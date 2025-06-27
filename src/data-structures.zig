const std = @import("std");
const utils = @import("test-utils.zig");

pub const DSError = error{
    InvalidCapacity, // A Queues capacity must be a power of 2
    QueueFull, // A thread is trying to push to a full queue
    AlreadyStolen, // A thread has stolen a value concurrently from the queue
};

/// A VyukovQueue to add values to. This VyukovQueue is implemented
/// with a lock-free ring buffer backend to make it extremely
/// performant.
pub fn VyukovQueue(comptime T: type) type {
    return struct {
        /// Represents a single slot in the VyukovQueue.
        /// sequence: std.atomic.Value - Used to keep track of insertions/deletions.
        /// value: T - The value associated with the slot.
        const Slot = struct {
            sequence: std.atomic.Value(usize),
            value: T,
        };

        /// buffer: []Slot - The data of the queue.
        /// buffer_mask: usize - A constant value used to update the pointers.
        /// push_pos: std.atomic.Value(usize) - The push position pointer.
        /// pop_pos: std.atomic.Value(usize) - The pop position pointer.
        /// allocator: std.mem.Allocator - An allocator to create the slot buffer.
        pad0: u16 = std.atomic.cache_line,
        buffer: []Slot,
        buffer_mask: usize,
        pad1: u16 = std.atomic.cache_line,
        push_pos: std.atomic.Value(usize),
        pad2: u16 = std.atomic.cache_line,
        pop_pos: std.atomic.Value(usize),
        pad3: u16 = std.atomic.cache_line,
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Instantiate a VyukovQueue object.
        /// Arguments:
        ///     allocator: std.mem.Allocator - An allocator to instantiate the buffer with.
        ///     capacity: usize - An initial capacity of the queue. Must be a power of two.
        /// Returns:
        ///     An instantiated VyukovQueue or error on allocation fail.
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            if (capacity & (capacity - 1) != 0) {
                return DSError.InvalidCapacity;
            }
            const buffer_mask: usize = capacity - 1;
            const buffer = try allocator.alloc(Slot, capacity);
            for (buffer, 0..) |_, i| {
                buffer[i].sequence.store(i, .release);
            }
            return Self{ .buffer = buffer, .buffer_mask = buffer_mask, .push_pos = std.atomic.Value(usize).init(0), .pop_pos = std.atomic.Value(usize).init(0), .allocator = allocator };
        }

        /// Destroy an instance of a VyukovQueue.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }

        /// Push a value to the VyukovQueue.
        /// Arguments:
        ///     value: T - The value to push.
        /// Returns:
        ///     void on success, ThreadPoolError.QueueFull if the queue is full.
        pub fn push(self: *Self, value: T) !void {
            // Load the position of the push pointer (might change during pushing)
            var slot: *Slot = undefined;
            var pos = self.push_pos.load(.acquire);
            while (true) {
                // Grab the slot at pos and grab the slot's sequence number
                slot = &self.buffer[pos & self.buffer_mask];
                const seq = slot.sequence.load(.acquire);
                if (seq == pos) {
                    // Sequnce and position match up, we can probably push
                    if (self.push_pos.cmpxchgWeak(pos, pos + 1, .acq_rel, .acquire)) |_| {
                        // Position has been updated since last read
                        std.atomic.spinLoopHint();
                    } else {
                        // This thread now owns the slot!
                        break;
                    }
                } else if (seq < pos) {
                    // The queue is full
                    return DSError.QueueFull;
                } else {
                    // The slot has been updated since last checking
                    pos = self.push_pos.load(.acquire);
                }
            }

            // This thread now owns the slot!
            slot.value = value;
            slot.sequence.store(pos + 1, .release);
        }

        /// Pop a value from the VyukovQueue.
        /// Returns:
        ///     A value on success, null if the queue is empty.
        pub fn pop(self: *Self) ?T {
            // Load the position of the pop pointer (might change during popping)
            var slot: *Slot = undefined;
            var pos = self.pop_pos.load(.acquire);
            var test_val = pos + 1;
            while (true) {
                // Grab the slot at pos and grab the slot's sequence number
                slot = &self.buffer[pos & self.buffer_mask];
                const seq = slot.sequence.load(.acquire);
                if (seq == test_val) {
                    // Sequence and position match up, we can probably push
                    if (self.pop_pos.cmpxchgWeak(pos, pos + 1, .acq_rel, .acquire)) |_| {
                        // Position has been update since last read
                        std.atomic.spinLoopHint();
                    } else {
                        // This thread now owns the slot!
                        break;
                    }
                } else if (seq < test_val) {
                    // The queue is empty
                    return null;
                } else {
                    // The slot has already been read since last checking
                    pos = self.pop_pos.load(.acquire);
                    test_val = pos + 1;
                }
            }

            // This thread now owns the slot!
            const value = slot.value;
            slot.sequence.store(pos + self.buffer_mask + 1, .release);
            return value;
        }
    };
}

/// This queue is a Single Producer/Multiple Consumer deque.
/// This queue is a ring buffer that requires a mutex for the stealing index.
pub fn SPMCDeque(comptime T: type) type {
    return struct {
        /// buffer: []T - A Ring buffer that holds the data.
        /// buffer_mask: usize - A value used to index into the buffer.
        /// lock: std.Thread.Mutex - Locks the top pointer.
        /// top: usize - Pointer to steal from.
        /// bottom: usize - Pointer to push/pop from. Owned by the current thread.
        /// allocator: std.mem.Allcator - The allocator used to create the buffer.
        buffer: []T,
        buffer_mask: usize,
        lock: std.Thread.Mutex,
        top: usize,
        bottom: usize,
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Initialize a SPMCDeque object.
        /// Arguments:
        ///     allocator: std.mem.Allocator - An allocator to create the ring buffer.
        ///     capacity: usize - The initial capacity (must be power of 2).
        /// Returns:
        ///     Initialized SPMCDeque on success, error on allocation error.
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            if (capacity & (capacity - 1) != 0) {
                return DSError.InvalidCapacity;
            }

            const buffer = try allocator.alloc(T, capacity);
            return Self{ .buffer = buffer, .buffer_mask = capacity - 1, .lock = .{}, .top = 0, .bottom = 0, .allocator = allocator };
        }

        /// Destroy an instance of the SPMCDeque
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }

        /// Push a value to the deque.
        /// Arguments:
        ///     value: T - The value to push.
        /// Returns:
        ///     void on success, error on queue full.
        pub fn push(self: *Self, value: T) !void {
            // Must read the top value
            self.lock.lock();

            if (self.bottom - self.top >= self.buffer.len) {
                // Deque is full
                return DSError.QueueFull;
            }

            self.buffer[self.bottom & self.buffer_mask] = value;
            self.bottom += 1;
            self.lock.unlock();
        }

        /// Steal a value from the deque.
        /// Returns:
        ///     A value on success, null on queue empty, error if another thread is stealing.
        pub fn steal(self: *Self) !?T {
            if (self.lock.tryLock()) {
                // No other thread is stealing right now
                if (self.bottom == self.top) {
                    // Queue is empty
                    self.lock.unlock();
                    return null;
                }

                // We own the top so grab the top value
                const value = self.buffer[self.top & self.buffer_mask];
                self.top += 1;
                self.lock.unlock();
                return value;
            }
            return DSError.AlreadyStolen;
        }

        /// Pop a value from the queue.
        /// Returns:
        ///     A value on success, null on queue empty.
        pub fn pop(self: *Self) ?T {
            // Must read the current value of top
            self.lock.lock();

            if (self.bottom == self.top) {
                self.lock.unlock();
                return null;
            }

            // Decrement bottom to point at the last item in the buffer
            self.bottom -= 1;

            // Grab the value in the bottom and return it if the top hasn't been stolen
            const value = self.buffer[self.bottom & self.buffer_mask];
            if (self.bottom >= self.top) {
                self.lock.unlock();
                return value;
            }
            self.bottom = self.top;
            self.lock.unlock();
            return null;
        }
    };
}

/// Functions used for testing
const test_func = struct {
    /// Test the push functionality of the VyokovQueue
    fn push_value_to_queue(comptime T: type, queue: *VyukovQueue(T), value: T) !void {
        try queue.push(value);
    }

    /// Test the pop functionality of the VyokovQueue
    fn consume(comptime T: type, queue: *VyukovQueue(T), sum: *std.atomic.Value(T)) void {
        const has_value = queue.pop();
        if (has_value) |value| {
            _ = sum.fetchAdd(value, .acq_rel);
        }
    }

    /// Test the steal functionality of the deque
    fn steal_from_deque(comptime T: type, deque: *SPMCDeque(T), sum: *std.atomic.Value(T)) void {
        const has_value = deque.steal() catch {
            return;
        };
        if (has_value) |value| {
            //std.debug.print("{d}\n", .{value});
            _ = sum.fetchAdd(value, .acq_rel);
        }
    }

    /// Time the VyukovQueue push method
    fn time_queue_push(comptime T: type, queue: *VyukovQueue(T), value: T, min_time: *std.atomic.Value(usize), max_time: *std.atomic.Value(usize), total_time: *std.atomic.Value(usize)) !void {
        var min: usize = 1_000_000_000_000;
        var max: usize = 0;
        for (0..512) |_| {
            const start_time = try std.time.Instant.now();
            queue.push(value) catch {};
            const end_time = try std.time.Instant.now();
            const elapsed = end_time.since(start_time);
            if (elapsed < min)
                min = elapsed;
            if (elapsed > max)
                max = elapsed;
            _ = total_time.fetchAdd(elapsed, .acq_rel);
        }
        if (min < min_time.load(.acquire)) {
            min_time.store(min, .release);
        }
        if (max > max_time.load(.acquire)) {
            max_time.store(max, .release);
        }
    }

    /// Time the VyukovQueue pop method
    fn time_queue_pop(comptime T: type, queue: *VyukovQueue(T), min_time: *std.atomic.Value(usize), max_time: *std.atomic.Value(usize), total_time: *std.atomic.Value(usize)) !void {
        var min: usize = 1_000_000_000_000;
        var max: usize = 0;
        for (0..512) |_| {
            const start_time = try std.time.Instant.now();
            _ = queue.pop();
            const end_time = try std.time.Instant.now();
            const elapsed = end_time.since(start_time);
            if (elapsed < min)
                min = elapsed;
            if (elapsed > max)
                max = elapsed;
            _ = total_time.fetchAdd(elapsed, .acq_rel);
        }
        if (min < min_time.load(.acquire)) {
            min_time.store(min, .release);
        }
        if (max > max_time.load(.acquire)) {
            max_time.store(max, .release);
        }
    }

    /// Time the SPMCDeque push method
    fn time_deque_push(comptime T: type, queue: *SPMCDeque(T), value: T, min_time: *std.atomic.Value(usize), max_time: *std.atomic.Value(usize), total_time: *std.atomic.Value(usize)) !void {
        var min: usize = 1_000_000_000_000;
        var max: usize = 0;
        for (0..4096) |_| {
            const start_time = try std.time.Instant.now();
            queue.push(value) catch {};
            const end_time = try std.time.Instant.now();
            const elapsed = end_time.since(start_time);
            if (elapsed < min)
                min = elapsed;
            if (elapsed > max)
                max = elapsed;
            _ = total_time.fetchAdd(elapsed, .acq_rel);
        }
        if (min < min_time.load(.acquire)) {
            min_time.store(min, .release);
        }
        if (max > max_time.load(.acquire)) {
            max_time.store(max, .release);
        }
    }

    /// Time the SPMCDeque pop method
    fn time_deque_pop(comptime T: type, queue: *SPMCDeque(T), min_time: *std.atomic.Value(usize), max_time: *std.atomic.Value(usize), total_time: *std.atomic.Value(usize)) !void {
        var min: usize = 1_000_000_000_000;
        var max: usize = 0;
        for (0..4096) |_| {
            const start_time = try std.time.Instant.now();
            _ = queue.pop();
            const end_time = try std.time.Instant.now();
            const elapsed = end_time.since(start_time);
            if (elapsed < min)
                min = elapsed;
            if (elapsed > max)
                max = elapsed;
            _ = total_time.fetchAdd(elapsed, .acq_rel);
        }
        if (min < min_time.load(.acquire)) {
            min_time.store(min, .release);
        }
        if (max > max_time.load(.acquire)) {
            max_time.store(max, .release);
        }
    }

    /// Time the SPMCDeque steal method
    fn time_deque_steal(comptime T: type, queue: *SPMCDeque(T), min_time: *std.atomic.Value(usize), max_time: *std.atomic.Value(usize), total_time: *std.atomic.Value(usize)) !void {
        var min: usize = 1_000_000_000_000;
        var max: usize = 0;
        for (0..256) |_| {
            const start_time = try std.time.Instant.now();
            _ = queue.steal() catch {};
            const end_time = try std.time.Instant.now();
            const elapsed = end_time.since(start_time);
            if (elapsed < min)
                min = elapsed;
            if (elapsed > max)
                max = elapsed;
            _ = total_time.fetchAdd(elapsed, .acq_rel);
        }
        if (min < min_time.load(.acquire)) {
            min_time.store(min, .release);
        }
        if (max > max_time.load(.acquire)) {
            max_time.store(max, .release);
        }
    }
};

test "VyukovQueue single threaded push/pop" {
    // Test of basic VyukovQueue methods (ensure a single thread can push/pop).
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var queue = try VyukovQueue(i32).init(allocator, 16);
    defer queue.deinit();

    for (0..10) |_| {
        try queue.push(test_int);
    }

    var sum: i32 = 0;
    for (0..10) |_| {
        const has_value = queue.pop();
        if (has_value) |value| {
            sum += value;
        }
    }

    try std.testing.expect(sum == 10);
}

test "VyukovQueue fill/empty/fill" {
    // Test of VyokovQueue edge case (ensure the queue can be emptied and refilled).
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var queue = try VyukovQueue(i32).init(allocator, 16);
    defer queue.deinit();

    var sum: i32 = 0;

    for (0..10) |_| {
        try queue.push(test_int);
    }

    for (0..10) |_| {
        const has_value = queue.pop();
        if (has_value) |value| {
            sum += value;
        }
    }

    for (0..10) |_| {
        try queue.push(test_int);
    }

    for (0..10) |_| {
        const has_value = queue.pop();
        if (has_value) |value| {
            sum += value;
        }
    }

    try std.testing.expect(sum == 20);
}

test "VyukovQueue multi threaded push single threaded pop" {
    // Ensure that multiple threads can push to the queue.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var queue = try VyukovQueue(i32).init(allocator, 16);
    defer queue.deinit();
    var threads: [10]std.Thread = undefined;

    for (0..10) |i| {
        threads[i] = try std.Thread.spawn(.{}, test_func.push_value_to_queue, .{ i32, &queue, test_int });
    }

    for (threads) |thread| {
        thread.join();
    }

    var sum: i32 = 0;
    for (0..10) |_| {
        const has_value = queue.pop();
        if (has_value) |value| {
            sum += value;
        }
    }

    try std.testing.expect(sum == 10);
}

test "VyukovQueue single threaded push multi threaded pop" {
    // Ensure that multiple threads can pop from the queue
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var threads: [10]std.Thread = undefined;
    var queue = try VyukovQueue(i32).init(allocator, 16);
    defer queue.deinit();

    for (0..10) |_| {
        try queue.push(test_int);
    }

    var sum = std.atomic.Value(i32).init(0);
    for (0..10) |i| {
        threads[i] = try std.Thread.spawn(.{}, test_func.consume, .{ i32, &queue, &sum });
    }

    for (threads) |thread| {
        thread.join();
    }

    try std.testing.expect(sum.load(.acquire) == 10);
}

test "VyukovQueue multi threaded push multi threaded pop" {
    // Ensure that multiple threads can push/pop from the queue.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var producer_threads: [10]std.Thread = undefined;
    var consumer_threads: [10]std.Thread = undefined;
    var queue = try VyukovQueue(i32).init(allocator, 16);
    defer queue.deinit();

    var sum = std.atomic.Value(i32).init(0);

    for (0..10) |i| {
        producer_threads[i] = try std.Thread.spawn(.{}, test_func.push_value_to_queue, .{ i32, &queue, test_int });
    }

    for (producer_threads) |thread| {
        thread.join();
    }

    for (0..10) |i| {
        consumer_threads[i] = try std.Thread.spawn(.{}, test_func.consume, .{ i32, &queue, &sum });
    }

    for (consumer_threads) |thread| {
        thread.join();
    }

    try std.testing.expect(sum.load(.acquire) == 10);
}

test "SPMCDeque test" {
    // Test the SPMCDeque
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var sum = std.atomic.Value(i32).init(0);

    var deque = try SPMCDeque(i32).init(allocator, 1024);
    defer deque.deinit();

    var threads: [32]std.Thread = undefined;

    for (0..1000) |_| {
        try deque.push(test_int);
    }

    for (0..32) |i| {
        threads[i] = try std.Thread.spawn(.{}, test_func.steal_from_deque, .{ i32, &deque, &sum });
    }

    while (deque.pop()) |value| {
        _ = sum.fetchAdd(value, .acq_rel);
    }

    for (threads) |thread| {
        thread.join();
    }

    try std.testing.expect(sum.load(.acquire) == 1000);
}

test "VyukovQueue performance test" {
    // Test the performance of the VyukovQueue in a multithreaded setting
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: usize = 1;
    var min_push_time = std.atomic.Value(usize).init(1_000_000_000_000);
    var min_pop_time = std.atomic.Value(usize).init(1_000_000_000_000);
    var max_push_time = std.atomic.Value(usize).init(0);
    var max_pop_time = std.atomic.Value(usize).init(0);
    var total_push_time = std.atomic.Value(usize).init(0);
    var total_pop_time = std.atomic.Value(usize).init(0);

    var queue = try VyukovQueue(usize).init(allocator, 4096);
    defer queue.deinit();

    var producer_threads: [8]std.Thread = undefined;
    var consumer_threads: [8]std.Thread = undefined;

    for (0..producer_threads.len) |i| {
        producer_threads[i] = try std.Thread.spawn(.{}, test_func.time_queue_push, .{ usize, &queue, test_int, &min_push_time, &max_push_time, &total_push_time });
        consumer_threads[i] = try std.Thread.spawn(.{}, test_func.time_queue_pop, .{ usize, &queue, &min_pop_time, &max_pop_time, &total_pop_time });
    }

    for (producer_threads ++ consumer_threads) |thread| {
        thread.join();
    }

    const min_push = min_push_time.load(.acquire);
    const min_pop = min_pop_time.load(.acquire);
    const max_push = max_push_time.load(.acquire);
    const max_pop = max_pop_time.load(.acquire);
    const total_push = total_push_time.load(.acquire);
    const total_pop = total_pop_time.load(.acquire);

    utils.header("Vyukov performance test");
    std.debug.print("Push Stats:\n", .{});
    utils.short_line();
    std.debug.print("Min: {d}\nMax: {d}\nAverage: {d}\n", .{ min_push, max_push, total_push / (8 * 512) });
    utils.short_line();
    std.debug.print("Pop Stats:\n", .{});
    utils.short_line();
    std.debug.print("Min: {d}\nMax: {d}\nAverage: {d}\n", .{ min_pop, max_pop, total_pop / (8 * 512) });
    utils.long_line();
}

test "SPMCDeque performance test" {
    // Test the performance of the SPMCDeque in a multithreaded setting
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: usize = 1;
    var min_push_time = std.atomic.Value(usize).init(1_000_000_000_000);
    var min_pop_time = std.atomic.Value(usize).init(1_000_000_000_000);
    var min_steal_time = std.atomic.Value(usize).init(1_000_000_000_000);
    var max_push_time = std.atomic.Value(usize).init(0);
    var max_pop_time = std.atomic.Value(usize).init(0);
    var max_steal_time = std.atomic.Value(usize).init(0);
    var total_push_time = std.atomic.Value(usize).init(0);
    var total_pop_time = std.atomic.Value(usize).init(0);
    var total_steal_time = std.atomic.Value(usize).init(0);

    var queue = try SPMCDeque(usize).init(allocator, 4096);
    defer queue.deinit();

    var stealer_threads: [8]std.Thread = undefined;

    for (0..stealer_threads.len) |i| {
        stealer_threads[i] = try std.Thread.spawn(.{}, test_func.time_deque_steal, .{ usize, &queue, &min_steal_time, &max_steal_time, &total_steal_time });
    }

    try test_func.time_deque_push(usize, &queue, test_int, &min_push_time, &max_push_time, &total_push_time);
    try test_func.time_deque_pop(usize, &queue, &min_pop_time, &max_pop_time, &total_pop_time);

    const min_push = min_push_time.load(.acquire);
    const min_pop = min_pop_time.load(.acquire);
    const min_steal = min_steal_time.load(.acquire);
    const max_push = max_push_time.load(.acquire);
    const max_pop = max_pop_time.load(.acquire);
    const max_steal = max_steal_time.load(.acquire);
    const total_push = total_push_time.load(.acquire);
    const total_pop = total_pop_time.load(.acquire);
    const total_steal = total_steal_time.load(.acquire);

    utils.header("SPMCDeque performance test");
    std.debug.print("Push Stats:\n", .{});
    utils.short_line();
    std.debug.print("Min: {d}\nMax: {d}\nAverage: {d}\n", .{ min_push, max_push, total_push / 4096 });
    utils.short_line();
    std.debug.print("Pop Stats:\n", .{});
    utils.short_line();
    std.debug.print("Min: {d}\nMax: {d}\nAverage: {d}\n", .{ min_pop, max_pop, total_pop / 4096 });
    utils.short_line();
    std.debug.print("Steal Stats:\n", .{});
    utils.short_line();
    std.debug.print("Min: {d}\nMax: {d}\nAverage: {d}\n", .{ min_steal, max_steal, total_steal / (8 * 256) });
    utils.long_line();
}
