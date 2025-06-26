const std = @import("std");

pub const LockFreeError = error{
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
        /// Returns:
        ///     An instantiated VyukovQueue or error on allocation fail.
        pub fn init(allocator: std.mem.Allocator) !Self {
            const buffer_mask: usize = 4095;
            const buffer = try allocator.alloc(Slot, buffer_mask + 1);
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
                    return LockFreeError.QueueFull;
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

/// A lock-free double ended queue.
/// This is based on the Chase-Lev deque (https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf)
/// The Chase-Lev deque offers a lock-free work-stealing queue
/// that overcomes the buffer overflow issue of many current
/// work stealing deques.
pub fn ChaseLevDeque(comptime T: type) type {
    return struct {
        /// bottom: usize - Values are only pushed and popped from bottom in the owner thread.
        /// top: std.atomic.Value(usize) - Values are only stolen (never pushed) from the top by other threads.
        /// buffer: []T - The queue is a circular buffer of values.
        /// buffer_mask: usize - Value used to quickly determine the value associated with an index.
        /// allocator: std.mem.Allocator - An allocator to allocate the buffer.
        bottom: usize,
        top: std.atomic.Value(usize),
        buffer: []T,
        buffer_mask: usize,
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Initialize a ChaseLevDeque
        /// Arguments:
        ///     allocator: std.mem.Allocator - An allocator to allocate the buffer.
        /// Returns:
        ///     An initialized ChaseLevDeque object.
        pub fn init(allocator: std.mem.Allocator) !Self {
            const buffer_mask = 4095;
            const buffer = try allocator.alloc(T, buffer_mask + 1);
            return Self{ .bottom = 0, .top = std.atomic.Value(usize).init(0), .buffer = buffer, .buffer_mask = buffer_mask, .allocator = allocator };
        }

        /// Destroy a ChaseLevDeque object
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }

        /// Perform a CAS operation on the top pointer.
        /// Arguments:
        ///     expected_value: usize - Check if the value is equal to this expected value.
        ///     new_value: usize - If the comparison works, substitute this value.
        /// Returns:
        ///     true on success, false on failure.
        pub fn casTop(self: *Self, expected_value: usize, new_value: usize) bool {
            // @cmpxchgStrong returns null on sucess, so this seems backwards
            if (self.top.cmpxchgStrong(expected_value, new_value, .acq_rel, .acquire)) |_| {
                return false;
            }
            return true;
        }

        /// Push a value to the bottom of the deque. This is an easy operation
        /// since only the owner thread will be pushing to the deque.
        /// Arguments:
        ///     value: T - The value to push to the bottom of the queue.
        /// Returns:
        ///     void on success, LockFreeError.QueueFull when the queue is full.
        pub fn push(self: *Self, value: T) !void {
            const t = self.top.load(.acquire);
            const size = self.bottom - t;
            if (size >= self.buffer_mask) {
                return LockFreeError.QueueFull;
            }
            self.buffer[self.bottom & self.buffer_mask] = value;
            self.bottom += 1;
        }

        /// Steal a value from the top of the deque. The stealing thread must ensure
        /// that another thread has not already stole the bottom value.
        /// Returns:
        ///     A value if one is available, null on empty, LockFreeError.AlreadyStolen
        ///     if another concurrent thread stole the top before we had a chance.
        pub fn steal(self: *Self) !?T {
            const t = self.top.load(.acquire);

            // Deque is empty
            if (self.bottom <= t) {
                return null;
            }

            // Must grab value before CAS because after CAS, this value
            // may be written over with another value by push (due to circular buffer)
            const value = self.buffer[t & self.buffer_mask];
            if (!self.casTop(t, t + 1)) {
                return LockFreeError.AlreadyStolen;
            }
            return value;
        }

        /// Pop a value from the bottom of the deque. Pop is a little more complicated
        /// because a pop/steal race may arise.
        /// Retunrs:
        ///     A value if one is available, null on empty.
        pub fn pop(self: *Self) ?T {
            // Decrement the bottom pointer to get the next Task
            self.bottom = self.bottom - 1;
            const t = self.top.load(.acquire);

            // The deque is empty
            if (self.bottom < t) {
                // Reset the deque back to a canonical empty state (bottom = top)
                self.bottom = t;
                return null;
            }

            // Deque is not empty, but a concurrent steal operation could still
            // race on this task
            const value = self.buffer[self.bottom & self.buffer_mask];
            if (self.bottom > t) {
                // Race condition not possible (more than one item in deque)
                return value;
            }

            // Race condition possible (one item left in deque)
            // Try to increment top by one to test if top has been accessed
            if (!self.casTop(t, t + 1)) {
                // Last element was stolen, reset deque back to canonical empty state (bottom = top)
                // We can set bottom = t + 1 because top will be t + 1 regardless of the
                // result of the CAS (either we changed it or a steal operation did)
                self.bottom = t + 1;
                return null;
            }

            // No race condition, reset deque back to canonical empty state (bottom = top)
            self.bottom = t + 1;
            return value;
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
    fn steal_from_deque(comptime T: type, deque: *ChaseLevDeque(T), sum: *std.atomic.Value(T)) void {
        const has_value = deque.steal() catch {
            return;
        };
        if (has_value) |value| {
            _ = sum.fetchAdd(value, .acq_rel);
        }
    }
};

test "VyukovQueue single threaded push/pop" {
    // Test of basic VyukovQueue methods (ensure a single thread can push/pop).
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var queue = try VyukovQueue(i32).init(allocator);
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

    var queue = try VyukovQueue(i32).init(allocator);
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

    var queue = try VyukovQueue(i32).init(allocator);
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
    var queue = try VyukovQueue(i32).init(allocator);
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
    var queue = try VyukovQueue(i32).init(allocator);
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

test "ChaseLevDeque test" {
    // Test the ChaseLevTaskQueue
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    const test_int: i32 = 1;

    var sum = std.atomic.Value(i32).init(0);

    var deque = try ChaseLevDeque(i32).init(allocator);
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
