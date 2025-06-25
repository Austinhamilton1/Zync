const std = @import("std");

/// Potential ThreadPool Errors
pub const ThreadPoolError = error{
    TooManyThreads, // Trying to spawn more than 1024 threads
    ForkError, // One of the spawns did not work in the fork method; discard results
    QueueFull, // BoundedTaskQueue can't accept any more tasks
};

/// Represents a Task needing to be run.
/// Tasks are added to the TaskQueue, where
/// they wait for a thread to consume them and use them.
const Task = struct {
    /// func: *const fn (*const anyopaque) void - A generic function to run.
    /// ctx: *const anyopaque - The arguments to pass the function.
    func: *const fn (*const anyopaque) void,
    ctx: *const anyopaque,
    deinit: *const fn (*const anyopaque, std.mem.Allocator) void,
};

/// An UnboundedTaskQueue to add Tasks to. The Tasks wait
/// in the queue until they are ready to be used by
/// a Thread in the ThreadPool. This TaskQueue is implemented
/// with a lock-free linked list backend to make it unbounded.
pub const UnboundedTaskQueue = struct {
    /// A single node in the linked list.
    const Node = struct {
        /// task: Task - The task to be completed.
        /// next: std.atomic.Value(usize) - A pointer to the next node in the list.
        task: Task,
        next: std.atomic.Value(?*Node),

        const NodeSelf = @This();

        /// Return an instantiated Node.
        /// Arguments:
        ///     task: Task - The task associated with the node.
        /// Returns:
        ///     An instantiated Node object.
        pub fn init(task: Task) NodeSelf {
            return NodeSelf{ .task = task, .next = std.atomic.Value(?*Node).init(null) };
        }
    };

    /// head: std.atomic.Value(usize) - The read pointer of the queue.
    /// tail: std.atomic.Value(usize) - The write pointer of the queue.
    /// len: std.atomic.Value(usize) - How many elements are in the queue.
    /// allocator: std.mem.Allocator - A thread safe allocator to allocate new nodes.
    head: std.atomic.Value(?*Node),
    tail: std.atomic.Value(?*Node),
    len: std.atomic.Value(usize),
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize an UnboundedTaskQueue object.
    /// Arguments:
    ///     allocator: std.mem.Allocator - MUST BE A THREAD SAFE ALLOCATOR.
    /// Returns:
    ///     An instantiated TaskQueue object
    pub fn init(allocator: std.mem.Allocator) !Self {
        const dummy = try allocator.create(Node);
        dummy.* = Node.init(Task{
            .func = undefined,
            .ctx = undefined,
            .deinit = undefined,
        });
        return Self{ .head = std.atomic.Value(?*Node).init(dummy), .tail = std.atomic.Value(?*Node).init(dummy), .len = std.atomic.Value(usize).init(0), .allocator = allocator };
    }

    /// Destroy an UnboundedTaskQueue object.
    pub fn deinit(self: *Self) void {
        // Traverse the linked list and destroy each node.
        var ptr = self.head.load(.acquire);
        while (ptr) |node| {
            ptr = node.next.load(.acquire);
            self.allocator.destroy(node);
        }
    }

    /// Push a Task to the back of the UnboundedTaskQueue.
    /// Arguments:
    ///     task: Task - The task for the Node to hold.
    /// Returns:
    ///     void on success, error on allocation error.
    pub fn push(self: *Self, task: Task) !void {
        const node = try self.allocator.create(Node);
        node.* = Node.init(task);
        var old_tail: *Node = undefined;

        while (true) {
            old_tail = self.tail.load(.acquire).?;
            const next_ptr = old_tail.next.load(.acquire);

            // Check if tail hasn't changed
            if (old_tail == self.tail.load(.acquire)) {
                if (next_ptr) |next| {
                    // Tail is lagging; advance it
                    _ = self.tail.cmpxchgWeak(old_tail, next, .acq_rel, .acquire);
                } else {
                    // Tail is truly the last node
                    if (old_tail.next.cmpxchgWeak(next_ptr, node, .acq_rel, .acquire)) |_| {
                        std.atomic.spinLoopHint();
                    } else {
                        break;
                    }
                }
            }

            // We failed to grab the tail, try again.
            std.atomic.spinLoopHint();
        }
    }

    /// Pop a Task from the front of the UnboundedTaskQueue.
    /// Returns:
    ///     A Task on success, null on empty queue.
    pub fn pop(self: *Self) ?Task {
        var old_head: *Node = undefined;

        while (true) {
            old_head = self.head.load(.acquire) orelse return null;
            const old_tail = self.tail.load(.acquire) orelse return null;
            const next_ptr = old_head.next.load(.acquire);

            // Check if head hasn't changed
            if (old_head == self.head.load(.acquire)) {
                // Queue might be empty
                if (old_head == old_tail) {
                    if (next_ptr) |next| {
                        // Advance the tail.
                        _ = self.tail.cmpxchgWeak(old_tail, next, .acq_rel, .acquire);
                    } else {
                        // Queue is empty
                        return null;
                    }
                } else {
                    // Queue is not empty
                    if (self.head.cmpxchgWeak(old_head, next_ptr, .acq_rel, .acquire)) |_| {
                        std.atomic.spinLoopHint();
                    } else {
                        self.allocator.destroy(old_head);
                        return next_ptr.?.task;
                    }
                }
            }

            // We failed to grab the head, try again
            std.atomic.spinLoopHint();
        }
    }
};

/// A BoundedTaskQueue to add Tasks to. The Tasks wait
/// in the queue until they are ready to be used by
/// a Thread in the ThreadPool. This BoundedTaskQueue is implemented
/// with a lock-free ring buffer backend to make it extremely
/// performant.
pub const BoundedTaskQueue = struct {
    /// Represents a single slot in the BoundedTaskQueue.
    /// sequence: std.atomic.Value - Used to keep track of insertions/deletions.
    /// task: Task - The task associated with the slot.
    const Slot = struct {
        sequence: std.atomic.Value(usize),
        task: Task,
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

    /// Instantiate a BoundedTaskQueue object.
    /// Arguments:
    ///     allocator: std.mem.Allocator - An allocator to instantiate the buffer with.
    /// Returns:
    ///     An instantiated BoundedTaskQueue or error on allocation fail.
    pub fn init(allocator: std.mem.Allocator) !Self {
        const buffer_mask: usize = 1023;
        const buffer = try allocator.alloc(Slot, buffer_mask + 1);
        for (buffer, 0..) |_, i| {
            buffer[i].sequence.store(i, .release);
        }
        return Self{ .buffer = buffer, .buffer_mask = buffer_mask, .push_pos = std.atomic.Value(usize).init(0), .pop_pos = std.atomic.Value(usize).init(0), .allocator = allocator };
    }

    /// Destroy an instance of a BoundedTaskQueue.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buffer);
    }

    /// Push a Task to the BoundedTaskQueue.
    /// Arguments:
    ///     task: Task - The task to push.
    /// Returns:
    ///     void on success, ThreadPoolError.QueueFull if the queue is full.
    pub fn push(self: *Self, task: Task) !void {
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
                return ThreadPoolError.QueueFull;
            } else {
                // The slot has been updated since last checking
                pos = self.push_pos.load(.acquire);
            }
        }

        // This thread now owns the slot!
        slot.task = task;
        slot.sequence.store(pos + 1, .release);
    }

    /// Pop a Task from the BoundedTaskQueue.
    /// Returns:
    ///     A Task on success, null if the queue is empty.
    pub fn pop(self: *Self) ?Task {
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
        const task = slot.task;
        slot.sequence.store(pos + self.buffer_mask + 1, .release);
        return task;
    }
};

/// Logical grouping of a set of Threads.
/// Allows for waiting on all threads to hit some
/// sync point.
pub const ThreadGroup = struct {
    counter: std.atomic.Value(usize),

    const Self = @This();

    /// Create an instantiated ThreadGroup instance.
    /// Returns:
    ///     An instantiated ThreadGroup.
    pub fn init() Self {
        return Self{ .counter = std.atomic.Value(usize).init(0) };
    }

    /// Add a reference to the ThreadGroup
    pub fn increment(self: *Self) void {
        _ = self.counter.fetchAdd(1, .acq_rel);
    }

    /// Subtract a reference from the ThreadGroup
    pub fn decrement(self: *Self) void {
        _ = self.counter.fetchSub(1, .acq_rel);
    }

    /// Wait on all threads to finish (counter == 0)
    pub fn wait(self: *Self) void {
        while (self.counter.load(.acquire) > 0) {}
    }
};

/// Context needed to run a consumer thread.
/// QueueType: type - The type of queue to implement the consumer with.
/// queue: *QueueType - Consumer will grab tasks from this queue.
/// group: *ThreadGroup - Consumer will update this group as it completes tasks.
/// running: *bool - Consumer will check this to see if it needs to continue running.
/// allocator: std.mem.Allocator - The allocator used to instantiate the function context.
pub fn ConsumerContext(comptime QueueType: type) type {
    return struct {
        queue: *QueueType,
        group: *ThreadGroup,
        running: *bool,
    };
}

/// Consumer for the ThreadPool.
/// Arguments:
///     comptime QueueType: type - The type of queue that is being used on the backend.
///     ctx: ConsumerContext - The context to run the consumer under.
fn consumer(QueueType: type, ctx: ConsumerContext(QueueType)) void {
    while (true) {
        // Check if the thread should continue running.
        const running = ctx.running.*;
        if (!running) {
            break;
        }

        // Pop a task off of the queue and run it
        const task_obj = ctx.queue.pop();
        if (task_obj) |task| {
            task.func(task.ctx);

            // Free the context memory now that the task is finished
            task.deinit(task.ctx, ctx.queue.allocator);

            ctx.group.decrement(); // Decrement the ThreadGroup (This thread is currently idle).
        }
    }
}

/// Context needed to create a ThreadPool
/// n_jobs: usize - How many threads to spawn (<= 1024)
/// ignore_errors: bool - If a thread gives an error, should the pool ignore it or handle it
pub const ThreadPoolContext = struct {
    n_jobs: usize,
    handle_errors: bool = false,
};

/// A thread pool object. Keeps and manages a list of threads
/// to consume tasks as they are produced.
/// Arguments
///     comptime QueueType: type - The type of task queue to use as a backend.
pub fn ThreadPool(comptime QueueType: type) type {
    return struct {
        /// n_jobs: usize - How many threads the ThreadPool is managing.
        /// handle_errors: bool - Should the pool allow error functions?
        /// workers: []Thread - The worker threads.
        /// running: *bool - The current state of the ThreadPool.
        /// queue: *TaskQueue - The queue that tasks come into the pool through.
        /// group: *ThreadGroup - A ThreadGroup that observes the state of the pool's threads.
        /// allocator: std.mem.Allocator - Allocator shared by TaskQueue (MUST BE THREAD SAFE!!!)
        n_jobs: usize,
        handle_errors: bool,
        workers: []std.Thread,
        running: *bool, // This is a pointer because the pool needs to be ready to run immediately after intialization
        group: *ThreadGroup,
        queue: *QueueType,
        allocator: std.mem.Allocator,

        // Might need to add a list of ThreadGroups to allow more fine grain control
        // over thread execution.

        const Self = @This();

        /// Return an initialized ThreadPool object
        /// Arguments:
        ///     allocator: std.mem.Allocator - A !!!thread safe!!! allocator.
        ///     ctx: ThreadPoolContext - The context to initiate the pool.
        /// Returns:
        ///     A fully instantiated ThreadPool object or an error on allocation error.
        pub fn init(allocator: std.mem.Allocator, ctx: ThreadPoolContext) !Self {
            // Ensure reasonable number of threads
            if (ctx.n_jobs > 1024) {
                return ThreadPoolError.TooManyThreads;
            }

            // Declare worker threads
            const workers = try allocator.alloc(std.Thread, ctx.n_jobs);

            // Initialize running = true (need to be ready to consume immediately)
            const running = try allocator.create(bool);
            running.* = true;

            // Initialize the ThreadGroup. This is a global thread group that observes all threads.
            const group = try allocator.create(ThreadGroup);
            group.* = ThreadGroup.init();

            // Initialize the queue and then give it the allocator (the pool will no longer need it).
            const queue = try allocator.create(QueueType);
            queue.* = try QueueType.init(allocator);

            const consumer_ctx = ConsumerContext(QueueType){
                .queue = queue,
                .group = group,
                .running = running,
            };

            // Instantiate worker threads
            for (0..ctx.n_jobs) |i| {
                workers[i] = try std.Thread.spawn(.{}, consumer, .{ QueueType, consumer_ctx });
            }

            return Self{ .n_jobs = ctx.n_jobs, .handle_errors = ctx.handle_errors, .workers = workers, .running = running, .group = group, .queue = queue, .allocator = allocator };
        }

        /// Destroy an instance of a ThreadPool.
        /// Join all threads and clean up memory used.
        pub fn deinit(self: *Self) void {
            // Join all threads.
            self.running.* = false;
            for (self.workers) |thread| {
                thread.join();
            }

            // Clean up allocated memory
            // Deallocate in reverse order of allocation
            self.queue.deinit();
            self.allocator.destroy(self.queue);
            self.allocator.destroy(self.group);
            self.allocator.destroy(self.running);
            self.allocator.free(self.workers);
        }

        /// Add a Task to the task queue.
        /// Arguments:
        ///     comptime FuncType: type - The method signature of the spawned task function.
        ///     F: FuncType - A function matching the supplied FuncType.
        ///     args: anytype - The arguments to pass into the function.
        /// Returns:
        ///     void on success, error on allocation error.
        pub fn spawn(self: *Self, comptime FuncType: type, F: FuncType, args: anytype) !void {
            // Resolve the types of the function and arguments
            const ArgType = @TypeOf(args);
            const Context = struct {
                f: FuncType,
                args: ArgType,
            };

            // Create a context to add to the task
            const context = try self.allocator.create(Context);
            context.* = .{
                .f = F,
                .args = args,
            };

            // Convert from Context to Task
            const trampoline = struct {
                fn call(ctx: *const anyopaque) void {
                    const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                    @call(.auto, call_ctx.f, call_ctx.args);
                }
            }.call;

            // Create a task deallocation function
            const deinit_fn = struct {
                fn call(ctx: *const anyopaque, allocator: std.mem.Allocator) void {
                    const casted: *const Context = @alignCast(@ptrCast(ctx));
                    allocator.destroy(casted);
                }
            }.call;

            // Create a task to push to the queue
            const task = Task{
                .func = trampoline,
                .ctx = context,
                .deinit = deinit_fn,
            };

            // Increment the group counter.
            self.group.increment();

            // try to push the task to the queue
            try self.queue.push(task);
            errdefer self.allocator.destroy(context);
        }

        /// Wait for all threads to enter an idle state
        pub fn join(self: *Self) void {
            self.group.wait();
        }
    };
}

/// Add one to an argument.
/// Arguments:
///     arg: *i32 - An integer to add one to.
fn addOne(arg: *i32) void {
    arg.* = arg.* + 1;
}

/// Square arr[index].
/// Arguments:
///     index: usize - The index of the array place to square.
///     arr: []u64 - A slice to square an element of.
fn square(index: usize, arr: []u64) void {
    arr[index] = arr[index] * arr[index];
}

/// Parallel squaring function
/// Arguments:
///     index: usize - The index of the array place to start squaring.
///     arr: []u64 - A slice to square.
///     step: usize - A step value.
fn par_square(index: usize, arr: []u64, step: usize) void {
    var i = index;
    while (i < arr.len) : (i += step) {
        arr[i] = arr[i] * arr[i];
    }
}

/// Parallel vector addition function (results end up in a)
/// Arguments:
///     index: usize - The index of the array place to start adding
///     a: []u64 - A slice to add
///     b: []u64 - Another slice to add
///     step: usize - A step value
fn par_add(index: usize, a: []u64, b: []u64, step: usize) void {
    var i = index;
    while (i < a.len) : (i += step) {
        a[i] = a[i] + b[i];
    }
}

/// Push a task to a TaskQueue.
/// Arguments:
///     comptime QueueType: type - The type of queue to push to.
///     queue: *QueueType - The queue to push to.
///     task: Task - The task to push.
fn push_task_to_queue(comptime QueueType: type, queue: *QueueType, task: Task) !void {
    try queue.push(task);
}

/// Consume a task from a TaskQueue.
/// Arguments:
///     comptime QueueType: type - The type of queue to consume.
///     queue - *QueueType - The queue to pop from.
fn consume(comptime QueueType: type, queue: *QueueType) void {
    const task_obj = queue.pop();
    if (task_obj) |task| {
        task.func(task.ctx);
    }
}

/// Tests a ThreadGroup. Makes the threads wait for two seconds before
/// updating an atomic variable. Finally decrements the reference count to the
/// ThreadGroup before exiting.
/// Arguments:
///     group: *ThreadGroup - A ThreadGroup to test.
///     arg: *std.atomic.Value(i32) - A value to test with.
fn thread_group_test(group: *ThreadGroup, arg: *std.atomic.Value(i32)) void {
    std.time.sleep(2_000_000_000);
    _ = arg.fetchAdd(1, .acq_rel);
    group.decrement();
}

test "Basic Task run" {
    // Create a basic Task and try to run it from the
    // Task context.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var test_int: i32 = 10;

    const args = .{&test_int};
    const FuncType = *const fn (*i32) void;
    const ArgType = @TypeOf(args);
    const Context = struct {
        f: FuncType,
        args: ArgType,
    };

    const context = try allocator.create(Context);
    defer allocator.destroy(context);

    context.* = .{
        .f = addOne,
        .args = args,
    };

    const trampoline = struct {
        fn call(ctx: *const anyopaque) void {
            const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
            @call(.auto, call_ctx.f, call_ctx.args);
        }
    }.call;

    // Create a task deallocation function
    const deinit_fn = struct {
        fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
            const casted: *const Context = @alignCast(@ptrCast(ctx));
            task_allocator.destroy(casted);
        }
    }.call;

    const task = Task{
        .func = trampoline,
        .ctx = context,
        .deinit = deinit_fn,
    };

    task.func(task.ctx);

    try std.testing.expect(test_int == 11);
}

test "Complex Task run" {
    // Create a more complex Task and try to run it from the
    // Task context.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    for (0..test_arr.len) |i| {
        const args = .{ i, arr_slice };
        const FuncType = *const fn (usize, []u64) void;
        const ArgType = @TypeOf(args);
        const Context = struct {
            f: FuncType,
            args: ArgType,
        };

        const context = try allocator.create(Context);
        defer allocator.destroy(context);

        context.* = .{
            .f = square,
            .args = args,
        };

        const trampoline = struct {
            fn call(ctx: *const anyopaque) void {
                const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                @call(.auto, call_ctx.f, call_ctx.args);
            }
        }.call;

        // Create a task deallocation function
        const deinit_fn = struct {
            fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
                const casted: *const Context = @alignCast(@ptrCast(ctx));
                task_allocator.destroy(casted);
            }
        }.call;

        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        task.func(task.ctx);
    }

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }
}

test "UnboundedTaskQueue single threaded push/pop" {
    // Test of basic TaskQueue methods (ensure a single thread can push/pop).
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_int: i32 = 10;

    const args = .{&test_int};
    const FuncType = *const fn (*i32) void;
    const ArgType = @TypeOf(args);
    const Context = struct {
        f: FuncType,
        args: ArgType,
    };

    const context = try allocator.create(Context);

    context.* = .{
        .f = addOne,
        .args = args,
    };

    const trampoline = struct {
        fn call(ctx: *const anyopaque) void {
            const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
            @call(.auto, call_ctx.f, call_ctx.args);
        }
    }.call;

    // Create a task deallocation function
    const deinit_fn = struct {
        fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
            const casted: *const Context = @alignCast(@ptrCast(ctx));
            task_allocator.destroy(casted);
        }
    }.call;

    var queue = try UnboundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..10) |_| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    try std.testing.expect(test_int == 20);
}

test "BoundedTaskQueue single threaded push/pop" {
    // Test of basic TaskQueue methods (ensure a single thread can push/pop).
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_int: i32 = 10;

    const args = .{&test_int};
    const FuncType = *const fn (*i32) void;
    const ArgType = @TypeOf(args);
    const Context = struct {
        f: FuncType,
        args: ArgType,
    };

    const context = try allocator.create(Context);

    context.* = .{
        .f = addOne,
        .args = args,
    };

    const trampoline = struct {
        fn call(ctx: *const anyopaque) void {
            const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
            @call(.auto, call_ctx.f, call_ctx.args);
        }
    }.call;

    // Create a task deallocation function
    const deinit_fn = struct {
        fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
            const casted: *const Context = @alignCast(@ptrCast(ctx));
            task_allocator.destroy(casted);
        }
    }.call;

    var queue = try BoundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..10) |_| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    try std.testing.expect(test_int == 20);
}

test "UnboundedTaskQueue fill/empty/fill" {
    // Test of TaskQueue edge case (ensure the queue can be emptied and refilled).
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_int: i32 = 10;

    const args = .{&test_int};
    const FuncType = *const fn (*i32) void;
    const ArgType = @TypeOf(args);
    const Context = struct {
        f: FuncType,
        args: ArgType,
    };

    const context = try allocator.create(Context);

    context.* = .{
        .f = addOne,
        .args = args,
    };

    const trampoline = struct {
        fn call(ctx: *const anyopaque) void {
            const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
            @call(.auto, call_ctx.f, call_ctx.args);
        }
    }.call;

    // Create a task deallocation function
    const deinit_fn = struct {
        fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
            const casted: *const Context = @alignCast(@ptrCast(ctx));
            task_allocator.destroy(casted);
        }
    }.call;

    var queue = try UnboundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..10) |_| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    for (0..10) |_| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    try std.testing.expect(test_int == 30);
}

test "BoundedTaskQueue fill/empty/fill" {
    // Test of TaskQueue edge case (ensure the queue can be emptied and refilled).
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_int: i32 = 10;

    const args = .{&test_int};
    const FuncType = *const fn (*i32) void;
    const ArgType = @TypeOf(args);
    const Context = struct {
        f: FuncType,
        args: ArgType,
    };

    const context = try allocator.create(Context);

    context.* = .{
        .f = addOne,
        .args = args,
    };

    const trampoline = struct {
        fn call(ctx: *const anyopaque) void {
            const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
            @call(.auto, call_ctx.f, call_ctx.args);
        }
    }.call;

    // Create a task deallocation function
    const deinit_fn = struct {
        fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
            const casted: *const Context = @alignCast(@ptrCast(ctx));
            task_allocator.destroy(casted);
        }
    }.call;

    var queue = try BoundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..10) |_| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    for (0..10) |_| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    try std.testing.expect(test_int == 30);
}

test "UnboundedTaskQueue multi threaded push single threaded pop" {
    // Ensure that multiple threads can push to the queue.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_int: i32 = 10;

    const args = .{&test_int};
    const FuncType = *const fn (*i32) void;
    const ArgType = @TypeOf(args);
    const Context = struct {
        f: FuncType,
        args: ArgType,
    };

    const context = try allocator.create(Context);

    context.* = .{
        .f = addOne,
        .args = args,
    };

    const trampoline = struct {
        fn call(ctx: *const anyopaque) void {
            const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
            @call(.auto, call_ctx.f, call_ctx.args);
        }
    }.call;

    // Create a task deallocation function
    const deinit_fn = struct {
        fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
            const casted: *const Context = @alignCast(@ptrCast(ctx));
            task_allocator.destroy(casted);
        }
    }.call;

    var queue = try UnboundedTaskQueue.init(allocator);
    defer queue.deinit();
    var threads: [10]std.Thread = undefined;

    for (0..10) |i| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ UnboundedTaskQueue, &queue, task });
    }

    for (threads) |thread| {
        thread.join();
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    try std.testing.expect(test_int == 20);
}

test "BoundedTaskQueue multi threaded push single threaded pop" {
    // Ensure that multiple threads can push to the queue.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_int: i32 = 10;

    const args = .{&test_int};
    const FuncType = *const fn (*i32) void;
    const ArgType = @TypeOf(args);
    const Context = struct {
        f: FuncType,
        args: ArgType,
    };

    const context = try allocator.create(Context);

    context.* = .{
        .f = addOne,
        .args = args,
    };

    const trampoline = struct {
        fn call(ctx: *const anyopaque) void {
            const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
            @call(.auto, call_ctx.f, call_ctx.args);
        }
    }.call;

    // Create a task deallocation function
    const deinit_fn = struct {
        fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
            const casted: *const Context = @alignCast(@ptrCast(ctx));
            task_allocator.destroy(casted);
        }
    }.call;

    var queue = try BoundedTaskQueue.init(allocator);
    defer queue.deinit();
    var threads: [10]std.Thread = undefined;

    for (0..10) |i| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ BoundedTaskQueue, &queue, task });
    }

    for (threads) |thread| {
        thread.join();
    }

    for (0..10) |_| {
        const task = queue.pop();
        if (task) |task_to_run| {
            task_to_run.func(task_to_run.ctx);
        }
    }

    try std.testing.expect(test_int == 20);
}

test "UnboundedTaskQueue single threaded push multi threaded pop" {
    // Ensure that multiple threads can pop from the queue
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var threads: [5]std.Thread = undefined;
    var queue = try UnboundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..test_arr.len) |i| {
        const args = .{ i, arr_slice };
        const FuncType = *const fn (usize, []u64) void;
        const ArgType = @TypeOf(args);
        const Context = struct {
            f: FuncType,
            args: ArgType,
        };

        const context = try allocator.create(Context);

        context.* = .{
            .f = square,
            .args = args,
        };

        const trampoline = struct {
            fn call(ctx: *const anyopaque) void {
                const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                @call(.auto, call_ctx.f, call_ctx.args);
            }
        }.call;

        // Create a task deallocation function
        const deinit_fn = struct {
            fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
                const casted: *const Context = @alignCast(@ptrCast(ctx));
                task_allocator.destroy(casted);
            }
        }.call;

        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..5) |i| {
        threads[i] = try std.Thread.spawn(.{}, consume, .{ UnboundedTaskQueue, &queue });
    }

    for (threads) |thread| {
        thread.join();
    }

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }
}

test "BoundedTaskQueue single threaded push multi threaded pop" {
    // Ensure that multiple threads can pop from the queue
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var threads: [5]std.Thread = undefined;
    var queue = try BoundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..test_arr.len) |i| {
        const args = .{ i, arr_slice };
        const FuncType = *const fn (usize, []u64) void;
        const ArgType = @TypeOf(args);
        const Context = struct {
            f: FuncType,
            args: ArgType,
        };

        const context = try allocator.create(Context);

        context.* = .{
            .f = square,
            .args = args,
        };

        const trampoline = struct {
            fn call(ctx: *const anyopaque) void {
                const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                @call(.auto, call_ctx.f, call_ctx.args);
            }
        }.call;

        // Create a task deallocation function
        const deinit_fn = struct {
            fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
                const casted: *const Context = @alignCast(@ptrCast(ctx));
                task_allocator.destroy(casted);
            }
        }.call;

        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        try queue.push(task);
    }

    for (0..5) |i| {
        threads[i] = try std.Thread.spawn(.{}, consume, .{ BoundedTaskQueue, &queue });
    }

    for (threads) |thread| {
        thread.join();
    }

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }
}

test "UnboundedTaskQueue multi threaded push multi threaded pop" {
    // Ensure that multiple threads can push/pop from the queue.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var producer_threads: [5]std.Thread = undefined;
    var consumer_threads: [5]std.Thread = undefined;
    var queue = try UnboundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..test_arr.len) |i| {
        const args = .{ i, arr_slice };
        const FuncType = *const fn (usize, []u64) void;
        const ArgType = @TypeOf(args);
        const Context = struct {
            f: FuncType,
            args: ArgType,
        };

        const context = try allocator.create(Context);

        context.* = .{
            .f = square,
            .args = args,
        };

        const trampoline = struct {
            fn call(ctx: *const anyopaque) void {
                const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                @call(.auto, call_ctx.f, call_ctx.args);
            }
        }.call;

        // Create a task deallocation function
        const deinit_fn = struct {
            fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
                const casted: *const Context = @alignCast(@ptrCast(ctx));
                task_allocator.destroy(casted);
            }
        }.call;

        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        producer_threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ UnboundedTaskQueue, &queue, task });
    }

    for (producer_threads) |thread| {
        thread.join();
    }

    for (0..5) |i| {
        consumer_threads[i] = try std.Thread.spawn(.{}, consume, .{ UnboundedTaskQueue, &queue });
    }

    for (consumer_threads) |thread| {
        thread.join();
    }

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }
}

test "BoundedTaskQueue multi threaded push multi threaded pop" {
    // Ensure that multiple threads can push/pop from the queue.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var producer_threads: [5]std.Thread = undefined;
    var consumer_threads: [5]std.Thread = undefined;
    var queue = try BoundedTaskQueue.init(allocator);
    defer queue.deinit();

    for (0..test_arr.len) |i| {
        const args = .{ i, arr_slice };
        const FuncType = *const fn (usize, []u64) void;
        const ArgType = @TypeOf(args);
        const Context = struct {
            f: FuncType,
            args: ArgType,
        };

        const context = try allocator.create(Context);

        context.* = .{
            .f = square,
            .args = args,
        };

        const trampoline = struct {
            fn call(ctx: *const anyopaque) void {
                const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                @call(.auto, call_ctx.f, call_ctx.args);
            }
        }.call;

        // Create a task deallocation function
        const deinit_fn = struct {
            fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
                const casted: *const Context = @alignCast(@ptrCast(ctx));
                task_allocator.destroy(casted);
            }
        }.call;

        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        producer_threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ BoundedTaskQueue, &queue, task });
    }

    for (producer_threads) |thread| {
        thread.join();
    }

    for (0..5) |i| {
        consumer_threads[i] = try std.Thread.spawn(.{}, consume, .{ BoundedTaskQueue, &queue });
    }

    for (consumer_threads) |thread| {
        thread.join();
    }

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }
}

test "ThreadGroup wait test" {
    // Test a ThreadGroup. Can it wait on 10 threads to run?
    var arg = std.atomic.Value(i32).init(0);
    var threads: [10]std.Thread = undefined;
    var t_group = ThreadGroup.init();

    for (0..threads.len) |i| {
        t_group.increment();
        threads[i] = try std.Thread.spawn(.{}, thread_group_test, .{ &t_group, &arg });
    }

    t_group.wait();
    try std.testing.expect(t_group.counter.load(.acquire) == 0);

    for (threads) |thread| {
        thread.join();
    }

    try std.testing.expect(arg.load(.acquire) == 10);
}

test "Unbounded consumer/ThreadGroup wait test" {
    // Ensure that the consumer function actually consumes Tasks.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var producer_threads: [5]std.Thread = undefined;
    var consumer_threads: [5]std.Thread = undefined;
    var queue = try UnboundedTaskQueue.init(allocator);
    defer queue.deinit();
    var group = ThreadGroup.init();
    var running = true;

    for (0..test_arr.len) |i| {
        const args = .{ i, arr_slice };
        const FuncType = *const fn (usize, []u64) void;
        const ArgType = @TypeOf(args);
        const Context = struct {
            f: FuncType,
            args: ArgType,
        };

        const context = try allocator.create(Context);

        context.* = .{
            .f = square,
            .args = args,
        };

        const trampoline = struct {
            fn call(ctx: *const anyopaque) void {
                const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                @call(.auto, call_ctx.f, call_ctx.args);
            }
        }.call;

        // Create a task deallocation function
        const deinit_fn = struct {
            fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
                const casted: *const Context = @alignCast(@ptrCast(ctx));
                task_allocator.destroy(casted);
            }
        }.call;

        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        group.increment();
        producer_threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ UnboundedTaskQueue, &queue, task });
    }

    for (producer_threads) |thread| {
        thread.join();
    }

    for (0..5) |i| {
        const ctx = ConsumerContext(UnboundedTaskQueue){ .queue = &queue, .group = &group, .running = &running };
        consumer_threads[i] = try std.Thread.spawn(.{}, consumer, .{ UnboundedTaskQueue, ctx });
    }

    group.wait();
    running = false;

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }

    for (consumer_threads) |thread| {
        thread.join();
    }
}

test "Bounded consumer/ThreadGroup wait test" {
    // Ensure that the consumer function actually consumes Tasks.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var producer_threads: [5]std.Thread = undefined;
    var consumer_threads: [5]std.Thread = undefined;
    var queue = try BoundedTaskQueue.init(allocator);
    defer queue.deinit();
    var group = ThreadGroup.init();
    var running = true;

    for (0..test_arr.len) |i| {
        const args = .{ i, arr_slice };
        const FuncType = *const fn (usize, []u64) void;
        const ArgType = @TypeOf(args);
        const Context = struct {
            f: FuncType,
            args: ArgType,
        };

        const context = try allocator.create(Context);

        context.* = .{
            .f = square,
            .args = args,
        };

        const trampoline = struct {
            fn call(ctx: *const anyopaque) void {
                const call_ctx: *const Context = @alignCast(@ptrCast(ctx));
                @call(.auto, call_ctx.f, call_ctx.args);
            }
        }.call;

        // Create a task deallocation function
        const deinit_fn = struct {
            fn call(ctx: *const anyopaque, task_allocator: std.mem.Allocator) void {
                const casted: *const Context = @alignCast(@ptrCast(ctx));
                task_allocator.destroy(casted);
            }
        }.call;

        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        group.increment();
        producer_threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ BoundedTaskQueue, &queue, task });
    }

    for (producer_threads) |thread| {
        thread.join();
    }

    for (0..5) |i| {
        const ctx = ConsumerContext(BoundedTaskQueue){ .queue = &queue, .group = &group, .running = &running };
        consumer_threads[i] = try std.Thread.spawn(.{}, consumer, .{ BoundedTaskQueue, ctx });
    }

    group.wait();
    running = false;

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }

    for (consumer_threads) |thread| {
        thread.join();
    }
}

test "Unbounded ThreadPool test" {
    // Ensure that ThreadPool spawning/joining works
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var t_pool = try ThreadPool(UnboundedTaskQueue).init(allocator, .{ .n_jobs = test_arr.len });
    defer t_pool.deinit();

    for (0..test_arr.len) |i| {
        try t_pool.spawn(*const fn (usize, []u64) void, square, .{ i, arr_slice });
    }

    t_pool.join();

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }
}

test "Bounded ThreadPool test" {
    // Ensure that ThreadPool spawning/joining works
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var t_pool = try ThreadPool(BoundedTaskQueue).init(allocator, .{ .n_jobs = test_arr.len });
    defer t_pool.deinit();

    for (0..test_arr.len) |i| {
        try t_pool.spawn(*const fn (usize, []u64) void, square, .{ i, arr_slice });
    }

    t_pool.join();

    var j: u32 = 0;
    while (j < test_arr.len) : (j += 1) {
        const square_value = (j + 1) * (j + 1);
        try std.testing.expect(arr_slice[j] == square_value);
        try std.testing.expect(test_arr[j] == square_value);
    }
}

test "Unbounded ThreadPool performance test" {
    //Ensure that the ThreadPool is performant compared to sequential computation
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr1 = try allocator.alloc(u64, 4096 * 4096);
    var test_arr2 = try allocator.alloc(u64, 4096 * 4096);
    defer allocator.free(test_arr1);
    defer allocator.free(test_arr2);

    var i: u64 = 0;
    while (i < test_arr1.len) : (i += 1) {
        test_arr1[i] = i;
        test_arr2[i] = i;
    }

    var t_pool = try ThreadPool(UnboundedTaskQueue).init(allocator, .{ .n_jobs = 8 });
    defer t_pool.deinit();

    const seq_start = try std.time.Instant.now();
    for (0..test_arr1.len) |index| {
        square(index, test_arr1);
    }
    const seq_end = try std.time.Instant.now();
    const seq_elapsed: f64 = @floatFromInt(seq_end.since(seq_start));

    const par_start = try std.time.Instant.now();
    for (0..t_pool.n_jobs) |index| {
        try t_pool.spawn(*const fn (usize, []u64, usize) void, par_square, .{ index, test_arr2, t_pool.n_jobs });
    }
    t_pool.join();
    const par_end = try std.time.Instant.now();
    const par_elapsed: f64 = @floatFromInt(par_end.since(par_start));

    try std.testing.expect(par_elapsed < seq_elapsed);

    std.debug.print("Unbounded ThreadPool performance test:\n", .{});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
    std.debug.print("Sequential time to square large array: {d}\n", .{seq_elapsed});
    std.debug.print("Parallel time to square large array: {d}\n", .{par_elapsed});
    std.debug.print("Parallel time reduction: {d:.1}%\n", .{(seq_elapsed - par_elapsed) / seq_elapsed * 100.0});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
}

test "Bounded ThreadPool performance test" {
    //Ensure that the ThreadPool is performant compared to sequential computation
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr1 = try allocator.alloc(u64, 4096 * 4096);
    var test_arr2 = try allocator.alloc(u64, 4096 * 4096);
    defer allocator.free(test_arr1);
    defer allocator.free(test_arr2);

    var i: u64 = 0;
    while (i < test_arr1.len) : (i += 1) {
        test_arr1[i] = i;
        test_arr2[i] = i;
    }

    var t_pool = try ThreadPool(BoundedTaskQueue).init(allocator, .{ .n_jobs = 16 });
    defer t_pool.deinit();

    const seq_start = try std.time.Instant.now();
    for (0..test_arr1.len) |index| {
        square(index, test_arr1);
    }
    const seq_end = try std.time.Instant.now();
    const seq_elapsed: f64 = @floatFromInt(seq_end.since(seq_start));

    const par_start = try std.time.Instant.now();
    for (0..t_pool.n_jobs) |index| {
        try t_pool.spawn(*const fn (usize, []u64, usize) void, par_square, .{ index, test_arr2, t_pool.n_jobs });
    }
    t_pool.join();
    const par_end = try std.time.Instant.now();
    const par_elapsed: f64 = @floatFromInt(par_end.since(par_start));

    try std.testing.expect(par_elapsed < seq_elapsed);

    std.debug.print("Bounded ThreadPool performance test:\n", .{});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
    std.debug.print("Sequential time to square large array: {d}\n", .{seq_elapsed});
    std.debug.print("Parallel time to square large array: {d}\n", .{par_elapsed});
    std.debug.print("Parallel time reduction: {d:.1}%\n", .{(seq_elapsed - par_elapsed) / seq_elapsed * 100.0});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
}

test "Bounded/Unbounded ThreadPool performance test" {
    //Compare performance of Bounded and Unbounded thread pools.
    var b_gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var u_gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const b_allocator = b_gpa.allocator();
    const u_allocator = u_gpa.allocator();

    var b_a = try b_allocator.alloc(u64, 4096 * 4096 * 16);
    var b_b = try b_allocator.alloc(u64, 4096 * 4096 * 16);
    var u_a = try u_allocator.alloc(u64, 4096 * 4096 * 16);
    var u_b = try u_allocator.alloc(u64, 4096 * 4096 * 16);
    defer b_allocator.free(b_a);
    defer b_allocator.free(b_b);
    defer u_allocator.free(u_a);
    defer u_allocator.free(u_b);

    var i: u64 = 0;
    while (i < b_a.len) : (i += 1) {
        b_a[i] = i;
        b_b[i] = i;
        u_a[i] = i;
        u_b[i] = i;
    }

    var bt_pool = try ThreadPool(BoundedTaskQueue).init(b_allocator, .{ .n_jobs = 8 });
    var ut_pool = try ThreadPool(UnboundedTaskQueue).init(u_allocator, .{ .n_jobs = 8 });
    defer bt_pool.deinit();
    defer ut_pool.deinit();

    const b_start = try std.time.Instant.now();
    for (0..bt_pool.n_jobs) |index| {
        try bt_pool.spawn(*const fn (usize, []u64, []u64, usize) void, par_add, .{ index, b_a, b_b, bt_pool.n_jobs });
    }
    bt_pool.join();
    const b_end = try std.time.Instant.now();
    const b_elapsed = b_end.since(b_start);

    const u_start = try std.time.Instant.now();
    for (0..ut_pool.n_jobs) |index| {
        try ut_pool.spawn(*const fn (usize, []u64, []u64, usize) void, par_add, .{ index, b_a, b_b, bt_pool.n_jobs });
    }
    ut_pool.join();
    const u_end = try std.time.Instant.now();
    const u_elapsed = u_end.since(u_start);

    std.debug.print("Bounded/Unbounded ThreadPool performance test:\n", .{});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
    std.debug.print("Bounded ThreadPool large vector addition runtime: {d}\n", .{b_elapsed});
    std.debug.print("Unbounded ThreadPool large vector addition runtime: {d}\n", .{u_elapsed});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
}
