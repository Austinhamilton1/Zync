const std = @import("std");

/// Potential ThreadPool Errors
pub const ThreadPoolError = error{
    TooManyThreads, // Trying to spawn more than 1024 threads
    ForkError, // One of the spawns did not work in the fork method; discard results
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

/// A TaskQueue to add Tasks to. The Tasks wait
/// in the queue until they are ready to be used by
/// a Thread in the ThreadPool. This TaskQueue is implemented
/// with a lock-free linked list backend to make it extremely
/// performant.
const TaskQueue = struct {
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

    /// Initialize a TaskQueue object.
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

    /// Destroy a TaskQueue object.
    pub fn deinit(self: *Self) void {
        // Traverse the linked list and destroy each node.
        var ptr = self.head.load(.acquire);
        while (ptr) |node| {
            ptr = node.next.load(.acquire);
            self.allocator.destroy(node);
        }
    }

    /// Push a Task to the back of the TaskQueue.
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

    /// Pop a Task from the front of the TaskQueue.
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
/// queue: *TaskQueue - Consumer will grab tasks from this queue.
/// group: *ThreadGroup - Consumer will update this group as it completes tasks.
/// running: *bool - Consumer will check this to see if it needs to continue running.
/// allocator: std.mem.Allocator - The allocator used to instantiate the function context.
const ConsumerContext = struct {
    queue: *TaskQueue,
    group: *ThreadGroup,
    running: *bool,
};

/// Consumer for the ThreadPool.
/// Arguments:
///     ctx: ConsumerContext - The context to run the consumer under.
fn consumer(ctx: ConsumerContext) void {
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
pub const ThreadPool = struct {
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
    queue: *TaskQueue,
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
        const queue = try allocator.create(TaskQueue);
        queue.* = try TaskQueue.init(allocator);

        const consumer_ctx = ConsumerContext{
            .queue = queue,
            .group = group,
            .running = running,
        };

        // Instantiate worker threads
        for (0..ctx.n_jobs) |i| {
            workers[i] = try std.Thread.spawn(.{}, consumer, .{consumer_ctx});
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

/// Push a task to a TaskQueue.
/// Arguments:
///     queue: *TaskQueue - The queue to push to.
///     task: Task - The task to push.
fn push_task_to_queue(queue: *TaskQueue, task: Task) !void {
    try queue.push(task);
}

/// Consume a task from a TaskQueue.
/// Arguments:
///     queue - *TaskQueue - The queue to pop from.
fn consume(queue: *TaskQueue) void {
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

test "TaskQueue single threaded push/pop" {
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

    var queue = try TaskQueue.init(allocator);
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

test "TaskQueue fill/empty/fill" {
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

    var queue = try TaskQueue.init(allocator);
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

test "TaskQueue multi threaded push single threaded pop" {
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

    var queue = try TaskQueue.init(allocator);
    defer queue.deinit();
    var threads: [10]std.Thread = undefined;

    for (0..10) |i| {
        const task = Task{
            .func = trampoline,
            .ctx = context,
            .deinit = deinit_fn,
        };

        threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ &queue, task });
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

test "TaskQueue single threaded push multi threaded pop" {
    // Ensure that multiple threads can pop from the queue
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var threads: [5]std.Thread = undefined;
    var queue = try TaskQueue.init(allocator);
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
        threads[i] = try std.Thread.spawn(.{}, consume, .{&queue});
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

test "TaskQueue multi threaded push multi threaded pop" {
    // Ensure that multiple threads can push/pop from the queue.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var producer_threads: [5]std.Thread = undefined;
    var consumer_threads: [5]std.Thread = undefined;
    var queue = try TaskQueue.init(allocator);
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

        producer_threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ &queue, task });
    }

    for (producer_threads) |thread| {
        thread.join();
    }

    for (0..5) |i| {
        consumer_threads[i] = try std.Thread.spawn(.{}, consume, .{&queue});
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

test "consumer/ThreadGroup wait test" {
    // Ensure that the consumer function actually consumes Tasks.
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var producer_threads: [5]std.Thread = undefined;
    var consumer_threads: [5]std.Thread = undefined;
    var queue = try TaskQueue.init(allocator);
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
        producer_threads[i] = try std.Thread.spawn(.{}, push_task_to_queue, .{ &queue, task });
    }

    for (producer_threads) |thread| {
        thread.join();
    }

    for (0..5) |i| {
        const ctx = ConsumerContext{ .queue = &queue, .group = &group, .running = &running };
        consumer_threads[i] = try std.Thread.spawn(.{}, consumer, .{ctx});
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

test "ThreadPool test" {
    // Ensure that ThreadPool spawning/joining works
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr = [_]u64{ 1, 2, 3, 4, 5 };
    const arr_slice = test_arr[0..];

    var t_pool = try ThreadPool.init(allocator, .{ .n_jobs = test_arr.len });
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

test "ThreadPool performance test" {
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

    var t_pool = try ThreadPool.init(allocator, .{ .n_jobs = 8 });
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

    std.debug.print("ThreadPool performance test:\n", .{});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
    std.debug.print("Sequential time to square large array: {d}\n", .{seq_elapsed});
    std.debug.print("Parallel time to square large array: {d}\n", .{par_elapsed});
    std.debug.print("Parallel time reduction: {d:.1}%\n", .{(seq_elapsed - par_elapsed) / seq_elapsed * 100.0});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
}
