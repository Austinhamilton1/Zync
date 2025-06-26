const std = @import("std");
const data = @import("lock-free.zig");

/// Potential ThreadPool Errors
pub const ThreadPoolError = error{
    TooManyThreads, // Trying to spawn more than 1024 threads
    MismatchedWorkStealing, // The steal_pool is instantiated as a different size then the passed in slice
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

/// More robust consumer thread implementation.
pub const Consumer = struct {
    /// id: usize - A unique identifier for a thread.
    /// queue: *ChaseLevDeque - Work to be completed by the Consumer should be submitted here.
    /// worker: std.Thread - The actual thread doing work for the RobustThread.
    /// group: *ThreadGroup - A ThreadGroup that the Consumer is a part of.
    /// running: *bool - Whether the Consumer should continue running or not.
    /// steal_from: []Consumer - Other consumers to steal from.
    /// prng: std.Random.DefaultPrng - A random number generator to pull consumer indexes from.
    /// allocator: std.mem.Allocator - Used by the Consumer to allocate the work queue.
    id: usize,
    queue: *data.ChaseLevDeque(Task),
    worker: std.Thread,
    group: *ThreadGroup,
    running: *bool,
    pool_size: usize,
    steal_from: []Consumer,
    prng: *std.Random.Xoshiro256,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize a Consumer
    /// Arguments:
    ///     allocator: std.mem.Allocator - An allocator to create the Consumer's work queue.
    ///     id: usize - A !!!UNIQUE!!! identifier of the Consumer.
    ///     group: *ThreadGroup - A ThreadGroup to track completion of Tasks.
    ///     pool_size - The size fo the pool the Consumer is going to be in (n_jobs).
    pub fn init(allocator: std.mem.Allocator, id: usize, group: *ThreadGroup, pool_size: usize) !Self {
        // Allocate the per-worker queue
        const queue = try allocator.create(data.ChaseLevDeque(Task));
        queue.* = try data.ChaseLevDeque(Task).init(allocator);

        // Allocate the running boolean sentinel value
        const running = try allocator.create(bool);

        // Allocate the steal_from field (this needs to be initialized in a separate call)
        const steal_from = try allocator.alloc(Consumer, pool_size);

        // Allocate the random number generator for selecting other consumers to steal from
        const prng = try allocator.create(std.Random.DefaultPrng);
        prng.* = std.Random.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.posix.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });

        return Self{ .id = id, .queue = queue, .worker = undefined, .group = group, .running = running, .pool_size = pool_size, .steal_from = steal_from, .prng = prng, .allocator = allocator };
    }

    /// Destroy a consumer.
    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self.queue);
        self.allocator.destroy(self.running);
        self.allocator.free(self.steal_from);
        self.allocator.destroy(self.prng);
    }

    /// Set the steal_from pool. This should be done right after all consumers have been
    /// allocated.
    /// Arguments:
    ///     consumers: []Consumer - A slice of Consumers with pool_size
    pub fn set_steal_pool(self: *Self, consumers: []Consumer) !void {
        if (consumers.len != self.pool_size) {
            return ThreadPoolError.MismatchedWorkStealing;
        }
        self.steal_from = consumers;
    }

    /// Continuously consume tasks from the Consumer's task queue. If the task
    /// queue is empty, steal from other workers.
    fn consume(self: *Self) void {
        // Check if the Consumer should continue running
        while (self.running.*) {
            // See if a task is ready
            const task_ready = self.queue.pop();
            if (task_ready) |task| {
                // Task is ready, run the task.
                task.func(task.ctx);
                task.deinit(task.ctx, self.allocator);

                // Let the ThreadGroup know that this task is complete
                self.group.decrement();
            } else {
                // No Task ready, need to steal a Task

                // Try to steal a Task from a random consumer, if it is this consumer
                // steal from the next consumer
                const rand = self.prng.random();
                var idx = rand.intRangeAtMostBiased(usize, 0, self.pool_size - 1);
                if (idx == self.id) {
                    idx = (self.id + 1) % self.pool_size;
                }

                // Now we have a consumer to steal from
                const other = &self.steal_from[idx];
                const has_task = other.queue.steal() catch {
                    // Another Consumer stole first, ignore and try again
                    std.time.sleep(10);
                    continue;
                };

                // Deque might be empty
                if (has_task) |task| {
                    // Task was successfully stolen, run the task
                    task.func(task.ctx);
                    task.deinit(task.ctx, other.allocator);

                    // Let the ThreadGroup know that this task is complete
                    other.group.decrement();
                } else {
                    // Sleep for a short period before trying again
                    std.time.sleep(10);
                }
            }
        }
    }

    // Start up a Consumer
    pub fn start(self: *Self) !void {
        self.running.* = true;
        self.worker = try std.Thread.spawn(.{}, Consumer.consume, .{self});
    }

    // Stop a Consumer (allows the Consumer to exit gracefully).
    pub fn stop(self: *Self) void {
        self.running.* = false;
        self.worker.join();
    }
};

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
    /// queue: *TaskQueue - The queue that tasks come into the pool through.
    /// group: *ThreadGroup - A ThreadGroup that observes the state of the pool's threads.
    /// allocator: std.mem.Allocator - Allocator shared by TaskQueue (MUST BE THREAD SAFE!!!)
    n_jobs: usize,
    handle_errors: bool,
    workers: []Consumer,
    group: *ThreadGroup,
    queue: *data.VyukovQueue(Task),
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
        if (ctx.n_jobs > 64) {
            return ThreadPoolError.TooManyThreads;
        }

        // Initialize the ThreadGroup. This is a global thread group that observes all threads.
        const group = try allocator.create(ThreadGroup);
        group.* = ThreadGroup.init();

        // Declare worker threads
        const workers = try allocator.alloc(Consumer, ctx.n_jobs);
        for (0..workers.len) |i| {
            workers[i] = try Consumer.init(allocator, i, group, ctx.n_jobs);
        }
        for (0..workers.len) |i| {
            try workers[i].set_steal_pool(workers);
        }

        // Initialize the queue and then give it the allocator
        const queue = try allocator.create(data.VyukovQueue(Task));
        queue.* = try data.VyukovQueue(Task).init(allocator);

        return Self{ .n_jobs = ctx.n_jobs, .handle_errors = ctx.handle_errors, .workers = workers, .group = group, .queue = queue, .allocator = allocator };
    }

    /// Destroy an instance of a ThreadPool.
    /// Join all threads and clean up memory used.
    pub fn deinit(self: *Self) void {
        // Join all threads.
        for (self.workers) |consumer| {
            consumer.stop();
        }

        // Clean up allocated memory
        // Deallocate in reverse order of allocation
        self.queue.deinit();
        self.allocator.destroy(self.group);
        self.allocator.free(self.workers);
        self.allocator.destroy(self.queue);
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

    /// Populate a consumer thread with Tasks to run.
    /// Arguments:
    ///     id: usize - The id of the Consumer.
    fn populate_consumer_tasks(self: *Self, current_consumer: *std.atomic.Value(usize)) void {
        while (self.queue.pop()) |task| {
            const id = current_consumer.fetchAdd(1, .acq_rel);
            self.workers[id % self.n_jobs].queue.push(task) catch {
                break;
            };
        }
    }

    /// Schedule all the tasks currently in the queue to the worker threads.
    pub fn schedule(self: *Self) !void {
        // For each worker, start adding tasks to the worker's queue
        var producer_threads = try self.allocator.alloc(std.Thread, self.n_jobs);
        var current_consumer = std.atomic.Value(usize).init(0);
        defer self.allocator.free(producer_threads);
        for (0..self.n_jobs) |i| {
            producer_threads[i] = try std.Thread.spawn(.{}, ThreadPool.populate_consumer_tasks, .{ self, &current_consumer });
        }
        for (producer_threads) |thread| {
            thread.join();
        }
    }

    /// Start the thread pool workers.
    pub fn start(self: *Self) !void {
        for (0..self.workers.len) |i| {
            try self.workers[i].start();
        }
    }

    /// Wait for all threads to enter an idle state.
    pub fn join(self: *Self) void {
        self.group.wait();
    }
};

const test_func = struct {
    /// Add one to an argument.
    /// Arguments:
    ///     arg: *i32 - An integer to add one to.
    fn add_one(arg: *i32) void {
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
};

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
        .f = test_func.add_one,
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
            .f = test_func.square,
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

test "ThreadGroup wait test" {
    // Test a ThreadGroup. Can it wait on 10 threads to run?
    var arg = std.atomic.Value(i32).init(0);
    var threads: [10]std.Thread = undefined;
    var t_group = ThreadGroup.init();

    for (0..threads.len) |i| {
        t_group.increment();
        threads[i] = try std.Thread.spawn(.{}, test_func.thread_group_test, .{ &t_group, &arg });
    }

    t_group.wait();
    try std.testing.expect(t_group.counter.load(.acquire) == 0);

    for (threads) |thread| {
        thread.join();
    }

    try std.testing.expect(arg.load(.acquire) == 10);
}

test "ThreadPool correctness test" {
    // Test the ThreadPool to ensure it is working
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var test_arr: [1024]u64 = undefined;
    var i: u64 = 0;
    while (i < test_arr.len) : (i += 1) {
        test_arr[i] = i;
    }
    const test_slice = test_arr[0..];

    const n_jobs: usize = 8;
    var t_pool = try ThreadPool.init(allocator, .{ .n_jobs = n_jobs });
    for (0..n_jobs) |j| {
        try t_pool.spawn(*const fn (usize, []u64, usize) void, test_func.par_square, .{ j, test_slice, n_jobs });
    }

    try t_pool.schedule();
    const start_time = try std.time.Instant.now();
    try t_pool.start();
    t_pool.join();
    const end_time = try std.time.Instant.now();
    const elapsed = end_time.since(start_time);

    i = 0;
    while (i < test_arr.len) : (i += 1) {
        const square_value = i * i;
        try std.testing.expect(test_arr[i] == square_value);
    }

    std.debug.print("ThreadPool correctness test\n", .{});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
    std.debug.print("Time to complete squaring: {d}\n", .{elapsed});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
}

test "ThreadPool performance test" {
    // Test the ThreadPool to ensure performance
    // Test the ThreadPool to ensure it is working
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var seq_arr = try allocator.alloc(u64, 4096 * 4096);
    var par_arr = try allocator.alloc(u64, 4096 * 4096);
    defer allocator.free(seq_arr);
    defer allocator.free(par_arr);

    var i: u64 = 0;
    while (i < seq_arr.len) : (i += 1) {
        seq_arr[i] = i;
        par_arr[i] = i;
    }

    const seq_start_time = try std.time.Instant.now();
    for (0..seq_arr.len) |j| {
        seq_arr[j] *= seq_arr[j];
    }
    const seq_end_time = try std.time.Instant.now();
    const seq_elapsed = seq_end_time.since(seq_start_time);

    const n_jobs: usize = 16;
    var t_pool = try ThreadPool.init(allocator, .{ .n_jobs = n_jobs });
    for (0..n_jobs) |j| {
        try t_pool.spawn(*const fn (usize, []u64, usize) void, test_func.par_square, .{ j, par_arr, n_jobs });
    }

    try t_pool.schedule();
    const par_start_time = try std.time.Instant.now();
    try t_pool.start();
    t_pool.join();
    const par_end_time = try std.time.Instant.now();
    const par_elapsed = par_end_time.since(par_start_time);

    i = 0;
    while (i < par_arr.len) : (i += 1) {
        const square_value = i * i;
        try std.testing.expect(seq_arr[i] == square_value);
        try std.testing.expect(par_arr[i] == square_value);
    }

    std.debug.print("ThreadPool performance test\n", .{});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
    std.debug.print("Time to complete squaring on 4096 * 4096 array (sequential): {d}\n", .{seq_elapsed});
    std.debug.print("Time to complete squaring on 4096 * 4096 array (parallel): {d}\n", .{par_elapsed});
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
}
