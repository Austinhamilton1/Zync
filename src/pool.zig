const std = @import("std");
const data = @import("data-structures.zig");
const utils = @import("test-utils.zig");

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
    /// deinit: *const fn (*const anyopaque, std.mem.Allocator) void: deinitialize the Task.
    /// allocator: std.mem.Allocator - The allocator used to allocate the Task.
    func: *const fn (*const anyopaque) void,
    ctx: *const anyopaque,
    deinit: *const fn (*const anyopaque, std.mem.Allocator) void,
    allocator: std.mem.Allocator,
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

/// Statistics about a Consumer thread.
/// id: usize - The id of the Consumer this is meant for.
/// executed_own: std.atomic.Value(usize) - How many times this Consumer executed from its own queue.
/// executed_stolen: std.atomic.Value(usize) - How many times this Consumer executed from another queue.
/// steal_failures: std.atomic.Value(usize) - How many times this Consumer failed to steal.
/// total_active_time: std.atomic.Value(usize) - How long this Consumer was active.
/// total_idle_time: std.atomic.Value(usize) - How long this Consumer was idle.
const ConsumerProfile = struct {
    id: usize,
    executed_own: std.atomic.Value(usize) = .init(0),
    executed_stolen: std.atomic.Value(usize) = .init(0),
    steal_failures: std.atomic.Value(usize) = .init(0),
    total_active_time: std.atomic.Value(usize) = .init(0),
    total_idle_time: std.atomic.Value(usize) = .init(0),
};

/// More robust consumer thread implementation.
pub const Consumer = struct {
    /// id: usize - A unique identifier for a thread.
    /// queue: *SPMCDeque - Work to be completed by the Consumer should be submitted here.
    /// worker: std.Thread - The actual thread doing work for the RobustThread.
    /// group: *ThreadGroup - A ThreadGroup that the Consumer is a part of.
    /// running: bool - Whether the Consumer should continue running or not.
    /// steal_from: []Consumer - Other consumers to steal from.
    /// prng: std.Random.DefaultPrng - A random number generator to pull consumer indexes from.
    /// profile: ?ConsumerProfile - If profiling is active, this will act as a benchmark for the Consumer otherwise it will be null.
    /// allocator: std.mem.Allocator - Used by the Consumer to allocate the work queue.
    id: usize,
    queue: *data.SPMCDeque(Task),
    worker: std.Thread,
    group: *ThreadGroup,
    running: bool,
    pool_size: usize,
    steal_from: []Consumer,
    prng: *std.Random.Xoshiro256,
    profile: ?ConsumerProfile,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize a Consumer
    /// Arguments:
    ///     allocator: std.mem.Allocator - An allocator to create the Consumer's work queue.
    ///     id: usize - A !!!UNIQUE!!! identifier of the Consumer.
    ///     group: *ThreadGroup - A ThreadGroup to track completion of Tasks.
    ///     pool_size: usize - The size fo the pool the Consumer is going to be in (n_jobs).
    ///     queue_size: usize - The size of the queue.
    ///     profiling: bool - Whether this Consumer should profile or not.
    pub fn init(allocator: std.mem.Allocator, id: usize, group: *ThreadGroup, pool_size: usize, queue_size: usize, profiling: bool) !Self {
        // Allocate the per-worker queue
        const queue = try allocator.create(data.SPMCDeque(Task));
        queue.* = try data.SPMCDeque(Task).init(allocator, queue_size);

        // Allocate the steal_from field (this needs to be initialized in a separate call)
        const steal_from = try allocator.alloc(Consumer, pool_size);

        // Allocate the random number generator for selecting other consumers to steal from
        const prng = try allocator.create(std.Random.DefaultPrng);
        prng.* = std.Random.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.posix.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });

        // Set the profile for the Consumer.
        var profile: ?ConsumerProfile = null;
        if (profiling) {
            profile = .{ .id = id };
        }

        return Self{ .id = id, .queue = queue, .worker = undefined, .group = group, .running = true, .pool_size = pool_size, .steal_from = steal_from, .prng = prng, .profile = profile, .allocator = allocator };
    }

    /// Destroy a consumer.
    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self.queue);
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
    fn consume(self: *Self) !void {
        // Check if the Consumer should continue running
        while (self.running) {
            // See if a task is ready
            const task_ready = self.queue.pop();

            if (task_ready) |task| {
                // Task is ready, run the task.
                task.func(task.ctx);
                task.deinit(task.ctx, task.allocator);

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
                while (true) {
                    const has_task = other.queue.steal() catch {
                        // Another Consumer stole first, ignore and try again
                        std.time.sleep(1);
                        break;
                    };

                    // Deque might be empty
                    if (has_task) |task| {
                        // Task was successfully stolen, run the task
                        task.func(task.ctx);
                        task.deinit(task.ctx, task.allocator);

                        // Let the ThreadGroup know that this task is complete
                        other.group.decrement();
                    } else {
                        // Give up a cycle before trying again
                        std.time.sleep(1);
                        break;
                    }
                }
            }
        }
    }

    // Consume function with profiling set to on.
    fn consume_with_profile(self: *Self) !void {
        // Check if the Consumer should continue running
        while (self.running) {
            // See if a task is ready
            const task_ready = self.queue.pop();

            // Used to calculate task time and idle time respectively
            var task_time: u64 = 0;
            const total_start: std.time.Instant = try std.time.Instant.now();

            if (task_ready) |task| {
                // Task is ready, run the task.
                _ = self.profile.?.executed_own.fetchAdd(1, .acq_rel);
                const task_start = try std.time.Instant.now();
                task.func(task.ctx);
                task.deinit(task.ctx, task.allocator);
                const task_end = try std.time.Instant.now();
                task_time = task_end.since(task_start);

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
                while (true) {
                    const has_task = other.queue.steal() catch {
                        // Another Consumer stole first, ignore and try again
                        std.time.sleep(1);
                        _ = self.profile.?.steal_failures.fetchAdd(1, .acq_rel);
                        const total_end = try std.time.Instant.now();
                        const idle_time = total_end.since(total_start) - task_time;
                        _ = self.profile.?.total_idle_time.fetchAdd(idle_time, .acq_rel);
                        _ = self.profile.?.total_active_time.fetchAdd(task_time, .acq_rel);
                        break;
                    };

                    // Deque might be empty
                    if (has_task) |task| {
                        // Task was successfully stolen, run the task
                        _ = self.profile.?.executed_stolen.fetchAdd(1, .acq_rel);
                        const task_start = try std.time.Instant.now();
                        task.func(task.ctx);
                        task.deinit(task.ctx, task.allocator);
                        const task_end = try std.time.Instant.now();
                        task_time = task_end.since(task_start);

                        // Let the ThreadGroup know that this task is complete
                        other.group.decrement();
                    } else {
                        // Sleep for a short period before trying again
                        std.time.sleep(1);
                        _ = self.profile.?.steal_failures.fetchAdd(1, .acq_rel);
                        break;
                    }
                }
            }

            // Calculate the total idle time for the current consumer loop
            const total_end = try std.time.Instant.now();
            const idle_time = total_end.since(total_start) - task_time;
            _ = self.profile.?.total_idle_time.fetchAdd(idle_time, .acq_rel);
            _ = self.profile.?.total_active_time.fetchAdd(task_time, .acq_rel);
        }
    }

    // Start up a Consumer
    pub fn start(self: *Self) !void {
        self.running = true;
        if (self.profile) |_| {
            self.worker = try std.Thread.spawn(.{}, Consumer.consume_with_profile, .{self});
        } else {
            self.worker = try std.Thread.spawn(.{}, Consumer.consume, .{self});
        }
    }

    // Stop a Consumer (allows the Consumer to exit gracefully).
    pub fn stop(self: *Self) void {
        self.running = false;
        self.worker.join();
    }
};

/// Context needed to create a ThreadPool.
/// n_jobs: usize - How many threads to spawn (<= 64).
/// queue_size: uszie - The size of the queue.
/// profiling: bool - Should the ThreadPool enable profiling?
/// ignore_errors: bool - If a thread gives an error, should the pool ignore it or handle it?
pub const ThreadPoolContext = struct {
    n_jobs: usize,
    queue_size: usize = 4096,
    profiling: bool = false,
    handle_errors: bool = false,
};

/// Profile for a ThreadPool
pub const ThreadPoolProfile = struct {
    threads_used: usize = 0,
    tasks_scheduled: usize = 0,
    tasks_completed: usize = 0,
    local_execution: usize = 0,
    stolen_execution: usize = 0,
    stolen_failures: usize = 0,
    total_time: usize = 0,
    total_idle_time: usize = 0,
    total_active_time: usize = 0,
    tasks_per_second: usize = 0,
};

/// A thread pool object. Keeps and manages a list of threads
/// to consume tasks as they are produced.
pub const ThreadPool = struct {
    /// n_jobs: usize - How many threads the ThreadPool is managing.
    /// handle_errors: bool - Should the pool allow error functions?
    /// workers: []Thread - The worker threads.
    /// queue: *TaskQueue - The queue that tasks come into the pool through.
    /// group: *ThreadGroup - A ThreadGroup that observes the state of the pool's threads.
    /// profile: ?ThreadPoolProfile - Used for profiling a ThreadPool, null if profiling is false.
    /// running: bool - If the ThreadPool is running.
    /// allocator: std.mem.Allocator - Allocator shared by TaskQueue (MUST BE THREAD SAFE!!!)
    n_jobs: usize,
    handle_errors: bool,
    workers: []Consumer,
    group: *ThreadGroup,
    queue: *data.VyukovQueue(Task),
    profile: ?ThreadPoolProfile,
    running: bool,
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
            workers[i] = try Consumer.init(allocator, i, group, ctx.n_jobs, ctx.queue_size, ctx.profiling);
        }
        for (0..workers.len) |i| {
            try workers[i].set_steal_pool(workers);
        }

        // Initialize the queue and then give it the allocator
        const queue = try allocator.create(data.VyukovQueue(Task));
        queue.* = try data.VyukovQueue(Task).init(allocator, ctx.queue_size);

        var profile: ?ThreadPoolProfile = null;
        if (ctx.profiling) {
            profile = .{ .threads_used = ctx.n_jobs };
        }

        return Self{ .n_jobs = ctx.n_jobs, .handle_errors = ctx.handle_errors, .workers = workers, .group = group, .queue = queue, .profile = profile, .running = false, .allocator = allocator };
    }

    /// Destroy an instance of a ThreadPool.
    /// Join all threads and clean up memory used.
    pub fn deinit(self: *Self) void {
        // Join all threads.
        if (self.running) {
            for (0..self.n_jobs) |i| {
                self.workers[i].stop();
            }
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
            .allocator = self.allocator,
        };

        // Try to push the task to the queue
        try self.queue.push(task);
        errdefer self.allocator.destroy(context);

        // If profiling, add one to the scheduled tasks count
        if (self.profile) |*profile| {
            profile.*.tasks_scheduled += 1;
        }
    }

    /// Populate a consumer thread with Tasks to run.
    /// Arguments:
    ///     id: usize - The id of the Consumer.
    fn populate_consumer_tasks(self: *Self, current_consumer: *std.atomic.Value(usize)) void {
        // Add a task to each Consumer in a round-robin fashion
        while (self.queue.pop()) |task| {
            const id = current_consumer.fetchAdd(1, .acq_rel);
            self.workers[id % self.n_jobs].queue.push(task) catch {
                break;
            };

            // Increament the ThreadGroup reference count
            self.group.increment();
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
        self.running = true;
        for (0..self.workers.len) |i| {
            try self.workers[i].start();
        }
    }

    /// Wait for all threads to enter an idle state.
    pub fn join(self: *Self) void {
        self.group.wait();
    }

    // Update the ThreadPool's profile
    pub fn stat(self: *Self) void {
        if (self.profile) |*pool_profile| {
            // Clear the current stats.
            pool_profile.*.local_execution = 0;
            pool_profile.*.stolen_execution = 0;
            pool_profile.*.tasks_completed = 0;
            pool_profile.*.stolen_failures = 0;
            pool_profile.*.total_active_time = 0;
            pool_profile.*.total_idle_time = 0;
            pool_profile.*.total_time = 0;
            pool_profile.*.tasks_per_second = 0;

            var total_time: usize = 0;

            // Aggregate worker stats
            for (self.workers) |consumer| {
                // Calculate values
                const consumer_profile = consumer.profile.?;
                const local_execution = consumer_profile.executed_own.load(.acquire);
                const stolen_execution = consumer_profile.executed_stolen.load(.acquire);
                const total_tasks = local_execution + stolen_execution;
                const total_active_time = consumer_profile.total_active_time.load(.acquire);
                const total_idle_time = consumer_profile.total_idle_time.load(.acquire);

                // Assign values
                total_time += (total_active_time + total_idle_time);
                pool_profile.*.local_execution += local_execution;
                pool_profile.*.stolen_execution += stolen_execution;
                pool_profile.*.tasks_completed += local_execution + stolen_execution;
                pool_profile.*.stolen_failures += consumer_profile.steal_failures.load(.acquire);
                pool_profile.*.total_active_time += total_active_time;
                pool_profile.*.total_idle_time += total_idle_time;
                pool_profile.*.tasks_per_second += total_tasks * 1_000_000_000_000;
            }

            // Calculate with total time
            pool_profile.*.total_time = total_time;
            pool_profile.*.tasks_per_second /= total_time;
        }
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

    /// Test a lightweight CPU bound task
    fn spin() void {
        var n: usize = 1000;
        while (n > 0) : (n -= 1) {
            _ = n * n;
        }
    }

    // Helper function for fib
    fn fib_recursive(n: usize) usize {
        if (n <= 1) {
            return n;
        }
        return test_func.fib_recursive(n - 1) + test_func.fib_recursive(n - 2);
    }

    /// Test a medium CPU bound task
    fn fib() void {
        _ = test_func.fib_recursive(25);
    }

    /// Test simulated I/O
    fn sleep() void {
        std.time.sleep(1_000_000); // 1ms
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
        .allocator = allocator,
    };

    task.func(task.ctx);

    try std.testing.expectEqual(11, test_int);
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
            .allocator = allocator,
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

    try std.testing.expectEqual(10, arg.load(.acquire));
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

    utils.header("ThreadPool correctness test");
    std.debug.print("Time to complete squaring: {d}\n", .{elapsed});
    utils.long_line();
}

test "ThreadPool performance test" {
    // Test the ThreadPool to ensure performance
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
    const seq_elapsed: f64 = @floatFromInt(seq_end_time.since(seq_start_time));

    const n_jobs: usize = 32;
    var t_pool = try ThreadPool.init(allocator, .{ .n_jobs = n_jobs, .profiling = true });
    for (0..n_jobs) |j| {
        try t_pool.spawn(*const fn (usize, []u64, usize) void, test_func.par_square, .{ j, par_arr, n_jobs });
    }

    try t_pool.schedule();
    const par_start_time = try std.time.Instant.now();
    try t_pool.start();
    t_pool.join();
    const par_end_time = try std.time.Instant.now();
    const par_elapsed: f64 = @floatFromInt(par_end_time.since(par_start_time));

    i = 0;
    while (i < par_arr.len) : (i += 1) {
        const square_value = i * i;
        try std.testing.expect(seq_arr[i] == square_value);
        try std.testing.expect(par_arr[i] == square_value);
    }

    utils.header("ThreadPool performance test");
    std.debug.print("Time to complete squaring on 4096 * 4096 array (sequential): {d}\n", .{seq_elapsed});
    std.debug.print("Time to complete squaring on 4096 * 4096 array (parallel): {d}\n", .{par_elapsed});
    std.debug.print("Percent reduction: {d:0.1}%\n", .{(seq_elapsed - par_elapsed) * 100 / seq_elapsed});
    utils.long_line();
}

test "ThreadPool benchmarks" {
    // Benchmark the ThreadPool on different Tasks
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var simple_t_pool = try ThreadPool.init(allocator, .{ .n_jobs = 1, .profiling = true, .queue_size = 16384 });

    for (0..10000) |_| {
        try simple_t_pool.spawn(*const fn () void, test_func.spin, .{});
    }

    try simple_t_pool.schedule();
    try simple_t_pool.start();
    simple_t_pool.join();

    utils.header("ThreadPool benchmarks");

    simple_t_pool.stat();
    var profile = simple_t_pool.profile.?;
    std.debug.print("Simple ThreadPool Benchmark\n", .{});
    utils.short_line();
    std.debug.print("Threads used: {d}\nTasks scheduled: {d}\nTasks completed: {d}\nLocal Execution: {d}\nStolen tasks: {d}\nSteal failures: {d}\nTotal active time: {d}\nTotal idle time: {d}\nTotal time: {d}\nEstimated tasks/sec: {d}\n", .{ profile.threads_used, profile.tasks_scheduled, profile.tasks_completed, profile.local_execution, profile.stolen_execution, profile.stolen_failures, profile.total_active_time, profile.total_idle_time, profile.total_time, profile.tasks_per_second });
    utils.short_line();

    simple_t_pool.deinit();

    const n_jobs: usize = 16;
    var t_pool = try ThreadPool.init(allocator, .{ .n_jobs = n_jobs, .profiling = true, .queue_size = 16384 });
    defer t_pool.deinit();

    for (0..10000) |_| {
        try t_pool.spawn(*const fn () void, test_func.spin, .{});
    }

    try t_pool.schedule();
    try t_pool.start();
    t_pool.join();

    t_pool.stat();
    profile = t_pool.profile.?;
    std.debug.print("ThreadPool Spin Benchmark\n", .{});
    utils.short_line();
    std.debug.print("Threads used: {d}\nTasks scheduled: {d}\nTasks completed: {d}\nLocal Execution: {d}\nStolen tasks: {d}\nSteal failures: {d}\nTotal active time: {d}\nTotal idle time: {d}\nTotal time: {d}\nEstimated tasks/sec: {d}\n", .{ profile.threads_used, profile.tasks_scheduled, profile.tasks_completed, profile.local_execution, profile.stolen_execution, profile.stolen_failures, profile.total_active_time, profile.total_idle_time, profile.total_time, profile.tasks_per_second });
    utils.short_line();

    for (0..10000) |_| {
        try t_pool.spawn(*const fn () void, test_func.fib, .{});
    }

    try t_pool.schedule();
    try t_pool.start();
    t_pool.join();

    t_pool.stat();
    profile = t_pool.profile.?;
    std.debug.print("ThreadPool Fib Benchmark\n", .{});
    utils.short_line();
    std.debug.print("Threads used: {d}\nTasks scheduled: {d}\nTasks completed: {d}\nLocal Execution: {d}\nStolen tasks: {d}\nSteal failures: {d}\nTotal active time: {d}\nTotal idle time: {d}\nTotal time: {d}\nEstimated tasks/sec: {d}\n", .{ profile.threads_used, profile.tasks_scheduled, profile.tasks_completed, profile.local_execution, profile.stolen_execution, profile.stolen_failures, profile.total_active_time, profile.total_idle_time, profile.total_time, profile.tasks_per_second });
    utils.short_line();

    for (0..10000) |_| {
        try t_pool.spawn(*const fn () void, test_func.sleep, .{});
    }

    try t_pool.schedule();
    try t_pool.start();
    t_pool.join();

    t_pool.stat();
    profile = t_pool.profile.?;
    std.debug.print("ThreadPool Sleep Benchmark\n", .{});
    utils.short_line();
    std.debug.print("Threads used: {d}\nTasks scheduled: {d}\nTasks completed: {d}\nLocal Execution: {d}\nStolen tasks: {d}\nSteal failures: {d}\nTotal active time: {d}\nTotal idle time: {d}\nTotal time: {d}\nEstimated tasks/sec: {d}\n", .{ profile.threads_used, profile.tasks_scheduled, profile.tasks_completed, profile.local_execution, profile.stolen_execution, profile.stolen_failures, profile.total_active_time, profile.total_idle_time, profile.total_time, profile.tasks_per_second });
    utils.short_line();
}
