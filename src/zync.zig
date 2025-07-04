const std = @import("std");
const pool = @import("pool.zig");
const util = @import("test-utils.zig");

/// Determines what hardware/software will run when iterating.
/// cpu - Run on a CPU based ThreadPool.
/// opencl - Run on the GPU using OpenCL shaders.
pub const backends = enum {
    cpu,
    opencl,
};

pub fn Zync(comptime T: type) type {
    return struct {
        /// A parallel iterator that can complete work on a slice
        /// of data concurrently.
        const ParallelIterator = struct {
            /// pool: pool.ThreadPool - A thread pool used for divvying out tasks on the slice.
            /// tasks: usize - How many tasks the ThreadPool should complete in total.
            /// data: []T - The data to do work on.
            /// backend: backends - The backend of the ParallelIterator.
            /// allocator: std.mem.Allocator - An allocator to manage the iterator.
            t_pool: pool.ThreadPool,
            tasks: usize,
            data: []T,
            backend_type: backends,
            allocator: std.mem.Allocator,

            const ParallelIteratorSelf = @This();

            /// Initialize a ParallelIterator.
            /// Arguments:
            ///     data: []T - A slice to initialize the iterator on.
            ///     allocator: std.mem.Allocator - An allocator to manage the internal ThreadPool.
            pub fn init(data: []T, allocator: std.mem.Allocator) ParallelIteratorSelf {
                return ParallelIteratorSelf{ .t_pool = undefined, .tasks = 0, .data = data, .backend_type = undefined, .allocator = allocator };
            }

            /// Set the backend of the ParallelIterator
            /// Arguments:
            ///     backend_type: backends - The backend to run the parallel computation on.
            /// Returns:
            ///     A ParallelIterator (for easy chaining).
            pub fn backend(self: *ParallelIteratorSelf, backend_type: backends) *ParallelIterator {
                self.backend_type = backend_type;
                switch (self.backend_type) {
                    .cpu => {
                        // Calculate how many cores are on the current machine to get maximum performance
                        const core_count = std.Thread.getCpuCount() catch unreachable;

                        // Calculate ceil(sqrt(data.len))
                        var sqrt = std.math.sqrt(self.data.len);
                        if (sqrt * sqrt != self.data.len) {
                            sqrt += 1;
                        }

                        // Set the number of tasks to run in parallel
                        self.tasks = sqrt;

                        // Set the queue size equal to the lowest power of two greater than
                        // ceil(sqrt(data.len))
                        var queue_size: usize = 1;
                        while (queue_size <= sqrt) : (queue_size <<= 1) {}

                        // Create a thread pool to work with
                        self.t_pool = pool.ThreadPool.init(self.allocator, .{ .n_jobs = core_count, .queue_size = queue_size }) catch unreachable;
                        return self;
                    },
                    .opencl => {
                        return self;
                    },
                }
            }

            /// Deinitialize a ParallelIterator.
            pub fn deinit(self: *ParallelIteratorSelf) void {
                switch (self.backend_type) {
                    .cpu => {
                        self.t_pool.deinit();
                    },
                    .opencl => {
                        return;
                    },
                }
            }

            /// The worker function for the public map function. Each thread will run
            /// an instance of this function.
            /// Arguments:
            ///     data: []T - The slice to do the work on (data of the ParallelIterator).
            ///     func: *const fn (T) T - The function run on each value in the slice (slice data will be repaced with the return value of this function).
            ///     start: usize - For this thread, run the function starting at this index.
            ///     stop: usize - For this thread, run the function until this index.
            fn par_map(data: []T, func: *const fn (T) T, start: usize, stop: usize) void {
                for (start..stop) |i| {
                    data[i] = func(data[i]);
                }
            }

            /// Run a function in parallel on the data of the ParallelIterator.
            /// Arguments:
            ///     func: *const fn (T) T - A function to run on each value in the slice (each value will be replaced with the return value of this function).
            /// Returns:
            ///     A ParallelIterator (for easy chaining).
            pub fn map(self: *ParallelIteratorSelf, func: *const fn (T) T) *ParallelIteratorSelf {
                // Generate enough tasks to fill the global work queue
                const block_size = self.data.len / self.tasks;
                if (self.data.len % self.tasks > 0) {
                    self.tasks += 1;
                }

                // Add tasks to  the queue
                for (0..self.tasks) |i| {
                    const task_start = block_size * i;
                    const task_end = @min(task_start + block_size, self.data.len);
                    self.t_pool.spawn(*const fn ([]T, *const fn (T) T, usize, usize) void, ParallelIterator.par_map, .{ self.data, func, task_start, task_end }) catch unreachable;
                }

                // Run the jobs
                self.t_pool.schedule() catch unreachable;
                self.t_pool.start() catch unreachable;
                self.t_pool.join();

                return self;
            }

            /// The worker function for the public do function. Each thread will run
            /// an instance of this function.
            /// Arguments:
            ///     data: []T - The slice to do the work on (data of the ParallelIterator).
            ///     func: *const fn (T) void - The function to run on each value in the slice (return nothing).
            ///     start: usize - For this thread, run the function starting at this index.
            ///     stop: usize - For this thread, run the function until this index.
            fn par_do(data: []T, func: *const fn (T) void, start: usize, stop: usize) void {
                for (start..stop) |i| {
                    func(data[i]);
                }
            }

            /// Run a funciton in parallel on the data of the ParallelIterator.
            /// Arguments:
            ///     func: *const fn (T) void - A function to run on each value in the slice (returns nothing).
            /// Returns:
            ///     A ParallelIterator (for easy chaining).
            pub fn do(self: *ParallelIteratorSelf, func: *const fn (T) void) *ParallelIteratorSelf {
                // Generate enough tasks to fill the global work queue
                const block_size = self.data.len / self.tasks;
                if (self.data.len % self.tasks > 0) {
                    self.tasks += 1;
                }

                // Add tasks to the queue
                for (0..self.tasks) |i| {
                    const task_start = block_size * i;
                    const task_end = @min(task_start + block_size, self.data.len);
                    self.t_pool.spawn(*const fn ([]T, *const fn (T) void, usize, usize) void, ParallelIterator.par_do, .{ self.data, func, task_start, task_end }) catch unreachable;
                }

                // Run the jobs
                self.t_pool.schedule() catch unreachable;
                self.t_pool.start() catch unreachable;
                self.t_pool.join();

                return self;
            }
        };

        /// Return a ParallelIterator on a slice of data.
        /// Arguments:
        ///     data: []T - The data to iterate over.
        ///     allocator: std.mem.Allocator - An allocator to manage the internal ThreadPool.
        pub fn par_iter(data: []T, allocator: std.mem.Allocator) ParallelIterator {
            return ParallelIterator.init(data, allocator);
        }
    };
}

const test_func = struct {
    fn square(n: usize) usize {
        return n * n;
    }

    fn io_sim(n: usize) void {
        _ = n;
        std.time.sleep(1_000_000_000);
    }
};

test "Zync map large memory performance test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var seq_square = try allocator.alloc(usize, 4096 * 4096 * 16);
    var par_square = try allocator.alloc(usize, 4096 * 4096 * 16);
    defer allocator.free(seq_square);
    defer allocator.free(par_square);

    for (0..seq_square.len) |i| {
        seq_square[i] = i;
        par_square[i] = i;
    }

    const seq_start = try std.time.Instant.now();
    for (0..seq_square.len) |i| {
        seq_square[i] = test_func.square(seq_square[i]);
    }
    const seq_end = try std.time.Instant.now();
    const seq_elapsed: f64 = @floatFromInt(seq_end.since(seq_start));

    var iter = Zync(usize).par_iter(par_square, allocator);
    defer iter.deinit();

    const par_start = try std.time.Instant.now();
    _ = iter.backend(.cpu)
        .map(test_func.square);
    const par_end = try std.time.Instant.now();
    const par_elapsed: f64 = @floatFromInt(par_end.since(par_start));

    try std.testing.expect(par_elapsed < seq_elapsed);

    util.header("Zync map performance test");
    std.debug.print("Sequential time to complete large squaring calculation: {d}\n", .{seq_elapsed});
    std.debug.print("Zync time to complete large squaring calculation: {d}\n", .{par_elapsed});
    std.debug.print("Percent reduction: {d:0.1}%\n", .{(seq_elapsed - par_elapsed) * 100 / seq_elapsed});
    util.long_line();
}

test "Zync do performance test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    const allocator = gpa.allocator();

    var seq_io = try allocator.alloc(usize, 60);
    var par_io = try allocator.alloc(usize, 60);
    defer allocator.free(seq_io);
    defer allocator.free(par_io);

    for (0..seq_io.len) |i| {
        seq_io[i] = i;
        par_io[i] = i;
    }

    const seq_start = try std.time.Instant.now();
    for (0..seq_io.len) |i| {
        test_func.io_sim(seq_io[i]);
    }
    const seq_end = try std.time.Instant.now();
    const seq_elapsed: f64 = @floatFromInt(seq_end.since(seq_start));

    var iter = Zync(usize).par_iter(par_io, allocator);
    defer iter.deinit();

    const par_start = try std.time.Instant.now();
    _ = iter.backend(.cpu)
        .do(test_func.io_sim);
    const par_end = try std.time.Instant.now();
    const par_elapsed: f64 = @floatFromInt(par_end.since(par_start));

    try std.testing.expect(par_elapsed < seq_elapsed);

    util.header("Zync map performance test");
    std.debug.print("Sequential time to complete IO simulation: {d}\n", .{seq_elapsed});
    std.debug.print("Zync time to complete IO simulation: {d}\n", .{par_elapsed});
    std.debug.print("Percent reduction: {d:0.1}%\n", .{(seq_elapsed - par_elapsed) * 100 / seq_elapsed});
    util.long_line();
}
