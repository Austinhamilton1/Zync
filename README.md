# Zync

### What Zync is.

Zync (**Z**ig-s**ync**hronous) is an open-source (contributors welcome) parallel processing library for Zig. The intended functionality of Zync is to be used like libraries in other languages such as Rust's Rayon or Python's Dask.
The intention of Zync is to make use of Zig's powerful parallel processing potential while remaining simple and intuitive to use.

Zync includes many powerful programming paradigms, including a per-thread, work-stealing queue architecture for thread scheduling. Zync comes pre-rolled with a ThreadPool implementation as well as more robust
Thread objects. These can be used outside of the Zync library for custom development outside of the Zync environment. However, the intended functionality of Zync is to be used as a supplemental library, not for
its types.

### What Zync is not.

Zync currently has no support for hardware accelerated computing (i.e., it has no OpenCL, Vulkan, CUDA, etc. backend).

Zync is not currently a "plug-and-play" library. A working knowledge of the inner-workings of the library will help with implementations.

*You can skip this part if you only want to use Zync instead of understanding its fundamentals*

## Data Types of Zync

### Consumer

The `Consumer` is the heart of the `ThreadPool`. A `Consumer` is a wrapper for a worker thread that consumes tasks from the global work queue. Each consumer has an ID, a local work-queue, a pointer to a `ThreadGroup`, and a pool of other consumers to steal tasks from.

The consume function loops as long as the `Consumer` is set to running. It attempts to pop a task from its own queue. If it succeeds, it runs the task and tries again. If it fails to `pop` a task (i.e., the queue is empty), the `Consumer` attempts to steal from a randomly selected `Consumer`. If it succeeds at `steal`ing, it immediately tries to `steal` again (the idea is that if a `Consumer` can be stolen from it is probably busy doing some task so we should probably be able to `steal` again). 

The `Consumer`s start running when their `run` method is called. The `Consumer`s run until their `stop` method is invoked. This is the difference between finishing a task and finsishing the thread. If you call `stop` the Thread will finish up and the `Consumer` will wait in an idle state to start again.

### VynukovQueue

The `VyukovQueue` is a MPMCQueue based on Vyokov's implementation. It acts as the global task queue to allow tasks to be submitted from the main thread and then delegated to the `Consumer`s in an intelligent way.

### SPMCDeque

The `SPMCDeque` is what allows for the work-stealing paradigm issued by the `ThreadPool`. It is a ring buffer with `top` and `bottom` pointers. Only the owner thread may `push`/`pop` from the `bottom`. The `top` is available to the rest of the Threads. Threads try to grab the `top` quickly when `steal`ing. This ensures that `steal`ing does not make the current thread stop its own work.

### ThreadGroup

A `ThreadGroup` is what allows a `ThreadPool` object to wait on `Consumer` threads to finish the current queue of tasks. Since `Consumer`s stay in an infinite loop, they need something to act as a pseudo-join. `ThreadGroup` lets you do this.

A `ThreadGroup` is just a thin wrapper around a global reference `counter`. When a task is submitted to the task queue, the reference count is `increment`ed. When a task is completed the ThreadGroup is `decrement`ed. The `wait` method of ThreadGroup just loops infinitely until the reference count is equal to 0.

### ThreadPool

The `ThreadPool` manages a group of `Consumer`s. It initially creates all the `Consumer`s, and waits. When the `spawn` method is called on the `ThreadPool`, the spawned Task is added to the `ThreadPool`'s global task queue. It isn't until all tasks are added and the `ThreadPool`'s `schedule` method is called that the `Consumer`s have access to the tasks.

Once the tasks have been `schedule`d, the `run` method of the `ThreadPool` is called, which officially begins the work of the `Consumer`s. Once `run` is called, the `Consumer`s are active and completing work. The `ThreadPool` runs in this mode until the `ThreadPool` calls `join`. This stops all `Consumer`s and leaves them in an idle state.

## Features of Zync

*Zync is not functional yet, so this remains to be seen

## Zync Examples

*Zync is not functional yet, so there are currently no examples*

## Benchmarks

*Zync hasn't been created yet, so no benchmarks yet*

# TODO

- [X] Task structure for completing generic tasks
- [X] UnboundedTaskQueue for adding tasks to a queue to be completed (first iteration)
  - [X] Atomic linked list data structure
  - [X] Pop
  - [X] Push
- [X] BoundedTaskQueue for adding tasks to a queue to be completed (second iteration)
  - [X] Atomic ring buffer data structure
  - [X] Pop
  - [X] Push
- [X] SPMC for work stealing (third iteration)
  - [X] Local thread based pop
  - [X] Local thread based push
  - [X] Other thread steal
- [X] ThreadGroup implementation
  - [X] Count task completion status
  - [X] Allow for generic synchronization of Threads  
- [X] Implement Consumer (ID, thread-local work queue, error recovery, etc.)
- [X] Add work stealing ability to Consumer
- [ ] ThreadPool implementation
  - [X] Allocate `n_jobs` worker threads that await a task.
  - [X] Global task queue for workers to grab from.
  - [X] Work stealing scheduling
  - [X] Optimize work stealing for faster context switches
- [ ] Start working on Zync library features
  - [ ] par_sort
  - [ ] par_iter for slices and arrays
  - [ ] par_map for par_iter
