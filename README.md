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

## Data Types of Zync

### Consumer

*Put a description of Consumer here.*

### VynukovQueue

*Put a description of the BoundedTaskQueue here.

### ChaseLevDeque

*Put a description of the ChaseLevDeque here.*

### ThreadPool

*Put a description of the ThreadPool here.*

## Features of Zync

*Zync is not functional yet, so this remains to be seen*

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
- [X] ChaseLevDeque for work stealing (third iteration)
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
  - [ ] Optimize work stealing for faster context switches
  - [ ] Fork/join setup
- [ ] Start working on Zync library features
  - [ ] par_sort
  - [ ] par_iter for slices and arrays
  - [ ] par_map for par_iter
