# ILGPU Deep Dive Research Request

Please perform a comprehensive research and analysis of ILGPU (Intermediate Language GPU) and create a detailed developer cheat sheet.

## Research Objectives

1. **Core ILGPU Concepts**
   - What is ILGPU and how does it work?
   - Architecture overview (Accelerator, MemoryBuffer, ArrayView, Index, etc.)
   - Compilation model (how C# becomes GPU kernels)
   - Thread/block execution model

2. **Memory Management**
   - How to allocate GPU memory correctly
   - Best practices for CPU ↔ GPU data transfer
   - MemoryBuffer1D, MemoryBuffer2D, MemoryBuffer3D usage
   - ArrayView vs MemoryBuffer - when to use each
   - Memory pooling patterns
   - Zero-copy techniques

3. **Data Transfer Patterns**
   - Correct methods for copying data from CPU to GPU
   - Correct methods for copying data from GPU to CPU
   - Span<T> integration with ILGPU
   - ReadOnlySpan<T> vs Span<T> usage
   - Batch transfer optimizations

4. **Kernel Development**
   - How to write ILGPU kernels
   - Kernel compilation and caching
   - Index1D, Index2D, Index3D usage
   - Thread synchronization primitives
   - Shared memory usage
   - Kernel parameter passing best practices

5. **Type Constraints**
   - Why `where T : unmanaged` is required
   - Which .NET types are unmanaged
   - Working with generic numeric types in ILGPU
   - Constraint propagation through type hierarchy

6. **Vectorization with ILGPU**
   - Patterns for vector operations (add, subtract, multiply, divide)
   - Matrix operations on GPU
   - Tensor operations on GPU
   - Broadcasting patterns
   - Reduction operations (sum, max, min, etc.)

7. **Performance Best Practices**
   - When to use GPU vs CPU
   - Batch size considerations
   - Minimizing CPU ↔ GPU transfers
   - Kernel optimization techniques
   - Memory coalescing
   - Occupancy optimization

8. **Error Handling**
   - ILGPU exception types
   - GPU memory exhaustion handling
   - Accelerator failure recovery
   - Debugging GPU kernels

9. **Common Pitfalls**
   - Mistakes to avoid
   - Anti-patterns
   - Memory leaks
   - Race conditions

10. **Integration Patterns**
    - Singleton pattern for Accelerator
    - Memory pool implementation
    - Fallback to CPU when GPU unavailable
    - Multi-GPU support

## Deliverable Format

Please create a comprehensive cheat sheet that includes:

1. **Quick Reference Section**: Common operations with code examples
2. **API Reference**: Key ILGPU types and methods with descriptions
3. **Code Templates**: Ready-to-use patterns for common operations
4. **Decision Trees**: When to use GPU vs CPU, when to use different memory types
5. **Best Practices Summary**: Do's and don'ts
6. **Troubleshooting Guide**: Common errors and solutions

## Current Context

We're building a .NET AI library (AiDotNet) that needs to:
- Support both CPU and GPU execution
- Work with generic numeric types through INumericOperations<T>
- Perform vector/matrix/tensor operations efficiently
- Handle automatic fallback when GPU unavailable
- Use memory pooling for GPU buffers
- Support multiple numeric types (float, double, int, etc.)

## Specific Questions to Address

1. What is the CORRECT way to copy Span<T> data to MemoryBuffer1D<T>?
2. What is the CORRECT way to copy MemoryBuffer1D<T> data back to Span<T>?
3. Do we use `buffer.CopyFromCPU()`, `buffer.View.CopyFromCPU()`, or `accelerator.Copy()`?
4. How do extension methods like `CopyFromCPU` work with generic constraints?
5. What are the signature requirements for MemoryBuffer copy methods?
6. How do we handle the `IContiguousArrayView<T>` constraint properly?

## Output Format

Please provide:
1. A comprehensive markdown document (ILGPU-CHEATSHEET.md)
2. Code examples for every pattern described
3. Performance characteristics for each approach
4. Links to official ILGPU documentation for further reading
5. Version information (we're using ILGPU 1.5.3+)

Focus on practical, production-ready patterns that our development team can use immediately.
