using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Marker interface for IR operations that can be JIT-compiled.
/// </summary>
/// <remarks>
/// <para>
/// This interface represents operations in the Intermediate Representation (IR) layer
/// that can be Just-In-Time compiled for optimized execution on different hardware backends.
/// </para>
/// <para>
/// IR operations serve as a bridge between high-level neural network layers and
/// low-level execution engines (CPU, GPU, TPU).
/// </para>
/// </remarks>
public interface IROp
{
    // Marker interface - no methods required
    // Implementations will define their own Forward/Backward signatures
}
