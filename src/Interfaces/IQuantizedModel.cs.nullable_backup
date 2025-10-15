using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that have been quantized.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IQuantizedModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
    where T : unmanaged
{
    /// <summary>
    /// Gets the bit width used for quantization.
    /// </summary>
    int QuantizationBitWidth { get; }
    
    /// <summary>
    /// Gets whether symmetric quantization was used.
    /// </summary>
    bool IsSymmetric { get; }
    
    /// <summary>
    /// Gets the scale factors used for quantization.
    /// </summary>
    /// <remarks>
    /// Scale factors are typically floating-point values that determine how to scale
    /// the quantized values back to their original range.
    /// </remarks>
    Vector<T> GetScaleFactors();
    
    /// <summary>
    /// Gets the zero points used for quantization (if asymmetric).
    /// </summary>
    /// <remarks>
    /// Zero points are typically integer values that represent the quantized value
    /// corresponding to zero in the original range. For symmetric quantization,
    /// these will all be zero. Implementations should handle appropriate type
    /// conversions when T is not an integer type.
    /// </remarks>
    Vector<T> GetZeroPoints();
}