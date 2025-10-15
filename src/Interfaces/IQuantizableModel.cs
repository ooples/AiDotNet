namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that support quantization compression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TModel">The type of the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IQuantizableModel<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a quantized version of this model.
    /// </summary>
    /// <param name="bitWidth">The target bit width for quantization.</param>
    /// <param name="useSymmetric">Whether to use symmetric quantization.</param>
    /// <returns>A quantized version of the model.</returns>
    TModel Quantize(int bitWidth, bool useSymmetric);
}