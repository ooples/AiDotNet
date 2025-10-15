using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Compression.Quantization;

/// <summary>
/// Represents a single quantized parameter from a model.
/// </summary>
/// <typeparam name="T">The numeric type used for scale factors and calculations.</typeparam>
/// <remarks>
/// <para>
/// This class encapsulates a quantized parameter along with the metadata needed
/// to dequantize it.
/// </para>
/// <para><b>For Beginners:</b> This represents one quantized weight matrix or vector.
/// 
/// It contains:
/// - The quantized values themselves (in low precision)
/// - The information needed to convert back to original values
/// - Metadata about the quantization process used
/// </para>
/// </remarks>
public class QuantizedParameter<T> where T : unmanaged
{
    /// <summary>
    /// Gets or sets the original shape of the parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the shape (dimensions) of the parameter before quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This records the original dimensions of the parameter.
    /// 
    /// For example, a weight matrix might have shape [1000, 500], meaning 1000 rows and 500 columns.
    /// </para>
    /// </remarks>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the quantized values as a tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the low-precision values that represent the original parameter.
    /// The tensor contains the raw quantized values which need to be interpreted
    /// based on the BitWidth property.
    /// </para>
    /// <para><b>For Beginners:</b> These are the actual compressed values.
    /// 
    /// For example, if using 8-bit quantization, this tensor would contain values
    /// that fit in 8 bits instead of the original 32-bit floats.
    /// </para>
    /// </remarks>
    public Tensor<T> QuantizedValues { get; set; } = null!;

    /// <summary>
    /// Gets or sets the scale factor for dequantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This scale factor is used to convert quantized values back to their original range.
    /// </para>
    /// <para><b>For Beginners:</b> This is the multiplier to convert back to original values.
    /// 
    /// During dequantization:
    /// original_value ≈ (quantized_value - zero_point) * scale
    /// </para>
    /// </remarks>
    public T Scale { get; set; } = default!;

    /// <summary>
    /// Gets or sets the zero point for asymmetric quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For asymmetric quantization, this is the quantized value that corresponds to 0
    /// in the original parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is the value that represents zero in the quantized space.
    /// 
    /// For asymmetric quantization, we need to know which quantized value maps to zero
    /// in the original space. This zero point is subtracted before scaling during dequantization.
    /// </para>
    /// </remarks>
    public int ZeroPoint { get; set; }

    /// <summary>
    /// Gets or sets the bit width used for quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the number of bits used to represent each quantized value.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many bits each quantized value uses.
    /// 
    /// Common values are 8 bits or 4 bits, compared to the original 32 bits used by floating-point values.
    /// </para>
    /// </remarks>
    public int BitWidth { get; set; }

    /// <summary>
    /// Gets or sets the quantization method used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This specifies whether symmetric, asymmetric, or per-channel quantization was used.
    /// </para>
    /// <para><b>For Beginners:</b> This tells which quantization approach was used.
    /// 
    /// Different methods have different tradeoffs and require different dequantization approaches.
    /// </para>
    /// </remarks>
    public QuantizationMethod Method { get; set; }

    /// <summary>
    /// Gets or sets the per-channel scale factors for per-channel quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For per-channel quantization, each channel has its own scale factor.
    /// </para>
    /// <para><b>For Beginners:</b> These are the multipliers for each channel.
    /// 
    /// When using per-channel quantization:
    /// - Each output channel has its own scaling factor
    /// - This vector contains one scale value per channel
    /// - Dequantization uses the appropriate scale for each channel
    /// </para>
    /// </remarks>
    public Vector<T> ChannelScales { get; set; }

    /// <summary>
    /// Gets or sets the per-channel zero points for per-channel asymmetric quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For per-channel asymmetric quantization, each channel has its own zero point.
    /// </para>
    /// <para><b>For Beginners:</b> These are the zero values for each channel.
    /// 
    /// When using per-channel asymmetric quantization:
    /// - Each output channel has its own zero point
    /// - This vector contains one zero point per channel
    /// </para>
    /// </remarks>
    public Vector<int> ChannelZeroPoints { get; set; }
}