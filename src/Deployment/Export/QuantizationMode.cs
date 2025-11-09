namespace AiDotNet.Deployment.Export;

/// <summary>
/// Quantization modes for model export.
/// </summary>
public enum QuantizationMode
{
    /// <summary>No quantization applied</summary>
    None,

    /// <summary>8-bit integer quantization</summary>
    Int8,

    /// <summary>16-bit floating point quantization</summary>
    Float16,

    /// <summary>Dynamic quantization (quantize at runtime)</summary>
    Dynamic,

    /// <summary>Mixed precision (some layers quantized, some not)</summary>
    Mixed
}
