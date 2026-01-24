namespace AiDotNet.Enums;

/// <summary>
/// Specifies the quantization mode for model optimization and export.
/// </summary>
public enum QuantizationMode
{
    /// <summary>No quantization applied</summary>
    None,

    /// <summary>8-bit integer quantization</summary>
    Int8,

    /// <summary>16-bit floating point quantization</summary>
    Float16,

    /// <summary>32-bit floating point (full precision, no quantization)</summary>
    Float32,

    /// <summary>Dynamic quantization (quantize at runtime)</summary>
    Dynamic,

    /// <summary>Mixed precision (some layers quantized, some not)</summary>
    Mixed
}
