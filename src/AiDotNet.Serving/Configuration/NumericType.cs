namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Numeric types supported for model inference.
/// </summary>
public enum NumericType
{
    /// <summary>64-bit floating point (double precision).</summary>
    Double,

    /// <summary>32-bit floating point (single precision).</summary>
    Float,

    /// <summary>128-bit decimal type.</summary>
    Decimal
}

