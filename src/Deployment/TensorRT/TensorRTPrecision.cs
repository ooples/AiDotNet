namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// TensorRT precision modes for inference.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Precision affects speed vs accuracy trade-off:
/// - FP32 (32-bit float): Full precision, most accurate, slowest
/// - FP16 (16-bit float): Half precision, good balance, ~2x faster than FP32
/// - INT8 (8-bit integer): Quantized, fastest, requires calibration data
///
/// For most use cases, FP16 provides a good balance. Use INT8 only when
/// you have calibration data and need maximum throughput.
/// </para>
/// </remarks>
public enum TensorRTPrecision
{
    /// <summary>32-bit floating point (full precision)</summary>
    FP32,

    /// <summary>16-bit floating point (half precision, ~2x faster)</summary>
    FP16,

    /// <summary>8-bit integer (quantized, fastest, requires calibration)</summary>
    INT8
}
