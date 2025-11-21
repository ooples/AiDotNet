namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Configurable thresholds for adaptive execution (CPU vs GPU routing).
/// </summary>
/// <remarks>
/// <para>
/// GPU operations have overhead (memory transfer, kernel launch). For small operations,
/// this overhead exceeds the computation time, making CPU faster. Adaptive thresholds
/// automatically route small operations to CPU and large operations to GPU.
/// </para>
/// <para><b>Phase B: US-GPU-004 - Adaptive Execution</b>
///
/// Benefits:
/// - Optimal performance across all operation sizes
/// - No performance penalty for small operations
/// - Maximum GPU speedup for large operations
/// - User-configurable for different hardware
///
/// Default thresholds are conservative and work well on most systems.
/// Adjust based on benchmarking for your specific hardware.
/// </para>
/// </remarks>
public class AdaptiveThresholds
{
    /// <summary>
    /// Threshold for vector Add operation (elements).
    /// Operations with fewer elements use CPU, more use GPU.
    /// </summary>
    /// <remarks>
    /// Default: 10,000 elements
    /// - Below threshold: CPU is faster due to low GPU overhead
    /// - Above threshold: GPU is 10-100x faster
    /// </remarks>
    public int VectorAdd { get; set; } = 10_000;

    /// <summary>
    /// Threshold for vector Subtract operation (elements).
    /// </summary>
    /// <remarks>
    /// Default: 10,000 elements
    /// Similar characteristics to Add operation.
    /// </remarks>
    public int VectorSubtract { get; set; } = 10_000;

    /// <summary>
    /// Threshold for vector Multiply (element-wise) operation (elements).
    /// </summary>
    /// <remarks>
    /// Default: 10,000 elements
    /// Similar characteristics to Add operation.
    /// </remarks>
    public int VectorMultiply { get; set; } = 10_000;

    /// <summary>
    /// Threshold for vector Divide operation (elements).
    /// </summary>
    /// <remarks>
    /// Default: 10,000 elements
    /// Division is slightly more expensive than Add/Multiply.
    /// </remarks>
    public int VectorDivide { get; set; } = 10_000;

    /// <summary>
    /// Threshold for vector Sqrt operation (elements).
    /// </summary>
    /// <remarks>
    /// Default: 5,000 elements
    /// Sqrt is more expensive than basic arithmetic, benefits from GPU earlier.
    /// </remarks>
    public int VectorSqrt { get; set; } = 5_000;

    /// <summary>
    /// Threshold for vector Power operation (elements).
    /// </summary>
    /// <remarks>
    /// Default: 5,000 elements
    /// Power is expensive, benefits from GPU earlier.
    /// </remarks>
    public int VectorPower { get; set; } = 5_000;

    /// <summary>
    /// Threshold for matrix multiplication (matrix dimension).
    /// </summary>
    /// <remarks>
    /// Default: 256 (256x256 matrix)
    /// GEMM is O(nÃƒâ€šÃ‚Â³), so GPU benefits kick in quickly.
    /// Below 256x256: CPU is competitive
    /// Above 256x256: GPU is 100-1000x faster
    /// </remarks>
    public int MatrixMultiply { get; set; } = 256;

    /// <summary>
    /// Threshold for matrix-vector multiply (matrix dimension).
    /// </summary>
    /// <remarks>
    /// Default: 512 (512x512 matrix)
    /// GEMV is O(nÃƒâ€šÃ‚Â²), less benefit than GEMM.
    /// </remarks>
    public int MatrixVectorMultiply { get; set; } = 512;

    /// <summary>
    /// Threshold for 2D convolution (input elements).
    /// </summary>
    /// <remarks>
    /// Default: 1,000 input elements
    /// Convolution is expensive, benefits from GPU earlier.
    /// Typical use: > 32x32 images benefit from GPU
    /// </remarks>
    public int Convolution { get; set; } = 1_000;

    /// <summary>
    /// Threshold for pooling operations (input elements).
    /// </summary>
    /// <remarks>
    /// Default: 2,000 input elements
    /// Pooling is simpler than convolution, needs larger size for GPU benefit.
    /// </remarks>
    public int Pooling { get; set; } = 2_000;

    /// <summary>
    /// Threshold for batched matrix multiplication (matrix dimension).
    /// </summary>
    /// <remarks>
    /// Default: 128 (128x128 matrices in batch)
    /// BatchMatMul benefits from GPU earlier than single GEMM due to parallel batch processing.
    /// Below 128x128: CPU is competitive
    /// Above 128x128: GPU is 50-500x faster due to batch parallelism
    /// </remarks>
    public int BatchMatMul { get; set; } = 128;

    /// <summary>
    /// Gets the default thresholds optimized for typical desktop GPUs.
    /// </summary>
    public static AdaptiveThresholds Default => new AdaptiveThresholds();

    /// <summary>
    /// Gets thresholds optimized for high-end GPUs (lower thresholds, more GPU usage).
    /// </summary>
    public static AdaptiveThresholds HighEndGpu => new AdaptiveThresholds
    {
        VectorAdd = 5_000,
        VectorSubtract = 5_000,
        VectorMultiply = 5_000,
        VectorDivide = 5_000,
        VectorSqrt = 2_000,
        VectorPower = 2_000,
        MatrixMultiply = 128,
        MatrixVectorMultiply = 256,
        Convolution = 500,
        Pooling = 1_000,
        BatchMatMul = 64
    };

    /// <summary>
    /// Gets thresholds optimized for low-end GPUs or integrated graphics (higher thresholds, less GPU usage).
    /// </summary>
    public static AdaptiveThresholds LowEndGpu => new AdaptiveThresholds
    {
        VectorAdd = 50_000,
        VectorSubtract = 50_000,
        VectorMultiply = 50_000,
        VectorDivide = 50_000,
        VectorSqrt = 20_000,
        VectorPower = 20_000,
        MatrixMultiply = 512,
        MatrixVectorMultiply = 1024,
        Convolution = 5_000,
        Pooling = 10_000,
        BatchMatMul = 256
    };

    /// <summary>
    /// Gets thresholds that always prefer CPU (for testing or systems without GPU).
    /// </summary>
    public static AdaptiveThresholds AlwaysCpu => new AdaptiveThresholds
    {
        VectorAdd = int.MaxValue,
        VectorSubtract = int.MaxValue,
        VectorMultiply = int.MaxValue,
        VectorDivide = int.MaxValue,
        VectorSqrt = int.MaxValue,
        VectorPower = int.MaxValue,
        MatrixMultiply = int.MaxValue,
        MatrixVectorMultiply = int.MaxValue,
        Convolution = int.MaxValue,
        Pooling = int.MaxValue,
        BatchMatMul = int.MaxValue
    };

    /// <summary>
    /// Gets thresholds that always prefer GPU (for testing or dedicated GPU workloads).
    /// </summary>
    public static AdaptiveThresholds AlwaysGpu => new AdaptiveThresholds
    {
        VectorAdd = 0,
        VectorSubtract = 0,
        VectorMultiply = 0,
        VectorDivide = 0,
        VectorSqrt = 0,
        VectorPower = 0,
        MatrixMultiply = 0,
        MatrixVectorMultiply = 0,
        Convolution = 0,
        Pooling = 0,
        BatchMatMul = 0
    };

    /// <summary>
    /// Returns a string describing the current threshold configuration.
    /// </summary>
    public override string ToString()
    {
        return $"AdaptiveThresholds: Vector={VectorAdd}, Matrix={MatrixMultiply}, Conv={Convolution}";
    }
}
