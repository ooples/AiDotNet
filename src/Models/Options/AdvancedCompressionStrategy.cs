namespace AiDotNet.Models.Options;

/// <summary>
/// Advanced gradient compression strategies for federated learning communication efficiency.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These strategies go beyond basic top-k or quantization to achieve
/// 100-1000x compression while maintaining model quality. They are designed for production FL
/// deployments with bandwidth constraints (mobile, satellite, edge devices).</para>
/// </remarks>
public enum AdvancedCompressionStrategy
{
    /// <summary>
    /// PowerSGD: low-rank gradient approximation using SVD-like factorization.
    /// Compresses a D-dimensional gradient into two factor matrices of rank R,
    /// reducing communication from O(D) to O(D*R/D) = O(R). Very effective for large models.
    /// </summary>
    PowerSGD,

    /// <summary>
    /// Gradient sketching: Count Sketch-based compression that maps gradients into
    /// a compact sketch using hash functions. Supports efficient top-k recovery.
    /// </summary>
    GradientSketch,

    /// <summary>
    /// 1-bit SGD: extreme compression where each gradient component is encoded as a single bit
    /// (sign only). Must be paired with error feedback for convergence.
    /// </summary>
    OneBitSGD,

    /// <summary>
    /// Adaptive compression: dynamically adjusts compression ratio per client based on
    /// estimated bandwidth, gradient importance, and staleness.
    /// </summary>
    Adaptive
}
