namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Specifies the type of discrete Laplacian operator to compute for mesh processing.
/// </summary>
/// <remarks>
/// The Laplacian operator is fundamental to many mesh processing algorithms including
/// smoothing, deformation, parameterization, and spectral analysis.
/// </remarks>
public enum LaplacianType
{
    /// <summary>
    /// Simple adjacency-based Laplacian that ignores geometry.
    /// Each neighbor has equal weight (1/degree).
    /// Fast to compute but doesn't preserve geometric properties.
    /// </summary>
    Uniform,

    /// <summary>
    /// Geometry-aware Laplacian using cotangent weights.
    /// Preserves angles and is suitable for discrete differential geometry.
    /// The standard choice for mesh processing applications.
    /// </summary>
    Cotangent,

    /// <summary>
    /// Cotangent Laplacian normalized by vertex areas.
    /// Provides scale-invariant smoothing properties.
    /// Also known as the "symmetric" or "area-normalized" Laplacian.
    /// </summary>
    Normalized
}
