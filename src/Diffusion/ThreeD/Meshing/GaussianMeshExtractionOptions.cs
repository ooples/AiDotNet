namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Settings for extracting a mesh from 3D Gaussians (<see cref="GaussianMeshExtractor"/>). The defaults are those of
/// DreamGaussian (Tang et al., ICLR 2024) and its official code (<c>extract_fields</c> / <c>extract_mesh</c>).
/// </summary>
public class GaussianMeshExtractionOptions
{
    /// <summary>
    /// Samples per axis of the density grid. Default: 128.
    /// </summary>
    public int Resolution { get; set; } = 128;

    /// <summary>
    /// Blocks per axis for local density queries: each block only evaluates the Gaussians whose centre lies near it.
    /// Must divide <see cref="Resolution"/>. Default: 16.
    /// </summary>
    public int BlockCount { get; set; } = 16;

    /// <summary>
    /// How far beyond a block, in block sizes, a Gaussian's centre may lie and still be evaluated there. Default: 1.5.
    /// </summary>
    public double RelaxRatio { get; set; } = 1.5;

    /// <summary>
    /// The density level of the extracted surface. Default: 1.
    /// </summary>
    public double DensityThreshold { get; set; } = 1.0;

    /// <summary>
    /// Gaussians with opacity at or below this are skipped before the density query. Default: 0.005.
    /// </summary>
    public double OpacityCullThreshold { get; set; } = 0.005;

    /// <summary>
    /// Half the extent the Gaussian centres are normalised to inside the [-1, 1] grid cube (the reference maps the
    /// largest side of their bounding box to 1.8). Default: 0.9.
    /// </summary>
    public double NormalizedHalfExtent { get; set; } = 0.9;
}
