using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// A density field sampled on a regular grid in world coordinates.
/// </summary>
/// <typeparam name="T">The numeric type of the samples.</typeparam>
/// <remarks>
/// Sample (i, j, k) of <see cref="Density"/> lies at <c>(OriginX + i * Spacing, OriginY + j * Spacing,
/// OriginZ + k * Spacing)</c>, which is the layout <see cref="MarchingCubes.Extract{T}"/> takes.
/// </remarks>
public sealed class GaussianDensityGrid<T>
{
    /// <summary>
    /// Creates a density grid.
    /// </summary>
    public GaussianDensityGrid(Tensor<T> density, double originX, double originY, double originZ, double spacing)
    {
        Density = density ?? throw new ArgumentNullException(nameof(density));
        if (density.Shape.Length != 3)
            throw new ArgumentException("The density must have shape [nx, ny, nz].", nameof(density));
        if (!(spacing > 0.0))
            throw new ArgumentOutOfRangeException(nameof(spacing), "The spacing must be positive.");

        OriginX = originX;
        OriginY = originY;
        OriginZ = originZ;
        Spacing = spacing;
    }

    /// <summary>The density samples, [nx, ny, nz].</summary>
    public Tensor<T> Density { get; }

    /// <summary>World x of sample index 0.</summary>
    public double OriginX { get; }

    /// <summary>World y of sample index 0.</summary>
    public double OriginY { get; }

    /// <summary>World z of sample index 0.</summary>
    public double OriginZ { get; }

    /// <summary>World distance between neighbouring samples.</summary>
    public double Spacing { get; }
}
