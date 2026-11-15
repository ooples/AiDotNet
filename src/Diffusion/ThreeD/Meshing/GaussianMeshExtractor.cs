using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Extracts a mesh from 3D Gaussians through a dense density grid, as DreamGaussian does (Tang, Ren, Zhou, Liu &amp;
/// Zeng, "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation", ICLR 2024, section 3.2,
/// "Efficient Mesh Extraction").
/// </summary>
/// <remarks>
/// <para>
/// The Gaussian centres are normalised so the largest side of their bounding box spans 1.8 inside [-1, 1]^3. The
/// density at a grid point x is
/// </para>
/// <code>
/// d(x) = sum_i alpha_i exp(-1/2 (x - mu_i)^T Sigma_i^-1 (x - mu_i))
/// </code>
/// <para>
/// with Sigma_i = R_i S_i S_i^T R_i^T. The grid is split into blocks and each block only sums the Gaussians whose centre
/// lies within <see cref="GaussianMeshExtractionOptions.RelaxRatio"/> block sizes of it (the paper's "local density
/// query"); Gaussians of negligible opacity are culled first. The surface is the level set d = threshold, extracted by
/// marching cubes and mapped back to the Gaussians' own coordinates. The density is invariant under the normalisation
/// (it scales offsets and standard deviations alike), so it is evaluated directly in world coordinates.
/// </para>
/// <para>
/// The reference then cleans, remeshes and decimates the mesh with PyMeshLab; those are mesh post-processing steps and
/// are not part of the extraction here. The arithmetic runs in double precision: the inverse covariances of small
/// Gaussians exceed the range of half precision.
/// </para>
/// </remarks>
public static class GaussianMeshExtractor
{
    /// <summary>
    /// Samples the Gaussians' density on the grid of the local density query.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gaussians">The Gaussians.</param>
    /// <param name="options">Grid settings; null uses DreamGaussian's defaults.</param>
    /// <returns>The density grid, [R, R, R], in world coordinates.</returns>
    public static GaussianDensityGrid<T> ExtractDensityGrid<T>(
        GaussianCloud<T> gaussians, GaussianMeshExtractionOptions? options = null)
    {
        if (gaussians is null)
            throw new ArgumentNullException(nameof(gaussians));

        var settings = options ?? new GaussianMeshExtractionOptions();
        int resolution = settings.Resolution;
        int blocks = settings.BlockCount;
        if (resolution < 2)
            throw new ArgumentOutOfRangeException(nameof(options), "Resolution must be at least 2.");
        if (blocks < 1 || resolution % blocks != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "BlockCount must be positive and divide Resolution.");
        if (!(settings.NormalizedHalfExtent > 0.0) || settings.NormalizedHalfExtent > 1.0)
            throw new ArgumentOutOfRangeException(nameof(options), "NormalizedHalfExtent must be in (0, 1].");
        if (settings.RelaxRatio < 0.0)
            throw new ArgumentOutOfRangeException(nameof(options), "RelaxRatio must be non-negative.");

        int count = gaussians.Count;
        double[] positions = gaussians.Positions.Cast<double>().ToArray();
        double[] scales = gaussians.Scales.Cast<double>().ToArray();
        double[] rotations = gaussians.Rotations.Cast<double>().ToArray();
        double[] opacities = gaussians.Opacities.Cast<double>().ToArray();

        // Opacity pre-filter, then the normalisation into the grid cube.
        var kept = new List<int>();
        for (int i = 0; i < count; i++)
        {
            if (opacities[i] > settings.OpacityCullThreshold)
                kept.Add(i);
        }

        var min = new[] { double.MaxValue, double.MaxValue, double.MaxValue };
        var max = new[] { double.MinValue, double.MinValue, double.MinValue };
        foreach (int i in kept)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                min[axis] = Math.Min(min[axis], positions[(3 * i) + axis]);
                max[axis] = Math.Max(max[axis], positions[(3 * i) + axis]);
            }
        }

        var center = new double[3];
        double scale = 1.0;
        if (kept.Count > 0)
        {
            double extent = 0.0;
            for (int axis = 0; axis < 3; axis++)
            {
                center[axis] = 0.5 * (min[axis] + max[axis]);
                extent = Math.Max(extent, max[axis] - min[axis]);
            }

            if (extent > 0.0)
                scale = 2.0 * settings.NormalizedHalfExtent / extent;
        }

        // Grid sample i on each axis: normalised -1 + i * step, world normalised / scale + center.
        double step = 2.0 / (resolution - 1);
        double worldSpacing = step / scale;
        double originX = center[0] - (1.0 / scale), originY = center[1] - (1.0 / scale), originZ = center[2] - (1.0 / scale);

        int kc = kept.Count;
        var mu = new double[3 * kc];
        var normalizedMu = new double[3 * kc];
        var inverse = new double[6 * kc];
        var alpha = new double[kc];
        for (int g = 0; g < kc; g++)
        {
            int i = kept[g];
            for (int axis = 0; axis < 3; axis++)
            {
                mu[(3 * g) + axis] = positions[(3 * i) + axis];
                normalizedMu[(3 * g) + axis] = (positions[(3 * i) + axis] - center[axis]) * scale;
            }

            alpha[g] = opacities[i];
            InverseCovariance(scales, rotations, i, inverse, 6 * g);
        }

        int split = resolution / blocks;
        double blockSize = 2.0 / blocks;
        double relax = blockSize * settings.RelaxRatio;
        var density = new double[resolution * resolution * resolution];
        System.Threading.Tasks.Parallel.For(0, blocks * blocks * blocks, block =>
        {
            int bx = block / (blocks * blocks), by = (block / blocks) % blocks, bz = block % blocks;
            int x0 = bx * split, y0 = by * split, z0 = bz * split;
            // The block's sample bounds (normalised), widened by the relax margin, select its Gaussians.
            double lowX = -1.0 + (x0 * step) - relax, highX = -1.0 + ((x0 + split - 1) * step) + relax;
            double lowY = -1.0 + (y0 * step) - relax, highY = -1.0 + ((y0 + split - 1) * step) + relax;
            double lowZ = -1.0 + (z0 * step) - relax, highZ = -1.0 + ((z0 + split - 1) * step) + relax;
            var local = new List<int>();
            for (int g = 0; g < kc; g++)
            {
                double nx = normalizedMu[3 * g], ny = normalizedMu[(3 * g) + 1], nz = normalizedMu[(3 * g) + 2];
                if (nx > lowX && nx < highX && ny > lowY && ny < highY && nz > lowZ && nz < highZ)
                    local.Add(g);
            }

            if (local.Count == 0)
                return;

            for (int ix = x0; ix < x0 + split; ix++)
            {
                double px = originX + (ix * worldSpacing);
                for (int iy = y0; iy < y0 + split; iy++)
                {
                    double py = originY + (iy * worldSpacing);
                    for (int iz = z0; iz < z0 + split; iz++)
                    {
                        double pz = originZ + (iz * worldSpacing);
                        double sum = 0.0;
                        foreach (int g in local)
                        {
                            double dx = px - mu[3 * g], dy = py - mu[(3 * g) + 1], dz = pz - mu[(3 * g) + 2];
                            int o = 6 * g;
                            double power = (-0.5 * ((dx * dx * inverse[o]) + (dy * dy * inverse[o + 3]) + (dz * dz * inverse[o + 5])))
                                - (dx * dy * inverse[o + 1]) - (dx * dz * inverse[o + 2]) - (dy * dz * inverse[o + 4]);
                            if (power <= 0.0)
                                sum += alpha[g] * Math.Exp(power);
                        }

                        density[(((ix * resolution) + iy) * resolution) + iz] = sum;
                    }
                }
            }
        });

        var tensor = new Tensor<double>(new[] { resolution, resolution, resolution }, new Vector<double>(density)).Cast<T>();
        return new GaussianDensityGrid<T>(tensor, originX, originY, originZ, worldSpacing);
    }

    /// <summary>
    /// Extracts the mesh of the Gaussians' density level set.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gaussians">The Gaussians.</param>
    /// <param name="options">Grid and threshold settings; null uses DreamGaussian's defaults.</param>
    /// <returns>The mesh, in the Gaussians' coordinates, wound outward with per-vertex normals.</returns>
    public static Mesh3D<T> ExtractMesh<T>(GaussianCloud<T> gaussians, GaussianMeshExtractionOptions? options = null)
    {
        var settings = options ?? new GaussianMeshExtractionOptions();
        var grid = ExtractDensityGrid(gaussians, settings);
        return MarchingCubes.Extract(grid.Density, settings.DensityThreshold, IsoSurfaceInterior.AboveIsoLevel,
            grid.OriginX, grid.OriginY, grid.OriginZ, grid.Spacing);
    }

    /// <summary>
    /// Writes the inverse of Sigma = R S S^T R^T as (a, b, c, d, e, f) = (xx, xy, xz, yy, yz, zz) at
    /// <paramref name="offset"/>, guarded as the reference's <c>gaussian_3d_coeff</c> is (determinant + 1e-24).
    /// </summary>
    internal static void InverseCovariance(double[] scales, double[] rotations, int i, double[] output, int offset)
    {
        double w = rotations[4 * i], x = rotations[(4 * i) + 1], y = rotations[(4 * i) + 2], z = rotations[(4 * i) + 3];
        double norm = Math.Sqrt((w * w) + (x * x) + (y * y) + (z * z));
        if (norm > 0.0)
        {
            w /= norm;
            x /= norm;
            y /= norm;
            z /= norm;
        }
        else
        {
            w = 1.0;
        }

        var r = new double[9]
        {
            1.0 - (2.0 * ((y * y) + (z * z))), 2.0 * ((x * y) - (w * z)), 2.0 * ((x * z) + (w * y)),
            2.0 * ((x * y) + (w * z)), 1.0 - (2.0 * ((x * x) + (z * z))), 2.0 * ((y * z) - (w * x)),
            2.0 * ((x * z) - (w * y)), 2.0 * ((y * z) + (w * x)), 1.0 - (2.0 * ((x * x) + (y * y)))
        };
        double s0 = scales[3 * i], s1 = scales[(3 * i) + 1], s2 = scales[(3 * i) + 2];
        var variance = new[] { s0 * s0, s1 * s1, s2 * s2 };
        // Sigma[p, q] = sum_k R[p, k] s_k^2 R[q, k].
        double Sigma(int p, int q) =>
            (r[(3 * p) + 0] * variance[0] * r[(3 * q) + 0]) + (r[(3 * p) + 1] * variance[1] * r[(3 * q) + 1])
            + (r[(3 * p) + 2] * variance[2] * r[(3 * q) + 2]);

        double a = Sigma(0, 0), b = Sigma(0, 1), c = Sigma(0, 2), d = Sigma(1, 1), e = Sigma(1, 2), f = Sigma(2, 2);
        double inverseDeterminant = 1.0 / ((a * d * f) + (2.0 * e * c * b) - (e * e * a) - (c * c * d) - (b * b * f) + 1e-24);
        output[offset] = ((d * f) - (e * e)) * inverseDeterminant;
        output[offset + 1] = ((e * c) - (b * f)) * inverseDeterminant;
        output[offset + 2] = ((e * b) - (c * d)) * inverseDeterminant;
        output[offset + 3] = ((a * f) - (c * c)) * inverseDeterminant;
        output[offset + 4] = ((b * c) - (e * a)) * inverseDeterminant;
        output[offset + 5] = ((a * d) - (b * b)) * inverseDeterminant;
    }
}
