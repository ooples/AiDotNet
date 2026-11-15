namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// Bilinear and trilinear interpolation of a feature grid at host-known positions, as gathers of the grid's corner
/// rows weighted by host-computed corner weights, with the exact spatial derivative.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is <c>F.grid_sample(..., mode='bilinear', padding_mode='zeros', align_corners=True)</c> on 4-D and 5-D inputs:
/// a corner outside the grid contributes zero. The sample positions of NeuS-style rendering are constants of the
/// sampling, so the corner indices and weights are built on the host, and the result
/// <c>Σₖ wₖ(p) · grid[idxₖ]</c> is differentiable with respect to the grid (a 3-D CNN's output, image features).
/// </para>
/// <para>
/// The trilinear derivative <c>∂/∂p Σₖ wₖ(p) grid[idxₖ] = Σₖ (∂wₖ/∂p) grid[idxₖ]</c> uses the same gathers with the
/// weights' derivatives; this is what autograd through <c>grid_sample</c> gives SparseNeuS's SDF gradient, and it
/// enters <see cref="NeuSSdfMlp{T}.Forward"/> as the latent Jacobian.
/// </para>
/// </remarks>
internal sealed class NeuSGridInterpolation<T>
{
    private readonly int[][] _indices;
    private readonly double[][] _weights;
    private readonly double[][][]? _derivatives;
    private readonly int _count;

    private NeuSGridInterpolation(int count, int[][] indices, double[][] weights, double[][][]? derivatives)
    {
        _count = count;
        _indices = indices;
        _weights = weights;
        _derivatives = derivatives;
    }

    /// <summary>The number of interpolated positions.</summary>
    public int Count => _count;

    /// <summary>
    /// An interpolation from precomputed corners: per corner, each position's row index, weight and (optionally) the
    /// weight's derivatives with respect to x, y and z. Lets a grid with its own indexing (a hash grid's levels) reuse
    /// the gathers.
    /// </summary>
    /// <param name="count">The number of positions.</param>
    /// <param name="indices">Per corner, the row of each position.</param>
    /// <param name="weights">Per corner, the weight of each position.</param>
    /// <param name="derivatives">Per axis, per corner, the weight's derivative at each position; or null.</param>
    public static NeuSGridInterpolation<T> FromCorners(int count, int[][] indices, double[][] weights, double[][][]? derivatives)
    {
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (indices.Length == 0 || indices.Length != weights.Length)
            throw new ArgumentException("Every corner needs indices and weights.", nameof(weights));
        if (derivatives is not null && derivatives.Length != 3)
            throw new ArgumentException("The derivatives need three axes.", nameof(derivatives));
        return new NeuSGridInterpolation<T>(count, indices, weights, derivatives);
    }

    /// <summary>
    /// Trilinear interpolation of a <c>[X, Y, Z]</c> grid (flattened x-slowest) at continuous grid coordinates.
    /// </summary>
    /// <param name="gridCoordinates">Positions in voxel units, <c>[N, 3]</c> (index i at coordinate i).</param>
    /// <param name="dims">X, Y, Z.</param>
    /// <param name="voxelSize">World units per voxel, for derivatives with respect to world position.</param>
    /// <param name="withDerivatives">Whether to prepare ∂/∂x, ∂/∂y, ∂/∂z.</param>
    public static NeuSGridInterpolation<T> Trilinear(double[] gridCoordinates, int[] dims, double voxelSize, bool withDerivatives)
    {
        int n = gridCoordinates.Length / 3;
        var indices = new int[8][];
        var weights = new double[8][];
        var derivatives = withDerivatives ? new double[3][][] : null;
        for (int k = 0; k < 8; k++)
        {
            indices[k] = new int[n];
            weights[k] = new double[n];
        }

        if (derivatives is not null)
            for (int d = 0; d < 3; d++)
            {
                derivatives[d] = new double[8][];
                for (int k = 0; k < 8; k++) derivatives[d][k] = new double[n];
            }

        var lower = new int[3];
        var frac = new double[3];
        for (int p = 0; p < n; p++)
        {
            for (int d = 0; d < 3; d++)
            {
                double c = gridCoordinates[p * 3 + d];
                double f = Math.Floor(c);
                lower[d] = double.IsNaN(c) ? int.MinValue / 2 : (int)Math.Max(int.MinValue / 2, Math.Min(int.MaxValue / 2, f));
                frac[d] = c - f;
            }

            for (int k = 0; k < 8; k++)
            {
                int ox = (k >> 2) & 1, oy = (k >> 1) & 1, oz = k & 1;
                int ix = lower[0] + ox, iy = lower[1] + oy, iz = lower[2] + oz;
                bool inside = ix >= 0 && ix < dims[0] && iy >= 0 && iy < dims[1] && iz >= 0 && iz < dims[2];
                double wx = ox == 1 ? frac[0] : 1.0 - frac[0];
                double wy = oy == 1 ? frac[1] : 1.0 - frac[1];
                double wz = oz == 1 ? frac[2] : 1.0 - frac[2];
                indices[k][p] = inside ? (ix * dims[1] + iy) * dims[2] + iz : 0;
                weights[k][p] = inside ? wx * wy * wz : 0.0;
                if (derivatives is not null)
                {
                    double sx = ox == 1 ? 1.0 : -1.0, sy = oy == 1 ? 1.0 : -1.0, sz = oz == 1 ? 1.0 : -1.0;
                    derivatives[0][k][p] = inside ? sx * wy * wz / voxelSize : 0.0;
                    derivatives[1][k][p] = inside ? wx * sy * wz / voxelSize : 0.0;
                    derivatives[2][k][p] = inside ? wx * wy * sz / voxelSize : 0.0;
                }
            }
        }

        return new NeuSGridInterpolation<T>(n, indices, weights, derivatives);
    }

    /// <summary>
    /// Bilinear interpolation of an <c>[H, W]</c> grid (flattened row-major) at pixel coordinates (u = column, v = row),
    /// with <c>align_corners=True</c> semantics.
    /// </summary>
    /// <param name="u">Columns, <c>[N]</c>.</param>
    /// <param name="v">Rows, <c>[N]</c>.</param>
    /// <param name="height">H.</param>
    /// <param name="width">W.</param>
    public static NeuSGridInterpolation<T> Bilinear(double[] u, double[] v, int height, int width)
    {
        int n = u.Length;
        var indices = new int[4][];
        var weights = new double[4][];
        for (int k = 0; k < 4; k++)
        {
            indices[k] = new int[n];
            weights[k] = new double[n];
        }

        for (int p = 0; p < n; p++)
        {
            bool finite = !double.IsNaN(u[p]) && !double.IsNaN(v[p]) && !double.IsInfinity(u[p]) && !double.IsInfinity(v[p]);
            double fu = finite ? Math.Floor(u[p]) : 0.0, fv = finite ? Math.Floor(v[p]) : 0.0;
            double au = u[p] - fu, av = v[p] - fv;
            int x0 = finite ? (int)Math.Max(-2, Math.Min(width + 1, fu)) : -2;
            int y0 = finite ? (int)Math.Max(-2, Math.Min(height + 1, fv)) : -2;
            for (int k = 0; k < 4; k++)
            {
                int ox = k & 1, oy = (k >> 1) & 1;
                int x = x0 + ox, y = y0 + oy;
                bool inside = finite && x >= 0 && x < width && y >= 0 && y < height;
                indices[k][p] = inside ? y * width + x : 0;
                weights[k][p] = inside ? (ox == 1 ? au : 1.0 - au) * (oy == 1 ? av : 1.0 - av) : 0.0;
            }
        }

        return new NeuSGridInterpolation<T>(n, indices, weights, null);
    }

    /// <summary>The interpolated rows, <c>[N, C]</c>, of a <c>[M, C]</c> grid.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="grid">The grid, one row per cell.</param>
    public Tensor<T> Sample(IEngine engine, Tensor<T> grid) => Combine(engine, grid, _weights);

    /// <summary>∂(interpolated rows)/∂(world x, y, z): three <c>[N, C]</c> tensors.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="grid">The grid, one row per cell.</param>
    public Tensor<T>[] SpatialDerivatives(IEngine engine, Tensor<T> grid)
    {
        var derivatives = _derivatives ?? throw new InvalidOperationException("This interpolation was built without derivatives.");
        return new[] { Combine(engine, grid, derivatives[0]), Combine(engine, grid, derivatives[1]), Combine(engine, grid, derivatives[2]) };
    }

    /// <summary>Interpolates a host scalar grid (a validity mask), <c>[N]</c>.</summary>
    /// <param name="values">The grid values, one per cell.</param>
    public double[] SampleHost(double[] values)
    {
        var result = new double[_count];
        for (int k = 0; k < _indices.Length; k++)
            for (int p = 0; p < _count; p++)
                result[p] += _weights[k][p] * values[_indices[k][p]];
        return result;
    }

    private Tensor<T> Combine(IEngine engine, Tensor<T> grid, double[][] weights)
    {
        Tensor<T>? total = null;
        for (int k = 0; k < _indices.Length; k++)
        {
            var rows = engine.TensorIndexSelect(grid, NeuSHost<T>.Indices(_indices[k]), 0);
            var term = engine.TensorMultiply(rows, NeuSHost<T>.FromDoubles(weights[k], _count, 1));
            total = total is null ? term : engine.TensorAdd(total, term);
        }

        return total ?? throw new InvalidOperationException("No interpolation corners.");
    }
}
