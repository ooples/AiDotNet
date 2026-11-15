using AiDotNet.NeuralRadianceFields.NeuS;

namespace AiDotNet.NeuralRadianceFields.Encodings;

/// <summary>
/// The multiresolution hash encoding of Instant-NGP (Müller et al. 2022, "Instant Neural Graphics Primitives with a
/// Multiresolution Hash Encoding", Sec. 3) in tiny-cuda-nn's <c>HashGrid</c> layout, with the encoding's exact spatial
/// derivative for SDF networks that need <c>∇ₓf</c> (NeuS-style Eikonal terms and normals).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Level <c>l</c> has the scale <c>s_l = b · σˡ − 1</c> and the resolution <c>⌈s_l⌉ + 1</c> (tiny-cuda-nn
/// <c>grid_scale</c> / <c>grid_resolution</c>, with base resolution <c>b</c> and per-level scale <c>σ</c>). A position
/// <c>x ∈ [0, 1]³</c> maps to <c>s_l · x + 0.5</c>; its eight surrounding grid vertices index the level's table densely
/// (<c>x + y·R + z·R²</c>) while the level's grid fits the table, and through the spatial hash
/// <c>(x·1) ⊕ (y·2654435761) ⊕ (z·805459861)</c> otherwise, both modulo the table size
/// <c>min(2^log2TableSize, R³ rounded up to a multiple of 8)</c>. The feature is the trilinear blend of the corners'
/// entries; the tables start uniform in <c>±10⁻⁴</c>.
/// </para>
/// <para>
/// The lookups are gathers of table rows weighted by host-computed corner weights
/// (<see cref="NeuSGridInterpolation{T}"/>), so the features — and their spatial derivatives
/// <c>Σₖ (∂wₖ/∂x) · table[idxₖ]</c>, the same gathers with the weights' derivatives — differentiate into the tables
/// on the tape. The engine's fused <c>MultiresolutionHashEncoding</c> that <see cref="AiDotNet.NeuralRadianceFields.Models.InstantNGP{T}"/> uses has no
/// position derivative, which is why an SDF backbone needs this form.
/// </para>
/// <para>
/// <see cref="ActiveLevels"/> implements instant-nsr-pl's <c>ProgressiveBandHashGrid</c>: the finer levels output
/// zero until they are switched on.
/// </para>
/// <para><b>For Beginners:</b> Instead of one big network, a hash grid stores small learned vectors at the corners of
/// many grids, from coarse to fine; a point's encoding blends the vectors of the corners around it. Training then only
/// needs a tiny network on top, which is why it is fast.</para>
/// </remarks>
internal sealed class MultiresolutionHashGrid<T>
{
    private const uint Prime1 = 2654435761u;
    private const uint Prime2 = 805459861u;

    private readonly Tensor<T>[] _tables;
    private int _activeLevels;

    /// <summary>Creates a grid with freshly initialized tables.</summary>
    /// <param name="levels">The level count L.</param>
    /// <param name="featuresPerLevel">The features per level F.</param>
    /// <param name="log2TableSize">The base-2 logarithm of the largest table.</param>
    /// <param name="baseResolution">The coarsest resolution b.</param>
    /// <param name="perLevelScale">The growth σ between levels.</param>
    /// <param name="random">The initialization generator.</param>
    public MultiresolutionHashGrid(int levels, int featuresPerLevel, int log2TableSize, int baseResolution,
        double perLevelScale, Random random)
    {
        if (levels <= 0) throw new ArgumentOutOfRangeException(nameof(levels), "At least one level is needed.");
        if (featuresPerLevel <= 0) throw new ArgumentOutOfRangeException(nameof(featuresPerLevel));
        if (log2TableSize <= 0 || log2TableSize > 30) throw new ArgumentOutOfRangeException(nameof(log2TableSize));
        if (baseResolution <= 0) throw new ArgumentOutOfRangeException(nameof(baseResolution));
        if (!(perLevelScale >= 1) || double.IsInfinity(perLevelScale)) throw new ArgumentOutOfRangeException(nameof(perLevelScale));
        if (random is null) throw new ArgumentNullException(nameof(random));
        Levels = levels;
        FeaturesPerLevel = featuresPerLevel;
        Scales = new double[levels];
        Resolutions = new int[levels];
        TableSizes = new int[levels];
        _tables = new Tensor<T>[levels];
        for (int l = 0; l < levels; l++)
        {
            Scales[l] = LevelScale(l, baseResolution, perLevelScale);
            Resolutions[l] = (int)Math.Ceiling(Scales[l]) + 1;
            TableSizes[l] = TableSize(Resolutions[l], log2TableSize);
            var values = new double[TableSizes[l] * featuresPerLevel];
            for (int i = 0; i < values.Length; i++) values[i] = (random.NextDouble() * 2.0 - 1.0) * 1e-4;
            _tables[l] = NeuSHost<T>.FromDoubles(values, TableSizes[l], featuresPerLevel);
        }

        _activeLevels = levels;
    }

    /// <summary>The level count.</summary>
    public int Levels { get; }

    /// <summary>The features per level.</summary>
    public int FeaturesPerLevel { get; }

    /// <summary>The encoded width, <c>Levels · FeaturesPerLevel</c>.</summary>
    public int OutputDims => Levels * FeaturesPerLevel;

    /// <summary>Each level's scale <c>s_l</c>.</summary>
    public double[] Scales { get; }

    /// <summary>Each level's grid resolution <c>⌈s_l⌉ + 1</c>.</summary>
    public int[] Resolutions { get; }

    /// <summary>Each level's table rows.</summary>
    public int[] TableSizes { get; }

    /// <summary>The tables, one <c>[TableSizes[l], FeaturesPerLevel]</c> tensor per level.</summary>
    public IReadOnlyList<Tensor<T>> Tables => _tables;

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => TableSizes.Sum() * FeaturesPerLevel;

    /// <summary>
    /// How many levels, coarsest first, contribute; the rest output zero (instant-nsr-pl's progressive band).
    /// </summary>
    public int ActiveLevels
    {
        get => _activeLevels;
        set
        {
            if (value < 0 || value > Levels) throw new ArgumentOutOfRangeException(nameof(value));
            _activeLevels = value;
        }
    }

    /// <summary>Level <paramref name="level"/>'s scale, <c>b · σˡ − 1</c> (tiny-cuda-nn <c>grid_scale</c>).</summary>
    /// <param name="level">The level.</param>
    /// <param name="baseResolution">The base resolution.</param>
    /// <param name="perLevelScale">The per-level scale.</param>
    internal static double LevelScale(int level, int baseResolution, double perLevelScale)
        => Math.Pow(perLevelScale, level) * baseResolution - 1.0;

    /// <summary>A level's table rows: <c>R³</c> rounded up to a multiple of 8, at most <c>2^log2TableSize</c>.</summary>
    /// <param name="resolution">The level's resolution.</param>
    /// <param name="log2TableSize">The base-2 logarithm of the largest table.</param>
    internal static int TableSize(int resolution, int log2TableSize)
    {
        long dense = (long)resolution * resolution * resolution;
        dense = (dense + 7) / 8 * 8;
        return (int)Math.Min(dense, 1L << log2TableSize);
    }

    /// <summary>
    /// The table row of grid vertex <c>(x, y, z)</c> (tiny-cuda-nn <c>grid_index</c>): dense while the level's grid fits
    /// the table, the spatial hash otherwise, modulo the table size.
    /// </summary>
    /// <param name="x">The vertex's x.</param>
    /// <param name="y">The vertex's y.</param>
    /// <param name="z">The vertex's z.</param>
    /// <param name="resolution">The level's resolution.</param>
    /// <param name="tableSize">The level's table rows.</param>
    internal static int VertexIndex(uint x, uint y, uint z, int resolution, int tableSize)
    {
        unchecked
        {
            uint size = (uint)tableSize;
            ulong stride = 1;
            ulong index = 0;
            var position = new[] { x, y, z };
            for (int d = 0; d < 3 && stride <= size; d++)
            {
                index += position[d] * stride;
                stride *= (ulong)resolution;
            }

            if (size < stride) index = x ^ (y * Prime1) ^ (z * Prime2);
            return (int)(index % size);
        }
    }

    /// <summary>
    /// Encodes <c>[N, 3]</c> positions in <c>[0, 1]³</c> (clamped there): the features <c>[N, OutputDims]</c>, level by
    /// level, and, when <paramref name="withJacobian"/>, their derivatives with respect to the three coordinates.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="unitPositions">The positions, <c>N · 3</c> values.</param>
    /// <param name="withJacobian">Whether to compute ∂features/∂x, ∂y, ∂z.</param>
    public (Tensor<T> Features, Tensor<T>[]? Jacobian) Encode(IEngine engine, double[] unitPositions, bool withJacobian)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (unitPositions is null || unitPositions.Length % 3 != 0)
            throw new ArgumentException("The positions must be N · 3 values.", nameof(unitPositions));
        int n = unitPositions.Length / 3;
        var features = new Tensor<T>[Levels];
        var jacobian = withJacobian ? new[] { new Tensor<T>[Levels], new Tensor<T>[Levels], new Tensor<T>[Levels] } : null;
        for (int l = 0; l < Levels; l++)
        {
            if (l >= _activeLevels)
            {
                features[l] = new Tensor<T>(new[] { n, FeaturesPerLevel });
                if (jacobian is not null)
                    for (int d = 0; d < 3; d++) jacobian[d][l] = new Tensor<T>(new[] { n, FeaturesPerLevel });
                continue;
            }

            var interpolation = Interpolation(unitPositions, l, withJacobian);
            features[l] = interpolation.Sample(engine, _tables[l]);
            if (jacobian is not null)
            {
                var derivatives = interpolation.SpatialDerivatives(engine, _tables[l]);
                for (int d = 0; d < 3; d++) jacobian[d][l] = derivatives[d];
            }
        }

        var encoded = Levels == 1 ? features[0] : engine.TensorConcatenate(features, 1);
        if (jacobian is null) return (encoded, null);
        var spatial = new Tensor<T>[3];
        for (int d = 0; d < 3; d++) spatial[d] = Levels == 1 ? jacobian[d][0] : engine.TensorConcatenate(jacobian[d], 1);
        return (encoded, spatial);
    }

    /// <summary>Level <paramref name="level"/>'s corner rows, weights and weight derivatives at the positions.</summary>
    internal NeuSGridInterpolation<T> Interpolation(double[] unitPositions, int level, bool withDerivatives)
    {
        int n = unitPositions.Length / 3;
        double scale = Scales[level];
        int resolution = Resolutions[level];
        int size = TableSizes[level];
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

        var lower = new uint[3];
        var frac = new double[3];
        for (int p = 0; p < n; p++)
        {
            for (int d = 0; d < 3; d++)
            {
                double u = unitPositions[p * 3 + d];
                u = double.IsNaN(u) ? 0.0 : Math.Max(0.0, Math.Min(1.0, u));
                double g = scale * u + 0.5;
                double f = Math.Floor(g);
                lower[d] = (uint)f;
                frac[d] = g - f;
            }

            for (int k = 0; k < 8; k++)
            {
                int ox = (k >> 2) & 1, oy = (k >> 1) & 1, oz = k & 1;
                double wx = ox == 1 ? frac[0] : 1.0 - frac[0];
                double wy = oy == 1 ? frac[1] : 1.0 - frac[1];
                double wz = oz == 1 ? frac[2] : 1.0 - frac[2];
                indices[k][p] = VertexIndex(lower[0] + (uint)ox, lower[1] + (uint)oy, lower[2] + (uint)oz, resolution, size);
                weights[k][p] = wx * wy * wz;
                if (derivatives is not null)
                {
                    double sx = ox == 1 ? scale : -scale, sy = oy == 1 ? scale : -scale, sz = oz == 1 ? scale : -scale;
                    derivatives[0][k][p] = sx * wy * wz;
                    derivatives[1][k][p] = wx * sy * wz;
                    derivatives[2][k][p] = wx * wy * sz;
                }
            }
        }

        return NeuSGridInterpolation<T>.FromCorners(n, indices, weights, derivatives);
    }

    /// <summary>The tables under the names <c>level{l}</c>.</summary>
    /// <param name="prefix">The name prefix.</param>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters(string prefix)
    {
        for (int l = 0; l < Levels; l++) yield return new KeyValuePair<string, Tensor<T>>(prefix + "level" + l, _tables[l]);
    }

    /// <summary>Loads one named table; returns whether the name belongs to this grid.</summary>
    /// <param name="prefix">The name prefix.</param>
    /// <param name="name">The name.</param>
    /// <param name="value">The table.</param>
    public bool TryLoad(string prefix, string name, Tensor<T> value)
    {
        if (name is null || value is null) return false;
        for (int l = 0; l < Levels; l++)
        {
            if (name != prefix + "level" + l) continue;
            if (value.Length != _tables[l].Length)
                throw new ArgumentException($"Hash grid level {l} has {_tables[l].Length} values; got {value.Length}.", nameof(value));
            value.AsSpan().CopyTo(_tables[l].AsWritableSpan());
            return true;
        }

        return false;
    }
}
