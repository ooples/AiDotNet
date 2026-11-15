namespace AiDotNet.Classification.Trees;

/// <summary>
/// Training features ranked once per feature, for growing decision trees: every value replaced by its position among
/// the feature's distinct values, column-major. A forest ranks its data once and every tree grows on it.
/// </summary>
/// <typeparam name="T">The numeric type of the features.</typeparam>
/// <remarks>
/// <para>
/// Ranks keep exactly the order and the equalities of the original values, so a builder working on ranks finds the
/// same splits as one working on the values: it sorts integers (by radix for large nodes) instead of values, and
/// reads the original values back from <see cref="DistinctValues"/> only to form split thresholds, in T.
/// </para>
/// <para>
/// The ranking is engine work: each feature column is sorted, a new rank starts wherever the sorted value changes, and
/// the ranks are put back in row order through the inverse permutation. All NaNs of a feature share one rank,
/// <see cref="NanRank"/>, which is <see cref="Rows"/>: above every numeric rank, so a NaN is never taken for a genuine
/// <c>MinValue</c> (the value NaN is sorted as).
/// </para>
/// </remarks>
internal sealed class DenseTreeData<T>
{
    /// <summary>Rank of row <c>i</c> in feature <c>f</c>, at <c>f * Rows + i</c>.</summary>
    public int[] Ranks { get; }

    /// <summary>
    /// Every feature's distinct values in ascending order, one after another: feature <c>f</c>'s value of rank
    /// <c>r</c> is at <c>ValueOffsets[f] + r</c>.
    /// </summary>
    public Tensor<T> DistinctValues { get; }

    /// <summary>Where each feature's distinct values start in <see cref="DistinctValues"/>.</summary>
    public int[] ValueOffsets { get; }

    /// <summary>The rank feature <c>f</c>'s NaNs share, or -1 when it has none.</summary>
    public int[] NanRank { get; }

    /// <summary>Bits needed to hold the largest rank of any feature.</summary>
    public int RankBits { get; }

    public int Rows { get; }

    public int Features { get; }

    public DenseTreeData(Matrix<T> x)
    {
        int rows = x.Rows;
        int features = x.Columns;
        Rows = rows;
        Features = features;
        if (rows == 0 || features == 0)
        {
            Ranks = new int[0];
            DistinctValues = new Tensor<T>(new[] { 0 });
            ValueOffsets = new int[features];
            NanRank = Enumerable.Repeat(-1, features).ToArray();
            RankBits = 0;
            return;
        }

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // [features, rows]; NaN becomes the smallest value so it sorts first, flagged by a 1/0 mask.
        var columns = engine.TensorTranspose(Tensor<T>.FromMatrix(x));
        var isNan = engine.TensorIsNan(columns);
        var nanMask = engine.TensorWhere(isNan, engine.TensorOnesLike(columns), engine.TensorZerosLike(columns)).Cast<int>();
        var keys = engine.TensorWhere(isNan, engine.TensorFullLike(columns, numOps.MinValue), columns);
        var (sorted, order) = engine.TensorSort(keys, 1, false);

        // A rank starts at the first sorted value and wherever the value changes.
        Tensor<int> starts;
        if (rows > 1)
        {
            var changed = engine.TensorNotEquals(
                engine.TensorSlice(sorted, new[] { 0, 1 }, new[] { features, rows - 1 }),
                engine.TensorSlice(sorted, new[] { 0, 0 }, new[] { features, rows - 1 })).Cast<int>();
            var first = new Tensor<int>(Enumerable.Repeat(1, features).ToArray(), new[] { features, 1 });
            starts = engine.TensorConcatenate(new[] { first, changed }, 1);
        }
        else
        {
            starts = new Tensor<int>(Enumerable.Repeat(1, features).ToArray(), new[] { features, 1 });
        }

        var sortedRanks = engine.TensorSubtractScalar(engine.TensorCumSum(starts, 1), 1);
        var ranks = engine.TensorTakeAlongDim(sortedRanks, engine.TensorArgsort(order, 1, false), 1);
        // NaN sorted as MinValue and so shares its rank with a genuine MinValue; give the NaNs their own rank, rows,
        // which no numeric rank reaches.
        ranks = engine.TensorWhere(nanMask, engine.TensorFullLike(ranks, rows), ranks);
        Ranks = ranks.ToArray();

        // The distinct values are the sorted values at the rank starts, feature after feature.
        var startFlags = starts.ToArray();
        var distinctPositions = Enumerable.Range(0, startFlags.Length).Where(i => startFlags[i] == 1).ToArray();
        DistinctValues = engine.TensorTake(
            engine.Reshape(sorted, new[] { features * rows }),
            new Tensor<int>(distinctPositions, new[] { distinctPositions.Length }));
        var distinctCounts = engine.ReduceSum(starts, new[] { 1 }, false).ToArray();
        ValueOffsets = new int[features];
        int offset = 0;
        for (int f = 0; f < features; f++)
        {
            ValueOffsets[f] = offset;
            offset += distinctCounts[f];
        }

        // A NaN row's rank, or -1: rank * isNaN + isNaN - 1 is the rank on NaN rows and -1 elsewhere.
        NanRank = engine.ReduceMax(
            engine.TensorSubtractScalar(engine.TensorAdd(engine.TensorMultiply(ranks, nanMask), nanMask), 1),
            new[] { 1 }, false).ToArray();

        int maxRank = Math.Max(distinctCounts.Max() - 1, NanRank.Max());
        int bits = 0;
        while (bits < 31 && (maxRank >> bits) != 0) bits++;
        RankBits = bits;
    }
}
