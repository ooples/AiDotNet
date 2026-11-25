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
/// Each feature column is sorted with its row indices, a new rank starts wherever the sorted value changes, and each
/// row's rank is written back through the sort order; features rank in parallel. All NaNs of a feature share one
/// rank, <see cref="NanRank"/>, which is <see cref="Rows"/>: above every numeric rank, so a NaN is never taken for a
/// genuine <c>MinValue</c> (the value NaN is sorted as).
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
    public T[] DistinctValues { get; }

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
            DistinctValues = new T[0];
            ValueOffsets = new int[features];
            NanRank = Enumerable.Repeat(-1, features).ToArray();
            RankBits = 0;
            return;
        }

        // Row-major copy of the data; each worker reads its feature's column out of it.
        var data = Tensor<T>.FromMatrix(x).ToArray();
        var ranks = new int[rows * features];
        var distinct = new T[features][];
        var nanRank = new int[features];
        var comparer = ValueComparer();
        System.Threading.Tasks.Parallel.For(0, features,
            () => (Keys: new T[rows], Order: new int[rows]),
            (f, _, buffers) =>
            {
                RankFeature(data, f, features, rows, buffers.Keys, buffers.Order, comparer, ranks, out distinct[f],
                    out nanRank[f]);
                return buffers;
            },
            _ => { });

        Ranks = ranks;
        NanRank = nanRank;
        ValueOffsets = new int[features];
        int offset = 0;
        int maxRank = 0;
        for (int f = 0; f < features; f++)
        {
            ValueOffsets[f] = offset;
            offset += distinct[f].Length;
            maxRank = Math.Max(maxRank, Math.Max(distinct[f].Length - 1, nanRank[f]));
        }

        DistinctValues = new T[offset];
        for (int f = 0; f < features; f++)
            Array.Copy(distinct[f], 0, DistinctValues, ValueOffsets[f], distinct[f].Length);

        int bits = 0;
        while (bits < 31 && (maxRank >> bits) != 0) bits++;
        RankBits = bits;
    }

    /// <summary>
    /// Ranks feature <paramref name="feature"/>: writes each row's rank into its column of <paramref name="ranks"/> and
    /// returns the distinct values ascending and the NaN rank (<paramref name="rows"/>, or -1 without NaNs). A NaN sorts
    /// as <c>MinValue</c>, so a feature with NaNs keeps <c>MinValue</c> among its distinct values.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Sorts one feature column with its row indices and walks it once; the engine form cast and masked the whole matrix and took 0.8 s on 20,000 x 20.")]
    private static void RankFeature(T[] data, int feature, int features, int rows, T[] keys, int[] order,
        IComparer<T> comparer, int[] ranks, out T[] distinct, out int nanRank)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T lowest = numOps.MinValue;
        bool hasNaN = false;
        for (int i = 0; i < rows; i++)
        {
            T value = data[i * features + feature];
            bool isNaN = numOps.IsNaN(value);
            hasNaN |= isNaN;
            keys[i] = isNaN ? lowest : value;
            order[i] = i;
        }

        Array.Sort(keys, order, 0, rows, comparer);

        var values = new List<T>();
        int column = feature * rows;
        int rank = -1;
        for (int i = 0; i < rows; i++)
        {
            if (rank < 0 || comparer.Compare(keys[i], values[rank]) != 0)
            {
                values.Add(keys[i]);
                rank++;
            }

            ranks[column + order[i]] = rank;
        }

        distinct = values.ToArray();
        nanRank = -1;
        if (!hasNaN)
            return;

        // NaN sorted as MinValue and so shares its rank with a genuine MinValue; give the NaNs their own rank, rows,
        // which no numeric rank reaches.
        nanRank = rows;
        for (int i = 0; i < rows; i++)
        {
            if (numOps.IsNaN(data[i * features + feature]))
                ranks[column + i] = rows;
        }
    }

    /// <summary>Ascending order of T: the type's own comparison for double and float, NumOps.Compare otherwise.</summary>
    private static IComparer<T> ValueComparer()
    {
        if (typeof(T) == typeof(double) || typeof(T) == typeof(float))
            return Comparer<T>.Default;

        var numOps = MathHelper.GetNumericOperations<T>();
        return Comparer<T>.Create((a, b) => numOps.Compare(a, b));
    }
}
