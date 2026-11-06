namespace AiDotNet.Classification.Trees;

/// <summary>
/// Training features ranked once per feature, for growing decision trees: every value replaced by its position among
/// the feature's distinct values, column-major. A forest ranks its data once and every tree grows on it.
/// </summary>
/// <remarks>
/// <para>
/// Ranks keep exactly the order and the equalities of the original values, so a builder working on ranks finds the
/// same splits as one working on the values: it sorts integers (by radix for large nodes) instead of doubles, and
/// reads the original values back from <see cref="Values"/> only to form split thresholds.
/// </para>
/// <para>
/// NaN sorts below every number and all NaNs of a feature share rank 0, flagged by <see cref="HasNaN"/>.
/// </para>
/// </remarks>
internal sealed class DenseTreeData
{
    /// <summary>Rank of row <c>i</c> in feature <c>f</c>, at <c>f * Rows + i</c>.</summary>
    public int[] Ranks { get; }

    /// <summary>For each feature, its distinct values in ascending order: <c>Values[f][rank]</c>.</summary>
    public double[][] Values { get; }

    /// <summary>Whether feature <c>f</c>'s rank 0 stands for NaN.</summary>
    public bool[] HasNaN { get; }

    /// <summary>Bits needed to hold the largest rank of any feature.</summary>
    public int RankBits { get; }

    public int Rows { get; }

    public int Features { get; }

    public DenseTreeData(double[] columnMajor, int rows, int features)
    {
        Rows = rows;
        Features = features;
        Ranks = new int[rows * features];
        Values = new double[features][];
        HasNaN = new bool[features];

        var keys = new double[rows];
        var order = new int[rows];
        var distinct = new double[rows];
        int maxRank = 0;
        for (int f = 0; f < features; f++)
        {
            Array.Copy(columnMajor, f * rows, keys, 0, rows);
            for (int i = 0; i < rows; i++) order[i] = i;

            // double's default ordering puts NaN first and treats NaNs as equal.
            Array.Sort(keys, order);
            int rank = -1;
            int column = f * rows;
            for (int i = 0; i < rows; i++)
            {
                if (rank < 0 || !SameValue(keys[i], distinct[rank]))
                    distinct[++rank] = keys[i];
                Ranks[column + order[i]] = rank;
            }

            Values[f] = new double[rank + 1];
            Array.Copy(distinct, Values[f], rank + 1);
            HasNaN[f] = rows > 0 && double.IsNaN(Values[f][0]);
            maxRank = Math.Max(maxRank, rank);
        }

        int bits = 0;
        while (bits < 31 && (maxRank >> bits) != 0) bits++;
        RankBits = bits;
    }

    private static bool SameValue(double a, double b) => a == b || (double.IsNaN(a) && double.IsNaN(b));
}
