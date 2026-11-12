using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// The maximal information coefficient (MIC) of Reshef et al. 2011 ("Detecting Novel Associations in Large Data
/// Sets", Science 334), computed with the ApproxMaxMI heuristic of the paper's supplement (SOM, Algorithms 2-5)
/// exactly as minepy/libmine implements it (<c>mine_compute_score</c> with <c>EST_MIC_APPROX</c>, then
/// <c>mine_mic</c>).
/// </summary>
/// <remarks>
/// <para>
/// For a sample of size n the grid budget is <c>B = max(n^alpha, 4)</c> when <c>0 &lt; alpha &lt;= 1</c> and
/// <c>B = min(alpha, n)</c> when <c>alpha &gt;= 4</c>. For every y-axis size <c>y = 2 .. floor(B/2)</c> the y axis is
/// equipartitioned, the x axis is split into at most <c>c * (m + 1)</c> superclumps (<c>m = floor(B/y) - 1</c>) and
/// the dynamic program OptimizeXAxis finds, for every x-axis size <c>2 .. m + 1</c>, the x partition with the largest
/// mutual information. Each value is normalized by <c>log(min(x, q))</c>; the same is done with the axes swapped and
/// MIC is the largest normalized value.
/// </para>
/// <para>
/// The partition maps are integer label data; the entropies and the dynamic program run as whole-array engine
/// operations in T over the cumulative histogram of the grid.
/// </para>
/// </remarks>
internal static class MaximalInformationCoefficient
{
    /// <summary>The paper's default exponent of the grid budget <c>B(n) = n^alpha</c>.</summary>
    public const double DefaultAlpha = 0.6;

    /// <summary>The paper's default clump factor (at most <c>c * x</c> superclumps on the x axis).</summary>
    public const double DefaultClumps = 15;

    /// <summary>Checks the MINE parameters the way <c>mine_check_parameter</c> does.</summary>
    public static void Validate(double alpha, double c, string alphaName, string cName)
    {
        if (!((alpha > 0.0 && alpha <= 1.0) || alpha >= 4.0))
            throw new ArgumentOutOfRangeException(alphaName, "alpha must be in (0, 1] or at least 4.");
        if (!(c > 0.0))
            throw new ArgumentOutOfRangeException(cName, "c must be greater than 0.");
    }

    /// <summary>
    /// MIC between every column of <paramref name="x"/> ([n, p]) and <paramref name="y"/> ([n]); returns [p].
    /// </summary>
    public static Tensor<T> Scores<T>(Tensor<T> x, Tensor<T> y, double alpha, double c)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (p == 0)
            return new Tensor<T>(new[] { 0 });

        var yColumn = engine.Reshape(y, new[] { n });
        var yOrder = FeatureSelectionTensorOps.StableOrder(yColumn, descending: false);
        var yGroups = TieGroups(yColumn, yOrder);

        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var xColumn = engine.Reshape(engine.TensorSlice(x, new[] { 0, j }, new[] { n, 1 }), new[] { n });
            var xOrder = FeatureSelectionTensorOps.StableOrder(xColumn, descending: false);
            columns[j] = Mic<T>(n, xOrder, TieGroups(xColumn, xOrder), yOrder, yGroups, alpha, c);
        }

        return engine.TensorMax(engine.TensorConcatenate(columns, 0), MathHelper.GetNumericOperations<T>().Zero);
    }

    /// <summary>
    /// For the values in ascending order (<paramref name="order"/>), the index of the first sorted value equal to
    /// each one, so tied values share a label.
    /// </summary>
    private static int[] TieGroups<T>(Tensor<T> values, int[] order)
    {
        var engine = AiDotNetEngine.Current;
        if (order.Length == 0)
            return Array.Empty<int>();

        var sorted = engine.TensorIndexSelect(values, new Tensor<int>((int[])order.Clone(), new[] { order.Length }), 0);
        return engine.TensorSearchSorted(sorted, sorted, false).ToArray();
    }

    /// <summary>MIC of one variable pair given the sort orders and tie labels of both axes; returns a [1] tensor.</summary>
    private static Tensor<T> Mic<T>(int n, int[] xOrder, int[] xGroups, int[] yOrder, int[] yGroups, double alpha, double c)
    {
        var engine = AiDotNetEngine.Current;
        double budget = alpha > 0.0 && alpha <= 1.0 ? Math.Max(Math.Pow(n, alpha), 4) : Math.Min(alpha, n);
        int rows = Math.Max((int)Math.Floor(budget / 2.0), 2) - 1;

        var scores = new List<Tensor<T>>();
        var temp = new int[n];
        for (int i = 0; i < rows; i++)
        {
            int columnsMax = (int)Math.Floor(budget / (i + 2)) - 1;
            int clumps = Math.Max((int)(c * (columnsMax + 1)), 1);

            // x vs. y: equipartition y into i + 2 rows, superclumps on x
            var q = EquipartitionYAxis(yGroups, n, i + 2, out int qCount);
            var qByX = new int[n];
            for (int k = 0; k < n; k++)
                temp[yOrder[k]] = q[k];
            for (int k = 0; k < n; k++)
                qByX[k] = temp[xOrder[k]];
            var pMap = GetSuperclumpsPartition(xGroups, n, clumps, qByX, out int pCount);
            scores.Add(OptimizeXAxis<T>(n, qByX, qCount, pMap, pCount, columnsMax + 1));

            // y vs. x: equipartition x into i + 2 rows, superclumps on y
            q = EquipartitionYAxis(xGroups, n, i + 2, out qCount);
            var qByY = new int[n];
            for (int k = 0; k < n; k++)
                temp[xOrder[k]] = q[k];
            for (int k = 0; k < n; k++)
                qByY[k] = temp[yOrder[k]];
            pMap = GetSuperclumpsPartition(yGroups, n, clumps, qByY, out pCount);
            scores.Add(OptimizeXAxis<T>(n, qByY, qCount, pMap, pCount, columnsMax + 1));
        }

        var nonEmpty = scores.Where(s => s.Length > 0).ToArray();
        if (nonEmpty.Length == 0)
            return new Tensor<T>(new[] { 1 });

        return engine.ReduceMax(engine.TensorConcatenate(nonEmpty, 0), new[] { 0 }, true);
    }

    /// <summary>
    /// SOM Algorithm 3 (libmine <c>EquipartitionYAxis</c>): splits the sorted points into at most
    /// <paramref name="y"/> rows of about equal size without separating tied values. <paramref name="groups"/>
    /// holds equal labels for tied values, in sorted order.
    /// </summary>
    internal static int[] EquipartitionYAxis(int[] groups, int n, int y, out int q)
    {
        var map = new int[n];
        double rowSize = (double)n / y;
        int i = 0;
        int h = 0;
        int current = 0;
        while (i < n)
        {
            int s = 1;
            for (int j = i + 1; j < n && groups[j] == groups[i]; j++)
                s++;

            double withRun = Math.Abs((double)h + s - rowSize);
            double without = Math.Abs((double)h - rowSize);
            if (h != 0 && withRun >= without)
            {
                current++;
                h = 0;
                rowSize = (double)(n - i) / (y - current);
            }

            for (int j = 0; j < s; j++)
                map[i + j] = current;
            i += s;
            h += s;
        }

        q = current + 1;
        return map;
    }

    /// <summary>
    /// libmine <c>GetClumpsPartition</c>: consecutive points (in x order) with the same row form a clump, and a run
    /// of tied x values that spans several rows forms a clump of its own.
    /// </summary>
    private static int[] GetClumpsPartition(int[] xGroups, int n, int[] qMap, out int p)
    {
        var tilde = (int[])qMap.Clone();
        int label = -1;
        int i = 0;
        while (i < n)
        {
            int s = 1;
            bool mixed = false;
            for (int j = i + 1; j < n && xGroups[j] == xGroups[i]; j++)
            {
                if (tilde[i] != tilde[j])
                    mixed = true;
                s++;
            }

            if (s > 1 && mixed)
            {
                for (int j = 0; j < s; j++)
                    tilde[i + j] = label;
                label--;
            }

            i += s;
        }

        var map = new int[n];
        int clump = 0;
        for (int j = 1; j < n; j++)
        {
            if (tilde[j] != tilde[j - 1])
                clump++;
            map[j] = clump;
        }

        p = clump + 1;
        return map;
    }

    /// <summary>
    /// SOM Algorithm 5 (libmine <c>GetSuperclumpsPartition</c>): the clumps, merged into at most
    /// <paramref name="maxClumps"/> superclumps by equipartitioning the clump labels when there are too many.
    /// </summary>
    private static int[] GetSuperclumpsPartition(int[] xGroups, int n, int maxClumps, int[] qMap, out int p)
    {
        var map = GetClumpsPartition(xGroups, n, qMap, out p);
        if (p > maxClumps)
            map = EquipartitionYAxis(map, n, maxClumps, out p);
        return map;
    }

    /// <summary>
    /// SOM Algorithm 2 (libmine <c>OptimizeXAxis</c>): for every x-axis size l = 2 .. <paramref name="maxColumns"/>
    /// the largest mutual information over the x partitions built from the <paramref name="p"/> superclumps, with
    /// the rows fixed to the <paramref name="q"/>-row partition <paramref name="qMap"/>, normalized by
    /// <c>log(min(l, q))</c>. Returns [maxColumns - 1].
    /// </summary>
    private static Tensor<T> OptimizeXAxis<T>(int n, int[] qMap, int q, int[] pMap, int p, int maxColumns)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int width = Math.Max(maxColumns - 1, 0);
        if (width == 0 || p == 1 || q == 1)
            return new Tensor<T>(new[] { width }); // libmine scores 0 (or 0/0, which mine_mic ignores)

        T one = ops.One;
        T minusOne = ops.Negate(one);

        // Cumulative histogram along the superclumps: cumhist[r, t] = #{points in row r and clumps 0 .. t}
        var keys = new int[n];
        for (int k = 0; k < n; k++)
            keys[k] = qMap[k] * p + pMap[k];
        var counts = engine.TensorBinCount(new Tensor<int>(keys, new[] { n }), q * p).Cast<T>();
        var cumhist = engine.TensorCumSum(engine.Reshape(counts, new[] { q, p }), 1); // [q, p]
        var cumulative = engine.ReduceSum(cumhist, new[] { 0 }, false); // c_1 .. c_p, [p]

        // Pairs (s, t) are indexed [s - 1, t - 1]
        var cS = engine.TensorBroadcastTo(engine.Reshape(cumulative, new[] { p, 1 }), new[] { p, p });
        var cT = engine.TensorBroadcastTo(engine.Reshape(cumulative, new[] { 1, p }), new[] { p, p });
        var cS3 = engine.TensorBroadcastTo(engine.Reshape(cumulative, new[] { 1, p, 1 }), new[] { q, p, p });
        var cT3 = engine.TensorBroadcastTo(engine.Reshape(cumulative, new[] { 1, 1, p }), new[] { q, p, p });
        var aS = engine.TensorBroadcastTo(engine.Reshape(cumhist, new[] { q, p, 1 }), new[] { q, p, p });
        var aT = engine.TensorBroadcastTo(engine.Reshape(cumhist, new[] { q, 1, p }), new[] { q, p, p });
        var between = engine.TensorSubtract(aT, aS);

        // H(Q), and the partition entropies of SOM Algorithm 2 (libmine hp3, hp3q and hp2q)
        var rowTotals = engine.TensorSlice(cumhist, new[] { 0, p - 1 }, new[] { q, 1 });
        T hq = engine.TensorSum(EntropyTerms(rowTotals, FeatureSelectionTensorOps.Filled(new[] { q, 1 }, ops.FromDouble(n))));
        var hp3 = engine.TensorAdd(EntropyTerms(cS, cT), EntropyTerms(engine.TensorSubtract(cT, cS), cT));
        var hp3q = engine.ReduceSum(engine.TensorAdd(EntropyTerms(aS, cT3), EntropyTerms(between, cT3)), new[] { 0 }, false);
        var hp2q = engine.ReduceSum(EntropyTerms(between, engine.TensorSubtract(cT3, cS3)), new[] { 0 }, false);

        // Every partition score lies within [-2 log n, log n]; invalid (s, t) pairs drop below that range
        T exclusion = ops.FromDouble(4 * Math.Log(Math.Max(n, 2)) + 4);
        var ones = FeatureSelectionTensorOps.Filled(new[] { p, p }, one);

        // Partitions into two columns: I[t][2] = H(Q) + max_{1 <= s <= t} (hp3(s, t) - hp3q(s, t))
        var level = MaskedColumnMax(engine.TensorSubtract(hp3, hp3q), engine.TensorTriu(ones, 0), minusOne, exclusion);
        level = engine.TensorAddScalar(level, hq);
        var atLastClump = new List<Tensor<T>> { engine.TensorSlice(level, new[] { p - 1 }, new[] { 1 }) };

        // I[t][l] = H(Q) + max_{l-1 <= s <= t} (c_s / c_t (I[s][l-1] - H(Q)) - (c_t - c_s) / c_t hp2q(s, t))
        var ratio = engine.TensorDivide(cS, cT);
        var weighted = engine.TensorMultiply(engine.TensorDivide(engine.TensorSubtract(cT, cS), cT), hp2q);
        T minusHq = ops.Negate(hq);
        int lastLevel = Math.Min(maxColumns, p);
        for (int l = 3; l <= lastLevel; l++)
        {
            int first = l - 2; // row of s = l - 1
            int count = p - first;
            var previous = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorSlice(level, new[] { first }, new[] { count }), new[] { count, 1 }), new[] { count, p });
            var candidates = engine.TensorSubtract(
                engine.TensorMultiply(
                    engine.TensorSlice(ratio, new[] { first, 0 }, new[] { count, p }),
                    engine.TensorAddScalar(previous, minusHq)),
                engine.TensorSlice(weighted, new[] { first, 0 }, new[] { count, p }));
            var valid = engine.TensorTriu(engine.TensorSlice(ones, new[] { first, 0 }, new[] { count, p }), first);
            level = engine.TensorAddScalar(MaskedColumnMax(candidates, valid, minusOne, exclusion), hq);
            atLastClump.Add(engine.TensorSlice(level, new[] { p - 1 }, new[] { 1 }));
        }

        // SOM Algorithm 2, line 19: sizes beyond the number of superclumps keep I[p][p]
        var best = atLastClump[atLastClump.Count - 1];
        while (atLastClump.Count < width)
            atLastClump.Add(best);

        // score[l - 2] = I[p][l] / min(log l, log q)
        var sizes = engine.TensorAddScalar(engine.TensorCumSum(FeatureSelectionTensorOps.Filled(new[] { width }, one), 0), one);
        var normalizer = engine.TensorMin(engine.TensorLog(sizes), ops.FromDouble(Math.Log(q)));
        return engine.TensorDivide(engine.TensorConcatenate(atLastClump.ToArray(), 0), normalizer);
    }

    /// <summary>
    /// Column-wise maximum of <paramref name="values"/> over the entries where <paramref name="mask"/> is 1 (every
    /// masked entry is pushed <paramref name="exclusion"/> below its value).
    /// </summary>
    private static Tensor<T> MaskedColumnMax<T>(Tensor<T> values, Tensor<T> mask, T minusOne, T exclusion)
    {
        var engine = AiDotNetEngine.Current;
        var penalty = engine.TensorMultiplyScalar(engine.TensorAddScalar(mask, minusOne), exclusion);
        return engine.ReduceMax(engine.TensorAdd(values, penalty), new[] { 0 }, false);
    }

    /// <summary>
    /// The entropy terms <c>-(a / total) log(a / total)</c> of non-negative integer counts, 0 where a is 0 (and
    /// finite, unused values where the inputs are not valid counts).
    /// </summary>
    private static Tensor<T> EntropyTerms<T>(Tensor<T> counts, Tensor<T> total)
    {
        var engine = AiDotNetEngine.Current;
        T one = MathHelper.GetNumericOperations<T>().One;
        var safeTotal = engine.TensorMax(total, one);
        var logRatio = engine.TensorSubtract(engine.TensorLog(engine.TensorMax(counts, one)), engine.TensorLog(safeTotal));
        return engine.TensorNegate(engine.TensorMultiply(engine.TensorDivide(counts, safeTotal), logRatio));
    }
}
