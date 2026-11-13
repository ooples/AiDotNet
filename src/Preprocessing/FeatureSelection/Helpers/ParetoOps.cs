using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Engine-op building blocks of the multi-objective subset searches (NSGA-II, SPEA2, MOEA/D): Pareto dominance,
/// non-dominated fronts, crowding distance and the binary-mask variation operators. The two objectives are a
/// gain to maximize and a cost to minimize.
/// </summary>
internal static class ParetoOps
{
    /// <summary>
    /// D [m, m] with D[i, j] = 1 when solution i dominates j: gain_i &gt;= gain_j and cost_i &lt;= cost_j with at
    /// least one strict.
    /// </summary>
    public static Tensor<T> Dominance<T>(Tensor<T> gain, Tensor<T> cost)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = gain.Length;
        var shape = new[] { m, m };
        var gi = engine.TensorBroadcastTo(engine.Reshape(gain, new[] { m, 1 }), shape);
        var gj = engine.TensorBroadcastTo(engine.Reshape(gain, new[] { 1, m }), shape);
        var ci = engine.TensorBroadcastTo(engine.Reshape(cost, new[] { m, 1 }), shape);
        var cj = engine.TensorBroadcastTo(engine.Reshape(cost, new[] { 1, m }), shape);
        var noWorse = engine.TensorMultiply(engine.TensorGreaterEqual(gi, gj), engine.TensorGreaterEqual(cj, ci));
        var better = engine.TensorClamp(
            engine.TensorAdd(engine.TensorGreaterThan(gi, gj), engine.TensorLessThan(ci, cj)), ops.Zero, ops.One);
        return engine.TensorMultiply(noWorse, better);
    }

    /// <summary>
    /// The non-dominated fronts of Deb et al. (2002) from a dominance matrix: front k holds the solutions dominated
    /// only by solutions of earlier fronts. <paramref name="rank"/> [m] is the front index of each solution.
    /// </summary>
    public static List<int[]> Fronts<T>(Tensor<T> dominance, out Tensor<T> rank)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = dominance.Shape[0];
        var remaining = PreprocessingTensorOps.Filled(new[] { m }, ops.One);
        rank = new Tensor<T>(new[] { m });
        var fronts = new List<int[]>();
        while (PopulationSearchOps.Any(remaining))
        {
            // dominators[j] = number of remaining solutions that dominate j
            var dominators = engine.Reshape(
                engine.TensorMatMul(engine.Reshape(remaining, new[] { 1, m }), dominance), new[] { m });
            var front = engine.TensorMultiply(remaining, engine.TensorEquals(dominators, ops.Zero));
            fronts.Add(PreprocessingTensorOps.MaskPositions(front));
            remaining = engine.TensorSubtract(remaining, front);
            rank = engine.TensorAdd(rank, remaining);
        }

        return fronts;
    }

    /// <summary>
    /// Crowding distance of Deb et al. (2002) of every solution, computed within its front: per objective the
    /// front is sorted, both boundary solutions get an infinite distance (here the largest value of
    /// <typeparamref name="T"/>) and every interior one adds (f[next] - f[previous]) / (f_max - f_min).
    /// </summary>
    public static Tensor<T> CrowdingDistance<T>(IReadOnlyList<int[]> fronts, params Tensor<T>[] objectives)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = objectives[0].Length;
        var distance = new Tensor<T>(new[] { m });
        var boundary = new Tensor<T>(new[] { m });
        foreach (var front in fronts)
        {
            int count = front.Length;
            if (count == 0)
                continue;

            if (count <= 2)
            {
                boundary = PopulationSearchOps.ScatterAt(
                    boundary, front, PreprocessingTensorOps.Filled(new[] { count }, ops.One));
                continue;
            }

            var members = new Tensor<int>((int[])front.Clone(), new[] { count });
            foreach (var values in objectives.Select(objective => engine.TensorIndexSelect(objective, members, 0)))
            {
                var order = PreprocessingTensorOps.StableOrder(values, descending: false);
                var sorted = engine.TensorIndexSelect(values, new Tensor<int>((int[])order.Clone(), new[] { count }), 0);
                var sortedMembers = new int[count];
                for (int r = 0; r < count; r++)
                    sortedMembers[r] = front[order[r]];

                boundary = PopulationSearchOps.ScatterAt(
                    boundary, new[] { sortedMembers[0], sortedMembers[count - 1] },
                    PreprocessingTensorOps.Filled(new[] { 2 }, ops.One));

                var range = engine.TensorSubtract(
                    engine.TensorSlice(sorted, new[] { count - 1 }, new[] { 1 }),
                    engine.TensorSlice(sorted, new[] { 0 }, new[] { 1 }));
                if (!PopulationSearchOps.Greater(range, new Tensor<T>(new[] { 1 })))
                    continue;

                var gaps = engine.TensorSubtract(
                    engine.TensorSlice(sorted, new[] { 2 }, new[] { count - 2 }),
                    engine.TensorSlice(sorted, new[] { 0 }, new[] { count - 2 }));
                var interior = engine.TensorDivide(gaps, engine.TensorBroadcastTo(range, new[] { count - 2 }));
                var interiorMembers = new int[count - 2];
                Array.Copy(sortedMembers, 1, interiorMembers, 0, count - 2);
                distance = engine.TensorAdd(
                    distance, PopulationSearchOps.ScatterAt(new Tensor<T>(new[] { m }), interiorMembers, interior));
            }
        }

        return engine.TensorWhere(
            engine.TensorGreaterThan(boundary, ops.Zero),
            PreprocessingTensorOps.Filled(new[] { m }, ops.MaxValue),
            distance);
    }

    /// <summary>
    /// Random initial subsets [m, p]: each row selects between 1 and <paramref name="maxCount"/> features
    /// (uniformly many), chosen uniformly without replacement.
    /// </summary>
    public static Tensor<T> RandomSubsets<T>(Random random, int m, int p, int maxCount)
    {
        var engine = AiDotNetEngine.Current;
        int limit = Math.Max(1, Math.Min(p, maxCount));
        var rows = new Tensor<T>[m];
        for (int i = 0; i < m; i++)
        {
            int count = random.Next(1, limit + 1);
            var keys = PopulationSearchOps.Uniform<T>(random, p);
            rows[i] = engine.Reshape(
                PopulationSearchOps.IndexMask<T>(PreprocessingTensorOps.BottomKIndicesSorted(keys, count), p),
                new[] { 1, p });
        }

        return engine.TensorConcatenate(rows, 0);
    }

    /// <summary>One-point crossover: the child takes positions below <paramref name="point"/> from the first parent.</summary>
    public static Tensor<T> OnePointCrossover<T>(Tensor<T> first, Tensor<T> second, int point)
    {
        var engine = AiDotNetEngine.Current;
        int p = first.Length;
        // positions compared as int: in Half the positions past 2048 are not exact and land on the wrong side
        var head = engine.TensorLessThan(engine.TensorArange<int>(0, p, 1), point).Cast<T>();
        return engine.TensorWhere(head, first, second);
    }

    /// <summary>Uniform crossover: position j comes from the first parent when <paramref name="draws"/>[j] &lt; 1/2.</summary>
    public static Tensor<T> UniformCrossover<T>(Tensor<T> first, Tensor<T> second, Tensor<T> draws)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorWhere(engine.TensorLessThan(draws, ops.FromDouble(0.5)), first, second);
    }

    /// <summary>Bit-flip mutation: position j flips when <paramref name="draws"/>[j] &lt; <paramref name="rate"/>.</summary>
    public static Tensor<T> BitFlip<T>(Tensor<T> mask, Tensor<T> draws, double rate)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorWhere(
            engine.TensorLessThan(draws, ops.FromDouble(rate)), PreprocessingTensorOps.OneMinus(mask), mask);
    }

    /// <summary>Selects one uniformly drawn feature when <paramref name="mask"/> selects nothing.</summary>
    public static Tensor<T> EnsureNonEmpty<T>(Tensor<T> mask, Random random) =>
        PopulationSearchOps.Any(mask) ? mask : FeatureSelectionTensorOps.OneHot<T>(random.Next(mask.Length), mask.Length);

    /// <summary>The smallest value of a [m] tensor as a [1] tensor.</summary>
    public static Tensor<T> Min<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(values), new[] { 0 }, true));
    }

    /// <summary>The largest value of a [m] tensor as a [1] tensor.</summary>
    public static Tensor<T> Max<T>(Tensor<T> values) => AiDotNetEngine.Current.ReduceMax(values, new[] { 0 }, true);
}
