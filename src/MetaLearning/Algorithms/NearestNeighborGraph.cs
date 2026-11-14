namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Whole-tensor k-nearest-neighbor selection for the transductive graph meta-learners.
/// </summary>
/// <remarks>
/// Neighbors are ranked by the composite key (distance, column index): the smaller distance comes first, and
/// equal distances keep ascending index order (a stable sort on distance). A node is never its own neighbor.
/// </remarks>
internal static class NearestNeighborGraph
{
    /// <summary>
    /// For each row i of an [n, n] distance matrix, a 0/1 mask selecting the min(k, n - 1) columns j != i with the
    /// smallest (distance, index) key.
    /// </summary>
    internal static Tensor<T> NearestMask<T>(IEngine engine, Tensor<T> distances, int k)
    {
        if (typeof(T) != typeof(double))
        {
            return NearestMask(engine, distances.Cast<double>(), k).Cast<T>();
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = distances.Shape[0];
        int take = Math.Min(k, n - 1);
        if (take <= 0)
        {
            return engine.TensorZerosLike(distances);
        }

        var square = new[] { n, n };
        var eye = engine.TensorEye<T>(n);
        var others = engine.TensorSubtract(engine.TensorOnesLike(eye), eye);

        // Infinity keeps self from displacing another node even when a valid distance is infinite.
        var withoutSelf = engine.TensorWhere(eye,
            engine.TensorFullLike(distances, numOps.FromDouble(double.PositiveInfinity)), distances);
        var sorted = engine.TensorSort(withoutSelf, 1, false).Values;
        var threshold = engine.TensorBroadcastTo(
            engine.TensorSlice(sorted, new[] { 0, take - 1 }, new[] { n, 1 }), square);

        // Every node strictly closer than the threshold, then the lowest-index ties until take are chosen.
        var closer = engine.TensorMultiply(engine.TensorLessThan(distances, threshold), others);
        var tied = engine.TensorMultiply(engine.TensorEquals(distances, threshold), others);
        var remaining = engine.TensorSubtract(
            engine.TensorFullLike(engine.ReduceSum(closer, new[] { 1 }, true), numOps.FromDouble(take + 0.5)),
            engine.ReduceSum(closer, new[] { 1 }, true));
        var tieRank = engine.TensorCumSum(tied, 1);
        var chosenTies = engine.TensorMultiply(
            tied, engine.TensorLessThan(tieRank, engine.TensorBroadcastTo(remaining, square)));
        return engine.TensorAdd(closer, chosenTies);
    }

    /// <summary>
    /// The symmetric [n, n] mask with an edge (i, j) wherever j is among i's nearest neighbors or i among j's.
    /// </summary>
    internal static Tensor<T> SymmetricNearestMask<T>(IEngine engine, Tensor<T> distances, int k)
    {
        var mask = NearestMask(engine, distances, k);
        return engine.TensorMax(mask, engine.TensorTranspose(mask));
    }
}
