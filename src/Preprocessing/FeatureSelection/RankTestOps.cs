using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Rank helpers for the nonparametric feature-selection tests, as whole-tensor engine ops generic in
/// <typeparamref name="T"/>.
/// </summary>
internal static class RankTestOps
{
    /// <summary>
    /// Average ranks (1-based; tied values share the mean of their positions, scipy <c>rankdata(method="average")</c>)
    /// of every column of an [n, m] tensor, as [n, m]. <paramref name="tieTerm"/> receives, per column, the tie
    /// term <c>sum over tie groups of (t^3 - t)</c> used by the tie corrections of the Kruskal-Wallis and
    /// Mann-Whitney tests (scipy <c>tiecorrect</c>), as [m]. The tie term is a count that reaches n^3 - n (past
    /// Half's 65504 at n = 41 and past float's exact integers at n = 257), so it is formed from the integer tie-group
    /// sizes in double, where it is exact.
    /// </summary>
    public static Tensor<T> AverageRanksWithTies<T>(Tensor<T> x, out Tensor<double> tieTerm)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int m = x.Shape[1];
        if (m == 0)
        {
            tieTerm = new Tensor<double>(new[] { 0 });
            return new Tensor<T>(new[] { n, 0 });
        }

        var columns = new Tensor<T>[m];
        var ties = new Tensor<double>[m];
        for (int j = 0; j < m; j++)
        {
            var column = engine.Reshape(engine.TensorSliceAxis(x, 1, j), new[] { n });
            var sorted = engine.TensorSort(column, 0, false).Values;
            var below = engine.TensorSearchSorted(sorted, column, false);
            var atMost = engine.TensorSearchSorted(sorted, column, true);
            columns[j] = engine.Reshape(engine.TensorAdd(below, atMost).Cast<T>(), new[] { n, 1 });

            // Each value's tie group has t = atMost - below members; summing t^2 - 1 over the values gives
            // sum over groups of t (t^2 - 1) = t^3 - t
            var groupSize = engine.TensorSubtract(atMost, below).Cast<double>();
            ties[j] = engine.ReduceSum(engine.Reshape(
                engine.TensorMultiply(groupSize, groupSize), new[] { 1, n }), new[] { 1 }, false);
        }

        // (below + atMost + 1) / 2 = mean of the 1-based positions below + 1 .. atMost
        tieTerm = engine.TensorSubtractScalar(engine.TensorConcatenate(ties, 0), (double)n);
        var sum = engine.TensorConcatenate(columns, 1);
        return engine.TensorDivideScalar(engine.TensorAddScalar(sum, ops.One), ops.FromDouble(2));
    }
}
