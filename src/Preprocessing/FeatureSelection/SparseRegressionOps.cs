using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Penalised least-squares solvers shared by the sparse feature selectors, as whole-tensor engine ops generic in
/// <typeparamref name="T"/>.
/// </summary>
internal static class SparseRegressionOps
{
    /// <summary>
    /// Minimises the sparse-group elastic-net objective
    /// <c>(1/2n) ||y - X b||^2 + l1 ||b||_1 + (l2 / 2) ||b||^2 + group sum_g sqrt(p_g) ||b_g||_2</c>, which covers the
    /// lasso and elastic net of scikit-learn (<c>Lasso</c>, <c>ElasticNet</c>), the group lasso of Yuan and Lin (2006)
    /// and the sparse group lasso of Simon et al. (2013). Uses FISTA (Beck and Teboulle, 2009) with step 1 / L, where
    /// L is the largest eigenvalue of <c>X'X / n + l2 I</c>; the proximal step soft-thresholds by l1 and then shrinks
    /// each group's norm by its group penalty. Stops once no coefficient moves by <paramref name="tolerance"/> or more.
    /// </summary>
    /// <param name="x">The [n, p] design.</param>
    /// <param name="y">The [n, 1] response.</param>
    /// <param name="groups">The group label of every feature (any integers).</param>
    /// <param name="l1">The lasso weight.</param>
    /// <param name="l2">The ridge weight.</param>
    /// <param name="group">The group-lasso weight.</param>
    /// <param name="maxIterations">The iteration limit.</param>
    /// <param name="tolerance">The largest coefficient change that counts as converged.</param>
    /// <returns>The coefficients, [p].</returns>
    public static Tensor<T> SparseGroupElasticNet<T>(Tensor<T> x, Tensor<T> y, int[] groups,
        double l1, double l2, double group, int maxIterations, double tolerance)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var coefficients = new Tensor<T>(new[] { p, 1 });
        if (p == 0 || n == 0)
            return engine.Reshape(coefficients, new[] { p });

        // Group membership [p, m] and the per-group weight sqrt(p_g) as [m, 1]
        var labels = groups.Distinct().OrderBy(g => g).ToArray();
        var position = groups.Select(g => Array.BinarySearch(labels, g)).ToArray();
        int m = labels.Length;
        var membership = engine.TensorOneHot<T>(new Tensor<int>(position, new[] { p }), m);
        var membershipT = engine.TensorTranspose(membership);
        // (group sizes counted as int: a 0/1 sum in T stops counting at 2048 for Half)
        var groupSizes = engine.ReduceSum(membership.Cast<int>(), new[] { 0 }, false).Cast<T>();
        var groupWeights = engine.Reshape(engine.TensorSqrt(groupSizes), new[] { m, 1 });

        // Smooth part: gradient H b - X'y / n with H = X'X / n + l2 I
        T count = ops.FromDouble(n);
        var xt = engine.TensorTranspose(x);
        var hessian = engine.TensorAdd(engine.TensorDivideScalar(engine.TensorMatMul(xt, x), count),
            engine.TensorMultiplyScalar(engine.TensorEye<T>(p), ops.FromDouble(l2)));
        var xty = engine.TensorDivideScalar(engine.TensorMatMul(xt, y), count);
        T lipschitz = engine.TensorMaxValue(LinalgDispatch<T>.Eigh(hessian).Values);
        if (!ops.GreaterThan(lipschitz, ops.Zero))
            return engine.Reshape(coefficients, new[] { p });

        T step = ops.Divide(ops.One, lipschitz);
        T l1Threshold = ops.Multiply(step, ops.FromDouble(l1));
        var groupThreshold = engine.TensorMultiplyScalar(groupWeights, ops.Multiply(step, ops.FromDouble(group)));
        var groupOnes = PreprocessingTensorOps.Filled(new[] { m, 1 }, ops.One);
        T zero = ops.Zero;
        T tol = ops.FromDouble(tolerance);
        var momentumPoint = coefficients;
        double t = 1;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Gradient step from the momentum point
            var gradient = engine.TensorSubtract(engine.TensorMatMul(hessian, momentumPoint), xty);
            var v = engine.TensorSubtract(momentumPoint, engine.TensorMultiplyScalar(gradient, step));

            // Proximal step: soft-threshold by l1, then scale each group by (1 - threshold / ||s_g||)_+
            var s = engine.TensorMultiply(engine.TensorSign(v),
                engine.TensorMax(engine.TensorSubtractScalar(engine.TensorAbs(v), l1Threshold), zero));
            var groupNorm = engine.TensorSqrt(engine.TensorMatMul(membershipT, engine.TensorMultiply(s, s)));
            var hasNorm = engine.TensorGreaterThan(groupNorm, zero);
            var safeNorm = engine.TensorWhere(hasNorm, groupNorm, groupOnes);
            var factor = SelectorTensorOps.ZeroUnless(hasNorm,
                engine.TensorMax(PreprocessingTensorOps.OneMinus(engine.TensorDivide(groupThreshold, safeNorm)), zero));
            var next = engine.TensorMultiply(s, engine.TensorMatMul(membership, factor));

            // FISTA momentum
            double tNext = (1 + Math.Sqrt(1 + 4 * t * t)) / 2;
            momentumPoint = engine.TensorAdd(next,
                engine.TensorMultiplyScalar(engine.TensorSubtract(next, coefficients), Scalar<T>((t - 1) / tNext)));
            t = tNext;

            T change = engine.TensorMaxValue(engine.TensorAbs(engine.TensorSubtract(next, coefficients)));
            coefficients = next;
            if (PreprocessingTensorOps.IsBelow(change, tol))
                break;
        }

        return engine.Reshape(coefficients, new[] { p });
    }

    /// <summary>A double constant as a scalar of type T.</summary>
    private static T Scalar<T>(double value) => MathHelper.GetNumericOperations<T>().FromDouble(value);
}
