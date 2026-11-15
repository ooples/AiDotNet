namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Reductions over a whole <see cref="Vector{T}"/> that accumulate in double, for the meta-learners whose
/// scalar loops summed in double before they were vectorized.
/// </summary>
/// <remarks>
/// The engine's vector reductions (<c>Sum</c>, <c>Mean</c>, <c>Norm</c>, <c>DotProduct</c>) accumulate in T. For
/// Half that running sum stops growing once it passes 2048 (the spacing between representable values exceeds
/// the terms being added) and overflows to infinity past 65504, so a sum over a feature or parameter vector is
/// wrong long before the inputs are. The element-wise work stays in T; only the accumulation is widened, and
/// each result is a single scalar.
/// </remarks>
internal static class WideReduction
{
    /// <summary>The sum of <paramref name="values"/>, accumulated in double.</summary>
    internal static double Sum<T>(IEngine engine, Vector<T> values)
    {
        return values.Length == 0 ? 0.0 : engine.TensorSum(Tensor<T>.FromVector(values).Cast<double>());
    }

    /// <summary>The mean of <paramref name="values"/>, accumulated in double; zero for an empty vector.</summary>
    internal static double Mean<T>(IEngine engine, Vector<T> values)
    {
        return values.Length == 0 ? 0.0 : Sum(engine, values) / values.Length;
    }

    /// <summary>The sum of squares of <paramref name="values"/>: squared and accumulated in double.</summary>
    internal static double SumOfSquares<T>(IEngine engine, Vector<T> values)
    {
        if (values.Length == 0)
        {
            return 0.0;
        }

        var wide = Tensor<T>.FromVector(values).Cast<double>().ToVector();
        return engine.DotProduct(wide, wide);
    }

    /// <summary>
    /// KL(N(mean, diag exp(logVar)) || N(prior, I)) = 0.5 * sum_d [v_d - 1 - log v_d + (mean_d - prior_d)^2] over
    /// the first <paramref name="dimensions"/> entries.
    /// </summary>
    /// <remarks>
    /// Per dimension v_d - 1 - log v_d is computed as expm1(log v_d) - log v_d, which is nonnegative and near zero
    /// when log v_d is. Summing those small terms (rather than sum(v_d) - D - sum(log v_d), two sums near D whose
    /// difference is the answer) keeps the KL exact when the posterior is close to the prior, and the squared
    /// mean shift is accumulated in double so a tiny shift is not lost against a large sum.
    /// </remarks>
    internal static T DiagonalGaussianKL<T>(
        IEngine engine, Vector<T> logVar, Vector<T> mean, Vector<T> prior, int dimensions)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var lv = logVar.Length == dimensions ? logVar : logVar.GetSubVector(0, dimensions);
        var shift = engine.Subtract(
            mean.Length == dimensions ? mean : mean.GetSubVector(0, dimensions),
            prior.Length == dimensions ? prior : prior.GetSubVector(0, dimensions));
        double varianceTerm = Sum(engine, engine.Subtract(engine.ExpM1(lv), lv));
        return numOps.FromDouble(0.5 * (varianceTerm + SumOfSquares(engine, shift)));
    }
}
