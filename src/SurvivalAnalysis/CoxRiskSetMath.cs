namespace AiDotNet.SurvivalAnalysis;

/// <summary>Stable double-precision scans used by the Cox Breslow estimator.</summary>
internal static class CoxRiskSetMath
{
    internal static Tensor<double> LogAdd(Tensor<double> left, Tensor<double> right)
    {
        var engine = AiDotNetEngine.Current;
        var maximum = engine.TensorWhere(engine.TensorGreaterThan(left, right), left, right);
        return engine.TensorAdd(maximum, engine.TensorLog(engine.TensorAdd(
            engine.TensorExp(engine.TensorSubtract(left, maximum)),
            engine.TensorExp(engine.TensorSubtract(right, maximum)))));
    }

    internal static Tensor<double> CumulativeLogSum(Tensor<double> values)
    {
        var engine = AiDotNetEngine.Current;
        int n = values.Length;
        for (int offset = 1; offset < n; offset = offset > n / 2 ? n : offset * 2)
        {
            var left = engine.TensorSlice(values, new[] { 0 }, new[] { n - offset });
            var right = engine.TensorSlice(values, new[] { offset }, new[] { n - offset });
            values = engine.TensorConcatenate(new[]
            {
                engine.TensorSlice(values, new[] { 0 }, new[] { offset }), LogAdd(left, right)
            }, 0);
        }
        return values;
    }

    /// <summary>Inclusive prefix risk sums and weighted covariate means, without forming exp(eta).</summary>
    internal static (Tensor<double> LogRisk, Tensor<double> Mean) PrefixMoments(
        Tensor<double> eta, Tensor<double> design)
    {
        var engine = AiDotNetEngine.Current;
        int n = design.Shape[0];
        int p = design.Shape[1];
        var mean = design;
        for (int offset = 1; offset < n; offset = offset > n / 2 ? n : offset * 2)
        {
            int count = n - offset;
            var left = engine.TensorSlice(eta, new[] { 0 }, new[] { count });
            var right = engine.TensorSlice(eta, new[] { offset }, new[] { count });
            var maximum = engine.TensorWhere(engine.TensorGreaterThan(left, right), left, right);
            var leftWeight = engine.TensorExp(engine.TensorSubtract(left, maximum));
            var rightWeight = engine.TensorExp(engine.TensorSubtract(right, maximum));
            var sum = engine.TensorAdd(leftWeight, rightWeight);
            var shape = new[] { count, p };
            var leftFraction = engine.TensorBroadcastTo(engine.TensorDivide(leftWeight, sum).Reshape(new[] { count, 1 }), shape);
            var rightFraction = engine.TensorBroadcastTo(engine.TensorDivide(rightWeight, sum).Reshape(new[] { count, 1 }), shape);
            var combinedMean = engine.TensorAdd(
                engine.TensorMultiply(engine.TensorSlice(mean, new[] { 0, 0 }, shape), leftFraction),
                engine.TensorMultiply(engine.TensorSlice(mean, new[] { offset, 0 }, shape), rightFraction));
            mean = engine.TensorConcatenate(new[]
            {
                engine.TensorSlice(mean, new[] { 0, 0 }, new[] { offset, p }), combinedMean
            }, 0);
            eta = engine.TensorConcatenate(new[]
            {
                engine.TensorSlice(eta, new[] { 0 }, new[] { offset }),
                engine.TensorAdd(maximum, engine.TensorLog(sum))
            }, 0);
        }
        return (eta, mean);
    }
}
