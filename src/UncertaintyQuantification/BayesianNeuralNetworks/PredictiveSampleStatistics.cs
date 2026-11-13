namespace AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;

/// <summary>
/// Moments and uncertainty decompositions of a set of stochastic predictions (MC dropout samples,
/// Bayes-by-Backprop weight samples or deep-ensemble members), built with engine ops and generic in T.
/// </summary>
/// <remarks>
/// <para>For class-probability outputs (last axis sums to one) the decomposition follows
/// Depeweg et al. 2018 and Kwon et al. 2020: epistemic = Var_s[p_s], aleatoric = E_s[p_s (1 - p_s)], and the
/// metrics are the predictive entropy H[E_s p_s] and the mutual information (BALD, Houlsby et al. 2011)
/// H[E_s p_s] - E_s H[p_s].</para>
/// <para>For other outputs the network does not predict a noise variance, so the sample variance is all
/// epistemic (Kendall and Gal 2017), the aleatoric part is zero and the entropy metrics are zero.</para>
/// </remarks>
internal static class PredictiveSampleStatistics<T>
{
    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>The samples stacked along a new leading axis: [S, ...].</summary>
    internal static Tensor<T> Stack(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            throw new ArgumentException("Cannot compute mean of empty prediction list");

        return AiDotNetEngine.Current.TensorStack(predictions.ToArray(), 0);
    }

    /// <summary>The sample mean E_s[y_s].</summary>
    internal static Tensor<T> Mean(List<Tensor<T>> predictions)
        => AiDotNetEngine.Current.ReduceMean(Stack(predictions), new[] { 0 }, keepDims: false);

    /// <summary>The population variance E_s[(y_s - mean)^2] (divides by S, as before).</summary>
    internal static Tensor<T> Variance(List<Tensor<T>> predictions, Tensor<T> mean)
    {
        var engine = AiDotNetEngine.Current;
        var stacked = Stack(predictions);
        var diff = engine.TensorSubtract(stacked, engine.TensorBroadcastTo(engine.TensorExpandDims(mean, 0), stacked._shape));
        return engine.ReduceMean(engine.TensorSquare(diff), new[] { 0 }, keepDims: false);
    }

    /// <summary>E_s[p_s (1 - p_s)] for probability outputs; zeros otherwise.</summary>
    internal static Tensor<T> AleatoricVariance(List<Tensor<T>> predictions, Tensor<T> mean)
    {
        if (!TryGetProbabilityLayout(mean, out _, out _))
            return new Tensor<T>(mean._shape);

        var engine = AiDotNetEngine.Current;
        var stacked = Stack(predictions);
        var bernoulli = engine.TensorMultiply(stacked, engine.ScalarMinusTensor(NumOps.One, stacked));
        return engine.ReduceMean(bernoulli, new[] { 0 }, keepDims: false);
    }

    /// <summary>Per-row predictive entropy and mutual information ([batch] each; zeros for non-probability outputs).</summary>
    internal static IReadOnlyDictionary<string, Tensor<T>> Metrics(List<Tensor<T>> predictions, Tensor<T> mean)
    {
        var engine = AiDotNetEngine.Current;
        if (!TryGetProbabilityLayout(mean, out int batch, out int classes))
        {
            int rows = mean.Rank <= 1 ? 1 : Math.Max(1, mean.Length / Math.Max(1, mean._shape[mean.Rank - 1]));
            var zeros = new Tensor<T>([rows]);
            return new Dictionary<string, Tensor<T>>
            {
                ["predictive_entropy"] = zeros,
                ["mutual_information"] = zeros
            };
        }

        var predictive = Entropy(engine.Reshape(mean, new[] { batch, classes }));                         // [batch]
        var stacked = engine.Reshape(Stack(predictions), new[] { predictions.Count * batch, classes });
        var expected = engine.ReduceMean(
            engine.Reshape(Entropy(stacked), new[] { predictions.Count, batch }), new[] { 0 }, keepDims: false); // [batch]
        var mutualInformation = engine.TensorClampMin(engine.TensorSubtract(predictive, expected), NumOps.Zero);

        return new Dictionary<string, Tensor<T>>
        {
            ["predictive_entropy"] = predictive,
            ["mutual_information"] = mutualInformation
        };
    }

    /// <summary>-sum_c p log p over the last axis of a [rows, C] tensor; zero-probability terms contribute 0.</summary>
    private static Tensor<T> Entropy(Tensor<T> probabilities)
    {
        var engine = AiDotNetEngine.Current;

        // Select rather than multiply by a mask, so log(0) = -inf never meets a 0 factor.
        var positive = engine.TensorGreaterThan(probabilities, NumOps.FromDouble(1e-12));
        var plogp = engine.TensorWhere(positive, engine.TensorMultiply(probabilities, engine.TensorLog(probabilities)),
            new Tensor<T>(probabilities._shape));
        return engine.TensorNegate(engine.ReduceSum(plogp, new[] { 1 }, keepDims: false));
    }

    /// <summary>
    /// True when the last axis holds class probabilities: more than one class, every value in [0, 1]
    /// and every row summing to one within 1e-2 (the rule AiModelResult uses).
    /// </summary>
    private static bool TryGetProbabilityLayout(Tensor<T> tensor, out int batch, out int classes)
    {
        batch = 0;
        classes = 0;
        if (tensor.Rank == 0 || tensor.Length == 0)
            return false;

        classes = tensor._shape[tensor.Rank - 1];
        if (classes <= 1)
            return false;

        batch = tensor.Rank == 1 ? 1 : tensor.Length / classes;
        if (batch <= 0)
            return false;

        var engine = AiDotNetEngine.Current;
        var rows = engine.Reshape(tensor, new[] { batch, classes });
        var outOfRange = engine.TensorAdd(engine.TensorLessThan(rows, NumOps.Zero), engine.TensorGreaterThan(rows, NumOps.One)).Cast<int>();
        if (engine.TensorSum(outOfRange) > 0)
            return false;

        var rowError = engine.TensorAbs(engine.TensorSubtractScalar(engine.ReduceSum(rows, new[] { 1 }, keepDims: false), NumOps.One));
        return engine.TensorSum(engine.TensorGreaterThan(rowError, NumOps.FromDouble(1e-2)).Cast<int>()) == 0;
    }
}
