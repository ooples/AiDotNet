using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// Base class for Bayesian causal discovery algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Bayesian methods maintain a posterior distribution over possible DAG structures given
/// the data. They can represent uncertainty about the causal structure and are naturally
/// suited for model averaging. Methods include MCMC sampling over graphs, variational
/// inference, and gradient-based approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of returning a single "best guess" graph, Bayesian methods
/// give probabilities for each possible connection. This tells you not just "X probably
/// causes Y" but also "we're 90% confident about this." The trade-off is higher computation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class BayesianCausalBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.Bayesian;

    /// <summary>
    /// Number of MCMC samples or variational iterations.
    /// Per Friedman and Koller (2003), MCMC methods need sufficient samples
    /// for the chain to mix properly. Default 5000.
    /// </summary>
    protected int NumSamples { get; set; } = 5000;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    protected int Seed { get; set; } = 42;

    /// <summary>
    /// Applies Bayesian-specific options.
    /// </summary>
    protected void ApplyBayesianOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.MaxIterations.HasValue) NumSamples = options.MaxIterations.Value;
        if (options.Seed.HasValue) Seed = options.Seed.Value;
    }

    /// <summary>
    /// Pairwise OLS slopes W[i, j] = Cov(i, j) / Var(i), zero where Var(i) is not above 1e-10.
    /// </summary>
    private protected Tensor<T> ComputePairwiseOLSWeights(Matrix<T> cov)
    {
        int d = cov.Rows;
        var c = Tensor<T>.FromMatrix(cov);
        var variance = Engine.TensorBroadcastTo(Engine.TensorDiagonal(c).Reshape(new[] { d, 1 }), new[] { d, d });
        var usable = Engine.TensorGreaterThan(variance, NumOps.FromDouble(1e-10));
        var safeVariance = Engine.TensorWhere(usable, variance, Engine.TensorOnesLike(variance));
        return Engine.TensorWhere(usable, Engine.TensorDivide(c, safeVariance), Engine.TensorZerosLike(c));
    }

    /// <summary>Entries of <paramref name="values"/> where <paramref name="criterion"/> &gt; threshold, else zero.</summary>
    private protected Tensor<T> KeepWhereAbove(Tensor<T> criterion, double threshold, Tensor<T> values)
    {
        return Engine.TensorWhere(Engine.TensorGreaterThan(criterion, NumOps.FromDouble(threshold)),
            values, Engine.TensorZerosLike(values));
    }

    /// <summary>Adds 1 to every entry of <paramref name="edgeCounts"/> where <paramref name="dag"/> is nonzero.</summary>
    private protected Tensor<T> AccumulateEdgeCounts(Tensor<T> edgeCounts, Matrix<T> dag)
    {
        return Engine.TensorAdd(edgeCounts, Engine.TensorNotEquals(Tensor<T>.FromMatrix(dag), NumOps.Zero));
    }

    /// <summary>Posterior edge frequencies counts / samples, keeping only those above 0.5.</summary>
    private protected Matrix<T> MajorityEdgeFrequencies(Tensor<T> edgeCounts, int sampleCount)
    {
        var frequency = Engine.TensorDivideScalar(edgeCounts, NumOps.FromDouble(sampleCount));
        return KeepWhereAbove(frequency, 0.5, frequency).ToMatrix();
    }

}
