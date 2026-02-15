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
    /// </summary>
    protected int NumSamples { get; set; } = 1000;

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
    /// Converts double array to Matrix&lt;T&gt;.
    /// </summary>
    protected Matrix<T> DoubleArrayToMatrix(double[,] data)
    {
        int rows = data.GetLength(0), cols = data.GetLength(1);
        var result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = NumOps.FromDouble(data[i, j]);
        return result;
    }
}
