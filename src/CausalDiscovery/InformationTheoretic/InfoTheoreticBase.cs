using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery.InformationTheoretic;

/// <summary>
/// Base class for information-theoretic causal discovery algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Information-theoretic methods use entropy, mutual information, and transfer entropy to
/// discover causal relationships. These measures quantify the amount of information one
/// variable provides about another, either unconditionally or conditioned on other variables.
/// </para>
/// <para>
/// <b>For Beginners:</b> These methods measure how much "information" flows between variables.
/// If knowing variable X tells you a lot about variable Y (beyond what other variables tell
/// you), that suggests X has a causal influence on Y.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class InfoTheoreticBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.InformationTheoretic;

    /// <summary>
    /// Number of nearest neighbors for MI estimation (Kraskov method).
    /// </summary>
    protected int KNeighbors { get; set; } = 5;

    /// <summary>
    /// Applies information-theoretic options.
    /// </summary>
    protected void ApplyInfoOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        // KNeighbors uses MaxParents as a reasonable general-purpose integer setting
        if (options.MaxParents.HasValue) KNeighbors = options.MaxParents.Value;
    }

    /// <summary>
    /// Computes Gaussian mutual information between two columns.
    /// </summary>
    protected double ComputeGaussianMI(Matrix<T> data, int col1, int col2)
    {
        return NumOps.ToDouble(CausalTensorHelpers.GaussianMutualInformation(
            Engine.GetColumn(data, col1).AsTensor(), Engine.GetColumn(data, col2).AsTensor()));
    }

    /// <summary>
    /// Computes Shannon entropy (Gaussian approximation) of a column.
    /// </summary>
    protected double ComputeEntropy(Matrix<T> data, int col)
    {
        T variance = NumOps.Divide(CausalTensorHelpers.CenteredSumOfSquares(data, col), NumOps.FromDouble(data.Rows));
        T entropy = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Log(NumOps.Multiply(
            NumOps.FromDouble(2 * Math.PI * Math.E), NumOps.Add(variance, NumOps.FromDouble(1e-15)))));
        return NumOps.ToDouble(entropy);
    }

}
