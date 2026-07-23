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
        int n = data.Rows;
        double m1 = 0, m2 = 0;
        for (int i = 0; i < n; i++) { m1 += NumOps.ToDouble(data[i, col1]); m2 += NumOps.ToDouble(data[i, col2]); }
        m1 /= n; m2 /= n;

        double s11 = 0, s22 = 0, s12 = 0;
        for (int i = 0; i < n; i++)
        {
            double d1 = NumOps.ToDouble(data[i, col1]) - m1, d2 = NumOps.ToDouble(data[i, col2]) - m2;
            s11 += d1 * d1; s22 += d2 * d2; s12 += d1 * d2;
        }

        s11 /= n; s22 /= n; s12 /= n;
        double corr = (s11 > 1e-15 && s22 > 1e-15) ? s12 / Math.Sqrt(s11 * s22) : 0;
        corr = Math.Max(-0.9999, Math.Min(0.9999, corr));
        return -0.5 * Math.Log(1 - corr * corr);
    }

    /// <summary>
    /// Computes Shannon entropy (Gaussian approximation) of a column.
    /// </summary>
    protected double ComputeEntropy(Matrix<T> data, int col)
    {
        int n = data.Rows;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += NumOps.ToDouble(data[i, col]);
        mean /= n;

        double variance = 0;
        for (int i = 0; i < n; i++) { double d = NumOps.ToDouble(data[i, col]) - mean; variance += d * d; }
        variance /= n;

        return 0.5 * Math.Log(2 * Math.PI * Math.E * (variance + 1e-15));
    }

}
