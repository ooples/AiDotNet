using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Represents the interventional distribution P(Y | do(X = x)) from Pearl's do-calculus.
/// </summary>
/// <remarks>
/// <para>
/// An interventional distribution answers the question: "What would happen to variable Y if
/// we actively SET variable X to value x?" This is fundamentally different from conditioning
/// (P(Y | X = x)) because it breaks all causal arrows INTO X, simulating an experiment.
/// </para>
/// <para>
/// <b>Truncated Factorization Formula:</b>
/// P(Y | do(X = x)) = Σ_pa(X) P(Y | X, pa(X)) * P(pa(X))
/// where pa(X) are the parents of X in the causal graph.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you discover that ice cream sales and drowning rates are
/// correlated. The observational distribution P(Drowning | IceCream = high) is high.
/// But the interventional distribution P(Drowning | do(IceCream = high)) — what happens
/// if we FORCE ice cream sales to be high — is NOT high, because the causal arrow is
/// actually Temperature → IceCream and Temperature → Drowning. Interventions break the
/// confounding by the temperature variable.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// // Discover the causal graph
/// var graph = algorithm.DiscoverStructure(data, featureNames);
///
/// // Compute interventional distribution: what happens to variable 2 if we set variable 0 = 1.5?
/// var intDist = graph.ComputeInterventionalDistribution(
///     interventionVariable: 0,
///     interventionValue: 1.5,
///     targetVariable: 2,
///     observationalData: data);
///
/// // Query the result
/// double expectedValue = intDist.Mean;
/// double uncertainty = intDist.StandardDeviation;
/// double[] samples = intDist.Samples;
/// </code>
/// </para>
/// <para>
/// Reference: Pearl (2009), "Causality: Models, Reasoning, and Inference", Cambridge University Press, Ch. 3.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InterventionalDistribution<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the name of the intervention variable (the variable being set via do-operator).
    /// </summary>
    public string InterventionVariableName { get; }

    /// <summary>
    /// Gets the index of the intervention variable.
    /// </summary>
    public int InterventionVariableIndex { get; }

    /// <summary>
    /// Gets the value the intervention variable is set to.
    /// </summary>
    public T InterventionValue { get; }

    /// <summary>
    /// Gets the name of the target variable whose distribution is being computed.
    /// </summary>
    public string TargetVariableName { get; }

    /// <summary>
    /// Gets the index of the target variable.
    /// </summary>
    public int TargetVariableIndex { get; }

    /// <summary>
    /// Gets the interventional samples of the target variable under the do-intervention.
    /// </summary>
    public double[] Samples { get; }

    /// <summary>
    /// Gets the number of interventional samples.
    /// </summary>
    public int SampleCount => Samples.Length;

    /// <summary>
    /// Gets the mean of the interventional distribution.
    /// </summary>
    public double Mean { get; }

    /// <summary>
    /// Gets the variance of the interventional distribution.
    /// </summary>
    public double Variance { get; }

    /// <summary>
    /// Gets the standard deviation of the interventional distribution.
    /// </summary>
    public double StandardDeviation => Math.Sqrt(Variance);

    /// <summary>
    /// Gets the median of the interventional distribution.
    /// </summary>
    public double Median { get; }

    /// <summary>
    /// Gets the minimum value in the interventional samples.
    /// </summary>
    public double Min { get; }

    /// <summary>
    /// Gets the maximum value in the interventional samples.
    /// </summary>
    public double Max { get; }

    /// <summary>
    /// Gets the Average Causal Effect (ACE): E[Y | do(X=x)] - E[Y_observational].
    /// </summary>
    /// <remarks>
    /// <para>A positive ACE means the intervention increases the target on average.
    /// A negative ACE means the intervention decreases the target on average.</para>
    /// </remarks>
    public double AverageCausalEffect { get; }

    /// <summary>
    /// Creates a new InterventionalDistribution from computed interventional samples.
    /// </summary>
    internal InterventionalDistribution(
        int interventionVariableIndex,
        string interventionVariableName,
        T interventionValue,
        int targetVariableIndex,
        string targetVariableName,
        double[] samples,
        double observationalMean)
    {
        InterventionVariableIndex = interventionVariableIndex;
        InterventionVariableName = interventionVariableName;
        InterventionValue = interventionValue;
        TargetVariableIndex = targetVariableIndex;
        TargetVariableName = targetVariableName;
        Samples = samples;

        if (samples.Length == 0)
        {
            Mean = 0;
            Variance = 0;
            Median = 0;
            Min = 0;
            Max = 0;
            AverageCausalEffect = 0;
            return;
        }

        // Compute statistics
        double sum = 0;
        double min = double.MaxValue;
        double max = double.MinValue;
        for (int i = 0; i < samples.Length; i++)
        {
            sum += samples[i];
            if (samples[i] < min) min = samples[i];
            if (samples[i] > max) max = samples[i];
        }
        Mean = sum / samples.Length;
        Min = min;
        Max = max;

        double varSum = 0;
        for (int i = 0; i < samples.Length; i++)
        {
            double diff = samples[i] - Mean;
            varSum += diff * diff;
        }
        Variance = samples.Length > 1 ? varSum / (samples.Length - 1) : 0;

        // Compute median
        var sorted = new double[samples.Length];
        Array.Copy(samples, sorted, samples.Length);
        Array.Sort(sorted);
        Median = samples.Length % 2 == 0
            ? (sorted[samples.Length / 2 - 1] + sorted[samples.Length / 2]) / 2.0
            : sorted[samples.Length / 2];

        AverageCausalEffect = Mean - observationalMean;
    }

    /// <summary>
    /// Computes the specified quantile of the interventional distribution.
    /// </summary>
    /// <param name="p">Probability value in [0, 1].</param>
    /// <returns>The quantile value.</returns>
    public double Quantile(double p)
    {
        if (p < 0 || p > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (Samples.Length == 0) return 0;

        var sorted = new double[Samples.Length];
        Array.Copy(Samples, sorted, Samples.Length);
        Array.Sort(sorted);

        double idx = p * (sorted.Length - 1);
        int lower = (int)Math.Floor(idx);
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        double frac = idx - lower;

        return sorted[lower] * (1 - frac) + sorted[upper] * frac;
    }

    /// <summary>
    /// Computes a confidence interval for the interventional mean.
    /// </summary>
    /// <param name="confidence">Confidence level (e.g., 0.95 for 95% CI). Default: 0.95.</param>
    /// <returns>A tuple of (lower, upper) bounds.</returns>
    public (double Lower, double Upper) ConfidenceInterval(double confidence = 0.95)
    {
        double alpha = 1.0 - confidence;
        double lower = Quantile(alpha / 2);
        double upper = Quantile(1 - alpha / 2);
        return (lower, upper);
    }

    /// <summary>
    /// Computes an empirical estimate of the probability density at a given value using a Gaussian kernel.
    /// </summary>
    /// <param name="value">The value at which to estimate the density.</param>
    /// <returns>The estimated probability density.</returns>
    public double EstimateDensity(double value)
    {
        if (Samples.Length == 0) return 0;

        // Silverman's rule of thumb for bandwidth
        double h = 1.06 * StandardDeviation * Math.Pow(Samples.Length, -0.2);
        if (h < 1e-10) h = 1.0; // fallback for zero variance

        double sum = 0;
        for (int i = 0; i < Samples.Length; i++)
        {
            double u = (value - Samples[i]) / h;
            sum += Math.Exp(-0.5 * u * u);
        }

        return sum / (Samples.Length * h * Math.Sqrt(2.0 * Math.PI));
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"P({TargetVariableName} | do({InterventionVariableName} = {NumOps.ToDouble(InterventionValue):F3})) " +
               $"~ Mean={Mean:F4}, Std={StandardDeviation:F4}, ACE={AverageCausalEffect:F4}, " +
               $"95% CI=[{ConfidenceInterval().Lower:F4}, {ConfidenceInterval().Upper:F4}]";
    }
}
