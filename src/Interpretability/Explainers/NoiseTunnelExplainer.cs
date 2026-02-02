using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Noise Tunnel wrapper that smooths attributions by averaging over noisy inputs.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TExplanation">The type of explanation produced by the wrapped explainer.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Gradient-based attribution methods can produce noisy results
/// because gradients are sensitive to small input perturbations. NoiseTunnel addresses
/// this by averaging attributions computed on multiple noisy versions of the input.
///
/// <b>How it works:</b>
/// 1. Take the original input
/// 2. Create multiple copies with small random noise added
/// 3. Compute attributions for each noisy copy
/// 4. Average (or aggregate) the attributions
///
/// <b>Aggregation methods:</b>
/// - SmoothGrad: Simple average of attributions (most common)
/// - SmoothGrad-Squared: Average of squared attributions (emphasizes important features)
/// - VarGrad: Variance of attributions (shows where gradients vary most)
///
/// <b>Benefits:</b>
/// - Reduces noise in attribution maps
/// - Makes explanations more visually interpretable
/// - More stable under small input changes
///
/// <b>Parameters:</b>
/// - NumSamples: More samples = smoother results (default: 5-10)
/// - StdDev: Standard deviation of noise (typical: 0.1-0.3 of input range)
///
/// <b>Reference:</b>
/// Smilkov et al., "SmoothGrad: removing noise by adding noise" (2017)
/// </para>
/// </remarks>
public class NoiseTunnelExplainer<T, TExplanation>
    where TExplanation : class
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ILocalExplainer<T, TExplanation> _baseExplainer;
    private readonly NoiseTunnelType _noiseTunnelType;
    private readonly int _numSamples;
    private readonly double _stdDev;
    private readonly Func<TExplanation, Vector<T>> _attributionExtractor;
    private readonly Func<Vector<T>, TExplanation, TExplanation> _attributionReplacer;
    private readonly Random _random;

    /// <summary>Gets the name of this method.</summary>
    public string MethodName => $"NoiseTunnel({_baseExplainer.MethodName}, {_noiseTunnelType})";

    /// <summary>
    /// Initializes a NoiseTunnel wrapper around a base explainer.
    /// </summary>
    /// <param name="baseExplainer">The underlying explainer to wrap.</param>
    /// <param name="attributionExtractor">Function to extract attribution vector from explanation.</param>
    /// <param name="attributionReplacer">Function to create new explanation with modified attributions.</param>
    /// <param name="noiseTunnelType">Type of aggregation (SmoothGrad, SquaredSmoothGrad, VarGrad).</param>
    /// <param name="numSamples">Number of noisy samples to average.</param>
    /// <param name="stdDev">Standard deviation of Gaussian noise.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>baseExplainer:</b> Any gradient-based explainer (SHAP, Integrated Gradients, etc.)
    /// - <b>attributionExtractor:</b> How to get the attribution vector from an explanation
    /// - <b>attributionReplacer:</b> How to create a new explanation with smoothed attributions
    /// - <b>numSamples:</b> Higher = smoother but slower (5-20 is typical)
    /// - <b>stdDev:</b> How much noise to add (0.1-0.3 is typical)
    /// </para>
    /// </remarks>
    public NoiseTunnelExplainer(
        ILocalExplainer<T, TExplanation> baseExplainer,
        Func<TExplanation, Vector<T>> attributionExtractor,
        Func<Vector<T>, TExplanation, TExplanation> attributionReplacer,
        NoiseTunnelType noiseTunnelType = NoiseTunnelType.SmoothGrad,
        int numSamples = 10,
        double stdDev = 0.15,
        int? seed = null)
    {
        _baseExplainer = baseExplainer ?? throw new ArgumentNullException(nameof(baseExplainer));
        _attributionExtractor = attributionExtractor ?? throw new ArgumentNullException(nameof(attributionExtractor));
        _attributionReplacer = attributionReplacer ?? throw new ArgumentNullException(nameof(attributionReplacer));
        _noiseTunnelType = noiseTunnelType;
        _numSamples = numSamples > 0 ? numSamples : throw new ArgumentException("numSamples must be positive");
        _stdDev = stdDev > 0 ? stdDev : throw new ArgumentException("stdDev must be positive");
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Explains an input with noise-smoothed attributions.
    /// </summary>
    /// <param name="input">The input to explain.</param>
    /// <returns>Smoothed explanation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Creates multiple noisy copies of the input
    /// 2. Gets attributions for each noisy copy
    /// 3. Aggregates according to the NoiseTunnelType
    /// 4. Returns an explanation with smoothed attributions
    /// </para>
    /// </remarks>
    public TExplanation Explain(Vector<T> input)
    {
        var allAttributions = new List<Vector<T>>(_numSamples);

        // Collect attributions from noisy samples
        for (int i = 0; i < _numSamples; i++)
        {
            var noisyInput = AddGaussianNoise(input);
            var explanation = _baseExplainer.Explain(noisyInput);
            var attributions = _attributionExtractor(explanation);
            allAttributions.Add(attributions);
        }

        // Aggregate attributions based on tunnel type
        var aggregated = AggregateAttributions(allAttributions);

        // Get the base explanation (for the original input) and replace attributions
        var baseExplanation = _baseExplainer.Explain(input);
        return _attributionReplacer(aggregated, baseExplanation);
    }

    /// <summary>
    /// Explains multiple inputs with noise-smoothed attributions.
    /// </summary>
    /// <param name="inputs">The inputs to explain.</param>
    /// <returns>Array of smoothed explanations.</returns>
    public TExplanation[] ExplainBatch(Matrix<T> inputs)
    {
        var results = new TExplanation[inputs.Rows];
        for (int i = 0; i < inputs.Rows; i++)
        {
            results[i] = Explain(inputs.GetRow(i));
        }
        return results;
    }

    /// <summary>
    /// Adds Gaussian noise to an input vector.
    /// </summary>
    /// <param name="input">Original input.</param>
    /// <returns>Input with added noise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gaussian (normal) noise is added to each feature.
    /// The noise has mean 0 and standard deviation _stdDev.
    /// This simulates small perturbations to the input.
    /// </para>
    /// </remarks>
    private Vector<T> AddGaussianNoise(Vector<T> input)
    {
        var noisy = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            double noise = GenerateGaussian() * _stdDev;
            double original = NumOps.ToDouble(input[i]);
            noisy[i] = NumOps.FromDouble(original + noise);
        }
        return new Vector<T>(noisy);
    }

    /// <summary>
    /// Generates a sample from standard normal distribution.
    /// </summary>
    /// <returns>A random value from N(0,1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses Box-Muller transform to convert uniform
    /// random numbers to Gaussian (normal) distributed numbers.
    /// </para>
    /// </remarks>
    private double GenerateGaussian()
    {
        // Box-Muller transform
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Aggregates multiple attribution vectors based on the tunnel type.
    /// </summary>
    /// <param name="attributions">List of attribution vectors.</param>
    /// <returns>Aggregated attribution vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different aggregation methods provide different insights:
    /// - SmoothGrad: Average reduces noise, shows stable important features
    /// - SquaredSmoothGrad: Square before averaging emphasizes strong signals
    /// - VarGrad: Variance shows where the model is most uncertain
    /// </para>
    /// </remarks>
    private Vector<T> AggregateAttributions(List<Vector<T>> attributions)
    {
        if (attributions.Count == 0)
            throw new InvalidOperationException("No attributions to aggregate.");

        int length = attributions[0].Length;
        var result = new T[length];

        switch (_noiseTunnelType)
        {
            case NoiseTunnelType.SmoothGrad:
                result = ComputeMean(attributions);
                break;

            case NoiseTunnelType.SquaredSmoothGrad:
                result = ComputeSquaredMean(attributions);
                break;

            case NoiseTunnelType.VarGrad:
                result = ComputeVariance(attributions);
                break;

            default:
                result = ComputeMean(attributions);
                break;
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes element-wise mean of attribution vectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Standard SmoothGrad - averages all attributions.
    /// If feature i has high attribution in most noisy samples, the average
    /// will be high. Random noise cancels out in the average.
    /// </para>
    /// </remarks>
    private T[] ComputeMean(List<Vector<T>> attributions)
    {
        int length = attributions[0].Length;
        var result = new T[length];
        double n = attributions.Count;

        for (int i = 0; i < length; i++)
        {
            double sum = 0;
            foreach (var attr in attributions)
            {
                sum += NumOps.ToDouble(attr[i]);
            }
            result[i] = NumOps.FromDouble(sum / n);
        }

        return result;
    }

    /// <summary>
    /// Computes element-wise mean of squared attributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SmoothGrad-Squared - squares each attribution before averaging.
    /// This emphasizes features with consistently large (positive or negative) attributions
    /// and de-emphasizes features that fluctuate around zero.
    /// </para>
    /// </remarks>
    private T[] ComputeSquaredMean(List<Vector<T>> attributions)
    {
        int length = attributions[0].Length;
        var result = new T[length];
        double n = attributions.Count;

        for (int i = 0; i < length; i++)
        {
            double sum = 0;
            foreach (var attr in attributions)
            {
                double val = NumOps.ToDouble(attr[i]);
                sum += val * val;
            }
            result[i] = NumOps.FromDouble(Math.Sqrt(sum / n));
        }

        return result;
    }

    /// <summary>
    /// Computes element-wise variance of attributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VarGrad - computes variance instead of mean.
    /// High variance at a feature means the model's sensitivity to that
    /// feature changes a lot with small input perturbations. This can
    /// indicate features at decision boundaries.
    /// </para>
    /// </remarks>
    private T[] ComputeVariance(List<Vector<T>> attributions)
    {
        int length = attributions[0].Length;
        var result = new T[length];
        double n = attributions.Count;

        for (int i = 0; i < length; i++)
        {
            // Compute mean
            double mean = 0;
            foreach (var attr in attributions)
            {
                mean += NumOps.ToDouble(attr[i]);
            }
            mean /= n;

            // Compute variance with Bessel's correction
            double variance = 0;
            foreach (var attr in attributions)
            {
                double diff = NumOps.ToDouble(attr[i]) - mean;
                variance += diff * diff;
            }

            // Apply Bessel's correction only when n > 1 to avoid division by zero
            if (n > 1)
            {
                variance /= (n - 1);
            }
            else
            {
                // With only one sample, variance is undefined; use 0 as a fallback
                variance = 0;
            }

            result[i] = NumOps.FromDouble(Math.Sqrt(variance));
        }

        return result;
    }
}

/// <summary>
/// Types of noise tunnel aggregation methods.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These methods determine how multiple noisy attributions
/// are combined into a single final attribution.
/// </para>
/// </remarks>
public enum NoiseTunnelType
{
    /// <summary>
    /// SmoothGrad: Simple average of attributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The standard method. Averages out random noise while
    /// preserving consistent signals. Most commonly used.
    /// </para>
    /// </remarks>
    SmoothGrad,

    /// <summary>
    /// SmoothGrad-Squared: Average of squared attributions, then sqrt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Squares values before averaging, so large values
    /// have more influence. Good when you want to emphasize strong attributions
    /// regardless of sign.
    /// </para>
    /// </remarks>
    SquaredSmoothGrad,

    /// <summary>
    /// VarGrad: Variance of attributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Shows where attributions vary most with input noise.
    /// High variance means the model is very sensitive to small changes in that
    /// feature, often indicating decision boundaries.
    /// </para>
    /// </remarks>
    VarGrad
}

/// <summary>
/// Convenience factory for creating NoiseTunnel wrappers for common explainer types.
/// </summary>
public static class NoiseTunnelFactory
{
    /// <summary>
    /// Creates a NoiseTunnel wrapper for SHAP explainers.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="shapExplainer">The SHAP explainer to wrap.</param>
    /// <param name="noiseTunnelType">Type of aggregation.</param>
    /// <param name="numSamples">Number of noisy samples.</param>
    /// <param name="stdDev">Standard deviation of noise.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>NoiseTunnel-wrapped SHAP explainer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper creates a NoiseTunnel for SHAP that
    /// automatically extracts and replaces SHAP values.
    /// </para>
    /// </remarks>
    public static NoiseTunnelExplainer<T, SHAPExplanation<T>> ForShap<T>(
        ILocalExplainer<T, SHAPExplanation<T>> shapExplainer,
        NoiseTunnelType noiseTunnelType = NoiseTunnelType.SmoothGrad,
        int numSamples = 10,
        double stdDev = 0.15,
        int? seed = null)
    {
        return new NoiseTunnelExplainer<T, SHAPExplanation<T>>(
            shapExplainer,
            explanation => explanation.ShapValues,
            (newValues, original) => new SHAPExplanation<T>(
                newValues,
                original.BaselineValue,
                original.Prediction,
                original.FeatureNames),
            noiseTunnelType,
            numSamples,
            stdDev,
            seed);
    }

    /// <summary>
    /// Creates a NoiseTunnel wrapper for IntegratedGradients explainers.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="igExplainer">The IntegratedGradients explainer to wrap.</param>
    /// <param name="noiseTunnelType">Type of aggregation.</param>
    /// <param name="numSamples">Number of noisy samples.</param>
    /// <param name="stdDev">Standard deviation of noise.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>NoiseTunnel-wrapped IntegratedGradients explainer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper creates a NoiseTunnel for Integrated Gradients
    /// that automatically extracts and replaces the attribution values.
    /// </para>
    /// </remarks>
    public static NoiseTunnelExplainer<T, IntegratedGradientsExplanation<T>> ForIntegratedGradients<T>(
        ILocalExplainer<T, IntegratedGradientsExplanation<T>> igExplainer,
        NoiseTunnelType noiseTunnelType = NoiseTunnelType.SmoothGrad,
        int numSamples = 10,
        double stdDev = 0.15,
        int? seed = null)
    {
        return new NoiseTunnelExplainer<T, IntegratedGradientsExplanation<T>>(
            igExplainer,
            explanation => explanation.Attributions,
            (newValues, original) => new IntegratedGradientsExplanation<T>
            {
                Attributions = newValues,
                Baseline = original.Baseline,
                Input = original.Input,
                BaselinePrediction = original.BaselinePrediction,
                InputPrediction = original.InputPrediction
            },
            noiseTunnelType,
            numSamples,
            stdDev,
            seed);
    }

    /// <summary>
    /// Creates a NoiseTunnel wrapper for GuidedBackprop explainers.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gbpExplainer">The GuidedBackprop explainer to wrap.</param>
    /// <param name="noiseTunnelType">Type of aggregation.</param>
    /// <param name="numSamples">Number of noisy samples.</param>
    /// <param name="stdDev">Standard deviation of noise.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>NoiseTunnel-wrapped GuidedBackprop explainer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper creates a NoiseTunnel for GuidedBackprop
    /// that smooths the gradient visualization.
    /// </para>
    /// </remarks>
    public static NoiseTunnelExplainer<T, GuidedBackpropExplanation<T>> ForGuidedBackprop<T>(
        ILocalExplainer<T, GuidedBackpropExplanation<T>> gbpExplainer,
        NoiseTunnelType noiseTunnelType = NoiseTunnelType.SmoothGrad,
        int numSamples = 10,
        double stdDev = 0.15,
        int? seed = null)
    {
        return new NoiseTunnelExplainer<T, GuidedBackpropExplanation<T>>(
            gbpExplainer,
            explanation => explanation.GuidedGradients,
            (newValues, original) => new GuidedBackpropExplanation<T>(
                original.Input,
                newValues,
                original.TargetClass,
                original.Prediction,
                original.InputShape,
                original.GradientTensor),
            noiseTunnelType,
            numSamples,
            stdDev,
            seed);
    }

}
