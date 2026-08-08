using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// The feature-wise transformation layer of Tseng et al. (arXiv:2001.08735): a per-channel affine
/// perturbation whose scale and bias are SAMPLED from learned hyper-parameters.
/// </summary>
/// <remarks>
/// <para>
/// Given an activation <c>z</c> with C channels, the modulated activation is
/// </para>
/// <code>
///   gamma_c ~ N(1, softplus(theta_gamma_c))      beta_c ~ N(0, softplus(theta_beta_c))
///   zhat_(c,h,w) = gamma_c * z_(c,h,w) + beta_c
/// </code>
/// <para>
/// Three details carry the method, and each is easy to get subtly wrong:
/// </para>
/// <list type="number">
/// <item><description>The scale is centred on ONE and the bias on ZERO, so the expected
/// transformation is the identity. It perturbs the feature distribution without biasing it.</description></item>
/// <item><description>The hyper-parameters pass through SOFTPLUS to become standard deviations,
/// which keeps them positive under unconstrained optimization. They are the spread of the
/// perturbation, not the perturbation itself.</description></item>
/// <item><description>gamma and beta are RESAMPLED every application. A fixed draw would be one more
/// deterministic layer for the encoder to absorb; it is the variation across draws that simulates
/// "various feature distributions under different domains".</description></item>
/// </list>
/// <para>
/// <b>For Beginners:</b> This jiggles the numbers coming out of the feature extractor — multiplying
/// each channel by something near 1 and adding something near 0. Because the jiggle is different
/// every time, whatever comes next cannot rely on the exact values, only on their pattern. How big
/// the jiggle is allowed to be is the thing being learned.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class FeatureWiseTransformation<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly Random _random;

    /// <summary>theta_gamma — one entry per channel, the scale term's pre-softplus spread.</summary>
    public Vector<T> ScaleHyperparameters { get; private set; }

    /// <summary>theta_beta — one entry per channel, the bias term's pre-softplus spread.</summary>
    public Vector<T> BiasHyperparameters { get; private set; }

    /// <summary>Gets the channel count C the transformation is defined over.</summary>
    public int FeatureDimension => ScaleHyperparameters.Length;

    /// <summary>
    /// Initializes the transformation with the paper's pre-determined hyper-parameter values.
    /// </summary>
    /// <param name="featureDimension">Channel count C.</param>
    /// <param name="initialScale">Initial theta_gamma (the paper's hand-tuned value is 0.3).</param>
    /// <param name="initialBias">Initial theta_beta (the paper's hand-tuned value is 0.5).</param>
    /// <param name="random">RNG used to draw gamma and beta; supply a seeded one for reproducibility.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the feature dimension is not positive.</exception>
    public FeatureWiseTransformation(int featureDimension, double initialScale, double initialBias, Random random)
    {
        if (featureDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(featureDimension), featureDimension,
                "Feature dimension (channel count C) must be positive.");
        }

        _random = random ?? throw new ArgumentNullException(nameof(random));
        ScaleHyperparameters = new Vector<T>(featureDimension);
        BiasHyperparameters = new Vector<T>(featureDimension);
        for (int c = 0; c < featureDimension; c++)
        {
            ScaleHyperparameters[c] = Ops.FromDouble(initialScale);
            BiasHyperparameters[c] = Ops.FromDouble(initialBias);
        }
    }

    /// <summary>
    /// Applies one freshly-sampled transformation to a feature vector.
    /// </summary>
    /// <remarks>
    /// Resamples gamma and beta on every call — see the class remarks for why that is the mechanism
    /// rather than an implementation choice.
    /// </remarks>
    public Vector<T> Apply(Vector<T> features)
    {
        var result = new Vector<T>(features.Length);
        int channels = Math.Min(features.Length, FeatureDimension);

        for (int c = 0; c < channels; c++)
        {
            double gamma = 1.0 + SampleGaussian() * Softplus(Ops.ToDouble(ScaleHyperparameters[c]));
            double beta = SampleGaussian() * Softplus(Ops.ToDouble(BiasHyperparameters[c]));
            result[c] = Ops.FromDouble(gamma * Ops.ToDouble(features[c]) + beta);
        }

        // Channels past C are passed through untouched rather than dropped: the transformation is
        // defined per-channel over C, and silently truncating the feature vector would change the
        // encoder's output width instead of perturbing it.
        for (int c = channels; c < features.Length; c++) result[c] = features[c];

        return result;
    }

    /// <summary>
    /// Replaces both hyper-parameter vectors, for the learning-to-learn update.
    /// </summary>
    public void SetHyperparameters(Vector<T> scale, Vector<T> bias)
    {
        ScaleHyperparameters = scale ?? throw new ArgumentNullException(nameof(scale));
        BiasHyperparameters = bias ?? throw new ArgumentNullException(nameof(bias));
    }

    /// <summary>
    /// The standard deviation actually used for channel <paramref name="channel"/>'s scale term,
    /// i.e. <c>softplus(theta_gamma_c)</c>.
    /// </summary>
    public double EffectiveScaleStdDev(int channel) => Softplus(Ops.ToDouble(ScaleHyperparameters[channel]));

    /// <summary>
    /// The standard deviation actually used for channel <paramref name="channel"/>'s bias term,
    /// i.e. <c>softplus(theta_beta_c)</c>.
    /// </summary>
    public double EffectiveBiasStdDev(int channel) => Softplus(Ops.ToDouble(BiasHyperparameters[channel]));

    /// <summary>
    /// <c>log(1 + exp(x))</c>, computed in the overflow-safe form so a large hyper-parameter cannot
    /// produce infinity.
    /// </summary>
    private static double Softplus(double x) => x > 30.0 ? x : Math.Log(1.0 + Math.Exp(x));

    private double SampleGaussian()
    {
        // Box-Muller through the supplied RNG, so a seeded algorithm stays reproducible.
        double u1 = 1.0 - _random.NextDouble();   // in (0, 1] keeps the log finite
        double u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
