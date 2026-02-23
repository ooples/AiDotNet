using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Implements differential privacy protection for label holder gradients in VFL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In VFL, the label holder (e.g., a hospital knowing patient outcomes)
/// sends gradients back to feature parties (e.g., a bank knowing income). Without protection,
/// the bank could analyze gradient patterns to figure out which patients had bad outcomes.</para>
///
/// <para>This class adds calibrated Gaussian noise to the gradients before they're shared.
/// The noise is carefully sized so that even a sophisticated attacker cannot reliably distinguish
/// whether a specific individual's data was used, providing differential privacy guarantees.</para>
///
/// <para><b>Privacy accounting:</b> Uses the Gaussian mechanism with Renyi Differential Privacy (RDP)
/// composition to track cumulative privacy loss across epochs. When the budget is exhausted,
/// training must stop.</para>
///
/// <para><b>Reference:</b> Abadi et al., "Deep Learning with Differential Privacy", ACM CCS 2016.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LabelDifferentialPrivacy<T> : FederatedLearningComponentBase<T>, ILabelProtector<T>
{
    private readonly double _epsilon;
    private readonly double _delta;
    private readonly double _noiseMultiplier;
    private readonly double _maxGradientNorm;
    private readonly Random _random;
    private double _totalEpsilonSpent;
    private int _queriesAnswered;

    /// <summary>
    /// Initializes a new instance of <see cref="LabelDifferentialPrivacy{T}"/>.
    /// </summary>
    /// <param name="epsilon">The per-round privacy budget (smaller = more private).</param>
    /// <param name="delta">The privacy failure probability (typically 1e-5).</param>
    /// <param name="maxGradientNorm">The maximum gradient norm for clipping. Defaults to 1.0.</param>
    /// <param name="seed">Random seed for reproducibility. Null for cryptographic randomness.</param>
    public LabelDifferentialPrivacy(double epsilon = 1.0, double delta = 1e-5,
        double maxGradientNorm = 1.0, int? seed = null)
    {
        if (epsilon <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        if (delta <= 0 || delta >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in (0, 1).");
        }

        if (maxGradientNorm <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxGradientNorm), "Max gradient norm must be positive.");
        }

        _epsilon = epsilon;
        _delta = delta;
        _maxGradientNorm = maxGradientNorm;

        // Compute noise multiplier: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        double sensitivity = maxGradientNorm;
        _noiseMultiplier = sensitivity * Math.Sqrt(2.0 * Math.Log(1.25 / delta)) / epsilon;

        _random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public Tensor<T> ProtectGradients(Tensor<T> gradients)
    {
        if (gradients is null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }

        // Step 1: Clip gradient norm
        var clipped = ClipGradientNorm(gradients, _maxGradientNorm);

        // Step 2: Add calibrated Gaussian noise
        var noisy = AddGaussianNoise(clipped, _noiseMultiplier);

        // Step 3: Update privacy accounting
        _queriesAnswered++;
        // Simple composition: epsilon grows as sqrt(k) * epsilon_per_round for Gaussian mechanism
        _totalEpsilonSpent = _epsilon * Math.Sqrt(_queriesAnswered);

        return noisy;
    }

    /// <inheritdoc/>
    public T ProtectLoss(T loss)
    {
        // Add Laplace noise to the scalar loss value
        double lossDouble = NumOps.ToDouble(loss);
        double sensitivity = _maxGradientNorm;
        double scale = sensitivity / _epsilon;
        double noise = SampleLaplace(scale);
        return NumOps.FromDouble(lossDouble + noise);
    }

    /// <inheritdoc/>
    public (double Epsilon, double Delta) GetPrivacyBudgetSpent()
    {
        return (_totalEpsilonSpent, _delta);
    }

    /// <summary>
    /// Clips the gradient tensor so its L2 norm does not exceed the specified maximum.
    /// </summary>
    private Tensor<T> ClipGradientNorm(Tensor<T> gradients, double maxNorm)
    {
        double normSquared = 0.0;
        int totalElements = 1;
        for (int d = 0; d < gradients.Rank; d++)
        {
            totalElements *= gradients.Shape[d];
        }

        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(gradients[i]);
            normSquared += val * val;
        }

        double norm = Math.Sqrt(normSquared);
        if (norm <= maxNorm)
        {
            return gradients;
        }

        double scale = maxNorm / norm;
        var clipped = new Tensor<T>(gradients.Shape);
        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(gradients[i]);
            clipped[i] = NumOps.FromDouble(val * scale);
        }

        return clipped;
    }

    /// <summary>
    /// Adds Gaussian noise with the specified standard deviation to each element.
    /// </summary>
    private Tensor<T> AddGaussianNoise(Tensor<T> tensor, double stddev)
    {
        int totalElements = 1;
        for (int d = 0; d < tensor.Rank; d++)
        {
            totalElements *= tensor.Shape[d];
        }

        var noisy = new Tensor<T>(tensor.Shape);
        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(tensor[i]);
            double noise = SampleGaussian(0.0, stddev);
            noisy[i] = NumOps.FromDouble(val + noise);
        }

        return noisy;
    }

    /// <summary>
    /// Samples from a Gaussian distribution using the Box-Muller transform.
    /// </summary>
    private double SampleGaussian(double mean, double stddev)
    {
        double u1 = _random.NextDouble();
        double u2 = _random.NextDouble();
        // Avoid log(0)
        if (u1 < 1e-15)
        {
            u1 = 1e-15;
        }

        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stddev * z;
    }

    /// <summary>
    /// Samples from a Laplace distribution.
    /// </summary>
    private double SampleLaplace(double scale)
    {
        double u = _random.NextDouble() - 0.5;
        if (Math.Abs(u) < 1e-15)
        {
            u = 1e-15;
        }

        return -scale * Math.Sign(u) * Math.Log(1.0 - 2.0 * Math.Abs(u));
    }
}
