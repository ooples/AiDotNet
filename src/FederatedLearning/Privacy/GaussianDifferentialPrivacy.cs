namespace AiDotNet.FederatedLearning.Privacy;

using System;
using AiDotNet.Tensors.Helpers;

/// <summary>
/// Implements differential privacy using the Gaussian mechanism.
/// </summary>
/// <remarks>
/// The Gaussian mechanism provides (ε, δ)-differential privacy by adding calibrated
/// Gaussian (normal distribution) noise to model parameters. This is one of the most
/// widely used privacy mechanisms in federated learning.
///
/// <b>For Beginners:</b> Differential privacy is like adding static to a phone conversation.
/// You add just enough noise that individual voices can't be identified, but the overall
/// message still gets through clearly.
///
/// How Gaussian Differential Privacy works:
/// 1. Clip gradients/parameters to bound sensitivity (maximum change any single data point can cause)
/// 2. Add Gaussian noise: noise ~ N(0, σ²) where σ depends on ε, δ, and sensitivity
/// 3. The noise calibration ensures (ε, δ)-DP guarantee
///
/// Mathematical formulation:
/// - Sensitivity Δ: Maximum L2 norm of gradient for any single training example
/// - Noise scale σ = (Δ/ε) × sqrt(2 × ln(1.25/δ))
/// - For each parameter w: w_private = w + N(0, σ²)
///
/// Privacy parameters:
/// - ε (epsilon): Privacy budget - smaller is more private
///   * ε = 0.1: Very strong privacy, significant noise
///   * ε = 1.0: Strong privacy, moderate noise (recommended)
///   * ε = 10: Weak privacy, minimal noise
///
/// - δ (delta): Failure probability - should be very small
///   * Typically δ = 1/n² where n is dataset size
///   * Common choice: δ = 1e-5
///
/// For example, protecting hospital patient data:
/// - Original gradient: [0.5, -0.3, 0.8, -0.2]
/// - Clip to max norm 1.0: [0.45, -0.27, 0.72, -0.18] (clipped)
/// - Add Gaussian noise with σ=0.1: [0.47, -0.29, 0.75, -0.21]
/// - Result: Individual patient influence is masked by noise
///
/// Privacy composition:
/// - Each time you share data, you consume privacy budget
/// - After T rounds with ε per round: total ε_total ≈ ε × sqrt(2T × ln(1/δ))
/// - This is more efficient than naive composition (ε × T)
///
/// Trade-offs:
/// - More privacy (smaller ε) → more noise → lower accuracy
/// - Less privacy (larger ε) → less noise → higher accuracy
/// - Must find acceptable balance for your application
///
/// When to use Gaussian DP:
/// - Need provable privacy guarantees
/// - Working with sensitive data (healthcare, finance)
/// - Regulatory requirements (GDPR, HIPAA)
/// - Publishing models or sharing with untrusted parties
///
/// Reference: Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy."
/// Abadi, M., et al. (2016). "Deep Learning with Differential Privacy." CCS 2016.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public class GaussianDifferentialPrivacy<T> : PrivacyMechanismBase<Dictionary<string, T[]>, T>
{
    private readonly object _sync = new object();
    private double _privacyBudgetConsumed;
    private readonly double _clipNorm;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the <see cref="GaussianDifferentialPrivacy{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a differential privacy mechanism with a specified
    /// gradient clipping threshold.
    ///
    /// Gradient clipping (clipNorm) is crucial for DP:
    /// - Bounds the maximum influence any single data point can have
    /// - Makes noise calibration possible
    /// - Common values: 0.1 - 10.0 depending on model and data
    ///
    /// Lower clipNorm:
    /// - Stronger privacy guarantee
    /// - More aggressive clipping
    /// - May slow convergence
    ///
    /// Higher clipNorm:
    /// - Less clipping
    /// - Faster convergence
    /// - Requires more noise for same privacy
    ///
    /// Recommendations:
    /// - Start with clipNorm = 1.0
    /// - Monitor gradient norms during training
    /// - Adjust based on typical gradient magnitudes
    /// </remarks>
    /// <param name="clipNorm">The maximum L2 norm for gradient clipping (sensitivity bound).</param>
    /// <param name="randomSeed">Optional random seed for reproducibility.</param>
    public GaussianDifferentialPrivacy(double clipNorm = 1.0, int? randomSeed = null)
    {
        if (clipNorm <= 0)
        {
            throw new ArgumentException("Clip norm must be positive.", nameof(clipNorm));
        }

        _clipNorm = clipNorm;
        _privacyBudgetConsumed = 0.0;
        _random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Applies differential privacy to model parameters by adding calibrated Gaussian noise.
    /// </summary>
    /// <remarks>
    /// This method implements the Gaussian mechanism for (ε, δ)-differential privacy:
    ///
    /// <b>For Beginners:</b> This adds carefully calculated random noise to protect privacy
    /// while maintaining model utility.
    ///
    /// Step-by-step process:
    /// 1. Calculate current L2 norm of model parameters
    /// 2. If norm > clipNorm, scale down parameters to clipNorm
    /// 3. Calculate noise scale σ based on ε, δ, and sensitivity
    /// 4. Add Gaussian noise N(0, σ²) to each parameter
    /// 5. Update privacy budget consumed
    ///
    /// Mathematical details:
    /// - Sensitivity Δ = clipNorm (worst-case parameter change)
    /// - σ = (Δ/ε) × sqrt(2 × ln(1.25/δ))
    /// - Noise ~ N(0, σ²) added to each parameter independently
    ///
    /// For example, with ε=1.0, δ=1e-5, clipNorm=1.0:
    /// - σ = (1.0/1.0) × sqrt(2 × ln(125000)) ≈ 4.7
    /// - Each parameter gets noise from N(0, 4.7²)
    /// - Original params: [0.5, -0.3, 0.8]
    /// - Noisy params: [0.52, -0.35, 0.83] (example with small noise realization)
    ///
    /// Privacy accounting:
    /// - Each call consumes ε privacy budget
    /// - Total budget accumulates: ε_total = ε_1 + ε_2 + ... (simplified)
    /// - Advanced: Use Rényi DP for tighter composition bounds
    /// </remarks>
    /// <param name="model">The model parameters to add noise to.</param>
    /// <param name="epsilon">Privacy budget for this operation (smaller = more private).</param>
    /// <param name="delta">Failure probability (typically 1e-5 or smaller).</param>
    /// <returns>The model with differential privacy applied.</returns>
    public override Dictionary<string, T[]> ApplyPrivacy(Dictionary<string, T[]> model, double epsilon, double delta)
    {
        if (model == null || model.Count == 0)
        {
            throw new ArgumentException("Model cannot be null or empty.", nameof(model));
        }

        if (epsilon <= 0)
        {
            throw new ArgumentException("Epsilon must be positive.", nameof(epsilon));
        }

        if (delta <= 0 || delta >= 1)
        {
            throw new ArgumentException("Delta must be in (0, 1).", nameof(delta));
        }

        // Create a copy of the model
        var noisyModel = new Dictionary<string, T[]>();
        foreach (var layer in model)
        {
            noisyModel[layer.Key] = (T[])layer.Value.Clone();
        }

        // Step 1: Gradient clipping - Calculate L2 norm of all parameters
        var l2Norm = CalculateL2Norm(noisyModel);
        var clipNormT = NumOps.FromDouble(_clipNorm);

        // If norm exceeds clip threshold, scale down
        if (NumOps.GreaterThan(l2Norm, clipNormT))
        {
            var scaleFactor = NumOps.Divide(clipNormT, l2Norm);

            foreach (var parameters in noisyModel.Values)
            {
                for (int i = 0; i < parameters.Length; i++)
                {
                    parameters[i] = NumOps.Multiply(parameters[i], scaleFactor);
                }
            }
        }

        // Step 2: Calculate noise scale based on Gaussian mechanism
        // σ = (Δ/ε) × sqrt(2 × ln(1.25/δ))
        // where Δ = clipNorm (sensitivity)
        double sensitivity = _clipNorm;
        double noiseSigma = (sensitivity / epsilon) * Math.Sqrt(2.0 * Math.Log(1.25 / delta));

        // Step 3: Add Gaussian noise to each parameter
        foreach (var parameters in noisyModel.Values)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                double noise = GenerateGaussianNoise(0.0, noiseSigma);
                parameters[i] = NumOps.Add(parameters[i], NumOps.FromDouble(noise));
            }
        }

        // Update privacy budget consumed
        lock (_sync)
        {
            _privacyBudgetConsumed += epsilon;
        }

        return noisyModel;
    }

    /// <summary>
    /// Calculates the L2 norm (Euclidean norm) of all model parameters.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> L2 norm is the "length" of the parameter vector in
    /// high-dimensional space. It's calculated as sqrt(sum of squares).
    ///
    /// For example, with parameters [3, 4]:
    /// - L2 norm = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
    ///
    /// Used for gradient clipping to bound sensitivity.
    /// </remarks>
    /// <param name="model">The model to calculate norm for.</param>
    /// <returns>The L2 norm of all parameters.</returns>
    private T CalculateL2Norm(Dictionary<string, T[]> model)
    {
        var sumOfSquares = NumOps.Zero;

        foreach (var param in model.Values.SelectMany(layer => layer))
        {
            sumOfSquares = NumOps.Add(sumOfSquares, NumOps.Square(param));
        }

        return NumOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Generates a sample from a Gaussian (normal) distribution.
    /// </summary>
    /// <remarks>
    /// Uses the Box-Muller transform to generate Gaussian random variables from
    /// uniform random variables.
    ///
    /// <b>For Beginners:</b> This creates random noise from a bell curve distribution.
    /// Most noise values will be close to the mean, with rare large values.
    ///
    /// Box-Muller transform:
    /// - Generate two uniform random numbers U1, U2 in [0, 1]
    /// - Z = sqrt(-2 × ln(U1)) × cos(2π × U2)
    /// - Z follows standard normal N(0, 1)
    /// - Scale and shift: X = mean + sigma × Z
    /// </remarks>
    /// <param name="mean">The mean of the Gaussian distribution.</param>
    /// <param name="sigma">The standard deviation of the Gaussian distribution.</param>
    /// <returns>A random sample from N(mean, sigma²).</returns>
    private double GenerateGaussianNoise(double mean, double sigma)
    {
        // Box-Muller transform
        double u1;
        double u2;
        lock (_sync)
        {
            u1 = 1.0 - _random.NextDouble(); // Uniform(0,1]
            u2 = 1.0 - _random.NextDouble();
        }
        double standardNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

        return mean + sigma * standardNormal;
    }

    /// <summary>
    /// Gets the total privacy budget consumed so far.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Returns how much privacy budget has been used up.
    /// Privacy budget is cumulative - once spent, it's gone.
    ///
    /// For example:
    /// - Round 1: ε = 0.5 consumed, total = 0.5
    /// - Round 2: ε = 0.5 consumed, total = 1.0
    /// - Round 3: ε = 0.5 consumed, total = 1.5
    ///
    /// If you started with total budget 10.0, you have 8.5 remaining.
    ///
    /// Note: This uses basic composition. Advanced composition (Rényi DP) gives
    /// tighter bounds and would show less budget consumed.
    /// </remarks>
    public override double GetPrivacyBudgetConsumed()
    {
        lock (_sync)
        {
            return _privacyBudgetConsumed;
        }
    }

    /// <summary>
    /// Gets the name of the privacy mechanism.
    /// </summary>
    /// <returns>A string describing the mechanism.</returns>
    public override string GetMechanismName()
    {
        return $"Gaussian DP (clip={_clipNorm})";
    }

    /// <summary>
    /// Gets the gradient clipping norm used for sensitivity bounding.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Returns the maximum allowed parameter norm.
    /// Parameters larger than this are scaled down before adding noise.
    /// </remarks>
    /// <returns>The clipping norm value.</returns>
    public double GetClipNorm()
    {
        return _clipNorm;
    }

    /// <summary>
    /// Resets the privacy budget counter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Resets the privacy budget tracker to zero.
    ///
    /// WARNING: This should only be used when starting a completely new training run.
    /// Do not reset during active training as it would give false privacy accounting.
    ///
    /// Use cases:
    /// - Starting new experiment with same mechanism instance
    /// - Testing and debugging
    /// - Separate training phases with independent privacy guarantees
    /// </remarks>
    public void ResetPrivacyBudget()
    {
        lock (_sync)
        {
            _privacyBudgetConsumed = 0.0;
        }
    }
}
