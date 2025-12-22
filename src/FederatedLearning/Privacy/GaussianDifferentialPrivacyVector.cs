using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Implements Gaussian differential privacy for vector-based model updates.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This adds carefully calibrated random noise to a parameter vector so that
/// individual data points cannot be inferred from the update, while the overall signal remains useful.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public sealed class GaussianDifferentialPrivacyVector<T> : PrivacyMechanismBase<Vector<T>, T>
{
    private readonly object _sync = new object();
    private double _privacyBudgetConsumed;
    private readonly double _clipNorm;
    private readonly Random _random;

    public GaussianDifferentialPrivacyVector(double clipNorm = 1.0, int? randomSeed = null)
    {
        if (clipNorm <= 0.0)
        {
            throw new ArgumentException("Clip norm must be positive.", nameof(clipNorm));
        }

        _clipNorm = clipNorm;
        _privacyBudgetConsumed = 0.0;
        _random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> ApplyPrivacy(Vector<T> model, double epsilon, double delta)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (epsilon <= 0.0)
        {
            throw new ArgumentException("Epsilon must be positive.", nameof(epsilon));
        }

        if (delta <= 0.0 || delta >= 1.0)
        {
            throw new ArgumentException("Delta must be in (0, 1).", nameof(delta));
        }

        var noisy = new Vector<T>(model.ToArray());

        // Clip by L2 norm.
        var l2Norm = CalculateL2Norm(noisy);
        var clipNormT = NumOps.FromDouble(_clipNorm);
        if (NumOps.GreaterThan(l2Norm, clipNormT))
        {
            var scale = NumOps.Divide(clipNormT, l2Norm);
            for (int i = 0; i < noisy.Length; i++)
            {
                noisy[i] = NumOps.Multiply(noisy[i], scale);
            }
        }

        // Gaussian noise sigma: (clipNorm/epsilon)*sqrt(2*ln(1.25/delta)).
        double sigma = (_clipNorm / epsilon) * Math.Sqrt(2.0 * Math.Log(1.25 / delta));
        lock (_sync)
        {
            for (int i = 0; i < noisy.Length; i++)
            {
                double noise = GenerateGaussianNoise(0.0, sigma);
                noisy[i] = NumOps.Add(noisy[i], NumOps.FromDouble(noise));
            }

            _privacyBudgetConsumed += epsilon;
        }

        return noisy;
    }

    public override double GetPrivacyBudgetConsumed()
    {
        lock (_sync)
        {
            return _privacyBudgetConsumed;
        }
    }

    public override string GetMechanismName() => "Gaussian Mechanism (Vector)";

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
    /// </remarks>
    public void ResetPrivacyBudget()
    {
        lock (_sync)
        {
            _privacyBudgetConsumed = 0.0;
        }
    }

    private T CalculateL2Norm(Vector<T> vector)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Square(vector[i]));
        }

        return NumOps.Sqrt(sum);
    }

    private double GenerateGaussianNoise(double mean, double stdDev)
    {
        // Box-Muller transform.
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}
