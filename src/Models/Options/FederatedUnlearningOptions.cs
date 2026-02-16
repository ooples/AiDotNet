namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for federated unlearning (right to be forgotten).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how the system removes a client's contribution
/// from the trained model when they exercise their right to be forgotten (GDPR, CCPA, LGPD).</para>
/// </remarks>
public class FederatedUnlearningOptions
{
    /// <summary>
    /// Gets or sets the unlearning method. Default is GradientAscent (fast, approximate).
    /// </summary>
    public UnlearningMethod Method { get; set; } = UnlearningMethod.GradientAscent;

    /// <summary>
    /// Gets or sets whether to verify unlearning correctness. Default is true.
    /// </summary>
    public bool VerificationEnabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum epochs for approximate unlearning methods. Default is 10.
    /// </summary>
    public int MaxUnlearningEpochs { get; set; } = 10;

    /// <summary>
    /// Gets or sets the learning rate for gradient ascent unlearning. Default is 0.01.
    /// </summary>
    public double UnlearningLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the noise scale for diffusive noise unlearning. Default is 0.1.
    /// </summary>
    public double NoiseScale { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the tolerance for influence function convergence. Default is 1e-4.
    /// </summary>
    public double InfluenceTolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the maximum number of iterations for influence function computation.
    /// Default is 100.
    /// </summary>
    public int MaxInfluenceIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the verification threshold: maximum cosine similarity between the unlearned
    /// model and a model trained with the target client that is considered acceptable.
    /// Default is 0.95 (lower = stricter verification).
    /// </summary>
    public double VerificationThreshold { get; set; } = 0.95;
}
