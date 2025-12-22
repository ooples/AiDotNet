using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for the synthetic federated tabular benchmark suite.
/// </summary>
/// <remarks>
/// <para>
/// This suite generates a deterministic synthetic dataset with non-IID client distributions, allowing
/// benchmark runs without external dataset dependencies.
/// </para>
/// <para><b>For Beginners:</b> Synthetic means AiDotNet generates the data automatically instead of reading
/// files from disk. This is useful for CI and quick sanity checks.
/// </para>
/// </remarks>
public sealed class SyntheticTabularFederatedBenchmarkOptions
{
    /// <summary>
    /// Gets or sets the number of federated clients to simulate (null uses defaults).
    /// </summary>
    public int? ClientCount { get; set; }

    /// <summary>
    /// Gets or sets the number of input features per sample (null uses defaults).
    /// </summary>
    public int? FeatureCount { get; set; }

    /// <summary>
    /// Gets or sets the number of training samples per client (null uses defaults).
    /// </summary>
    public int? TrainSamplesPerClient { get; set; }

    /// <summary>
    /// Gets or sets the number of test samples per client (null uses defaults).
    /// </summary>
    public int? TestSamplesPerClient { get; set; }

    /// <summary>
    /// Gets or sets the number of classes for classification tasks (null uses defaults).
    /// </summary>
    public int? ClassCount { get; set; }

    /// <summary>
    /// Gets or sets the synthetic task type.
    /// </summary>
    public SyntheticTabularTaskType TaskType { get; set; } = SyntheticTabularTaskType.MultiClassClassification;

    /// <summary>
    /// Gets or sets the Dirichlet concentration parameter controlling label skew across clients.
    /// </summary>
    /// <remarks>
    /// Smaller values create more heterogeneous (more non-IID) client label distributions.
    /// </remarks>
    public double DirichletAlpha { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the noise standard deviation applied to generated targets/scores.
    /// </summary>
    public double NoiseStdDev { get; set; } = 0.1;
}

