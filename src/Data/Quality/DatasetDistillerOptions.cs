namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for dataset distillation.
/// </summary>
/// <remarks>
/// Dataset distillation synthesizes a small set of examples that capture the essence
/// of the full training set. Training on the distilled set produces similar model quality
/// to training on the full dataset.
/// </remarks>
public sealed class DatasetDistillerOptions
{
    /// <summary>Number of synthetic samples per class to generate. Default is 10.</summary>
    public int SamplesPerClass { get; set; } = 10;
    /// <summary>Learning rate for optimizing distilled samples. Default is 0.01.</summary>
    public double DistillLearningRate { get; set; } = 0.01;
    /// <summary>Number of distillation optimization steps. Default is 1000.</summary>
    public int NumSteps { get; set; } = 1000;
    /// <summary>Random seed for reproducibility. Default is null (random).</summary>
    public int? Seed { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SamplesPerClass <= 0) throw new ArgumentOutOfRangeException(nameof(SamplesPerClass), "SamplesPerClass must be positive.");
        if (DistillLearningRate <= 0 || double.IsNaN(DistillLearningRate) || double.IsInfinity(DistillLearningRate))
            throw new ArgumentOutOfRangeException(nameof(DistillLearningRate), "DistillLearningRate must be a positive finite number.");
        if (NumSteps <= 0) throw new ArgumentOutOfRangeException(nameof(NumSteps), "NumSteps must be positive.");
    }
}
