namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Result from fine-tuning an SSL pretrained encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
public class FineTuningResult<T>
{
    /// <summary>Training accuracy per epoch.</summary>
    public List<double> TrainAccuracies { get; set; } = [];

    /// <summary>Validation accuracy per epoch.</summary>
    public List<double> ValidAccuracies { get; set; } = [];

    /// <summary>Final training accuracy.</summary>
    public double FinalTrainAccuracy { get; set; }

    /// <summary>Final validation accuracy.</summary>
    public double FinalValidAccuracy { get; set; }

    /// <summary>Best validation accuracy achieved.</summary>
    public double BestValidAccuracy { get; set; }

    /// <summary>Epoch where best validation accuracy was achieved.</summary>
    public int BestEpoch { get; set; }
}
