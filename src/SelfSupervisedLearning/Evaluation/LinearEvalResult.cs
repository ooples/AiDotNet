namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Results from linear evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
public class LinearEvalResult<T>
{
    /// <summary>
    /// Training accuracy per epoch.
    /// </summary>
    public List<double> TrainAccuracies { get; set; } = [];

    /// <summary>
    /// Validation accuracy per epoch.
    /// </summary>
    public List<double> ValidAccuracies { get; set; } = [];

    /// <summary>
    /// Final training accuracy.
    /// </summary>
    public double FinalTrainAccuracy { get; set; }

    /// <summary>
    /// Final validation accuracy.
    /// </summary>
    public double FinalValidAccuracy { get; set; }

    /// <summary>
    /// Best validation accuracy achieved.
    /// </summary>
    public double BestValidAccuracy => ValidAccuracies.Count > 0 ? ValidAccuracies.Max() : 0;
}
