namespace AiDotNet.SelfSupervisedLearning.Core;

/// <summary>
/// Training history from SSL pretraining.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
public class SSLTrainingHistory<T>
{
    /// <summary>
    /// Loss values per epoch.
    /// </summary>
    public List<T> LossHistory { get; set; } = [];

    /// <summary>
    /// Representation standard deviation per epoch (for collapse detection).
    /// </summary>
    public List<T> StdHistory { get; set; } = [];

    /// <summary>
    /// k-NN accuracy per epoch (if computed).
    /// </summary>
    public List<double> KNNHistory { get; set; } = [];

    /// <summary>
    /// Learning rate per epoch.
    /// </summary>
    public List<double> LearningRateHistory { get; set; } = [];

    /// <summary>
    /// Momentum value per epoch (for methods with momentum encoder).
    /// </summary>
    public List<double> MomentumHistory { get; set; } = [];

    /// <summary>
    /// Custom metrics per epoch.
    /// </summary>
    public Dictionary<string, List<T>> CustomMetrics { get; set; } = [];

    /// <summary>
    /// Adds metrics from a training step.
    /// </summary>
    public void AddEpochMetrics(T loss, T std, double knnAcc, double lr, double momentum)
    {
        LossHistory.Add(loss);
        StdHistory.Add(std);
        KNNHistory.Add(knnAcc);
        LearningRateHistory.Add(lr);
        MomentumHistory.Add(momentum);
    }

    /// <summary>
    /// Adds a custom metric value.
    /// </summary>
    public void AddCustomMetric(string name, T value)
    {
        if (!CustomMetrics.TryGetValue(name, out List<T>? metricList))
        {
            metricList = [];
            CustomMetrics[name] = metricList;
        }
        metricList.Add(value);
    }
}
