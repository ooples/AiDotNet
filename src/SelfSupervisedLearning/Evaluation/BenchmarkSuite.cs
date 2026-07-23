namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Complete benchmark suite results.
/// </summary>
public class BenchmarkSuite<T>
{
    /// <summary>
    /// Name of the dataset.
    /// </summary>
    public string DatasetName { get; set; } = "";

    /// <summary>
    /// Number of classes.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Number of training samples.
    /// </summary>
    public int NumTrainSamples { get; set; }

    /// <summary>
    /// Number of test samples.
    /// </summary>
    public int NumTestSamples { get; set; }

    /// <summary>
    /// Linear probing result.
    /// </summary>
    public BenchmarkResult<T>? LinearProbingResult { get; set; }

    /// <summary>
    /// k-NN evaluation result.
    /// </summary>
    public BenchmarkResult<T>? KNNResult { get; set; }

    /// <summary>
    /// Few-shot evaluation results by percentage.
    /// </summary>
    public Dictionary<double, BenchmarkResult<T>> FewShotResults { get; set; } = [];
}
