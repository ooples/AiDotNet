using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Tasks;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Base interface for meta-learning algorithms that learn to adapt quickly to new tasks.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// For Beginners:
/// Meta-learning algorithms learn "how to learn" by training on many tasks.
/// They use a two-loop structure:
/// - Inner loop: Fast adaptation to a specific task using the support set
/// - Outer loop: Meta-optimization to improve the adaptation process across tasks
///
/// Common meta-learning algorithms include:
/// - MAML (Model-Agnostic Meta-Learning): Uses gradients for both loops
/// - Reptile: Simpler version of MAML with combined gradients
/// - SEAL: Self-adapting meta-learner with adaptive mechanisms
/// </remarks>
public interface IMetaLearner<T> where T : struct
{
    /// <summary>
    /// Gets the base model used for meta-learning.
    /// </summary>
    INeuralNetwork<T> BaseModel { get; }

    /// <summary>
    /// Gets the meta-learner configuration.
    /// </summary>
    IMetaLearnerConfig<T> Config { get; }

    /// <summary>
    /// Performs one meta-training step on a batch of episodes.
    /// </summary>
    /// <param name="episodes">Batch of training episodes</param>
    /// <returns>Training metrics (loss, accuracy, etc.)</returns>
    MetaTrainingMetrics MetaTrainStep(IEpisode<T>[] episodes);

    /// <summary>
    /// Evaluates the meta-learner on a batch of episodes.
    /// </summary>
    /// <param name="episodes">Batch of evaluation episodes</param>
    /// <returns>Evaluation metrics</returns>
    MetaEvaluationMetrics Evaluate(IEpisode<T>[] episodes);

    /// <summary>
    /// Adapts the model to a specific task using the support set,
    /// then evaluates on the query set.
    /// </summary>
    /// <param name="episode">The episode to adapt to and evaluate on</param>
    /// <returns>Adaptation metrics</returns>
    AdaptationMetrics AdaptAndEvaluate(IEpisode<T> episode);

    /// <summary>
    /// Saves the meta-learned model to a file.
    /// </summary>
    /// <param name="path">Path to save the model</param>
    void Save(string path);

    /// <summary>
    /// Loads a meta-learned model from a file.
    /// </summary>
    /// <param name="path">Path to load the model from</param>
    void Load(string path);

    /// <summary>
    /// Gets the current meta-training iteration.
    /// </summary>
    int CurrentIteration { get; }

    /// <summary>
    /// Resets the meta-learner to its initial state.
    /// </summary>
    void Reset();
}

/// <summary>
/// Metrics collected during meta-training.
/// </summary>
public class MetaTrainingMetrics
{
    /// <summary>
    /// Meta-training loss (outer loop loss).
    /// </summary>
    public double MetaLoss { get; set; }

    /// <summary>
    /// Average task loss across episodes (inner loop loss).
    /// </summary>
    public double TaskLoss { get; set; }

    /// <summary>
    /// Average accuracy on query sets after adaptation.
    /// </summary>
    public double Accuracy { get; set; }

    /// <summary>
    /// Number of episodes in the batch.
    /// </summary>
    public int NumEpisodes { get; set; }

    /// <summary>
    /// Additional metrics specific to the algorithm.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}

/// <summary>
/// Metrics collected during meta-evaluation.
/// </summary>
public class MetaEvaluationMetrics
{
    /// <summary>
    /// Average accuracy on query sets after adaptation.
    /// </summary>
    public double Accuracy { get; set; }

    /// <summary>
    /// Standard deviation of accuracy across episodes.
    /// </summary>
    public double AccuracyStd { get; set; }

    /// <summary>
    /// 95% confidence interval for accuracy.
    /// </summary>
    public (double Lower, double Upper) ConfidenceInterval { get; set; }

    /// <summary>
    /// Average loss on query sets.
    /// </summary>
    public double Loss { get; set; }

    /// <summary>
    /// Number of episodes evaluated.
    /// </summary>
    public int NumEpisodes { get; set; }

    /// <summary>
    /// Additional metrics specific to the algorithm.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}

/// <summary>
/// Metrics collected during adaptation to a single task.
/// </summary>
public class AdaptationMetrics
{
    /// <summary>
    /// Query set accuracy after adaptation.
    /// </summary>
    public double QueryAccuracy { get; set; }

    /// <summary>
    /// Query set loss after adaptation.
    /// </summary>
    public double QueryLoss { get; set; }

    /// <summary>
    /// Support set accuracy after adaptation (should be high).
    /// </summary>
    public double SupportAccuracy { get; set; }

    /// <summary>
    /// Number of adaptation steps performed.
    /// </summary>
    public int AdaptationSteps { get; set; }

    /// <summary>
    /// Time taken for adaptation in milliseconds.
    /// </summary>
    public double AdaptationTimeMs { get; set; }

    /// <summary>
    /// Additional metrics specific to the algorithm.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}
