using AiDotNet.ActiveLearning.Config;
using AiDotNet.ActiveLearning.Results;

namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for active learners.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> An active learner is a machine learning system that
/// actively selects which data points should be labeled. Instead of labeling all data,
/// it strategically chooses the most informative samples to learn from.</para>
///
/// <para><b>Active Learning Workflow:</b></para>
/// <list type="number">
/// <item><description>Start with a small labeled dataset and large unlabeled pool</description></item>
/// <item><description>Train the model on labeled data</description></item>
/// <item><description>Use a query strategy to select informative unlabeled samples</description></item>
/// <item><description>Get labels for selected samples (from oracle/human)</description></item>
/// <item><description>Add newly labeled samples to training set</description></item>
/// <item><description>Repeat until stopping criterion is met</description></item>
/// </list>
///
/// <para><b>When to Use Active Learning:</b></para>
/// <list type="bullet">
/// <item><description>Labeling is expensive (requires human experts)</description></item>
/// <item><description>Large amounts of unlabeled data available</description></item>
/// <item><description>Model can benefit from strategic sample selection</description></item>
/// </list>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ActiveLearner")]
public interface IActiveLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the configuration for this active learner.
    /// </summary>
    ActiveLearnerConfig<T> Configuration { get; }

    /// <summary>
    /// Gets the underlying model being trained.
    /// </summary>
    IFullModel<T, TInput, TOutput> Model { get; }

    /// <summary>
    /// Gets the query strategy used for sample selection.
    /// </summary>
    IQueryStrategy<T, TInput, TOutput> QueryStrategy { get; }

    /// <summary>
    /// Gets the current labeled dataset.
    /// </summary>
    IDataset<T, TInput, TOutput> LabeledPool { get; }

    /// <summary>
    /// Gets the current unlabeled pool.
    /// </summary>
    IDataset<T, TInput, TOutput> UnlabeledPool { get; }

    /// <summary>
    /// Gets the number of queries (labeling requests) made so far.
    /// </summary>
    int TotalQueries { get; }

    /// <summary>
    /// Gets the number of active learning iterations completed.
    /// </summary>
    int IterationsCompleted { get; }

    /// <summary>
    /// Initializes the active learner with labeled and unlabeled data.
    /// </summary>
    /// <param name="initialLabeled">Initial labeled dataset.</param>
    /// <param name="unlabeledPool">Pool of unlabeled samples.</param>
    void Initialize(
        IDataset<T, TInput, TOutput> initialLabeled,
        IDataset<T, TInput, TOutput> unlabeledPool);

    /// <summary>
    /// Runs a single iteration of active learning.
    /// </summary>
    /// <param name="oracle">The oracle that provides labels for selected samples.</param>
    /// <returns>Result of this iteration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> One iteration involves:
    /// 1. Training on current labeled data
    /// 2. Selecting samples to query
    /// 3. Getting labels from the oracle
    /// 4. Adding newly labeled samples to training set</para>
    /// </remarks>
    ActiveLearningIterationResult<T> RunIteration(IOracle<TInput, TOutput> oracle);

    /// <summary>
    /// Runs active learning until a stopping criterion is met.
    /// </summary>
    /// <param name="oracle">The oracle that provides labels.</param>
    /// <param name="stoppingCriterion">Optional custom stopping criterion.</param>
    /// <returns>Final result of the active learning process.</returns>
    ActiveLearningResult<T> Run(
        IOracle<TInput, TOutput> oracle,
        IStoppingCriterion<T>? stoppingCriterion = null);

    /// <summary>
    /// Selects the next batch of samples to query.
    /// </summary>
    /// <returns>Indices of samples to query from the unlabeled pool.</returns>
    int[] SelectNextBatch();

    /// <summary>
    /// Adds newly labeled samples to the training set.
    /// </summary>
    /// <param name="indices">Indices of labeled samples in the unlabeled pool.</param>
    /// <param name="labels">Labels for the samples.</param>
    void AddLabeledSamples(int[] indices, TOutput[] labels);

    /// <summary>
    /// Trains the model on the current labeled dataset.
    /// </summary>
    /// <returns>Training metrics.</returns>
    TrainingMetrics<T> TrainModel();

    /// <summary>
    /// Evaluates the model on a test dataset.
    /// </summary>
    /// <param name="testData">The test dataset.</param>
    /// <returns>Evaluation metrics.</returns>
    EvaluationMetrics<T> Evaluate(IDataset<T, TInput, TOutput> testData);

    /// <summary>
    /// Gets the learning curve (performance vs. number of labeled samples).
    /// </summary>
    /// <returns>The learning curve data.</returns>
    LearningCurve<T> GetLearningCurve();

    /// <summary>
    /// Event raised when an iteration is completed.
    /// </summary>
    event EventHandler<ActiveLearningIterationResult<T>>? IterationCompleted;

    /// <summary>
    /// Event raised when samples are selected for labeling.
    /// </summary>
    event EventHandler<SamplesSelectedEventArgs<TInput>>? SamplesSelected;

    /// <summary>
    /// Event raised when the learning process is complete.
    /// </summary>
    event EventHandler<ActiveLearningResult<T>>? LearningCompleted;
}

/// <summary>
/// Interface for oracles (labeling providers) in active learning.
/// </summary>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output (label) data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> An oracle is the source of labels for unlabeled data.
/// In real applications, this is typically a human expert. In experiments, it can be
/// a simulator using ground-truth labels.</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Oracle")]
public interface IOracle<TInput, TOutput>
{
    /// <summary>
    /// Provides a label for a single sample.
    /// </summary>
    /// <param name="input">The input sample to label.</param>
    /// <returns>The label for the sample.</returns>
    TOutput Label(TInput input);

    /// <summary>
    /// Provides labels for a batch of samples.
    /// </summary>
    /// <param name="inputs">The input samples to label.</param>
    /// <returns>Labels for all samples.</returns>
    TOutput[] LabelBatch(TInput[] inputs);

    /// <summary>
    /// Gets the cost of labeling a sample.
    /// </summary>
    /// <param name="input">The sample to potentially label.</param>
    /// <returns>The cost (e.g., time, money) of labeling this sample.</returns>
    T GetLabelingCost<T>(TInput input);
}

/// <summary>
/// Event arguments for when samples are selected for labeling.
/// </summary>
/// <typeparam name="TInput">The input data type.</typeparam>
public class SamplesSelectedEventArgs<TInput> : EventArgs
{
    /// <summary>
    /// Gets the indices of selected samples in the unlabeled pool.
    /// </summary>
    public int[] SelectedIndices { get; }

    /// <summary>
    /// Gets the selected samples.
    /// </summary>
    public TInput[] SelectedSamples { get; }

    /// <summary>
    /// Gets the informativeness scores of selected samples.
    /// </summary>
    public double[] InformativenessScores { get; }

    /// <summary>
    /// Initializes a new instance of the SamplesSelectedEventArgs class.
    /// </summary>
    /// <param name="selectedIndices">Indices of selected samples.</param>
    /// <param name="selectedSamples">The selected samples.</param>
    /// <param name="scores">Informativeness scores.</param>
    public SamplesSelectedEventArgs(int[] selectedIndices, TInput[] selectedSamples, double[] scores)
    {
        SelectedIndices = selectedIndices;
        SelectedSamples = selectedSamples;
        InformativenessScores = scores;
    }
}

/// <summary>
/// Training metrics from a model training iteration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TrainingMetrics<T>
{
    /// <summary>
    /// Gets or sets the training loss.
    /// </summary>
    public T Loss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the training accuracy.
    /// </summary>
    public T Accuracy { get; set; } = default!;

    /// <summary>
    /// Gets or sets the number of epochs trained.
    /// </summary>
    public int EpochsTrained { get; set; }

    /// <summary>
    /// Gets or sets the training time.
    /// </summary>
    public TimeSpan TrainingTime { get; set; }
}

/// <summary>
/// Evaluation metrics from model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EvaluationMetrics<T>
{
    /// <summary>
    /// Gets or sets the evaluation loss.
    /// </summary>
    public T Loss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the evaluation accuracy.
    /// </summary>
    public T Accuracy { get; set; } = default!;

    /// <summary>
    /// Gets or sets the precision (for classification).
    /// </summary>
    public T? Precision { get; set; }

    /// <summary>
    /// Gets or sets the recall (for classification).
    /// </summary>
    public T? Recall { get; set; }

    /// <summary>
    /// Gets or sets the F1 score (for classification).
    /// </summary>
    public T? F1Score { get; set; }

    /// <summary>
    /// Gets or sets the AUC-ROC (for binary classification).
    /// </summary>
    public T? AucRoc { get; set; }
}

/// <summary>
/// Represents a learning curve showing performance vs. number of labeled samples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LearningCurve<T>
{
    /// <summary>
    /// Gets or sets the number of labeled samples at each point.
    /// </summary>
    public int[] SampleCounts { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the accuracy at each point.
    /// </summary>
    public T[] Accuracies { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the loss at each point.
    /// </summary>
    public T[] Losses { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the validation accuracy at each point (if available).
    /// </summary>
    public T[]? ValidationAccuracies { get; set; }

    /// <summary>
    /// Gets or sets the validation loss at each point (if available).
    /// </summary>
    public T[]? ValidationLosses { get; set; }

    /// <summary>
    /// Gets the area under the learning curve (AULC).
    /// </summary>
    public T? AreaUnderCurve { get; set; }
}
