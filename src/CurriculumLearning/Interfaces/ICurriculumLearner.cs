using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Results;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning.Interfaces;

/// <summary>
/// Interface for curriculum learning trainers that train models using a structured curriculum.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Curriculum learning is inspired by how humans learn - starting
/// with easy examples and gradually progressing to harder ones. Just like a student learns
/// basic arithmetic before calculus, a model trained with curriculum learning sees simple
/// examples first, then progressively harder ones.</para>
///
/// <para><b>Why Curriculum Learning?</b></para>
/// <list type="bullet">
/// <item><description>Faster convergence: Easy examples help establish basic patterns</description></item>
/// <item><description>Better generalization: Gradual difficulty prevents overfitting</description></item>
/// <item><description>Improved final accuracy: Models learn more robust representations</description></item>
/// <item><description>More stable training: Avoids early exposure to noisy/hard examples</description></item>
/// </list>
///
/// <para><b>Key Components:</b></para>
/// <list type="bullet">
/// <item><description><b>Difficulty Estimator:</b> Measures how hard each sample is</description></item>
/// <item><description><b>Curriculum Scheduler:</b> Decides when to introduce harder samples</description></item>
/// <item><description><b>Curriculum Learner:</b> Orchestrates the training process</description></item>
/// </list>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Bengio et al. "Curriculum Learning" (ICML 2009)</description></item>
/// <item><description>Kumar et al. "Self-Paced Learning for Latent Variable Models" (NIPS 2010)</description></item>
/// <item><description>Soviany et al. "Curriculum Learning: A Survey" (IJCV 2022)</description></item>
/// </list>
/// </remarks>
public interface ICurriculumLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the underlying model being trained.
    /// </summary>
    IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Gets the configuration for the curriculum learner.
    /// </summary>
    ICurriculumLearnerConfig<T> Config { get; }

    /// <summary>
    /// Gets the difficulty estimator used to rank samples.
    /// </summary>
    IDifficultyEstimator<T, TInput, TOutput> DifficultyEstimator { get; }

    /// <summary>
    /// Gets the curriculum scheduler that controls training progression.
    /// </summary>
    ICurriculumScheduler<T> Scheduler { get; }

    /// <summary>
    /// Gets the current training phase (0-1, where 1 means all samples are available).
    /// </summary>
    T CurrentPhase { get; }

    /// <summary>
    /// Gets the current epoch number.
    /// </summary>
    int CurrentEpoch { get; }

    /// <summary>
    /// Gets whether training is currently in progress.
    /// </summary>
    bool IsTraining { get; }

    /// <summary>
    /// Trains the model using curriculum learning.
    /// </summary>
    /// <param name="trainingData">The full training dataset.</param>
    /// <returns>Result containing training metrics and curriculum progression.</returns>
    /// <remarks>
    /// <para>This method will:
    /// 1. Estimate difficulty for all samples
    /// 2. Sort samples by difficulty (easy to hard)
    /// 3. Train in phases, gradually introducing harder samples
    /// 4. Track curriculum progression and model performance
    /// </para>
    /// </remarks>
    CurriculumLearningResult<T> Train(IDataset<T, TInput, TOutput> trainingData);

    /// <summary>
    /// Trains the model with a validation set for monitoring.
    /// </summary>
    /// <param name="trainingData">The full training dataset.</param>
    /// <param name="validationData">The validation dataset.</param>
    /// <returns>Result containing training metrics and curriculum progression.</returns>
    CurriculumLearningResult<T> Train(
        IDataset<T, TInput, TOutput> trainingData,
        IDataset<T, TInput, TOutput> validationData);

    /// <summary>
    /// Trains with pre-computed difficulty scores.
    /// </summary>
    /// <param name="trainingData">The full training dataset.</param>
    /// <param name="difficultyScores">Pre-computed difficulty scores for each sample.</param>
    /// <returns>Result containing training metrics and curriculum progression.</returns>
    /// <remarks>
    /// <para>Use this method when:
    /// - Difficulty scores are computed externally (e.g., by domain experts)
    /// - You want to reuse difficulty scores across training runs
    /// - The difficulty estimation is too expensive to repeat
    /// </para>
    /// </remarks>
    CurriculumLearningResult<T> TrainWithDifficulty(
        IDataset<T, TInput, TOutput> trainingData,
        Vector<T> difficultyScores);

    /// <summary>
    /// Estimates difficulty scores for all samples in a dataset.
    /// </summary>
    /// <param name="dataset">The dataset to estimate difficulties for.</param>
    /// <returns>Vector of difficulty scores (higher = harder).</returns>
    Vector<T> EstimateDifficulties(IDataset<T, TInput, TOutput> dataset);

    /// <summary>
    /// Gets the indices of samples available at the current curriculum phase.
    /// </summary>
    /// <param name="allDifficulties">Difficulty scores for all samples.</param>
    /// <returns>Indices of samples available for training at current phase.</returns>
    int[] GetCurrentCurriculumIndices(Vector<T> allDifficulties);

    /// <summary>
    /// Advances the curriculum to the next phase.
    /// </summary>
    /// <returns>True if advanced, false if already at final phase.</returns>
    bool AdvancePhase();

    /// <summary>
    /// Resets the curriculum to the initial phase.
    /// </summary>
    void ResetCurriculum();

    /// <summary>
    /// Gets the training history.
    /// </summary>
    /// <returns>List of results from each curriculum phase.</returns>
    IReadOnlyList<CurriculumPhaseResult<T>> GetPhaseHistory();

    /// <summary>
    /// Saves the curriculum learner state.
    /// </summary>
    /// <param name="path">Path to save the state.</param>
    void Save(string path);

    /// <summary>
    /// Loads the curriculum learner state.
    /// </summary>
    /// <param name="path">Path to load the state from.</param>
    void Load(string path);

    /// <summary>
    /// Event raised when a curriculum phase starts.
    /// </summary>
    event EventHandler<CurriculumPhaseEventArgs<T>>? PhaseStarted;

    /// <summary>
    /// Event raised when a curriculum phase completes.
    /// </summary>
    event EventHandler<CurriculumPhaseCompletedEventArgs<T>>? PhaseCompleted;

    /// <summary>
    /// Event raised when training completes.
    /// </summary>
    event EventHandler<CurriculumTrainingCompletedEventArgs<T>>? TrainingCompleted;
}

/// <summary>
/// Event arguments for curriculum phase events.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class CurriculumPhaseEventArgs<T> : EventArgs
{
    /// <summary>
    /// Gets the phase number (0-indexed).
    /// </summary>
    public int PhaseNumber { get; }

    /// <summary>
    /// Gets the total number of phases.
    /// </summary>
    public int TotalPhases { get; }

    /// <summary>
    /// Gets the fraction of data available in this phase (0-1).
    /// </summary>
    public T DataFraction { get; }

    /// <summary>
    /// Gets the number of samples available in this phase.
    /// </summary>
    public int SampleCount { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CurriculumPhaseEventArgs{T}"/> class.
    /// </summary>
    public CurriculumPhaseEventArgs(int phaseNumber, int totalPhases, T dataFraction, int sampleCount)
    {
        PhaseNumber = phaseNumber;
        TotalPhases = totalPhases;
        DataFraction = dataFraction;
        SampleCount = sampleCount;
    }
}

/// <summary>
/// Event arguments for curriculum phase completion events.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class CurriculumPhaseCompletedEventArgs<T> : CurriculumPhaseEventArgs<T>
{
    /// <summary>
    /// Gets the phase result.
    /// </summary>
    public CurriculumPhaseResult<T> Result { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CurriculumPhaseCompletedEventArgs{T}"/> class.
    /// </summary>
    public CurriculumPhaseCompletedEventArgs(
        int phaseNumber,
        int totalPhases,
        T dataFraction,
        int sampleCount,
        CurriculumPhaseResult<T> result)
        : base(phaseNumber, totalPhases, dataFraction, sampleCount)
    {
        Result = result;
    }
}

/// <summary>
/// Event arguments for training completion.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class CurriculumTrainingCompletedEventArgs<T> : EventArgs
{
    /// <summary>
    /// Gets the overall training result.
    /// </summary>
    public CurriculumLearningResult<T> Result { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CurriculumTrainingCompletedEventArgs{T}"/> class.
    /// </summary>
    public CurriculumTrainingCompletedEventArgs(CurriculumLearningResult<T> result)
    {
        Result = result;
    }
}
