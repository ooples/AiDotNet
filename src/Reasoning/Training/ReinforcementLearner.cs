using System.IO;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Strategies;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning.Training;

/// <summary>
/// Orchestrates reinforcement learning training for reasoning models.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is the complete training pipeline that combines all RL
/// components to train reasoning models like ChatGPT o1/o3.
///
/// **What is ReinforcementLearner?**
/// The master controller that orchestrates the entire RL training process:
/// 1. Generate reasoning chains
/// 2. Evaluate with reward models
/// 3. Collect training data
/// 4. Update the policy
/// 5. Monitor progress
/// 6. Iterate until convergence
///
/// **Complete training pipeline:**
///
/// ```
/// ┌─────────────────────────────────────────┐
/// │  1. Problem Sampling                     │
/// │     - Select from training set           │
/// │     - Curriculum learning (easy → hard)  │
/// └─────────┬───────────────────────────────┘
///           │
///           ▼
/// ┌─────────────────────────────────────────┐
/// │  2. Chain Generation                     │
/// │     - Model generates reasoning chains   │
/// │     - Multiple samples per problem       │
/// └─────────┬───────────────────────────────┘
///           │
///           ▼
/// ┌─────────────────────────────────────────┐
/// │  3. Reward Calculation                   │
/// │     - PRM: Step-by-step rewards          │
/// │     - ORM: Final answer rewards          │
/// │     - Hybrid: Combined rewards           │
/// └─────────┬───────────────────────────────┘
///           │
///           ▼
/// ┌─────────────────────────────────────────┐
/// │  4. Data Collection                      │
/// │     - Store chains with rewards          │
/// │     - Filter low-quality samples         │
/// │     - Balance dataset                    │
/// └─────────┬───────────────────────────────┘
///           │
///           ▼
/// ┌─────────────────────────────────────────┐
/// │  5. Policy Update                        │
/// │     - Calculate gradients                │
/// │     - Update model parameters            │
/// │     - Apply regularization               │
/// └─────────┬───────────────────────────────┘
///           │
///           ▼
/// ┌─────────────────────────────────────────┐
/// │  6. Evaluation                           │
/// │     - Test on validation set             │
/// │     - Calculate metrics                  │
/// │     - Early stopping check               │
/// └─────────┬───────────────────────────────┘
///           │
///           ▼
///       Converged? ──No──┐
///           │            │
///          Yes           │
///           │            │
///           ▼            │
///     Save Model    ─────┘
/// ```
///
/// **Training strategies:**
///
/// *1. Standard RL:*
/// Generate chains → Evaluate → Update → Repeat
///
/// *2. Self-Taught Reasoner (STaR):*
/// Generate many chains → Keep only correct ones → Train on correct chains → Repeat
///
/// *3. Iterative refinement:*
/// Train → Evaluate → Collect failures → Train on failures → Repeat
///
/// *4. Curriculum learning:*
/// Start with easy problems → Gradually increase difficulty
///
/// *5. Best-of-N sampling:*
/// Generate N chains → Pick best → Train on best
///
/// **Example usage:**
///
/// ```csharp
/// // Setup
/// var chatModel = /* your chat model */;
/// var prm = new ProcessRewardModel<double>(chatModel);
/// var orm = new OutcomeRewardModel<double>(chatModel);
/// var rewardModel = new HybridRewardModel<double>(prm, orm, 0.5, 0.5);
///
/// var learner = new ReinforcementLearner<double>(
///     chatModel,
///     rewardModel,
///     config: new RLConfig
///     {
///         Epochs = 10,
///         BatchSize = 32,
///         LearningRate = 0.0001,
///         ValidationFrequency = 100
///     }
/// );
///
/// // Training data
/// var trainingProblems = await LoadProblemsAsync("gsm8k_train.json");
/// var validationProblems = await LoadProblemsAsync("gsm8k_val.json");
///
/// // Train
/// var results = await learner.TrainAsync(
///     trainingProblems,
///     validationProblems
/// );
///
/// Console.WriteLine($"Final accuracy: {results.BestAccuracy:P2}");
/// Console.WriteLine($"Epochs trained: {results.EpochsTrained}");
///
/// // Save trained model
/// await learner.SaveCheckpointAsync("model_checkpoint.bin");
/// ```
///
/// **Progress monitoring:**
/// ```csharp
/// learner.OnEpochComplete += (sender, metrics) =>
/// {
///     Console.WriteLine($"Epoch {metrics.Epoch}:");
///     Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
///     Console.WriteLine($"  Average Reward: {metrics.AverageReward:F3}");
///     Console.WriteLine($"  Loss: {metrics.Loss:F4}");
/// };
///
/// learner.OnBatchComplete += (sender, progress) =>
/// {
///     Console.WriteLine($"Batch {progress.BatchNumber}/{progress.TotalBatches}");
/// };
/// ```
///
/// **Hyperparameter tuning:**
/// - Learning rate: 0.0001 - 0.001 (smaller = stable, larger = fast)
/// - Batch size: 16 - 128 (larger = stable, smaller = diverse)
/// - Discount factor: 0.95 - 0.99 (how much to value future rewards)
/// - Entropy coefficient: 0.01 - 0.1 (exploration vs exploitation)
///
/// **Best practices:**
/// 1. Start with supervised pre-training
/// 2. Use curriculum learning (easy → hard)
/// 3. Monitor validation accuracy (early stopping)
/// 4. Save checkpoints frequently
/// 5. Use diverse training data
/// 6. Balance exploration and exploitation
/// 7. Gradually decrease learning rate
///
/// **Research:**
/// - "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
/// - "Let's Verify Step by Step" (Lightman et al., 2023)
/// - "Self-Taught Reasoner (STaR)" (Zelikman et al., 2022)
/// - "Proximal Policy Optimization" (Schulman et al., 2017)
/// </para>
/// </remarks>
internal class ReinforcementLearner<T>
{
    private readonly IChatModel<T> _model;
    private readonly IRewardModel<T> _rewardModel;
    private readonly PolicyGradientTrainer<T> _trainer;
    private readonly TrainingDataCollector<T> _dataCollector;
    private readonly RLConfig _config;
    private readonly IReasoningStrategy<T> _reasoningStrategy;

    // Training state for checkpoint/resume functionality
    private int _currentEpoch;
    private double _bestAccuracy;
    private int _bestEpoch;
    private int _epochsWithoutImprovement;

    /// <summary>
    /// Event raised when an epoch completes.
    /// </summary>
    public event EventHandler<EpochMetrics<T>>? OnEpochComplete;

    /// <summary>
    /// Event raised when a batch completes.
    /// </summary>
    public event EventHandler<BatchProgress>? OnBatchComplete;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReinforcementLearner{T}"/> class.
    /// </summary>
    public ReinforcementLearner(
        IChatModel<T> model,
        IRewardModel<T> rewardModel,
        RLConfig? config = null)
    {
        Guard.NotNull(model);
        _model = model;
        Guard.NotNull(rewardModel);
        _rewardModel = rewardModel;
        _config = config ?? RLConfig.Default;

        _trainer = new PolicyGradientTrainer<T>(
            _model,
            _rewardModel,
            learningRate: _config.LearningRate,
            discountFactor: _config.DiscountFactor,
            entropyCoefficient: _config.EntropyCoefficient
        );

        _dataCollector = new TrainingDataCollector<T>();
        _reasoningStrategy = new ChainOfThoughtStrategy<T>(_model);
    }

    /// <summary>
    /// Trains the model using reinforcement learning.
    /// </summary>
    public async Task<TrainingResults<T>> TrainAsync(
        List<(string problem, string answer)> trainingSet,
        List<(string problem, string answer)> validationSet,
        CancellationToken cancellationToken = default)
    {
        var results = new TrainingResults<T>();

        for (int epoch = _currentEpoch; epoch < _config.Epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _currentEpoch = epoch;

            Console.WriteLine($"\n=== Epoch {epoch + 1}/{_config.Epochs} ===");

            // Training phase
            var epochMetrics = await TrainEpochAsync(trainingSet, epoch, cancellationToken);

            // Validation phase
            if ((epoch + 1) % _config.ValidationFrequency == 0)
            {
                var evalMetrics = await EvaluateAsync(validationSet, cancellationToken);

                epochMetrics.Accuracy = evalMetrics.Accuracy;
                epochMetrics.ValidationReward = evalMetrics.AverageReward;

                Console.WriteLine($"Validation Accuracy: {evalMetrics.Accuracy:P2}");

                // Track best model
                if (evalMetrics.Accuracy > _bestAccuracy)
                {
                    _bestAccuracy = evalMetrics.Accuracy;
                    _bestEpoch = epoch + 1;
                    results.BestAccuracy = _bestAccuracy;
                    results.BestEpoch = _bestEpoch;
                    _epochsWithoutImprovement = 0;

                    // Save checkpoint
                    if (_config.SaveCheckpoints)
                    {
                        await SaveCheckpointAsync($"checkpoint_epoch_{epoch + 1}.bin", cancellationToken);
                    }
                }
                else
                {
                    _epochsWithoutImprovement++;
                }

                // Early stopping
                if (_epochsWithoutImprovement >= _config.EarlyStoppingPatience)
                {
                    Console.WriteLine($"Early stopping at epoch {epoch + 1}");
                    break;
                }
            }

            // Fire event
            OnEpochComplete?.Invoke(this, epochMetrics);
            results.EpochMetrics.Add(epochMetrics);
        }

        results.EpochsTrained = results.EpochMetrics.Count;
        results.DataCollector = _dataCollector;

        return results;
    }

    /// <summary>
    /// Trains using Self-Taught Reasoner (STaR) approach.
    /// </summary>
    public async Task<TrainingResults<T>> TrainSTaRAsync(
        List<(string problem, string answer)> trainingSet,
        List<(string problem, string answer)> validationSet,
        int samplesPerProblem = 5,
        CancellationToken cancellationToken = default)
    {
        var results = new TrainingResults<T>();
        double bestAccuracy = 0.0;

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            Console.WriteLine($"\n=== STaR Epoch {epoch + 1}/{_config.Epochs} ===");

            var problems = trainingSet.Select(t => t.problem).ToList();
            var answers = trainingSet.Select(t => t.answer).ToList();

            // Train with STaR
            var trainingMetrics = await _trainer.TrainSTaRAsync(
                problems,
                answers,
                async (problem) => await GenerateMultipleChainsAsync(problem, samplesPerProblem, cancellationToken),
                samplesPerProblem,
                cancellationToken
            );

            Console.WriteLine($"STaR Chains Collected: {trainingMetrics.ChainCount}");
            Console.WriteLine($"Average Reward: {Convert.ToDouble(trainingMetrics.AverageReward):F3}");

            // Validate
            if ((epoch + 1) % _config.ValidationFrequency == 0)
            {
                var evalMetrics = await EvaluateAsync(validationSet, cancellationToken);

                if (evalMetrics.Accuracy > bestAccuracy)
                {
                    bestAccuracy = evalMetrics.Accuracy;
                    results.BestAccuracy = bestAccuracy;
                    results.BestEpoch = epoch + 1;
                }

                Console.WriteLine($"Validation Accuracy: {evalMetrics.Accuracy:P2}");
            }
        }

        results.EpochsTrained = _config.Epochs;
        return results;
    }

    private async Task<EpochMetrics<T>> TrainEpochAsync(
        List<(string problem, string answer)> trainingSet,
        int epoch,
        CancellationToken cancellationToken)
    {
        // Shuffle training set
        var shuffled = trainingSet.OrderBy(_ => Guid.NewGuid()).ToList();

        // Create batches
        var batches = new List<List<(string problem, string answer)>>();
        for (int i = 0; i < shuffled.Count; i += _config.BatchSize)
        {
            batches.Add(shuffled.Skip(i).Take(_config.BatchSize).ToList());
        }

        var epochMetrics = new EpochMetrics<T> { Epoch = epoch + 1 };
        var allTrainingMetrics = new List<TrainingMetrics<T>>();

        // Train on each batch
        for (int batchIdx = 0; batchIdx < batches.Count; batchIdx++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var batch = batches[batchIdx];

            // Generate chains for batch
            var chains = new List<ReasoningChain<T>>();
            var answers = new List<string>();

            foreach (var (problem, answer) in batch)
            {
                var result = await _reasoningStrategy.ReasonAsync(problem, cancellationToken: cancellationToken);

                if (result.ReasoningChain != null)
                {
                    chains.Add(result.ReasoningChain);
                    answers.Add(answer);

                    // Collect training sample
                    var reward = await _rewardModel.CalculateChainRewardAsync(result.ReasoningChain, answer, cancellationToken);
                    var outcomeReward = await _rewardModel.CalculateChainRewardAsync(result.ReasoningChain, answer, cancellationToken);

                    _dataCollector.AddSample(new TrainingSample<T>
                    {
                        Problem = problem,
                        ReasoningChain = result.ReasoningChain,
                        CorrectAnswer = answer,
                        ChainReward = reward,
                        OutcomeReward = outcomeReward
                    });
                }
            }

            // Train on batch
            if (chains.Count > 0)
            {
                var batchMetrics = await _trainer.TrainBatchAsync(chains, answers, cancellationToken);
                allTrainingMetrics.Add(batchMetrics);
            }

            // Progress callback
            OnBatchComplete?.Invoke(this, new BatchProgress
            {
                BatchNumber = batchIdx + 1,
                TotalBatches = batches.Count,
                Epoch = epoch + 1
            });

            if ((batchIdx + 1) % 10 == 0)
            {
                Console.WriteLine($"  Batch {batchIdx + 1}/{batches.Count}");
            }
        }

        // Aggregate epoch metrics
        if (allTrainingMetrics.Count > 0)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            epochMetrics.AverageReward = numOps.FromDouble(
                allTrainingMetrics.Average(m => Convert.ToDouble(m.AverageReward))
            );
            epochMetrics.AverageLoss = numOps.FromDouble(
                allTrainingMetrics.Average(m => Convert.ToDouble(m.TotalLoss))
            );
        }

        return epochMetrics;
    }

    private async Task<EvaluationMetrics<T>> EvaluateAsync(
        List<(string problem, string answer)> validationSet,
        CancellationToken cancellationToken)
    {
        return await _trainer.EvaluateAsync(
            validationSet,
            async (problem) =>
            {
                var result = await _reasoningStrategy.ReasonAsync(problem, cancellationToken: cancellationToken);
                return result.ReasoningChain ?? new ReasoningChain<T>();
            },
            cancellationToken
        );
    }

    private async Task<List<ReasoningChain<T>>> GenerateMultipleChainsAsync(
        string problem,
        int count,
        CancellationToken cancellationToken)
    {
        var chains = new List<ReasoningChain<T>>();

        for (int i = 0; i < count; i++)
        {
            var result = await _reasoningStrategy.ReasonAsync(problem, cancellationToken: cancellationToken);
            if (result.ReasoningChain != null)
            {
                chains.Add(result.ReasoningChain);
            }
        }

        return chains;
    }

    /// <summary>
    /// Saves a training checkpoint including training state, configuration, and collected data.
    /// </summary>
    /// <param name="filePath">Path to save the checkpoint file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <remarks>
    /// The checkpoint includes:
    /// - Current training state (epoch, best accuracy, early stopping counters)
    /// - Training configuration (RLConfig)
    /// - Collected training data (TrainingDataCollector)
    /// - Note: Model weights are not currently serialized (future enhancement)
    /// </remarks>
    public async Task SaveCheckpointAsync(string filePath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        var checkpoint = new TrainingCheckpoint<T>
        {
            CurrentEpoch = _currentEpoch,
            BestAccuracy = _bestAccuracy,
            BestEpoch = _bestEpoch,
            EpochsWithoutImprovement = _epochsWithoutImprovement,
            Config = _config,
            TrainingData = _dataCollector
        };

        string json = JsonConvert.SerializeObject(checkpoint, Formatting.Indented);
        byte[] bytes = Encoding.UTF8.GetBytes(json);

        using (var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 4096, useAsync: true))
        {
            await fileStream.WriteAsync(bytes, 0, bytes.Length, cancellationToken);
        }
    }

    /// <summary>
    /// Loads a training checkpoint and restores training state.
    /// </summary>
    /// <param name="filePath">Path to the checkpoint file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <remarks>
    /// This restores:
    /// - Training state (allows resuming from the saved epoch)
    /// - Collected training data
    /// Note: The loaded config is for reference only; the current _config is not replaced.
    /// </remarks>
    public async Task LoadCheckpointAsync(string filePath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Checkpoint file not found: {filePath}", filePath);

#if NET7_0_OR_GREATER
        byte[] bytes = await File.ReadAllBytesAsync(filePath, cancellationToken);
#else
        byte[] bytes = await Task.Run(() => File.ReadAllBytes(filePath), cancellationToken);
#endif

        string json = Encoding.UTF8.GetString(bytes);
        var checkpoint = JsonConvert.DeserializeObject<TrainingCheckpoint<T>>(json);

        if (checkpoint == null)
            throw new InvalidOperationException($"Failed to deserialize checkpoint from {filePath}");

        // Restore training state
        _currentEpoch = checkpoint.CurrentEpoch;
        _bestAccuracy = checkpoint.BestAccuracy;
        _bestEpoch = checkpoint.BestEpoch;
        _epochsWithoutImprovement = checkpoint.EpochsWithoutImprovement;

        // Restore training data from checkpoint
        if (checkpoint.TrainingData != null)
        {
            _dataCollector.Clear();
            var samples = checkpoint.TrainingData.GetAllSamples();
            _dataCollector.AddSamples(samples);
        }
    }

    /// <summary>
    /// Resets training state to initial values (useful when starting fresh training).
    /// </summary>
    public void ResetTrainingState()
    {
        _currentEpoch = 0;
        _bestAccuracy = 0.0;
        _bestEpoch = 0;
        _epochsWithoutImprovement = 0;
    }
}

/// <summary>
/// Represents a training checkpoint that can be saved and loaded to resume training.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring.</typeparam>
internal class TrainingCheckpoint<T>
{
    /// <summary>
    /// Gets or sets the current epoch number (0-based).
    /// </summary>
    public int CurrentEpoch { get; set; }

    /// <summary>
    /// Gets or sets the best accuracy achieved so far.
    /// </summary>
    public double BestAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the epoch at which best accuracy was achieved.
    /// </summary>
    public int BestEpoch { get; set; }

    /// <summary>
    /// Gets or sets the number of epochs without improvement (for early stopping).
    /// </summary>
    public int EpochsWithoutImprovement { get; set; }

    /// <summary>
    /// Gets or sets the training configuration.
    /// </summary>
    public RLConfig Config { get; set; } = RLConfig.Default;

    /// <summary>
    /// Gets or sets the collected training data.
    /// </summary>
    public TrainingDataCollector<T>? TrainingData { get; set; }
}

/// <summary>
/// Configuration for reinforcement learning training.
/// </summary>
internal class RLConfig
{
    public int Epochs { get; set; } = 10;
    public int BatchSize { get; set; } = 32;
    public double LearningRate { get; set; } = 0.0001;
    public double DiscountFactor { get; set; } = 0.99;
    public double EntropyCoefficient { get; set; } = 0.01;
    public int ValidationFrequency { get; set; } = 1;
    public int EarlyStoppingPatience { get; set; } = 3;
    public bool SaveCheckpoints { get; set; } = true;

    public static RLConfig Default => new RLConfig();
}

/// <summary>
/// Metrics for a training epoch.
/// </summary>
internal class EpochMetrics<T>
{
    public int Epoch { get; set; }
    public T AverageReward { get; set; } = default!;
    public T AverageLoss { get; set; } = default!;
    public double Accuracy { get; set; }
    public T ValidationReward { get; set; } = default!;
}

/// <summary>
/// Progress information for a training batch.
/// </summary>
internal class BatchProgress
{
    public int BatchNumber { get; set; }
    public int TotalBatches { get; set; }
    public int Epoch { get; set; }
}

/// <summary>
/// Results from complete training run.
/// </summary>
internal class TrainingResults<T>
{
    public int EpochsTrained { get; set; }
    public double BestAccuracy { get; set; }
    public int BestEpoch { get; set; }
    public List<EpochMetrics<T>> EpochMetrics { get; set; } = new();
    public TrainingDataCollector<T>? DataCollector { get; set; }
}
