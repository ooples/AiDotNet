using System.Text.Json;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.CurriculumLearning.Results;
using AiDotNet.CurriculumLearning.Schedulers;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning;

/// <summary>
/// Main orchestrator for curriculum learning that coordinates difficulty estimation,
/// scheduling, and model training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input features.</typeparam>
/// <typeparam name="TOutput">The type of output labels.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Curriculum learning is a training strategy that presents
/// samples from easy to hard, mimicking how humans learn. This class coordinates the
/// entire process:</para>
///
/// <list type="number">
/// <item><description>Estimates difficulty of each training sample</description></item>
/// <item><description>Sorts samples from easy to hard</description></item>
/// <item><description>Trains the model on progressively harder samples</description></item>
/// <item><description>Monitors progress and can stop early if model stops improving</description></item>
/// </list>
///
/// <para><b>Key Components:</b></para>
/// <list type="bullet">
/// <item><description><b>DifficultyEstimator:</b> Measures how hard each sample is</description></item>
/// <item><description><b>Scheduler:</b> Controls when to introduce harder samples</description></item>
/// <item><description><b>Config:</b> Settings for training (epochs, batch size, early stopping)</description></item>
/// </list>
///
/// <para><b>Example Usage:</b></para>
/// <code>
/// var learner = new CurriculumLearner&lt;double, Vector&lt;double&gt;, double&gt;(
///     model,
///     config,
///     new LossBasedDifficultyEstimator&lt;double, Vector&lt;double&gt;, double&gt;());
///
/// var result = learner.Train(trainingData, validationData);
/// </code>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Bengio et al. "Curriculum Learning" (ICML 2009)</description></item>
/// <item><description>Soviany et al. "Curriculum Learning: A Survey" (IJCV 2022)</description></item>
/// </list>
/// </remarks>
public class CurriculumLearner<T, TInput, TOutput> : ICurriculumLearner<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _baseModel;
    private readonly ICurriculumLearnerConfig<T> _config;
    private readonly IDifficultyEstimator<T, TInput, TOutput> _difficultyEstimator;
    private readonly ICurriculumScheduler<T> _scheduler;
    private readonly List<CurriculumPhaseResult<T>> _phaseHistory;
    private readonly Random _random;

    private bool _isTraining;
    private T _bestValidationLoss;
    private int _epochsWithoutImprovement;
    private Vector<T>? _currentDifficulties;
    private int[]? _sortedIndices;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> BaseModel => _baseModel;

    /// <inheritdoc/>
    public ICurriculumLearnerConfig<T> Config => _config;

    /// <inheritdoc/>
    public IDifficultyEstimator<T, TInput, TOutput> DifficultyEstimator => _difficultyEstimator;

    /// <inheritdoc/>
    public ICurriculumScheduler<T> Scheduler => _scheduler;

    /// <inheritdoc/>
    public T CurrentPhase => _scheduler.CurrentPhase;

    /// <inheritdoc/>
    public int CurrentEpoch => _scheduler.CurrentEpoch;

    /// <inheritdoc/>
    public bool IsTraining => _isTraining;

    /// <inheritdoc/>
    public event EventHandler<CurriculumPhaseEventArgs<T>>? PhaseStarted;

    /// <inheritdoc/>
    public event EventHandler<CurriculumPhaseCompletedEventArgs<T>>? PhaseCompleted;

    /// <inheritdoc/>
    public event EventHandler<CurriculumTrainingCompletedEventArgs<T>>? TrainingCompleted;

    /// <summary>
    /// Initializes a new instance of the <see cref="CurriculumLearner{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="baseModel">The model to train.</param>
    /// <param name="config">Configuration for curriculum learning.</param>
    /// <param name="difficultyEstimator">Estimator for sample difficulties.</param>
    /// <param name="scheduler">Optional custom scheduler. If null, created based on config.</param>
    public CurriculumLearner(
        IFullModel<T, TInput, TOutput> baseModel,
        ICurriculumLearnerConfig<T> config,
        IDifficultyEstimator<T, TInput, TOutput> difficultyEstimator,
        ICurriculumScheduler<T>? scheduler = null)
    {
        if (baseModel is null) throw new ArgumentNullException(nameof(baseModel));
        if (config is null) throw new ArgumentNullException(nameof(config));
        if (difficultyEstimator is null) throw new ArgumentNullException(nameof(difficultyEstimator));

        _baseModel = baseModel;
        _config = config;
        _difficultyEstimator = difficultyEstimator;
        _scheduler = scheduler ?? CreateSchedulerFromConfig(config);
        _phaseHistory = new List<CurriculumPhaseResult<T>>();
        _random = config.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(config.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        _isTraining = false;
        _bestValidationLoss = NumOps.FromDouble(double.MaxValue);
        _epochsWithoutImprovement = 0;
    }

    /// <inheritdoc/>
    public CurriculumLearningResult<T> Train(IDataset<T, TInput, TOutput> trainingData)
    {
        return Train(trainingData, null);
    }

    /// <inheritdoc/>
    public CurriculumLearningResult<T> Train(
        IDataset<T, TInput, TOutput> trainingData,
        IDataset<T, TInput, TOutput>? validationData)
    {
        if (trainingData is null) throw new ArgumentNullException(nameof(trainingData));

        // Estimate difficulties
        var difficulties = EstimateDifficulties(trainingData);

        return TrainWithDifficulty(trainingData, difficulties, validationData);
    }

    /// <inheritdoc/>
    public CurriculumLearningResult<T> TrainWithDifficulty(
        IDataset<T, TInput, TOutput> trainingData,
        Vector<T> difficultyScores)
    {
        return TrainWithDifficulty(trainingData, difficultyScores, null);
    }

    /// <summary>
    /// Trains the model using curriculum learning with pre-computed difficulty scores and optional validation.
    /// </summary>
    private CurriculumLearningResult<T> TrainWithDifficulty(
        IDataset<T, TInput, TOutput> trainingData,
        Vector<T> difficultyScores,
        IDataset<T, TInput, TOutput>? validationData)
    {
        if (trainingData is null) throw new ArgumentNullException(nameof(trainingData));
        if (difficultyScores is null) throw new ArgumentNullException(nameof(difficultyScores));

        if (difficultyScores.Length != trainingData.Count)
        {
            throw new ArgumentException(
                $"Difficulty scores count ({difficultyScores.Length}) must match training data count ({trainingData.Count}).");
        }

        _isTraining = true;
        _currentDifficulties = difficultyScores;

        // Normalize difficulties if configured
        if (_config.NormalizeDifficulties)
        {
            _currentDifficulties = NormalizeDifficulties(_currentDifficulties);
        }

        // Sort indices by difficulty (easy to hard)
        _sortedIndices = _difficultyEstimator.GetSortedIndices(_currentDifficulties);

        var result = new CurriculumLearningResult<T>
        {
            TotalEpochs = 0,
            PhaseResults = [],
            CurriculumProgression = [],
            TrainingSamples = trainingData.Count,
            ValidationSamples = validationData?.Count
        };

        try
        {
            // Reset scheduler for fresh training
            _scheduler.Reset();
            _phaseHistory.Clear();
            _bestValidationLoss = NumOps.FromDouble(double.MaxValue);
            _epochsWithoutImprovement = 0;

            var currentPhaseNumber = -1;
            var phaseStartEpoch = 0;
            var phaseTrainingLosses = new List<T>();
            var phaseValidationLosses = new List<T>();

            // Training loop
            for (int epoch = 0; epoch < _config.TotalEpochs; epoch++)
            {
                // Check for phase change
                if (_scheduler.CurrentPhaseNumber != currentPhaseNumber)
                {
                    // Complete previous phase if not first
                    if (currentPhaseNumber >= 0)
                    {
                        CompletePreviousPhase(
                            result, currentPhaseNumber, phaseStartEpoch, epoch,
                            phaseTrainingLosses, phaseValidationLosses);
                    }

                    // Start new phase
                    currentPhaseNumber = _scheduler.CurrentPhaseNumber;
                    phaseStartEpoch = epoch;
                    phaseTrainingLosses.Clear();
                    phaseValidationLosses.Clear();

                    RaisePhaseStarted(currentPhaseNumber, _scheduler.GetDataFraction());

                    LogPhaseStart(currentPhaseNumber);
                }

                // Get current curriculum indices
                var curriculumIndices = GetCurrentCurriculumIndices(_currentDifficulties);

                // Shuffle within phase if configured
                if (_config.ShuffleWithinPhase)
                {
                    ShuffleArray(curriculumIndices);
                }

                // Create phase-specific training subset
                var phaseData = trainingData.Subset(curriculumIndices);

                // Train one epoch
                var epochLoss = TrainEpoch(phaseData, epoch);
                phaseTrainingLosses.Add(epochLoss);

                // Validate if validation data provided
                T? validationLoss = default;
                T? validationAccuracy = default;
                if (validationData != null)
                {
                    (validationLoss, validationAccuracy) = Evaluate(validationData);
                    phaseValidationLosses.Add(validationLoss);
                }

                // Record progression
                result.CurriculumProgression.Add(new CurriculumProgressionEntry<T>
                {
                    Epoch = epoch,
                    Phase = currentPhaseNumber,
                    DataFraction = _scheduler.GetDataFraction(),
                    TrainingLoss = epochLoss,
                    ValidationLoss = validationLoss,
                    SamplesUsed = curriculumIndices.Length
                });

                // Create epoch metrics for scheduler
                var epochMetrics = new CurriculumEpochMetrics<T>
                {
                    Epoch = epoch,
                    TrainingLoss = epochLoss,
                    ValidationLoss = validationLoss,
                    TrainingAccuracy = default,
                    ValidationAccuracy = validationAccuracy,
                    Improved = CheckImprovement(validationLoss ?? epochLoss),
                    SamplesUsed = curriculumIndices.Length
                };

                // Update scheduler (may advance phase)
                _scheduler.StepEpoch(epochMetrics);

                // Recalculate difficulties if configured
                if (_config.RecalculateDifficulties &&
                    epoch > 0 &&
                    epoch % _config.DifficultyRecalculationFrequency == 0)
                {
                    _difficultyEstimator.Update(epoch, _baseModel);
                    _currentDifficulties = EstimateDifficulties(trainingData);
                    if (_config.NormalizeDifficulties)
                    {
                        _currentDifficulties = NormalizeDifficulties(_currentDifficulties);
                    }
                    _sortedIndices = _difficultyEstimator.GetSortedIndices(_currentDifficulties);
                }

                // Check early stopping
                if (_config.UseEarlyStopping && CheckEarlyStopping(validationLoss ?? epochLoss))
                {
                    result.EarlyStopTriggered = true;
                    LogMessage($"Early stopping triggered at epoch {epoch + 1}");
                    result.TotalEpochs = epoch + 1;
                    break;
                }

                result.TotalEpochs = epoch + 1;
            }

            // Complete final phase
            if (currentPhaseNumber >= 0)
            {
                CompletePreviousPhase(
                    result, currentPhaseNumber, phaseStartEpoch, result.TotalEpochs,
                    phaseTrainingLosses, phaseValidationLosses);
            }

            // Finalize result
            FinalizeResult(result, validationData);

            // Raise training completed event
            RaiseTrainingCompleted(result);
        }
        finally
        {
            _isTraining = false;
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> EstimateDifficulties(IDataset<T, TInput, TOutput> dataset)
    {
        if (dataset is null) throw new ArgumentNullException(nameof(dataset));

        return _difficultyEstimator.EstimateDifficulties(dataset, _baseModel);
    }

    /// <inheritdoc/>
    public int[] GetCurrentCurriculumIndices(Vector<T> allDifficulties)
    {
        if (allDifficulties is null) throw new ArgumentNullException(nameof(allDifficulties));

        // Use cached sorted indices if available and correct size
        var sortedIndices = _sortedIndices;
        if (sortedIndices == null || sortedIndices.Length != allDifficulties.Length)
        {
            sortedIndices = _difficultyEstimator.GetSortedIndices(allDifficulties);
        }

        return _scheduler.GetCurrentIndices(sortedIndices, allDifficulties.Length);
    }

    /// <inheritdoc/>
    public bool AdvancePhase()
    {
        return _scheduler.AdvancePhase();
    }

    /// <inheritdoc/>
    public void ResetCurriculum()
    {
        _scheduler.Reset();
        _difficultyEstimator.Reset();
        _phaseHistory.Clear();
        _currentDifficulties = null;
        _sortedIndices = null;
        _bestValidationLoss = NumOps.FromDouble(double.MaxValue);
        _epochsWithoutImprovement = 0;
    }

    /// <inheritdoc/>
    public IReadOnlyList<CurriculumPhaseResult<T>> GetPhaseHistory()
    {
        return _phaseHistory.AsReadOnly();
    }

    /// <inheritdoc/>
    public void Save(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));

        var state = new CurriculumLearnerState<T>
        {
            CurrentEpoch = _scheduler.CurrentEpoch,
            CurrentPhaseNumber = _scheduler.CurrentPhaseNumber,
            BestValidationLoss = NumOps.ToDouble(_bestValidationLoss),
            EpochsWithoutImprovement = _epochsWithoutImprovement,
            PhaseHistory = _phaseHistory.ToList(),
            SchedulerStatistics = _scheduler.GetStatistics()
        };

        var json = System.Text.Json.JsonSerializer.Serialize(state, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);

        // Save model state in same directory
        var modelPath = Path.Combine(
            Path.GetDirectoryName(path) ?? ".",
            Path.GetFileNameWithoutExtension(path) + "_model.bin");
        _baseModel.SaveModel(modelPath);
    }

    /// <inheritdoc/>
    public void Load(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));

        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Curriculum learner state file not found.", path);
        }

        var json = File.ReadAllText(path);
        var state = System.Text.Json.JsonSerializer.Deserialize<CurriculumLearnerState<T>>(json);

        if (state != null)
        {
            _bestValidationLoss = NumOps.FromDouble(state.BestValidationLoss);
            _epochsWithoutImprovement = state.EpochsWithoutImprovement;
            _phaseHistory.Clear();
            if (state.PhaseHistory != null)
            {
                _phaseHistory.AddRange(state.PhaseHistory);
            }
        }

        // Load model state
        var modelPath = Path.Combine(
            Path.GetDirectoryName(path) ?? ".",
            Path.GetFileNameWithoutExtension(path) + "_model.bin");
        if (File.Exists(modelPath))
        {
            _baseModel.LoadModel(modelPath);
        }
    }

    /// <summary>
    /// Trains the model for one epoch on the provided data.
    /// </summary>
    private T TrainEpoch(IDataset<T, TInput, TOutput> data, int epoch)
    {
        var totalLoss = NumOps.Zero;
        var batchCount = 0;

        // Process in batches
        var indices = Enumerable.Range(0, data.Count).ToArray();
        if (_config.ShuffleWithinPhase)
        {
            ShuffleArray(indices);
        }

        for (int i = 0; i < data.Count; i += _config.BatchSize)
        {
            var batchEnd = Math.Min(i + _config.BatchSize, data.Count);
            var batchLoss = NumOps.Zero;
            var batchSamples = 0;

            for (int j = i; j < batchEnd; j++)
            {
                var idx = indices[j];
                var (input, output) = data.GetSample(idx);

                // Compute gradients
                var gradients = _baseModel.ComputeGradients(input, output);

                // Apply gradients with learning rate
                _baseModel.ApplyGradients(gradients, _config.LearningRate);

                // Accumulate loss (use gradient magnitude as proxy for loss)
                var sampleLoss = ComputeLoss(input, output);
                batchLoss = NumOps.Add(batchLoss, sampleLoss);
                batchSamples++;
            }

            if (batchSamples > 0)
            {
                batchLoss = NumOps.Divide(batchLoss, NumOps.FromDouble(batchSamples));
                totalLoss = NumOps.Add(totalLoss, batchLoss);
                batchCount++;
            }
        }

        // Return average loss
        if (batchCount > 0)
        {
            return NumOps.Divide(totalLoss, NumOps.FromDouble(batchCount));
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Computes the loss for a single sample.
    /// </summary>
    private T ComputeLoss(TInput input, TOutput expectedOutput)
    {
        // Use the model's default loss function - convert TOutput to Vector<T>
        var prediction = _baseModel.Predict(input);
        var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
        var expectedVector = ConversionsHelper.ConvertToVector<T, TOutput>(expectedOutput);
        return _baseModel.DefaultLossFunction.CalculateLoss(predictionVector, expectedVector);
    }

    /// <summary>
    /// Evaluates the model on a dataset.
    /// </summary>
    private (T Loss, T? Accuracy) Evaluate(IDataset<T, TInput, TOutput> data)
    {
        var totalLoss = NumOps.Zero;
        var correctCount = 0;

        for (int i = 0; i < data.Count; i++)
        {
            var (input, output) = data.GetSample(i);
            var loss = ComputeLoss(input, output);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Compute accuracy using argmax for classification outputs
            var prediction = _baseModel.Predict(input);
            if (prediction != null && IsCorrectPrediction(prediction, output))
            {
                correctCount++;
            }
        }

        var avgLoss = data.Count > 0
            ? NumOps.Divide(totalLoss, NumOps.FromDouble(data.Count))
            : NumOps.Zero;

        var accuracy = data.Count > 0
            ? NumOps.FromDouble((double)correctCount / data.Count)
            : NumOps.Zero;

        return (avgLoss, accuracy);
    }

    /// <summary>
    /// Determines if a prediction is correct by comparing with expected output.
    /// Uses argmax comparison for vector outputs (classification) or threshold-based
    /// comparison for scalar outputs (regression).
    /// </summary>
    private bool IsCorrectPrediction(TOutput prediction, TOutput expected)
    {
        try
        {
            // Convert to vectors for comparison
            var predVec = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
            var expVec = ConversionsHelper.ConvertToVector<T, TOutput>(expected);

            // For multi-class classification (vector outputs), compare argmax
            if (predVec.Length > 1 && expVec.Length > 1)
            {
                return ArgMax(predVec) == ArgMax(expVec);
            }

            // For scalar outputs (regression), use threshold-based comparison
            if (predVec.Length == 1 && expVec.Length == 1)
            {
                var diff = NumOps.Abs(NumOps.Subtract(predVec[0], expVec[0]));
                var threshold = NumOps.FromDouble(0.01); // 1% tolerance for regression
                var absExpected = NumOps.Abs(expVec[0]);
                var one = NumOps.FromDouble(1.0);
                // Use max(|expected|, 1.0) as normalizer to avoid division issues
                var normalizer = NumOps.Compare(absExpected, one) > 0 ? absExpected : one;
                var relativeError = NumOps.Divide(diff, normalizer);
                return NumOps.Compare(relativeError, threshold) <= 0;
            }

            // Fallback: check if vectors are equal element-wise
            if (predVec.Length != expVec.Length)
            {
                return false;
            }

            for (int i = 0; i < predVec.Length; i++)
            {
                if (NumOps.Compare(predVec[i], expVec[i]) != 0)
                {
                    return false;
                }
            }

            return true;
        }
        catch (InvalidOperationException)
        {
            // If conversion fails, use direct equality comparison
            return prediction != null && prediction.Equals(expected);
        }
    }

    /// <summary>
    /// Returns the index of the maximum element in a vector.
    /// </summary>
    private static int ArgMax(Vector<T> vec)
    {
        if (vec.Length == 0)
        {
            return -1;
        }

        int maxIdx = 0;
        T maxVal = vec[0];

        for (int i = 1; i < vec.Length; i++)
        {
            if (NumOps.Compare(vec[i], maxVal) > 0)
            {
                maxVal = vec[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /// <summary>
    /// Normalizes difficulty scores to [0, 1] range.
    /// </summary>
    private Vector<T> NormalizeDifficulties(Vector<T> difficulties)
    {
        var minVal = difficulties.Minimum();
        var maxVal = difficulties.Max();
        var range = NumOps.Subtract(maxVal, minVal);

        if (NumOps.Compare(range, NumOps.Zero) <= 0)
        {
            // All same difficulty - return uniform values
            return new Vector<T>(Enumerable.Repeat(NumOps.FromDouble(0.5), difficulties.Length));
        }

        var normalized = new Vector<T>(difficulties.Length);
        for (int i = 0; i < difficulties.Length; i++)
        {
            normalized[i] = NumOps.Divide(
                NumOps.Subtract(difficulties[i], minVal),
                range);
        }

        return normalized;
    }

    /// <summary>
    /// Checks if the current loss represents an improvement.
    /// </summary>
    private bool CheckImprovement(T currentLoss)
    {
        var improvement = NumOps.Subtract(_bestValidationLoss, currentLoss);
        if (NumOps.Compare(improvement, _config.EarlyStoppingMinDelta) > 0)
        {
            _bestValidationLoss = currentLoss;
            _epochsWithoutImprovement = 0;
            return true;
        }

        _epochsWithoutImprovement++;
        return false;
    }

    /// <summary>
    /// Checks if early stopping should be triggered.
    /// </summary>
    private bool CheckEarlyStopping(T currentLoss)
    {
        CheckImprovement(currentLoss);
        return _epochsWithoutImprovement >= _config.EarlyStoppingPatience;
    }

    /// <summary>
    /// Shuffles an array in place using Fisher-Yates algorithm.
    /// </summary>
    private void ShuffleArray(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    /// <summary>
    /// Completes a training phase and records results.
    /// </summary>
    private void CompletePreviousPhase(
        CurriculumLearningResult<T> result,
        int phaseNumber,
        int startEpoch,
        int endEpoch,
        List<T> trainingLosses,
        List<T> validationLosses)
    {
        var phaseResult = new CurriculumPhaseResult<T>
        {
            PhaseNumber = phaseNumber,
            StartEpoch = startEpoch,
            EndEpoch = endEpoch,
            DataFraction = _scheduler.GetDataFraction(),
            TrainingLosses = trainingLosses.ToList(),
            ValidationLosses = validationLosses.ToList(),
            FinalTrainingLoss = trainingLosses.Count > 0 ? trainingLosses[^1] : NumOps.Zero,
            FinalValidationLoss = validationLosses.Count > 0 ? validationLosses[^1] : default,
            NumSamples = (int)Math.Ceiling(
                NumOps.ToDouble(_scheduler.GetDataFraction()) *
                (result.TrainingSamples ?? 0))
        };

        result.PhaseResults.Add(phaseResult);
        _phaseHistory.Add(phaseResult);
        result.PhasesCompleted = phaseNumber + 1;

        RaisePhaseCompleted(phaseResult);
        LogPhaseCompleted(phaseResult);
    }

    /// <summary>
    /// Finalizes the training result with summary statistics.
    /// </summary>
    private void FinalizeResult(
        CurriculumLearningResult<T> result,
        IDataset<T, TInput, TOutput>? validationData)
    {
        // Calculate best and final losses
        if (result.CurriculumProgression.Count > 0)
        {
            result.FinalTrainingLoss = result.CurriculumProgression[^1].TrainingLoss;
            result.FinalValidationLoss = result.CurriculumProgression[^1].ValidationLoss;

            result.BestTrainingLoss = result.CurriculumProgression
                .Select(p => p.TrainingLoss)
                .Aggregate((a, b) => NumOps.Compare(a, b) < 0 ? a : b);

            if (validationData != null)
            {
                result.BestValidationLoss = _bestValidationLoss;
            }
        }

        result.Success = !result.EarlyStopTriggered ||
                         (result.PhasesCompleted >= _config.NumPhases / 2);

        // Store final difficulties
        result.FinalDifficulties = _currentDifficulties;

        // Store scheduler statistics
        result.SchedulerStatistics = _scheduler.GetStatistics();
    }

    /// <summary>
    /// Creates a scheduler based on configuration.
    /// </summary>
    private static ICurriculumScheduler<T> CreateSchedulerFromConfig(ICurriculumLearnerConfig<T> config)
    {
        return config.ScheduleType switch
        {
            CurriculumScheduleType.Linear => new LinearScheduler<T>(
                config.TotalEpochs,
                config.InitialDataFraction,
                config.FinalDataFraction),

            CurriculumScheduleType.Exponential => new ExponentialScheduler<T>(
                config.TotalEpochs,
                growthRate: 3.0,
                config.InitialDataFraction,
                config.FinalDataFraction),

            CurriculumScheduleType.Step => new StepScheduler<T>(
                config.TotalEpochs,
                config.NumPhases,
                config.InitialDataFraction,
                config.FinalDataFraction),

            CurriculumScheduleType.SelfPaced => new SelfPacedScheduler<T>(
                config.TotalEpochs),

            CurriculumScheduleType.CompetenceBased => new CompetenceBasedScheduler<T>(
                config.TotalEpochs,
                totalPhases: config.NumPhases),

            CurriculumScheduleType.Polynomial => new PolynomialScheduler<T>(
                config.TotalEpochs,
                power: 2.0,
                config.InitialDataFraction,
                config.FinalDataFraction),

            CurriculumScheduleType.Cosine => new CosineScheduler<T>(
                config.TotalEpochs,
                config.InitialDataFraction,
                config.FinalDataFraction),

            _ => new LinearScheduler<T>(
                config.TotalEpochs,
                config.InitialDataFraction,
                config.FinalDataFraction)
        };
    }

    /// <summary>
    /// Raises the PhaseStarted event.
    /// </summary>
    private void RaisePhaseStarted(int phaseNumber, T dataFraction)
    {
        var sampleCount = _currentDifficulties != null
            ? GetCurrentCurriculumIndices(_currentDifficulties).Length
            : 0;
        PhaseStarted?.Invoke(this, new CurriculumPhaseEventArgs<T>(
            phaseNumber,
            _scheduler.TotalPhases,
            dataFraction,
            sampleCount));
    }

    /// <summary>
    /// Raises the PhaseCompleted event.
    /// </summary>
    private void RaisePhaseCompleted(CurriculumPhaseResult<T> phaseResult)
    {
        PhaseCompleted?.Invoke(this, new CurriculumPhaseCompletedEventArgs<T>(
            phaseResult.PhaseNumber,
            _scheduler.TotalPhases,
            phaseResult.DataFraction,
            phaseResult.SampleCount,
            phaseResult));
    }

    /// <summary>
    /// Raises the TrainingCompleted event.
    /// </summary>
    private void RaiseTrainingCompleted(CurriculumLearningResult<T> result)
    {
        TrainingCompleted?.Invoke(this, new CurriculumTrainingCompletedEventArgs<T>(result));
    }

    /// <summary>
    /// Logs a phase start message based on verbosity.
    /// </summary>
    private void LogPhaseStart(int phaseNumber)
    {
        if (_config.Verbosity >= CurriculumVerbosity.Normal)
        {
            LogMessage($"Starting phase {phaseNumber + 1}/{_scheduler.TotalPhases} " +
                      $"with {NumOps.ToDouble(_scheduler.GetDataFraction()) * 100:F1}% of data");
        }
    }

    /// <summary>
    /// Logs a phase completion message based on verbosity.
    /// </summary>
    private void LogPhaseCompleted(CurriculumPhaseResult<T> phaseResult)
    {
        if (_config.Verbosity >= CurriculumVerbosity.Normal)
        {
            LogMessage($"Phase {phaseResult.PhaseNumber + 1} completed: " +
                      $"Training Loss = {NumOps.ToDouble(phaseResult.FinalTrainingLoss):F4}");
        }
    }

    /// <summary>
    /// Logs a message using the configured logging action or Console.WriteLine as fallback.
    /// </summary>
    /// <param name="message">The message to log.</param>
    private void LogMessage(string message)
    {
        // Use configured logging action if available, otherwise fall back to Console
        if (_config.LogAction != null)
        {
            _config.LogAction($"[CurriculumLearner] {message}");
        }
        else
        {
            Console.WriteLine($"[CurriculumLearner] {message}");
        }
    }
}

/// <summary>
/// State for serializing curriculum learner progress.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class CurriculumLearnerState<T>
{
    /// <summary>
    /// Gets or sets the current epoch.
    /// </summary>
    public int CurrentEpoch { get; set; }

    /// <summary>
    /// Gets or sets the current phase number.
    /// </summary>
    public int CurrentPhaseNumber { get; set; }

    /// <summary>
    /// Gets or sets the best validation loss as a double.
    /// </summary>
    public double BestValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets epochs without improvement.
    /// </summary>
    public int EpochsWithoutImprovement { get; set; }

    /// <summary>
    /// Gets or sets the phase history.
    /// </summary>
    public List<CurriculumPhaseResult<T>>? PhaseHistory { get; set; }

    /// <summary>
    /// Gets or sets scheduler statistics.
    /// </summary>
    public Dictionary<string, object>? SchedulerStatistics { get; set; }
}
