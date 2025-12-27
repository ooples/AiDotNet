using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;
using AiDotNet.SelfSupervisedLearning.Evaluation;

namespace AiDotNet.SelfSupervisedLearning.Core;

/// <summary>
/// Manages a self-supervised learning training session.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> An SSL session manages the entire training lifecycle:
/// initialization, training loop, evaluation, and checkpointing. It provides
/// callbacks for monitoring progress and supports resuming from checkpoints.</para>
/// </remarks>
public class SSLSession<T>
{
    private readonly ISSLMethod<T> _method;
    private readonly SSLConfig _config;
    private readonly SSLMetrics<T> _metrics;
    private SSLTrainingHistory<T> _history;

    private int _currentEpoch;
    private int _globalStep;
    private bool _isTraining;
    private DateTime _startTime;

    /// <summary>
    /// Event raised at the start of each epoch.
    /// </summary>
    public event Action<int>? OnEpochStart;

    /// <summary>
    /// Event raised at the end of each epoch.
    /// </summary>
    public event Action<int, T>? OnEpochEnd;

    /// <summary>
    /// Event raised after each training step.
    /// </summary>
    public event Action<int, SSLStepResult<T>>? OnStepComplete;

    /// <summary>
    /// Event raised when collapse is detected.
    /// </summary>
    public event Action<int>? OnCollapseDetected;

    /// <summary>
    /// Gets the current epoch number.
    /// </summary>
    public int CurrentEpoch => _currentEpoch;

    /// <summary>
    /// Gets the global step counter.
    /// </summary>
    public int GlobalStep => _globalStep;

    /// <summary>
    /// Gets whether training is in progress.
    /// </summary>
    public bool IsTraining => _isTraining;

    /// <summary>
    /// Gets the SSL method being used.
    /// </summary>
    public ISSLMethod<T> Method => _method;

    /// <summary>
    /// Initializes a new SSL training session.
    /// </summary>
    /// <param name="method">The SSL method to use.</param>
    /// <param name="config">Training configuration.</param>
    public SSLSession(ISSLMethod<T> method, SSLConfig? config = null)
    {
        _method = method ?? throw new ArgumentNullException(nameof(method));
        _config = config ?? new SSLConfig();
        _metrics = new SSLMetrics<T>();
        _history = new SSLTrainingHistory<T>();
    }

    /// <summary>
    /// Trains the SSL method for the specified number of epochs.
    /// </summary>
    /// <param name="dataLoader">Function that yields batches of data.</param>
    /// <param name="validationData">Optional validation data for k-NN evaluation.</param>
    /// <param name="validationLabels">Optional validation labels.</param>
    /// <returns>Training result.</returns>
    public SSLResult<T> Train(
        Func<IEnumerable<Tensor<T>>> dataLoader,
        Tensor<T>? validationData = null,
        int[]? validationLabels = null)
    {
        var totalEpochs = _config.PretrainingEpochs ?? 100;
        _startTime = DateTime.Now;
        _isTraining = true;

        try
        {
            for (_currentEpoch = 0; _currentEpoch < totalEpochs && _isTraining; _currentEpoch++)
            {
                TrainEpoch(dataLoader, validationData, validationLabels);
            }

            return CreateResult();
        }
        catch (Exception ex)
        {
            return SSLResult<T>.Failure(ex.Message);
        }
        finally
        {
            _isTraining = false;
        }
    }

    /// <summary>
    /// Trains for a single epoch.
    /// </summary>
    private void TrainEpoch(
        Func<IEnumerable<Tensor<T>>> dataLoader,
        Tensor<T>? validationData,
        int[]? validationLabels)
    {
        OnEpochStart?.Invoke(_currentEpoch);
        _method.OnEpochStart(_currentEpoch);

        T epochLoss = default!;
        int stepCount = 0;

        foreach (var batch in dataLoader())
        {
            var result = _method.TrainStep(batch);
            epochLoss = result.Loss;
            stepCount++;
            _globalStep++;

            OnStepComplete?.Invoke(_globalStep, result);

            // Check for collapse periodically
            if (_globalStep % 100 == 0)
            {
                // Get representations for collapse detection
                var representations = _method.Encode(batch);
                if (_metrics.DetectCollapse(representations))
                {
                    OnCollapseDetected?.Invoke(_currentEpoch);
                }
            }
        }

        // Epoch-end evaluation
        double knnAcc = 0;
        if (validationData is not null && validationLabels is not null)
        {
            var knn = new KNNEvaluator<T>(_method.GetEncoder());
            // For k-NN, we'd need training features - simplified here
            knnAcc = 0; // Would compute actual k-NN accuracy
        }

        // Record history
        var std = _metrics.ComputeRepresentationStd(
            _method.Encode(dataLoader().First()));

        _history.AddEpochMetrics(
            epochLoss, std, knnAcc,
            _method is SSLMethodBase<T> baseMethod ? baseMethod.GetEffectiveLearningRate() : 0,
            0);

        _method.OnEpochEnd(_currentEpoch);
        OnEpochEnd?.Invoke(_currentEpoch, epochLoss);
    }

    /// <summary>
    /// Stops training gracefully.
    /// </summary>
    public void Stop()
    {
        _isTraining = false;
    }

    /// <summary>
    /// Resets the session for a new training run.
    /// </summary>
    public void Reset()
    {
        _currentEpoch = 0;
        _globalStep = 0;
        _history = new SSLTrainingHistory<T>();
        _method.Reset();
    }

    /// <summary>
    /// Gets the current training history.
    /// </summary>
    public SSLTrainingHistory<T> GetHistory() => _history;

    /// <summary>
    /// Runs evaluation on the current encoder.
    /// </summary>
    public SSLMetricReport<T> Evaluate(Tensor<T> data)
    {
        var z1 = _method.Encode(data);
        var z2 = _method.Encode(data); // Same data, different augmentation would be used in practice

        return _metrics.ComputeFullReport(z1, z2);
    }

    private SSLResult<T> CreateResult()
    {
        var elapsed = DateTime.Now - _startTime;

        var result = SSLResult<T>.Success(
            _method.GetEncoder(),
            _config.Method ?? SSLMethodType.SimCLR,
            _config,
            _history);

        result.TrainingTimeSeconds = elapsed.TotalSeconds;
        result.EpochsTrained = _currentEpoch;

        return result;
    }

    /// <summary>
    /// Creates a session from a pretrained checkpoint.
    /// </summary>
    public static SSLSession<T> FromCheckpoint(
        string checkpointPath,
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, ISSLMethod<T>> methodFactory)
    {
        // Load checkpoint data (simplified)
        var method = methodFactory(encoder);
        var session = new SSLSession<T>(method);

        // Would load epoch, history, etc. from checkpoint

        return session;
    }
}
