using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;
using AiDotNet.SelfSupervisedLearning.Evaluation;
using JsonSerializer = System.Text.Json.JsonSerializer;

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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ISSLMethod<T> _method;
    private readonly SSLConfig _config;
    private readonly SSLMetrics<T> _metrics;
    private SSLTrainingHistory<T> _history;

    private int _currentEpoch;
    private int _globalStep;
    private bool _isTraining;
    private DateTime _startTime;

    // Storage for k-NN evaluation
    private Tensor<T>? _cachedTrainingFeatures;
    private int[]? _cachedTrainingLabels;

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

        // Epoch-end k-NN evaluation
        double knnAcc = 0;
        if (validationData is not null && validationLabels is not null &&
            _cachedTrainingFeatures is not null && _cachedTrainingLabels is not null)
        {
            knnAcc = ComputeKNNAccuracy(validationData, validationLabels);
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
    /// Caches training data for k-NN evaluation.
    /// </summary>
    /// <param name="trainingData">Training data batches.</param>
    /// <param name="trainingLabels">Corresponding labels.</param>
    public void CacheTrainingFeaturesForKNN(IEnumerable<Tensor<T>> trainingData, int[] trainingLabels)
    {
        // Encode all training data
        var allFeatures = new List<T[]>();
        foreach (var batch in trainingData)
        {
            var encoded = _method.Encode(batch);
            var batchSize = encoded.Shape[0];
            var dim = encoded.Shape[1];

            for (int i = 0; i < batchSize; i++)
            {
                var features = new T[dim];
                for (int j = 0; j < dim; j++)
                {
                    features[j] = encoded[i, j];
                }
                allFeatures.Add(features);
            }
        }

        // Convert to tensor
        var numSamples = allFeatures.Count;
        var featureDim = allFeatures[0].Length;
        var flatFeatures = new T[numSamples * featureDim];

        for (int i = 0; i < numSamples; i++)
        {
            Array.Copy(allFeatures[i], 0, flatFeatures, i * featureDim, featureDim);
        }

        _cachedTrainingFeatures = new Tensor<T>(flatFeatures, [numSamples, featureDim]);
        _cachedTrainingLabels = trainingLabels;
    }

    /// <summary>
    /// Computes k-NN accuracy on validation data using cached training features.
    /// </summary>
    private double ComputeKNNAccuracy(Tensor<T> validationData, int[] validationLabels, int k = 20)
    {
        if (_cachedTrainingFeatures is null || _cachedTrainingLabels is null)
        {
            return 0;
        }

        var validationFeatures = _method.Encode(validationData);
        var numVal = validationFeatures.Shape[0];
        var numTrain = _cachedTrainingFeatures.Shape[0];
        var dim = validationFeatures.Shape[1];

        int correctCount = 0;

        for (int v = 0; v < numVal; v++)
        {
            // Compute distances to all training samples
            var distances = new (double distance, int label)[numTrain];

            for (int t = 0; t < numTrain; t++)
            {
                double dist = 0;
                for (int d = 0; d < dim; d++)
                {
                    var diff = NumOps.ToDouble(validationFeatures[v, d]) -
                               NumOps.ToDouble(_cachedTrainingFeatures[t, d]);
                    dist += diff * diff;
                }
                distances[t] = (dist, _cachedTrainingLabels[t]);
            }

            // Find k nearest neighbors
            Array.Sort(distances, (a, b) => a.distance.CompareTo(b.distance));

            // Majority vote among k nearest
            var labelCounts = new Dictionary<int, int>();
            for (int i = 0; i < Math.Min(k, numTrain); i++)
            {
                var label = distances[i].label;
                labelCounts[label] = labelCounts.GetValueOrDefault(label, 0) + 1;
            }

            // Find label with maximum count (manual implementation for .NET 4.7.1 compatibility)
            var predictedLabel = 0;
            var maxCount = -1;
            foreach (var kv in labelCounts)
            {
                if (kv.Value > maxCount)
                {
                    maxCount = kv.Value;
                    predictedLabel = kv.Key;
                }
            }

            if (predictedLabel == validationLabels[v])
            {
                correctCount++;
            }
        }

        return (double)correctCount / numVal;
    }

    /// <summary>
    /// Saves a checkpoint to disk.
    /// </summary>
    /// <param name="checkpointPath">Path to save the checkpoint.</param>
    public void SaveCheckpoint(string checkpointPath)
    {
        var checkpointData = new SSLCheckpointData
        {
            CurrentEpoch = _currentEpoch,
            GlobalStep = _globalStep,
            MethodType = _method.Name,
            ConfigJson = JsonSerializer.Serialize(_config),
            HistoryJson = JsonSerializer.Serialize(_history)
        };

        // Save checkpoint metadata
        var metadataPath = checkpointPath + ".meta";
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(checkpointData));

        // Save encoder parameters
        var encoderParams = _method.GetEncoder().GetParameters();
        var paramsPath = checkpointPath + ".params";
        SaveParameters(paramsPath, encoderParams);
    }

    private static void SaveParameters(string path, Vector<T> parameters)
    {
        var doubleParams = new double[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            doubleParams[i] = NumOps.ToDouble(parameters[i]);
        }
        File.WriteAllBytes(path, DoubleArrayToBytes(doubleParams));
    }

    private static byte[] DoubleArrayToBytes(double[] data)
    {
        var bytes = new byte[data.Length * sizeof(double)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static double[] BytesToDoubleArray(byte[] bytes)
    {
        var data = new double[bytes.Length / sizeof(double)];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return data;
    }

    /// <summary>
    /// Creates a session from a pretrained checkpoint.
    /// </summary>
    /// <param name="checkpointPath">Path to the checkpoint file.</param>
    /// <param name="encoder">The encoder network to load weights into.</param>
    /// <param name="methodFactory">Factory to create the SSL method from the encoder.</param>
    /// <returns>A session restored from the checkpoint.</returns>
    public static SSLSession<T> FromCheckpoint(
        string checkpointPath,
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, ISSLMethod<T>> methodFactory)
    {
        // Load checkpoint metadata
        var metadataPath = checkpointPath + ".meta";
        if (!File.Exists(metadataPath))
        {
            throw new FileNotFoundException($"Checkpoint metadata not found: {metadataPath}");
        }

        var metadataJson = File.ReadAllText(metadataPath);
        var checkpointData = JsonSerializer.Deserialize<SSLCheckpointData>(metadataJson)
            ?? throw new InvalidDataException("Failed to deserialize checkpoint metadata");

        // Load encoder parameters
        var paramsPath = checkpointPath + ".params";
        if (!File.Exists(paramsPath))
        {
            throw new FileNotFoundException($"Checkpoint parameters not found: {paramsPath}");
        }

        var paramBytes = File.ReadAllBytes(paramsPath);
        var doubleParams = BytesToDoubleArray(paramBytes);
        var parameters = new T[doubleParams.Length];
        for (int i = 0; i < doubleParams.Length; i++)
        {
            parameters[i] = NumOps.FromDouble(doubleParams[i]);
        }
        encoder.UpdateParameters(new Vector<T>(parameters));

        // Deserialize config
        var config = JsonSerializer.Deserialize<SSLConfig>(checkpointData.ConfigJson)
            ?? new SSLConfig();

        // Create method and session
        var method = methodFactory(encoder);
        var session = new SSLSession<T>(method, config)
        {
            _currentEpoch = checkpointData.CurrentEpoch,
            _globalStep = checkpointData.GlobalStep
        };

        // Restore history if available
        if (!string.IsNullOrEmpty(checkpointData.HistoryJson))
        {
            var history = JsonSerializer.Deserialize<SSLTrainingHistory<T>>(checkpointData.HistoryJson);
            if (history is not null)
            {
                session._history = history;
            }
        }

        return session;
    }
}

/// <summary>
/// Data structure for SSL checkpoint serialization.
/// </summary>
internal class SSLCheckpointData
{
    public int CurrentEpoch { get; set; }
    public int GlobalStep { get; set; }
    public string MethodType { get; set; } = string.Empty;
    public string ConfigJson { get; set; } = string.Empty;
    public string HistoryJson { get; set; } = string.Empty;
}
