namespace AiDotNet.TimeSeries.AnomalyDetection;

/// <summary>
/// Implements DeepANT (Deep Learning for Anomaly Detection in Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// DeepANT is a deep learning-based approach for unsupervised anomaly detection in time series.
/// It uses a convolutional neural network to learn normal patterns and identifies anomalies
/// as data points that deviate significantly from the learned patterns.
/// </para>
/// <para>
/// Key features:
/// - Time series prediction using CNN
/// - Anomaly detection based on prediction error
/// - Unsupervised learning (no labeled anomalies needed)
/// - Effective for both point anomalies and contextual anomalies
/// </para>
/// <para><b>For Beginners:</b> DeepANT learns what "normal" looks like in your time series,
/// then flags anything unusual as an anomaly. It works by:
/// 1. Learning to predict the next value based on past values
/// 2. Comparing actual values to predictions
/// 3. Marking large prediction errors as anomalies
///
/// Think of it like a system that learns your daily routine - if you suddenly do something
/// very different, it notices and flags it as unusual.
/// </para>
/// </remarks>
public class DeepANT<T> : TimeSeriesModelBase<T>
{
    private readonly DeepANTOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    // CNN layers
    private List<ConvLayer<T>> _convLayers = new List<ConvLayer<T>>();
    private Matrix<T> _fcWeights = new Matrix<T>(0, 0);
    private Vector<T> _fcBias = new Vector<T>(0);

    // Anomaly detection threshold
    private T _anomalyThreshold;

    /// <summary>
    /// Initializes a new instance of the DeepANT class.
    /// </summary>
    public DeepANT(DeepANTOptions<T>? options = null)
        : this(options ?? new DeepANTOptions<T>(), initializeModel: true)
    {
    }

    /// <summary>
    /// Private constructor for proper options instance management.
    /// </summary>
    private DeepANT(DeepANTOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _convLayers = new List<ConvLayer<T>>();
        _anomalyThreshold = _numOps.FromDouble(3.0); // 3 sigma by default

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        var random = new Random(42);

        // Initialize convolutional layers with different seeds
        _convLayers.Clear();
        _convLayers.Add(new ConvLayer<T>(_options.WindowSize, 32, 3, seed: 42));
        _convLayers.Add(new ConvLayer<T>(32, 32, 3, seed: 1042));

        // Initialize fully connected output layer
        double stddev = Math.Sqrt(2.0 / 32);
        _fcWeights = new Matrix<T>(1, 32);
        for (int i = 0; i < _fcWeights.Rows; i++)
            for (int j = 0; j < _fcWeights.Columns; j++)
                _fcWeights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _fcBias = new Vector<T>(1);
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        List<T> predictionErrors = new List<T>();

        // Training loop with batch processing
        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            predictionErrors.Clear();

            // Process in batches using BatchSize
            for (int batchStart = 0; batchStart < x.Rows; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, x.Rows);

                for (int i = batchStart; i < batchEnd; i++)
                {
                    Vector<T> input = x.GetRow(i);
                    T target = y[i];
                    T prediction = PredictSingle(input);

                    // Compute prediction error
                    T error = _numOps.Subtract(target, prediction);
                    predictionErrors.Add(_numOps.Abs(error));
                }

                // Update weights once per batch (instead of periodically)
                if (batchEnd > batchStart)
                {
                    // Use a sample from the batch for gradient computation
                    Vector<T> sampleInput = x.GetRow(batchStart);
                    T sampleTarget = y[batchStart];
                    UpdateWeightsNumerically(sampleInput, sampleTarget, learningRate);
                }
            }
        }

        // Compute anomaly threshold based on training errors
        if (predictionErrors.Count > 0)
        {
            // Calculate mean and std of errors
            T mean = _numOps.Zero;
            foreach (var error in predictionErrors)
                mean = _numOps.Add(mean, error);
            mean = _numOps.Divide(mean, _numOps.FromDouble(predictionErrors.Count));

            T variance = _numOps.Zero;
            foreach (var error in predictionErrors)
            {
                T diff = _numOps.Subtract(error, mean);
                variance = _numOps.Add(variance, _numOps.Multiply(diff, diff));
            }
            variance = _numOps.Divide(variance, _numOps.FromDouble(predictionErrors.Count));
            T std = _numOps.Sqrt(variance);

            // Threshold = mean + 3 * std
            _anomalyThreshold = _numOps.Add(mean, _numOps.Multiply(_numOps.FromDouble(3.0), std));
        }
    }

    private void UpdateWeightsNumerically(Vector<T> input, T target, T learningRate)
    {
        T epsilon = _numOps.FromDouble(1e-5);
        T twoEpsilon = _numOps.Multiply(_numOps.FromDouble(2.0), epsilon);

        // Update ALL weights in the FC layer
        for (int i = 0; i < _fcWeights.Columns; i++)
        {
            T original = _fcWeights[0, i];

            _fcWeights[0, i] = _numOps.Add(original, epsilon);
            T predPlus = PredictSingle(input);
            T lossPlus = _numOps.Multiply(
                _numOps.Subtract(target, predPlus),
                _numOps.Subtract(target, predPlus)
            );

            _fcWeights[0, i] = _numOps.Subtract(original, epsilon);
            T predMinus = PredictSingle(input);
            T lossMinus = _numOps.Multiply(
                _numOps.Subtract(target, predMinus),
                _numOps.Subtract(target, predMinus)
            );

            _fcWeights[0, i] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            _fcWeights[0, i] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }

        // Also update FC bias
        for (int b = 0; b < _fcBias.Length; b++)
        {
            T original = _fcBias[b];

            _fcBias[b] = _numOps.Add(original, epsilon);
            T predPlus = PredictSingle(input);
            T lossPlus = _numOps.Multiply(
                _numOps.Subtract(target, predPlus),
                _numOps.Subtract(target, predPlus)
            );

            _fcBias[b] = _numOps.Subtract(original, epsilon);
            T predMinus = PredictSingle(input);
            T lossMinus = _numOps.Multiply(
                _numOps.Subtract(target, predMinus),
                _numOps.Subtract(target, predMinus)
            );

            _fcBias[b] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            _fcBias[b] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }
    }

    public override T PredictSingle(Vector<T> input)
    {
        // Forward pass through convolutional layers
        Vector<T> features = input.Clone();

        foreach (var conv in _convLayers)
        {
            features = conv.Forward(features);
        }

        // Fully connected output using features directly
        T output = _fcBias[0];
        for (int j = 0; j < Math.Min(_fcWeights.Columns, features.Length); j++)
        {
            output = _numOps.Add(output, _numOps.Multiply(_fcWeights[0, j], features[j]));
        }

        return output;
    }

    /// <summary>
    /// Detects anomalies in a time series.
    /// </summary>
    /// <param name="data">Time series data.</param>
    /// <returns>Boolean array where true indicates an anomaly.</returns>
    public bool[] DetectAnomalies(Vector<T> data)
    {
        if (data.Length <= _options.WindowSize)
            throw new ArgumentException($"Data length must be greater than {_options.WindowSize}");

        // Each anomaly score corresponds to comparing prediction from window ending at i with actual value at i
        int numResults = data.Length - _options.WindowSize;
        bool[] anomalies = new bool[numResults];

        for (int i = 0; i < numResults; i++)
        {
            // Extract window ending at position i + WindowSize - 1
            Vector<T> window = new Vector<T>(_options.WindowSize);
            for (int j = 0; j < _options.WindowSize; j++)
                window[j] = data[i + j];

            // Predict next value and compare with actual
            T prediction = PredictSingle(window);
            T actual = data[i + _options.WindowSize];
            T error = _numOps.Abs(_numOps.Subtract(actual, prediction));

            anomalies[i] = _numOps.GreaterThan(error, _anomalyThreshold);
        }

        return anomalies;
    }

    /// <summary>
    /// Computes anomaly scores for a time series.
    /// </summary>
    public Vector<T> ComputeAnomalyScores(Vector<T> data)
    {
        if (data.Length <= _options.WindowSize)
            throw new ArgumentException($"Data length must be greater than {_options.WindowSize}");

        // Each score corresponds to comparing prediction from window ending at i with actual value at i
        int numResults = data.Length - _options.WindowSize;
        var scores = new Vector<T>(numResults);

        for (int i = 0; i < numResults; i++)
        {
            Vector<T> window = new Vector<T>(_options.WindowSize);
            for (int j = 0; j < _options.WindowSize; j++)
                window[j] = data[i + j];

            T prediction = PredictSingle(window);
            T actual = data[i + _options.WindowSize];
            T error = _numOps.Abs(_numOps.Subtract(actual, prediction));
            scores[i] = error;
        }

        return scores;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.WindowSize);
        writer.Write(Convert.ToDouble(_anomalyThreshold));

        // Serialize conv layers
        writer.Write(_convLayers.Count);
        foreach (var conv in _convLayers)
        {
            conv.Serialize(writer);
        }

        // Serialize FC weights
        writer.Write(_fcWeights.Rows);
        writer.Write(_fcWeights.Columns);
        for (int i = 0; i < _fcWeights.Rows; i++)
            for (int j = 0; j < _fcWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_fcWeights[i, j]));

        writer.Write(_fcBias.Length);
        for (int i = 0; i < _fcBias.Length; i++)
            writer.Write(Convert.ToDouble(_fcBias[i]));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        int savedWindowSize = reader.ReadInt32();
        if (savedWindowSize != _options.WindowSize)
        {
            throw new InvalidOperationException(
                $"Serialized WindowSize ({savedWindowSize}) doesn't match options ({_options.WindowSize})");
        }
        _anomalyThreshold = _numOps.FromDouble(reader.ReadDouble());

        // Deserialize conv layers
        int convLayerCount = reader.ReadInt32();
        _convLayers.Clear();
        for (int i = 0; i < convLayerCount; i++)
        {
            _convLayers.Add(ConvLayer<T>.Deserialize(reader));
        }

        // Deserialize FC weights
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _fcWeights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                _fcWeights[i, j] = _numOps.FromDouble(reader.ReadDouble());

        int biasLen = reader.ReadInt32();
        _fcBias = new Vector<T>(biasLen);
        for (int i = 0; i < biasLen; i++)
            _fcBias[i] = _numOps.FromDouble(reader.ReadDouble());
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DeepANT",
            ModelType = ModelType.TimeSeriesRegression,
            Description = "Deep learning for anomaly detection in time series using CNN",
            Complexity = ParameterCount,
            FeatureCount = _options.WindowSize,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "WindowSize", _options.WindowSize },
                { "AnomalyThreshold", Convert.ToDouble(_anomalyThreshold) }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new DeepANT<T>(new DeepANTOptions<T>(_options));
    }

    public override int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var conv in _convLayers)
                count += conv.ParameterCount;
            count += _fcWeights.Rows * _fcWeights.Columns + _fcBias.Length;
            return count;
        }
    }
}

/// <summary>
/// Options for DeepANT model.
/// </summary>
public class DeepANTOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int WindowSize { get; set; } = 30;
    public double LearningRate { get; set; } = 0.001;
    public int Epochs { get; set; } = 50;
    public int BatchSize { get; set; } = 32;

    public DeepANTOptions() { }

    public DeepANTOptions(DeepANTOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Copy DeepANT-specific properties
        WindowSize = other.WindowSize;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;

        // Copy TimeSeriesRegressionOptions properties
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;

        // Copy RegressionOptions properties
        DecompositionMethod = other.DecompositionMethod;
        UseIntercept = other.UseIntercept;
    }
}

/// <summary>
/// Simplified 1D convolutional layer.
/// </summary>
internal class ConvLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private int _outputChannels;
    private int _kernelSize;
    private Matrix<T> _kernels;
    private Vector<T> _biases;

    public int ParameterCount => _kernels.Rows * _kernels.Columns + _biases.Length;

    public ConvLayer(int inputChannels, int outputChannels, int kernelSize, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;

        var random = new Random(seed);
        // Use kernelSize weights per output channel for 1D convolution
        double stddev = Math.Sqrt(2.0 / kernelSize);

        _kernels = new Matrix<T>(outputChannels, kernelSize);
        for (int i = 0; i < _kernels.Rows; i++)
            for (int j = 0; j < _kernels.Columns; j++)
                _kernels[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _biases = new Vector<T>(outputChannels);
    }

    /// <summary>
    /// Creates a ConvLayer for deserialization.
    /// </summary>
    private ConvLayer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _outputChannels = 0;
        _kernelSize = 0;
        _kernels = new Matrix<T>(0, 0);
        _biases = new Vector<T>(0);
    }

    public Vector<T> Forward(Vector<T> input)
    {
        // Proper 1D convolution with global average pooling
        // For each output channel, slide kernel across input and aggregate
        var output = new Vector<T>(_outputChannels);

        // Number of valid convolution positions (no padding)
        int numPositions = Math.Max(1, input.Length - _kernelSize + 1);

        for (int outChannel = 0; outChannel < _outputChannels; outChannel++)
        {
            T channelSum = _numOps.Zero;

            // Slide kernel across all valid positions
            for (int pos = 0; pos < numPositions; pos++)
            {
                T positionSum = _biases[outChannel];

                // Apply kernel at this position - use all kernelSize weights
                for (int k = 0; k < _kernelSize && (pos + k) < input.Length; k++)
                {
                    T weight = _kernels[outChannel, k];
                    T inputVal = input[pos + k];
                    positionSum = _numOps.Add(positionSum, _numOps.Multiply(weight, inputVal));
                }

                // ReLU activation at each position
                T activated = _numOps.GreaterThan(positionSum, _numOps.Zero) ? positionSum : _numOps.Zero;
                channelSum = _numOps.Add(channelSum, activated);
            }

            // Global average pooling: average over all positions
            output[outChannel] = _numOps.Divide(channelSum, _numOps.FromDouble(numPositions));
        }

        return output;
    }

    /// <summary>
    /// Serializes the convolutional layer weights.
    /// </summary>
    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_outputChannels);
        writer.Write(_kernelSize);

        // Serialize kernels
        writer.Write(_kernels.Rows);
        writer.Write(_kernels.Columns);
        for (int i = 0; i < _kernels.Rows; i++)
            for (int j = 0; j < _kernels.Columns; j++)
                writer.Write(Convert.ToDouble(_kernels[i, j]));

        // Serialize biases
        writer.Write(_biases.Length);
        for (int i = 0; i < _biases.Length; i++)
            writer.Write(Convert.ToDouble(_biases[i]));
    }

    /// <summary>
    /// Deserializes a convolutional layer from binary data.
    /// </summary>
    public static ConvLayer<T> Deserialize(BinaryReader reader)
    {
        var layer = new ConvLayer<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        layer._outputChannels = reader.ReadInt32();
        layer._kernelSize = reader.ReadInt32();

        // Deserialize kernels
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        layer._kernels = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                layer._kernels[i, j] = numOps.FromDouble(reader.ReadDouble());

        // Deserialize biases
        int biasLen = reader.ReadInt32();
        layer._biases = new Vector<T>(biasLen);
        for (int i = 0; i < biasLen; i++)
            layer._biases[i] = numOps.FromDouble(reader.ReadDouble());

        return layer;
    }
}
