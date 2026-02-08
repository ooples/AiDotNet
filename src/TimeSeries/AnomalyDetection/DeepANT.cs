using AiDotNet.Tensors.LinearAlgebra;

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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    // CNN layers (Tensor-based)
    private readonly List<ConvLayerTensor<T>> _convLayers = new List<ConvLayerTensor<T>>();

    // Fully connected layer (Tensor-based)
    private Tensor<T> _fcWeights;      // [1, numChannels]
    private Tensor<T> _fcBias;         // [1]

    // Gradient accumulators for batch training
    private Tensor<T> _fcWeightsGrad;
    private Tensor<T> _fcBiasGrad;

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
        _convLayers = new List<ConvLayerTensor<T>>();
        _anomalyThreshold = _numOps.FromDouble(3.0); // 3 sigma by default

        // Initialize with default tensors
        _fcWeights = new Tensor<T>(new[] { 1, 1 });
        _fcBias = new Tensor<T>(new[] { 1 });
        _fcWeightsGrad = new Tensor<T>(new[] { 1, 1 });
        _fcBiasGrad = new Tensor<T>(new[] { 1 });

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        int numChannels = 32;

        // Initialize convolutional layers (fixed random features)
        _convLayers.Clear();
        _convLayers.Add(new ConvLayerTensor<T>(numChannels, 3, seed: 42));
        _convLayers.Add(new ConvLayerTensor<T>(numChannels, 3, seed: 1042));

        // Initialize fully connected output layer with Xavier initialization
        double stddev = Math.Sqrt(2.0 / numChannels);
        _fcWeights = new Tensor<T>(new[] { 1, numChannels });
        for (int j = 0; j < numChannels; j++)
            _fcWeights[j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _fcBias = new Tensor<T>(new[] { 1 });
        _fcBias[0] = _numOps.Zero;

        // Initialize gradient accumulators
        _fcWeightsGrad = new Tensor<T>(new[] { 1, numChannels });
        _fcBiasGrad = new Tensor<T>(new[] { 1 });
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        List<T> predictionErrors = new List<T>();

        // Training loop with batch processing and proper backpropagation
        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            predictionErrors.Clear();

            // Process in batches
            for (int batchStart = 0; batchStart < x.Rows; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, x.Rows);
                int batchSize = batchEnd - batchStart;

                // Reset gradient accumulators
                ResetGradients();

                // Accumulate gradients for batch
                for (int i = batchStart; i < batchEnd; i++)
                {
                    Vector<T> input = x.GetRow(i);
                    T target = y[i];

                    // Forward pass with feature caching
                    var (prediction, features) = ForwardWithCache(input);

                    // Compute prediction error for anomaly threshold
                    T error = _numOps.Subtract(target, prediction);
                    predictionErrors.Add(_numOps.Abs(error));

                    // Compute gradients analytically and accumulate
                    ComputeAndAccumulateGradients(features, error);
                }

                // Apply accumulated gradients
                ApplyGradients(learningRate, batchSize);
            }
        }

        // Compute anomaly threshold based on training errors
        ComputeAnomalyThreshold(predictionErrors);
    }

    private void ResetGradients()
    {
        for (int i = 0; i < _fcWeightsGrad.Length; i++)
            _fcWeightsGrad[i] = _numOps.Zero;
        for (int i = 0; i < _fcBiasGrad.Length; i++)
            _fcBiasGrad[i] = _numOps.Zero;
    }

    /// <summary>
    /// Forward pass that returns both prediction and cached features for backpropagation.
    /// </summary>
    private (T prediction, Tensor<T> features) ForwardWithCache(Vector<T> input)
    {
        // Forward pass through convolutional layers (fixed random features)
        Tensor<T> features = new Tensor<T>(new[] { input.Length });
        for (int i = 0; i < input.Length; i++)
            features[i] = input[i];

        foreach (var conv in _convLayers)
        {
            features = conv.Forward(features);
        }

        // Fully connected output: output = sum(weights * features) + bias
        T output = _fcBias[0];
        int numFeatures = Math.Min(_fcWeights.Length, features.Length);
        for (int j = 0; j < numFeatures; j++)
        {
            output = _numOps.Add(output, _numOps.Multiply(_fcWeights[j], features[j]));
        }

        return (output, features);
    }

    /// <summary>
    /// Computes analytical gradients for the FC layer and accumulates them.
    /// </summary>
    /// <remarks>
    /// <para><b>Design Note:</b> This implementation uses proper analytical gradients for the FC layer.
    /// The convolutional layers remain fixed (Random Features approach) as this has been shown to be
    /// effective for time series anomaly detection while being computationally efficient.</para>
    /// <para>
    /// For MSE loss L = (target - prediction)^2, the gradients are:
    /// - dL/dW_j = -2 * error * features[j]
    /// - dL/dBias = -2 * error
    /// </para>
    /// </remarks>
    private void ComputeAndAccumulateGradients(Tensor<T> features, T error)
    {
        // dL/d(prediction) = -2 * error for MSE loss
        T dLoss = _numOps.Multiply(_numOps.FromDouble(-2.0), error);

        // Accumulate gradients for FC weights: dL/dW_j = dLoss * features[j]
        int numFeatures = Math.Min(_fcWeightsGrad.Length, features.Length);
        for (int j = 0; j < numFeatures; j++)
        {
            T grad = _numOps.Multiply(dLoss, features[j]);
            _fcWeightsGrad[j] = _numOps.Add(_fcWeightsGrad[j], grad);
        }

        // Accumulate gradient for FC bias: dL/dBias = dLoss
        _fcBiasGrad[0] = _numOps.Add(_fcBiasGrad[0], dLoss);
    }

    /// <summary>
    /// Applies accumulated gradients with SGD.
    /// </summary>
    private void ApplyGradients(T learningRate, int batchSize)
    {
        T batchSizeT = _numOps.FromDouble(batchSize);

        // Update FC weights
        for (int j = 0; j < _fcWeights.Length; j++)
        {
            T avgGrad = _numOps.Divide(_fcWeightsGrad[j], batchSizeT);
            T update = _numOps.Multiply(learningRate, avgGrad);
            _fcWeights[j] = _numOps.Subtract(_fcWeights[j], update);
        }

        // Update FC bias
        for (int b = 0; b < _fcBias.Length; b++)
        {
            T avgGrad = _numOps.Divide(_fcBiasGrad[b], batchSizeT);
            T update = _numOps.Multiply(learningRate, avgGrad);
            _fcBias[b] = _numOps.Subtract(_fcBias[b], update);
        }
    }

    /// <summary>
    /// Computes anomaly threshold based on training prediction errors.
    /// </summary>
    private void ComputeAnomalyThreshold(List<T> predictionErrors)
    {
        if (predictionErrors.Count == 0) return;

        // Calculate mean
        T mean = _numOps.Zero;
        foreach (var error in predictionErrors)
            mean = _numOps.Add(mean, error);
        mean = _numOps.Divide(mean, _numOps.FromDouble(predictionErrors.Count));

        // Calculate variance and std
        T variance = predictionErrors.Select(error =>
        {
            T diff = _numOps.Subtract(error, mean);
            return _numOps.Multiply(diff, diff);
        }).Aggregate(_numOps.Zero, (sum, val) => _numOps.Add(sum, val));
        variance = _numOps.Divide(variance, _numOps.FromDouble(predictionErrors.Count));
        T std = _numOps.Sqrt(variance);

        // Threshold = mean + 3 * std
        _anomalyThreshold = _numOps.Add(mean, _numOps.Multiply(_numOps.FromDouble(3.0), std));
    }

    public override T PredictSingle(Vector<T> input)
    {
        // Forward pass through convolutional layers
        Tensor<T> features = new Tensor<T>(new[] { input.Length });
        for (int i = 0; i < input.Length; i++)
            features[i] = input[i];

        foreach (var conv in _convLayers)
        {
            features = conv.Forward(features);
        }

        // Fully connected output
        T output = _fcBias[0];
        int numFeatures = Math.Min(_fcWeights.Length, features.Length);
        for (int j = 0; j < numFeatures; j++)
        {
            output = _numOps.Add(output, _numOps.Multiply(_fcWeights[j], features[j]));
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

        int numResults = data.Length - _options.WindowSize;
        bool[] anomalies = new bool[numResults];

        for (int i = 0; i < numResults; i++)
        {
            Vector<T> window = new Vector<T>(_options.WindowSize);
            for (int j = 0; j < _options.WindowSize; j++)
                window[j] = data[i + j];

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
        writer.Write(_numOps.ToDouble(_anomalyThreshold));

        // Serialize conv layers
        writer.Write(_convLayers.Count);
        foreach (var conv in _convLayers)
        {
            conv.Serialize(writer);
        }

        // Serialize FC weights tensor
        writer.Write(_fcWeights.Shape.Length);
        foreach (int dim in _fcWeights.Shape)
            writer.Write(dim);
        writer.Write(_fcWeights.Length);
        for (int i = 0; i < _fcWeights.Length; i++)
            writer.Write(_numOps.ToDouble(_fcWeights[i]));

        // Serialize FC bias tensor
        writer.Write(_fcBias.Shape.Length);
        foreach (int dim in _fcBias.Shape)
            writer.Write(dim);
        writer.Write(_fcBias.Length);
        for (int i = 0; i < _fcBias.Length; i++)
            writer.Write(_numOps.ToDouble(_fcBias[i]));
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
            _convLayers.Add(ConvLayerTensor<T>.Deserialize(reader));
        }

        // Deserialize FC weights tensor
        int weightsRank = reader.ReadInt32();
        int[] weightsShape = new int[weightsRank];
        for (int i = 0; i < weightsRank; i++)
            weightsShape[i] = reader.ReadInt32();
        int weightsLength = reader.ReadInt32();
        _fcWeights = new Tensor<T>(weightsShape);
        // Clamp by tensor length but consume all serialized values to keep stream aligned
        for (int i = 0; i < weightsLength; i++)
        {
            double v = reader.ReadDouble();
            if (i < _fcWeights.Length)
                _fcWeights[i] = _numOps.FromDouble(v);
        }

        // Deserialize FC bias tensor
        int biasRank = reader.ReadInt32();
        int[] biasShape = new int[biasRank];
        for (int i = 0; i < biasRank; i++)
            biasShape[i] = reader.ReadInt32();
        int biasLength = reader.ReadInt32();
        _fcBias = new Tensor<T>(biasShape);
        // Clamp by tensor length but consume all serialized values to keep stream aligned
        for (int i = 0; i < biasLength; i++)
        {
            double v = reader.ReadDouble();
            if (i < _fcBias.Length)
                _fcBias[i] = _numOps.FromDouble(v);
        }

        // Initialize gradient accumulators
        _fcWeightsGrad = new Tensor<T>(weightsShape);
        _fcBiasGrad = new Tensor<T>(biasShape);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DeepANT",
            ModelType = AiDotNet.Enums.ModelType.TimeSeriesRegression,
            Description = "Deep learning for anomaly detection in time series using CNN",
            Complexity = ParameterCount,
            FeatureCount = _options.WindowSize,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "WindowSize", _options.WindowSize },
                { "AnomalyThreshold", _numOps.ToDouble(_anomalyThreshold) }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new DeepANT<T>(new DeepANTOptions<T>(_options), initializeModel: false);
    }

    public override int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var conv in _convLayers)
                count += conv.ParameterCount;
            count += _fcWeights.Length + _fcBias.Length;
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
/// Tensor-based 1D convolutional layer for DeepANT.
/// </summary>
/// <remarks>
/// <para>This layer uses fixed random weights (Random Features approach) which has been shown
/// to be effective for time series feature extraction while being computationally efficient.</para>
/// </remarks>
internal class ConvLayerTensor<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private int _outputChannels;
    private int _kernelSize;
    private Tensor<T> _kernels;  // [outputChannels, kernelSize]
    private Tensor<T> _biases;   // [outputChannels]

    public int ParameterCount => _kernels.Length + _biases.Length;

    public ConvLayerTensor(int outputChannels, int kernelSize, int seed = 42)
    {
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / kernelSize);

        _kernels = new Tensor<T>(new[] { outputChannels, kernelSize });
        for (int i = 0; i < _kernels.Length; i++)
            _kernels[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _biases = new Tensor<T>(new[] { outputChannels });
        for (int i = 0; i < _biases.Length; i++)
            _biases[i] = _numOps.Zero;
    }

    /// <summary>
    /// Creates a ConvLayerTensor for deserialization.
    /// </summary>
    private ConvLayerTensor()
    {
        _outputChannels = 0;
        _kernelSize = 0;
        _kernels = new Tensor<T>(new[] { 1, 1 });
        _biases = new Tensor<T>(new[] { 1 });
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // Proper 1D convolution with global average pooling
        var output = new Tensor<T>(new[] { _outputChannels });

        // Number of valid convolution positions (no padding)
        int numPositions = Math.Max(1, input.Length - _kernelSize + 1);

        for (int outChannel = 0; outChannel < _outputChannels; outChannel++)
        {
            T channelSum = _numOps.Zero;

            // Slide kernel across all valid positions
            for (int pos = 0; pos < numPositions; pos++)
            {
                T positionSum = _biases[outChannel];

                // Apply kernel at this position
                for (int k = 0; k < _kernelSize && (pos + k) < input.Length; k++)
                {
                    int kernelIdx = outChannel * _kernelSize + k;
                    T weight = _kernels[kernelIdx];
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

        // Serialize kernels tensor
        writer.Write(_kernels.Shape.Length);
        foreach (int dim in _kernels.Shape)
            writer.Write(dim);
        writer.Write(_kernels.Length);
        for (int i = 0; i < _kernels.Length; i++)
            writer.Write(_numOps.ToDouble(_kernels[i]));

        // Serialize biases tensor
        writer.Write(_biases.Shape.Length);
        foreach (int dim in _biases.Shape)
            writer.Write(dim);
        writer.Write(_biases.Length);
        for (int i = 0; i < _biases.Length; i++)
            writer.Write(_numOps.ToDouble(_biases[i]));
    }

    /// <summary>
    /// Deserializes a convolutional layer from binary data.
    /// </summary>
    public static ConvLayerTensor<T> Deserialize(BinaryReader reader)
    {
        var layer = new ConvLayerTensor<T>();

        layer._outputChannels = reader.ReadInt32();
        layer._kernelSize = reader.ReadInt32();

        // Deserialize kernels tensor
        int kernelsRank = reader.ReadInt32();
        int[] kernelsShape = new int[kernelsRank];
        for (int i = 0; i < kernelsRank; i++)
            kernelsShape[i] = reader.ReadInt32();
        int kernelsLength = reader.ReadInt32();
        layer._kernels = new Tensor<T>(kernelsShape);
        for (int i = 0; i < kernelsLength; i++)
            layer._kernels[i] = _numOps.FromDouble(reader.ReadDouble());

        // Deserialize biases tensor
        int biasesRank = reader.ReadInt32();
        int[] biasesShape = new int[biasesRank];
        for (int i = 0; i < biasesRank; i++)
            biasesShape[i] = reader.ReadInt32();
        int biasesLength = reader.ReadInt32();
        layer._biases = new Tensor<T>(biasesShape);
        for (int i = 0; i < biasesLength; i++)
            layer._biases[i] = _numOps.FromDouble(reader.ReadDouble());

        return layer;
    }
}
