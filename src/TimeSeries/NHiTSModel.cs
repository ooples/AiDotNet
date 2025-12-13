namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements N-HiTS (Neural Hierarchical Interpolation for Time Series) for efficient long-horizon forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// N-HiTS is an evolution of N-BEATS that addresses limitations in long-horizon forecasting through:
/// </para>
/// <list type="bullet">
/// <item>Multi-rate data sampling via hierarchical interpolation</item>
/// <item>Stack-specific input pooling to capture patterns at different frequencies</item>
/// <item>More efficient parameterization compared to N-BEATS</item>
/// <item>Interpolation-based basis functions for smoother predictions</item>
/// </list>
/// <para>
/// Original paper: Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023).
/// </para>
/// <para><b>For Beginners:</b> N-HiTS improves upon N-BEATS by using a "zoom lens" approach to time series.
/// It looks at your data at three different zoom levels:
/// - Zoomed out (low resolution): Captures long-term trends like yearly seasonality
/// - Medium zoom: Captures medium-term patterns like monthly cycles
/// - Zoomed in (high resolution): Captures short-term fluctuations like daily variations
///
/// By combining insights from all three levels, it produces more accurate forecasts,
/// especially for predicting far into the future.
/// </para>
/// </remarks>
public class NHiTSModel<T> : TimeSeriesModelBase<T>
{
    private readonly NHiTSOptions<T> _options;
    private readonly INumericOperations<T> _numOps;
    private List<NHiTSStack<T>> _stacks;

    /// <summary>
    /// Initializes a new instance of the NHiTSModel class.
    /// </summary>
    /// <param name="options">Configuration options for N-HiTS.</param>
    public NHiTSModel(NHiTSOptions<T>? options = null)
        : base(options ?? new NHiTSOptions<T>())
    {
        _options = options ?? new NHiTSOptions<T>();
        _numOps = MathHelper.GetNumericOperations<T>();
        _stacks = new List<NHiTSStack<T>>();

        ValidateNHiTSOptions();
        InitializeStacks();
    }

    /// <summary>
    /// Validates N-HiTS specific options.
    /// </summary>
    private void ValidateNHiTSOptions()
    {
        if (_options.NumStacks <= 0)
            throw new ArgumentException("Number of stacks must be positive.");

        if (_options.PoolingKernelSizes?.Length != _options.NumStacks)
            throw new ArgumentException($"Pooling kernel sizes length must match number of stacks ({_options.NumStacks}).");

        if (_options.LookbackWindow <= 0)
            throw new ArgumentException("Lookback window must be positive.");

        if (_options.ForecastHorizon <= 0)
            throw new ArgumentException("Forecast horizon must be positive.");
    }

    /// <summary>
    /// Initializes all stacks with their respective pooling and interpolation configurations.
    /// </summary>
    private void InitializeStacks()
    {
        _stacks.Clear();

        for (int i = 0; i < _options.NumStacks; i++)
        {
            int poolingSize = _options.PoolingKernelSizes![i];
            int downsampledLength = _options.LookbackWindow / poolingSize;

            var stack = new NHiTSStack<T>(
                downsampledLength > 0 ? downsampledLength : 1,
                _options.ForecastHorizon,
                _options.HiddenLayerSize,
                _options.NumHiddenLayers,
                _options.NumBlocksPerStack,
                poolingSize
            );

            _stacks.Add(stack);
        }
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        int numSamples = x.Rows;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            T epochLoss = _numOps.Zero;

            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);
                T batchLoss = ComputeBatchLoss(x, y, batchStart, batchEnd);
                epochLoss = _numOps.Add(epochLoss, batchLoss);

                // Simplified weight update
                UpdateStackWeights(x, y, batchStart, batchEnd, learningRate);
            }
        }
    }

    /// <summary>
    /// Computes the mean squared error loss for a batch.
    /// </summary>
    private T ComputeBatchLoss(Matrix<T> x, Vector<T> y, int batchStart, int batchEnd)
    {
        T totalLoss = _numOps.Zero;
        int batchSize = batchEnd - batchStart;

        for (int i = batchStart; i < batchEnd; i++)
        {
            Vector<T> input = x.GetRow(i);
            T prediction = PredictSingle(input);
            T error = _numOps.Subtract(prediction, y[i]);
            T squaredError = _numOps.Multiply(error, error);
            totalLoss = _numOps.Add(totalLoss, squaredError);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Updates stack weights using numerical gradients.
    /// </summary>
    private void UpdateStackWeights(Matrix<T> x, Vector<T> y, int batchStart, int batchEnd, T learningRate)
    {
        T epsilon = _numOps.FromDouble(1e-6);

        // Update parameters for first stack only (for efficiency)
        if (_stacks.Count > 0)
        {
            var stack = _stacks[0];
            Vector<T> stackParams = stack.GetParameters();
            int sampleSize = Math.Min(20, stackParams.Length);

            for (int i = 0; i < sampleSize; i++)
            {
                T original = stackParams[i];

                stackParams[i] = _numOps.Add(original, epsilon);
                stack.SetParameters(stackParams);
                T lossPlus = ComputeBatchLoss(x, y, batchStart, batchEnd);

                stackParams[i] = _numOps.Subtract(original, epsilon);
                stack.SetParameters(stackParams);
                T lossMinus = ComputeBatchLoss(x, y, batchStart, batchEnd);

                stackParams[i] = original;

                T gradient = _numOps.Divide(
                    _numOps.Subtract(lossPlus, lossMinus),
                    _numOps.Multiply(_numOps.FromDouble(2.0), epsilon)
                );

                stackParams[i] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
            }

            stack.SetParameters(stackParams);
        }
    }

    public override T PredictSingle(Vector<T> input)
    {
        Vector<T> forecast = ForecastHorizon(input);
        return forecast[0]; // Return first step
    }

    /// <summary>
    /// Generates forecasts for the full horizon using hierarchical processing.
    /// </summary>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

        // Process through each stack
        for (int stackIdx = 0; stackIdx < _stacks.Count; stackIdx++)
        {
            // Apply pooling to input
            int kernelSize = _options.PoolingKernelSizes![stackIdx];
            Vector<T> pooledInput = ApplyPooling(input, kernelSize, _options.PoolingModes![stackIdx]);

            // Get forecast from this stack
            Vector<T> stackForecast = _stacks[stackIdx].Forward(pooledInput);

            // Interpolate to full forecast horizon if needed
            Vector<T> interpolatedForecast = ApplyInterpolation(stackForecast, _options.ForecastHorizon, _options.InterpolationModes![stackIdx]);

            // Add to aggregated forecast
            for (int i = 0; i < _options.ForecastHorizon; i++)
            {
                aggregatedForecast[i] = _numOps.Add(aggregatedForecast[i], interpolatedForecast[i]);
            }
        }

        return aggregatedForecast;
    }

    /// <summary>
    /// Applies pooling to downsample the input.
    /// </summary>
    private Vector<T> ApplyPooling(Vector<T> input, int kernelSize, string mode)
    {
        if (kernelSize <= 1)
            return input.Clone();

        int outputLength = (input.Length + kernelSize - 1) / kernelSize;
        var pooled = new Vector<T>(outputLength);

        for (int i = 0; i < outputLength; i++)
        {
            int start = i * kernelSize;
            int end = Math.Min(start + kernelSize, input.Length);

            if (mode == "MaxPool")
            {
                T maxVal = input[start];
                for (int j = start + 1; j < end; j++)
                {
                    if (_numOps.GreaterThan(input[j], maxVal))
                        maxVal = input[j];
                }
                pooled[i] = maxVal;
            }
            else // AvgPool
            {
                T sum = _numOps.Zero;
                for (int j = start; j < end; j++)
                {
                    sum = _numOps.Add(sum, input[j]);
                }
                pooled[i] = _numOps.Divide(sum, _numOps.FromDouble(end - start));
            }
        }

        return pooled;
    }

    /// <summary>
    /// Applies interpolation to upsample the forecast.
    /// </summary>
    private Vector<T> ApplyInterpolation(Vector<T> input, int targetLength, string mode)
    {
        if (input.Length == targetLength)
            return input.Clone();

        var interpolated = new Vector<T>(targetLength);

        if (mode == "Linear")
        {
            double scale = (double)(input.Length - 1) / (targetLength - 1);

            for (int i = 0; i < targetLength; i++)
            {
                double srcIdx = i * scale;
                int idx1 = (int)Math.Floor(srcIdx);
                int idx2 = Math.Min(idx1 + 1, input.Length - 1);
                double weight = srcIdx - idx1;

                T val1 = input[idx1];
                T val2 = input[idx2];
                T interpolatedVal = _numOps.Add(
                    _numOps.Multiply(val1, _numOps.FromDouble(1.0 - weight)),
                    _numOps.Multiply(val2, _numOps.FromDouble(weight))
                );

                interpolated[i] = interpolatedVal;
            }
        }
        else
        {
            // Fallback: simple repetition
            for (int i = 0; i < targetLength; i++)
            {
                int srcIdx = (i * input.Length) / targetLength;
                interpolated[i] = input[srcIdx];
            }
        }

        return interpolated;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.NumStacks);
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);

        writer.Write(_stacks.Count);
        foreach (var stack in _stacks)
        {
            Vector<T> parameters = stack.GetParameters();
            writer.Write(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
                writer.Write(Convert.ToDouble(parameters[i]));
        }
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.NumStacks = reader.ReadInt32();
        _options.LookbackWindow = reader.ReadInt32();
        _options.ForecastHorizon = reader.ReadInt32();

        InitializeStacks();

        int stackCount = reader.ReadInt32();
        for (int s = 0; s < stackCount && s < _stacks.Count; s++)
        {
            int paramCount = reader.ReadInt32();
            var parameters = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
                parameters[i] = _numOps.FromDouble(reader.ReadDouble());
            _stacks[s].SetParameters(parameters);
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "N-HiTS",
            ModelType = ModelType.TimeSeriesRegression,
            Description = "Neural Hierarchical Interpolation for Time Series with multi-rate sampling",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumStacks", _options.NumStacks },
                { "LookbackWindow", _options.LookbackWindow },
                { "ForecastHorizon", _options.ForecastHorizon },
                { "PoolingKernelSizes", _options.PoolingKernelSizes! }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new NHiTSModel<T>(new NHiTSOptions<T>(_options));
    }

    public override int ParameterCount
    {
        get
        {
            int total = 0;
            foreach (var stack in _stacks)
                total += stack.ParameterCount;
            return total;
        }
    }
}

/// <summary>
/// Represents a single stack in the N-HiTS architecture.
/// </summary>
internal class NHiTSStack<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _inputLength;
    private readonly int _outputLength;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly int _poolingSize;
    private List<Matrix<T>> _weights;
    private List<Vector<T>> _biases;

    public int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var w in _weights)
                count += w.Rows * w.Columns;
            foreach (var b in _biases)
                count += b.Length;
            return count;
        }
    }

    public NHiTSStack(int inputLength, int outputLength, int hiddenSize, int numLayers, int numBlocks, int poolingSize)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _inputLength = inputLength;
        _outputLength = outputLength;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _poolingSize = poolingSize;
        _weights = new List<Matrix<T>>();
        _biases = new List<Vector<T>>();

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        var random = new Random(42);

        // Input layer
        double stddev = Math.Sqrt(2.0 / (_inputLength + _hiddenSize));
        _weights.Add(CreateRandomMatrix(_hiddenSize, _inputLength, stddev, random));
        _biases.Add(new Vector<T>(_hiddenSize));

        // Hidden layers
        for (int i = 1; i < _numLayers; i++)
        {
            stddev = Math.Sqrt(2.0 / (_hiddenSize + _hiddenSize));
            _weights.Add(CreateRandomMatrix(_hiddenSize, _hiddenSize, stddev, random));
            _biases.Add(new Vector<T>(_hiddenSize));
        }

        // Output layer
        stddev = Math.Sqrt(2.0 / (_hiddenSize + _outputLength));
        _weights.Add(CreateRandomMatrix(_outputLength, _hiddenSize, stddev, random));
        _biases.Add(new Vector<T>(_outputLength));
    }

    private Matrix<T> CreateRandomMatrix(int rows, int cols, double stddev, Random random)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return matrix;
    }

    public Vector<T> Forward(Vector<T> input)
    {
        Vector<T> x = input.Clone();

        // Ensure input matches expected size
        if (x.Length != _inputLength)
        {
            var resized = new Vector<T>(_inputLength);
            for (int i = 0; i < _inputLength; i++)
            {
                int srcIdx = (i * x.Length) / _inputLength;
                resized[i] = x[Math.Min(srcIdx, x.Length - 1)];
            }
            x = resized;
        }

        // Forward through layers
        for (int layer = 0; layer < _weights.Count; layer++)
        {
            Vector<T> linear = new Vector<T>(_weights[layer].Rows);
            for (int i = 0; i < _weights[layer].Rows; i++)
            {
                T sum = _biases[layer][i];
                for (int j = 0; j < Math.Min(x.Length, _weights[layer].Columns); j++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(_weights[layer][i, j], x[j]));
                }
                linear[i] = sum;
            }

            // ReLU activation for all but last layer
            if (layer < _weights.Count - 1)
            {
                x = new Vector<T>(linear.Length);
                for (int i = 0; i < linear.Length; i++)
                {
                    x[i] = _numOps.GreaterThan(linear[i], _numOps.Zero) ? linear[i] : _numOps.Zero;
                }
            }
            else
            {
                x = linear;
            }
        }

        return x;
    }

    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        foreach (var weight in _weights)
            for (int i = 0; i < weight.Rows; i++)
                for (int j = 0; j < weight.Columns; j++)
                    allParams.Add(weight[i, j]);

        foreach (var bias in _biases)
            for (int i = 0; i < bias.Length; i++)
                allParams.Add(bias[i]);

        return new Vector<T>(allParams.ToArray());
    }

    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        foreach (var weight in _weights)
        {
            for (int i = 0; i < weight.Rows; i++)
                for (int j = 0; j < weight.Columns; j++)
                    if (idx < parameters.Length)
                        weight[i, j] = parameters[idx++];
        }

        foreach (var bias in _biases)
        {
            for (int i = 0; i < bias.Length; i++)
                if (idx < parameters.Length)
                    bias[i] = parameters[idx++];
        }
    }
}
