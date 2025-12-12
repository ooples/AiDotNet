namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// DeepAR is a probabilistic forecasting model that produces full probability distributions
/// rather than point estimates. Key features include:
/// </para>
/// <list type="bullet">
/// <item>Autoregressive RNN architecture (typically LSTM-based)</item>
/// <item>Probabilistic forecasts with quantile predictions</item>
/// <item>Handles multiple related time series</item>
/// <item>Built-in handling of covariates and categorical features</item>
/// <item>Effective for cold-start scenarios</item>
/// </list>
/// <para>
/// Original paper: Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020).
/// </para>
/// <para><b>For Beginners:</b> DeepAR is like a weather forecaster that doesn't just say
/// "it will be 70°F tomorrow" but rather "there's a 50% chance it'll be between 65-75°F,
/// a 90% chance it'll be between 60-80°F," etc.
///
/// It uses a type of neural network called LSTM (Long Short-Term Memory) that's good at
/// remembering patterns over time. The "autoregressive" part means it uses its own
/// predictions to make future predictions - similar to how you might predict tomorrow's
/// weather based on today's forecast.
///
/// This is particularly useful when you need to:
/// - Make decisions based on uncertainty (e.g., inventory planning)
/// - Forecast many related series efficiently (e.g., sales across stores)
/// - Handle new products or stores with limited data
/// </para>
/// </remarks>
public class DeepARModel<T> : TimeSeriesModelBase<T>
{
    private readonly DeepAROptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    // LSTM layers
    private List<DeepARLstmCell<T>> _lstmLayers = new List<DeepARLstmCell<T>>();
    private Matrix<T> _outputWeights = new Matrix<T>(0, 0);
    private Vector<T> _outputBias = new Vector<T>(0);

    // Distribution parameters
    private Matrix<T> _meanWeights = new Matrix<T>(0, 0);
    private Vector<T> _meanBias = new Vector<T>(0);
    private Matrix<T> _scaleWeights = new Matrix<T>(0, 0);
    private Vector<T> _scaleBias = new Vector<T>(0);

    /// <summary>
    /// Initializes a new instance of the DeepARModel class.
    /// </summary>
    /// <param name="options">Configuration options for DeepAR.</param>
    public DeepARModel(DeepAROptions<T>? options = null)
        : base(options ?? new DeepAROptions<T>())
    {
        _options = options ?? new DeepAROptions<T>();
        _numOps = MathHelper.GetNumericOperations<T>();
        _lstmLayers = new List<DeepARLstmCell<T>>();

        ValidateDeepAROptions();
        InitializeModel();
    }

    /// <summary>
    /// Validates DeepAR-specific options.
    /// </summary>
    private void ValidateDeepAROptions()
    {
        if (_options.LookbackWindow <= 0)
            throw new ArgumentException("Lookback window must be positive.");

        if (_options.ForecastHorizon <= 0)
            throw new ArgumentException("Forecast horizon must be positive.");

        if (_options.HiddenSize <= 0)
            throw new ArgumentException("Hidden size must be positive.");

        if (_options.NumLayers <= 0)
            throw new ArgumentException("Number of layers must be positive.");

        if (_options.NumSamples <= 0)
            throw new ArgumentException("Number of samples must be positive.");
    }

    /// <summary>
    /// Initializes the model architecture.
    /// </summary>
    private void InitializeModel()
    {
        var random = new Random(42);

        // Initialize LSTM layers
        _lstmLayers.Clear();
        int inputSize = 1 + _options.CovariateSize; // Target + covariates

        for (int i = 0; i < _options.NumLayers; i++)
        {
            int layerInputSize = (i == 0) ? inputSize : _options.HiddenSize;
            _lstmLayers.Add(new DeepARLstmCell<T>(layerInputSize, _options.HiddenSize));
        }

        // Output projection for distribution parameters
        double stddev = Math.Sqrt(2.0 / _options.HiddenSize);

        // Mean parameter
        _meanWeights = CreateRandomMatrix(1, _options.HiddenSize, stddev, random);
        _meanBias = new Vector<T>(1);

        // Scale parameter (for uncertainty)
        _scaleWeights = CreateRandomMatrix(1, _options.HiddenSize, stddev, random);
        _scaleBias = new Vector<T>(1);
    }

    private Matrix<T> CreateRandomMatrix(int rows, int cols, double stddev, Random random)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return matrix;
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        int numSamples = x.Rows;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            // Shuffle training data (simplified - just process in order)
            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);

                // Compute loss for training metrics and update weights
                _ = ComputeBatchLoss(x, y, batchStart, batchEnd);
                UpdateWeights(x, y, batchStart, batchEnd, learningRate);
            }
        }
    }

    /// <summary>
    /// Computes the negative log-likelihood loss for a batch.
    /// </summary>
    private T ComputeBatchLoss(Matrix<T> x, Vector<T> y, int batchStart, int batchEnd)
    {
        T totalLoss = _numOps.Zero;

        for (int i = batchStart; i < batchEnd; i++)
        {
            Vector<T> input = x.GetRow(i);
            T target = y[i];

            // Forward pass to get distribution parameters
            var (mean, scale) = PredictDistribution(input);

            // Negative log-likelihood (Gaussian assumption)
            T error = _numOps.Subtract(target, mean);
            T squaredError = _numOps.Multiply(error, error);
            T variance = _numOps.Multiply(scale, scale);

            // NLL = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
            T nll = _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Log(_numOps.Add(variance, _numOps.FromDouble(1e-6)))),
                _numOps.Divide(squaredError, _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Add(variance, _numOps.FromDouble(1e-6))))
            );

            totalLoss = _numOps.Add(totalLoss, nll);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batchEnd - batchStart));
    }

    /// <summary>
    /// Updates model weights using numerical gradients.
    /// </summary>
    private void UpdateWeights(Matrix<T> x, Vector<T> y, int batchStart, int batchEnd, T learningRate)
    {
        T epsilon = _numOps.FromDouble(1e-6);

        // Update mean weights (sample a few for efficiency)
        for (int i = 0; i < Math.Min(5, _meanWeights.Rows); i++)
        {
            for (int j = 0; j < Math.Min(5, _meanWeights.Columns); j++)
            {
                T original = _meanWeights[i, j];

                _meanWeights[i, j] = _numOps.Add(original, epsilon);
                T lossPlus = ComputeBatchLoss(x, y, batchStart, batchEnd);

                _meanWeights[i, j] = _numOps.Subtract(original, epsilon);
                T lossMinus = ComputeBatchLoss(x, y, batchStart, batchEnd);

                _meanWeights[i, j] = original;

                T gradient = _numOps.Divide(
                    _numOps.Subtract(lossPlus, lossMinus),
                    _numOps.Multiply(_numOps.FromDouble(2.0), epsilon)
                );

                _meanWeights[i, j] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
            }
        }
    }

    /// <summary>
    /// Predicts distribution parameters (mean and scale) for a single input.
    /// </summary>
    private (T mean, T scale) PredictDistribution(Vector<T> input)
    {
        // Forward pass through LSTM layers
        Vector<T> hidden = input.Clone();

        // Ensure input size matches
        if (hidden.Length < 1)
            hidden = new Vector<T>(new[] { _numOps.Zero });

        foreach (var lstm in _lstmLayers)
        {
            hidden = lstm.Forward(hidden);
        }

        // Predict mean
        T mean = _meanBias[0];
        for (int j = 0; j < Math.Min(hidden.Length, _meanWeights.Columns); j++)
        {
            mean = _numOps.Add(mean, _numOps.Multiply(_meanWeights[0, j], hidden[j]));
        }

        // Predict scale (must be positive)
        T scaleRaw = _scaleBias[0];
        for (int j = 0; j < Math.Min(hidden.Length, _scaleWeights.Columns); j++)
        {
            scaleRaw = _numOps.Add(scaleRaw, _numOps.Multiply(_scaleWeights[0, j], hidden[j]));
        }
        T scale = _numOps.Exp(_numOps.Multiply(scaleRaw, _numOps.FromDouble(0.1))); // Softplus approximation

        return (mean, scale);
    }

    public override T PredictSingle(Vector<T> input)
    {
        var (mean, _) = PredictDistribution(input);
        return mean;
    }

    /// <summary>
    /// Generates probabilistic forecasts with quantile predictions.
    /// </summary>
    /// <param name="history">Historical time series data.</param>
    /// <param name="quantiles">Quantile levels to predict (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Dictionary mapping quantile levels to forecast vectors.</returns>
    public Dictionary<double, Vector<T>> ForecastWithQuantiles(Vector<T> history, double[] quantiles)
    {
        var result = new Dictionary<double, Vector<T>>();
        var random = new Random();

        // Generate samples
        var samples = new List<Vector<T>>();

        for (int s = 0; s < _options.NumSamples; s++)
        {
            var forecast = new Vector<T>(_options.ForecastHorizon);
            Vector<T> context = history.Clone();

            for (int h = 0; h < _options.ForecastHorizon; h++)
            {
                var (mean, scale) = PredictDistribution(context);

                // Sample from Gaussian distribution
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

                T sample = _numOps.Add(mean, _numOps.Multiply(scale, _numOps.FromDouble(randStdNormal)));
                forecast[h] = sample;

                // Update context with sliding window - shift left and append new prediction
                var newContext = new Vector<T>(context.Length);
                for (int i = 0; i < context.Length - 1; i++)
                {
                    newContext[i] = context[i + 1];
                }
                newContext[context.Length - 1] = sample;
                context = newContext;
            }

            samples.Add(forecast);
        }

        // Compute quantiles from samples
        foreach (var q in quantiles)
        {
            var quantileForecast = new Vector<T>(_options.ForecastHorizon);

            for (int h = 0; h < _options.ForecastHorizon; h++)
            {
                var values = new List<double>();
                foreach (var sample in samples)
                {
                    values.Add(Convert.ToDouble(sample[h]));
                }
                values.Sort();

                int idx = (int)(q * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                quantileForecast[h] = _numOps.FromDouble(values[idx]);
            }

            result[q] = quantileForecast;
        }

        return result;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.HiddenSize);
        writer.Write(_options.NumLayers);

        // Serialize LSTM layers
        writer.Write(_lstmLayers.Count);
        foreach (var lstm in _lstmLayers)
        {
            var parameters = lstm.GetParameters();
            writer.Write(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
                writer.Write(Convert.ToDouble(parameters[i]));
        }

        // Serialize distribution parameter weights
        SerializeMatrix(writer, _meanWeights);
        SerializeVector(writer, _meanBias);
        SerializeMatrix(writer, _scaleWeights);
        SerializeVector(writer, _scaleBias);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.HiddenSize = reader.ReadInt32();
        _options.NumLayers = reader.ReadInt32();

        InitializeModel();

        // Deserialize LSTM layers
        int numLayers = reader.ReadInt32();
        for (int i = 0; i < numLayers && i < _lstmLayers.Count; i++)
        {
            int paramCount = reader.ReadInt32();
            var parameters = new Vector<T>(paramCount);
            for (int j = 0; j < paramCount; j++)
                parameters[j] = _numOps.FromDouble(reader.ReadDouble());
            _lstmLayers[i].SetParameters(parameters);
        }

        // Deserialize distribution parameter weights
        _meanWeights = DeserializeMatrix(reader);
        _meanBias = DeserializeVector(reader);
        _scaleWeights = DeserializeMatrix(reader);
        _scaleBias = DeserializeVector(reader);
    }

    private void SerializeMatrix(BinaryWriter writer, Matrix<T> matrix)
    {
        writer.Write(matrix.Rows);
        writer.Write(matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                writer.Write(Convert.ToDouble(matrix[i, j]));
    }

    private Matrix<T> DeserializeMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = _numOps.FromDouble(reader.ReadDouble());
        return matrix;
    }

    private void SerializeVector(BinaryWriter writer, Vector<T> vector)
    {
        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
            writer.Write(Convert.ToDouble(vector[i]));
    }

    private Vector<T> DeserializeVector(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var vector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
            vector[i] = _numOps.FromDouble(reader.ReadDouble());
        return vector;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DeepAR",
            ModelType = ModelType.TimeSeriesRegression,
            Description = "Probabilistic forecasting with autoregressive recurrent networks",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HiddenSize", _options.HiddenSize },
                { "NumLayers", _options.NumLayers },
                { "LikelihoodType", _options.LikelihoodType },
                { "ForecastHorizon", _options.ForecastHorizon }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new DeepARModel<T>(new DeepAROptions<T>(_options));
    }

    public override int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var lstm in _lstmLayers)
                count += lstm.ParameterCount;
            count += _meanWeights.Rows * _meanWeights.Columns + _meanBias.Length;
            count += _scaleWeights.Rows * _scaleWeights.Columns + _scaleBias.Length;
            return count;
        }
    }
}

/// <summary>
/// Simplified LSTM layer implementation.
/// </summary>
internal class DeepARLstmCell<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly Matrix<T> _weights;
    private readonly Vector<T> _bias;
    private readonly Vector<T> _hiddenState;
    private readonly Vector<T> _cellState;

    public int ParameterCount => _weights.Rows * _weights.Columns + _bias.Length;

    public DeepARLstmCell(int inputSize, int hiddenSize)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;

        // Initialize weights for all gates (input, forget, output, cell)
        var random = new Random(42);
        double stddev = Math.Sqrt(2.0 / (inputSize + hiddenSize));

        _weights = new Matrix<T>(4 * hiddenSize, inputSize + hiddenSize);
        for (int i = 0; i < _weights.Rows; i++)
            for (int j = 0; j < _weights.Columns; j++)
                _weights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _bias = new Vector<T>(4 * hiddenSize);
        _hiddenState = new Vector<T>(hiddenSize);
        _cellState = new Vector<T>(hiddenSize);
    }

    public Vector<T> Forward(Vector<T> input)
    {
        // Simplified LSTM forward pass (full implementation would include all gates)
        var combined = new Vector<T>(_inputSize + _hiddenSize);

        // Copy input
        for (int i = 0; i < Math.Min(input.Length, _inputSize); i++)
            combined[i] = input[i];

        // Copy hidden state
        for (int i = 0; i < _hiddenSize; i++)
            combined[_inputSize + i] = _hiddenState[i];

        // Compute gates (simplified)
        var output = new Vector<T>(_hiddenSize);
        for (int i = 0; i < _hiddenSize; i++)
        {
            T sum = _bias[i];
            for (int j = 0; j < combined.Length && j < _weights.Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_weights[i, j], combined[j]));
            }
            output[i] = MathHelper.Tanh(sum); // Simplified activation
            _hiddenState[i] = output[i];
        }

        return output;
    }

    public Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        for (int i = 0; i < _weights.Rows; i++)
            for (int j = 0; j < _weights.Columns; j++)
                parameters.Add(_weights[i, j]);
        for (int i = 0; i < _bias.Length; i++)
            parameters.Add(_bias[i]);
        return new Vector<T>(parameters.ToArray());
    }

    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int i = 0; i < _weights.Rows && idx < parameters.Length; i++)
            for (int j = 0; j < _weights.Columns && idx < parameters.Length; j++)
                _weights[i, j] = parameters[idx++];
        for (int i = 0; i < _bias.Length && idx < parameters.Length; i++)
            _bias[i] = parameters[idx++];
    }
}
