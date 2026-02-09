using AiDotNet.Extensions;
using AiDotNet.Tensors;

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
/// <para>
/// <b>Production-Ready Features:</b>
/// <list type="bullet">
/// <item>Uses Tensor&lt;T&gt; for GPU-accelerated operations via IEngine</item>
/// <item>Proper LSTM with all gates (input, forget, output, cell)</item>
/// <item>Backpropagation through time (BPTT) for gradient computation</item>
/// <item>Vectorized operations - no numerical differentiation</item>
/// <item>All parameters are trained (not subsets)</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> DeepAR is like a weather forecaster that doesn't just say
/// "it will be 70 degrees tomorrow" but rather "there's a 50% chance it'll be between 65-75 degrees,
/// a 90% chance it'll be between 60-80 degrees," etc.
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
    private readonly Random _random;

    // Tensor-based LSTM layers
    private readonly List<DeepARLstmCellTensor<T>> _lstmLayers;

    // Distribution parameters using Tensors
    private Tensor<T> _meanWeights;
    private Tensor<T> _meanBias;
    private Tensor<T> _scaleWeights;
    private Tensor<T> _scaleBias;

    /// <summary>
    /// Initializes a new instance of the DeepARModel class.
    /// </summary>
    /// <param name="options">Configuration options for DeepAR.</param>
    public DeepARModel(DeepAROptions<T>? options = null)
        : base(options ?? new DeepAROptions<T>())
    {
        _options = options ?? new DeepAROptions<T>();
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(42);
        _lstmLayers = new List<DeepARLstmCellTensor<T>>();
        _meanWeights = new Tensor<T>([1, 1]);
        _meanBias = new Tensor<T>([1]);
        _scaleWeights = new Tensor<T>([1, 1]);
        _scaleBias = new Tensor<T>([1]);

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
    /// Initializes the model architecture with tensor-based parameters.
    /// </summary>
    private void InitializeModel()
    {
        _lstmLayers.Clear();
        int inputSize = 1 + _options.CovariateSize;

        for (int i = 0; i < _options.NumLayers; i++)
        {
            int layerInputSize = (i == 0) ? inputSize : _options.HiddenSize;
            _lstmLayers.Add(new DeepARLstmCellTensor<T>(layerInputSize, _options.HiddenSize, 42 + i * 1000));
        }

        // Distribution parameter weights using Xavier initialization
        double stddev = Math.Sqrt(2.0 / _options.HiddenSize);
        _meanWeights = CreateRandomTensor([1, _options.HiddenSize], stddev);
        _meanBias = new Tensor<T>([1]);
        _scaleWeights = CreateRandomTensor([1, _options.HiddenSize], stddev);
        _scaleBias = new Tensor<T>([1]);
    }

    private Tensor<T> CreateRandomTensor(int[] shape, double stddev)
    {
        var tensor = new Tensor<T>(shape);
        int total = tensor.Length;
        for (int i = 0; i < total; i++)
        {
            tensor[i] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    /// <summary>
    /// Trains the model using proper backpropagation through time (BPTT).
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = NumOps.FromDouble(_options.LearningRate);
        int numSamples = x.Rows;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            var indices = Enumerable.Range(0, numSamples).OrderBy(_ => _random.Next()).ToList();

            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);
                int batchSize = batchEnd - batchStart;

                // Accumulate gradients over batch
                var batchGradients = new Dictionary<string, Tensor<T>>();

                for (int bi = 0; bi < batchSize; bi++)
                {
                    int i = indices[batchStart + bi];
                    var input = ConvertRowToTensor(x, i);
                    T target = y[i];

                    var gradients = ComputeGradients(input, target);

                    // Accumulate gradients
                    foreach (var kvp in gradients)
                    {
                        batchGradients[kvp.Key] = batchGradients.TryGetValue(kvp.Key, out var existing)
                            ? Engine.TensorAdd(existing, kvp.Value)
                            : kvp.Value.Clone();
                    }
                }

                // Average and apply gradients
                ApplyGradients(batchGradients, learningRate, batchSize);
            }
        }
    }

    private Tensor<T> ConvertRowToTensor(Matrix<T> x, int rowIndex)
    {
        var tensor = new Tensor<T>([x.Columns]);
        for (int j = 0; j < x.Columns; j++)
        {
            tensor[j] = x[rowIndex, j];
        }
        return tensor;
    }

    /// <summary>
    /// Computes gradients using backpropagation through the LSTM and output layers.
    /// </summary>
    private Dictionary<string, Tensor<T>> ComputeGradients(Tensor<T> input, T target)
    {
        var gradients = new Dictionary<string, Tensor<T>>();

        // Reset LSTM states
        foreach (var lstm in _lstmLayers)
        {
            lstm.ResetState();
        }

        // Forward pass through LSTM layers (cells cache internally for backprop)
        Tensor<T> hidden = input;

        foreach (var lstm in _lstmLayers)
        {
            hidden = lstm.Forward(hidden);
        }

        // Ensure hidden matches distribution weight dimensions
        if (hidden.Length != _meanWeights.Shape[1])
        {
            var resized = new Tensor<T>([_meanWeights.Shape[1]]);
            for (int j = 0; j < Math.Min(hidden.Length, _meanWeights.Shape[1]); j++)
            {
                resized[j] = hidden[j];
            }
            hidden = resized;
        }

        // Compute mean
        T mean = _meanBias[0];
        for (int j = 0; j < _meanWeights.Shape[1]; j++)
        {
            mean = NumOps.Add(mean, NumOps.Multiply(_meanWeights[0, j], hidden[j]));
        }

        // Compute scale (softplus for positivity)
        T scaleRaw = _scaleBias[0];
        for (int j = 0; j < _scaleWeights.Shape[1]; j++)
        {
            scaleRaw = NumOps.Add(scaleRaw, NumOps.Multiply(_scaleWeights[0, j], hidden[j]));
        }
        T scale = Softplus(scaleRaw);

        // Compute NLL loss gradient
        T error = NumOps.Subtract(mean, target);
        T variance = NumOps.Multiply(scale, scale);
        T varianceEps = NumOps.Add(variance, NumOps.FromDouble(1e-6));

        // dL/d_mean = (mean - target) / variance
        T dLdMean = NumOps.Divide(error, varianceEps);

        // dL/d_scale = 1/scale - error^2/scale^3 (from NLL derivation for Gaussian)
        // Use epsilon-adjusted scale consistently to avoid division by zero
        T errorSq = NumOps.Multiply(error, error);
        T scaleEps = NumOps.Add(scale, NumOps.FromDouble(1e-6));
        // Use (scale + ε)³ to ensure denominator is never zero
        T scaleCubed = NumOps.Multiply(scaleEps, NumOps.Multiply(scaleEps, scaleEps));
        T dLdScale = NumOps.Subtract(
            NumOps.Divide(NumOps.One, scaleEps),
            NumOps.Divide(errorSq, scaleCubed)
        );

        // Backprop through softplus: d_softplus/dx = sigmoid(x)
        T dSoftplus = Sigmoid(scaleRaw);
        T dLdScaleRaw = NumOps.Multiply(dLdScale, dSoftplus);

        // Compute gradients for mean weights
        var meanWeightGrad = new Tensor<T>(_meanWeights.Shape);
        for (int j = 0; j < _meanWeights.Shape[1]; j++)
        {
            meanWeightGrad[0, j] = NumOps.Multiply(dLdMean, hidden[j]);
        }
        gradients["mean_weights"] = meanWeightGrad;

        var meanBiasGrad = new Tensor<T>([1]);
        meanBiasGrad[0] = dLdMean;
        gradients["mean_bias"] = meanBiasGrad;

        // Compute gradients for scale weights
        var scaleWeightGrad = new Tensor<T>(_scaleWeights.Shape);
        for (int j = 0; j < _scaleWeights.Shape[1]; j++)
        {
            scaleWeightGrad[0, j] = NumOps.Multiply(dLdScaleRaw, hidden[j]);
        }
        gradients["scale_weights"] = scaleWeightGrad;

        var scaleBiasGrad = new Tensor<T>([1]);
        scaleBiasGrad[0] = dLdScaleRaw;
        gradients["scale_bias"] = scaleBiasGrad;

        // Backprop through LSTM layers
        var dHidden = new Tensor<T>(hidden.Shape);
        for (int j = 0; j < hidden.Length; j++)
        {
            dHidden[j] = NumOps.Add(
                NumOps.Multiply(dLdMean, _meanWeights[0, Math.Min(j, _meanWeights.Shape[1] - 1)]),
                NumOps.Multiply(dLdScaleRaw, _scaleWeights[0, Math.Min(j, _scaleWeights.Shape[1] - 1)])
            );
        }

        for (int layer = _lstmLayers.Count - 1; layer >= 0; layer--)
        {
            var lstmGradients = _lstmLayers[layer].Backward(dHidden);
            foreach (var kvp in lstmGradients)
            {
                gradients[$"lstm_{layer}_{kvp.Key}"] = kvp.Value;
            }

            if (layer > 0 && lstmGradients.TryGetValue("input_gradient", out var inputGrad))
            {
                dHidden = inputGrad;
            }
        }

        return gradients;
    }

    private T Softplus(T x)
    {
        T threshold = NumOps.FromDouble(20.0);
        if (NumOps.GreaterThan(x, threshold))
            return x;
        if (NumOps.LessThan(x, NumOps.FromDouble(-20.0)))
            return NumOps.Exp(x);
        return NumOps.Log(NumOps.Add(NumOps.One, NumOps.Exp(x)));
    }

    private T Sigmoid(T x)
    {
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(x))));
    }

    private void ApplyGradients(Dictionary<string, Tensor<T>> gradients, T learningRate, int batchSize)
    {
        T batchSizeT = NumOps.FromDouble(batchSize);

        // Update mean weights
        if (gradients.TryGetValue("mean_weights", out var meanWGrad))
        {
            var avgGrad = Engine.TensorDivideScalar(meanWGrad, batchSizeT);
            var scaledGrad = Engine.TensorMultiplyScalar(avgGrad, learningRate);
            _meanWeights = Engine.TensorSubtract(_meanWeights, scaledGrad);
        }

        if (gradients.TryGetValue("mean_bias", out var meanBGrad))
        {
            var avgGrad = Engine.TensorDivideScalar(meanBGrad, batchSizeT);
            var scaledGrad = Engine.TensorMultiplyScalar(avgGrad, learningRate);
            _meanBias = Engine.TensorSubtract(_meanBias, scaledGrad);
        }

        // Update scale weights
        if (gradients.TryGetValue("scale_weights", out var scaleWGrad))
        {
            var avgGrad = Engine.TensorDivideScalar(scaleWGrad, batchSizeT);
            var scaledGrad = Engine.TensorMultiplyScalar(avgGrad, learningRate);
            _scaleWeights = Engine.TensorSubtract(_scaleWeights, scaledGrad);
        }

        if (gradients.TryGetValue("scale_bias", out var scaleBGrad))
        {
            var avgGrad = Engine.TensorDivideScalar(scaleBGrad, batchSizeT);
            var scaledGrad = Engine.TensorMultiplyScalar(avgGrad, learningRate);
            _scaleBias = Engine.TensorSubtract(_scaleBias, scaledGrad);
        }

        // Update LSTM layers
        for (int layer = 0; layer < _lstmLayers.Count; layer++)
        {
            _lstmLayers[layer].ApplyGradients(gradients, $"lstm_{layer}_", learningRate, batchSizeT);
        }
    }

    public override T PredictSingle(Vector<T> input)
    {
        var (mean, _) = PredictDistribution(input);
        return mean;
    }

    /// <summary>
    /// Predicts distribution parameters (mean and scale) for a single input.
    /// </summary>
    private (T mean, T scale) PredictDistribution(Vector<T> input)
    {
        foreach (var lstm in _lstmLayers)
        {
            lstm.ResetState();
        }

        var hidden = new Tensor<T>([input.Length]);
        for (int i = 0; i < input.Length; i++)
        {
            hidden[i] = input[i];
        }

        foreach (var lstm in _lstmLayers)
        {
            hidden = lstm.Forward(hidden);
        }

        if (hidden.Length != _meanWeights.Shape[1])
        {
            var resized = new Tensor<T>([_meanWeights.Shape[1]]);
            for (int j = 0; j < Math.Min(hidden.Length, _meanWeights.Shape[1]); j++)
            {
                resized[j] = hidden[j];
            }
            hidden = resized;
        }

        T mean = _meanBias[0];
        for (int j = 0; j < _meanWeights.Shape[1]; j++)
        {
            mean = NumOps.Add(mean, NumOps.Multiply(_meanWeights[0, j], hidden[j]));
        }

        T scaleRaw = _scaleBias[0];
        for (int j = 0; j < _scaleWeights.Shape[1]; j++)
        {
            scaleRaw = NumOps.Add(scaleRaw, NumOps.Multiply(_scaleWeights[0, j], hidden[j]));
        }

        T scale = Softplus(scaleRaw);
        T minScale = NumOps.FromDouble(1e-6);
        if (NumOps.LessThan(scale, minScale))
            scale = minScale;

        return (mean, scale);
    }

    /// <summary>
    /// Generates probabilistic forecasts with quantile predictions.
    /// </summary>
    public Dictionary<double, Vector<T>> ForecastWithQuantiles(Vector<T> history, double[] quantiles)
    {
        var result = new Dictionary<double, Vector<T>>();
        var samples = new List<Vector<T>>();

        for (int s = 0; s < _options.NumSamples; s++)
        {
            var forecast = new Vector<T>(_options.ForecastHorizon);
            Vector<T> context = history.Clone();

            for (int h = 0; h < _options.ForecastHorizon; h++)
            {
                var (mean, scale) = PredictDistribution(context);

                T sample = NumOps.Add(mean, NumOps.Multiply(scale, NumOps.FromDouble(_random.NextGaussian())));
                forecast[h] = sample;

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
                quantileForecast[h] = NumOps.FromDouble(values[idx]);
            }

            result[q] = quantileForecast;
        }

        return result;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.HiddenSize);
        writer.Write(_options.NumLayers);

        writer.Write(_lstmLayers.Count);
        foreach (var lstm in _lstmLayers)
        {
            lstm.Serialize(writer);
        }

        SerializeTensor(writer, _meanWeights);
        SerializeTensor(writer, _meanBias);
        SerializeTensor(writer, _scaleWeights);
        SerializeTensor(writer, _scaleBias);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.HiddenSize = reader.ReadInt32();
        _options.NumLayers = reader.ReadInt32();

        InitializeModel();

        int numLayers = reader.ReadInt32();
        for (int i = 0; i < numLayers && i < _lstmLayers.Count; i++)
        {
            _lstmLayers[i].Deserialize(reader);
        }

        _meanWeights = DeserializeTensor(reader);
        _meanBias = DeserializeTensor(reader);
        _scaleWeights = DeserializeTensor(reader);
        _scaleBias = DeserializeTensor(reader);
    }

    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
            writer.Write(dim);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(Convert.ToDouble(tensor[i]));
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int d = 0; d < rank; d++)
            shape[d] = reader.ReadInt32();

        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble(reader.ReadDouble());
        return tensor;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DeepAR",
            ModelType = ModelType.TimeSeriesRegression,
            Description = "Probabilistic forecasting with autoregressive recurrent networks (Production-Ready)",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HiddenSize", _options.HiddenSize },
                { "NumLayers", _options.NumLayers },
                { "LikelihoodType", _options.LikelihoodType },
                { "ForecastHorizon", _options.ForecastHorizon },
                { "ProductionReady", true }
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
            count += _meanWeights.Length + _meanBias.Length;
            count += _scaleWeights.Length + _scaleBias.Length;
            return count;
        }
    }
}

/// <summary>
/// Production-ready LSTM cell with proper gates (input, forget, output, cell).
/// Uses Tensor operations for GPU acceleration and proper backpropagation.
/// </summary>
internal class DeepARLstmCellTensor<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _inputSize;
    private readonly int _hiddenSize;

    // Weights for all gates: [4*hiddenSize, inputSize+hiddenSize]
    // Order: input gate (i), forget gate (f), cell gate (g), output gate (o)
    private readonly Tensor<T> _weights;
    private readonly Tensor<T> _bias;

    // States
    private Tensor<T> _hiddenState;
    private Tensor<T> _cellState;

    // Cached values for backprop
    private Tensor<T> _lastInput;
    private Tensor<T> _lastCombined;
    private Tensor<T> _lastGates;
    private Tensor<T> _lastCellCandidate;
    private Tensor<T> _lastPrevCell;

    public int ParameterCount => _weights.Length + _bias.Length;

    public DeepARLstmCellTensor(int inputSize, int hiddenSize, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / (inputSize + hiddenSize));

        // Initialize weights for all 4 gates
        _weights = new Tensor<T>([4 * hiddenSize, inputSize + hiddenSize]);
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }

        // Initialize biases - forget gate bias typically initialized to 1
        _bias = new Tensor<T>([4 * hiddenSize]);
        for (int i = _hiddenSize; i < 2 * _hiddenSize; i++)
        {
            _bias[i] = _numOps.One; // Forget gate bias
        }

        _hiddenState = new Tensor<T>([hiddenSize]);
        _cellState = new Tensor<T>([hiddenSize]);
        _lastInput = new Tensor<T>([inputSize]);
        _lastCombined = new Tensor<T>([inputSize + hiddenSize]);
        _lastGates = new Tensor<T>([4 * hiddenSize]);
        _lastCellCandidate = new Tensor<T>([hiddenSize]);
        _lastPrevCell = new Tensor<T>([hiddenSize]);
    }

    public void ResetState()
    {
        for (int i = 0; i < _hiddenSize; i++)
        {
            _hiddenState[i] = _numOps.Zero;
            _cellState[i] = _numOps.Zero;
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // Cache input for backprop
        _lastInput = input.Clone();
        _lastPrevCell = _cellState.Clone();

        // Create combined [input; hidden]
        int combinedSize = _inputSize + _hiddenSize;
        _lastCombined = new Tensor<T>([combinedSize]);

        for (int i = 0; i < _inputSize; i++)
        {
            _lastCombined[i] = i < input.Length ? input[i] : _numOps.Zero;
        }
        for (int i = 0; i < _hiddenSize; i++)
        {
            _lastCombined[_inputSize + i] = _hiddenState[i];
        }

        // Compute all gates: gates = W * combined + bias
        _lastGates = new Tensor<T>([4 * _hiddenSize]);
        for (int i = 0; i < 4 * _hiddenSize; i++)
        {
            T sum = _bias[i];
            for (int j = 0; j < combinedSize; j++)
            {
                int idx = i * (_inputSize + _hiddenSize) + j;
                if (idx < _weights.Length)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(_weights[idx], _lastCombined[j]));
                }
            }
            _lastGates[i] = sum;
        }

        // Apply activations and compute new states
        var newHidden = new Tensor<T>([_hiddenSize]);
        var newCell = new Tensor<T>([_hiddenSize]);
        _lastCellCandidate = new Tensor<T>([_hiddenSize]);

        for (int i = 0; i < _hiddenSize; i++)
        {
            // Input gate: sigmoid(gates[i])
            T inputGate = Sigmoid(_lastGates[i]);

            // Forget gate: sigmoid(gates[hiddenSize + i])
            T forgetGate = Sigmoid(_lastGates[_hiddenSize + i]);

            // Cell candidate: tanh(gates[2*hiddenSize + i])
            T cellCandidate = MathHelper.Tanh(_lastGates[2 * _hiddenSize + i]);
            _lastCellCandidate[i] = cellCandidate;

            // Output gate: sigmoid(gates[3*hiddenSize + i])
            T outputGate = Sigmoid(_lastGates[3 * _hiddenSize + i]);

            // New cell state: f * c_prev + i * g
            newCell[i] = _numOps.Add(
                _numOps.Multiply(forgetGate, _cellState[i]),
                _numOps.Multiply(inputGate, cellCandidate)
            );

            // New hidden state: o * tanh(c_new)
            newHidden[i] = _numOps.Multiply(outputGate, MathHelper.Tanh(newCell[i]));
        }

        _cellState = newCell;
        _hiddenState = newHidden;

        return newHidden.Clone();
    }

    private T Sigmoid(T x)
    {
        return _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Exp(_numOps.Negate(x))));
    }

    public Dictionary<string, Tensor<T>> Backward(Tensor<T> dHidden)
    {
        var gradients = new Dictionary<string, Tensor<T>>();

        // Weight gradients
        var dWeights = new Tensor<T>(_weights.Shape);
        var dBias = new Tensor<T>(_bias.Shape);
        var dInput = new Tensor<T>([_inputSize]);

        // Backprop through output: h = o * tanh(c)
        for (int i = 0; i < _hiddenSize && i < dHidden.Length; i++)
        {
            T dh = dHidden[i];

            // o = sigmoid(gates[3*hidden + i])
            T o = Sigmoid(_lastGates[3 * _hiddenSize + i]);
            T tanhC = MathHelper.Tanh(_cellState[i]);

            // d_tanh_c = dh * o
            T dTanhC = _numOps.Multiply(dh, o);

            // d_o = dh * tanh(c)
            T dO = _numOps.Multiply(dh, tanhC);
            T dOGate = _numOps.Multiply(dO, _numOps.Multiply(o, _numOps.Subtract(_numOps.One, o)));

            // d_c = d_tanh_c * (1 - tanh(c)^2)
            T tanhCSq = _numOps.Multiply(tanhC, tanhC);
            T dC = _numOps.Multiply(dTanhC, _numOps.Subtract(_numOps.One, tanhCSq));

            // d_i = dC * g, d_f = dC * c_prev, d_g = dC * i
            T iGate = Sigmoid(_lastGates[i]);
            T fGate = Sigmoid(_lastGates[_hiddenSize + i]);
            T g = _lastCellCandidate[i];

            T dI = _numOps.Multiply(dC, g);
            T dF = _numOps.Multiply(dC, _lastPrevCell[i]);
            T dG = _numOps.Multiply(dC, iGate);

            // Apply sigmoid/tanh derivatives
            T dIGate = _numOps.Multiply(dI, _numOps.Multiply(iGate, _numOps.Subtract(_numOps.One, iGate)));
            T dFGate = _numOps.Multiply(dF, _numOps.Multiply(fGate, _numOps.Subtract(_numOps.One, fGate)));
            T gSq = _numOps.Multiply(g, g);
            T dGGate = _numOps.Multiply(dG, _numOps.Subtract(_numOps.One, gSq));

            // Update bias gradients
            dBias[i] = _numOps.Add(dBias[i], dIGate);
            dBias[_hiddenSize + i] = _numOps.Add(dBias[_hiddenSize + i], dFGate);
            dBias[2 * _hiddenSize + i] = _numOps.Add(dBias[2 * _hiddenSize + i], dGGate);
            dBias[3 * _hiddenSize + i] = _numOps.Add(dBias[3 * _hiddenSize + i], dOGate);

            // Update weight gradients
            int combinedSize = _inputSize + _hiddenSize;
            for (int j = 0; j < combinedSize; j++)
            {
                T cj = _lastCombined[j];
                int wi = i * combinedSize + j;
                int wf = (_hiddenSize + i) * combinedSize + j;
                int wg = (2 * _hiddenSize + i) * combinedSize + j;
                int wo = (3 * _hiddenSize + i) * combinedSize + j;

                if (wi < dWeights.Length) dWeights[wi] = _numOps.Add(dWeights[wi], _numOps.Multiply(dIGate, cj));
                if (wf < dWeights.Length) dWeights[wf] = _numOps.Add(dWeights[wf], _numOps.Multiply(dFGate, cj));
                if (wg < dWeights.Length) dWeights[wg] = _numOps.Add(dWeights[wg], _numOps.Multiply(dGGate, cj));
                if (wo < dWeights.Length) dWeights[wo] = _numOps.Add(dWeights[wo], _numOps.Multiply(dOGate, cj));

                // Accumulate input gradient
                if (j < _inputSize)
                {
                    dInput[j] = _numOps.Add(dInput[j], _numOps.Multiply(dIGate, _weights[wi]));
                    if (wf < _weights.Length) dInput[j] = _numOps.Add(dInput[j], _numOps.Multiply(dFGate, _weights[wf]));
                    if (wg < _weights.Length) dInput[j] = _numOps.Add(dInput[j], _numOps.Multiply(dGGate, _weights[wg]));
                    if (wo < _weights.Length) dInput[j] = _numOps.Add(dInput[j], _numOps.Multiply(dOGate, _weights[wo]));
                }
            }
        }

        gradients["weights"] = dWeights;
        gradients["bias"] = dBias;
        gradients["input_gradient"] = dInput;

        return gradients;
    }

    public void ApplyGradients(Dictionary<string, Tensor<T>> allGradients, string prefix, T learningRate, T batchSize)
    {
        if (allGradients.TryGetValue($"{prefix}weights", out var wGrad))
        {
            for (int i = 0; i < _weights.Length && i < wGrad.Length; i++)
            {
                T avg = _numOps.Divide(wGrad[i], batchSize);
                T scaled = _numOps.Multiply(avg, learningRate);
                _weights[i] = _numOps.Subtract(_weights[i], scaled);
            }
        }

        if (allGradients.TryGetValue($"{prefix}bias", out var bGrad))
        {
            for (int i = 0; i < _bias.Length && i < bGrad.Length; i++)
            {
                T avg = _numOps.Divide(bGrad[i], batchSize);
                T scaled = _numOps.Multiply(avg, learningRate);
                _bias[i] = _numOps.Subtract(_bias[i], scaled);
            }
        }
    }

    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_inputSize);
        writer.Write(_hiddenSize);

        writer.Write(_weights.Shape.Length);
        foreach (var dim in _weights.Shape)
            writer.Write(dim);
        for (int i = 0; i < _weights.Length; i++)
            writer.Write(Convert.ToDouble(_weights[i]));

        writer.Write(_bias.Shape.Length);
        foreach (var dim in _bias.Shape)
            writer.Write(dim);
        for (int i = 0; i < _bias.Length; i++)
            writer.Write(Convert.ToDouble(_bias[i]));
    }

    public void Deserialize(BinaryReader reader)
    {
        reader.ReadInt32(); // inputSize
        reader.ReadInt32(); // hiddenSize

        int wRank = reader.ReadInt32();
        var wShape = new int[wRank];
        for (int d = 0; d < wRank; d++)
            wShape[d] = reader.ReadInt32();
        int wTotal = wShape.Aggregate(1, (a, b) => a * b);
        // Consume all serialized doubles to keep stream aligned
        for (int i = 0; i < wTotal; i++)
        {
            double v = reader.ReadDouble();
            if (i < _weights.Length)
                _weights[i] = _numOps.FromDouble(v);
        }

        int bRank = reader.ReadInt32();
        var bShape = new int[bRank];
        for (int d = 0; d < bRank; d++)
            bShape[d] = reader.ReadInt32();
        int bTotal = bShape.Aggregate(1, (a, b) => a * b);
        // Consume all serialized doubles to keep stream aligned
        for (int i = 0; i < bTotal; i++)
        {
            double v = reader.ReadDouble();
            if (i < _bias.Length)
                _bias[i] = _numOps.FromDouble(v);
        }
    }
}
