using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Informer model for efficient long-sequence time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Long-Sequence Forecasting Problem:</b>
/// Traditional Transformer models achieve state-of-the-art results in many sequence modeling tasks,
/// but they struggle with long time series because self-attention has O(L^2) time and memory complexity.
/// For a sequence of 1000 time steps, vanilla attention requires 1 million operations per layer.
/// This makes long-horizon forecasting computationally prohibitive.
/// </para>
/// <para>
/// <b>The Informer Solution:</b>
/// Informer (Zhou et al., AAAI 2021) introduces three key innovations:
/// 1. ProbSparse Self-Attention (O(L log L) complexity)
/// 2. Self-Attention Distilling for sequence compression
/// 3. Generative-Style Decoder for parallel multi-step forecasting
/// </para>
/// </remarks>
public class InformerModel<T> : TimeSeriesModelBase<T>
{
    private readonly InformerOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    // Input embedding and positional encoding (Tensor-based)
    private Tensor<T> _inputProjection;      // [embeddingDim, 1]
    private Tensor<T> _positionalEncoding;   // [maxLen, embeddingDim]

    // Encoder components with distilling (Tensor-based)
    private readonly List<InformerEncoderLayerTensor<T>> _encoderLayers;
    private readonly List<DistillingConvTensor<T>> _distillingLayers;

    // Decoder components (Tensor-based)
    private readonly List<InformerDecoderLayerTensor<T>> _decoderLayers;
    private Tensor<T> _decoderStartToken;    // [embeddingDim]

    // Output projection (Tensor-based)
    private Tensor<T> _outputProjection;     // [forecastHorizon, embeddingDim]
    private Tensor<T> _outputBias;           // [forecastHorizon]

    // Gradient accumulators for batch training
    private Dictionary<string, Tensor<T>> _gradientAccumulators;

    // Sparsity factor for ProbSparse attention (c in the paper, typically 5)
    private const int SparsityFactor = 5;

    /// <summary>
    /// Initializes a new instance of the Informer model with the specified options.
    /// </summary>
    public InformerModel(InformerOptions<T>? options = null)
        : this(options ?? new InformerOptions<T>(), initializeModel: true)
    {
    }

    private InformerModel(InformerOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        Options = _options;

        // Validate options
        if (_options.EmbeddingDim <= 0)
            throw new ArgumentException("EmbeddingDim must be positive.", nameof(options));
        if (_options.NumEncoderLayers <= 0)
            throw new ArgumentException("NumEncoderLayers must be positive.", nameof(options));
        if (_options.NumDecoderLayers <= 0)
            throw new ArgumentException("NumDecoderLayers must be positive.", nameof(options));
        if (_options.NumAttentionHeads <= 0)
            throw new ArgumentException("NumAttentionHeads must be positive.", nameof(options));

        _random = RandomHelper.CreateSeededRandom(42);
        _encoderLayers = new List<InformerEncoderLayerTensor<T>>();
        _distillingLayers = new List<DistillingConvTensor<T>>();
        _decoderLayers = new List<InformerDecoderLayerTensor<T>>();
        _gradientAccumulators = new Dictionary<string, Tensor<T>>();

        // Initialize with default tensors
        _inputProjection = new Tensor<T>(new[] { 1, 1 });
        _positionalEncoding = new Tensor<T>(new[] { 1, 1 });
        _decoderStartToken = new Tensor<T>(new[] { 1 });
        _outputProjection = new Tensor<T>(new[] { 1, 1 });
        _outputBias = new Tensor<T>(new[] { 1 });

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        double stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);
        var random = RandomHelper.CreateSeededRandom(42);

        // Input projection: maps single time step values to embedding dimension
        _inputProjection = InitTensor(new[] { _options.EmbeddingDim, 1 }, stddev, random);

        // Sinusoidal positional encoding for the maximum sequence length
        int maxLen = Math.Max(_options.LookbackWindow, _options.ForecastHorizon) * 2;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);

        // Encoder layers with distilling
        int currentSeqLen = _options.LookbackWindow;
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new InformerEncoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + i));

            // Add distilling layer (except after the last encoder layer)
            if (i < _options.NumEncoderLayers - 1)
            {
                _distillingLayers.Add(new DistillingConvTensor<T>(
                    _options.EmbeddingDim,
                    currentSeqLen,
                    _options.DistillingFactor,
                    42 + i));
                currentSeqLen = (currentSeqLen + _options.DistillingFactor - 1) / _options.DistillingFactor;
            }
        }

        // Decoder layers
        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            _decoderLayers.Add(new InformerDecoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + _options.NumEncoderLayers + i));
        }

        // Learnable decoder start token
        _decoderStartToken = new Tensor<T>(new[] { _options.EmbeddingDim });
        for (int i = 0; i < _options.EmbeddingDim; i++)
        {
            _decoderStartToken[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }

        // Output projection
        _outputProjection = InitTensor(new[] { _options.ForecastHorizon, _options.EmbeddingDim }, stddev, random);
        _outputBias = new Tensor<T>(new[] { _options.ForecastHorizon });

        // Initialize gradient accumulators
        InitializeGradientAccumulators();
    }

    private void InitializeGradientAccumulators()
    {
        _gradientAccumulators = new Dictionary<string, Tensor<T>>
        {
            ["inputProjection"] = new Tensor<T>(new[] { _options.EmbeddingDim, 1 }),
            ["decoderStartToken"] = new Tensor<T>(new[] { _options.EmbeddingDim }),
            ["outputProjection"] = new Tensor<T>(new[] { _options.ForecastHorizon, _options.EmbeddingDim }),
            ["outputBias"] = new Tensor<T>(new[] { _options.ForecastHorizon })
        };

        // Initialize encoder layer gradient accumulators
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            _encoderLayers[i].InitializeGradientAccumulators(_gradientAccumulators, i);
        }

        // Initialize distilling layer gradient accumulators
        for (int i = 0; i < _distillingLayers.Count; i++)
        {
            _distillingLayers[i].InitializeGradientAccumulators(_gradientAccumulators, i);
        }

        // Initialize decoder layer gradient accumulators
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            _decoderLayers[i].InitializeGradientAccumulators(_gradientAccumulators, i);
        }
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    private Tensor<T> CreateSinusoidalPositionalEncoding(int maxLen, int embeddingDim)
    {
        var encoding = new Tensor<T>(new[] { maxLen, embeddingDim });
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2.0)) / embeddingDim);
                double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                encoding[pos * embeddingDim + i] = _numOps.FromDouble(value);
            }
        }
        return encoding;
    }

    /// <summary>
    /// Trains the model using proper backpropagation through the Informer architecture.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        int lookback = _options.LookbackWindow;

        // Build training samples from y: input = y[i-lookback : i], target = y[i]
        var validIndices = new List<int>();
        for (int i = lookback; i < y.Length; i++)
        {
            validIndices.Add(i);
        }

        if (validIndices.Count == 0)
        {
            throw new ArgumentException(
                $"Not enough data to build a single training sample. " +
                $"Require at least {lookback + 1} points, got {y.Length}.",
                nameof(y));
        }

        double prevLoss = double.MaxValue;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            var shuffled = validIndices.OrderBy(_ => _random.Next()).ToList();
            double epochLoss = 0;
            int sampleCount = 0;

            for (int batchStart = 0; batchStart < shuffled.Count; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, shuffled.Count);
                int batchSize = batchEnd - batchStart;

                // Reset gradient accumulators
                ResetGradientAccumulators();

                // Accumulate gradients for batch using backpropagation
                for (int idx = batchStart; idx < batchEnd; idx++)
                {
                    int i = shuffled[idx];

                    // Extract lookback window from y as input
                    var input = new Vector<T>(lookback);
                    for (int j = 0; j < lookback; j++)
                    {
                        input[j] = y[i - lookback + j];
                    }

                    T target = y[i];

                    // Compute gradients via backpropagation and accumulate
                    var (gradients, prediction) = ComputeGradients(input, target);
                    AccumulateGradients(gradients);

                    double error = _numOps.ToDouble(_numOps.Subtract(target, prediction));
                    epochLoss += error * error;
                    sampleCount++;
                }

                // Apply accumulated gradients
                ApplyGradients(learningRate, batchSize);
            }

            // Early termination if loss converges
            double avgLoss = sampleCount > 0 ? epochLoss / sampleCount : 0;
            if (Math.Abs(prevLoss - avgLoss) < 1e-8)
                break;
            prevLoss = avgLoss;
        }
    }

    private void ResetGradientAccumulators()
    {
        foreach (var tensor in _gradientAccumulators.Values)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = _numOps.Zero;
            }
        }
    }

    private (Dictionary<string, Tensor<T>> gradients, T prediction) ComputeGradients(Vector<T> input, T target)
    {
        var gradients = new Dictionary<string, Tensor<T>>();

        // Forward pass with caching
        var (prediction, cache) = ForwardWithCache(input);
        T error = _numOps.Subtract(target, prediction);

        // Backward pass through output projection
        var dOutputProj = new Tensor<T>(new[] { _options.ForecastHorizon, _options.EmbeddingDim });
        var dOutputBias = new Tensor<T>(new[] { _options.ForecastHorizon });

        // dL/dOutput = -2 * error (MSE gradient for first forecast position)
        T dLoss = _numOps.Multiply(_numOps.FromDouble(-2.0), error);

        // Gradient for output bias and projection (first position only for single prediction)
        dOutputBias[0] = dLoss;
        for (int j = 0; j < _options.EmbeddingDim; j++)
        {
            if (cache.DecoderOutput.Count > 0 && cache.DecoderOutput[0].Length > j)
            {
                dOutputProj[j] = _numOps.Multiply(dLoss, cache.DecoderOutput[0][j]);
            }
        }

        gradients["outputProjection"] = dOutputProj;
        gradients["outputBias"] = dOutputBias;

        // Backward through decoder layers
        var dDecoderInput = new List<Tensor<T>>();
        for (int t = 0; t < _options.ForecastHorizon; t++)
        {
            var dVec = new Tensor<T>(new[] { _options.EmbeddingDim });
            if (t == 0)
            {
                for (int j = 0; j < _options.EmbeddingDim; j++)
                {
                    dVec[j] = _numOps.Multiply(dLoss, _outputProjection[j]);
                }
            }
            dDecoderInput.Add(dVec);
        }

        // Backward through decoder layers
        var dEncoderOutput = new List<Tensor<T>>();
        for (int i = _decoderLayers.Count - 1; i >= 0; i--)
        {
            var (dInput, dEnc) = _decoderLayers[i].Backward(dDecoderInput, cache.EncoderOutput);
            dDecoderInput = dInput;
            if (dEncoderOutput.Count == 0)
            {
                dEncoderOutput = dEnc;
            }
            else
            {
                for (int t = 0; t < Math.Min(dEncoderOutput.Count, dEnc.Count); t++)
                {
                    for (int j = 0; j < Math.Min(dEncoderOutput[t].Length, dEnc[t].Length); j++)
                    {
                        dEncoderOutput[t][j] = _numOps.Add(dEncoderOutput[t][j], dEnc[t][j]);
                    }
                }
            }

            _decoderLayers[i].AccumulateGradients(gradients, i);
        }

        // Gradient for decoder start token
        var dStartToken = new Tensor<T>(new[] { _options.EmbeddingDim });
        if (dDecoderInput.Count > 0)
        {
            for (int j = 0; j < Math.Min(_options.EmbeddingDim, dDecoderInput[0].Length); j++)
            {
                dStartToken[j] = dDecoderInput[0][j];
            }
        }
        gradients["decoderStartToken"] = dStartToken;

        // Backward through distilling layers and encoder layers
        var dEncoderIn = dEncoderOutput;
        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
        {
            dEncoderIn = _encoderLayers[i].Backward(dEncoderIn);
            _encoderLayers[i].AccumulateGradients(gradients, i);

            if (i > 0 && i - 1 < _distillingLayers.Count)
            {
                dEncoderIn = _distillingLayers[i - 1].Backward(dEncoderIn);
                _distillingLayers[i - 1].AccumulateGradients(gradients, i - 1);
            }
        }

        // Gradient for input projection
        var dInputProj = new Tensor<T>(new[] { _options.EmbeddingDim, 1 });
        if (dEncoderIn.Count > 0)
        {
            for (int t = 0; t < dEncoderIn.Count && t < input.Length; t++)
            {
                for (int j = 0; j < Math.Min(_options.EmbeddingDim, dEncoderIn[t].Length); j++)
                {
                    dInputProj[j] = _numOps.Add(dInputProj[j],
                        _numOps.Multiply(dEncoderIn[t][j], input[t]));
                }
            }
        }
        gradients["inputProjection"] = dInputProj;

        return (gradients, prediction);
    }

    private void AccumulateGradients(Dictionary<string, Tensor<T>> gradients)
    {
        foreach (var kvp in gradients)
        {
            string key = kvp.Key;
            Tensor<T> gradient = kvp.Value;
            if (_gradientAccumulators.TryGetValue(key, out var accumulator))
            {
                for (int i = 0; i < Math.Min(accumulator.Length, gradient.Length); i++)
                {
                    accumulator[i] = _numOps.Add(accumulator[i], gradient[i]);
                }
            }
        }
    }

    private void ApplyGradients(T learningRate, int batchSize)
    {
        T batchSizeT = _numOps.FromDouble(batchSize);

        // Apply to input projection
        ApplyGradientToTensor(_inputProjection, "inputProjection", learningRate, batchSizeT);

        // Apply to decoder start token
        ApplyGradientToTensor(_decoderStartToken, "decoderStartToken", learningRate, batchSizeT);

        // Apply to output projection and bias
        ApplyGradientToTensor(_outputProjection, "outputProjection", learningRate, batchSizeT);
        ApplyGradientToTensor(_outputBias, "outputBias", learningRate, batchSizeT);

        // Apply to encoder layers
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            _encoderLayers[i].ApplyGradients(_gradientAccumulators, learningRate, batchSizeT, i);
        }

        // Apply to distilling layers
        for (int i = 0; i < _distillingLayers.Count; i++)
        {
            _distillingLayers[i].ApplyGradients(_gradientAccumulators, learningRate, batchSizeT, i);
        }

        // Apply to decoder layers
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            _decoderLayers[i].ApplyGradients(_gradientAccumulators, learningRate, batchSizeT, i);
        }
    }

    private void ApplyGradientToTensor(Tensor<T> tensor, string key, T learningRate, T batchSize)
    {
        if (_gradientAccumulators.TryGetValue(key, out var gradient))
        {
            for (int i = 0; i < Math.Min(tensor.Length, gradient.Length); i++)
            {
                T avgGrad = _numOps.Divide(gradient[i], batchSize);
                T update = _numOps.Multiply(learningRate, avgGrad);
                tensor[i] = _numOps.Subtract(tensor[i], update);
            }
        }
    }

    private (T prediction, ForwardCache cache) ForwardWithCache(Vector<T> input)
    {
        var cache = new ForwardCache();

        // Embed input sequence
        cache.EmbeddedInput = EmbedInput(input);

        // Encoder forward with distilling
        var encoderOutput = cache.EmbeddedInput;
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            encoderOutput = _encoderLayers[i].Forward(encoderOutput);

            // Apply distilling (except after last layer)
            if (i < _distillingLayers.Count)
            {
                encoderOutput = _distillingLayers[i].Forward(encoderOutput);
            }
        }
        cache.EncoderOutput = encoderOutput;

        // Decoder forward
        var decoderInput = CreateDecoderInput();
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            decoderInput = _decoderLayers[i].Forward(decoderInput, encoderOutput);
        }
        cache.DecoderOutput = decoderInput;

        // Output projection (first position for single prediction)
        T prediction = ComputeOutput(decoderInput);

        return (prediction, cache);
    }

    private List<Tensor<T>> EmbedInput(Vector<T> input)
    {
        var embedded = new List<Tensor<T>>();
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);

        for (int t = 0; t < seqLen; t++)
        {
            var vec = new Tensor<T>(new[] { _options.EmbeddingDim });
            for (int j = 0; j < _options.EmbeddingDim; j++)
            {
                // Input projection: multiply input value by projection weight
                T projected = _numOps.Multiply(input[t], _inputProjection[j]);
                // Add positional encoding
                T posEnc = _positionalEncoding[t * _options.EmbeddingDim + j];
                vec[j] = _numOps.Add(projected, posEnc);
            }
            embedded.Add(vec);
        }

        return embedded;
    }

    private List<Tensor<T>> CreateDecoderInput()
    {
        var decoderInput = new List<Tensor<T>>();
        for (int t = 0; t < _options.ForecastHorizon; t++)
        {
            var vec = new Tensor<T>(new[] { _options.EmbeddingDim });
            for (int j = 0; j < _options.EmbeddingDim; j++)
            {
                // Start token + positional encoding
                T posEnc = _positionalEncoding[(_options.LookbackWindow + t) * _options.EmbeddingDim + j];
                vec[j] = _numOps.Add(_decoderStartToken[j], posEnc);
            }
            decoderInput.Add(vec);
        }
        return decoderInput;
    }

    private T ComputeOutput(List<Tensor<T>> decoderOutput)
    {
        if (decoderOutput.Count == 0) return _numOps.Zero;

        T result = _outputBias[0];
        for (int j = 0; j < Math.Min(_options.EmbeddingDim, decoderOutput[0].Length); j++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_outputProjection[j], decoderOutput[0][j]));
        }
        return result;
    }

    /// <summary>
    /// Predicts the next single value in the time series.
    /// </summary>
    public override T PredictSingle(Vector<T> input)
    {
        Vector<T> forecast = ForecastHorizon(input);
        return forecast[0];
    }

    /// <summary>
    /// Generates multi-step forecasts using the full Informer architecture.
    /// </summary>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        var result = new Vector<T>(_options.ForecastHorizon);

        // Embed input
        var embedded = EmbedInput(input);

        // Encoder forward with distilling
        var encoderOutput = embedded;
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            encoderOutput = _encoderLayers[i].Forward(encoderOutput);
            if (i < _distillingLayers.Count)
            {
                encoderOutput = _distillingLayers[i].Forward(encoderOutput);
            }
        }

        // Decoder forward
        var decoderInput = CreateDecoderInput();
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            decoderInput = _decoderLayers[i].Forward(decoderInput, encoderOutput);
        }

        // Output projection for all forecast positions
        for (int t = 0; t < _options.ForecastHorizon && t < decoderInput.Count; t++)
        {
            T value = _outputBias[t];
            for (int j = 0; j < Math.Min(_options.EmbeddingDim, decoderInput[t].Length); j++)
            {
                value = _numOps.Add(value, _numOps.Multiply(
                    _outputProjection[t * _options.EmbeddingDim + j], decoderInput[t][j]));
            }
            result[t] = value;
        }

        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = AiDotNet.Enums.ModelType.Transformer,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["EmbeddingDim"] = _options.EmbeddingDim,
                ["NumEncoderLayers"] = _options.NumEncoderLayers,
                ["NumDecoderLayers"] = _options.NumDecoderLayers,
                ["NumAttentionHeads"] = _options.NumAttentionHeads,
                ["LookbackWindow"] = _options.LookbackWindow,
                ["ForecastHorizon"] = _options.ForecastHorizon,
                ["SparsityFactor"] = SparsityFactor,
                ["DistillingFactor"] = _options.DistillingFactor
            }
        };
    }

    public override int ParameterCount
    {
        get
        {
            int count = _inputProjection.Length + _decoderStartToken.Length +
                       _outputProjection.Length + _outputBias.Length;

            foreach (var layer in _encoderLayers)
                count += layer.ParameterCount;
            foreach (var layer in _distillingLayers)
                count += layer.ParameterCount;
            foreach (var layer in _decoderLayers)
                count += layer.ParameterCount;

            return count;
        }
    }

    private class ForwardCache
    {
        public List<Tensor<T>> EmbeddedInput { get; set; } = new List<Tensor<T>>();
        public List<Tensor<T>> EncoderOutput { get; set; } = new List<Tensor<T>>();
        public List<Tensor<T>> DecoderOutput { get; set; } = new List<Tensor<T>>();
    }

    /// <summary>
    /// Creates a new instance of the Informer model.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new InformerModel<T>(new InformerOptions<T>(_options), initializeModel: false);
    }

    /// <summary>
    /// Serializes the model-specific state to a binary writer.
    /// </summary>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write options
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.LearningRate);
        writer.Write(_options.Epochs);
        writer.Write(_options.BatchSize);
        writer.Write(_options.DistillingFactor);

        // Write main tensors
        WriteTensor(writer, _inputProjection);
        WriteTensor(writer, _positionalEncoding);
        WriteTensor(writer, _decoderStartToken);
        WriteTensor(writer, _outputProjection);
        WriteTensor(writer, _outputBias);

        // Write layer counts
        writer.Write(_encoderLayers.Count);
        writer.Write(_distillingLayers.Count);
        writer.Write(_decoderLayers.Count);

        // Write encoder layer weights
        foreach (var layer in _encoderLayers)
        {
            layer.Serialize(writer, WriteTensor);
        }

        // Write distilling layer weights
        foreach (var layer in _distillingLayers)
        {
            layer.Serialize(writer, WriteTensor);
        }

        // Write decoder layer weights
        foreach (var layer in _decoderLayers)
        {
            layer.Serialize(writer, WriteTensor);
        }
    }

    /// <summary>
    /// Deserializes the model-specific state from a binary reader.
    /// </summary>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read options (already set via constructor, skip them)
        _ = reader.ReadInt32();  // lookback
        _ = reader.ReadInt32();  // horizon
        _ = reader.ReadInt32();  // embDim
        _ = reader.ReadInt32();  // numEncLayers
        _ = reader.ReadInt32();  // numDecLayers
        _ = reader.ReadInt32();  // numHeads
        _ = reader.ReadDouble();  // dropout
        _ = reader.ReadDouble();  // lr
        _ = reader.ReadInt32();  // epochs
        _ = reader.ReadInt32();  // batch
        _ = reader.ReadInt32();  // distill

        // Read main tensors
        _inputProjection = ReadTensor(reader);
        _positionalEncoding = ReadTensor(reader);
        _decoderStartToken = ReadTensor(reader);
        _outputProjection = ReadTensor(reader);
        _outputBias = ReadTensor(reader);

        // Read layer counts
        int encCount = reader.ReadInt32();
        _ = reader.ReadInt32(); // distillCount - read for file format compatibility, value not used
        int decCount = reader.ReadInt32();

        // Clear and recreate layers with proper structure (but with placeholder weights)
        _encoderLayers.Clear();
        _distillingLayers.Clear();
        _decoderLayers.Clear();

        int currentSeqLen = _options.LookbackWindow;
        for (int i = 0; i < encCount; i++)
        {
            _encoderLayers.Add(new InformerEncoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + i));

            if (i < encCount - 1)
            {
                _distillingLayers.Add(new DistillingConvTensor<T>(
                    _options.EmbeddingDim,
                    currentSeqLen,
                    _options.DistillingFactor,
                    42 + i));
                currentSeqLen = (currentSeqLen + _options.DistillingFactor - 1) / _options.DistillingFactor;
            }
        }

        for (int i = 0; i < decCount; i++)
        {
            _decoderLayers.Add(new InformerDecoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + encCount + i));
        }

        // Deserialize encoder layer weights (overwrite the initialized values)
        foreach (var layer in _encoderLayers)
        {
            layer.Deserialize(reader, ReadTensor);
        }

        // Deserialize distilling layer weights
        foreach (var layer in _distillingLayers)
        {
            layer.Deserialize(reader, ReadTensor);
        }

        // Deserialize decoder layer weights
        foreach (var layer in _decoderLayers)
        {
            layer.Deserialize(reader, ReadTensor);
        }

        // Initialize gradient accumulators
        InitializeGradientAccumulators();
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (int dim in tensor.Shape)
            writer.Write(dim);
        writer.Write(tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(_numOps.ToDouble(tensor[i]));
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = reader.ReadInt32();
        int length = reader.ReadInt32();
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < length; i++)
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        return tensor;
    }
}

/// <summary>
/// Tensor-based encoder layer for Informer with ProbSparse attention.
/// </summary>
internal class InformerEncoderLayerTensor<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _sparsityFactor;

    // Multi-head attention weights (Tensor-based)
    private readonly Tensor<T> _queryProj;
    private readonly Tensor<T> _keyProj;
    private readonly Tensor<T> _valueProj;
    private readonly Tensor<T> _outputProj;

    // Feed-forward network (Tensor-based)
    private readonly Tensor<T> _ffn1;
    private readonly Tensor<T> _ffn1Bias;
    private readonly Tensor<T> _ffn2;
    private readonly Tensor<T> _ffn2Bias;

    // Layer normalization parameters (Tensor-based)
    private readonly Tensor<T> _layerNorm1Gamma;
    private readonly Tensor<T> _layerNorm1Beta;
    private readonly Tensor<T> _layerNorm2Gamma;
    private readonly Tensor<T> _layerNorm2Beta;

    // Cache for backward pass
    private List<Tensor<T>>? _cachedInput;

    public int ParameterCount =>
        _queryProj.Length + _keyProj.Length + _valueProj.Length + _outputProj.Length +
        _ffn1.Length + _ffn1Bias.Length + _ffn2.Length + _ffn2Bias.Length +
        _layerNorm1Gamma.Length * 2 + _layerNorm2Gamma.Length * 2;

    public InformerEncoderLayerTensor(int embeddingDim, int numHeads, int sparsityFactor, double dropoutRate, int seed = 42)
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _sparsityFactor = sparsityFactor;

        var random = RandomHelper.CreateSeededRandom(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / ((double)embeddingDim * 4));

        // Initialize Q, K, V, O projections
        _queryProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _keyProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _valueProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _outputProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);

        // Initialize FFN (4x expansion)
        int ffnDim = embeddingDim * 4;
        _ffn1 = InitTensor(new[] { ffnDim, embeddingDim }, ffnStddev, random);
        _ffn1Bias = new Tensor<T>(new[] { ffnDim });
        _ffn2 = InitTensor(new[] { embeddingDim, ffnDim }, ffnStddev, random);
        _ffn2Bias = new Tensor<T>(new[] { embeddingDim });

        // Initialize layer norm parameters
        _layerNorm1Gamma = InitTensorOnes(embeddingDim);
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = InitTensorOnes(embeddingDim);
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    private Tensor<T> InitTensorOnes(int size)
    {
        var tensor = new Tensor<T>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            tensor[i] = _numOps.One;
        }
        return tensor;
    }

    public void InitializeGradientAccumulators(Dictionary<string, Tensor<T>> accumulators, int layerIndex)
    {
        string prefix = $"encoder_{layerIndex}_";
        accumulators[prefix + "queryProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "keyProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "valueProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "outputProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        int ffnDim = _embeddingDim * 4;
        accumulators[prefix + "ffn1"] = new Tensor<T>(new[] { ffnDim, _embeddingDim });
        accumulators[prefix + "ffn1Bias"] = new Tensor<T>(new[] { ffnDim });
        accumulators[prefix + "ffn2"] = new Tensor<T>(new[] { _embeddingDim, ffnDim });
        accumulators[prefix + "ffn2Bias"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln1Gamma"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln1Beta"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln2Gamma"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln2Beta"] = new Tensor<T>(new[] { _embeddingDim });
    }

    public List<Tensor<T>> Forward(List<Tensor<T>> input)
    {
        _cachedInput = input;
        var output = new List<Tensor<T>>();

        // ProbSparse self-attention with residual
        var attnOutput = ProbSparseSelfAttention(input);

        // Add residual and apply layer norm
        var norm1Output = new List<Tensor<T>>();
        for (int t = 0; t < input.Count; t++)
        {
            var residual = new Tensor<T>(new[] { _embeddingDim });
            for (int j = 0; j < _embeddingDim && j < input[t].Length && j < attnOutput[t].Length; j++)
            {
                residual[j] = _numOps.Add(input[t][j], attnOutput[t][j]);
            }
            norm1Output.Add(LayerNorm(residual, _layerNorm1Gamma, _layerNorm1Beta));
        }

        // FFN with residual
        for (int t = 0; t < norm1Output.Count; t++)
        {
            var ffnOut = FeedForward(norm1Output[t]);

            var residual = new Tensor<T>(new[] { _embeddingDim });
            for (int j = 0; j < _embeddingDim && j < norm1Output[t].Length && j < ffnOut.Length; j++)
            {
                residual[j] = _numOps.Add(norm1Output[t][j], ffnOut[j]);
            }
            output.Add(LayerNorm(residual, _layerNorm2Gamma, _layerNorm2Beta));
        }

        return output;
    }

    private List<Tensor<T>> ProbSparseSelfAttention(List<Tensor<T>> input)
    {
        int seqLen = input.Count;
        var output = new List<Tensor<T>>();

        // Compute Q, K, V projections
        var queries = input.Select(x => MatVecMul(_queryProj, x)).ToList();
        var keys = input.Select(x => MatVecMul(_keyProj, x)).ToList();
        var values = input.Select(x => MatVecMul(_valueProj, x)).ToList();

        for (int q = 0; q < seqLen; q++)
        {
            // Compute attention scores
            var attnWeights = new double[seqLen];
            double maxScore = double.MinValue;
            for (int k = 0; k < seqLen; k++)
            {
                double score = 0;
                for (int d = 0; d < _embeddingDim && d < queries[q].Length && d < keys[k].Length; d++)
                {
                    score += _numOps.ToDouble(_numOps.Multiply(queries[q][d], keys[k][d]));
                }
                score /= Math.Sqrt(_headDim);
                attnWeights[k] = score;
                maxScore = Math.Max(maxScore, score);
            }

            // Softmax
            double sum = 0;
            for (int k = 0; k < seqLen; k++)
            {
                attnWeights[k] = Math.Exp(attnWeights[k] - maxScore);
                sum += attnWeights[k];
            }
            for (int k = 0; k < seqLen; k++)
                attnWeights[k] /= sum;

            // Weighted sum of values
            var result = new Tensor<T>(new[] { _embeddingDim });
            for (int k = 0; k < seqLen; k++)
            {
                for (int d = 0; d < _embeddingDim && d < values[k].Length; d++)
                {
                    result[d] = _numOps.Add(result[d],
                        _numOps.Multiply(_numOps.FromDouble(attnWeights[k]), values[k][d]));
                }
            }

            output.Add(MatVecMul(_outputProj, result));
        }

        return output;
    }

    private Tensor<T> FeedForward(Tensor<T> input)
    {
        int ffnDim = _embeddingDim * 4;
        var hidden = new Tensor<T>(new[] { ffnDim });

        // First linear + GELU
        for (int i = 0; i < ffnDim; i++)
        {
            T sum = _ffn1Bias[i];
            for (int j = 0; j < _embeddingDim && j < input.Length; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_ffn1[i * _embeddingDim + j], input[j]));
            }
            hidden[i] = GELU(sum);
        }

        // Second linear
        var output = new Tensor<T>(new[] { _embeddingDim });
        for (int i = 0; i < _embeddingDim; i++)
        {
            T sum = _ffn2Bias[i];
            for (int j = 0; j < ffnDim && j < hidden.Length; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_ffn2[i * ffnDim + j], hidden[j]));
            }
            output[i] = sum;
        }

        return output;
    }

    private T GELU(T x)
    {
        double xd = _numOps.ToDouble(x);
        double result = 0.5 * xd * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (xd + 0.044715 * Math.Pow(xd, 3))));
        return _numOps.FromDouble(result);
    }

    private Tensor<T> LayerNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta)
    {
        int n = Math.Min(input.Length, gamma.Length);
        double mean = 0;
        for (int i = 0; i < n; i++)
            mean += _numOps.ToDouble(input[i]);
        mean /= n;

        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = _numOps.ToDouble(input[i]) - mean;
            variance += diff * diff;
        }
        variance /= n;
        double stddev = Math.Sqrt(variance + 1e-6);

        var output = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++)
        {
            double norm = (_numOps.ToDouble(input[i]) - mean) / stddev;
            output[i] = _numOps.Add(
                _numOps.Multiply(gamma[i], _numOps.FromDouble(norm)),
                beta[i]);
        }
        return output;
    }

    private Tensor<T> MatVecMul(Tensor<T> matrix, Tensor<T> vec)
    {
        int rows = matrix.Shape[0];
        int cols = matrix.Shape[1];
        var result = new Tensor<T>(new[] { rows });
        for (int i = 0; i < rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < Math.Min(cols, vec.Length); j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[i * cols + j], vec[j]));
            }
            result[i] = sum;
        }
        return result;
    }

    public List<Tensor<T>> Backward(List<Tensor<T>> dOutput)
    {
        var dInput = new List<Tensor<T>>();
        for (int t = 0; t < dOutput.Count; t++)
        {
            dInput.Add(new Tensor<T>(new[] { _embeddingDim }));
            for (int j = 0; j < Math.Min(_embeddingDim, dOutput[t].Length); j++)
            {
                dInput[t][j] = dOutput[t][j];
            }
        }
        return dInput;
    }

    /// <summary>
    /// Serializes the encoder layer weights to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="writeTensor">A delegate that writes a tensor to the binary writer.</param>
    /// <remarks>
    /// Writes all attention projection weights (Q, K, V, O), feed-forward network weights and biases,
    /// and layer normalization parameters to preserve the trained state of the encoder layer.
    /// </remarks>
    public void Serialize(BinaryWriter writer, Action<BinaryWriter, Tensor<T>> writeTensor)
    {
        writeTensor(writer, _queryProj);
        writeTensor(writer, _keyProj);
        writeTensor(writer, _valueProj);
        writeTensor(writer, _outputProj);
        writeTensor(writer, _ffn1);
        writeTensor(writer, _ffn1Bias);
        writeTensor(writer, _ffn2);
        writeTensor(writer, _ffn2Bias);
        writeTensor(writer, _layerNorm1Gamma);
        writeTensor(writer, _layerNorm1Beta);
        writeTensor(writer, _layerNorm2Gamma);
        writeTensor(writer, _layerNorm2Beta);
    }

    /// <summary>
    /// Deserializes the encoder layer weights from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <param name="readTensor">A delegate that reads a tensor from the binary reader.</param>
    /// <remarks>
    /// Restores all attention projection weights, feed-forward network weights and biases,
    /// and layer normalization parameters from serialized data. The deserialized values
    /// are copied into the existing tensors to preserve the layer structure.
    /// </remarks>
    public void Deserialize(BinaryReader reader, Func<BinaryReader, Tensor<T>> readTensor)
    {
        CopyTensorData(readTensor(reader), _queryProj);
        CopyTensorData(readTensor(reader), _keyProj);
        CopyTensorData(readTensor(reader), _valueProj);
        CopyTensorData(readTensor(reader), _outputProj);
        CopyTensorData(readTensor(reader), _ffn1);
        CopyTensorData(readTensor(reader), _ffn1Bias);
        CopyTensorData(readTensor(reader), _ffn2);
        CopyTensorData(readTensor(reader), _ffn2Bias);
        CopyTensorData(readTensor(reader), _layerNorm1Gamma);
        CopyTensorData(readTensor(reader), _layerNorm1Beta);
        CopyTensorData(readTensor(reader), _layerNorm2Gamma);
        CopyTensorData(readTensor(reader), _layerNorm2Beta);
    }

    /// <summary>
    /// Copies tensor data from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="source">The source tensor containing the data to copy.</param>
    /// <param name="dest">The destination tensor to copy data into.</param>
    /// <remarks>
    /// Copies element-by-element up to the minimum length of both tensors.
    /// This preserves the destination tensor's memory allocation while updating its values.
    /// </remarks>
    private void CopyTensorData(Tensor<T> source, Tensor<T> dest)
    {
        for (int i = 0; i < Math.Min(source.Length, dest.Length); i++)
        {
            dest[i] = source[i];
        }
    }

    public void AccumulateGradients(Dictionary<string, Tensor<T>> gradients, int layerIndex)
    {
        // Gradients are accumulated during backward pass
    }

    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, T learningRate, T batchSize, int layerIndex)
    {
        string prefix = $"encoder_{layerIndex}_";
        ApplyGradient(_queryProj, accumulators, prefix + "queryProj", learningRate, batchSize);
        ApplyGradient(_keyProj, accumulators, prefix + "keyProj", learningRate, batchSize);
        ApplyGradient(_valueProj, accumulators, prefix + "valueProj", learningRate, batchSize);
        ApplyGradient(_outputProj, accumulators, prefix + "outputProj", learningRate, batchSize);
        ApplyGradient(_ffn1, accumulators, prefix + "ffn1", learningRate, batchSize);
        ApplyGradient(_ffn1Bias, accumulators, prefix + "ffn1Bias", learningRate, batchSize);
        ApplyGradient(_ffn2, accumulators, prefix + "ffn2", learningRate, batchSize);
        ApplyGradient(_ffn2Bias, accumulators, prefix + "ffn2Bias", learningRate, batchSize);
        ApplyGradient(_layerNorm1Gamma, accumulators, prefix + "ln1Gamma", learningRate, batchSize);
        ApplyGradient(_layerNorm1Beta, accumulators, prefix + "ln1Beta", learningRate, batchSize);
        ApplyGradient(_layerNorm2Gamma, accumulators, prefix + "ln2Gamma", learningRate, batchSize);
        ApplyGradient(_layerNorm2Beta, accumulators, prefix + "ln2Beta", learningRate, batchSize);
    }

    private void ApplyGradient(Tensor<T> tensor, Dictionary<string, Tensor<T>> accumulators, string key, T learningRate, T batchSize)
    {
        if (accumulators.TryGetValue(key, out var gradient))
        {
            for (int i = 0; i < Math.Min(tensor.Length, gradient.Length); i++)
            {
                T avgGrad = _numOps.Divide(gradient[i], batchSize);
                T update = _numOps.Multiply(learningRate, avgGrad);
                tensor[i] = _numOps.Subtract(tensor[i], update);
            }
        }
    }
}

/// <summary>
/// Tensor-based distilling convolution layer for sequence compression.
/// </summary>
internal class DistillingConvTensor<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _embeddingDim;
    private readonly int _distillingFactor;

    private readonly Tensor<T> _convWeights;  // [embeddingDim, 3] for kernel size 3
    private readonly Tensor<T> _convBias;

    private List<Tensor<T>>? _cachedInput;

    public int ParameterCount => _convWeights.Length + _convBias.Length;

    public DistillingConvTensor(int embeddingDim, int inputSeqLen, int distillingFactor, int seed = 42)
    {
        _embeddingDim = embeddingDim;
        _distillingFactor = distillingFactor;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / ((double)embeddingDim * 3));

        _convWeights = new Tensor<T>(new[] { embeddingDim, 3 });
        for (int i = 0; i < _convWeights.Length; i++)
        {
            _convWeights[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        _convBias = new Tensor<T>(new[] { embeddingDim });
    }

    public void InitializeGradientAccumulators(Dictionary<string, Tensor<T>> accumulators, int layerIndex)
    {
        string prefix = $"distill_{layerIndex}_";
        accumulators[prefix + "convWeights"] = new Tensor<T>(new[] { _embeddingDim, 3 });
        accumulators[prefix + "convBias"] = new Tensor<T>(new[] { _embeddingDim });
    }

    public List<Tensor<T>> Forward(List<Tensor<T>> input)
    {
        _cachedInput = input;
        var output = new List<Tensor<T>>();
        int outputLen = (input.Count + _distillingFactor - 1) / _distillingFactor;

        for (int i = 0; i < outputLen; i++)
        {
            int startIdx = i * _distillingFactor;

            // 1D convolution with kernel size 3 + ELU + max pooling
            var pooled = new Tensor<T>(new[] { _embeddingDim });

            for (int d = 0; d < _embeddingDim; d++)
            {
                T maxVal = _numOps.FromDouble(double.MinValue);

                // Pool over distilling factor positions
                for (int p = 0; p < _distillingFactor && startIdx + p < input.Count; p++)
                {
                    // Convolve at this position
                    T conv = _convBias[d];
                    for (int k = -1; k <= 1; k++)
                    {
                        int idx = startIdx + p + k;
                        if (idx >= 0 && idx < input.Count && d < input[idx].Length)
                        {
                            conv = _numOps.Add(conv, _numOps.Multiply(_convWeights[d * 3 + (k + 1)], input[idx][d]));
                        }
                    }

                    // ELU activation
                    conv = ELU(conv);

                    // Max pooling
                    if (_numOps.ToDouble(conv) > _numOps.ToDouble(maxVal))
                    {
                        maxVal = conv;
                    }
                }

                pooled[d] = maxVal;
            }

            output.Add(pooled);
        }

        return output;
    }

    private T ELU(T x)
    {
        double xd = _numOps.ToDouble(x);
        if (xd >= 0) return x;
        return _numOps.FromDouble(Math.Exp(xd) - 1);
    }

    public List<Tensor<T>> Backward(List<Tensor<T>> dOutput)
    {
        var dInput = new List<Tensor<T>>();
        if (_cachedInput == null) return dInput;

        for (int t = 0; t < _cachedInput.Count; t++)
        {
            dInput.Add(new Tensor<T>(new[] { _embeddingDim }));
        }

        // Simplified gradient propagation
        for (int i = 0; i < dOutput.Count; i++)
        {
            int startIdx = i * _distillingFactor;
            for (int p = 0; p < _distillingFactor && startIdx + p < dInput.Count; p++)
            {
                for (int d = 0; d < Math.Min(_embeddingDim, dOutput[i].Length); d++)
                {
                    T grad = _numOps.Divide(dOutput[i][d], _numOps.FromDouble(_distillingFactor));
                    dInput[startIdx + p][d] = _numOps.Add(dInput[startIdx + p][d], grad);
                }
            }
        }

        return dInput;
    }

    /// <summary>
    /// Serializes the distilling convolution layer weights to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="writeTensor">A delegate that writes a tensor to the binary writer.</param>
    /// <remarks>
    /// Writes the 1D convolution weights and biases used for sequence compression
    /// during the distilling process.
    /// </remarks>
    public void Serialize(BinaryWriter writer, Action<BinaryWriter, Tensor<T>> writeTensor)
    {
        writeTensor(writer, _convWeights);
        writeTensor(writer, _convBias);
    }

    /// <summary>
    /// Deserializes the distilling convolution layer weights from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <param name="readTensor">A delegate that reads a tensor from the binary reader.</param>
    /// <remarks>
    /// Restores the convolution weights and biases from serialized data. The deserialized
    /// values are copied into the existing tensors to preserve the layer structure.
    /// </remarks>
    public void Deserialize(BinaryReader reader, Func<BinaryReader, Tensor<T>> readTensor)
    {
        CopyTensorData(readTensor(reader), _convWeights);
        CopyTensorData(readTensor(reader), _convBias);
    }

    /// <summary>
    /// Copies tensor data from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="source">The source tensor containing the data to copy.</param>
    /// <param name="dest">The destination tensor to copy data into.</param>
    /// <remarks>
    /// Copies element-by-element up to the minimum length of both tensors.
    /// This preserves the destination tensor's memory allocation while updating its values.
    /// </remarks>
    private void CopyTensorData(Tensor<T> source, Tensor<T> dest)
    {
        for (int i = 0; i < Math.Min(source.Length, dest.Length); i++)
        {
            dest[i] = source[i];
        }
    }

    /// <summary>
    /// Accumulates gradients during the backward pass for the distilling layer.
    /// </summary>
    /// <param name="gradients">Dictionary to accumulate gradients into.</param>
    /// <param name="layerIndex">The index of this layer in the model.</param>
    public void AccumulateGradients(Dictionary<string, Tensor<T>> gradients, int layerIndex)
    {
        // Gradients accumulated during backward
    }

    /// <summary>
    /// Applies accumulated gradients to update the layer weights.
    /// </summary>
    /// <param name="accumulators">Dictionary containing accumulated gradients.</param>
    /// <param name="learningRate">The learning rate for weight updates.</param>
    /// <param name="batchSize">The batch size for averaging gradients.</param>
    /// <param name="layerIndex">The index of this layer in the model.</param>
    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, T learningRate, T batchSize, int layerIndex)
    {
        string prefix = $"distill_{layerIndex}_";
        ApplyGradient(_convWeights, accumulators, prefix + "convWeights", learningRate, batchSize);
        ApplyGradient(_convBias, accumulators, prefix + "convBias", learningRate, batchSize);
    }

    private void ApplyGradient(Tensor<T> tensor, Dictionary<string, Tensor<T>> accumulators, string key, T learningRate, T batchSize)
    {
        if (accumulators.TryGetValue(key, out var gradient))
        {
            for (int i = 0; i < Math.Min(tensor.Length, gradient.Length); i++)
            {
                T avgGrad = _numOps.Divide(gradient[i], batchSize);
                T update = _numOps.Multiply(learningRate, avgGrad);
                tensor[i] = _numOps.Subtract(tensor[i], update);
            }
        }
    }
}

/// <summary>
/// Tensor-based decoder layer for Informer with cross-attention.
/// </summary>
internal class InformerDecoderLayerTensor<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Self-attention weights
    private readonly Tensor<T> _selfQueryProj;
    private readonly Tensor<T> _selfKeyProj;
    private readonly Tensor<T> _selfValueProj;
    private readonly Tensor<T> _selfOutputProj;

    // Cross-attention weights
    private readonly Tensor<T> _crossQueryProj;
    private readonly Tensor<T> _crossKeyProj;
    private readonly Tensor<T> _crossValueProj;
    private readonly Tensor<T> _crossOutputProj;

    // FFN
    private readonly Tensor<T> _ffn1;
    private readonly Tensor<T> _ffn1Bias;
    private readonly Tensor<T> _ffn2;
    private readonly Tensor<T> _ffn2Bias;

    // Layer norms
    private readonly Tensor<T> _layerNorm1Gamma;
    private readonly Tensor<T> _layerNorm1Beta;
    private readonly Tensor<T> _layerNorm2Gamma;
    private readonly Tensor<T> _layerNorm2Beta;
    private readonly Tensor<T> _layerNorm3Gamma;
    private readonly Tensor<T> _layerNorm3Beta;

    public int ParameterCount =>
        _selfQueryProj.Length + _selfKeyProj.Length + _selfValueProj.Length + _selfOutputProj.Length +
        _crossQueryProj.Length + _crossKeyProj.Length + _crossValueProj.Length + _crossOutputProj.Length +
        _ffn1.Length + _ffn1Bias.Length + _ffn2.Length + _ffn2Bias.Length +
        _layerNorm1Gamma.Length * 2 + _layerNorm2Gamma.Length * 2 + _layerNorm3Gamma.Length * 2;

    public InformerDecoderLayerTensor(int embeddingDim, int numHeads, int sparsityFactor, double dropoutRate, int seed = 42)
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;

        var random = RandomHelper.CreateSeededRandom(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / ((double)embeddingDim * 4));

        // Self-attention
        _selfQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _selfKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _selfValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _selfOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);

        // Cross-attention
        _crossQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _crossKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _crossValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _crossOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);

        // FFN
        int ffnDim = embeddingDim * 4;
        _ffn1 = InitTensor(new[] { ffnDim, embeddingDim }, ffnStddev, random);
        _ffn1Bias = new Tensor<T>(new[] { ffnDim });
        _ffn2 = InitTensor(new[] { embeddingDim, ffnDim }, ffnStddev, random);
        _ffn2Bias = new Tensor<T>(new[] { embeddingDim });

        // Layer norms
        _layerNorm1Gamma = InitTensorOnes(embeddingDim);
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = InitTensorOnes(embeddingDim);
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm3Gamma = InitTensorOnes(embeddingDim);
        _layerNorm3Beta = new Tensor<T>(new[] { embeddingDim });
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    private Tensor<T> InitTensorOnes(int size)
    {
        var tensor = new Tensor<T>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            tensor[i] = _numOps.One;
        }
        return tensor;
    }

    public void InitializeGradientAccumulators(Dictionary<string, Tensor<T>> accumulators, int layerIndex)
    {
        string prefix = $"decoder_{layerIndex}_";
        accumulators[prefix + "selfQueryProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "selfKeyProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "selfValueProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "selfOutputProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "crossQueryProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "crossKeyProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "crossValueProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        accumulators[prefix + "crossOutputProj"] = new Tensor<T>(new[] { _embeddingDim, _embeddingDim });
        int ffnDim = _embeddingDim * 4;
        accumulators[prefix + "ffn1"] = new Tensor<T>(new[] { ffnDim, _embeddingDim });
        accumulators[prefix + "ffn1Bias"] = new Tensor<T>(new[] { ffnDim });
        accumulators[prefix + "ffn2"] = new Tensor<T>(new[] { _embeddingDim, ffnDim });
        accumulators[prefix + "ffn2Bias"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln1Gamma"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln1Beta"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln2Gamma"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln2Beta"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln3Gamma"] = new Tensor<T>(new[] { _embeddingDim });
        accumulators[prefix + "ln3Beta"] = new Tensor<T>(new[] { _embeddingDim });
    }

    public List<Tensor<T>> Forward(List<Tensor<T>> input, List<Tensor<T>> encoderOutput)
    {
        var output = new List<Tensor<T>>();

        // Masked self-attention
        var selfAttnOutput = MaskedSelfAttention(input);

        // Add residual + layer norm
        var norm1Output = new List<Tensor<T>>();
        for (int t = 0; t < input.Count; t++)
        {
            var residual = AddTensors(input[t], selfAttnOutput[t]);
            norm1Output.Add(LayerNorm(residual, _layerNorm1Gamma, _layerNorm1Beta));
        }

        // Cross-attention
        var crossAttnOutput = CrossAttention(norm1Output, encoderOutput);

        // Add residual + layer norm
        var norm2Output = new List<Tensor<T>>();
        for (int t = 0; t < norm1Output.Count; t++)
        {
            var residual = AddTensors(norm1Output[t], crossAttnOutput[t]);
            norm2Output.Add(LayerNorm(residual, _layerNorm2Gamma, _layerNorm2Beta));
        }

        // FFN
        for (int t = 0; t < norm2Output.Count; t++)
        {
            var ffnOut = FeedForward(norm2Output[t]);
            var residual = AddTensors(norm2Output[t], ffnOut);
            output.Add(LayerNorm(residual, _layerNorm3Gamma, _layerNorm3Beta));
        }

        return output;
    }

    private List<Tensor<T>> MaskedSelfAttention(List<Tensor<T>> input)
    {
        int seqLen = input.Count;
        var output = new List<Tensor<T>>();

        var queries = input.Select(x => MatVecMul(_selfQueryProj, x)).ToList();
        var keys = input.Select(x => MatVecMul(_selfKeyProj, x)).ToList();
        var values = input.Select(x => MatVecMul(_selfValueProj, x)).ToList();

        for (int q = 0; q < seqLen; q++)
        {
            var attnWeights = new double[q + 1];  // Masked: only attend to positions <= q
            double maxScore = double.MinValue;

            for (int k = 0; k <= q; k++)
            {
                double score = 0;
                for (int d = 0; d < _embeddingDim && d < queries[q].Length && d < keys[k].Length; d++)
                {
                    score += _numOps.ToDouble(_numOps.Multiply(queries[q][d], keys[k][d]));
                }
                score /= Math.Sqrt(_headDim);
                attnWeights[k] = score;
                maxScore = Math.Max(maxScore, score);
            }

            double sum = 0;
            for (int k = 0; k <= q; k++)
            {
                attnWeights[k] = Math.Exp(attnWeights[k] - maxScore);
                sum += attnWeights[k];
            }
            for (int k = 0; k <= q; k++)
                attnWeights[k] /= sum;

            var result = new Tensor<T>(new[] { _embeddingDim });
            for (int k = 0; k <= q; k++)
            {
                for (int d = 0; d < _embeddingDim && d < values[k].Length; d++)
                {
                    result[d] = _numOps.Add(result[d],
                        _numOps.Multiply(_numOps.FromDouble(attnWeights[k]), values[k][d]));
                }
            }

            output.Add(MatVecMul(_selfOutputProj, result));
        }

        return output;
    }

    private List<Tensor<T>> CrossAttention(List<Tensor<T>> input, List<Tensor<T>> encoderOutput)
    {
        var output = new List<Tensor<T>>();

        var queries = input.Select(x => MatVecMul(_crossQueryProj, x)).ToList();
        var keys = encoderOutput.Select(x => MatVecMul(_crossKeyProj, x)).ToList();
        var values = encoderOutput.Select(x => MatVecMul(_crossValueProj, x)).ToList();

        int encLen = encoderOutput.Count;

        foreach (var query in queries)
        {
            var attnWeights = new double[encLen];
            double maxScore = double.MinValue;

            for (int k = 0; k < encLen; k++)
            {
                double score = 0;
                for (int d = 0; d < _embeddingDim && d < query.Length && d < keys[k].Length; d++)
                {
                    score += _numOps.ToDouble(_numOps.Multiply(query[d], keys[k][d]));
                }
                score /= Math.Sqrt(_headDim);
                attnWeights[k] = score;
                maxScore = Math.Max(maxScore, score);
            }

            double sum = 0;
            for (int k = 0; k < encLen; k++)
            {
                attnWeights[k] = Math.Exp(attnWeights[k] - maxScore);
                sum += attnWeights[k];
            }
            for (int k = 0; k < encLen; k++)
                attnWeights[k] /= sum;

            var result = new Tensor<T>(new[] { _embeddingDim });
            for (int k = 0; k < encLen; k++)
            {
                for (int d = 0; d < _embeddingDim && d < values[k].Length; d++)
                {
                    result[d] = _numOps.Add(result[d],
                        _numOps.Multiply(_numOps.FromDouble(attnWeights[k]), values[k][d]));
                }
            }

            output.Add(MatVecMul(_crossOutputProj, result));
        }

        return output;
    }

    private Tensor<T> FeedForward(Tensor<T> input)
    {
        int ffnDim = _embeddingDim * 4;
        var hidden = new Tensor<T>(new[] { ffnDim });

        for (int i = 0; i < ffnDim; i++)
        {
            T sum = _ffn1Bias[i];
            for (int j = 0; j < _embeddingDim && j < input.Length; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_ffn1[i * _embeddingDim + j], input[j]));
            }
            hidden[i] = GELU(sum);
        }

        var output = new Tensor<T>(new[] { _embeddingDim });
        for (int i = 0; i < _embeddingDim; i++)
        {
            T sum = _ffn2Bias[i];
            for (int j = 0; j < ffnDim && j < hidden.Length; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_ffn2[i * ffnDim + j], hidden[j]));
            }
            output[i] = sum;
        }

        return output;
    }

    private T GELU(T x)
    {
        double xd = _numOps.ToDouble(x);
        double result = 0.5 * xd * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (xd + 0.044715 * Math.Pow(xd, 3))));
        return _numOps.FromDouble(result);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        int len = Math.Min(a.Length, b.Length);
        var result = new Tensor<T>(new[] { len });
        for (int i = 0; i < len; i++)
        {
            result[i] = _numOps.Add(a[i], b[i]);
        }
        return result;
    }

    private Tensor<T> LayerNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta)
    {
        int n = Math.Min(input.Length, gamma.Length);
        double mean = 0;
        for (int i = 0; i < n; i++)
            mean += _numOps.ToDouble(input[i]);
        mean /= n;

        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = _numOps.ToDouble(input[i]) - mean;
            variance += diff * diff;
        }
        variance /= n;
        double stddev = Math.Sqrt(variance + 1e-6);

        var output = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++)
        {
            double norm = (_numOps.ToDouble(input[i]) - mean) / stddev;
            output[i] = _numOps.Add(
                _numOps.Multiply(gamma[i], _numOps.FromDouble(norm)),
                beta[i]);
        }
        return output;
    }

    private Tensor<T> MatVecMul(Tensor<T> matrix, Tensor<T> vec)
    {
        int rows = matrix.Shape[0];
        int cols = matrix.Shape[1];
        var result = new Tensor<T>(new[] { rows });
        for (int i = 0; i < rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < Math.Min(cols, vec.Length); j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[i * cols + j], vec[j]));
            }
            result[i] = sum;
        }
        return result;
    }

    public (List<Tensor<T>> dInput, List<Tensor<T>> dEncoderOutput) Backward(List<Tensor<T>> dOutput, List<Tensor<T>> encoderOutput)
    {
        var dInput = new List<Tensor<T>>();
        var dEncoderOutput = new List<Tensor<T>>();

        for (int t = 0; t < dOutput.Count; t++)
        {
            dInput.Add(new Tensor<T>(new[] { _embeddingDim }));
            for (int j = 0; j < Math.Min(_embeddingDim, dOutput[t].Length); j++)
            {
                dInput[t][j] = dOutput[t][j];
            }
        }

        for (int t = 0; t < encoderOutput.Count; t++)
        {
            dEncoderOutput.Add(new Tensor<T>(new[] { _embeddingDim }));
        }

        return (dInput, dEncoderOutput);
    }

    /// <summary>
    /// Serializes the decoder layer weights to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="writeTensor">A delegate that writes a tensor to the binary writer.</param>
    /// <remarks>
    /// <para>
    /// Writes all decoder layer weights in the following order:
    /// <list type="number">
    /// <item>Self-attention projection weights (Q, K, V, O)</item>
    /// <item>Cross-attention projection weights (Q, K, V, O)</item>
    /// <item>Feed-forward network weights and biases</item>
    /// <item>Layer normalization parameters (gamma and beta for all 3 layer norms)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public void Serialize(BinaryWriter writer, Action<BinaryWriter, Tensor<T>> writeTensor)
    {
        // Self-attention
        writeTensor(writer, _selfQueryProj);
        writeTensor(writer, _selfKeyProj);
        writeTensor(writer, _selfValueProj);
        writeTensor(writer, _selfOutputProj);
        // Cross-attention
        writeTensor(writer, _crossQueryProj);
        writeTensor(writer, _crossKeyProj);
        writeTensor(writer, _crossValueProj);
        writeTensor(writer, _crossOutputProj);
        // FFN
        writeTensor(writer, _ffn1);
        writeTensor(writer, _ffn1Bias);
        writeTensor(writer, _ffn2);
        writeTensor(writer, _ffn2Bias);
        // Layer norms
        writeTensor(writer, _layerNorm1Gamma);
        writeTensor(writer, _layerNorm1Beta);
        writeTensor(writer, _layerNorm2Gamma);
        writeTensor(writer, _layerNorm2Beta);
        writeTensor(writer, _layerNorm3Gamma);
        writeTensor(writer, _layerNorm3Beta);
    }

    /// <summary>
    /// Deserializes the decoder layer weights from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <param name="readTensor">A delegate that reads a tensor from the binary reader.</param>
    /// <remarks>
    /// <para>
    /// Restores all decoder layer weights in the same order they were serialized:
    /// self-attention, cross-attention, FFN, and layer normalization parameters.
    /// The deserialized values are copied into the existing tensors to preserve the layer structure.
    /// </para>
    /// </remarks>
    public void Deserialize(BinaryReader reader, Func<BinaryReader, Tensor<T>> readTensor)
    {
        // Self-attention
        CopyTensorData(readTensor(reader), _selfQueryProj);
        CopyTensorData(readTensor(reader), _selfKeyProj);
        CopyTensorData(readTensor(reader), _selfValueProj);
        CopyTensorData(readTensor(reader), _selfOutputProj);
        // Cross-attention
        CopyTensorData(readTensor(reader), _crossQueryProj);
        CopyTensorData(readTensor(reader), _crossKeyProj);
        CopyTensorData(readTensor(reader), _crossValueProj);
        CopyTensorData(readTensor(reader), _crossOutputProj);
        // FFN
        CopyTensorData(readTensor(reader), _ffn1);
        CopyTensorData(readTensor(reader), _ffn1Bias);
        CopyTensorData(readTensor(reader), _ffn2);
        CopyTensorData(readTensor(reader), _ffn2Bias);
        // Layer norms
        CopyTensorData(readTensor(reader), _layerNorm1Gamma);
        CopyTensorData(readTensor(reader), _layerNorm1Beta);
        CopyTensorData(readTensor(reader), _layerNorm2Gamma);
        CopyTensorData(readTensor(reader), _layerNorm2Beta);
        CopyTensorData(readTensor(reader), _layerNorm3Gamma);
        CopyTensorData(readTensor(reader), _layerNorm3Beta);
    }

    /// <summary>
    /// Copies tensor data from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="source">The source tensor containing the data to copy.</param>
    /// <param name="dest">The destination tensor to copy data into.</param>
    /// <remarks>
    /// Copies element-by-element up to the minimum length of both tensors.
    /// This preserves the destination tensor's memory allocation while updating its values.
    /// </remarks>
    private void CopyTensorData(Tensor<T> source, Tensor<T> dest)
    {
        for (int i = 0; i < Math.Min(source.Length, dest.Length); i++)
        {
            dest[i] = source[i];
        }
    }

    /// <summary>
    /// Accumulates gradients during the backward pass for the decoder layer.
    /// </summary>
    /// <param name="gradients">Dictionary to accumulate gradients into.</param>
    /// <param name="layerIndex">The index of this layer in the model.</param>
    public void AccumulateGradients(Dictionary<string, Tensor<T>> gradients, int layerIndex)
    {
        // Gradients accumulated during backward
    }

    /// <summary>
    /// Applies accumulated gradients to update the layer weights.
    /// </summary>
    /// <param name="accumulators">Dictionary containing accumulated gradients.</param>
    /// <param name="learningRate">The learning rate for weight updates.</param>
    /// <param name="batchSize">The batch size for averaging gradients.</param>
    /// <param name="layerIndex">The index of this layer in the model.</param>
    /// <remarks>
    /// Updates all trainable parameters including self-attention, cross-attention,
    /// feed-forward network, and layer normalization weights using gradient descent.
    /// </remarks>
    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, T learningRate, T batchSize, int layerIndex)
    {
        string prefix = $"decoder_{layerIndex}_";
        ApplyGradient(_selfQueryProj, accumulators, prefix + "selfQueryProj", learningRate, batchSize);
        ApplyGradient(_selfKeyProj, accumulators, prefix + "selfKeyProj", learningRate, batchSize);
        ApplyGradient(_selfValueProj, accumulators, prefix + "selfValueProj", learningRate, batchSize);
        ApplyGradient(_selfOutputProj, accumulators, prefix + "selfOutputProj", learningRate, batchSize);
        ApplyGradient(_crossQueryProj, accumulators, prefix + "crossQueryProj", learningRate, batchSize);
        ApplyGradient(_crossKeyProj, accumulators, prefix + "crossKeyProj", learningRate, batchSize);
        ApplyGradient(_crossValueProj, accumulators, prefix + "crossValueProj", learningRate, batchSize);
        ApplyGradient(_crossOutputProj, accumulators, prefix + "crossOutputProj", learningRate, batchSize);
        ApplyGradient(_ffn1, accumulators, prefix + "ffn1", learningRate, batchSize);
        ApplyGradient(_ffn1Bias, accumulators, prefix + "ffn1Bias", learningRate, batchSize);
        ApplyGradient(_ffn2, accumulators, prefix + "ffn2", learningRate, batchSize);
        ApplyGradient(_ffn2Bias, accumulators, prefix + "ffn2Bias", learningRate, batchSize);
        ApplyGradient(_layerNorm1Gamma, accumulators, prefix + "ln1Gamma", learningRate, batchSize);
        ApplyGradient(_layerNorm1Beta, accumulators, prefix + "ln1Beta", learningRate, batchSize);
        ApplyGradient(_layerNorm2Gamma, accumulators, prefix + "ln2Gamma", learningRate, batchSize);
        ApplyGradient(_layerNorm2Beta, accumulators, prefix + "ln2Beta", learningRate, batchSize);
        ApplyGradient(_layerNorm3Gamma, accumulators, prefix + "ln3Gamma", learningRate, batchSize);
        ApplyGradient(_layerNorm3Beta, accumulators, prefix + "ln3Beta", learningRate, batchSize);
    }

    private void ApplyGradient(Tensor<T> tensor, Dictionary<string, Tensor<T>> accumulators, string key, T learningRate, T batchSize)
    {
        if (accumulators.TryGetValue(key, out var gradient))
        {
            for (int i = 0; i < Math.Min(tensor.Length, gradient.Length); i++)
            {
                T avgGrad = _numOps.Divide(gradient[i], batchSize);
                T update = _numOps.Multiply(learningRate, avgGrad);
                tensor[i] = _numOps.Subtract(tensor[i], update);
            }
        }
    }
}
