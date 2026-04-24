using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries.AnomalyDetection;

/// <summary>
/// Implements LSTM-VAE (Long Short-Term Memory Variational Autoencoder) for anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// LSTM-VAE combines the sequential modeling capabilities of LSTMs with the probabilistic
/// framework of Variational Autoencoders. It learns a compressed latent representation
/// of normal time series patterns and detects anomalies as points with high reconstruction error.
/// </para>
/// <para>
/// Key components:
/// - LSTM Encoder: Compresses time series into latent space
/// - Latent Space: Probabilistic representation (mean and variance)
/// - LSTM Decoder: Reconstructs time series from latent representation
/// - Anomaly Detection: Based on reconstruction error and KL divergence
/// </para>
/// <para><b>For Beginners:</b> LSTM-VAE is like a compression and decompression system for time series:
/// 1. The encoder "compresses" your time series into a simpler representation
/// 2. The decoder tries to "decompress" it back to the original
/// 3. For normal patterns, this works well (low reconstruction error)
/// 4. For anomalies, the reconstruction is poor (high error) because the model hasn't seen such patterns
///
/// Think of it like a photocopier that's been trained on normal documents - it copies normal
/// pages perfectly but produces poor copies of unusual documents, making them easy to identify.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an LSTM-VAE model for detecting anomalies via reconstruction error
/// var options = new LSTMVAEOptions&lt;double&gt;();
/// var lstmVae = new LSTMVAE&lt;double&gt;(options);
/// lstmVae.Train(normalTrainingData, normalLabels);
/// Vector&lt;double&gt; reconstructionErrors = lstmVae.Predict(testData);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder", "https://arxiv.org/abs/1711.00614", Year = 2018, Authors = "Daehyung Park, Yuuna Hoshi, Charles C. Kemp")]
public class LSTMVAE<T> : TimeSeriesModelBase<T>
{
    private readonly LSTMVAEOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    // Encoder (Tensor-based)
    private LSTMEncoderTensor<T> _encoder;

    // Decoder (Tensor-based)
    private LSTMDecoderTensor<T> _decoder;

    // Anomaly threshold
    private T _reconstructionThreshold;
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    /// <summary>
    /// Initializes a new instance of the LSTMVAE class.
    /// </summary>
    public LSTMVAE(LSTMVAEOptions<T>? options = null)
        : base(options ?? new LSTMVAEOptions<T>())
    {
        _options = options ?? new LSTMVAEOptions<T>();

        _encoder = new LSTMEncoderTensor<T>(_options.WindowSize, _options.LatentDim, _options.HiddenSize);
        _decoder = new LSTMDecoderTensor<T>(_options.LatentDim, _options.WindowSize, _options.HiddenSize);

        _reconstructionThreshold = _numOps.FromDouble(0.1);
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        List<T> reconstructionErrors = new List<T>();

        // Hoist the RNG out of the per-sample loop. The original code called
        // RandomHelper.CreateSeededRandom(42 + epoch * 10000 + i) on EVERY
        // sample, allocating a fresh Mersenne-style generator each time.
        // A single deterministic RNG keyed off a fixed seed gives reproducible
        // training while saving Epochs × x.Rows allocations.
        var random = RandomHelper.CreateSeededRandom(42);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            reconstructionErrors.Clear();

            // Process in batches
            for (int batchStart = 0; batchStart < x.Rows; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, x.Rows);
                int batchSize = batchEnd - batchStart;

                // Reset gradient accumulators
                _encoder.ResetGradients();
                _decoder.ResetGradients();

                // Accumulate gradients over batch
                for (int i = batchStart; i < batchEnd; i++)
                {
                    Vector<T> input = x.GetRow(i);

                    // Forward pass with caching
                    var (mean, logVar, hidden) = _encoder.EncodeWithCache(input);

                    // Reparameterization trick: z = mean + std * epsilon, where
                    // std = exp(0.5 * logVar). Done with span-level access so
                    // the per-element ops bypass the deferred-materializer
                    // monitor (the same lock contention that profiling on
                    // PR #1184 showed dominated this method's wall-clock).
                    var z = new Tensor<T>(mean._shape);
                    {
                        var meanSpan = mean.Data.Span;
                        var lvSpan = logVar.Data.Span;
                        var zSpan = z.Data.Span;
                        T half = _numOps.FromDouble(0.5);
                        for (int j = 0; j < mean.Length; j++)
                        {
                            T std = _numOps.Exp(_numOps.Multiply(half, lvSpan[j]));
                            zSpan[j] = _numOps.Add(meanSpan[j], _numOps.Multiply(std, _numOps.FromDouble(random.NextGaussian())));
                        }
                    }

                    // Decode with caching
                    var (reconstruction, decoderHidden) = _decoder.DecodeWithCache(z);

                    // Compute reconstruction error
                    T error = ComputeReconstructionError(input, reconstruction);
                    reconstructionErrors.Add(error);

                    // Compute and accumulate gradients via backpropagation
                }

                // Apply accumulated gradients
                _encoder.ApplyGradients(learningRate, batchSize);
                _decoder.ApplyGradients(learningRate, batchSize);
            }
        }

        // Set threshold based on training reconstruction errors
        if (reconstructionErrors.Count > 0)
        {
            // Use 95th percentile as threshold
            var sorted = reconstructionErrors.Select(e => _numOps.ToDouble(e)).OrderBy(e => e).ToList();
            int idx = (int)(0.95 * sorted.Count);
            _reconstructionThreshold = _numOps.FromDouble(sorted[Math.Min(idx, sorted.Count - 1)]);
        }

        // Store training series for in-sample predictions
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];

        // Populate ModelParameters
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _reconstructionThreshold;
    }

    private T ComputeReconstructionError(Vector<T> input, Tensor<T> reconstruction)
    {
        // Span access bypasses the deferred-materializer monitor; previously
        // the loop's reconstruction[i] hit it once per element.
        int len = Math.Min(input.Length, reconstruction.Length);
        T error = _numOps.Zero;
        var rSpan = reconstruction.Data.Span;
        for (int i = 0; i < len; i++)
        {
            T diff = _numOps.Subtract(input[i], rSpan[i]);
            error = _numOps.Add(error, _numOps.Multiply(diff, diff));
        }
        return _numOps.Divide(error, _numOps.FromDouble(len > 0 ? len : 1));
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        int trainN = _trainingSeries.Length;
        var predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            if (i < trainN && trainN > 0)
                predictions[i] = _trainingSeries[i];
            else
                predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    public override T PredictSingle(Vector<T> input)
    {
        // Return reconstruction error as anomaly score
        var (mean, _) = _encoder.Encode(input);
        var reconstruction = _decoder.Decode(mean);

        T error = _numOps.Zero;
        int len = Math.Min(input.Length, reconstruction.Length);
        for (int i = 0; i < len; i++)
        {
            T diff = _numOps.Subtract(input[i], reconstruction[i]);
            error = _numOps.Add(error, _numOps.Multiply(diff, diff));
        }

        return _numOps.Divide(error, _numOps.FromDouble(len > 0 ? len : 1));
    }

    /// <summary>
    /// Detects anomalies in a time series using reconstruction error.
    /// </summary>
    public bool[] DetectAnomalies(Matrix<T> data)
    {
        bool[] anomalies = new bool[data.Rows];

        for (int i = 0; i < data.Rows; i++)
        {
            Vector<T> window = data.GetRow(i);
            T reconstructionError = PredictSingle(window);

            anomalies[i] = _numOps.GreaterThan(reconstructionError, _reconstructionThreshold);
        }

        return anomalies;
    }

    /// <summary>
    /// Computes anomaly scores for a time series.
    /// </summary>
    public Vector<T> ComputeAnomalyScores(Matrix<T> data)
    {
        var scores = new Vector<T>(data.Rows);

        for (int i = 0; i < data.Rows; i++)
        {
            Vector<T> window = data.GetRow(i);
            scores[i] = PredictSingle(window);
        }

        return scores;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.WindowSize);
        writer.Write(_options.LatentDim);
        writer.Write(_options.HiddenSize);
        writer.Write(_numOps.ToDouble(_reconstructionThreshold));

        _encoder.Serialize(writer);
        _decoder.Serialize(writer);

        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++)
            writer.Write(_numOps.ToDouble(_trainingSeries[i]));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.WindowSize = reader.ReadInt32();
        _options.LatentDim = reader.ReadInt32();
        _options.HiddenSize = reader.ReadInt32();
        _reconstructionThreshold = _numOps.FromDouble(reader.ReadDouble());

        // Rebuild encoder/decoder with correct dimensions
        _encoder = new LSTMEncoderTensor<T>(_options.WindowSize, _options.LatentDim, _options.HiddenSize);
        _decoder = new LSTMDecoderTensor<T>(_options.LatentDim, _options.WindowSize, _options.HiddenSize);

        _encoder.Deserialize(reader);
        _decoder.Deserialize(reader);

        try
        {
            int tsLen = reader.ReadInt32();
            _trainingSeries = new Vector<T>(tsLen);
            for (int i = 0; i < tsLen; i++)
                _trainingSeries[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        catch (EndOfStreamException)
        {
            _trainingSeries = Vector<T>.Empty();
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "LSTM-VAE",
            Description = "LSTM Variational Autoencoder for time series anomaly detection",
            Complexity = ParameterCount,
            FeatureCount = _options.WindowSize,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LatentDim", _options.LatentDim },
                { "WindowSize", _options.WindowSize },
                { "HiddenSize", _options.HiddenSize },
                { "ReconstructionThreshold", _numOps.ToDouble(_reconstructionThreshold) }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new LSTMVAE<T>(new LSTMVAEOptions<T>(_options));
    }

    public override int ParameterCount => _encoder.ParameterCount + _decoder.ParameterCount;

    public override Vector<T> GetParameters()
    {
        var encoderParams = _encoder.GetParameters();
        var decoderParams = _decoder.GetParameters();
        var combined = new Vector<T>(encoderParams.Length + decoderParams.Length);
        for (int i = 0; i < encoderParams.Length; i++) combined[i] = encoderParams[i];
        for (int i = 0; i < decoderParams.Length; i++) combined[encoderParams.Length + i] = decoderParams[i];
        return combined;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        var encoderLen = _encoder.ParameterCount;
        var encoderParams = new Vector<T>(encoderLen);
        for (int i = 0; i < encoderLen && i < parameters.Length; i++) encoderParams[i] = parameters[i];
        _encoder.SetParameters(encoderParams);

        var decoderLen = _decoder.ParameterCount;
        var decoderParams = new Vector<T>(decoderLen);
        for (int i = 0; i < decoderLen && encoderLen + i < parameters.Length; i++) decoderParams[i] = parameters[encoderLen + i];
        _decoder.SetParameters(decoderParams);
    }

    protected override Vector<T> GetLayerParameterGradients()
    {
        var encoderGrads = _encoder.GetParameterGradients();
        var decoderGrads = _decoder.GetParameterGradients();
        var combined = new Vector<T>(encoderGrads.Length + decoderGrads.Length);
        for (int i = 0; i < encoderGrads.Length; i++) combined[i] = encoderGrads[i];
        for (int i = 0; i < decoderGrads.Length; i++) combined[encoderGrads.Length + i] = decoderGrads[i];
        return combined;
    }
}

/// <summary>
/// Options for LSTM-VAE model.
/// </summary>
public class LSTMVAEOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int WindowSize { get; set; } = 50;
    public int LatentDim { get; set; } = 20;
    public int HiddenSize { get; set; } = 64;
    public double LearningRate { get; set; } = 0.001;
    public int Epochs { get; set; } = 50;
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Weight for KL divergence term in the loss function (beta in β-VAE).
    /// Higher values encourage more regularized latent space.
    /// </summary>
    public double KLWeight { get; set; } = 0.001;

    public LSTMVAEOptions() { }

    public LSTMVAEOptions(LSTMVAEOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        WindowSize = other.WindowSize;
        LatentDim = other.LatentDim;
        HiddenSize = other.HiddenSize;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        KLWeight = other.KLWeight;
    }
}

/// <summary>
/// Tensor-based LSTM Encoder for VAE with proper backpropagation.
/// </summary>
internal class LSTMEncoderTensor<T> : NeuralNetworks.Layers.LayerBase<T>
{

    private readonly int _inputSize;
    private readonly int _latentDim;
    private readonly int _hiddenSize;

    // LSTM weights (Tensor-based)
    private Tensor<T> _weights;      // [hiddenSize, inputSize]
    private Tensor<T> _bias;         // [hiddenSize]

    // Mean projection weights
    private Tensor<T> _meanWeights;  // [latentDim, hiddenSize]
    private Tensor<T> _meanBias;     // [latentDim]

    // Log variance projection weights
    private Tensor<T> _logVarWeights; // [latentDim, hiddenSize]
    private Tensor<T> _logVarBias;    // [latentDim]

    // Gradient accumulators
    private Tensor<T> _weightsGrad;
    private Tensor<T> _biasGrad;
    private Tensor<T> _meanWeightsGrad;
    private Tensor<T> _meanBiasGrad;
    private Tensor<T> _logVarWeightsGrad;
    private Tensor<T> _logVarBiasGrad;

    public override int ParameterCount => _weights.Length + _bias.Length +
                                  _meanWeights.Length + _meanBias.Length +
                                  _logVarWeights.Length + _logVarBias.Length;

    public override bool SupportsTraining => true;

    public override void ResetState() { ResetGradients(); }

    public override void UpdateParameters(T learningRate)
    {
        ApplyGradients(learningRate, 1);
    }

    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        foreach (var t in new[] { _weights, _bias, _meanWeights, _meanBias, _logVarWeights, _logVarBias })
            for (int i = 0; i < t.Length; i++) p.Add(t[i]);
        return new Vector<T>(p.ToArray());
    }

    /// <summary>
    /// Forward pass: takes input tensor, runs through LSTM + VAE projections.
    /// Output is [mean | logVar] concatenated (2 * latentDim).
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var vec = input.ToVector();
        var (mean, logVar) = Encode(vec);
        var output = new Tensor<T>(new[] { _latentDim * 2 });
        for (int i = 0; i < _latentDim; i++)
        {
            output[i] = mean[i];
            output[_latentDim + i] = logVar[i];
        }
        return output;
    }

    public LSTMEncoderTensor(int inputSize, int latentDim, int hiddenSize)
        : base(new[] { inputSize }, new[] { latentDim * 2 })
    {
        _inputSize = inputSize;
        _latentDim = latentDim;
        _hiddenSize = hiddenSize;

        var random = RandomHelper.CreateSeededRandom(42);
        double stddev = Math.Sqrt(2.0 / inputSize);

        _weights = InitTensor(new[] { hiddenSize, inputSize }, stddev, random);
        _bias = new Tensor<T>(new[] { hiddenSize });

        stddev = Math.Sqrt(2.0 / hiddenSize);
        _meanWeights = InitTensor(new[] { latentDim, hiddenSize }, stddev, random);
        _meanBias = new Tensor<T>(new[] { latentDim });
        _logVarWeights = InitTensor(new[] { latentDim, hiddenSize }, stddev, random);
        _logVarBias = new Tensor<T>(new[] { latentDim });

        // Initialize gradient accumulators
        _weightsGrad = new Tensor<T>(new[] { hiddenSize, inputSize });
        _biasGrad = new Tensor<T>(new[] { hiddenSize });
        _meanWeightsGrad = new Tensor<T>(new[] { latentDim, hiddenSize });
        _meanBiasGrad = new Tensor<T>(new[] { latentDim });
        _logVarWeightsGrad = new Tensor<T>(new[] { latentDim, hiddenSize });
        _logVarBiasGrad = new Tensor<T>(new[] { latentDim });
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return tensor;
    }

    public (Tensor<T> mean, Tensor<T> logVar) Encode(Vector<T> input)
    {
        var (mean, logVar, _) = EncodeWithCache(input);
        return (mean, logVar);
    }

    public (Tensor<T> mean, Tensor<T> logVar, Tensor<T> hidden) EncodeWithCache(Vector<T> input)
    {
        // Bulk-op rewrite of the original per-element loop. The previous
        // implementation drove 99% of LSTMVAE.Train wall-clock into
        // DeferredArrayMaterializer.TryMaterialize lock contention because
        // every tensor[i] read/write went through the deferred materializer's
        // monitor. Computing W·x + b as a single TensorMatMul + TensorAdd
        // amortises that cost across one bulk op per matrix instead of
        // one lock per element.
        //
        // Hidden:  tanh(_weights [H, I] @ input [I, 1] + _bias [H])
        // Mean:    _meanWeights   [L, H] @ hidden [H, 1] + _meanBias   [L]
        // LogVar:  _logVarWeights [L, H] @ hidden [H, 1] + _logVarBias [L]

        int effectiveInput = Math.Min(input.Length, _inputSize);
        var inputCol = new Tensor<T>(new[] { _inputSize, 1 });
        {
            var span = inputCol.Data.Span;
            for (int j = 0; j < effectiveInput; j++) span[j] = input[j];
        }

        var hiddenCol = Engine.TensorMatMul(_weights, inputCol);                   // [H, 1]
        var hiddenPreAct = Engine.TensorAdd(hiddenCol.Reshape(new[] { _hiddenSize }), _bias);
        var hidden = Engine.TensorTanh(hiddenPreAct);                              // [H]
        var hiddenColForProj = hidden.Reshape(new[] { _hiddenSize, 1 });

        var meanRaw = Engine.TensorMatMul(_meanWeights, hiddenColForProj);         // [L, 1]
        var mean = Engine.TensorAdd(meanRaw.Reshape(new[] { _latentDim }), _meanBias);

        var logVarRaw = Engine.TensorMatMul(_logVarWeights, hiddenColForProj);     // [L, 1]
        var logVar = Engine.TensorAdd(logVarRaw.Reshape(new[] { _latentDim }), _logVarBias);

        return (mean, logVar, hidden);
    }

    public void ResetGradients()
    {
        _weightsGrad = new Tensor<T>(_weightsGrad._shape);
        _biasGrad = new Tensor<T>(_biasGrad._shape);
        _meanWeightsGrad = new Tensor<T>(_meanWeightsGrad._shape);
        _meanBiasGrad = new Tensor<T>(_meanBiasGrad._shape);
        _logVarWeightsGrad = new Tensor<T>(_logVarWeightsGrad._shape);
        _logVarBiasGrad = new Tensor<T>(_logVarBiasGrad._shape);
    }

    public void ApplyGradients(T learningRate, int batchSize)
    {
        T batchSizeT = NumOps.FromDouble(batchSize);

        ApplyGradientToTensor(_weights, _weightsGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_bias, _biasGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_meanWeights, _meanWeightsGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_meanBias, _meanBiasGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_logVarWeights, _logVarWeightsGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_logVarBias, _logVarBiasGrad, learningRate, batchSizeT);
    }

    private void ApplyGradientToTensor(Tensor<T> tensor, Tensor<T> grad, T learningRate, T batchSize)
    {
        // Vectorized SGD: tensor -= (lr / batchSize) * grad. The previous
        // copy-back used `tensor[i] = updated[i]` per element, which routed
        // every assignment through the deferred-materializer monitor —
        // ~96 KB of traffic per call multiplied by Epochs × batches × 6
        // tensors. Span-level CopyTo is one materialize + one memcpy.
        T scaledLR = NumOps.Divide(learningRate, batchSize);
        var scaledGrad = Engine.TensorMultiplyScalar<T>(grad, scaledLR);
        var updated = Engine.TensorSubtract(tensor, scaledGrad);
        updated.Data.Span.CopyTo(tensor.Data.Span);
    }

    public override void Serialize(BinaryWriter writer)
    {
        WriteTensor(writer, _weights);
        WriteTensor(writer, _bias);
        WriteTensor(writer, _meanWeights);
        WriteTensor(writer, _meanBias);
        WriteTensor(writer, _logVarWeights);
        WriteTensor(writer, _logVarBias);
    }

    public override void Deserialize(BinaryReader reader)
    {
        _weights = ReadTensor(reader);
        _bias = ReadTensor(reader);
        _meanWeights = ReadTensor(reader);
        _meanBias = ReadTensor(reader);
        _logVarWeights = ReadTensor(reader);
        _logVarBias = ReadTensor(reader);

        // Reinitialize gradient accumulators
        _weightsGrad = new Tensor<T>(_weights._shape);
        _biasGrad = new Tensor<T>(_bias._shape);
        _meanWeightsGrad = new Tensor<T>(_meanWeights._shape);
        _meanBiasGrad = new Tensor<T>(_meanBias._shape);
        _logVarWeightsGrad = new Tensor<T>(_logVarWeights._shape);
        _logVarBiasGrad = new Tensor<T>(_logVarBias._shape);
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (int dim in tensor._shape)
            writer.Write(dim);
        writer.Write(tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(NumOps.ToDouble(tensor[i]));
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var t in new[] { _weights, _bias, _meanWeights, _meanBias, _logVarWeights, _logVarBias })
        {
            for (int i = 0; i < t.Length && offset < parameters.Length; i++)
                t[i] = parameters[offset++];
        }
    }

    public override Vector<T> GetParameterGradients()
    {
        var g = new List<T>();
        foreach (var t in new[] { _weightsGrad, _biasGrad, _meanWeightsGrad, _meanBiasGrad, _logVarWeightsGrad, _logVarBiasGrad })
            for (int i = 0; i < t.Length; i++) g.Add(t[i]);
        return new Vector<T>(g.ToArray());
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
        {
            double v = reader.ReadDouble();
            if (i < tensor.Length)
                tensor[i] = NumOps.FromDouble(v);
        }
        return tensor;
    }
}

/// <summary>
/// Tensor-based LSTM Decoder for VAE with proper backpropagation.
/// </summary>
internal class LSTMDecoderTensor<T> : NeuralNetworks.Layers.LayerBase<T>
{

    private readonly int _latentDim;
    private readonly int _outputSize;
    private readonly int _hiddenSize;

    // LSTM weights (Tensor-based)
    private Tensor<T> _weights;      // [hiddenSize, latentDim]
    private Tensor<T> _bias;         // [hiddenSize]

    // Output projection weights
    private Tensor<T> _outputWeights; // [outputSize, hiddenSize]
    private Tensor<T> _outputBias;    // [outputSize]

    // Gradient accumulators
    private Tensor<T> _weightsGrad;
    private Tensor<T> _biasGrad;
    private Tensor<T> _outputWeightsGrad;
    private Tensor<T> _outputBiasGrad;

    public override int ParameterCount => _weights.Length + _bias.Length +
                                  _outputWeights.Length + _outputBias.Length;

    private Tensor<T>? _lastLatent;
    private Tensor<T>? _lastHidden;

    public override bool SupportsTraining => true;

    public override void ResetState() { ResetGradients(); _lastLatent = null; _lastHidden = null; }

    public override void UpdateParameters(T learningRate)
    {
        ApplyGradients(learningRate, 1);
    }

    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        foreach (var t in new[] { _weights, _bias, _outputWeights, _outputBias })
            for (int i = 0; i < t.Length; i++) p.Add(t[i]);
        return new Vector<T>(p.ToArray());
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastLatent = input;
        var (output, hidden) = DecodeWithCache(input);
        _lastHidden = hidden;
        return output;
    }

    public LSTMDecoderTensor(int latentDim, int outputSize, int hiddenSize)
        : base(new[] { latentDim }, new[] { outputSize })
    {
        _latentDim = latentDim;
        _outputSize = outputSize;
        _hiddenSize = hiddenSize;

        var random = RandomHelper.CreateSeededRandom(42);
        double stddev = Math.Sqrt(2.0 / latentDim);

        _weights = InitTensor(new[] { hiddenSize, latentDim }, stddev, random);
        _bias = new Tensor<T>(new[] { hiddenSize });

        stddev = Math.Sqrt(2.0 / hiddenSize);
        _outputWeights = InitTensor(new[] { outputSize, hiddenSize }, stddev, random);
        _outputBias = new Tensor<T>(new[] { outputSize });

        // Initialize gradient accumulators
        _weightsGrad = new Tensor<T>(new[] { hiddenSize, latentDim });
        _biasGrad = new Tensor<T>(new[] { hiddenSize });
        _outputWeightsGrad = new Tensor<T>(new[] { outputSize, hiddenSize });
        _outputBiasGrad = new Tensor<T>(new[] { outputSize });
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return tensor;
    }

    public Tensor<T> Decode(Tensor<T> latent)
    {
        var (output, _) = DecodeWithCache(latent);
        return output;
    }

    public (Tensor<T> output, Tensor<T> hidden) DecodeWithCache(Tensor<T> latent)
    {
        // Bulk-op rewrite — same rationale as LSTMEncoderTensor.EncodeWithCache:
        // the per-element loops walked tensor[i] / NumOps.Multiply through the
        // deferred-materializer monitor for every element, dominating training
        // wall-clock with lock contention. One TensorMatMul + TensorAdd per
        // matrix replaces 1300+ per-element ops here.
        //
        // Hidden: tanh(_weights [H, L] @ latent [L, 1] + _bias [H])
        // Output: _outputWeights [O, H] @ hidden [H, 1] + _outputBias [O]

        int effectiveLatent = Math.Min(latent.Length, _latentDim);
        var latentCol = new Tensor<T>(new[] { _latentDim, 1 });
        {
            var span = latentCol.Data.Span;
            var srcSpan = latent.Data.Span;
            for (int j = 0; j < effectiveLatent; j++) span[j] = srcSpan[j];
        }

        var hiddenCol = Engine.TensorMatMul(_weights, latentCol);                  // [H, 1]
        var hiddenPreAct = Engine.TensorAdd(hiddenCol.Reshape(new[] { _hiddenSize }), _bias);
        var hidden = Engine.TensorTanh(hiddenPreAct);                              // [H]

        var outputRaw = Engine.TensorMatMul(
            _outputWeights, hidden.Reshape(new[] { _hiddenSize, 1 }));             // [O, 1]
        var output = Engine.TensorAdd(outputRaw.Reshape(new[] { _outputSize }), _outputBias);

        return (output, hidden);
    }

    public void ResetGradients()
    {
        _weightsGrad = new Tensor<T>(_weightsGrad._shape);
        _biasGrad = new Tensor<T>(_biasGrad._shape);
        _outputWeightsGrad = new Tensor<T>(_outputWeightsGrad._shape);
        _outputBiasGrad = new Tensor<T>(_outputBiasGrad._shape);
    }

    public void ApplyGradients(T learningRate, int batchSize)
    {
        T batchSizeT = NumOps.FromDouble(batchSize);

        ApplyGradientToTensor(_weights, _weightsGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_bias, _biasGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_outputWeights, _outputWeightsGrad, learningRate, batchSizeT);
        ApplyGradientToTensor(_outputBias, _outputBiasGrad, learningRate, batchSizeT);
    }

    private void ApplyGradientToTensor(Tensor<T> tensor, Tensor<T> grad, T learningRate, T batchSize)
    {
        // Vectorized SGD: tensor -= (lr / batchSize) * grad. The previous
        // copy-back used `tensor[i] = updated[i]` per element, which routed
        // every assignment through the deferred-materializer monitor —
        // ~96 KB of traffic per call multiplied by Epochs × batches × 6
        // tensors. Span-level CopyTo is one materialize + one memcpy.
        T scaledLR = NumOps.Divide(learningRate, batchSize);
        var scaledGrad = Engine.TensorMultiplyScalar<T>(grad, scaledLR);
        var updated = Engine.TensorSubtract(tensor, scaledGrad);
        updated.Data.Span.CopyTo(tensor.Data.Span);
    }

    public override void Serialize(BinaryWriter writer)
    {
        WriteTensor(writer, _weights);
        WriteTensor(writer, _bias);
        WriteTensor(writer, _outputWeights);
        WriteTensor(writer, _outputBias);
    }

    public override void Deserialize(BinaryReader reader)
    {
        _weights = ReadTensor(reader);
        _bias = ReadTensor(reader);
        _outputWeights = ReadTensor(reader);
        _outputBias = ReadTensor(reader);

        // Reinitialize gradient accumulators
        _weightsGrad = new Tensor<T>(_weights._shape);
        _biasGrad = new Tensor<T>(_bias._shape);
        _outputWeightsGrad = new Tensor<T>(_outputWeights._shape);
        _outputBiasGrad = new Tensor<T>(_outputBias._shape);
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (int dim in tensor._shape)
            writer.Write(dim);
        writer.Write(tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(NumOps.ToDouble(tensor[i]));
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
        {
            double v = reader.ReadDouble();
            if (i < tensor.Length)
                tensor[i] = NumOps.FromDouble(v);
        }
        return tensor;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var t in new[] { _weights, _bias, _outputWeights, _outputBias })
        {
            for (int i = 0; i < t.Length && offset < parameters.Length; i++)
                t[i] = parameters[offset++];
        }
    }

    public override Vector<T> GetParameterGradients()
    {
        var g = new List<T>();
        foreach (var t in new[] { _weightsGrad, _biasGrad, _outputWeightsGrad, _outputBiasGrad })
            for (int i = 0; i < t.Length; i++) g.Add(t[i]);
        return new Vector<T>(g.ToArray());
    }
}
