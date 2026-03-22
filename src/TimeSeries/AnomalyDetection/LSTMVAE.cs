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
[ModelPaper("A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder", "https://arxiv.org/abs/1711.00614", Year = 2018, Authors = "Daehyung Park, Yuuna Hoshi, Charles C. Kemp")]
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

                    // Reparameterization trick: z = mean + std * epsilon
                    var z = new Tensor<T>(mean.Shape);
                    var random = RandomHelper.CreateSeededRandom(42 + epoch * 10000 + i);
                    for (int j = 0; j < mean.Length; j++)
                    {
                        // z = mean + exp(0.5 * logVar) * epsilon
                        T std = _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(0.5), logVar[j]));
                        z[j] = _numOps.Add(mean[j], _numOps.Multiply(std, _numOps.FromDouble(random.NextGaussian())));
                    }

                    // Decode with caching
                    var (reconstruction, decoderHidden) = _decoder.DecodeWithCache(z);

                    // Compute reconstruction error
                    T error = ComputeReconstructionError(input, reconstruction);
                    reconstructionErrors.Add(error);

                    // Compute and accumulate gradients via backpropagation
                    ComputeAndAccumulateGradients(input, mean, logVar, hidden, z, reconstruction, decoderHidden);
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

    /// <summary>
    /// Computes gradients via backpropagation and accumulates them.
    /// </summary>
    private void ComputeAndAccumulateGradients(
        Vector<T> input,
        Tensor<T> mean,
        Tensor<T> logVar,
        Tensor<T> encoderHidden,
        Tensor<T> latent,
        Tensor<T> reconstruction,
        Tensor<T> decoderHidden)
    {
        int outputSize = Math.Min(input.Length, reconstruction.Length);

        // Compute reconstruction loss gradient: dL/d(reconstruction) = 2 * (reconstruction - input) / n
        var dReconstruction = new Tensor<T>(new[] { outputSize });
        for (int i = 0; i < outputSize; i++)
        {
            T diff = _numOps.Subtract(reconstruction[i], input[i]);
            dReconstruction[i] = _numOps.Divide(
                _numOps.Multiply(_numOps.FromDouble(2.0), diff),
                _numOps.FromDouble(outputSize));
        }

        // Backpropagate through decoder (pass latent for weight gradient computation)
        var dLatent = _decoder.Backward(dReconstruction, decoderHidden, latent);

        // Compute gradients through reparameterization trick
        // z = mean + exp(0.5 * logVar) * epsilon
        // dz/dmean = 1, dz/dlogVar = 0.5 * exp(0.5 * logVar) * epsilon = 0.5 * std * epsilon
        // Where std = exp(0.5 * logVar) and z = mean + std * epsilon
        // We can recover epsilon = (z - mean) / std
        T klWeight = _numOps.FromDouble(_options.KLWeight);
        var dMean = new Tensor<T>(mean.Shape);
        var dLogVar = new Tensor<T>(logVar.Shape);

        for (int i = 0; i < mean.Length; i++)
        {
            // Compute std and epsilon from stored latent and mean
            T std = _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(0.5), logVar[i]));
            T stdEps = _numOps.FromDouble(1e-8);
            T stdSafe = _numOps.Add(std, stdEps);
            T epsilon = _numOps.Divide(_numOps.Subtract(latent[i], mean[i]), stdSafe);

            // Gradient for mean: dL/dmean = dL/dz * dz/dmean + dKL/dmean
            // dz/dmean = 1, dKL/dmean = mean
            dMean[i] = _numOps.Add(dLatent[i], _numOps.Multiply(klWeight, mean[i]));

            // Gradient for logVar: dL/dlogVar = dL/dz * dz/dlogVar + dKL/dlogVar
            // dz/dlogVar = 0.5 * std * epsilon
            // dKL/dlogVar = 0.5 * (exp(logVar) - 1)
            T reparamGrad = _numOps.Multiply(dLatent[i],
                _numOps.Multiply(_numOps.FromDouble(0.5),
                    _numOps.Multiply(std, epsilon)));

            T expLogVar = _numOps.Exp(logVar[i]);
            T klGrad = _numOps.Multiply(_numOps.FromDouble(0.5),
                _numOps.Subtract(expLogVar, _numOps.One));

            dLogVar[i] = _numOps.Add(reparamGrad, _numOps.Multiply(klWeight, klGrad));
        }

        // Backpropagate through encoder (pass input for weight gradient computation)
        _encoder.Backward(dMean, dLogVar, encoderHidden, input);
    }

    private T ComputeReconstructionError(Vector<T> input, Tensor<T> reconstruction)
    {
        T error = _numOps.Zero;
        int len = Math.Min(input.Length, reconstruction.Length);

        for (int i = 0; i < len; i++)
        {
            T diff = _numOps.Subtract(input[i], reconstruction[i]);
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
    public override bool SupportsJitCompilation => true;

    public override void ResetState() { ResetGradients(); }

    public override void UpdateParameters(T learningRate)
    {
        ApplyGradients(learningRate, 1);
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> nodes)
    {
        return Autodiff.TensorOperations<T>.Variable(new Tensor<T>(new[] { _latentDim * 2 }), "lstm_encoder_output");
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

    /// <summary>
    /// Backward pass: dOutput is gradient w.r.t. [mean | logVar].
    /// Delegates to existing BackwardInternal with split gradients.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var dMean = new Tensor<T>(new[] { _latentDim });
        var dLogVar = new Tensor<T>(new[] { _latentDim });
        for (int i = 0; i < _latentDim && i < outputGradient.Length; i++)
        {
            dMean[i] = outputGradient[i];
            if (i + _latentDim < outputGradient.Length)
                dLogVar[i] = outputGradient[_latentDim + i];
        }
        // Use zero hidden and input for gradient computation
        var hidden = new Tensor<T>(new[] { _hiddenSize });
        var input = new Vector<T>(_inputSize);
        Backward(dMean, dLogVar, hidden, input);
        return new Tensor<T>(new[] { _inputSize });
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
        // Simple LSTM-like encoding: hidden = tanh(W * input + bias) using Engine
        int effectiveInput = Math.Min(input.Length, _inputSize);
        var inputVec = new Vector<T>(effectiveInput);
        for (int j = 0; j < effectiveInput; j++) inputVec[j] = input[j];

        var hidden = new Tensor<T>(new[] { _hiddenSize });
        for (int i = 0; i < _hiddenSize; i++)
        {
            var wSlice = new Vector<T>(effectiveInput);
            for (int j = 0; j < effectiveInput; j++) wSlice[j] = _weights[i * _inputSize + j];
            hidden[i] = MathHelper.Tanh(NumOps.Add(_bias[i], Engine.DotProduct(wSlice, inputVec)));
        }

        // Compute mean: mean = meanWeights * hidden + meanBias using Engine
        var hiddenVec = new Vector<T>(_hiddenSize);
        for (int j = 0; j < _hiddenSize; j++) hiddenVec[j] = hidden[j];

        var mean = new Tensor<T>(new[] { _latentDim });
        for (int i = 0; i < _latentDim; i++)
        {
            var mwSlice = new Vector<T>(_hiddenSize);
            for (int j = 0; j < _hiddenSize; j++) mwSlice[j] = _meanWeights[i * _hiddenSize + j];
            mean[i] = NumOps.Add(_meanBias[i], Engine.DotProduct(mwSlice, hiddenVec));
        }

        // Compute log variance: logVar = logVarWeights * hidden + logVarBias
        var logVar = new Tensor<T>(new[] { _latentDim });
        for (int i = 0; i < _latentDim; i++)
        {
            var lvSlice = new Vector<T>(_hiddenSize);
            for (int j = 0; j < _hiddenSize; j++) lvSlice[j] = _logVarWeights[i * _hiddenSize + j];
            logVar[i] = NumOps.Add(_logVarBias[i], Engine.DotProduct(lvSlice, hiddenVec));
        }

        return (mean, logVar, hidden);
    }

    public void Backward(Tensor<T> dMean, Tensor<T> dLogVar, Tensor<T> hidden, Vector<T> input)
    {
        // Gradients for mean projection: dMeanWeights = dMean * hidden^T, dMeanBias = dMean
        for (int i = 0; i < _latentDim; i++)
        {
            _meanBiasGrad[i] = NumOps.Add(_meanBiasGrad[i], dMean[i]);
            for (int j = 0; j < _hiddenSize; j++)
            {
                int idx = i * _hiddenSize + j;
                T grad = NumOps.Multiply(dMean[i], hidden[j]);
                _meanWeightsGrad[idx] = NumOps.Add(_meanWeightsGrad[idx], grad);
            }
        }

        // Gradients for logVar projection
        for (int i = 0; i < _latentDim; i++)
        {
            _logVarBiasGrad[i] = NumOps.Add(_logVarBiasGrad[i], dLogVar[i]);
            for (int j = 0; j < _hiddenSize; j++)
            {
                int idx = i * _hiddenSize + j;
                T grad = NumOps.Multiply(dLogVar[i], hidden[j]);
                _logVarWeightsGrad[idx] = NumOps.Add(_logVarWeightsGrad[idx], grad);
            }
        }

        // Compute gradient w.r.t. hidden using Engine.DotProduct per column
        var dMeanVec = new Vector<T>(_latentDim);
        var dLogVarVec = new Vector<T>(_latentDim);
        for (int i = 0; i < _latentDim; i++) { dMeanVec[i] = dMean[i]; dLogVarVec[i] = dLogVar[i]; }

        var dHidden = new Tensor<T>(new[] { _hiddenSize });
        for (int j = 0; j < _hiddenSize; j++)
        {
            var mwCol = new Vector<T>(_latentDim);
            var lvCol = new Vector<T>(_latentDim);
            for (int i = 0; i < _latentDim; i++)
            {
                mwCol[i] = _meanWeights[i * _hiddenSize + j];
                lvCol[i] = _logVarWeights[i * _hiddenSize + j];
            }
            dHidden[j] = NumOps.Add(Engine.DotProduct(mwCol, dMeanVec), Engine.DotProduct(lvCol, dLogVarVec));
        }

        // Apply tanh derivative: dHidden * (1 - hidden^2)
        for (int i = 0; i < _hiddenSize; i++)
        {
            T h = hidden[i];
            T tanhDeriv = NumOps.Subtract(NumOps.One, NumOps.Multiply(h, h));
            dHidden[i] = NumOps.Multiply(dHidden[i], tanhDeriv);
        }

        // Gradients for encoder weights: dWeights = dHidden * input^T, dBias = dHidden
        for (int i = 0; i < _hiddenSize; i++)
        {
            _biasGrad[i] = NumOps.Add(_biasGrad[i], dHidden[i]);
            for (int j = 0; j < Math.Min(input.Length, _inputSize); j++)
            {
                int idx = i * _inputSize + j;
                T grad = NumOps.Multiply(dHidden[i], input[j]);
                _weightsGrad[idx] = NumOps.Add(_weightsGrad[idx], grad);
            }
        }
    }

    public void ResetGradients()
    {
        for (int i = 0; i < _weightsGrad.Length; i++) _weightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _biasGrad.Length; i++) _biasGrad[i] = NumOps.Zero;
        for (int i = 0; i < _meanWeightsGrad.Length; i++) _meanWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _meanBiasGrad.Length; i++) _meanBiasGrad[i] = NumOps.Zero;
        for (int i = 0; i < _logVarWeightsGrad.Length; i++) _logVarWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _logVarBiasGrad.Length; i++) _logVarBiasGrad[i] = NumOps.Zero;
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
        for (int i = 0; i < tensor.Length; i++)
        {
            T avgGrad = NumOps.Divide(grad[i], batchSize);
            T update = NumOps.Multiply(learningRate, avgGrad);
            tensor[i] = NumOps.Subtract(tensor[i], update);
        }
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
        _weightsGrad = new Tensor<T>(_weights.Shape);
        _biasGrad = new Tensor<T>(_bias.Shape);
        _meanWeightsGrad = new Tensor<T>(_meanWeights.Shape);
        _meanBiasGrad = new Tensor<T>(_meanBias.Shape);
        _logVarWeightsGrad = new Tensor<T>(_logVarWeights.Shape);
        _logVarBiasGrad = new Tensor<T>(_logVarBias.Shape);
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (int dim in tensor.Shape)
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
        // Clamp by tensor length but consume all serialized values to keep stream aligned
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
    public override bool SupportsJitCompilation => true;

    public override void ResetState() { ResetGradients(); _lastLatent = null; _lastHidden = null; }

    public override void UpdateParameters(T learningRate)
    {
        ApplyGradients(learningRate, 1);
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> nodes)
    {
        return Autodiff.TensorOperations<T>.Variable(new Tensor<T>(new[] { _outputSize }), "lstm_decoder_output");
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

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var hidden = _lastHidden ?? new Tensor<T>(new[] { _hiddenSize });
        var latent = _lastLatent ?? new Tensor<T>(new[] { _latentDim });
        return Backward(outputGradient, hidden, latent);
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
        // Expand to hidden: hidden = tanh(W * latent + bias)
        var hidden = new Tensor<T>(new[] { _hiddenSize });
        for (int i = 0; i < _hiddenSize; i++)
        {
            T sum = _bias[i];
            for (int j = 0; j < Math.Min(latent.Length, _latentDim); j++)
            {
                int idx = i * _latentDim + j;
                sum = NumOps.Add(sum, NumOps.Multiply(_weights[idx], latent[j]));
            }
            hidden[i] = MathHelper.Tanh(sum);
        }

        // Decode to output: output = outputWeights * hidden + outputBias
        var output = new Tensor<T>(new[] { _outputSize });
        for (int i = 0; i < _outputSize; i++)
        {
            T sum = _outputBias[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                int idx = i * _hiddenSize + j;
                sum = NumOps.Add(sum, NumOps.Multiply(_outputWeights[idx], hidden[j]));
            }
            output[i] = sum;
        }

        return (output, hidden);
    }

    public Tensor<T> Backward(Tensor<T> dOutput, Tensor<T> hidden, Tensor<T> latent)
    {
        // Gradients for output projection: dOutputWeights = dOutput * hidden^T, dOutputBias = dOutput
        int dOutLen = dOutput.Length;
        int effectiveOutputSize = Math.Min(_outputSize, dOutLen);
        for (int i = 0; i < effectiveOutputSize; i++)
        {
            _outputBiasGrad[i] = NumOps.Add(_outputBiasGrad[i], dOutput[i]);
            for (int j = 0; j < _hiddenSize; j++)
            {
                int idx = i * _hiddenSize + j;
                T grad = NumOps.Multiply(dOutput[i], hidden[j]);
                _outputWeightsGrad[idx] = NumOps.Add(_outputWeightsGrad[idx], grad);
            }
        }

        // Compute gradient w.r.t. hidden: dHidden = outputWeights^T * dOutput
        var dHidden = new Tensor<T>(new[] { _hiddenSize });
        for (int j = 0; j < _hiddenSize; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < effectiveOutputSize; i++)
            {
                int idx = i * _hiddenSize + j;
                sum = NumOps.Add(sum, NumOps.Multiply(_outputWeights[idx], dOutput[i]));
            }
            dHidden[j] = sum;
        }

        // Apply tanh derivative: dHidden * (1 - hidden^2)
        for (int i = 0; i < _hiddenSize; i++)
        {
            T h = hidden[i];
            T tanhDeriv = NumOps.Subtract(NumOps.One, NumOps.Multiply(h, h));
            dHidden[i] = NumOps.Multiply(dHidden[i], tanhDeriv);
        }

        // Gradients for decoder weights: dWeights = dHidden * latent^T, dBias = dHidden
        for (int i = 0; i < _hiddenSize; i++)
        {
            _biasGrad[i] = NumOps.Add(_biasGrad[i], dHidden[i]);
            for (int j = 0; j < Math.Min(latent.Length, _latentDim); j++)
            {
                int idx = i * _latentDim + j;
                T grad = NumOps.Multiply(dHidden[i], latent[j]);
                _weightsGrad[idx] = NumOps.Add(_weightsGrad[idx], grad);
            }
        }

        // Compute gradient w.r.t. latent: dLatent = weights^T * dHidden
        var dLatent = new Tensor<T>(new[] { _latentDim });
        for (int j = 0; j < _latentDim; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < _hiddenSize; i++)
            {
                int idx = i * _latentDim + j;
                sum = NumOps.Add(sum, NumOps.Multiply(_weights[idx], dHidden[i]));
            }
            dLatent[j] = sum;
        }

        return dLatent;
    }

    public void ResetGradients()
    {
        for (int i = 0; i < _weightsGrad.Length; i++) _weightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _biasGrad.Length; i++) _biasGrad[i] = NumOps.Zero;
        for (int i = 0; i < _outputWeightsGrad.Length; i++) _outputWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _outputBiasGrad.Length; i++) _outputBiasGrad[i] = NumOps.Zero;
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
        for (int i = 0; i < tensor.Length; i++)
        {
            T avgGrad = NumOps.Divide(grad[i], batchSize);
            T update = NumOps.Multiply(learningRate, avgGrad);
            tensor[i] = NumOps.Subtract(tensor[i], update);
        }
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
        _weightsGrad = new Tensor<T>(_weights.Shape);
        _biasGrad = new Tensor<T>(_bias.Shape);
        _outputWeightsGrad = new Tensor<T>(_outputWeights.Shape);
        _outputBiasGrad = new Tensor<T>(_outputBias.Shape);
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (int dim in tensor.Shape)
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
        // Clamp by tensor length but consume all serialized values to keep stream aligned
        for (int i = 0; i < length; i++)
        {
            double v = reader.ReadDouble();
            if (i < tensor.Length)
                tensor[i] = NumOps.FromDouble(v);
        }
        return tensor;
    }
}
