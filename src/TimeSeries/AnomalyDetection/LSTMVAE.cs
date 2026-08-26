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
public partial class LSTMVAE<T> : TimeSeriesModelBase<T>
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
    [Buffer]
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
                    using var z = new Tensor<T>(mean._shape);
                    // epsilon and sigma are kept because the reparameterization gradient needs them:
                    // z = mu + sigma * eps means dz/dlogVar = 0.5 * sigma * eps. Regenerating eps
                    // after the forward would differentiate a DIFFERENT sample than the one decoded.
                    using var epsilon = new Tensor<T>(mean._shape);
                    using var sigma = new Tensor<T>(mean._shape);
                    var clampedActive = new bool[mean.Length];
                    {
                        var meanSpan = mean.Data.Span;
                        var lvSpan = logVar.Data.Span;
                        var zSpan = z.Data.Span;
                        T half = _numOps.FromDouble(0.5);

                        // CLAMP THE LOG-VARIANCE BEFORE EXPONENTIATING. exp(0.5 * logVar) is unbounded
                        // above: logVar is a free network output, so nothing stops the encoder from
                        // driving it past the exponent's range during training. At logVar ~ 710 the
                        // double overflows to +Infinity, std becomes Infinity, and z = mean + std * eps
                        // is Infinity or NaN -- which then flows through the decoder into the forecast
                        // (the observed "Out-of-sample forecast contains NaN or Infinity").
                        //
                        // Clamping to [-30, 20] is the standard guard for a diagonal-Gaussian latent:
                        // CompVis latent-diffusion's DiagonalGaussianDistribution applies exactly
                        // torch.clamp(logvar, -30.0, 20.0) before deriving std, and the same bounds are
                        // conventional across VAE implementations. They are deliberately wide rather
                        // than tuned -- exp(20/2) is about 2.2e4 and exp(-30/2) about 3.1e-7, so every
                        // variance a healthy encoder actually produces passes through untouched and
                        // only the divergent tail is bounded. The lower bound matters too: an
                        // unbounded-below logVar underflows std to exactly 0, which silently removes
                        // the sampling noise that makes this a VAE rather than an autoencoder.
                        T logVarFloor = _numOps.FromDouble(-30.0);
                        T logVarCeiling = _numOps.FromDouble(20.0);

                        var epsSpan = epsilon.Data.Span;
                        var sigmaSpan = sigma.Data.Span;

                        for (int j = 0; j < mean.Length; j++)
                        {
                            T clampedLogVar = lvSpan[j];
                            if (_numOps.LessThan(clampedLogVar, logVarFloor)) { clampedLogVar = logVarFloor; clampedActive[j] = true; }
                            else if (_numOps.GreaterThan(clampedLogVar, logVarCeiling)) { clampedLogVar = logVarCeiling; clampedActive[j] = true; }

                            T std = _numOps.Exp(_numOps.Multiply(half, clampedLogVar));
                            T eps = _numOps.FromDouble(random.NextGaussian());
                            sigmaSpan[j] = std;
                            epsSpan[j] = eps;
                            zSpan[j] = _numOps.Add(meanSpan[j], _numOps.Multiply(std, eps));
                        }
                    }

                    // Decode with caching
                    var (reconstruction, decoderHidden) = _decoder.DecodeWithCache(z);

                    // Compute reconstruction error
                    T error = ComputeReconstructionError(input, reconstruction);
                    reconstructionErrors.Add(error);

                    // BACKPROPAGATION OF THE ELBO. This block previously did not exist: the loop
                    // reset the gradient accumulators, ran the forward, and called ApplyGradients on
                    // accumulators nothing had written, so every update was `w -= lr/batch * 0` and
                    // the model never left its initialization. Nothing in the file wrote a gradient.
                    //
                    // The objective is the standard VAE bound (Kingma & Welling, arXiv:1312.6114),
                    // maximised as a minimised loss:
                    //     L = ||x - x_hat||^2 / n  +  beta * KL(q(z|x) || N(0, I))
                    //     KL = -0.5 * sum_j (1 + logVar_j - mu_j^2 - exp(logVar_j))
                    // with the Gaussian-likelihood reconstruction term reducing to MSE, exactly as
                    // ComputeReconstructionError already measures it, and the LSTM-VAE anomaly
                    // formulation of Park et al. (RA-L 2018) scoring by that same reconstruction term.
                    int reconLength = Math.Min(input.Length, reconstruction.Length);
                    T reconScale = _numOps.FromDouble(reconLength > 0 ? 2.0 / reconLength : 0.0);

                    // dL/dx_hat = 2 (x_hat - x) / n
                    using var dOutput = new Tensor<T>(reconstruction._shape);
                    {
                        var dOutSpan = dOutput.Data.Span;
                        var reconSpan = reconstruction.Data.Span;
                        for (int j = 0; j < reconLength; j++)
                        {
                            dOutSpan[j] = _numOps.Multiply(reconScale, _numOps.Subtract(reconSpan[j], input[j]));
                        }
                    }

                    using var dLatent = _decoder.AccumulateGradients(z, decoderHidden, dOutput);

                    // Reparameterization + KL, per latent coordinate.
                    using var dMean = new Tensor<T>(mean._shape);
                    using var dLogVar = new Tensor<T>(logVar._shape);
                    {
                        var dLatentSpan = dLatent.Data.Span;
                        var dMeanSpan = dMean.Data.Span;
                        var dLogVarSpan = dLogVar.Data.Span;
                        var meanSpan = mean.Data.Span;
                        var sigmaSpan = sigma.Data.Span;
                        var epsSpan = epsilon.Data.Span;
                        T half = _numOps.FromDouble(0.5);
                        T beta = _numOps.FromDouble(_options.KLWeight);

                        for (int j = 0; j < mean.Length; j++)
                        {
                            // z = mu + sigma * eps  =>  dz/dmu = 1, dz/dlogVar = 0.5 * sigma * eps.
                            T dz = dLatentSpan[j];
                            T dLogVarFromRecon = _numOps.Multiply(
                                dz, _numOps.Multiply(half, _numOps.Multiply(sigmaSpan[j], epsSpan[j])));

                            // d/dmu KL = mu ; d/dlogVar KL = 0.5 * (exp(logVar) - 1).
                            // sigma was computed from the clamped log-variance, so sigma^2 is the
                            // same bounded exp(clampedLogVar) used by the forward distribution. Using
                            // exp(raw logVar) here defeated the forward clamp and overflowed the
                            // gradient when an encoder coordinate had already diverged.
                            T dMeanFromKL = _numOps.Multiply(beta, meanSpan[j]);
                            T boundedVariance = _numOps.Multiply(sigmaSpan[j], sigmaSpan[j]);
                            T dLogVarFromKL = _numOps.Multiply(
                                beta,
                                _numOps.Multiply(half, _numOps.Subtract(boundedVariance, _numOps.One)));

                            dMeanSpan[j] = _numOps.Add(dz, dMeanFromKL);

                            // A saturated clamp has zero local derivative, so the reconstruction path
                            // contributes nothing there. Keep a bounded KL restoring gradient so a
                            // diverged coordinate can return to range without exp(raw logVar) overflowing.
                            dLogVarSpan[j] = clampedActive[j]
                                ? dLogVarFromKL
                                : _numOps.Add(dLogVarFromRecon, dLogVarFromKL);
                        }
                    }

                    _encoder.AccumulateGradients(input, hidden, dMean, dLogVar);
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
        if (TryPredictFromTimeIndexCalibration(input, _trainingSeries, out var calibratedPredictions))
        {
            return calibratedPredictions;
        }

        int n = input.Rows;
        var predictions = new Vector<T>(n);

        // Reconstruct every row from its own input window (see DeepARModel.Predict: the prior
        // i < _trainingSeries.Length shortcut returned memorized training values for OOS rows).
        for (int i = 0; i < n; i++)
        {
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
// Rank 1 only, and that is the shape the layer is CONSTRUCTED at rather than an assumption:
// `base(new[] { inputSize }, new[] { latentDim * 2 })`. ForwardTraced flattens with
// `input.ToVector()`, so a higher-rank input would not throw - but nothing in this layer states what
// one would mean, and the tensor it emits is rank 1 regardless.
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal partial class LSTMEncoderTensor<T> : NeuralNetworks.Layers.LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// HAND-WRITTEN because the emitted width is not carried through and is not a bare field: the
    /// forward allocates <c>new Tensor&lt;T&gt;(new[] { _latentDim * 2 })</c>, filling the first
    /// <c>_latentDim</c> slots with the mean and the next <c>_latentDim</c> with the log-variance.
    /// That doubling IS the VAE's [mean | logVar] packing, so the output width is set by the latent
    /// size alone and is independent of how wide the input window was.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 1 || _latentDim <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_latentDim * 2)),
        };
    }


    private readonly int _inputSize;
    private readonly int _latentDim;
    private readonly int _hiddenSize;

    // LSTM weights (Tensor-based)
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weights;      // [hiddenSize, inputSize]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _bias;         // [hiddenSize]

    // Mean projection weights
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _meanWeights;  // [latentDim, hiddenSize]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _meanBias;     // [latentDim]

    // Log variance projection weights
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _logVarWeights; // [latentDim, hiddenSize]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _logVarBias;    // [latentDim]

    // Gradient accumulators
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weightsGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _biasGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _meanWeightsGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _meanBiasGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _logVarWeightsGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _logVarBiasGrad;

    public override bool SupportsTraining => true;

    public override void ResetState() { ResetGradients(); }

    public override void UpdateParameters(T learningRate)
    {
        ApplyGradients(learningRate, 1);
    }

    /// <summary>
    /// Forward pass: takes input tensor, runs through LSTM + VAE projections.
    /// Output is [mean | logVar] concatenated (2 * latentDim).
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
        double stddev = Math.Sqrt(2.0 / Math.Max(1, inputSize));

        _weights = InitTensor(new[] { hiddenSize, inputSize }, stddev, random);
        _bias = new Tensor<T>(new[] { hiddenSize });

        stddev = Math.Sqrt(2.0 / Math.Max(1, hiddenSize));
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

    /// <summary>
    /// Backpropagates dL/dmu and dL/dlogVar through the encoder's two projection heads and its
    /// shared hidden layer.
    /// </summary>
    /// <param name="input">x, the example this forward consumed.</param>
    /// <param name="hidden">h = tanh(W x + b), cached from <see cref="EncodeWithCache"/>.</param>
    /// <param name="dMean">dL/dmu for this example, shape [latentDim].</param>
    /// <param name="dLogVar">dL/dlogVar for this example, shape [latentDim].</param>
    /// <remarks>
    /// mu and logVar are two independent affine heads over the SAME hidden vector, so the hidden
    /// gradient is the SUM of both paths: <c>dh = W_muᵀ dmu + W_logVarᵀ dlogVar</c>. Dropping either
    /// term is the classic VAE-backprop error — it silently stops the posterior variance (or the
    /// mean) from shaping the representation. From there the tanh derivative gives
    /// <c>dpre = dh ⊙ (1 - h²)</c>, and <c>dW = dpre ⊗ x</c>, <c>db = dpre</c>.
    /// </remarks>
    public void AccumulateGradients(Vector<T> input, Tensor<T> hidden, Tensor<T> dMean, Tensor<T> dLogVar)
    {
        var hiddenRow = hidden.Reshape(new[] { 1, _hiddenSize });                      // [1,H]
        var dMeanCol = dMean.Reshape(new[] { _latentDim, 1 });                         // [L,1]
        var dLogVarCol = dLogVar.Reshape(new[] { _latentDim, 1 });                     // [L,1]

        // Both projection heads.
        _meanWeightsGrad = Engine.TensorAdd(_meanWeightsGrad, Engine.TensorMatMul(dMeanCol, hiddenRow));
        _meanBiasGrad = Engine.TensorAdd(_meanBiasGrad, dMean);
        _logVarWeightsGrad = Engine.TensorAdd(_logVarWeightsGrad, Engine.TensorMatMul(dLogVarCol, hiddenRow));
        _logVarBiasGrad = Engine.TensorAdd(_logVarBiasGrad, dLogVar);

        // The hidden vector feeds BOTH heads, so its gradient is the sum of the two paths.
        var dHidden = Engine.TensorAdd(
            Engine.TensorMatMul(Engine.TensorTranspose(_meanWeights), dMeanCol).Reshape(new[] { _hiddenSize }),
            Engine.TensorMatMul(Engine.TensorTranspose(_logVarWeights), dLogVarCol).Reshape(new[] { _hiddenSize }));

        using var ones = new Tensor<T>(new[] { _hiddenSize });
        ones.Fill(NumOps.One);
        var tanhDerivative = Engine.TensorSubtract(ones, Engine.TensorMultiply(hidden, hidden));
        var dPre = Engine.TensorMultiply(dHidden, tanhDerivative);                     // [H]

        // Input projection. The forward pads or truncates x to _inputSize, so mirror that here.
        using var inputRow = new Tensor<T>(new[] { 1, _inputSize });
        {
            var span = inputRow.Data.Span;
            int effective = Math.Min(input.Length, _inputSize);
            for (int j = 0; j < effective; j++) span[j] = input[j];
        }

        _weightsGrad = Engine.TensorAdd(
            _weightsGrad, Engine.TensorMatMul(dPre.Reshape(new[] { _hiddenSize, 1 }), inputRow));
        _biasGrad = Engine.TensorAdd(_biasGrad, dPre);
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

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (int dim in tensor._shape)
            writer.Write(dim);
        writer.Write(tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(NumOps.ToDouble(tensor[i]));
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
// Rank 1 only, matching the shapes the layer is CONSTRUCTED with:
// `base(new[] { latentDim }, new[] { outputSize })`.
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal partial class LSTMDecoderTensor<T> : NeuralNetworks.Layers.LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// HAND-WRITTEN because the width is reconstruction size, not the latent size it was handed.
    /// DecodeWithCache copies at most <c>_latentDim</c> values out of whatever it is given
    /// (<c>Math.Min(latent.Length, _latentDim)</c>) and then projects through
    /// <c>_outputWeights</c> [O, H], returning <c>Reshape(new[] { _outputSize })</c> - so the input
    /// width does not reach the output at all.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 1 || _outputSize <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputSize)),
        };
    }


    private readonly int _latentDim;
    private readonly int _outputSize;
    private readonly int _hiddenSize;

    // LSTM weights (Tensor-based)
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weights;      // [hiddenSize, latentDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _bias;         // [hiddenSize]

    // Output projection weights
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputWeights; // [outputSize, hiddenSize]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputBias;    // [outputSize]

    // Gradient accumulators
    private Tensor<T> _weightsGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _biasGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputWeightsGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputBiasGrad;

    [Scratch]
    private Tensor<T>? _lastLatent;
    [Scratch]
    private Tensor<T>? _lastHidden;

    public override bool SupportsTraining => true;

    public override void ResetState() { ResetGradients(); _lastLatent = null; _lastHidden = null; }

    public override void UpdateParameters(T learningRate)
    {
        ApplyGradients(learningRate, 1);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
        double stddev = Math.Sqrt(2.0 / Math.Max(1, latentDim));

        _weights = InitTensor(new[] { hiddenSize, latentDim }, stddev, random);
        _bias = new Tensor<T>(new[] { hiddenSize });

        stddev = Math.Sqrt(2.0 / Math.Max(1, hiddenSize));
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

    /// <summary>
    /// Backpropagates the reconstruction gradient through the decoder and returns dL/dz, the
    /// gradient with respect to the latent sample.
    /// </summary>
    /// <param name="latent">z, the latent sample this forward consumed.</param>
    /// <param name="hidden">g = tanh(W_d z + b_d), cached from <see cref="DecodeWithCache"/>.</param>
    /// <param name="dOutput">dL/dx̂ for this example.</param>
    /// <returns>dL/dz, shape [latentDim].</returns>
    /// <remarks>
    /// The decoder is x̂ = W_o · tanh(W_d z + b_d) + b_o, so the chain is
    /// <c>dW_o = dx̂ ⊗ g</c>, <c>db_o = dx̂</c>, <c>dg = W_oᵀ dx̂</c>,
    /// <c>dpre = dg ⊙ (1 - g²)</c> (the tanh derivative), <c>dW_d = dpre ⊗ z</c>,
    /// <c>db_d = dpre</c>, and <c>dz = W_dᵀ dpre</c>. Gradients ACCUMULATE so a batch sums before
    /// <see cref="ApplyGradients"/> divides by the batch size.
    /// </remarks>
    public Tensor<T> AccumulateGradients(Tensor<T> latent, Tensor<T> hidden, Tensor<T> dOutput)
    {
        var dOutCol = dOutput.Reshape(new[] { _outputSize, 1 });                       // [O,1]
        var hiddenRow = hidden.Reshape(new[] { 1, _hiddenSize });                      // [1,H]

        // Output projection.
        _outputWeightsGrad = Engine.TensorAdd(_outputWeightsGrad, Engine.TensorMatMul(dOutCol, hiddenRow));
        _outputBiasGrad = Engine.TensorAdd(_outputBiasGrad, dOutput);

        // Into the hidden activation, then through tanh: d/dx tanh(x) = 1 - tanh(x)^2.
        var dHidden = Engine.TensorMatMul(Engine.TensorTranspose(_outputWeights), dOutCol)
            .Reshape(new[] { _hiddenSize });                                           // [H]
        using var ones = new Tensor<T>(new[] { _hiddenSize });
        ones.Fill(NumOps.One);
        var tanhDerivative = Engine.TensorSubtract(ones, Engine.TensorMultiply(hidden, hidden));
        var dPre = Engine.TensorMultiply(dHidden, tanhDerivative);                     // [H]

        // Latent projection. DecodeWithCache pads short latent vectors and truncates long ones;
        // backward must use the exact same effective input instead of requiring latent.Length == L.
        using var latentRow = new Tensor<T>(new[] { 1, _latentDim });                  // [1,L]
        int effectiveLatent = Math.Min(latent.Length, _latentDim);
        latent.Data.Span[..effectiveLatent].CopyTo(latentRow.Data.Span);
        var dPreCol = dPre.Reshape(new[] { _hiddenSize, 1 });                          // [H,1]
        _weightsGrad = Engine.TensorAdd(_weightsGrad, Engine.TensorMatMul(dPreCol, latentRow));
        _biasGrad = Engine.TensorAdd(_biasGrad, dPre);

        return Engine.TensorMatMul(Engine.TensorTranspose(_weights), dPreCol)
            .Reshape(new[] { _latentDim });                                            // dL/dz, [L]
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

    public override Vector<T> GetParameterGradients()
    {
        var g = new List<T>();
        foreach (var t in new[] { _weightsGrad, _biasGrad, _outputWeightsGrad, _outputBiasGrad })
            for (int i = 0; i < t.Length; i++) g.Add(t[i]);
        return new Vector<T>(g.ToArray());
    }
}
