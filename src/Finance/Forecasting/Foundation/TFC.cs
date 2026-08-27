using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// TF-C — Time-Frequency Consistency for Self-Supervised Time Series.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TF-C learns time series representations by enforcing consistency between time-domain
/// and frequency-domain representations via contrastive learning, capturing both
/// temporal and spectral patterns. It uses dual CNN encoders with a shared projection head.
/// </para>
/// <para><b>For Beginners:</b> TF-C learns to understand time series by looking at the same
/// data in two ways: as a sequence of values over time, and as a set of frequencies (like
/// breaking a musical chord into individual notes). By training the model to agree on what
/// it sees from both perspectives, it learns robust patterns that work well for downstream
/// tasks like forecasting and classification.</para>
/// <para>
/// <b>Reference:</b> Zhang et al., "Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency", NeurIPS 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TF-C model for self-supervised time series representation learning
/// // Enforces consistency between time-domain and frequency-domain representations
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with dual CNN encoders and contrastive learning
/// var model = new TFC&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TFC&lt;double&gt;(architecture, "tfc.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency", "https://arxiv.org/abs/2206.08496", Year = 2022, Authors = "Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis, Marinka Zitnik")]
public partial class TFC<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _timeInputProjection;
    private readonly List<ILayer<T>> _timeEncoderLayers = [];
    private ILayer<T>? _freqInputProjection;
    private readonly List<ILayer<T>> _freqEncoderLayers = [];
    private ILayer<T>? _projectionHead;
    private ILayer<T>? _forecastHead;

    // Fixed, non-trainable Fourier bases. The reference TF-C implementation uses
    // torch.fft.fft(...).abs(), whose output has exactly the same length as the
    // time-domain input. AiDotNet.Tensors.RFFT requires a power-of-two input and
    // pads other lengths internally, so using it for TF-C's paper-default 200-step
    // window silently produces 129 complex bins (a 256-point FFT) instead of 101.
    // Explicit DFT projections preserve the exact n-point transform while keeping
    // the operation visible to the gradient tape through TensorMatMul.
    [Scratch]
    private Tensor<T>? _dftRealBasis;
    [Scratch]
    private Tensor<T>? _dftImaginaryBasis;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TFCOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _projectionDimension;
    private int _numTimeLayers;
    private int _numFreqLayers;
    private double _dropout;
    private double _contrastiveTemperature;

    // RevIN (reversible instance normalization, Kim et al. 2022) statistics.
    // TFC normalizes each input series before the time/frequency encoders and
    // restores the level on the output so distinct input scales produce distinct
    // forecasts.
    /// <summary>Per-instance mean captured by <see cref="ApplyInstanceNormalization"/>,
    /// consumed by <see cref="DenormalizeForecast"/>. Tensor-shaped [B, 1] so it broadcasts
    /// against the forecast. NULL when no forward has run yet.</summary>
    [Scratch]
    private Tensor<T>? _revinMeanTensor;

    /// <summary>Per-instance standard deviation captured by <see cref="ApplyInstanceNormalization"/>,
    /// consumed by <see cref="DenormalizeForecast"/>. Tensor-shaped [B, 1]. NULL when no forward
    /// has run yet.</summary>
    [Scratch]
    private Tensor<T>? _revinStdTensor;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => 1;
    /// <inheritdoc/>
    public override int PatchSize => 1;
    /// <inheritdoc/>
    public override int Stride => 1;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;
    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;
    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TF-C model using a pretrained ONNX model.
    /// </summary>
    public TFC(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TFCOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TFCOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
    }

    /// <summary>
    /// Creates a TF-C model in native mode for training or fine-tuning.
    /// </summary>
    public TFC(
        NeuralNetworkArchitecture<T> architecture,
        TFCOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TFCOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(TFCOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _projectionDimension = options.ProjectionDimension;
        _numTimeLayers = options.NumTimeLayers;
        _numFreqLayers = options.NumFreqLayers;
        _dropout = options.DropoutRate;
        _contrastiveTemperature = options.ContrastiveTemperature;
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultTFCLayers(
                Architecture, _contextLength, _forecastHorizon, _hiddenDimension,
                _projectionDimension, _numTimeLayers, _numFreqLayers, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        int layersPerBlock = _dropout > 0 ? 3 : 2;

        // Time encoder input projection
        if (idx < Layers.Count)
            _timeInputProjection = Layers[idx++];

        // Time encoder layers
        _timeEncoderLayers.Clear();
        int totalTimeLayers = _numTimeLayers * layersPerBlock;
        for (int i = 0; i < totalTimeLayers && idx < Layers.Count; i++)
            _timeEncoderLayers.Add(Layers[idx++]);

        // Frequency encoder input projection
        if (idx < Layers.Count)
            _freqInputProjection = Layers[idx++];

        // Frequency encoder layers
        _freqEncoderLayers.Clear();
        int totalFreqLayers = _numFreqLayers * layersPerBlock;
        for (int i = 0; i < totalFreqLayers && idx < Layers.Count; i++)
            _freqEncoderLayers.Add(Layers[idx++]);

        // Shared projection head
        if (idx < Layers.Count)
            _projectionHead = Layers[idx++];

        // Forecast head
        if (idx < Layers.Count)
            _forecastHead = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        var loss = LossFunction;

        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();

        // GPU-RESIDENT fast path — compiled fused SGD on the combined supervised +
        // contrastive objective. Safe now that ApplyInstanceNormalization and
        // ComputeFrequencyRepresentation both use traceable Engine ops (ReduceMean /
        // ReduceVariance / TensorSqrt / broadcast for RevIN, TensorMatMul + magnitude
        // for the DFT) — both re-execute on every replay from
        // the current-step persistent slot data instead of freezing at trace time.
        var trainableLayers = Layers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableLayers.Count > 0)
        {
            // Closure-captured contrastive loss: ComputeContrastiveLossTape runs
            // INSIDE the forward closure so it consumes the CURRENT-step persistent
            // input (`inp`), not the outer `input` which would freeze at compile
            // time. Fwd/Loss ordering is guaranteed by the fused-step contract.
            Tensor<T>? capturedContrastive = null;
            Tensor<T> ForwardCombined(Tensor<T> inp)
            {
                capturedContrastive = ComputeContrastiveLossTape(inp);
                return ForwardForTraining(inp);
            }
            Tensor<T> ComputeLossCombined(Tensor<T> pred, Tensor<T> tgt)
            {
                var alignedT = tgt;
                if (pred.Rank > tgt.Rank && pred.Shape[0] == 1 && pred.Length == tgt.Length)
                    pred = Engine.Reshape(pred, tgt._shape);
                else if (tgt.Rank > pred.Rank && tgt.Shape[0] == 1 && tgt.Length == pred.Length)
                    alignedT = Engine.Reshape(tgt, pred._shape);
                var supervised = loss.ComputeTapeLoss(pred, alignedT);
                var contrastive = capturedContrastive
                    ?? throw new InvalidOperationException(
                        "TFC fused step: contrastive loss was not captured by ForwardCombined. " +
                        "This indicates the fused-step framework called the loss closure before " +
                        "the forward closure, which violates its documented Fwd-then-Loss ordering.");
                if (!supervised._shape.SequenceEqual(contrastive._shape)
                    && supervised.Length == contrastive.Length)
                    contrastive = Engine.Reshape(contrastive, supervised._shape);
                return Engine.TensorAdd(supervised, contrastive);
            }
            if (AiDotNet.Training.CompiledTapeTrainingStep<T>.TryStepWithFusedOptimizer(
                    trainableLayers, input, target,
                    forward: ForwardCombined, computeLoss: ComputeLossCombined,
                    optimizerType: AiDotNet.Tensors.Engines.Compilation.OptimizerType.SGD,
                    learningRate: 0.001f, beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f, weightDecay: 0f,
                    out T fusedLoss,
                    onGradients: gradients => PublishParameterGradients(gradients)))
            {
                LastLoss = fusedLoss;
                return;
            }
        }

        // Custom tape step: TFC's loss is supervised forecast + weighted
        // contrastive alignment between the time-domain and frequency-
        // domain encoder outputs. Both terms must be recorded under the
        // same GradientTape so the optimizer update reflects the full
        // objective.
        using var tape = new GradientTape<T>();

        // Supervised branch (reuse the forecast head's output).
        var forecast = ForwardForTraining(input);
        var alignedTarget = target;
        if (forecast.Rank > target.Rank && forecast.Shape[0] == 1 && forecast.Length == target.Length)
            forecast = Engine.Reshape(forecast, target._shape);
        else if (target.Rank > forecast.Rank && target.Shape[0] == 1 && target.Length == forecast.Length)
            alignedTarget = Engine.Reshape(target, forecast._shape);
        var supervisedLoss = loss.ComputeTapeLoss(forecast, alignedTarget);

        // Contrastive branch — separate forward through time+freq encoders
        // (tape-aware; see ComputeContrastiveLossTape below). Weight it
        // with _contrastiveTemperature-based scaling applied inside the
        // helper, so this stays a simple additive combination.
        var contrastiveLoss = ComputeContrastiveLossTape(input);

        // Total = supervised + contrastive. Using TensorAdd keeps both
        // losses on the same tape so tape.ComputeGradients(total, ...)
        // accumulates gradients from both terms into each shared
        // parameter (the projection head is shared, so its gradient is
        // the sum of contributions from both branches).
        // Shape-align on rank drift (e.g. supervisedLoss rank-0 [] vs
        // contrastiveLoss rank-1 [1]) so the engine's strict-shape add
        // accepts the pair. Both are scalar-valued so reshape is safe.
        if (!supervisedLoss._shape.SequenceEqual(contrastiveLoss._shape)
            && supervisedLoss.Length == contrastiveLoss.Length)
        {
            contrastiveLoss = Engine.Reshape(contrastiveLoss, supervisedLoss._shape);
        }
        var totalLoss = Engine.TensorAdd(supervisedLoss, contrastiveLoss);

        var allGrads = ComputeAndPublishParameterGradients(tape, totalLoss, sources: null);
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var param in trainableParams)
        {
            if (allGrads.TryGetValue(param, out var grad))
                grads[param] = grad;
        }

        T lossValue = totalLoss.Length > 0 ? totalLoss[0] : NumOps.Zero;
        LastLoss = lossValue;

        // Apply gradients via the registered optimizer. Mirrors the
        // simple SGD-style update path used in TapeTrainingStep so that
        // non-Adam optimizers still get the learning-rate-scaled
        // gradient descent semantics when a full IGradientBasedOptimizer
        // isn't wired up for Finance models yet.
        T lr = NumOps.FromDouble(0.001);
        foreach (var param in trainableParams)
        {
            if (grads.TryGetValue(param, out var grad))
            {
                var update = Engine.TensorMultiplyScalar(grad, lr);
                Engine.TensorSubtractInPlace(param, update);
            }
        }
    }

    /// <summary>
    /// Tape-aware version of <see cref="ComputeContrastiveLoss"/> that
    /// returns a <see cref="Tensor{T}"/> (not a <c>T</c> scalar) so the
    /// gradient tape records every op between the encoder outputs and
    /// the final loss. The old scalar version round-tripped through
    /// <c>double</c> at the last step, which made it invisible to
    /// backward.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Computes <c>-log(sigmoid(cos(time, freq) / T))</c> via the
    /// numerically stable <c>softplus(-logit)</c> identity. All ops go
    /// through <see cref="IEngine"/> so the tape can walk back through
    /// the time encoder, frequency encoder, projection head, and
    /// ultimately the input embeddings.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeContrastiveLossTape(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        var timeCurrent = normalized;
        if (timeCurrent.Rank == 1)
            timeCurrent = Engine.Reshape(timeCurrent, new[] { 1, timeCurrent.Length });

        // Time encoder.
        if (_timeInputProjection is not null)
            timeCurrent = _timeInputProjection.Forward(timeCurrent);
        foreach (var layer in _timeEncoderLayers)
            timeCurrent = layer.Forward(timeCurrent);

        // Frequency encoder.
        var freqInput = ComputeFrequencyRepresentation(normalized);
        if (freqInput.Rank == 1)
            freqInput = Engine.Reshape(freqInput, new[] { 1, freqInput.Length });
        var freqCurrent = freqInput;
        if (_freqInputProjection is not null)
            freqCurrent = _freqInputProjection.Forward(freqCurrent);
        foreach (var layer in _freqEncoderLayers)
            freqCurrent = layer.Forward(freqCurrent);

        // Shared projection head.
        Tensor<T> timeProj = timeCurrent, freqProj = freqCurrent;
        if (_projectionHead is not null)
        {
            timeProj = _projectionHead.Forward(timeCurrent);
            freqProj = _projectionHead.Forward(freqCurrent);
        }

        // Broadcast freqProj to timeProj shape if they differ (e.g., a
        // frequency encoder that drops the final length dim). Using
        // Engine.Reshape keeps the tape intact. Compare by converting
        // _shape to arrays so the Linq SequenceEqual binds to
        // IEnumerable<int> without the ReadOnlySpan inference ambiguity.
        var timeShape = timeProj._shape;
        var freqShape = freqProj._shape;
        if (!timeShape.AsEnumerable().SequenceEqual(freqShape))
        {
            if (timeProj.Length == freqProj.Length)
                freqProj = Engine.Reshape(freqProj, timeProj._shape);
            else
                throw new InvalidOperationException(
                    $"TFC contrastive loss: time/freq projections have incompatible shapes " +
                    $"({string.Join("x", timeProj.Shape.ToArray())} vs " +
                    $"{string.Join("x", freqProj.Shape.ToArray())}).");
        }

        // Cosine similarity via tape-aware ops:
        //   dot = sum(a * b)  (scalar tensor via ReduceSum across all axes)
        //   |a| = sqrt(sum(a^2)),  |b| = sqrt(sum(b^2))
        //   cos = dot / (|a| * |b| + eps)
        var allAxes = Enumerable.Range(0, timeProj.Rank).ToArray();

        var dotElements = Engine.TensorMultiply(timeProj, freqProj);
        var dotProduct = Engine.ReduceSum(dotElements, allAxes, keepDims: false);

        var timeSq = Engine.TensorMultiply(timeProj, timeProj);
        var freqSq = Engine.TensorMultiply(freqProj, freqProj);
        var timeNormSq = Engine.ReduceSum(timeSq, allAxes, keepDims: false);
        var freqNormSq = Engine.ReduceSum(freqSq, allAxes, keepDims: false);
        var timeNorm = Engine.TensorSqrt(timeNormSq);
        var freqNorm = Engine.TensorSqrt(freqNormSq);
        var normProduct = Engine.TensorMultiply(timeNorm, freqNorm);
        var normProductSafe = Engine.TensorAddScalar(normProduct, NumOps.FromDouble(1e-8));
        var cosSim = Engine.TensorDivide(dotProduct, normProductSafe);

        // logit = cos / temperature; softplus(-logit) = log(1 + exp(-logit)).
        T tempT = NumOps.FromDouble(Math.Max(1e-8, _contrastiveTemperature));
        var logit = Engine.TensorMultiplyScalar(cosSim, NumOps.Divide(NumOps.One, tempT));
        var negLogit = Engine.TensorNegate(logit);
        return Engine.Softplus(negLogit);
    }

    /// <summary>
    /// Contrastive loss between time and frequency encoder outputs.
    /// Computes a positive-pair similarity loss (-log sigmoid) between time-domain and
    /// frequency-domain representations. This is a single-sample approximation of InfoNCE;
    /// full InfoNCE requires a batch of negatives which will be supported when batch training is added.
    /// </summary>
    private T ComputeContrastiveLoss(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        var timeCurrent = normalized;
        if (timeCurrent.Rank == 1) timeCurrent = timeCurrent.Reshape(new[] { 1, timeCurrent.Length });

        // Time encoder
        if (_timeInputProjection is not null) timeCurrent = _timeInputProjection.Forward(timeCurrent);
        foreach (var layer in _timeEncoderLayers) timeCurrent = layer.Forward(timeCurrent);

        // Frequency encoder
        var freqInput = ComputeFrequencyRepresentation(normalized);
        if (freqInput.Rank == 1) freqInput = freqInput.Reshape(new[] { 1, freqInput.Length });
        var freqCurrent = freqInput;
        if (_freqInputProjection is not null) freqCurrent = _freqInputProjection.Forward(freqCurrent);
        foreach (var layer in _freqEncoderLayers) freqCurrent = layer.Forward(freqCurrent);

        // Project both to shared space
        Tensor<T> timeProj = timeCurrent, freqProj = freqCurrent;
        if (_projectionHead is not null)
        {
            timeProj = _projectionHead.Forward(timeCurrent);
            freqProj = _projectionHead.Forward(freqCurrent);
        }

        // Cosine similarity / temperature
        // Engine-accelerated cosine similarity
        int projLen = Math.Min(timeProj.Length, freqProj.Length);
        var tpVec = new Vector<T>(projLen);
        var fpVec = new Vector<T>(projLen);
        for (int i = 0; i < projLen; i++) { tpVec[i] = timeProj[i]; fpVec[i] = freqProj[i]; }
        T dotProduct = Engine.DotProduct(tpVec, fpVec);
        T normTime = Engine.DotProduct(tpVec, tpVec);
        T normFreq = Engine.DotProduct(fpVec, fpVec);
        T eps8 = NumOps.FromDouble(1e-8);
        T normProduct = NumOps.Add(NumOps.Multiply(NumOps.Sqrt(normTime), NumOps.Sqrt(normFreq)), eps8);
        T cosSim = NumOps.Divide(dotProduct, normProduct);
        T tempT = NumOps.FromDouble(Math.Max(1e-8, _contrastiveTemperature));
        T logit = NumOps.Divide(cosSim, tempT);

        // -log(sigmoid(logit)) for positive pair — use log-sum-exp for numerical stability
        // -log(sigmoid(x)) = log(1 + exp(-x))
        double logitD = NumOps.ToDouble(logit);
        double loss = Math.Log(1.0 + Math.Exp(-logitD));
        return NumOps.FromDouble(loss);
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TFC" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "ProjectionDimension", _projectionDimension },
                { "NumTimeLayers", _numTimeLayers },
                { "NumFreqLayers", _numFreqLayers },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("TFC does not support quantile forecasting. Pass null for point forecasts.");

        return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;
        int stepsRemaining = steps;

        while (stepsRemaining > 0)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);
            int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining);
            stepsRemaining -= stepsUsed;

            if (stepsRemaining > 0)
                currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed);
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
            count++;
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = NumOps.Sqrt(mse);
        return metrics;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Traceable RevIN forward (Kim et al. 2022). Uses <see cref="IEngine.ReduceMean{T}"/>,
    /// <see cref="IEngine.ReduceVariance{T}"/>, <see cref="IEngine.TensorSqrt{T}"/>, and
    /// <see cref="IEngine.TensorDivide{T}"/> so every op records on the tape and
    /// re-executes under the compiled fused plan. The per-instance mean/std are captured
    /// in tensor fields (not <see cref="Vector{T}"/> scalars) so <see cref="DenormalizeForecast"/>
    /// stays on-tape too — an inference call refreshes the tensors and a compiled-plan
    /// replay recomputes both on-device from the current slot data.
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        var (normalized, mean, std) = NormalizeWithStats(input);
        _revinMeanTensor = mean;
        _revinStdTensor = std;
        return normalized;
    }

    /// <summary>
    /// Stateless RevIN forward. Returns (normalized, mean, std) with mean/std as tensors
    /// shaped [B, 1] so downstream ops can broadcast them back. All ops go through
    /// <see cref="Engine"/> — no <c>.Data.Span</c> host loops — so the whole computation
    /// records on the autodiff tape and re-executes on every replay under a compiled plan.
    /// </summary>
    private (Tensor<T> Normalized, Tensor<T> Mean, Tensor<T> Std) NormalizeWithStats(Tensor<T> input)
    {
        int batchSize = input.Shape.Length > 1 ? input.Shape[0] : 1;
        int instanceSize = batchSize > 0 ? input.Length / batchSize : input.Length;
        if (instanceSize <= 0)
        {
            // Degenerate input — return input unchanged with identity mean/std.
            var meanIdentity = new Tensor<T>(new[] { 1, 1 });
            var stdIdentity = new Tensor<T>(new[] { 1, 1 });
            Engine.TensorFill(meanIdentity, NumOps.Zero);
            Engine.TensorFill(stdIdentity, NumOps.One);
            return (input, meanIdentity, stdIdentity);
        }

        bool reshaped = input.Rank != 2;
        var flat = reshaped ? Engine.Reshape(input, new[] { batchSize, instanceSize }) : input;

        // mean over the instance axis, keepDims so shape stays [B, 1] for broadcast.
        var mean = Engine.ReduceMean(flat, new[] { 1 }, keepDims: true);
        var variance = Engine.ReduceVariance(flat, new[] { 1 }, keepDims: true);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-5)));

        // (x - mean) / std via BroadcastSubtract + BroadcastDivide.
        var centered = Engine.TensorSubtract(flat, mean);
        var normalized = Engine.TensorDivide(centered, std);

        if (reshaped)
            normalized = Engine.Reshape(normalized, input._shape);
        return (normalized, mean, std);
    }

    /// <summary>
    /// RevIN reverse step (Kim et al. 2022): restores each instance's mean/std to the
    /// forecast so it is expressed on the input's original scale. All ops go through
    /// <see cref="Engine"/> so the forecast stays on the autodiff tape AND the compiled-plan
    /// replay uses the CURRENT input's stats (via <see cref="_revinMeanTensor"/> /
    /// <see cref="_revinStdTensor"/>, both refreshed by <see cref="ApplyInstanceNormalization"/>
    /// or <see cref="NormalizeWithStats"/> earlier in the same forward).
    /// </summary>
    private Tensor<T> DenormalizeForecast(Tensor<T> forecast)
    {
        return DenormalizeForecastWithStats(forecast, _revinMeanTensor, _revinStdTensor);
    }

    /// <summary>
    /// Stateless RevIN inverse. Takes explicit mean/std tensors so the compiled-plan
    /// path can thread the CURRENT-step stats through without touching class fields.
    /// </summary>
    private Tensor<T> DenormalizeForecastWithStats(Tensor<T> forecast, Tensor<T>? mean, Tensor<T>? std)
    {
        if (mean is null || std is null) return forecast;

        int batch = forecast.Shape.Length > 1 ? forecast.Shape[0] : 1;
        if (mean.Length != batch || std.Length != batch || forecast.Length % batch != 0)
            return forecast;

        bool reshaped = forecast.Rank != 2;
        var work = reshaped ? Engine.Reshape(forecast, new[] { batch, forecast.Length / batch }) : forecast;
        var scaled = Engine.TensorMultiply(work, std);
        var shifted = Engine.TensorAdd(scaled, mean);
        return reshaped ? Engine.Reshape(shifted, forecast._shape) : shifted;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["ProjectionDimension"] = NumOps.FromDouble(_projectionDimension),
            ["NumTimeLayers"] = NumOps.FromDouble(_numTimeLayers),
            ["NumFreqLayers"] = NumOps.FromDouble(_numFreqLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        // Thread mean/std as tensors through the same forward — under the compiled
        // fused plan, class-field capture at trace time would freeze the trace-batch
        // stats; tensor-threaded stats re-execute on every replay. The abstract
        // override caches them for external ApplyInstanceNormalization callers.
        var (normalized, mean, std) = NormalizeWithStats(input);
        _revinMeanTensor = mean;
        _revinStdTensor = std;
        var current = normalized;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // Time-domain encoder path
        if (_timeInputProjection is not null)
            current = _timeInputProjection.Forward(current);

        foreach (var layer in _timeEncoderLayers)
            current = layer.Forward(current);

        // Frequency-domain path: compute DFT magnitude spectrum as input
        var freqInput = ComputeFrequencyRepresentation(normalized);
        if (freqInput.Rank == 1)
            freqInput = freqInput.Reshape(new[] { 1, freqInput.Length });

        var freqCurrent = freqInput;
        if (_freqInputProjection is not null)
            freqCurrent = _freqInputProjection.Forward(freqCurrent);

        foreach (var layer in _freqEncoderLayers)
            freqCurrent = layer.Forward(freqCurrent);

        // Average time and frequency representations (contrastive fusion).
        // Must go through Engine ops so the gradient tape records the
        // combine — if we did a .Data.Span loop here, base.Train would
        // call Forward under a GradientTape and the freq encoder would
        // never see gradients because the tape can't see the assignment.
        // A shape mismatch between the two branches (e.g., freq encoder
        // output length ≠ time encoder output) gets reshaped through
        // Engine.Reshape so the tape still records it.
        if (!current._shape.AsEnumerable().SequenceEqual(freqCurrent._shape))
            freqCurrent = Engine.Reshape(freqCurrent, current._shape);
        current = Engine.TensorAdd(current, freqCurrent);
        current = Engine.TensorMultiplyScalar(current, NumOps.FromDouble(0.5));

        if (_projectionHead is not null)
            current = _projectionHead.Forward(current);

        if (_forecastHead is not null)
            current = _forecastHead.Forward(current);

        // RevIN reverse: restore the input's per-instance level/scale so distinct
        // input levels yield distinct forecasts (the encoders see only the
        // mean/std-normalized series). Pass mean/std as tensor locals so the
        // compiled-plan replay picks up the CURRENT-step stats, not the trace pass.
        current = DenormalizeForecastWithStats(current, mean, std);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = Engine.Reshape(current, new[] { current.Shape[1] });

        return current;
    }

    /// <summary>
    /// Training-mode forward. Routes through <see cref="ForwardNative"/> so training
    /// uses the same RevIN normalize/denormalize as inference and keeps training mode
    /// (dropout) active, instead of the base default that flips to inference.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return ForwardNative(input);
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        int features = input.Rank > 2 ? input.Shape[2] : 1;

        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++)
            inputData[i] = (float)NumOps.ToDouble(input[i]);

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData, new[] { batchSize, seqLen, features });

        string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input";
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);

        int totalElements = 1;
        foreach (var dim in outputShape) totalElements *= dim;

        for (int i = 0; i < totalElements && i < output.Length; i++)
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return output;
    }

    #endregion

    #region Frequency Transform

    /// <summary>
    /// Traceable full-length DFT magnitude spectrum. The TF-C reference implementation
    /// constructs its frequency view with <c>torch.fft.fft(x).abs()</c>, which returns
    /// one magnitude per input time step. Real and imaginary DFT bases are applied with
    /// <see cref="IEngine.TensorMatMul{T}"/>, followed by tape-aware magnitude operations.
    /// This also supports non-power-of-two windows exactly; <see cref="IEngine.RFFT{T}"/>
    /// pads such windows and therefore does not implement TF-C's required n-point FFT.
    /// </summary>
    /// <remarks>
    /// Output shape is <c>[..., n]</c>, matching both the time-domain view and the
    /// frequency encoder's configured context width. Magnitudes retain the existing
    /// <c>1/n</c> normalization so changing the transform implementation does not alter
    /// the model's established activation scale.
    /// </remarks>
    private Tensor<T> ComputeFrequencyRepresentation(Tensor<T> input)
    {
        // TF-C computes the DFT magnitude spectrum over the CONTEXT (time) axis.
        // The model is univariate (NumFeatures == 1), so every sample contributes
        // exactly _contextLength contiguous time values however it is shaped:
        // [context], [batch, context], or the framework's canonical time-series
        // layout [batch, context, 1]. Reading the transform length from the last
        // axis broke that canonical layout — it saw the trailing feature axis (1)
        // as the context and rejected the input with "expects a N-step context but
        // received 1 steps on the last axis". Derive the length from the configured
        // context instead and treat the flattened input as numSamples x context.
        int n = _contextLength;
        if (input.Length % n != 0)
        {
            throw new ArgumentException(
                $"TFC is univariate (NumFeatures = {NumFeatures}), so each sample must contain " +
                $"exactly {_contextLength} time steps, but the input holds {input.Length} elements, " +
                $"which is not a whole number of {_contextLength}-step samples.",
                nameof(input));
        }

        int numSamples = input.Length / n;
        // Only skip the reshape when the input is already exactly [numSamples, n];
        // any other rank or shape (including [batch, context, 1]) is normalized.
        bool reshaped = !(input.Rank == 2 && input.Shape[0] == numSamples && input.Shape[1] == n);
        var flat = reshaped ? Engine.Reshape(input, new[] { numSamples, n }) : input;

        var (dftReal, dftImaginary) = EnsureDftBases(n);
        var real = Engine.TensorMatMul(flat, dftReal);
        var imaginary = Engine.TensorMatMul(flat, dftImaginary);
        var realSquared = Engine.TensorMultiply(real, real);
        var imaginarySquared = Engine.TensorMultiply(imaginary, imaginary);
        var magnitude = Engine.TensorSqrt(Engine.TensorAdd(realSquared, imaginarySquared));
        var normalized = Engine.TensorMultiplyScalar(
            magnitude, NumOps.Divide(NumOps.One, NumOps.FromDouble(n)));

        return reshaped ? Engine.Reshape(normalized, input._shape) : normalized;
    }

    /// <summary>
    /// Creates the constant real and imaginary n-point DFT projection matrices.
    /// Matrix row <c>j</c>, column <c>k</c> contains the contribution from input
    /// sample <c>j</c> to frequency bin <c>k</c>.
    /// </summary>
    /// <returns>
    /// The real and imaginary basis matrices. Returned rather than left for the caller to read off
    /// the nullable fields, so the postcondition "both are non-null after this call" is expressed in
    /// the signature instead of asserted with a suppression at every use.
    /// </returns>
    private (Tensor<T> Real, Tensor<T> Imaginary) EnsureDftBases(int n)
    {
        if (_dftRealBasis is not null && _dftImaginaryBasis is not null &&
            _dftRealBasis.Shape[0] == n && _dftRealBasis.Shape[1] == n)
        {
            return (_dftRealBasis, _dftImaginaryBasis);
        }

        var real = new Tensor<T>(new[] { n, n });
        var imaginary = new Tensor<T>(new[] { n, n });
        double scale = 2.0 * Math.PI / n;
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                double angle = scale * j * k;
                real[j, k] = NumOps.FromDouble(Math.Cos(angle));
                imaginary[j, k] = NumOps.FromDouble(-Math.Sin(angle));
            }
        }

        _dftRealBasis = real;
        _dftImaginaryBasis = imaginary;
        return (real, imaginary);
    }

    #endregion

    #region Parameter Estimation

    private new int GetParameterCount()
    {
        // Time encoder
        long total = (long)_contextLength * _hiddenDimension + _hiddenDimension;
        long perTimeLayer = 2L * _hiddenDimension * _hiddenDimension + 2 * _hiddenDimension;
        total += perTimeLayer * _numTimeLayers;

        // Frequency encoder (same size)
        total += (long)_contextLength * _hiddenDimension + _hiddenDimension;
        long perFreqLayer = 2L * _hiddenDimension * _hiddenDimension + 2 * _hiddenDimension;
        total += perFreqLayer * _numFreqLayers;

        // Projection head
        total += (long)_hiddenDimension * _projectionDimension + _projectionDimension;

        // Forecast head
        total += (long)_projectionDimension * _forecastHorizon + _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
