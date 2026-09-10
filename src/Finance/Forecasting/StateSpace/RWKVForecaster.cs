using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.StateSpace;

/// <summary>
/// RWKV (Receptance Weighted Key Value) implementation for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// RWKV combines the efficient parallelizable training of Transformers with the efficient
/// inference of RNNs, achieving linear complexity for both training and inference.
/// This forecasting variant uses stacked RWKV layers for time series prediction.
/// </para>
/// <para><b>For Beginners:</b> RWKV is a unique architecture for sequence modeling:
///
/// <b>The Key Innovation:</b>
/// RWKV can be computed two ways:
/// 1. As a parallel operation during training (like a Transformer) — fast on GPUs
/// 2. As a recurrence during inference (like an RNN) — constant memory per token
///
/// <b>Architecture Components:</b>
/// - Time mixing: WKV (Weighted Key Value) attention with learned exponential decay
/// - Channel mixing: Feed-forward network with gating mechanism
/// - Token shift: Efficiently mixes current and previous token information
///
/// <b>For Time Series:</b>
/// - Linear complexity enables processing very long historical windows
/// - Constant memory during autoregressive forecasting
/// - Multi-head structure captures diverse temporal patterns
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
/// var model = new RWKVForecaster&lt;double&gt;(architecture);
/// var onnxModel = new RWKVForecaster&lt;double&gt;(architecture, "rwkv.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RWKV: Reinventing RNNs for the Transformer Era", "https://arxiv.org/abs/2305.13048", Year = 2023, Authors = "Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho")]
public partial class RWKVForecaster<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private bool _useNativeMode;
    #endregion

    #region Native Mode Fields
    private DenseLayer<T>? _inputEmbedding;
    /// <summary>The RWKV-7 stack. One layer owning N blocks, because the value residual is cross-layer.</summary>
    private Rwkv7Stack<T>? _rwkvStack;
    private List<DenseLayer<T>>? _outputProjectionLayers;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly RWKVForecastingOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _lastForwardSeqLen;
    private int _lastForwardBatchSize;
    private int _forecastHorizon;
    private int _modelDimension;
    private int _numHeads;
    private int _numLayers;
    private double _dropout;
    private int _numFeatures;

    // RevIN (reversible instance normalization, Kim et al. 2022) statistics.
    // RWKVForecaster normalizes each input series before the embedding and
    // restores the level on the output so distinct input scales produce
    // distinct forecasts.
    [Scratch]
    private Vector<T> _revinMean = new Vector<T>(0);
    [Scratch]
    private Vector<T> _revinStd = new Vector<T>(0);
    #endregion

    #region IForecastingModel Properties
    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;
    /// <inheritdoc/>
    public override int PatchSize => 1;
    /// <inheritdoc/>
    public override int Stride => 1;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;
    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>Gets the number of RWKV heads.</summary>
    public int NumHeads => _numHeads;
    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    public RWKVForecaster(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        RWKVForecastingOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new RWKVForecastingOptions<T>();
        _options = options;
        Options = _options;
        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        ApplyOptions(options);
        _numFeatures = 1;
        InitializeLayers();

        // Same ordering requirement as the native constructor: a stateful optimizer sizes its
        // per-parameter state from the model it is given, so it must not be constructed while the
        // model still has zero layers. See the native ctor for the measured failure signature.
        _optimizer = optimizer ?? CreateDefaultOptimizer(options);
    }

    /// <summary>
    /// Initializes a new instance in native mode for training.
    /// </summary>
    public RWKVForecaster(
        NeuralNetworkArchitecture<T> architecture,
        RWKVForecastingOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new RWKVForecastingOptions<T>();
        _options = options;
        Options = _options;
        _useNativeMode = true;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        ApplyOptions(options);
        _numFeatures = numFeatures;
        InitializeLayers();

        // Optimizer construction MUST come after InitializeLayers(). CreateDefaultOptimizer passes
        // `this` to the optimizer, and a stateful optimizer sizes its per-parameter state (Adam's
        // moments, SGD's momentum) from the model it is handed. Built before InitializeLayers() the
        // model still has ZERO layers and ZERO parameters, so that state was allocated against an
        // empty parameter set and its in-place writes then landed against a 17,576-element
        // ParameterBuffer they were never sized for.
        //
        // Measured signature of that mismatch on the FP32 fixture at batch 1: gradients finite at the
        // clipping check, parameters finite before the step, and after ONE step the ENTIRE buffer
        // ([0..17575]) non-finite, on 11-16 of 25 fresh draws. Stateless GradientDescentOptimizer was
        // clean 25/25 precisely because it has no per-parameter state to misalign, while Adam (15/25)
        // and SGD (16/25) both failed.
        _optimizer = optimizer ?? CreateDefaultOptimizer(options);
    }

    private void ApplyOptions(RWKVForecastingOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _modelDimension = options.ModelDimension;
        _numHeads = options.NumHeads;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer(
        RWKVForecastingOptions<T> options)
    {
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = options.LearningRate,
                Beta1 = options.AdamBeta1,
                Beta2 = options.AdamBeta2,
                Epsilon = options.AdamEpsilon,
                UseAdaptiveBetas = false,
                UseAMSGrad = false
            });
    }

    #endregion

    /// <summary>
    /// RWKV's time-mixing is a RECURRENT per-timestep update, so its forward is not a static operation
    /// graph that can be compiled once and replayed safely.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every sibling in this family already opts out — HiPPO in this same folder, plus
    /// GatedDeltaNet, GLA, Griffin and Hawk — for exactly this reason. RWKV was the outlier, silently
    /// inheriting the base default of <c>true</c>.
    /// </para>
    /// <para>
    /// The measured symptom was ForwardPass_ShouldBeFinite_AfterTraining and
    /// Clone_AfterTraining_ShouldPreserveLearnedWeights failing on roughly half of all runs (5 of 10
    /// draws locally), with the whole parameter buffer going NaN after a single training step. A replayed
    /// plan over a recurrent forward reads trace-time state that is no longer valid, which is
    /// order-dependent and therefore intermittent rather than deterministic.
    /// </para>
    /// <para>
    /// This also explains two observations that had made the fault look like a memory-allocator problem.
    /// It reproduced ONLY with stateful optimizers because the fused path requires
    /// <c>IFusedOptimizerSpec</c>, which plain gradient descent does not implement — so compiled training
    /// never engaged for GD. And it survived a zero learning rate because a NaN arriving through the
    /// graph is still NaN after being multiplied by zero.
    /// </para>
    /// </remarks>
    protected override bool SupportsFusedCompiledTraining => false;

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultRWKVForecastingLayers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _modelDimension, _numHeads, _numLayers, _dropout,
                _options.GlobalIclrMultiplier));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _rwkvStack = Layers.OfType<Rwkv7Stack<T>>().FirstOrDefault();
        _outputProjectionLayers = Layers.OfType<DenseLayer<T>>().Skip(1).ToList();
    }

    /// <inheritdoc/>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.OfType<RWKV7Block<T>>().Count() < 1)
            throw new ArgumentException("RWKV Forecaster requires at least one RWKV7Block.");

        if (layers.OfType<DenseLayer<T>>().Count() < 2)
            throw new ArgumentException("RWKV Forecaster requires at least input embedding and output projection DenseLayer layers.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        // Issue #1166: the old body computed a loss + gradient and then
        // called _optimizer.UpdateParameters(Layers) without a backward
        // pass, so every layer's UpdateParameters threw "Backward pass
        // must be called before updating parameters." Delegate to
        // FinancialModelBase.Train — it routes through the tape-based
        // NeuralNetworkBase.TrainWithTape flow (GradientTape forward +
        // tape.ComputeGradients + optimizer.Step) that every other
        // NeuralNetworkBase subclass uses.
        base.Train(input, target);
    }

    /// <summary>
    /// One-shot guard for <see cref="ResolveLazyLayerShapes"/>, mirroring the Wav2Vec2 precedent.
    /// </summary>
    private bool _shapesProbed;

    /// <inheritdoc/>
    /// <remarks>
    /// The real <see cref="Forward"/> reshapes [batch, seqLen, numFeatures] to
    /// [batch*seqLen, numFeatures] before the input embedding, so that embedding must resolve to
    /// inputSize = numFeatures (1 for a univariate series). The base sequential walk instead feeds
    /// the architecture's FLAT input shape ([contextLength]) straight in and resolves the embedding
    /// to contextLength -> 131,328 parameters instead of 512. The first real forward then silently
    /// REBUILDS the layer at the correct shape, so ParameterCount and GetParameters().Length CHANGE
    /// across a training step (measured 5,985,632 -> 5,854,816), which in turn makes every
    /// parameter-vector-sized consumer - optimizer moment state, Clone/serialization round-trips,
    /// and the param-L2 invariants - disagree with the model depending on whether anything queried
    /// parameters before the first forward. Probe the real forward once so every lazy layer resolves
    /// to what Forward actually feeds it. This is virtual for exactly this case (non-sequential
    /// forward topology) and is one-shot, so lazy initialization keeps its performance benefit.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_shapesProbed || !_useNativeMode || Layers.Count == 0) return;
        _shapesProbed = true;
        _ = Forward(new Tensor<T>(new[] { _contextLength, _numFeatures }));
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
                { "NetworkType", "RWKV" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "ModelDimension", _modelDimension },
                { "NumHeads", _numHeads },
                { "NumLayers", _numLayers },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelDataProvider = () => _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);
        if (quantiles is not null && quantiles.Length > 0)
            return GenerateQuantilePredictions(historicalData, quantiles);
        return output;
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
        T mse = NumOps.Zero, mae = NumOps.Zero;
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
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
        // RevIN forward (Kim et al. 2022) over each instance: every non-batch element of a row is
        // normalized together. Delegates to the shared tape-tracked helper -- the previous hand-rolled
        // loop accumulated the statistics with scalar NumOps arithmetic and wrote the output through
        // result.Data.Span[...], which the tape cannot see, so the normalized input came back as a
        // LEAF and nothing could differentiate through the normalization.
    {
        if (!_options.UseReversibleNormalization)
        {
            _revinMean = new Vector<T>(0);
            _revinStd = new Vector<T>(0);
            return input;
        }

        return NormalizeInstanceOnTape(input, _options.RevInEpsilon, out _revinMean, out _revinStd);
    }

    /// <summary>
    /// RevIN reverse step (Kim et al. 2022): restores each instance's mean/std to the
    /// forecast so it is expressed on the input's original scale. The multiply/add go
    /// through the Engine so the forecast stays on the autodiff tape.
    /// </summary>
    private Tensor<T> DenormalizeForecast(Tensor<T> forecast)
    {
        int batch = forecast.Shape.Length > 1 ? forecast.Shape[0] : 1;
        if (_revinMean.Length != batch || forecast.Length % batch != 0)
            return forecast;

        var meanT = new Tensor<T>(new[] { batch, 1 });
        var stdT = new Tensor<T>(new[] { batch, 1 });
        for (int b = 0; b < batch; b++)
        {
            meanT.Data.Span[b] = _revinMean[b];
            stdT.Data.Span[b] = _revinStd[b];
        }

        bool reshaped = forecast.Rank != 2;
        var work = reshaped ? Engine.Reshape(forecast, new[] { batch, forecast.Length / batch }) : forecast;
        // Implicit broadcasting: the explicit TensorBroadcast* entry points were removed in
        // AiDotNet.Tensors 0.121.0, and [batch, n] against [batch, 1] broadcasts on its own.
        var scaled = Engine.TensorMultiply(work, stdT);
        var shifted = Engine.TensorAdd(scaled, meanT);
        // Clone the shape: Reshape must not alias the source tensor's shape array.
        return reshaped ? Engine.Reshape(shifted, (int[])forecast._shape.Clone()) : shifted;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["ModelDimension"] = NumOps.FromDouble(_modelDimension),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    private Tensor<T> Forward(Tensor<T> input)
    {
        // RevIN forward: normalize so the embedding + RWKV recurrence see a
        // zero-mean unit-std series; the level is restored after the projection.
        var current = NormalizeInputTo3D(ApplyInstanceNormalization(input));
        int batchSize = current.Shape[0];
        int seqLen = current.Shape[1];

        if (seqLen != _contextLength)
            throw new ArgumentException(
                $"Input sequence length ({seqLen}) does not match expected context length ({_contextLength}).");
        int featureDim = current.Shape[2];
        if (featureDim != _numFeatures)
            throw new ArgumentException(
                $"Input feature dimension ({featureDim}) does not match expected numFeatures ({_numFeatures}).");

        _lastForwardSeqLen = seqLen;
        _lastForwardBatchSize = batchSize;

        // Input embedding: [batch*seqLen, numFeatures] -> [batch*seqLen, modelDim]
        if (_inputEmbedding is not null)
        {
            current = Engine.Reshape(current, new[] { batchSize * seqLen, _numFeatures });
            current = _inputEmbedding.Forward(current);
            current = Engine.Reshape(current, new[] { batchSize, seqLen, _modelDimension });
        }

        // RWKV layers: [batch, seqLen, modelDim] -> [batch, seqLen, modelDim]
        // One call: the stack threads v_first across its blocks internally. Looping the blocks here
        // would bypass that and silently drop the value residual.
        if (_rwkvStack is not null)
            current = _rwkvStack.Forward(current);

        // Output projection: take the last timestep's hidden state instead of flattening
        // [seqLen × modelDim]. RWKV is a causal recurrence whose final state has
        // integrated the entire context, so it is the natural fixed-size [batch,
        // modelDim] summary for the forecast head. The old flatten sized the first
        // output-projection Dense at seqLen·modelDim inputs at paper scale (a
        // multi-hundred-million-parameter, multi-GB weight) which overflowed the
        // serializer and OOM-cascaded the suite.
        if (current.Rank == 3)
            current = Engine.TensorSliceAxis(current, axis: 1, index: current.Shape[1] - 1);

        if (_outputProjectionLayers is not null)
        {
            foreach (var layer in _outputProjectionLayers)
                current = layer.Forward(current);
        }

        // RevIN reverse: restore the input's per-instance level/scale.
        return DenormalizeForecast(current);
    }

    /// <summary>
    /// Training-mode forward. Routes through <see cref="Forward"/> so training uses the
    /// same RevIN normalize/denormalize as inference and keeps training mode (dropout)
    /// active, instead of the base default that flips to inference.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Captures the per-layer activations along the model's real forward path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default <see cref="NeuralNetworkBase{T}.GetNamedLayerActivations"/> runs
    /// the <c>Layers</c> list sequentially, but RWKVForecaster reshapes the tensor
    /// between the embedding, the RWKV blocks, and the output projection (2D for the
    /// per-time-step Dense layers, 3D for the RWKV recurrence). Threading the raw
    /// layer list without those reshapes feeds the RWKV layer the wrong rank/feature
    /// count. This override reproduces the genuine forward so the activations are both
    /// non-empty and meaningful.
    /// </para>
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        if (!_useNativeMode)
            return activations;

        var current = NormalizeInputTo3D(input);
        int batchSize = current.Shape[0];
        int seqLen = current.Shape[1];

        if (_inputEmbedding is not null)
        {
            current = Engine.Reshape(current, new[] { batchSize * seqLen, _numFeatures });
            current = _inputEmbedding.Forward(current);
            current = Engine.Reshape(current, new[] { batchSize, seqLen, _modelDimension });
            activations["InputEmbedding"] = current.Clone();
        }

        // Per-block activations are no longer separable here: the stack must run as a unit so
        // v_first is threaded, so the whole stack reports under one name.
        if (_rwkvStack is not null)
        {
            current = _rwkvStack.Forward(current);
            activations["RWKVStack"] = current.Clone();
        }

        // Mirror Forward: take the last timestep's hidden state rather than flattening.
        if (current.Rank == 3)
            current = Engine.TensorSliceAxis(current, axis: 1, index: current.Shape[1] - 1);

        if (_outputProjectionLayers is not null)
        {
            for (int i = 0; i < _outputProjectionLayers.Count; i++)
            {
                current = _outputProjectionLayers[i].Forward(current);
                activations[$"OutputProjection_{i}"] = current.Clone();
            }
        }

        return activations;
    }

    private Tensor<T> NormalizeInputTo3D(Tensor<T> input)
    {
        if (input.Rank == 3) return input;
        if (input.Rank == 2) return Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] });
        if (input.Rank == 1)
        {
            int seqLen, features;
            if (_numFeatures > 1 && input.Length % _numFeatures == 0) { seqLen = input.Length / _numFeatures; features = _numFeatures; }
            else if (_contextLength > 0 && input.Length % _contextLength == 0) { seqLen = _contextLength; features = input.Length / _contextLength; }
            else { seqLen = input.Length; features = 1; }
            return Engine.Reshape(input, new[] { 1, seqLen, features });
        }
        int batchDims = 1;
        for (int i = 0; i < input.Rank - 2; i++) batchDims *= input.Shape[i];
        return Engine.Reshape(input, new[] { batchDims, input.Shape[input.Rank - 2], input.Shape[input.Rank - 1] });
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null) throw new InvalidOperationException("ONNX session not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = OnnxSession.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    #endregion

    #region Model-Specific Processing

    private Tensor<T> GenerateQuantilePredictions(Tensor<T> input, double[] quantiles)
    {
        int numSamples = 100;
        var samples = new List<Tensor<T>>();
        SetTrainingMode(true);
        for (int s = 0; s < numSamples; s++) samples.Add(Forward(input));
        SetTrainingMode(false);

        var result = new Tensor<T>(new[] { 1, _forecastHorizon, quantiles.Length });
        for (int t = 0; t < _forecastHorizon; t++)
        {
            var values = new List<double>();
            foreach (var sample in samples)
                if (t < sample.Length) values.Add(NumOps.ToDouble(sample.Data.Span[t]));
            values.Sort();
            for (int q = 0; q < quantiles.Length; q++)
            {
                int idx = Math.Min((int)(quantiles[q] * values.Count), values.Count - 1);
                result.Data.Span[t * quantiles.Length + q] = NumOps.FromDouble(values[idx]);
            }
        }
        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        var result = new Tensor<T>(input._shape);
        for (int i = 0; i < _contextLength - stepsUsed; i++)
            result.Data.Span[i] = input.Data.Span[i + stepsUsed];
        for (int i = 0; i < stepsUsed && i < predictions.Length; i++)
            result.Data.Span[_contextLength - stepsUsed + i] = predictions.Data.Span[i];
        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        var result = new Tensor<T>(new[] { 1, totalSteps, 1 });
        int position = 0;
        foreach (var pred in predictions)
        {
            int toCopy = Math.Min(pred.Length, totalSteps - position);
            for (int i = 0; i < toCopy; i++) result.Data.Span[position + i] = pred.Data.Span[i];
            position += toCopy;
        }
        return result;
    }

    #endregion

    #region IDisposable

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing) OnnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
