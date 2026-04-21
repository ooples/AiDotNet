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
public class TFC<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _timeInputProjection;
    private readonly List<ILayer<T>> _timeEncoderLayers = [];
    private ILayer<T>? _freqInputProjection;
    private readonly List<ILayer<T>> _freqEncoderLayers = [];
    private ILayer<T>? _projectionHead;
    private ILayer<T>? _forecastHead;

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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        var loss = LossFunction as LossFunctions.LossFunctionBase<T>
            ?? throw new InvalidOperationException(
                "LossFunction must derive from LossFunctionBase<T> for TFC tape-based training.");

        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();

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

        var allGrads = tape.ComputeGradients(totalLoss, sources: null);
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

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TFC<T>(Architecture, new TFCOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            ProjectionDimension = _projectionDimension,
            NumTimeLayers = _numTimeLayers,
            NumFreqLayers = _numFreqLayers,
            DropoutRate = _dropout,
            ContrastiveTemperature = _contrastiveTemperature
        });
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_projectionDimension);
        writer.Write(_numTimeLayers);
        writer.Write(_numFreqLayers);
        writer.Write(_dropout);
        writer.Write(_contrastiveTemperature);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _projectionDimension = reader.ReadInt32();
        _numTimeLayers = reader.ReadInt32();
        _numFreqLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _contrastiveTemperature = reader.ReadDouble();
    }

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
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input._shape);

        for (int b = 0; b < batchSize; b++)
        {
            T mean = NumOps.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length)
                    mean = NumOps.Add(mean, input[idx]);
            }
            mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen));

            T variance = NumOps.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length)
                {
                    var diff = NumOps.Subtract(input[idx], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen));
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length && idx < result.Length)
                    result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std);
            }
        }

        return result;
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
        var normalized = ApplyInstanceNormalization(input);
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

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });

        return current;
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
    /// Computes the DFT magnitude spectrum of the input time series.
    /// For rank-1 input, computes DFT directly. For batched input (rank > 1),
    /// computes DFT per sample along the last dimension.
    /// Returns |X[k]| for k = 0..N/2 (one-sided spectrum), same shape as input via mirroring.
    /// </summary>
    private Tensor<T> ComputeFrequencyRepresentation(Tensor<T> input)
    {
        // For rank-1 (unbatched), n = sequence length. For batched, n = last dimension.
        int n = input.Rank > 1 ? input.Shape[^1] : input.Length;
        int numSamples = input.Rank > 1 ? input.Length / n : 1;
        int halfN = n / 2 + 1;
        var result = new Tensor<T>(input._shape);
        T invN = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));

        for (int s = 0; s < numSamples; s++)
        {
            int offset = s * n;

            // DFT: X[k] = sum_{t=0}^{N-1} x[t] * exp(-2*pi*i*k*t/N)
            for (int k = 0; k < halfN; k++)
            {
                T realPart = NumOps.Zero;
                T imagPart = NumOps.Zero;
                for (int t = 0; t < n; t++)
                {
                    double angle = -2.0 * Math.PI * k * t / n;
                    T cosT = NumOps.FromDouble(Math.Cos(angle));
                    T sinT = NumOps.FromDouble(Math.Sin(angle));
                    realPart = NumOps.Add(realPart, NumOps.Multiply(input[offset + t], cosT));
                    imagPart = NumOps.Add(imagPart, NumOps.Multiply(input[offset + t], sinT));
                }
                T magSquared = NumOps.Add(NumOps.Multiply(realPart, realPart), NumOps.Multiply(imagPart, imagPart));
                result.Data.Span[offset + k] = NumOps.Multiply(NumOps.Sqrt(magSquared), invN);
            }

            // Mirror the one-sided spectrum for symmetric representation
            for (int k = halfN; k < n; k++)
                result.Data.Span[offset + k] = result[offset + (n - k)];
        }

        return result;
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
