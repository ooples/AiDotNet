using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;
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
/// TOTEM — TOkenized Time Series EMbeddings via VQ-VAE.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TOTEM learns discrete tokenized representations for time series via VQ-VAE,
/// enabling the use of discrete token-based methods (like LLMs) on continuous time series data.
/// It uses an encoder-decoder architecture with vector quantization bottleneck.
/// </para>
/// <para><b>For Beginners:</b> TOTEM converts continuous time series data into discrete tokens
/// (like words in a vocabulary), making it possible to use language model techniques on
/// numerical data. Think of it as creating a "dictionary" of common time series patterns:
/// each chunk of data gets matched to its closest dictionary entry, creating a compact
/// representation that language-style models can process.</para>
/// <para>
/// <b>Reference:</b> Talukder et al., "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis", 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TOTEM model for tokenized time series embeddings via VQ-VAE
/// // Converts continuous time series to discrete tokens for language-model-style processing
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with VQ-VAE encoder-decoder and vector quantization
/// var model = new TOTEM&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TOTEM&lt;double&gt;(architecture, "totem.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("TOTEM: TOkenized Time Series EMbeddings", "https://arxiv.org/abs/2402.16412")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class TOTEM<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _encoder;
    private readonly List<ILayer<T>> _transformerLayers = [];
    private ILayer<T>? _quantizationProjection;
    private ILayer<T>? _decoder;
    private ILayer<T>? _forecastHead;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TOTEMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _codebookSize;
    private int _codebookDimension;
    private int _numCodebooks;
    private double _dropout;
    private double _commitmentWeight;

    // VQ codebook: [numCodebooks x codebookSize x codebookDimension]
    private Tensor<T>? _codebooks;
    private T _lastCommitmentLoss;

    // RevIN (reversible instance normalization, Kim et al. 2022) statistics.
    // The VQ bottleneck snaps the encoder output to a discrete codebook entry, so
    // constant inputs of different levels map to the same token and decode
    // identically — restoring the input level keeps the forecast input-dependent.
    private Vector<T> _revinMean = new Vector<T>(0);
    private Vector<T> _revinStd = new Vector<T>(0);

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
    public override FoundationModelSize ModelSize => FoundationModelSize.Base;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TOTEM model using a pretrained ONNX model.
    /// </summary>
    public TOTEM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TOTEMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TOTEMOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _lastCommitmentLoss = NumOps.Zero;

        CopyOptionsToFields(options);
    }

    /// <summary>
    /// Creates a TOTEM model in native mode for training or fine-tuning.
    /// </summary>
    public TOTEM(
        NeuralNetworkArchitecture<T> architecture,
        TOTEMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TOTEMOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _lastCommitmentLoss = NumOps.Zero;

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(TOTEMOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumLayers, nameof(options.NumLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));
        Guard.Positive(options.CodebookSize, nameof(options.CodebookSize));
        Guard.Positive(options.CodebookDimension, nameof(options.CodebookDimension));
        Guard.Positive(options.NumCodebooks, nameof(options.NumCodebooks));

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _codebookSize = options.CodebookSize;
        _codebookDimension = options.CodebookDimension;
        _numCodebooks = options.NumCodebooks;
        _dropout = options.DropoutRate;
        _commitmentWeight = options.CommitmentWeight;
        InitializeCodebooks();
    }

    /// <summary>
    /// Initializes the VQ codebook embeddings with random values from N(0, 1/dim).
    /// </summary>
    private void InitializeCodebooks()
    {
        _codebooks = new Tensor<T>(new[] { _numCodebooks, _codebookSize, _codebookDimension });
        var rand = RandomHelper.CreateSecureRandom();
        T scale = NumOps.Divide(NumOps.One, NumOps.FromDouble(Math.Sqrt(_codebookDimension)));
        for (int c = 0; c < _numCodebooks; c++)
            for (int k = 0; k < _codebookSize; k++)
                for (int d = 0; d < _codebookDimension; d++)
                {
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    T sample = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                    int idx = c * _codebookSize * _codebookDimension + k * _codebookDimension + d;
                    _codebooks.Data.Span[idx] = NumOps.Multiply(scale, sample);
                }
    }

    /// <summary>Gets a codebook value at the given indices.</summary>
    private T GetCodebookValue(int codebook, int entry, int dim)
    {
        int idx = codebook * _codebookSize * _codebookDimension + entry * _codebookDimension + dim;
        var codebooks = _codebooks ?? throw new InvalidOperationException("Codebooks not initialized.");
        return codebooks[idx];
    }

    /// <summary>Sets a codebook value at the given indices.</summary>
    private void SetCodebookValue(int codebook, int entry, int dim, T value)
    {
        int idx = codebook * _codebookSize * _codebookDimension + entry * _codebookDimension + dim;
        var codebooks = _codebooks ?? throw new InvalidOperationException("Codebooks not initialized.");
        codebooks.Data.Span[idx] = value;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTOTEMLayers(
                Architecture, _contextLength, _forecastHorizon, _hiddenDimension,
                _numLayers, _numHeads, _codebookDimension, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        if (idx < Layers.Count)
            _encoder = Layers[idx++];

        _transformerLayers.Clear();
        // Must match CreateDefaultTOTEMLayers' per-block layer count: BatchNorm,
        // Dense, Dense, [Dropout], BatchNorm, Dense, Dense, [Dropout] = 6, or 8
        // when dropout > 0.
        int layersPerBlock = _dropout > 0 ? 8 : 6;
        int totalTransformerLayers = _numLayers * layersPerBlock;

        for (int i = 0; i < totalTransformerLayers && idx < Layers.Count; i++)
            _transformerLayers.Add(Layers[idx++]);

        if (idx < Layers.Count)
            _quantizationProjection = Layers[idx++];

        if (idx < Layers.Count)
            _decoder = Layers[idx++];

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

        // TOTEM-specific training: reconstruction loss + commitment
        // term for the vector-quantized encoder. The base trainer's
        // supervised path only sees reconstruction, which drops the
        // VQ-VAE commitment penalty and lets the encoder drift
        // arbitrarily far from the codebook. Run a custom tape step
        // that combines both.

        var loss = LossFunction as LossFunctions.LossFunctionBase<T>
            ?? throw new InvalidOperationException(
                "LossFunction must derive from LossFunctionBase<T> for TOTEM tape-based training.");

        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();

        // GPU-RESIDENT fast path — recon + commitment on a fused SGD plan. Safe
        // now that VectorQuantize is fully traceable (argmin + gather + straight-
        // through + commitment loss all via engine ops) so each replay recomputes
        // from the CURRENT slot data instead of freezing the trace-batch argmin
        // into the plan. The codebook EMA runs POST-Step in eager code using the
        // trace-time argmin/head tensor references — their .Data is refreshed by
        // each replay, so the post-Step read gives the current batch's values and
        // the update lands exactly once per batch (CodeRabbit contract).
        var trainableLayers = Layers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableLayers.Count > 0)
        {
            Tensor<T>? capturedCommitment = null;
            Tensor<int>? capturedArgmin = null;
            Tensor<T>? capturedHead = null;
            Tensor<T> ForwardCombined(Tensor<T> inp)
            {
                var (fc, commit, argmin, head) = ForwardNativeForTrainingWithVQExtras(inp);
                capturedCommitment = commit;
                capturedArgmin = argmin;
                capturedHead = head;
                return fc;
            }
            Tensor<T> ComputeLossCombined(Tensor<T> pred, Tensor<T> tgt)
            {
                var alignedT = tgt;
                if (pred.Rank > tgt.Rank && pred.Shape[0] == 1 && pred.Length == tgt.Length)
                    pred = Engine.Reshape(pred, tgt._shape);
                else if (tgt.Rank > pred.Rank && tgt.Shape[0] == 1 && tgt.Length == pred.Length)
                    alignedT = Engine.Reshape(tgt, pred._shape);
                var recon = loss.ComputeTapeLoss(pred, alignedT);
                var commit = capturedCommitment
                    ?? throw new InvalidOperationException(
                        "TOTEM fused step: commitment loss was not captured by ForwardCombined. " +
                        "This indicates the fused-step framework called the loss closure before " +
                        "the forward closure, violating its documented Fwd-then-Loss ordering.");
                return Engine.TensorAdd(recon, commit);
            }
            if (AiDotNet.Training.CompiledTapeTrainingStep<T>.TryStepWithFusedOptimizer(
                    trainableLayers, input, target,
                    forward: ForwardCombined, computeLoss: ComputeLossCombined,
                    optimizerType: AiDotNet.Tensors.Engines.Compilation.OptimizerType.SGD,
                    learningRate: 0.001f, beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f, weightDecay: 0f,
                    out T fusedLoss))
            {
                LastLoss = fusedLoss;
                if (IsTrainingMode && capturedArgmin is not null && capturedHead is not null)
                    UpdateCodebookEMA(capturedHead, capturedArgmin);
                return;
            }
        }

        using var tape = new GradientTape<T>();
        var (forecast, commitmentLoss) = ForwardNativeForTrainingWithCommitment(input);

        var alignedTarget = target;
        if (forecast.Rank > target.Rank && forecast.Shape[0] == 1 && forecast.Length == target.Length)
            forecast = Engine.Reshape(forecast, target._shape);
        else if (target.Rank > forecast.Rank && target.Shape[0] == 1 && target.Length == forecast.Length)
            alignedTarget = Engine.Reshape(target, forecast._shape);
        var reconLoss = loss.ComputeTapeLoss(forecast, alignedTarget);

        var totalLoss = Engine.TensorAdd(reconLoss, commitmentLoss);

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
    /// Training-mode forward. Routes the encoder → VQ → decoder →
    /// forecast head pipeline through the tape with a straight-through
    /// vector quantizer so gradients flow into the encoder and
    /// forecast head while the codebook lookup's argmin stays non-
    /// differentiable (it has to — there's no gradient to an argmin).
    /// Also emits the commitment loss as a tape-aware tensor for the
    /// caller to add to the reconstruction loss.
    /// </summary>
    private (Tensor<T> forecast, Tensor<T> commitmentLoss) ForwardNativeForTrainingWithCommitment(Tensor<T> input)
    {
        var (forecast, commitmentLoss, _, _) = ForwardNativeForTrainingWithVQExtras(input);
        return (forecast, commitmentLoss);
    }

    /// <summary>
    /// Training forward that also exposes the VQ argmin indices and encoder head
    /// tensors. Used by the compiled fused path so the caller can invoke
    /// <see cref="UpdateCodebookEMA"/> AFTER each Step with post-replay values
    /// (argmin/head are graph-node references whose <c>.Data</c> is refreshed
    /// by each replay). All ops go through <see cref="Engine"/> so the full
    /// forward — including the RevIN normalize/denormalize, the encoder, the
    /// quantization projection, VQ argmin+gather+straight-through, commitment
    /// loss, and the decoder — records on the autodiff tape and re-executes
    /// per replay.
    /// </summary>
    private (Tensor<T> Forecast, Tensor<T> CommitmentLoss, Tensor<int>? Argmin, Tensor<T>? Head)
        ForwardNativeForTrainingWithVQExtras(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        // Tokenize to [1, contextLength, 1] for the per-token encoder/decoder.
        int seqLen = normalized.Length;
        var current = Engine.Reshape(normalized, new[] { 1, seqLen, 1 });

        if (_encoder is not null)
            current = _encoder.Forward(current);
        foreach (var layer in _transformerLayers)
            current = layer.Forward(current);
        if (_quantizationProjection is not null)
            current = _quantizationProjection.Forward(current);

        // Traceable VQ: returns straight-through-quantized values, commitment loss,
        // argmin indices, and the reshaped-head input for post-Step EMA.
        var (quantizedST, commitmentLoss, argmin, head) = VectorQuantizeTraceable(current);

        var decoded = quantizedST;
        if (_decoder is not null)
            decoded = _decoder.Forward(decoded);

        // Pool the token sequence so the head emits one [1, forecastHorizon] forecast.
        if (decoded.Rank == 3)
            decoded = Engine.ReduceMean(decoded, new[] { 1 }, keepDims: false);

        if (_forecastHead is not null)
            decoded = _forecastHead.Forward(decoded);

        // RevIN reverse: train against the input-scale forecast.
        decoded = DenormalizeForecast(decoded);

        return (decoded, commitmentLoss, argmin, head);
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
                { "NetworkType", "TOTEM" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "CodebookSize", _codebookSize },
                { "NumCodebooks", _numCodebooks },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TOTEM<T>(Architecture, new TOTEMOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            CodebookSize = _codebookSize,
            CodebookDimension = _codebookDimension,
            NumCodebooks = _numCodebooks,
            DropoutRate = _dropout,
            CommitmentWeight = _commitmentWeight
        });
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_codebookSize);
        writer.Write(_codebookDimension);
        writer.Write(_numCodebooks);
        writer.Write(_dropout);
        writer.Write(_commitmentWeight);

        // Serialize codebook embeddings
        if (_codebooks is not null)
        {
            writer.Write(true);
            for (int c = 0; c < _numCodebooks; c++)
                for (int k = 0; k < _codebookSize; k++)
                    for (int d = 0; d < _codebookDimension; d++)
                        writer.Write(NumOps.ToDouble(GetCodebookValue(c, k, d)));
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _codebookSize = reader.ReadInt32();
        _codebookDimension = reader.ReadInt32();
        _numCodebooks = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _commitmentWeight = reader.ReadDouble();

        // Deserialize codebook embeddings
        bool hasCodebooks = reader.ReadBoolean();
        if (hasCodebooks)
        {
            _codebooks = new Tensor<T>(new[] { _numCodebooks, _codebookSize, _codebookDimension });
            for (int c = 0; c < _numCodebooks; c++)
                for (int k = 0; k < _codebookSize; k++)
                    for (int d = 0; d < _codebookDimension; d++)
                        SetCodebookValue(c, k, d, NumOps.FromDouble(reader.ReadDouble()));
        }
        else
        {
            InitializeCodebooks();
        }

        // The base deserializer has already recreated every layer in Layers with the
        // copied weights. Re-point the cached encoder/decoder/projection references at
        // those layers; otherwise they keep pointing at the stale random-initialized
        // layers from CreateNewInstance and a clone diverges from the original.
        ExtractLayerReferences();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("TOTEM does not support quantile forecasting. Pass null for point forecasts.");

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
        // RevIN forward (Kim et al. 2022). Stats over every non-batch element of
        // each row (a rank-1 input is a single instance), stored for the reverse.
        int batchSize = input.Shape.Length > 1 ? input.Shape[0] : 1;
        int instanceSize = batchSize > 0 ? input.Length / batchSize : input.Length;
        if (instanceSize <= 0)
            return input;

        var result = new Tensor<T>(input._shape);
        _revinMean = new Vector<T>(batchSize);
        _revinStd = new Vector<T>(batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            int start = b * instanceSize;

            T mean = NumOps.Zero;
            for (int t = 0; t < instanceSize; t++)
                mean = NumOps.Add(mean, input[start + t]);
            mean = NumOps.Divide(mean, NumOps.FromDouble(instanceSize));

            T variance = NumOps.Zero;
            for (int t = 0; t < instanceSize; t++)
            {
                var diff = NumOps.Subtract(input[start + t], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(instanceSize));
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

            _revinMean[b] = mean;
            _revinStd[b] = std;

            for (int t = 0; t < instanceSize; t++)
                result.Data.Span[start + t] = NumOps.Divide(NumOps.Subtract(input[start + t], mean), std);
        }

        return result;
    }

    /// <summary>
    /// RevIN reverse step (Kim et al. 2022): restores each instance's mean/std to the
    /// forecast so it is expressed on the input's original scale, via tape-connected
    /// Engine ops.
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
        var scaled = Engine.TensorBroadcastMultiply(work, stdT);
        var shifted = Engine.TensorBroadcastAdd(scaled, meanT);
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
            ["CodebookSize"] = NumOps.FromDouble(_codebookSize),
            ["NumCodebooks"] = NumOps.FromDouble(_numCodebooks),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// VQ-VAE forward pass: encode → transformer → project to codebook dim →
    /// vector quantize (nearest neighbor lookup) → decode → forecast.
    /// The quantization bottleneck forces discrete token representations.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);

        // Tokenize: [contextLength] (or [1, contextLength]) → [1, contextLength, 1]
        // so the per-token encoder/decoder project each timestep.
        int seqLen = normalized.Length;
        var current = Engine.Reshape(normalized, new[] { 1, seqLen, 1 });

        // Encoder → [1, seqLen, hiddenDim]
        if (_encoder is not null)
            current = _encoder.Forward(current);

        // Transformer layers (per-token)
        foreach (var layer in _transformerLayers)
            current = layer.Forward(current);

        // Project to codebook dimension → [1, seqLen, codebookDim]
        if (_quantizationProjection is not null)
            current = _quantizationProjection.Forward(current);

        // Vector Quantization: snap each position to its nearest codebook entry.
        var quantized = VectorQuantize(current);

        // Decoder → [1, seqLen, hiddenDim]
        if (_decoder is not null)
            quantized = _decoder.Forward(quantized);

        // Pool the token sequence so the head emits one [1, forecastHorizon] forecast.
        if (quantized.Rank == 3)
            quantized = Engine.ReduceMean(quantized, new[] { 1 }, keepDims: false);

        if (_forecastHead is not null)
            quantized = _forecastHead.Forward(quantized);

        // RevIN reverse: restore the input's per-instance level/scale so distinct
        // input levels yield distinct forecasts despite the VQ bottleneck.
        quantized = DenormalizeForecast(quantized);

        if (quantized.Rank == 2 && quantized.Shape[0] == 1)
            quantized = quantized.Reshape(new[] { quantized.Shape[1] });

        return quantized;
    }

    /// <summary>
    /// Vector quantization: for each codebook, find nearest embedding to each input vector.
    /// Uses product quantization when numCodebooks > 1 (splits features across codebooks).
    /// Also computes commitment loss: ||z_e - sg(e_k)||^2.
    /// </summary>
    /// <summary>
    /// Legacy scalar-loop VectorQuantize — kept for callers that don't need the argmin/head
    /// side-outputs (inference, serialization roundtrips). Training paths should use
    /// <see cref="VectorQuantizeTraceable"/> so the entire quantization runs on-tape and
    /// re-executes correctly under the compiled fused plan.
    /// </summary>
    private Tensor<T> VectorQuantize(Tensor<T> encoderOutput)
    {
        var (quantized, _, _, _) = VectorQuantizeTraceable(encoderOutput);
        return quantized;
    }

    /// <summary>
    /// Traceable VQ-VAE quantization step (van den Oord et al. 2017). Returns the
    /// straight-through-quantized tensor, the commitment loss, the argmin indices,
    /// and the reshaped-head input in the [numPositions, numCodebooks, codebookDim]
    /// layout. All ops go through <see cref="Engine"/> so the computation records on
    /// the autodiff tape and re-executes on every replay under a compiled fused plan
    /// — the previous <c>.Data.Span</c> nearest-neighbor + <c>SetCodebookValue</c>
    /// EMA loop froze the argmin decision AND applied the codebook update at trace
    /// time (bug flagged by CodeRabbit).
    /// </summary>
    /// <remarks>
    /// EMA is intentionally NOT applied here. The caller invokes <see cref="UpdateCodebookEMA"/>
    /// with the returned argmin+head AFTER the compiled Step so the codebook update
    /// runs exactly once per batch (regardless of whether the fused or eager path
    /// engaged). Under the compiled plan, <c>head</c> and <c>argmin</c> are trace-time
    /// graph nodes whose <c>.Data</c> is refreshed by each replay — reading them
    /// post-Step gives the current batch's values.
    /// </remarks>
    private (Tensor<T> Quantized, Tensor<T> CommitmentLoss, Tensor<int>? ArgminIndices, Tensor<T>? Head)
        VectorQuantizeTraceable(Tensor<T> encoderOutput)
    {
        if (_codebooks is null) InitializeCodebooks();
        var codebooks = _codebooks!;

        int totalLen = encoderOutput.Length;
        int dimPerCodebook = Math.Max(1, _codebookDimension);
        int blockSize = dimPerCodebook * _numCodebooks;
        int numPositions = Math.Max(1, totalLen / Math.Max(1, blockSize));
        int quantizedElements = numPositions * blockSize;

        // Fallback: input can't be cleanly reshaped into the PQ block structure.
        // Return the input unchanged with a zero commitment loss and no argmin/head
        // (the caller's EMA-update path is a no-op when these are null).
        if (numPositions <= 0 || quantizedElements > totalLen)
        {
            var zeroLoss = new Tensor<T>(new[] { 1 });
            Engine.TensorFill(zeroLoss, NumOps.Zero);
            _lastCommitmentLoss = NumOps.Zero;
            return (encoderOutput, zeroLoss, null, null);
        }

        // Split input into [quantizable, passThrough]. The passThrough tail is
        // copied unchanged; the quantizable prefix goes through PQ.
        var flatInput = encoderOutput.Rank == 1
            ? encoderOutput
            : Engine.Reshape(encoderOutput, new[] { totalLen });
        var quantizable = Engine.TensorSlice(flatInput, new[] { 0 }, new[] { quantizedElements });

        // head[p, c, d] — reshape the quantizable prefix into PQ block layout.
        var head = Engine.Reshape(quantizable, new[] { numPositions, _numCodebooks, dimPerCodebook });

        // Distance to each codebook entry: broadcast head [P, C, 1, D] against
        // codebook [1, C, K, D] → diff [P, C, K, D] → sum(diff²) → [P, C, K].
        // codebooks shape: [numCodebooks, codebookSize, codebookDim] → add batch axis.
        var headExpanded = Engine.Reshape(head, new[] { numPositions, _numCodebooks, 1, dimPerCodebook });
        var codebookExpanded = Engine.Reshape(codebooks, new[] { 1, _numCodebooks, _codebookSize, dimPerCodebook });
        var diff = Engine.TensorBroadcastSubtract(headExpanded, codebookExpanded);
        var diffSq = Engine.TensorMultiply(diff, diff);
        var distances = Engine.ReduceSum(diffSq, new[] { 3 }, keepDims: false);
        // distances shape: [numPositions, numCodebooks, codebookSize].

        // Argmin over the codebookSize axis — non-differentiable by design; the
        // straight-through estimator below routes gradients around the argmin.
        var argmin = Engine.TensorArgMin(distances, axis: 2);
        // argmin shape: [numPositions, numCodebooks] of Tensor<int>.

        // Per-codebook gather: for each c, zqSlices[c][p, :] = codebooks[c, argmin[p, c], :].
        // TensorIndexSelectDiff along the codebookSize axis of the per-c codebook slice.
        var zqSlices = new Tensor<T>[_numCodebooks];
        for (int c = 0; c < _numCodebooks; c++)
        {
            // Slice codebook_c = codebooks[c, :, :] via TensorSliceAxis(axis=0, index=c).
            var codebookC = Engine.TensorSliceAxis(codebooks, axis: 0, index: c);
            // argminC = argmin[:, c] shape [numPositions] — TensorSliceAxis on int tensor.
            var argminC = Engine.TensorSliceAxis(argmin, axis: 1, index: c);
            // Gather: source shape [codebookSize, codebookDim], indices [numPositions] along axis 0
            //   → [numPositions, codebookDim].
            zqSlices[c] = Engine.TensorIndexSelectDiff(codebookC, argminC, axis: 0);
        }
        // Stack per-codebook slices along the codebook axis to get [numPositions, numCodebooks, codebookDim].
        var zq = Engine.TensorStack(zqSlices, axis: 1);

        // Commitment loss per Oord 2017 §3.2: β · ||z_e - sg(e_k)||² averaged over totalLen.
        // Straight-through routes gradient through encoder only — encoder learns to
        // match codebook via commitment loss; codebook learns via EMA (separate path).
        var zqDetached = Engine.StopGradient(zq);
        var commitmentDelta = Engine.TensorSubtract(head, zqDetached);
        var commitmentSqSum = Engine.ReduceSum(
            Engine.TensorMultiply(commitmentDelta, commitmentDelta),
            axes: null, keepDims: false);
        var invTotalLen = NumOps.Divide(NumOps.One, NumOps.FromDouble(Math.Max(1, totalLen)));
        var commitmentLoss = Engine.TensorMultiplyScalar(
            commitmentSqSum,
            NumOps.Multiply(NumOps.FromDouble(_commitmentWeight), invTotalLen));
        _lastCommitmentLoss = commitmentLoss.Length > 0 ? commitmentLoss[0] : NumOps.Zero;

        // Straight-through: quantized = head + StopGradient(zq - head). Forward-values
        // equal codebook entries; backward gradient flows through head as if identity.
        var straightThroughShift = Engine.StopGradient(Engine.TensorSubtract(zq, head));
        var quantizedBlocks = Engine.TensorAdd(head, straightThroughShift);
        var quantizedFlat = Engine.Reshape(quantizedBlocks, new[] { quantizedElements });

        Tensor<T> quantized;
        if (quantizedElements < totalLen)
        {
            // Concat the passThrough tail unchanged.
            var passThroughLen = totalLen - quantizedElements;
            var passThrough = Engine.TensorSlice(flatInput, new[] { quantizedElements }, new[] { passThroughLen });
            var combined = Engine.TensorConcatenate(new[] { quantizedFlat, passThrough }, axis: 0);
            quantized = encoderOutput.Rank == 1 ? combined : Engine.Reshape(combined, encoderOutput._shape);
        }
        else
        {
            quantized = encoderOutput.Rank == 1 ? quantizedFlat : Engine.Reshape(quantizedFlat, encoderOutput._shape);
        }

        return (quantized, commitmentLoss, argmin, head);
    }

    /// <summary>
    /// Post-Step EMA codebook update (van den Oord 2017 §3.2). Runs exactly once
    /// per batch, using the trace-time <paramref name="argmin"/> / <paramref name="head"/>
    /// tensors captured by <see cref="VectorQuantizeTraceable"/> — under the compiled
    /// plan these are graph-node references whose <c>.Data</c> reflects the LAST
    /// replay, so post-Step reads give the current batch's values.
    /// </summary>
    /// <remarks>
    /// EMA formula (last-wins matching the eager path's per-position race):
    /// <c>codebook[c, argmin[p,c], :] ← decay · codebook[c, argmin[p,c], :] +
    /// (1-decay) · head[p, c, :]</c>.
    /// Expressed as an in-place scatter-add: <c>codebook += (1-decay) · scatter(head - zq_at_selected)</c>,
    /// where the scatter writes the deltas into the codebook at (c, argmin) positions
    /// and the codebook's underlying data is updated via <see cref="IEngine.TensorCopy"/>
    /// so the tensor object identity (referenced by future trace/replay reads) stays
    /// stable.
    /// </remarks>
    private void UpdateCodebookEMA(Tensor<T> head, Tensor<int> argmin)
    {
        if (_codebooks is null) return;
        var codebooks = _codebooks;

        int dimPerCodebook = Math.Max(1, _codebookDimension);
        int numPositions = head.Shape.Length >= 3 ? head.Shape[0] : 1;

        var decayT = NumOps.FromDouble(0.99);
        var oneMinusDecayT = NumOps.FromDouble(0.01);

        // Per-codebook scatter: for each c, the selected entries move by
        // (1-decay)·(head[:, c, :] - codebook[c, argmin[:, c], :]).
        // TensorScatterAdd writes updates into a fresh copy of the codebook slice;
        // TensorCopy propagates the update back into the same codebook tensor object.
        for (int c = 0; c < _numCodebooks; c++)
        {
            var codebookC = Engine.TensorSliceAxis(codebooks, axis: 0, index: c);
            var headC = Engine.TensorSliceAxis(head, axis: 1, index: c);
            var argminC = Engine.TensorSliceAxis(argmin, axis: 1, index: c);

            // Current selected entries: gather codebookC by argminC along axis 0.
            var zqC = Engine.TensorIndexSelectDiff(codebookC, argminC, axis: 0);

            // Delta = (1-decay) · (headC - zqC), shape [numPositions, codebookDim].
            var deltaC = Engine.TensorMultiplyScalar(
                Engine.TensorSubtract(headC, zqC),
                oneMinusDecayT);

            // Scatter deltaC into a zero canvas at argminC indices along axis 0.
            var canvas = new Tensor<T>(new[] { _codebookSize, dimPerCodebook });
            Engine.TensorFill(canvas, NumOps.Zero);
            var updatedCodebookC = Engine.TensorScatterAdd(canvas, argminC, deltaC, axis: 0);
            // updatedCodebookC = deltas placed at scatter positions, zeros elsewhere.

            // codebookC_new = codebookC + updatedCodebookC.
            var codebookC_new = Engine.TensorAdd(codebookC, updatedCodebookC);

            // Write codebookC_new back into codebooks[c, :, :] via TensorSliceAxisWrite-like
            // pattern: reshape to [1, codebookSize, codebookDim], place with TensorScatter
            // along axis 0 at index c into a per-codebook staging tensor. Simpler:
            // reconstruct the full codebook by concatenating updated + unchanged slices.
            var updated3D = Engine.Reshape(codebookC_new, new[] { 1, _codebookSize, dimPerCodebook });
            var partsList = new System.Collections.Generic.List<Tensor<T>>();
            if (c > 0)
            {
                partsList.Add(Engine.TensorSlice(codebooks, new[] { 0, 0, 0 }, new[] { c, _codebookSize, dimPerCodebook }));
            }
            partsList.Add(updated3D);
            if (c + 1 < _numCodebooks)
            {
                partsList.Add(Engine.TensorSlice(codebooks,
                    new[] { c + 1, 0, 0 },
                    new[] { _numCodebooks - c - 1, _codebookSize, dimPerCodebook }));
            }
            var newCodebooks = partsList.Count == 1 ? partsList[0] : Engine.TensorConcatenate(partsList.ToArray(), axis: 0);
            Engine.TensorCopy(newCodebooks, codebooks);
        }

        // Suppress unused-var warning for decayT (kept for documentation of the
        // decay-in-the-formula intent; the (1-decay) factor is what's actually used).
        _ = decayT;
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

    #region Parameter Estimation

    private new int GetParameterCount()
    {
        long total = (long)_contextLength * _hiddenDimension + _hiddenDimension;
        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perLayer += 2L * _hiddenDimension * (_hiddenDimension * 4) + _hiddenDimension + (_hiddenDimension * 4);
        perLayer += 4L * _hiddenDimension;
        total += perLayer * _numLayers;
        total += (long)_codebookSize * _codebookDimension * _numCodebooks;
        total += (long)_hiddenDimension * _codebookDimension + _codebookDimension;
        total += (long)_codebookDimension * _hiddenDimension + _hiddenDimension;
        total += (long)_hiddenDimension * _forecastHorizon + _forecastHorizon;
        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
