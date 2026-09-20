using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.LearningRateSchedulers;
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
public partial class TOTEM<T> : TimeSeriesFoundationModelBase<T>
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

    // VQ codebook: [numCodebooks x codebookSize x codebookDimension]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _codebooks;

    // RevIN (reversible instance normalization, Kim et al. 2022) statistics.
    // The VQ bottleneck snaps the encoder output to a discrete codebook entry, so
    // constant inputs of different levels map to the same token and decode
    // identically — restoring the input level keeps the forecast input-dependent.
    [Scratch]
    private Vector<T> _revinMean = new Vector<T>(0);
    [Scratch]
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
        : base(architecture, lossFunction ?? new HuberLoss<T>(), 1.0)
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

        CopyOptionsToFields(options);
        _optimizer = optimizer ?? CreateForecastingOptimizer(options);
        SetBaseTrainOptimizer(_optimizer);
        _lossFunction = lossFunction ?? new HuberLoss<T>();

    }

    /// <summary>
    /// Creates a TOTEM model in native mode for training or fine-tuning.
    /// </summary>
    public TOTEM(
        NeuralNetworkArchitecture<T> architecture,
        TOTEMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new HuberLoss<T>(), 1.0)
    {
        options ??= new TOTEMOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        CopyOptionsToFields(options);
        _optimizer = optimizer ?? CreateForecastingOptimizer(options);
        SetBaseTrainOptimizer(_optimizer);
        _lossFunction = lossFunction ?? new HuberLoss<T>();

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
        Guard.Positive(options.TotalTrainingSteps, nameof(options.TotalTrainingSteps));
        if (double.IsNaN(options.LearningRate) || double.IsInfinity(options.LearningRate) || options.LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(options.LearningRate), "Learning rate must be finite and positive.");
        if (double.IsNaN(options.DropoutRate) || double.IsInfinity(options.DropoutRate) ||
            options.DropoutRate < 0.0 || options.DropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(options.DropoutRate), "Dropout rate must be finite and in [0, 1).");
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _codebookSize = options.CodebookSize;
        _codebookDimension = options.CodebookDimension;
        _numCodebooks = options.NumCodebooks;
        _dropout = options.DropoutRate;
        InitializeCodebooks();
    }
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateForecastingOptimizer(
        TOTEMOptions<T> options)
    {
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = options.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                EnableGradientClipping = false,
                LearningRateScheduler = new OneCycleLRScheduler(
                    maxLearningRate: options.LearningRate,
                    totalSteps: options.TotalTrainingSteps),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
    }



    /// <summary>
    /// Initializes the VQ codebook embeddings with random values from N(0, 1/dim).
    /// </summary>
    private void InitializeCodebooks()
    {
        _codebooks = new Tensor<T>(new[] { _numCodebooks, _codebookSize, _codebookDimension });
        // Honour the deterministic init scope the layers use. The codebook is a raw tensor rather than
        // a layer, so it bypassed that scope entirely and always drew from CreateSecureRandom: the
        // codebook differed on every construction even when the caller had pinned an init seed. Most
        // draws train fine — TOTEM passes on its own — but some send the parameter L2 to NaN on the
        // very first step, which made it fail only when it shared a worker with other classes.
        // Falls back to the secure generator in production, where no scope is active.
        int? initSeed = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextSeedOrNull();
        var rand = initSeed.HasValue
            ? RandomHelper.CreateSeededRandom(initSeed.Value)
            : RandomHelper.CreateSecureRandom();
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
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return ForwardNativeForTrainingCore(input);
    }

    /// <inheritdoc/>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(
        IReadOnlyList<Tensor<T>> parameters)
    {
        // The forecasting stage in the official TOTEM pipeline consumes a frozen
        // tokenizer/codebook and optimizes only the time-series decoder. In this
        // integrated model those trainable forecasting components are the decoder
        // projection and forecast head.
        var forecastingParameters = new HashSet<Tensor<T>>(
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

        static void AddLayerParameters(
            ILayer<T>? layer,
            HashSet<Tensor<T>> destination)
        {
            if (layer is not ITrainableLayer<T> trainable)
                return;

            foreach (var parameter in trainable.GetTrainableParameters())
                destination.Add(parameter);
        }

        foreach (var layer in _transformerLayers)
            AddLayerParameters(layer, forecastingParameters);
        AddLayerParameters(_decoder, forecastingParameters);
        AddLayerParameters(_forecastHead, forecastingParameters);

        return parameters.Where(forecastingParameters.Contains).ToArray();
    }

    /// <summary>
    /// Runs the forecasting-stage forward pass through a frozen tokenizer and codebook.
    /// The official TOTEM forecasting pipeline optimizes the downstream transformer and
    /// forecast head only; tokenizer pretraining is a separate stage.
    /// </summary>
    private Tensor<T> ForwardNativeForTrainingCore(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        // Tokenize to [1, contextLength, 1] for the per-token encoder/decoder.
        int seqLen = normalized.Length;
        var current = Engine.Reshape(normalized, new[] { 1, seqLen, 1 });

        if (_encoder is not null)
            current = _encoder.Forward(current);
        if (_quantizationProjection is not null)
            current = _quantizationProjection.Forward(current);

        // The tokenizer and codebook are frozen during the forecasting stage. Quantization
        // remains on-tape only so downstream gradients traverse the straight-through values.
        var quantized = VectorQuantizeTraceable(current);

        var decoded = quantized;
        if (_decoder is not null)
            decoded = _decoder.Forward(decoded);
        foreach (var layer in _transformerLayers)
            decoded = layer.Forward(decoded);

        // Pool the token sequence so the head emits one [1, forecastHorizon] forecast.
        if (decoded.Rank == 3)
            decoded = Engine.ReduceMean(decoded, new[] { 1 }, keepDims: false);

        if (_forecastHead is not null)
            decoded = _forecastHead.Forward(decoded);

        // RevIN reverse: train against the input-scale forecast.
        decoded = DenormalizeForecast(decoded);

        return decoded;
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
        // RevIN forward (Kim et al. 2022), delegated to the shared tape-tracked helper. The previous
        // hand-rolled version accumulated mean/variance with scalar NumOps arithmetic and wrote the
        // output through result.Data.Span[...], which the autodiff tape cannot observe: the normalised
        // tensor came back as a LEAF, so no gradient could flow through the normalisation. RevIN is a
        // differentiable layer in the paper, not a preprocessing step.
        => NormalizeInstanceOnTape(input, DefaultRevInEpsilon, out _revinMean, out _revinStd);

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
        var scaled = Engine.TensorMultiply(work, stdT);
        var shifted = Engine.TensorAdd(scaled, meanT);
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

        // Project to codebook dimension → [1, seqLen, codebookDim]
        if (_quantizationProjection is not null)
            current = _quantizationProjection.Forward(current);

        // Vector Quantization: snap each position to its nearest codebook entry.
        var quantized = VectorQuantize(current);

        // Decoder → [1, seqLen, hiddenDim]
        if (_decoder is not null)
            quantized = _decoder.Forward(quantized);
        // Forecasting transformer consumes the frozen code embeddings.
        foreach (var layer in _transformerLayers)
            quantized = layer.Forward(quantized);

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
    /// Applies the frozen product-quantization tokenizer used by both inference and forecasting training.
    /// </summary>
    /// <remarks>
    /// The tokenizer/codebook objective is optimized in TOTEM's separate pretraining stage; this
    /// forecasting model consumes those discrete representations without mutating the codebook.
    /// </remarks>
    private Tensor<T> VectorQuantize(Tensor<T> encoderOutput)
    {
        return VectorQuantizeTraceable(encoderOutput);
    }

    /// <summary>
    /// Applies the frozen TOTEM product-quantization codebook and returns straight-through
    /// quantized values. The forecasting stage does not update the tokenizer or codebook;
    /// their VQ objective belongs to the separate tokenizer-pretraining stage.
    /// </summary>
    private Tensor<T> VectorQuantizeTraceable(Tensor<T> encoderOutput)
    {
        if (_codebooks is null) InitializeCodebooks();
        var codebooks = _codebooks!;

        int totalLen = encoderOutput.Length;
        int dimPerCodebook = Math.Max(1, _codebookDimension);
        int blockSize = dimPerCodebook * _numCodebooks;
        int numPositions = Math.Max(1, totalLen / Math.Max(1, blockSize));
        int quantizedElements = numPositions * blockSize;

        // If the input cannot form a complete PQ block, leave it unchanged.
        if (numPositions <= 0 || quantizedElements > totalLen)
            return encoderOutput;

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
        var diff = Engine.TensorSubtract(headExpanded, codebookExpanded);
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

        return quantized;
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
