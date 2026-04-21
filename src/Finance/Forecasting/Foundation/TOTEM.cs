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
        int layersPerBlock = _dropout > 0 ? 9 : 7;
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
    public override Tensor<T> Predict(Tensor<T> input)
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
        var normalized = ApplyInstanceNormalization(input);
        var current = normalized;
        if (current.Rank == 1)
            current = Engine.Reshape(current, new[] { 1, current.Length });

        if (_encoder is not null)
            current = _encoder.Forward(current);
        foreach (var layer in _transformerLayers)
            current = layer.Forward(current);
        if (_quantizationProjection is not null)
            current = _quantizationProjection.Forward(current);

        // Non-differentiable lookup: find nearest codebook entry per
        // position. VectorQuantize returns a plain Tensor<T> built by
        // .Data.Span — this is intentional here; we use it as the
        // stop-gradient target in the straight-through trick.
        var codebookValues = VectorQuantize(current);

        // Straight-through estimator:
        //   quantized = encoder + sg(codebook - encoder)
        //   forward: evaluates to codebook values (VQ behavior)
        //   backward: d quantized / d encoder = 1 (gradient passes through)
        var diff = Engine.TensorSubtract(codebookValues, current);
        var diffDetached = Engine.StopGradient(diff);
        var quantizedST = Engine.TensorAdd(current, diffDetached);

        // Commitment loss: mean((encoder - sg(codebook))^2), weighted.
        // Pulling encoder toward codebook encourages discrete
        // quantization without destabilizing codebook values.
        var codebookDetached = Engine.StopGradient(codebookValues);
        var commitDiff = Engine.TensorSubtract(current, codebookDetached);
        var commitSq = Engine.TensorMultiply(commitDiff, commitDiff);
        var allAxes = Enumerable.Range(0, commitSq.Rank).ToArray();
        var commitmentLoss = Engine.ReduceMean(commitSq, allAxes, keepDims: false);
        commitmentLoss = Engine.TensorMultiplyScalar(commitmentLoss,
            NumOps.FromDouble(_commitmentWeight));

        var decoded = quantizedST;
        if (_decoder is not null)
            decoded = _decoder.Forward(decoded);
        if (_forecastHead is not null)
            decoded = _forecastHead.Forward(decoded);

        return (decoded, commitmentLoss);
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
        var current = normalized;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // Encoder
        if (_encoder is not null)
            current = _encoder.Forward(current);

        // Transformer layers
        foreach (var layer in _transformerLayers)
            current = layer.Forward(current);

        // Project to codebook dimension
        if (_quantizationProjection is not null)
            current = _quantizationProjection.Forward(current);

        // Vector Quantization: find nearest codebook entry for each position
        var quantized = VectorQuantize(current);

        // Straight-through estimator: use quantized for forward, but pass gradient through
        // In this simplified version, we just use the quantized output
        if (_decoder is not null)
            quantized = _decoder.Forward(quantized);

        if (_forecastHead is not null)
            quantized = _forecastHead.Forward(quantized);

        if (addedBatchDim && quantized.Rank == 2 && quantized.Shape[0] == 1)
            quantized = quantized.Reshape(new[] { quantized.Shape[1] });

        return quantized;
    }

    /// <summary>
    /// Vector quantization: for each codebook, find nearest embedding to each input vector.
    /// Uses product quantization when numCodebooks > 1 (splits features across codebooks).
    /// Also computes commitment loss: ||z_e - sg(e_k)||^2.
    /// </summary>
    private Tensor<T> VectorQuantize(Tensor<T> encoderOutput)
    {
        if (_codebooks is null) InitializeCodebooks();

        int totalLen = encoderOutput.Length;
        var quantized = new Tensor<T>(encoderOutput._shape);
        T commitmentLoss = NumOps.Zero;

        // Product quantization: split encoded vector across codebooks
        int dimPerCodebook = Math.Max(1, _codebookDimension);
        int numPositions = Math.Max(1, totalLen / Math.Max(1, dimPerCodebook * _numCodebooks));

        for (int pos = 0; pos < numPositions; pos++)
        {
            for (int c = 0; c < _numCodebooks; c++)
            {
                int startIdx = pos * dimPerCodebook * _numCodebooks + c * dimPerCodebook;
                if (startIdx >= totalLen) break;

                int effectiveDim = Math.Min(dimPerCodebook, totalLen - startIdx);

                // Find nearest codebook entry
                int bestIdx = 0;
                T bestDist = NumOps.MaxValue;
                for (int k = 0; k < _codebookSize; k++)
                {
                    T dist = NumOps.Zero;
                    for (int d = 0; d < effectiveDim; d++)
                    {
                        int idx = startIdx + d;
                        if (idx >= totalLen) break;
                        T diff = NumOps.Subtract(encoderOutput[idx], GetCodebookValue(c, k, d % _codebookDimension));
                        dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                    }
                    if (NumOps.LessThan(dist, bestDist)) { bestDist = dist; bestIdx = k; }
                }

                // Replace encoder output with nearest codebook entry (straight-through)
                for (int d = 0; d < effectiveDim; d++)
                {
                    int idx = startIdx + d;
                    if (idx >= totalLen) break;
                    T codebookVal = GetCodebookValue(c, bestIdx, d % _codebookDimension);

                    // Straight-through: quantized = encoder_output + sg(codebook - encoder_output)
                    // This means forward uses codebook values, backward passes gradient through encoder
                    quantized.Data.Span[idx] = codebookVal;

                    // Commitment loss: ||z_e - sg(e_k)||^2
                    T diff = NumOps.Subtract(encoderOutput[idx], codebookVal);
                    commitmentLoss = NumOps.Add(commitmentLoss, NumOps.Multiply(diff, diff));
                }

                // EMA codebook update (during training): move codebook entry toward encoder output
                if (IsTrainingMode)
                {
                    T emaDecay = NumOps.FromDouble(0.99);
                    T oneMinusDecay = NumOps.FromDouble(0.01);
                    for (int d = 0; d < effectiveDim && d < _codebookDimension; d++)
                    {
                        int idx = startIdx + d;
                        if (idx >= totalLen) break;
                        T currentVal = GetCodebookValue(c, bestIdx, d);
                        T newVal = NumOps.Add(NumOps.Multiply(emaDecay, currentVal), NumOps.Multiply(oneMinusDecay, encoderOutput[idx]));
                        SetCodebookValue(c, bestIdx, d, newVal);
                    }
                }
            }
        }

        // Copy through any remaining elements that don't fit product quantization
        int quantizedElements = numPositions * _numCodebooks * dimPerCodebook;
        for (int i = quantizedElements; i < totalLen; i++)
            quantized.Data.Span[i] = encoderOutput[i];

        T commitWeightT = NumOps.FromDouble(_commitmentWeight);
        T invTotalLen = NumOps.Divide(NumOps.One, NumOps.FromDouble(Math.Max(1, totalLen)));
        _lastCommitmentLoss = NumOps.Multiply(NumOps.Multiply(commitmentLoss, commitWeightT), invTotalLen);
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
