using System.IO;
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
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// MOMENT (Multi-task Optimization through Masked Encoding for Time series) foundation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// MOMENT is a family of open time series foundation models from Carnegie Mellon University.
/// It uses a T5-based encoder-only transformer with patch-based input and RevIN to handle
/// five downstream tasks: forecasting, anomaly detection, classification, imputation, and
/// embedding generation — all from a single pretrained backbone.
/// </para>
/// <para>
/// <b>For Beginners:</b> MOMENT is the first true multi-task time series foundation model:
///
/// <b>Architecture Overview:</b>
/// 1. <b>Patch Embedding:</b> Divides the input into fixed-length patches (e.g., 64 steps each)
/// 2. <b>RevIN:</b> Reversible Instance Normalization handles different scales automatically
/// 3. <b>T5 Encoder:</b> A stack of transformer encoder layers processes the patches
/// 4. <b>Task Heads:</b> Separate output heads for each supported task
///
/// <b>Multi-Task Capability:</b>
/// - <b>Forecasting:</b> Linear projection from encoder output to future values
/// - <b>Anomaly Detection:</b> Reconstruction-based — anomalies have high reconstruction error
/// - <b>Classification:</b> Pooled encoder output fed to a classification head
/// - <b>Imputation:</b> Masked patches reconstructed using unmasked context
/// - <b>Embedding:</b> Mean-pooled encoder output serves as the representation
///
/// <b>Key Insight:</b> MOMENT's pretraining uses masked reconstruction (like BERT),
/// which naturally supports all five tasks without task-specific pretraining.
///
/// <b>Model Sizes (MOMENT family):</b>
/// - MOMENT-Small: ~40M parameters
/// - MOMENT-Base: ~385M parameters (default)
/// - MOMENT-Large: ~1B+ parameters
/// </para>
/// <para>
/// <b>Reference:</b> Goswami et al., "MOMENT: A Family of Open Time-Series Foundation Models",
/// ICML 2024. https://arxiv.org/abs/2402.03885
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is NOT thread-safe. Create separate instances for concurrent usage.
/// </para>
/// </remarks>
public class MOMENT<T> : TimeSeriesFoundationModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this model uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Patch embedding layer that projects raw patches to hidden dimension.
    /// </summary>
    private ILayer<T>? _patchEmbedding;

    /// <summary>
    /// Transformer encoder layers (T5-style).
    /// </summary>
    private readonly List<ILayer<T>> _transformerLayers = [];

    /// <summary>
    /// Final layer normalization before task heads.
    /// </summary>
    private ILayer<T>? _finalLayerNorm;

    /// <summary>
    /// Forecasting head: projects encoder output to forecast horizon.
    /// </summary>
    private ILayer<T>? _forecastHead;

    /// <summary>
    /// Reconstruction head: used for anomaly detection and imputation.
    /// </summary>
    private ILayer<T>? _reconstructionHead;

    /// <summary>
    /// Classification head: projects pooled output to class logits.
    /// </summary>
    private ILayer<T>? _classificationHead;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly MOMENTOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _patchLength;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _intermediateSize;
    private double _dropout;
    private FoundationModelSize _modelSize;
    private TimeSeriesFoundationModelTask _currentTask;
    private int? _numClasses;
    private double _maskRatio;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => 1;

    /// <inheritdoc/>
    public override int PatchSize => _patchLength;

    /// <inheritdoc/>
    public override int Stride => _patchLength;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => _modelSize;

    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;

    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override TimeSeriesFoundationModelTask CurrentTask => _currentTask;

    /// <inheritdoc/>
    public override IReadOnlyList<TimeSeriesFoundationModelTask> SupportedTasks { get; } = new[]
    {
        TimeSeriesFoundationModelTask.Forecasting,
        TimeSeriesFoundationModelTask.AnomalyDetection,
        TimeSeriesFoundationModelTask.Classification,
        TimeSeriesFoundationModelTask.Imputation,
        TimeSeriesFoundationModelTask.Embedding
    };

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a MOMENT model using a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Load pretrained MOMENT weights for immediate use on any
    /// of the five supported tasks.
    /// </para>
    /// </remarks>
    public MOMENT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        MOMENTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new MOMENTOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
    }

    /// <summary>
    /// Creates a MOMENT model in native mode for training or fine-tuning.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for fine-tuning on domain-specific data or training
    /// from scratch (requires significant compute for large variants).
    /// </para>
    /// </remarks>
    public MOMENT(
        NeuralNetworkArchitecture<T> architecture,
        MOMENTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new MOMENTOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(MOMENTOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _dropout = options.DropoutRate;
        _modelSize = options.ModelSize;
        _currentTask = options.Task;
        _numClasses = options.NumClasses;
        _maskRatio = options.MaskRatio;
    }

    #endregion

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMOMENTLayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _hiddenDimension, _numLayers, _numHeads, _intermediateSize,
                _dropout, _numClasses));

            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Patch embedding
        if (idx < Layers.Count)
            _patchEmbedding = Layers[idx++];

        // Transformer layers
        _transformerLayers.Clear();
        int layersPerBlock = _dropout > 0 ? 9 : 7;
        int totalTransformerLayers = _numLayers * layersPerBlock;

        for (int i = 0; i < totalTransformerLayers && idx < Layers.Count; i++)
        {
            _transformerLayers.Add(Layers[idx++]);
        }

        // Final layer norm
        if (idx < Layers.Count)
            _finalLayerNorm = Layers[idx++];

        // Forecasting head
        if (idx < Layers.Count)
            _forecastHead = Layers[idx++];

        // Reconstruction head (for anomaly detection / imputation)
        if (idx < Layers.Count)
            _reconstructionHead = Layers[idx++];

        // Classification head (optional)
        if (idx < Layers.Count)
            _classificationHead = Layers[idx++];
    }

    /// <inheritdoc/>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 4)
        {
            throw new ArgumentException(
                "MOMENT requires at least 4 layers (patch embed, transformer, norm, forecast head).",
                nameof(layers));
        }
    }

    private static void ValidateOptions(MOMENTOptions<T> options)
    {
        var errors = new List<string>();

        if (options.ContextLength < 1)
            errors.Add("ContextLength must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.PatchLength < 1)
            errors.Add("PatchLength must be at least 1.");
        if (options.ContextLength % options.PatchLength != 0)
            errors.Add("ContextLength must be divisible by PatchLength.");
        if (options.HiddenDimension < 1)
            errors.Add("HiddenDimension must be at least 1.");
        if (options.NumLayers < 1)
            errors.Add("NumLayers must be at least 1.");
        if (options.NumHeads < 1)
            errors.Add("NumHeads must be at least 1.");
        if (options.HiddenDimension % options.NumHeads != 0)
            errors.Add("HiddenDimension must be divisible by NumHeads.");
        if (options.IntermediateSize < 1)
            errors.Add("IntermediateSize must be at least 1.");
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            errors.Add("DropoutRate must be between 0 and 1 (exclusive).");
        if (options.MaskRatio < 0 || options.MaskRatio >= 1)
            errors.Add("MaskRatio must be between 0 and 1 (exclusive).");
        if (options.Task == TimeSeriesFoundationModelTask.Classification && (options.NumClasses == null || options.NumClasses < 2))
            errors.Add("NumClasses must be at least 2 when using classification task.");

        if (errors.Count > 0)
            throw new ArgumentException($"Invalid MOMENT options: {string.Join(", ", errors)}");
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

        SetTrainingMode(true);
        try
        {
            var output = ForwardNative(input);
            LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

            var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
            BackwardNative(Tensor<T>.FromVector(gradient, output.Shape));

            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "MOMENT" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "IntermediateSize", _intermediateSize },
                { "ModelSize", _modelSize.ToString() },
                { "CurrentTask", _currentTask.ToString() },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new MOMENTOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            IntermediateSize = _intermediateSize,
            DropoutRate = _dropout,
            ModelSize = _modelSize,
            Task = _currentTask,
            NumClasses = _numClasses,
            MaskRatio = _maskRatio
        };

        return new MOMENT<T>(Architecture, options);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_intermediateSize);
        writer.Write(_dropout);
        writer.Write((int)_modelSize);
        writer.Write((int)_currentTask);
        writer.Write(_numClasses ?? -1);
        writer.Write(_maskRatio);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _intermediateSize = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _modelSize = (FoundationModelSize)reader.ReadInt32();
        _currentTask = (TimeSeriesFoundationModelTask)reader.ReadInt32();
        int nc = reader.ReadInt32();
        _numClasses = nc >= 0 ? nc : null;
        _maskRatio = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
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
            {
                currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed);
            }
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
        // MOMENT uses RevIN (Reversible Instance Normalization)
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input.Shape);

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
                {
                    result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std);
                }
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
            ["PatchLength"] = NumOps.FromDouble(_patchLength),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Multi-Task Overrides

    /// <inheritdoc/>
    public override Tensor<T> DetectAnomalies(Tensor<T> series, double? threshold = null)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.AnomalyDetection);

        if (!_useNativeMode)
            throw new NotSupportedException(
                "Anomaly detection requires native mode. ONNX inference only supports forecasting.");

        // Anomaly detection via reconstruction error
        var normalized = ApplyInstanceNormalization(series);
        var reconstructed = ReconstructNative(normalized);

        int batchSize = series.Shape[0];
        int seqLen = series.Shape.Length > 1 ? series.Shape[1] : series.Length;
        var scores = new Tensor<T>(new[] { batchSize, seqLen, 1 });

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int srcIdx = b * seqLen + t;
                int dstIdx = b * seqLen + t;

                if (srcIdx < series.Length && srcIdx < reconstructed.Length && dstIdx < scores.Length)
                {
                    var diff = NumOps.Subtract(normalized.Data.Span[srcIdx], reconstructed.Data.Span[srcIdx]);
                    scores.Data.Span[dstIdx] = NumOps.Multiply(diff, diff); // Squared reconstruction error
                }
            }
        }

        return scores;
    }

    /// <inheritdoc/>
    public override Tensor<T> Classify(Tensor<T> series, int numClasses)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Classification);

        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "Number of classes must be positive.");

        if (!_useNativeMode)
            throw new NotSupportedException(
                "Classification requires native mode. ONNX inference only supports forecasting.");

        var normalized = ApplyInstanceNormalization(series);
        var encoded = ForwardEncoder(normalized);

        // Mean pool over sequence dimension
        int batchSize = encoded.Shape[0];
        int seqDim = encoded.Shape.Length > 1 ? encoded.Shape[1] : 1;
        int hiddenDim = encoded.Shape.Length > 2 ? encoded.Shape[2] : encoded.Length / (batchSize * seqDim);

        var pooled = new Tensor<T>(new[] { batchSize, hiddenDim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqDim; s++)
                {
                    int idx = b * seqDim * hiddenDim + s * hiddenDim + h;
                    if (idx < encoded.Length)
                        sum = NumOps.Add(sum, encoded[idx]);
                }
                int dstIdx = b * hiddenDim + h;
                if (dstIdx < pooled.Length)
                    pooled.Data.Span[dstIdx] = NumOps.Divide(sum, NumOps.FromDouble(seqDim));
            }
        }

        // Classification head
        if (_classificationHead is not null)
            return _classificationHead.Forward(pooled);

        // No classification head available — cannot classify
        throw new InvalidOperationException(
            "Classification head is not initialized. Ensure the model was created with classification layers.");
    }

    /// <inheritdoc/>
    public override Tensor<T> Impute(Tensor<T> series, Tensor<T> mask)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Imputation);

        if (!_useNativeMode)
            throw new NotSupportedException(
                "Imputation requires native mode. ONNX inference only supports forecasting.");

        // Apply mask to input (zero out missing values)
        var masked = new Tensor<T>(series.Shape);
        for (int i = 0; i < series.Length && i < mask.Length; i++)
        {
            masked.Data.Span[i] = NumOps.Multiply(series[i], mask[i]);
        }

        // Reconstruct full sequence
        var normalized = ApplyInstanceNormalization(masked);
        var reconstructed = ReconstructNative(normalized);

        // Blend: use original where mask=1, reconstruction where mask=0
        var result = new Tensor<T>(series.Shape);
        for (int i = 0; i < series.Length; i++)
        {
            if (i < mask.Length && NumOps.ToDouble(mask[i]) > 0.5)
            {
                result.Data.Span[i] = series[i];
            }
            else if (i < reconstructed.Length)
            {
                result.Data.Span[i] = reconstructed[i];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override Tensor<T> Embed(Tensor<T> series)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Embedding);

        if (!_useNativeMode)
            throw new NotSupportedException(
                "Embedding extraction requires native mode. ONNX inference only supports forecasting.");

        var normalized = ApplyInstanceNormalization(series);
        var encoded = ForwardEncoder(normalized);

        // Mean pool over sequence dimension to get fixed-size embedding
        int batchSize = encoded.Shape[0];
        int seqDim = encoded.Shape.Length > 1 ? encoded.Shape[1] : 1;
        int hiddenDim = encoded.Shape.Length > 2 ? encoded.Shape[2] : encoded.Length / (batchSize * seqDim);

        var embedding = new Tensor<T>(new[] { batchSize, hiddenDim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqDim; s++)
                {
                    int idx = b * seqDim * hiddenDim + s * hiddenDim + h;
                    if (idx < encoded.Length)
                        sum = NumOps.Add(sum, encoded[idx]);
                }
                int dstIdx = b * hiddenDim + h;
                if (dstIdx < embedding.Length)
                    embedding.Data.Span[dstIdx] = NumOps.Divide(sum, NumOps.FromDouble(seqDim));
            }
        }

        return embedding;
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the full native forward pass through the MOMENT architecture.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        var encoded = ForwardEncoder(normalized);

        // Apply task-specific head
        if (_forecastHead is not null)
            return _forecastHead.Forward(encoded);

        return encoded;
    }

    /// <summary>
    /// Runs the encoder portion only (patch embed + transformer layers + norm).
    /// </summary>
    private Tensor<T> ForwardEncoder(Tensor<T> input)
    {
        var current = input;

        // Add batch dimension if needed
        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // Patch embedding
        if (_patchEmbedding is not null)
            current = _patchEmbedding.Forward(current);

        // Transformer encoder layers
        foreach (var layer in _transformerLayers)
        {
            current = layer.Forward(current);
        }

        // Final layer norm
        if (_finalLayerNorm is not null)
            current = _finalLayerNorm.Forward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
        {
            current = current.Reshape(new[] { current.Shape[1] });
        }

        return current;
    }

    /// <summary>
    /// Reconstructs the input for anomaly detection and imputation tasks.
    /// </summary>
    private Tensor<T> ReconstructNative(Tensor<T> input)
    {
        var encoded = ForwardEncoder(input);

        if (_reconstructionHead is not null)
            return _reconstructionHead.Forward(encoded);

        return encoded;
    }

    /// <summary>
    /// Performs the backward pass through the MOMENT architecture.
    /// </summary>
    private Tensor<T> BackwardNative(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // Backward through forecast head
        if (_forecastHead is not null)
            current = _forecastHead.Backward(current);

        // Backward through final norm
        if (_finalLayerNorm is not null)
            current = _finalLayerNorm.Backward(current);

        // Backward through transformer layers (reverse order)
        for (int i = _transformerLayers.Count - 1; i >= 0; i--)
        {
            current = _transformerLayers[i].Backward(current);
        }

        // Backward through patch embedding
        if (_patchEmbedding is not null)
            current = _patchEmbedding.Backward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
        {
            current = current.Reshape(new[] { current.Shape[1] });
        }

        return current;
    }

    /// <summary>
    /// Runs inference using the ONNX model.
    /// </summary>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        int features = input.Rank > 2 ? input.Shape[2] : 1;

        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++)
        {
            inputData[i] = (float)NumOps.ToDouble(input[i]);
        }

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
        {
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return output;
    }

    #endregion

    #region Parameter Estimation

    private new int GetParameterCount()
    {
        int numPatches = _contextLength / _patchLength;
        int patchInputSize = _patchLength;

        // Patch embedding parameters
        long total = (long)patchInputSize * _hiddenDimension + _hiddenDimension;

        // Transformer layer parameters (per layer)
        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension; // attention
        perLayer += 2L * _hiddenDimension * _intermediateSize + _hiddenDimension + _intermediateSize; // FFN
        perLayer += 4L * _hiddenDimension; // layer norms
        total += perLayer * _numLayers;

        // Final layer norm
        total += 2L * _hiddenDimension;

        // Task heads
        total += (long)numPatches * _hiddenDimension * _forecastHorizon; // forecast
        total += (long)numPatches * _hiddenDimension * numPatches * _patchLength; // reconstruction

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
