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
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// VisionTS — Visual Masked Autoencoders as Zero-Shot Time Series Forecasters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VisionTS repurposes Visual Masked Autoencoders (MAE) pretrained on images for time series
/// forecasting. It converts time series into 2D image-like patch grids, processes them with
/// a pretrained ViT encoder, and reconstructs/forecasts using the decoder. This cross-modal
/// transfer demonstrates that vision foundation models generalize to time series.
/// </para>
/// <para><b>For Beginners:</b> VisionTS takes a surprising approach: it converts time series
/// data into images and uses a vision model (originally trained on photos) to forecast future
/// values. The data is arranged in a 2D grid like pixels, and the vision model fills in the
/// missing parts, effectively predicting future values. This works because patterns in time
/// series grids resemble visual textures that vision models already understand.</para>
/// <para>
/// <b>Reference:</b> "VisionTS: Visual Masked Autoencoders as Zero-Shot Time Series Forecasters",
/// ICML 2025.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a VisionTS model that repurposes visual MAE for time series forecasting
/// // Converts time series into 2D image-like patch grids for cross-modal transfer
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with ViT encoder and MAE decoder
/// var model = new VisionTS&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained vision model
/// var onnxModel = new VisionTS&lt;double&gt;(architecture, "visionts.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("VisionTS: Visual Masked Autoencoders as Zero-Shot Time Series Forecasters", "https://arxiv.org/abs/2408.17253")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public partial class VisionTS<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly VisionTSOptions<T> _options;

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
    private double _maskRatio;

    // RevIN (reversible instance normalization, Kim et al. 2022) statistics.
    // VisionTS normalizes each input series before the ViT and restores the level
    // on the output so distinct input scales produce distinct forecasts.
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

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a VisionTS model using a pretrained ONNX model.
    /// </summary>
    public VisionTS(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        VisionTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        var session = new InferenceSession(onnxModelPath);
        try
        {
            options ??= new VisionTSOptions<T>();
            _options = options;
            Options = _options;

            _useNativeMode = false;
            OnnxModelPath = onnxModelPath;

            CopyOptionsToFields(options);
            _optimizer = optimizer ?? CreatePaperOptimizer(options);
            _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
            SetBaseTrainOptimizer(_optimizer);
            OnnxSession = session;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a VisionTS model in native mode.
    /// </summary>
    public VisionTS(
        NeuralNetworkArchitecture<T> architecture,
        VisionTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new VisionTSOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        CopyOptionsToFields(options);
        _optimizer = optimizer ?? CreatePaperOptimizer(options);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        SetBaseTrainOptimizer(_optimizer);

        InitializeLayers();
    }

    private void CopyOptionsToFields(VisionTSOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.PatchLength, nameof(options.PatchLength));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumLayers, nameof(options.NumLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));
        Guard.Positive(options.IntermediateSize, nameof(options.IntermediateSize));
        Guard.Positive(options.LearningRate, nameof(options.LearningRate));

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _dropout = options.DropoutRate;
        _modelSize = options.ModelSize;
        _maskRatio = options.MaskRatio;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(VisionTSOptions<T> options)
    {
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = options.LearningRate,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                EnableGradientClipping = false
            });
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultVisionTSLayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _hiddenDimension, _numLayers, _numHeads, _intermediateSize, _dropout));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// The VisionTS full-shot recipe freezes the visual MAE and fine-tunes only
    /// LayerNorm affine parameters. Keep that selection at the shared tape
    /// boundary so Adam moments and clipping see exactly the paper's subset.
    /// </summary>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(
        IReadOnlyList<Tensor<T>> parameters)
    {
        var layerNormParameters = new HashSet<Tensor<T>>(
            AiDotNet.Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

        foreach (var layer in Layers)
        {
            if (layer is TransformerEncoderLayer<T> encoder)
            {
                foreach (var parameter in encoder.GetLayerNormalizationTrainableParameters())
                    layerNormParameters.Add(parameter);
            }
            else if (layer is LayerNormalizationLayer<T> normalization)
            {
                foreach (var parameter in normalization.GetTrainableParameters())
                    layerNormParameters.Add(parameter);
            }
        }

        var selected = parameters.Where(layerNormParameters.Contains).ToArray();
        if (parameters.Count > 0 && selected.Length == 0)
        {
            throw new InvalidOperationException(
                "VisionTS full-shot training requires initialized LayerNorm parameters, but the configured " +
                "architecture did not expose any trainable LayerNorm tensors.");
        }

        return selected;
    }

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

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "VisionTS" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "ModelSize", _modelSize.ToString() },
                { "MaskRatio", _maskRatio },
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
            throw new NotSupportedException("VisionTS does not support quantile forecasting. Pass null for point forecasts.");

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
        T mse = NumOps.Zero, mae = NumOps.Zero;
        int count = 0;
        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
            count++;
        }
        if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); }
        return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) };
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
            ["PatchLength"] = NumOps.FromDouble(_patchLength),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward

    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var current = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (current.Rank == 1) { current = current.Reshape(new[] { 1, current.Length }); addedBatchDim = true; }
        foreach (var layer in Layers)
            current = layer.Forward(current);
        // RevIN reverse: restore the input's per-instance level/scale so distinct
        // input levels yield distinct forecasts (the ViT sees only the normalized series).
        current = DenormalizeForecast(current);
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

    /// <inheritdoc/>
    /// <remarks>
    /// Mirrors <see cref="ForwardNative"/>'s preprocessing so the captured activations match the
    /// real forward pass: a bare rank-1 context is RevIN-normalized and given a leading batch
    /// axis before it reaches the patch <c>ReshapeLayer</c>.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode)
            return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();
        var current = ApplyInstanceNormalization(input);
        if (current.Rank == 1)
            current = current.Reshape(new[] { 1, current.Length });
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }
        return activations;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized.");
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        int features = input.Rank > 2 ? input.Shape[2] : 1;
        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]);
        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);
        int totalElements = 1;
        foreach (var dim in outputShape) totalElements *= dim;
        for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        return output;
    }

    #endregion

    #region Parameter Estimation

    private new int GetParameterCount()
    {
        int numPatches = _contextLength / _patchLength;
        long total = (long)_patchLength * _hiddenDimension + _hiddenDimension;

        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perLayer += 2L * _hiddenDimension * _intermediateSize + _hiddenDimension + _intermediateSize;
        perLayer += 4L * _hiddenDimension;
        total += perLayer * _numLayers;

        total += 2L * _hiddenDimension;
        total += (long)numPatches * _hiddenDimension * _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
