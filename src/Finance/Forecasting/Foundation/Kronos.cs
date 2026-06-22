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
/// Kronos — Foundation Model for the Language of Financial Markets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kronos is a decoder-only foundation model pre-trained on 12B+ K-line (candlestick) records
/// across 45 global exchanges. It natively understands OHLCV (Open, High, Low, Close, Volume)
/// candlestick patterns for financial market forecasting.
/// </para>
/// <para><b>For Beginners:</b> Kronos is a foundation model built specifically for financial
/// markets. It was trained on over 12 billion candlestick records from 45 exchanges worldwide,
/// so it natively understands the language of stock charts (open, high, low, close, volume).
/// Think of it as a model that has "read" every trading chart in history and can predict what
/// comes next based on patterns it has learned.</para>
/// <para>
/// <b>Reference:</b> "Kronos: A Foundation Model for the Language of Financial Markets", 2025.
/// https://arxiv.org/abs/2508.02739
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Kronos foundation model for financial market forecasting
/// // Pre-trained on 12B+ candlestick records across 45 global exchanges
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 5, inputDepth: 1, outputSize: 24);
///
/// // Training mode with OHLCV-native decoder-only architecture
/// var model = new Kronos&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new Kronos&lt;double&gt;(architecture, "kronos.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("Kronos: A Foundation Model for the Language of Financial Markets", "https://arxiv.org/abs/2508.02739")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class Kronos<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly KronosOptions<T> _options;

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
    private int _numCandlestickFeatures;

    // Per-feature RevIN statistics (Kim et al. 2022). The input is a single
    // multivariate OHLCV series [contextLength, numFeatures]; each feature is
    // normalized over the time axis and the forecast is denormalized with the
    // same per-feature stats so level-shifted inputs yield distinct forecasts.
    private Vector<T> _revinMean = new Vector<T>(0);
    private Vector<T> _revinStd = new Vector<T>(0);

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => _numCandlestickFeatures;
    /// <inheritdoc/>
    public override int PatchSize => _patchLength;
    /// <inheritdoc/>
    public override int Stride => _patchLength;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;
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
    /// Creates a Kronos model using a pretrained ONNX model.
    /// </summary>
    public Kronos(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        KronosOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new KronosOptions<T>();
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
    /// Creates a Kronos model in native mode for training or fine-tuning.
    /// </summary>
    public Kronos(
        NeuralNetworkArchitecture<T> architecture,
        KronosOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new KronosOptions<T>();
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

    private void CopyOptionsToFields(KronosOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.PatchLength, nameof(options.PatchLength));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumLayers, nameof(options.NumLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _dropout = options.DropoutRate;
        _modelSize = options.ModelSize;
        _numCandlestickFeatures = options.NumCandlestickFeatures;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultKronosLayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _hiddenDimension, _numLayers, _numHeads, _intermediateSize,
                _numCandlestickFeatures, _dropout));
        }
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
                { "NetworkType", "Kronos" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "NumCandlestickFeatures", _numCandlestickFeatures },
                { "ModelSize", _modelSize.ToString() },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var opts = new KronosOptions<T>
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
            NumCandlestickFeatures = _numCandlestickFeatures
        };

        if (!_useNativeMode && OnnxModelPath is not null)
            return new Kronos<T>(Architecture, OnnxModelPath, opts);

        return new Kronos<T>(Architecture, opts);
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
        writer.Write(_numCandlestickFeatures);
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
        _numCandlestickFeatures = reader.ReadInt32();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("Kronos does not support quantile forecasting. Pass null for point forecasts.");

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
        // The Kronos input is ONE multivariate OHLCV series laid out row-major as
        // [contextLength, numFeatures] (or a flat [contextLength*numFeatures]).
        // RevIN normalizes EACH feature over the time axis (the leading dimension
        // is time, NOT a batch). Per-feature mean/std are stored for the reverse
        // denormalization of the forecast.
        int features = input.Rank > 1 ? input.Shape[input.Rank - 1] : 1;
        int steps = features > 0 ? input.Length / features : input.Length;
        var result = new Tensor<T>(input._shape);
        _revinMean = new Vector<T>(features);
        _revinStd = new Vector<T>(features);

        for (int f = 0; f < features; f++)
        {
            T mean = NumOps.Zero;
            for (int t = 0; t < steps; t++)
            {
                int idx = t * features + f;
                if (idx < input.Length)
                    mean = NumOps.Add(mean, input[idx]);
            }
            mean = NumOps.Divide(mean, NumOps.FromDouble(steps));

            T variance = NumOps.Zero;
            for (int t = 0; t < steps; t++)
            {
                int idx = t * features + f;
                if (idx < input.Length)
                {
                    var diff = NumOps.Subtract(input[idx], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(steps));
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));
            _revinMean[f] = mean;
            _revinStd[f] = std;

            for (int t = 0; t < steps; t++)
            {
                int idx = t * features + f;
                if (idx < input.Length && idx < result.Length)
                    result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std);
            }
        }

        return result;
    }

    /// <summary>
    /// RevIN reverse step (Kim et al. 2022): restores each feature's mean/std to
    /// the forecast (laid out [..., numFeatures]) so it is on the input's original
    /// scale. Tape-connected (Engine broadcast ops) so the forecast head keeps its
    /// gradients.
    /// </summary>
    private Tensor<T> DenormalizeForecast(Tensor<T> forecast)
    {
        int features = _revinMean.Length;
        if (features <= 0 || forecast.Length % features != 0)
            return forecast;

        var meanRow = new Tensor<T>(new[] { 1, features });
        var stdRow = new Tensor<T>(new[] { 1, features });
        for (int f = 0; f < features; f++) { meanRow.Data.Span[f] = _revinMean[f]; stdRow.Data.Span[f] = _revinStd[f]; }

        int rows = forecast.Length / features;
        var fc2d = Engine.Reshape(forecast, new[] { rows, features });
        var scaled = Engine.TensorBroadcastMultiply(fc2d, stdRow);
        var shifted = Engine.TensorBroadcastAdd(scaled, meanRow);
        return Engine.Reshape(shifted, forecast._shape);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumCandlestickFeatures"] = NumOps.FromDouble(_numCandlestickFeatures),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        // Kronos (financial decoder-only model). Helper emits a flat
        // sequentially-composable layers list: Reshape → Dense(patch) → N ×
        // TransformerEncoderLayer (+ optional Dropout) → Flatten →
        // Dense(head). ForwardNative is a straight sequential dispatch.
        var current = ApplyInstanceNormalization(input);

        // Flatten the whole (possibly multivariate) series into a single batch row
        // [1, contextLength*numFeatures] so the patch ReshapeLayer reshapes the
        // entire sequence into [numPatches, patchLength*numFeatures]. Previously a
        // [contextLength, numFeatures] input left the leading time dimension intact,
        // which the ReshapeLayer mis-read as a batch axis and threw on.
        bool flattened = !(current.Rank == 2 && current.Shape[0] == 1);
        if (flattened)
            current = current.Reshape(new[] { 1, current.Length });

        foreach (var layer in Layers)
            current = layer.Forward(current);

        // RevIN reverse: put the forecast back on the input's per-feature scale.
        current = DenormalizeForecast(current);

        if (flattened && current.Rank == 2 && current.Shape[0] == 1)
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

    #region Parameter Estimation

    private new int GetParameterCount()
    {
        int numPatches = _contextLength / _patchLength;
        long total = (long)_patchLength * _numCandlestickFeatures * _hiddenDimension + _hiddenDimension;

        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perLayer += 2L * _hiddenDimension * _intermediateSize + _hiddenDimension + _intermediateSize;
        perLayer += 4L * _hiddenDimension;
        total += perLayer * _numLayers;

        total += 2L * _hiddenDimension;
        total += (long)numPatches * _hiddenDimension * _forecastHorizon * _numCandlestickFeatures;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
