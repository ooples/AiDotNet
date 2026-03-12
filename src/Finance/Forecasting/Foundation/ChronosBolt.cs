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
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// Chronos-Bolt — Fast Non-Autoregressive Time Series Forecasting from the Chronos Family.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chronos-Bolt uses an encoder-decoder architecture with direct quantile forecasting
/// (non-autoregressive), making it significantly faster than autoregressive Chronos v1/v2.
/// The encoder processes the input context and the decoder directly outputs all forecast
/// quantiles in a single forward pass without iterative generation.
/// </para>
/// <para>
/// <b>Reference:</b> Part of Amazon Chronos family, Nov 2024.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Chronos: Learning the Language of Time Series", "https://arxiv.org/abs/2403.07815", Year = 2024, Authors = "Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Yuyang Wang")]
public class ChronosBolt<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly List<ILayer<T>> _encoderLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];
    private ILayer<T>? _patchEmbedding;
    private ILayer<T>? _quantileHead;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly ChronosBoltOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _patchLength;
    private int _encoderHiddenDim;
    private int _decoderHiddenDim;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _numHeads;
    private double _dropout;
    private FoundationModelSize _modelSize;
    private int _numQuantiles;

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
    /// Creates a Chronos-Bolt model using a pretrained ONNX model.
    /// </summary>
    public ChronosBolt(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ChronosBoltOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new ChronosBoltOptions<T>();
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
    /// Creates a Chronos-Bolt model in native mode.
    /// </summary>
    public ChronosBolt(
        NeuralNetworkArchitecture<T> architecture,
        ChronosBoltOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new ChronosBoltOptions<T>();
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

    private void CopyOptionsToFields(ChronosBoltOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.PatchLength, nameof(options.PatchLength));
        Guard.Positive(options.EncoderHiddenDim, nameof(options.EncoderHiddenDim));
        Guard.Positive(options.DecoderHiddenDim, nameof(options.DecoderHiddenDim));
        Guard.Positive(options.NumEncoderLayers, nameof(options.NumEncoderLayers));
        Guard.Positive(options.NumDecoderLayers, nameof(options.NumDecoderLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));
        Guard.Positive(options.NumQuantiles, nameof(options.NumQuantiles));

        if (options.DropoutRate < 0.0 || options.DropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(options), $"DropoutRate must be in [0, 1) but was {options.DropoutRate}.");

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _encoderHiddenDim = options.EncoderHiddenDim;
        _decoderHiddenDim = options.DecoderHiddenDim;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _modelSize = options.ModelSize;
        _numQuantiles = options.NumQuantiles;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultChronosBoltLayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _encoderHiddenDim, _decoderHiddenDim,
                _numEncoderLayers, _numDecoderLayers, _numHeads,
                _numQuantiles, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        if (idx < Layers.Count)
            _patchEmbedding = Layers[idx++];

        _encoderLayers.Clear();
        for (int i = 0; i < _numEncoderLayers && idx < Layers.Count; i++)
            _encoderLayers.Add(Layers[idx++]);

        _decoderLayers.Clear();
        for (int i = 0; i < _numDecoderLayers && idx < Layers.Count; i++)
            _decoderLayers.Add(Layers[idx++]);

        if (idx < Layers.Count)
            _quantileHead = Layers[idx++];
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
        finally { SetTrainingMode(false); }
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
                { "NetworkType", "ChronosBolt" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "EncoderHiddenDim", _encoderHiddenDim },
                { "DecoderHiddenDim", _decoderHiddenDim },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "NumQuantiles", _numQuantiles },
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
        var opts = new ChronosBoltOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            EncoderHiddenDim = _encoderHiddenDim,
            DecoderHiddenDim = _decoderHiddenDim,
            NumEncoderLayers = _numEncoderLayers,
            NumDecoderLayers = _numDecoderLayers,
            NumHeads = _numHeads,
            DropoutRate = _dropout,
            ModelSize = _modelSize,
            NumQuantiles = _numQuantiles
        };

        if (!_useNativeMode && OnnxModelPath is not null)
            return new ChronosBolt<T>(Architecture, OnnxModelPath, opts);

        return new ChronosBolt<T>(Architecture, opts);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_encoderHiddenDim);
        writer.Write(_decoderHiddenDim);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_dropout);
        writer.Write((int)_modelSize);
        writer.Write(_numQuantiles);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _encoderHiddenDim = reader.ReadInt32();
        _decoderHiddenDim = reader.ReadInt32();
        _numEncoderLayers = reader.ReadInt32();
        _numDecoderLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _modelSize = (FoundationModelSize)reader.ReadInt32();
        _numQuantiles = reader.ReadInt32();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var rawOutput = _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);

        // If no specific quantiles requested, return the median (middle quantile)
        if (quantiles is null || quantiles.Length == 0)
        {
            // Raw output is [horizon * numQuantiles]; extract median (middle index)
            int medianIdx = _numQuantiles / 2;
            var median = new Tensor<T>(new[] { _forecastHorizon });
            for (int t = 0; t < _forecastHorizon && t * _numQuantiles + medianIdx < rawOutput.Length; t++)
                median.Data.Span[t] = rawOutput[t * _numQuantiles + medianIdx];
            return median;
        }

        // Map requested quantile levels to nearest model quantile indices
        // Model quantiles are evenly spaced: 1/(N+1), 2/(N+1), ..., N/(N+1)
        var result = new Tensor<T>(new[] { _forecastHorizon * quantiles.Length });
        for (int q = 0; q < quantiles.Length; q++)
        {
            // Find the closest model quantile index for the requested level
            int bestIdx = 0;
            double bestDist = double.MaxValue;
            for (int k = 0; k < _numQuantiles; k++)
            {
                double modelQ = (k + 1.0) / (_numQuantiles + 1.0);
                double dist = Math.Abs(modelQ - quantiles[q]);
                if (dist < bestDist) { bestDist = dist; bestIdx = k; }
            }
            for (int t = 0; t < _forecastHorizon && t * _numQuantiles + bestIdx < rawOutput.Length; t++)
                result.Data.Span[t * quantiles.Length + q] = rawOutput[t * _numQuantiles + bestIdx];
        }
        return result;
    }

    /// <inheritdoc/>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // Chronos-Bolt is non-autoregressive — direct multi-step output, truncated to requested steps
        var fullForecast = Forecast(input, null);
        if (steps >= fullForecast.Length) return fullForecast;

        var result = new Tensor<T>(new[] { steps });
        for (int i = 0; i < steps; i++)
            result.Data.Span[i] = fullForecast[i];
        return result;
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
    {
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            T mean = NumOps.Zero;
            for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) mean = NumOps.Add(mean, input[idx]); }
            mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen));
            T variance = NumOps.Zero;
            for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) { var diff = NumOps.Subtract(input[idx], mean); variance = NumOps.Add(variance, NumOps.Multiply(diff, diff)); } }
            variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen));
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));
            for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length && idx < result.Length) result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std); }
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
            ["NumQuantiles"] = NumOps.FromDouble(_numQuantiles),
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

        if (_patchEmbedding is not null)
            current = _patchEmbedding.Forward(current);

        foreach (var layer in _encoderLayers)
            current = layer.Forward(current);

        foreach (var layer in _decoderLayers)
            current = layer.Forward(current);

        if (_quantileHead is not null)
            current = _quantileHead.Forward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });
        return current;
    }

    private Tensor<T> BackwardNative(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        bool addedBatchDim = false;
        if (current.Rank == 1) { current = current.Reshape(new[] { 1, current.Length }); addedBatchDim = true; }
        for (int i = Layers.Count - 1; i >= 0; i--)
            current = Layers[i].Backward(current);
        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });
        return current;
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
        long total = (long)_patchLength * _encoderHiddenDim + _encoderHiddenDim;

        // Encoder layers
        long perEncLayer = 4L * _encoderHiddenDim * _encoderHiddenDim + 4 * _encoderHiddenDim;
        perEncLayer += 2L * _encoderHiddenDim * (_encoderHiddenDim * 4) + _encoderHiddenDim + _encoderHiddenDim * 4;
        perEncLayer += 4L * _encoderHiddenDim;
        total += perEncLayer * _numEncoderLayers;

        // Decoder layers
        long perDecLayer = 4L * _decoderHiddenDim * _decoderHiddenDim + 4 * _decoderHiddenDim;
        perDecLayer += 4L * _decoderHiddenDim * _encoderHiddenDim; // cross-attention
        perDecLayer += 2L * _decoderHiddenDim * (_decoderHiddenDim * 4) + _decoderHiddenDim + _decoderHiddenDim * 4;
        perDecLayer += 6L * _decoderHiddenDim; // 3 layer norms
        total += perDecLayer * _numDecoderLayers;

        // Quantile head
        total += (long)_decoderHiddenDim * _forecastHorizon * _numQuantiles;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
