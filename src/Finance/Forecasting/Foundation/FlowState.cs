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
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// FlowState — IBM's SSM-based Time Series Foundation Model (9.1M parameters).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FlowState is IBM's State-Space Model based time series foundation model. Despite having
/// only 9.1M parameters (smallest in GIFT-Eval top 10), it outperforms models 20x its size
/// and generalizes to unseen timescales. It uses structured state spaces for linear-time
/// processing of long sequences.
/// </para>
/// <para>
/// <b>Reference:</b> IBM Research, "SSM Time Series Model", 2025.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class FlowState<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly List<ILayer<T>> _ssmLayers = [];
    private ILayer<T>? _inputProjection;
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly FlowStateOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _stateDimension;
    private int _hiddenDimension;
    private int _numLayers;
    private double _dropout;
    private int _ssmRank;
    private bool _useDiscretization;
    private FoundationModelSize _modelSize;

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
    public override FoundationModelSize ModelSize => _modelSize;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a FlowState model using a pretrained ONNX model.
    /// </summary>
    public FlowState(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FlowStateOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new FlowStateOptions<T>();
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
    /// Creates a FlowState model in native mode.
    /// </summary>
    public FlowState(
        NeuralNetworkArchitecture<T> architecture,
        FlowStateOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new FlowStateOptions<T>();
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

    private void CopyOptionsToFields(FlowStateOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _stateDimension = options.StateDimension;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
        _ssmRank = options.SSMRank;
        _useDiscretization = options.UseDiscretization;
        _modelSize = options.ModelSize;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFlowStateLayers(
                Architecture, _contextLength, _forecastHorizon,
                _stateDimension, _hiddenDimension, _numLayers, _dropout,
                _ssmRank, _useDiscretization));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        if (idx < Layers.Count)
            _inputProjection = Layers[idx++];

        _ssmLayers.Clear();
        int layersPerBlock = _dropout > 0 ? 5 : 4;
        int totalSSMLayers = _numLayers * layersPerBlock;

        for (int i = 0; i < totalSSMLayers && idx < Layers.Count; i++)
            _ssmLayers.Add(Layers[idx++]);

        if (idx < Layers.Count)
            _outputProjection = Layers[idx++];
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "FlowState" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "StateDimension", _stateDimension },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
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
        return new FlowState<T>(Architecture, new FlowStateOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            StateDimension = _stateDimension,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            DropoutRate = _dropout,
            SSMRank = _ssmRank,
            UseDiscretization = _useDiscretization,
            ModelSize = _modelSize
        });
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_stateDimension);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_dropout);
        writer.Write(_ssmRank);
        writer.Write(_useDiscretization);
        writer.Write((int)_modelSize);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _stateDimension = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _ssmRank = reader.ReadInt32();
        _useDiscretization = reader.ReadBoolean();
        _modelSize = (FoundationModelSize)reader.ReadInt32();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("FlowState does not support quantile forecasting. Pass null for point forecasts.");

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
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
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
            ["StateDimension"] = NumOps.FromDouble(_stateDimension),
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

        if (_inputProjection is not null)
            current = _inputProjection.Forward(current);

        foreach (var layer in _ssmLayers)
            current = layer.Forward(current);

        if (_outputProjection is not null)
            current = _outputProjection.Forward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });
        return current;
    }

    private Tensor<T> BackwardNative(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        bool addedBatchDim = false;
        if (current.Rank == 1) { current = current.Reshape(new[] { 1, current.Length }); addedBatchDim = true; }

        if (_outputProjection is not null)
            current = _outputProjection.Backward(current);

        for (int i = _ssmLayers.Count - 1; i >= 0; i--)
            current = _ssmLayers[i].Backward(current);

        if (_inputProjection is not null)
            current = _inputProjection.Backward(current);

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
        // Input projection
        long total = (long)_contextLength * _hiddenDimension + _hiddenDimension;

        // SSM layers: each has A, B, C, D matrices + norm
        long perLayer = (long)_stateDimension * _stateDimension; // A
        perLayer += (long)_stateDimension * _hiddenDimension; // B
        perLayer += (long)_hiddenDimension * _stateDimension; // C
        perLayer += _hiddenDimension; // D
        perLayer += 2L * _hiddenDimension; // norm
        total += perLayer * _numLayers;

        // Output projection
        total += (long)_hiddenDimension * _forecastHorizon + _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
