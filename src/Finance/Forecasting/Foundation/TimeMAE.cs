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
/// TimeMAE — Masked Autoencoder for Time Series with Decoupled Masked Autoencoders.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeMAE applies masked autoencoding to time series, randomly masking patches of the input
/// and training a transformer to reconstruct the missing patches, learning rich temporal representations.
/// It uses an asymmetric encoder-decoder architecture where the encoder processes visible patches only.
/// </para>
/// <para><b>For Beginners:</b> TimeMAE learns about time series by hiding random chunks of
/// data and training a model to fill them back in, similar to how a student learns vocabulary
/// by doing fill-in-the-blank exercises. The model only processes the visible chunks with a
/// large encoder and uses a smaller decoder to reconstruct the hidden ones, making training
/// very efficient. The learned representations transfer well to forecasting tasks.</para>
/// <para>
/// <b>Reference:</b> Cheng et al., "TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders", 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TimeMAE masked autoencoder for time series representation learning
/// // Asymmetric encoder-decoder learns by reconstructing randomly masked patches
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with masked patch reconstruction
/// var model = new TimeMAE&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TimeMAE&lt;double&gt;(architecture, "timemae.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class TimeMAE<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _patchEmbedding;
    private readonly List<ILayer<T>> _encoderLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];
    private ILayer<T>? _reconstructionHead;
    private ILayer<T>? _forecastHead;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeMAEOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _patchLength;
    private int _hiddenDimension;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _numHeads;
    private double _maskRatio;
    private double _dropout;

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
    public override FoundationModelSize ModelSize => FoundationModelSize.Base;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TimeMAE model using a pretrained ONNX model.
    /// </summary>
    public TimeMAE(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeMAEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeMAEOptions<T>();
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
    /// Creates a TimeMAE model in native mode for training or fine-tuning.
    /// </summary>
    public TimeMAE(
        NeuralNetworkArchitecture<T> architecture,
        TimeMAEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeMAEOptions<T>();
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

    private void CopyOptionsToFields(TimeMAEOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.PatchLength, nameof(options.PatchLength));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumEncoderLayers, nameof(options.NumEncoderLayers));
        Guard.Positive(options.NumDecoderLayers, nameof(options.NumDecoderLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));

        if (options.MaskRatio <= 0.0 || options.MaskRatio >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(options), $"MaskRatio must be between 0 and 1 (exclusive) but was {options.MaskRatio}.");

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumHeads;
        _maskRatio = options.MaskRatio;
        _dropout = options.DropoutRate;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeMAELayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _hiddenDimension, _numEncoderLayers, _numDecoderLayers, _numHeads, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        int layersPerBlock = _dropout > 0 ? 9 : 7;

        if (idx < Layers.Count)
            _patchEmbedding = Layers[idx++];

        _encoderLayers.Clear();
        int totalEncoderLayers = _numEncoderLayers * layersPerBlock;
        for (int i = 0; i < totalEncoderLayers && idx < Layers.Count; i++)
            _encoderLayers.Add(Layers[idx++]);

        _decoderLayers.Clear();
        int totalDecoderLayers = _numDecoderLayers * layersPerBlock;
        for (int i = 0; i < totalDecoderLayers && idx < Layers.Count; i++)
            _decoderLayers.Add(Layers[idx++]);

        if (idx < Layers.Count)
            _reconstructionHead = Layers[idx++];

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
                { "NetworkType", "TimeMAE" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "HiddenDimension", _hiddenDimension },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "MaskRatio", _maskRatio },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var opts = new TimeMAEOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            HiddenDimension = _hiddenDimension,
            NumEncoderLayers = _numEncoderLayers,
            NumDecoderLayers = _numDecoderLayers,
            NumHeads = _numHeads,
            MaskRatio = _maskRatio,
            DropoutRate = _dropout
        };

        // Preserve ONNX mode if the original instance was created with an ONNX model
        if (!_useNativeMode && OnnxModelPath is not null)
            return new TimeMAE<T>(Architecture, OnnxModelPath, opts);

        return new TimeMAE<T>(Architecture, opts);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_hiddenDimension);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_maskRatio);
        writer.Write(_dropout);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numEncoderLayers = reader.ReadInt32();
        _numDecoderLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _maskRatio = reader.ReadDouble();
        _dropout = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("TimeMAE does not support quantile forecasting. Pass null for point forecasts.");

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
        if (input.Length == 0)
            return new Tensor<T>(input.Shape);

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
            ["PatchLength"] = NumOps.FromDouble(_patchLength),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumEncoderLayers"] = NumOps.FromDouble(_numEncoderLayers),
            ["NumDecoderLayers"] = NumOps.FromDouble(_numDecoderLayers),
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

        if (_patchEmbedding is not null)
            current = _patchEmbedding.Forward(current);

        foreach (var layer in _encoderLayers)
            current = layer.Forward(current);

        foreach (var layer in _decoderLayers)
            current = layer.Forward(current);

        if (_reconstructionHead is not null)
            current = _reconstructionHead.Forward(current);

        if (_forecastHead is not null)
            current = _forecastHead.Forward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });

        return current;
    }

    private Tensor<T> BackwardNative(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        if (_forecastHead is not null)
            current = _forecastHead.Backward(current);

        if (_reconstructionHead is not null)
            current = _reconstructionHead.Backward(current);

        for (int i = _decoderLayers.Count - 1; i >= 0; i--)
            current = _decoderLayers[i].Backward(current);

        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
            current = _encoderLayers[i].Backward(current);

        if (_patchEmbedding is not null)
            current = _patchEmbedding.Backward(current);

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

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
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
        long total = (long)_patchLength * _hiddenDimension + _hiddenDimension;

        long perEncoderLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perEncoderLayer += 2L * _hiddenDimension * (_hiddenDimension * 4) + _hiddenDimension + (_hiddenDimension * 4);
        perEncoderLayer += 4L * _hiddenDimension;
        total += perEncoderLayer * _numEncoderLayers;

        long perDecoderLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perDecoderLayer += 2L * _hiddenDimension * (_hiddenDimension * 4) + _hiddenDimension + (_hiddenDimension * 4);
        perDecoderLayer += 4L * _hiddenDimension;
        total += perDecoderLayer * _numDecoderLayers;

        total += (long)_hiddenDimension * _patchLength + _patchLength;
        total += (long)numPatches * _patchLength * _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
