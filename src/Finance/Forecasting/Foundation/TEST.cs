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
/// TEST — Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TEST generates text-prototype-aligned embeddings for time series by leveraging pretrained
/// language model knowledge. It translates seasonal/trend patterns into text descriptions
/// and aligns time series embeddings to these text prototypes via contrastive learning.
/// </para>
/// <para><b>For Beginners:</b> TEST bridges the gap between language models and time series
/// by creating text descriptions of temporal patterns (like "rising trend with weekly
/// seasonality") and teaching the model to align numerical patterns with these descriptions.
/// This lets a language model understand time series without converting numbers to text,
/// combining the best of both worlds.</para>
/// <para>
/// <b>Reference:</b> Sun et al., "TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series", 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TEST model that aligns time series embeddings with text prototypes
/// // Bridges language models and time series via contrastive learning on pattern descriptions
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with text-prototype alignment
/// var model = new TEST&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TEST&lt;double&gt;(architecture, "test_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class TEST<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _patchEmbedding;
    private readonly List<ILayer<T>> _encoderLayers = [];
    private ILayer<T>? _alignmentProjection;
    private ILayer<T>? _forecastHead;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TESTOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _patchLength;
    private int _hiddenDimension;
    private int _textEmbeddingDimension;
    private int _numLayers;
    private int _numHeads;
    private double _dropout;
    private FoundationModelSize _modelSize;
    private int _numPrototypes;
    private double _alignmentWeight;

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
    /// Creates a TEST model using a pretrained ONNX model.
    /// </summary>
    public TEST(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TESTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TESTOptions<T>();
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
    /// Creates a TEST model in native mode for training or fine-tuning.
    /// </summary>
    public TEST(
        NeuralNetworkArchitecture<T> architecture,
        TESTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TESTOptions<T>();
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

    private void CopyOptionsToFields(TESTOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.PatchLength, nameof(options.PatchLength));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.TextEmbeddingDimension, nameof(options.TextEmbeddingDimension));
        Guard.Positive(options.NumLayers, nameof(options.NumLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));
        Guard.Positive(options.NumPrototypes, nameof(options.NumPrototypes));

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _textEmbeddingDimension = options.TextEmbeddingDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _modelSize = options.ModelSize;
        _numPrototypes = options.NumPrototypes;
        _alignmentWeight = options.AlignmentWeight;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTESTLayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _hiddenDimension, _textEmbeddingDimension, _numLayers, _numHeads,
                _numPrototypes, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        if (idx < Layers.Count)
            _patchEmbedding = Layers[idx++];

        _encoderLayers.Clear();
        int layersPerBlock = _dropout > 0 ? 9 : 7;
        int totalEncoderLayers = _numLayers * layersPerBlock;

        for (int i = 0; i < totalEncoderLayers && idx < Layers.Count; i++)
            _encoderLayers.Add(Layers[idx++]);

        // Alignment projection (hidden -> text embedding space)
        if (idx < Layers.Count)
            _alignmentProjection = Layers[idx++];

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
            BackwardNative(Tensor<T>.FromVector(gradient, output.Shape.ToArray()));

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
                { "NetworkType", "TEST" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "HiddenDimension", _hiddenDimension },
                { "TextEmbeddingDimension", _textEmbeddingDimension },
                { "NumLayers", _numLayers },
                { "NumPrototypes", _numPrototypes },
                { "AlignmentWeight", _alignmentWeight },
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
        var opts = new TESTOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            HiddenDimension = _hiddenDimension,
            TextEmbeddingDimension = _textEmbeddingDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            DropoutRate = _dropout,
            ModelSize = _modelSize,
            NumPrototypes = _numPrototypes,
            AlignmentWeight = _alignmentWeight
        };

        if (!_useNativeMode && OnnxModelPath is not null)
            return new TEST<T>(Architecture, OnnxModelPath, opts);

        return new TEST<T>(Architecture, opts);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_hiddenDimension);
        writer.Write(_textEmbeddingDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_dropout);
        writer.Write((int)_modelSize);
        writer.Write(_numPrototypes);
        writer.Write(_alignmentWeight);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _textEmbeddingDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _modelSize = (FoundationModelSize)reader.ReadInt32();
        _numPrototypes = reader.ReadInt32();
        _alignmentWeight = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("TEST does not support quantile forecasting. Pass null for point forecasts.");

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
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input.Shape.ToArray());

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
            ["NumPrototypes"] = NumOps.FromDouble(_numPrototypes),
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

        // Alignment projection maps to text embedding space
        if (_alignmentProjection is not null)
            current = _alignmentProjection.Forward(current);

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

        if (_alignmentProjection is not null)
            current = _alignmentProjection.Backward(current);

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
        int intermediateSize = _hiddenDimension * 4;
        long total = (long)_patchLength * _hiddenDimension + _hiddenDimension;

        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perLayer += 2L * _hiddenDimension * intermediateSize + _hiddenDimension + intermediateSize;
        perLayer += 4L * _hiddenDimension;
        total += perLayer * _numLayers;

        // Alignment projection
        total += (long)_hiddenDimension * _textEmbeddingDimension + _textEmbeddingDimension;
        // Text prototypes
        total += (long)_numPrototypes * _textEmbeddingDimension;
        // Forecast head
        total += (long)_textEmbeddingDimension * _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
