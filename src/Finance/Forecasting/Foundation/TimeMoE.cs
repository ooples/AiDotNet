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
/// Time-MoE — Billion-Scale Time Series Foundation Models with Mixture of Experts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Time-MoE is the first billion-scale time series foundation model, using sparse Mixture
/// of Experts for efficient scaling up to 2.4B total parameters (~300M active per token).
/// It uses a decoder-only transformer where each feed-forward layer is replaced by an
/// MoE layer with a learned router.
/// </para>
/// <para><b>For Beginners:</b> Time-MoE is the first time series model to reach billions of
/// parameters by using a Mixture of Experts approach. Instead of one giant network processing
/// every input, it has many specialized "expert" sub-networks and a router that picks the best
/// 2-3 experts for each data point. This means only a fraction of the parameters are active
/// at any time, making it efficient despite its massive total size.</para>
/// <para>
/// <b>Reference:</b> Shi et al., "Time-MoE: Billion-Scale Time Series Foundation Models
/// with Mixture of Experts", ICLR 2025. https://openreview.net/forum?id=e1wDDFmlVu
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Time-MoE billion-scale foundation model with Mixture of Experts
/// // 2.4B total parameters with only ~300M active per token via sparse expert routing
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with MoE layers and learned expert routing
/// var model = new TimeMoE&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TimeMoE&lt;double&gt;(architecture, "time_moe.onnx");
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
[ResearchPaper("Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts", "https://arxiv.org/abs/2409.16040", Year = 2025, Authors = "Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, Ming Jin")]
public class TimeMoE<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeMoEOptions<T> _options;

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
    private int _numExperts;
    private int _numActiveExperts;

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
    /// Creates a Time-MoE model using a pretrained ONNX model.
    /// </summary>
    public TimeMoE(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeMoEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeMoEOptions<T>();
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
    /// Creates a Time-MoE model in native mode.
    /// </summary>
    public TimeMoE(
        NeuralNetworkArchitecture<T> architecture,
        TimeMoEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeMoEOptions<T>();
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

    private void CopyOptionsToFields(TimeMoEOptions<T> options)
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
        _numExperts = options.NumExperts;
        _numActiveExperts = options.NumActiveExperts;
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
            Guard.Positive(_numExperts, nameof(_numExperts));
            Guard.Positive(_numActiveExperts, nameof(_numActiveExperts));
            Guard.Positive(_numHeads, nameof(_numHeads));
            Guard.Positive(_hiddenDimension, nameof(_hiddenDimension));
            if (_numActiveExperts > _numExperts)
                throw new ArgumentOutOfRangeException(
                    nameof(_numActiveExperts),
                    $"NumActiveExperts ({_numActiveExperts}) must be <= NumExperts ({_numExperts}).");
            if (_hiddenDimension % _numHeads != 0)
                throw new ArgumentException(
                    $"HiddenDimension ({_hiddenDimension}) must be divisible by NumHeads ({_numHeads}).");

            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeMoELayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _hiddenDimension, _numLayers, _numHeads, _intermediateSize,
                _numExperts, _numActiveExperts, _dropout));
        }
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
                { "NetworkType", "TimeMoE" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumExperts", _numExperts },
                { "NumActiveExperts", _numActiveExperts },
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
        return new TimeMoE<T>(Architecture, new TimeMoEOptions<T>
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
            NumExperts = _numExperts,
            NumActiveExperts = _numActiveExperts
        });
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
        writer.Write(_numExperts);
        writer.Write(_numActiveExperts);
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
        _numExperts = reader.ReadInt32();
        _numActiveExperts = reader.ReadInt32();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// TimeMoE produces point forecasts only. The <paramref name="quantiles"/> parameter
    /// is accepted for interface compatibility but is not used.
    /// </remarks>
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
            ["NumExperts"] = NumOps.FromDouble(_numExperts),
            ["NumActiveExperts"] = NumOps.FromDouble(_numActiveExperts),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward

    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        // Time-MoE (Shi et al., 2024) is a decoder-only GPT-style transformer
        // whose per-patch tokens feed a stack of self-attention + FFN blocks,
        // then a flatten + linear forecast head. The helper emits a flat,
        // sequentially-composable Layers list (ReshapeLayer → DenseLayer
        // (patch embed) → N × TransformerEncoderLayer (+ optional DropoutLayer)
        // → FlattenLayer → DenseLayer (forecast head)), so ForwardNative is a
        // straight sequential dispatch. Causal masking for the decoder-only
        // semantics is applied by the attention block's own mask config, not
        // at this orchestration layer.
        var current = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        foreach (var layer in Layers)
            current = layer.Forward(current);

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
        // Use the first input name from the ONNX model, falling back to "input"
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
        // Matches the paper-architecture helper: patch embed + N TimeMoE
        // blocks (attention + MoE-FFN with numExperts experts + router +
        // layer norms) + flatten + forecast head. Per-block:
        //   attention Q/K/V/O      = 4 · H² + 4 · H
        //   MoE experts (dense FFN) = numExperts · (2·H·I + H + I)
        //   MoE router              = H · numExperts + numExperts (weight + bias)
        //   layer norms (2 pre-norm) = 4 · H
        int numPatches = _contextLength / _patchLength;
        long total = (long)_patchLength * _hiddenDimension + _hiddenDimension;

        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension; // QKV + out
        perLayer += (long)_numExperts * (2L * _hiddenDimension * _intermediateSize + _hiddenDimension + _intermediateSize); // MoE experts
        perLayer += (long)_hiddenDimension * _numExperts + _numExperts; // router weight + bias
        perLayer += 4L * _hiddenDimension; // 2 pre-norm layers
        total += perLayer * _numLayers;

        total += (long)numPatches * _hiddenDimension * _forecastHorizon + _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
