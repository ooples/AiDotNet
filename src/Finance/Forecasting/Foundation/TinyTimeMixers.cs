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
/// Tiny Time Mixers (TTM) foundation model for compact, high-performance time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// TTM is IBM Research's lightweight foundation model that uses an MLP-Mixer architecture
/// instead of attention-based transformers. With only 1-5 million parameters, it outperforms
/// or matches models 20-40x its size on standard forecasting benchmarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> TTM proves that bigger isn't always better:
///
/// <b>MLP-Mixer Architecture:</b>
/// Instead of expensive attention mechanisms, TTM alternates between two types of mixing:
/// 1. <b>Temporal Mixing:</b> An MLP that mixes information across time steps within each feature.
///    Think of it as learning "how does the pattern at time t relate to time t-1, t-2, etc.?"
/// 2. <b>Channel Mixing:</b> An MLP that mixes information across features at each time step.
///    Think of it as learning "how does price relate to volume at this moment?"
///
/// <b>Why So Small Yet Effective:</b>
/// - Time series have simpler structure than language or images
/// - MLP-Mixers capture temporal patterns efficiently without attention overhead
/// - Patch-based input reduces the sequence length the model needs to process
/// - Careful architectural choices maximize information per parameter
///
/// <b>Practical Benefits:</b>
/// - Runs on CPU in real-time (no GPU required)
/// - Trains in minutes instead of hours
/// - Perfect for edge deployment (IoT, mobile)
/// - Low memory footprint (~20MB model size)
/// </para>
/// <para>
/// <b>Reference:</b> Ekambaram et al., "Tiny Time Mixers (TTMs): Fast Pre-trained Models
/// for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series", NeurIPS 2024.
/// https://arxiv.org/abs/2401.03955
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is NOT thread-safe. Create separate instances for concurrent usage.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Tiny Time Mixers (TTM) for compact, high-performance forecasting
/// // MLP-Mixer architecture with only 1-5M parameters that rivals 20-40x larger models
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 7, inputDepth: 1, outputSize: 24);
///
/// // Training mode with temporal and channel mixing MLPs
/// var model = new TinyTimeMixers&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TinyTimeMixers&lt;double&gt;(architecture, "ttm_base.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series", "https://arxiv.org/abs/2401.03955", Year = 2024, Authors = "Vijay Ekambaram, Arindam Jati, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam")]
public class TinyTimeMixers<T> : TimeSeriesFoundationModelBase<T>
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
    /// Mixer layers (alternating temporal-mixing and channel-mixing blocks).
    /// </summary>
    private readonly List<ILayer<T>> _mixerLayers = [];

    /// <summary>
    /// Final layer normalization before the output head.
    /// </summary>
    private ILayer<T>? _finalLayerNorm;

    /// <summary>
    /// Output projection head that maps mixer output to forecast horizon.
    /// </summary>
    private ILayer<T>? _outputHead;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TinyTimeMixersOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _patchLength;
    private int _hiddenDimension;
    private int _numMixerLayers;
    private int _expansionFactor;
    private double _dropout;
    private FoundationModelSize _modelSize;
    private bool? _useAdaptivePatching;
    private int _numFeatures;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => _patchLength;

    /// <inheritdoc/>
    public override int Stride => _patchLength;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false; // TTM uses channel mixing

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
    /// Creates a Tiny Time Mixers model using a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Load pretrained TTM weights for fast zero-shot forecasting.
    /// </para>
    /// </remarks>
    public TinyTimeMixers(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TinyTimeMixersOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TinyTimeMixersOptions<T>();
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
    /// Creates a Tiny Time Mixers model in native mode for training or fine-tuning.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for fine-tuning or training from scratch.
    /// TTM trains very quickly — minutes instead of hours for larger models.
    /// </para>
    /// </remarks>
    public TinyTimeMixers(
        NeuralNetworkArchitecture<T> architecture,
        TinyTimeMixersOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TinyTimeMixersOptions<T>();
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

    private void CopyOptionsToFields(TinyTimeMixersOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numMixerLayers = options.NumMixerLayers;
        _expansionFactor = options.ExpansionFactor;
        _dropout = options.DropoutRate;
        _modelSize = options.ModelSize;
        _useAdaptivePatching = options.UseAdaptivePatching;
        _numFeatures = options.NumFeatures;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTinyTimeMixersLayers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _patchLength, _hiddenDimension, _numMixerLayers, _expansionFactor,
                _dropout));

            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Patch embedding
        if (idx < Layers.Count)
            _patchEmbedding = Layers[idx++];

        // Mixer layers (each block: temporal-mix MLP [expand+contract+dropout] + channel-mix MLP [expand+contract+dropout])
        _mixerLayers.Clear();
        int layersPerBlock = _dropout > 0 ? 6 : 4; // temporal(2) + channel(2) + optional dropouts
        int totalMixerLayers = _numMixerLayers * layersPerBlock;

        for (int i = 0; i < totalMixerLayers && idx < Layers.Count - 2; i++)
        {
            _mixerLayers.Add(Layers[idx++]);
        }

        // Final layer norm
        if (idx < Layers.Count)
            _finalLayerNorm = Layers[idx++];

        // Output head
        if (idx < Layers.Count)
            _outputHead = Layers[idx];
    }

    /// <inheritdoc/>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "TinyTimeMixers requires at least 3 layers (patch embed, mixer, output head).",
                nameof(layers));
        }
    }

    private static void ValidateOptions(TinyTimeMixersOptions<T> options)
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
        if (options.NumMixerLayers < 1)
            errors.Add("NumMixerLayers must be at least 1.");
        if (options.ExpansionFactor < 1)
            errors.Add("ExpansionFactor must be at least 1.");
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            errors.Add("DropoutRate must be between 0 and 1 (exclusive).");
        if (options.NumFeatures < 1)
            errors.Add("NumFeatures must be at least 1.");

        if (errors.Count > 0)
            throw new ArgumentException($"Invalid TinyTimeMixers options: {string.Join(", ", errors)}");
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
            BackwardNative(Tensor<T>.FromVector(gradient, output.Shape._dims));

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
                { "NetworkType", "TinyTimeMixers" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "HiddenDimension", _hiddenDimension },
                { "NumMixerLayers", _numMixerLayers },
                { "ExpansionFactor", _expansionFactor },
                { "ModelSize", _modelSize.ToString() },
                { "UseNativeMode", _useNativeMode },
                { "NumFeatures", _numFeatures },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TinyTimeMixersOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            HiddenDimension = _hiddenDimension,
            NumMixerLayers = _numMixerLayers,
            ExpansionFactor = _expansionFactor,
            DropoutRate = _dropout,
            ModelSize = _modelSize,
            UseAdaptivePatching = _useAdaptivePatching,
            NumFeatures = _numFeatures
        };

        return new TinyTimeMixers<T>(Architecture, options);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_hiddenDimension);
        writer.Write(_numMixerLayers);
        writer.Write(_expansionFactor);
        writer.Write(_dropout);
        writer.Write((int)_modelSize);
        writer.Write(_useAdaptivePatching.HasValue);
        if (_useAdaptivePatching.HasValue)
            writer.Write(_useAdaptivePatching.Value);
        writer.Write(_numFeatures);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numMixerLayers = reader.ReadInt32();
        _expansionFactor = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _modelSize = (FoundationModelSize)reader.ReadInt32();
        bool hasAdaptive = reader.ReadBoolean();
        _useAdaptivePatching = hasAdaptive ? reader.ReadBoolean() : null;
        _numFeatures = reader.ReadInt32();
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
        // RevIN (Reversible Instance Normalization)
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input.Shape._dims);

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
            ["NumMixerLayers"] = NumOps.FromDouble(_numMixerLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the full native forward pass through the TTM MLP-Mixer architecture.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var current = ApplyInstanceNormalization(input);

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

        // Mixer layers (temporal-mixing + channel-mixing blocks)
        foreach (var layer in _mixerLayers)
        {
            current = layer.Forward(current);
        }

        // Final layer norm
        if (_finalLayerNorm is not null)
            current = _finalLayerNorm.Forward(current);

        // Output head
        if (_outputHead is not null)
            current = _outputHead.Forward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
        {
            current = current.Reshape(new[] { current.Shape[1] });
        }

        return current;
    }

    /// <summary>
    /// Performs the backward pass through the TTM architecture.
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

        // Output head backward
        if (_outputHead is not null)
            current = _outputHead.Backward(current);

        // Final norm backward
        if (_finalLayerNorm is not null)
            current = _finalLayerNorm.Backward(current);

        // Mixer layers backward (reverse order)
        for (int i = _mixerLayers.Count - 1; i >= 0; i--)
        {
            current = _mixerLayers[i].Backward(current);
        }

        // Patch embedding backward
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

        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;

        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++)
        {
            inputData[i] = (float)NumOps.ToDouble(input[i]);
        }

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
        int patchInputSize = _patchLength * _numFeatures;
        int expandedDim = _hiddenDimension * _expansionFactor;

        // Patch embedding
        long total = (long)patchInputSize * _hiddenDimension + _hiddenDimension;

        // Mixer layers (per block: temporal MLP + channel MLP)
        long perBlock = 2L * numPatches * expandedDim + expandedDim + numPatches; // temporal
        perBlock += 2L * _hiddenDimension * expandedDim + expandedDim + _hiddenDimension; // channel
        total += perBlock * _numMixerLayers;

        // Final norm + output head
        total += 2L * _hiddenDimension; // norm
        total += (long)numPatches * _hiddenDimension * _forecastHorizon; // output

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
