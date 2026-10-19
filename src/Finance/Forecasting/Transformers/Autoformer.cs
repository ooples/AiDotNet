using AiDotNet.LearningRateSchedulers;
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
using AiDotNet.TimeSeries;
namespace AiDotNet.Finance.Forecasting.Transformers;

/// <summary>
/// Autoformer (Decomposition Transformers with Auto-Correlation) neural network for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Autoformer replaces the traditional attention mechanism with auto-correlation, which finds
/// period-based dependencies efficiently using FFT. It also progressively decomposes the series
/// into trend and seasonal components at each layer.
/// </para>
/// <para>
/// <b>For Beginners:</b> Autoformer is like a music analyst who finds repeating patterns
/// by checking if the melody matches itself at different time delays. Instead of comparing
/// every note to every other note (O(L²)), it uses FFT to find these patterns in O(L log L) time.
///
/// Key innovations:
/// - <b>Auto-Correlation:</b> Finds periodic patterns using FFT instead of attention
/// - <b>Series Decomposition:</b> Separates trend from seasonal patterns at each layer
/// - <b>Progressive Decomposition:</b> Decomposition happens multiple times for accuracy
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation",
/// NeurIPS 2021. https://arxiv.org/abs/2106.13008
/// </para>
/// <para>
/// The network is the official code's (<see cref="AiDotNet.TimeSeries.AutoformerNetwork{T}"/>, shared with
/// <see cref="AiDotNet.TimeSeries.AutoformerModel{T}"/> and checked against thuml/Autoformer), forecasting every input
/// series jointly. Standardize the data with training statistics first, as the reference does. The temporal embedding
/// needs timestamps, which this interface does not carry, so it is not used.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an Autoformer for long-term financial time series forecasting
/// // with auto-correlation attention and series decomposition
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 7, outputSize: 7);   // seven series (ETTh1), forecast jointly
///
/// // Training mode: [batch, 96, 7] in, [batch, 96, 7] out
/// var model = new Autoformer&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new Autoformer&lt;double&gt;(architecture, "autoformer_etth1.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", "https://arxiv.org/abs/2106.13008", Year = 2021, Authors = "Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, ReferenceBatchSize = 32,
                Schedule = LearningRateSchedulerType.Exponential, DecayRate = 0.5,
                ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                Source = "Wu et al. 2021, Implementation details: Adam with an initial learning rate of 1e-4 and batch size 32; the official code (lradj 'type1') halves the rate after every epoch.")]
[TaskVariantOf(typeof(AiDotNet.TimeSeries.AutoformerModel<>), "financial forecasting")]
public partial class Autoformer<T> : ForecastingModelBase<T>
{
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly AutoformerOptions<T> _options;
    private readonly int _numFeatures;
    private readonly Random _random;
    private AutoformerNetwork<T>? _network;

    public override ModelOptions GetOptions() => _options;

    #region IForecastingModel Properties

    public override int SequenceLength => _options.LookbackWindow;

    public override int PredictionHorizon => _options.ForecastHorizon;

    public override int NumFeatures => _numFeatures;

    public override int PatchSize => 1;

    public override int Stride => 1;

    public override bool IsChannelIndependent => false;

    public override bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an Autoformer with the reference's defaults for a single series.
    /// </summary>
    public Autoformer()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 1,
            outputSize: 1))
    {
    }

    /// <summary>
    /// Creates an Autoformer that runs a pre-trained ONNX model.
    /// </summary>
    /// <param name="architecture">The architecture; its input size is the number of series (channels).</param>
    /// <param name="onnxModelPath">The ONNX model file.</param>
    /// <param name="options">The Autoformer options.</param>
    /// <param name="optimizer">Unused in ONNX mode; accepted for symmetry.</param>
    /// <param name="lossFunction">The loss reported by evaluation (mean squared error by default).</param>
    public Autoformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        AutoformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model file not found: {onnxModelPath}", onnxModelPath);
        _options = options ?? new AutoformerOptions<T>();
        Options = _options;
        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;
        _numFeatures = architecture.InputSize;
        _random = RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                { InitialLearningRate = _options.LearningRate });
    }

    /// <summary>
    /// Creates a trainable Autoformer.
    /// </summary>
    /// <param name="architecture">The architecture; its input size is the number of series (channels), which the
    /// model forecasts jointly (the reference's <c>features='M'</c>).</param>
    /// <param name="options">The Autoformer options.</param>
    /// <param name="optimizer">The optimizer; the paper's Adam with the per-epoch halving when null.</param>
    /// <param name="lossFunction">The training loss (mean squared error by default, as in the reference).</param>
    public Autoformer(
        NeuralNetworkArchitecture<T> architecture,
        AutoformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _options = options ?? new AutoformerOptions<T>();
        Options = _options;
        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;
        _numFeatures = architecture.InputSize;
        _random = RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                { InitialLearningRate = _options.LearningRate });
        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc />
    /// <remarks>
    /// Builds the shared <see cref="AutoformerNetwork{T}"/> (the official code's model) and initializes it as the
    /// reference does; every one of its layers is in <c>Layers</c>.
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            throw new ArgumentException(
                "Autoformer builds its own encoder-decoder network; custom layers in the architecture are not supported.");
        var o = _options;
        var layers = new List<LayerBase<T>>();
        _network = new AutoformerNetwork<T>(Engine, layers, new AutoformerConfiguration(
            _numFeatures, _numFeatures, _numFeatures, o.EmbeddingDim, o.NumAttentionHeads, o.NumEncoderLayers,
            o.NumDecoderLayers, o.FeedForwardDim, o.MovingAverageKernel, o.AutoCorrelationFactor, o.DropoutRate, o.UseGelu,
            TimeFeatures: 0, o.LabelLength, o.ForecastHorizon));
        _network.InitializeLikeReference(RandomHelper.CreateSeededRandom(o.Seed ?? 42));
        Layers.AddRange(layers);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc />
    /// <remarks>Auto-Correlation chooses its lags from the current values, so a traced graph replayed on later steps
    /// would keep the first step's lags; training runs the true forward every step.</remarks>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <inheritdoc />
    /// <remarks>The forward is not a fold of <c>Layers</c> (decomposition, attention and the decoder branch off the
    /// list), so the activations are those the forecast actually computes.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
        => _useNativeMode ? ObservedLayerActivations(input) : new Dictionary<string, Tensor<T>>();

    protected override Tensor<T> PredictCore(Tensor<T> input) => Forecast(input);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = _options;
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "Autoformer" },
                { "SequenceLength", o.LookbackWindow },
                { "LabelLength", o.LabelLength },
                { "PredictionHorizon", o.ForecastHorizon },
                { "NumFeatures", _numFeatures },
                { "NumEncoderLayers", o.NumEncoderLayers },
                { "NumDecoderLayers", o.NumDecoderLayers },
                { "NumHeads", o.NumAttentionHeads },
                { "ModelDimension", o.EmbeddingDim },
                { "MovingAverageKernel", o.MovingAverageKernel },
                { "AutoCorrelationFactor", o.AutoCorrelationFactor },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelDataProvider = () => _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        return _useNativeMode ? ForecastNative(input, quantiles) : ForecastOnnx(input);
    }

    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be at least 1.");
        var currentInput = input;
        var allPredictions = new List<Tensor<T>>();
        int stepsRemaining = steps;
        while (stepsRemaining > 0)
        {
            var forecast = Forecast(currentInput);
            int stepsToUse = Math.Min(stepsRemaining, PredictionHorizon);
            allPredictions.Add(forecast);
            stepsRemaining -= stepsToUse;
            if (stepsRemaining > 0)
                currentInput = ShiftInputWithPredictions(currentInput, forecast, stepsToUse);
        }
        return ConcatenatePredictions(allPredictions, steps);
    }

    public override Dictionary<string, T> Evaluate(Tensor<T> inputs, Tensor<T> targets)
    {
        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));
        if (targets is null)
            throw new ArgumentNullException(nameof(targets));
        var predictions = Forecast(inputs);
        T maeSum = NumOps.Zero, mseSum = NumOps.Zero;
        int count = predictions.Length;
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            maeSum = NumOps.Add(maeSum, NumOps.Abs(diff));
            mseSum = NumOps.Add(mseSum, NumOps.Multiply(diff, diff));
        }
        var mse = NumOps.Divide(mseSum, NumOps.FromDouble(count));
        return new Dictionary<string, T>
        {
            ["MAE"] = NumOps.Divide(maeSum, NumOps.FromDouble(count)),
            ["MSE"] = mse,
            ["RMSE"] = NumOps.Sqrt(mse),
        };
    }

    /// <inheritdoc />
    /// <remarks>Autoformer normalizes nothing per instance: the reference standardizes the whole dataset with training
    /// statistics before the model sees it, so the input is returned unchanged.</remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) => input;

    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(SequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(PredictionHorizon),
            ["NumFeatures"] = NumOps.FromDouble(_numFeatures),
            ["ParameterCount"] = NumOps.FromDouble(GetParameterCount())
        };
    }

    #endregion

    #region Forward

    // [L, F] or [B, L, F] history → [B, H, F] forecast (the decoder input comes from the history's decomposition).
    private Tensor<T> Forward(Tensor<T> input, bool training)
    {
        if (_network is null) throw new InvalidOperationException("The native network is not initialized.");
        var history = input.Rank == 2 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
        if (history.Rank != 3 || history.Shape[2] != _numFeatures)
            throw new ArgumentException(
                $"Autoformer expects [batch, length, {_numFeatures}] (or [length, {_numFeatures}]), got [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        return _network.Forward(history, null, null, training, _random);
    }

    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Forward(input, training: false);
    }

    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input) => Forward(input, training: true);

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");
        var inputName = OnnxSession.InputMetadata.Keys.First();
        var onnxInput = new OnnxTensors.DenseTensor<float>(
            input.ToArray().Select(x => Convert.ToSingle(x)).ToArray(),
            input._shape.ToArray());
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = OnnxSession.Run(inputs);
        var output = results.First().AsTensor<float>();
        var outputArray = output.ToArray().Select(x => NumOps.FromDouble(x)).ToArray();
        return new Tensor<T>(outputArray, output.Dimensions.ToArray());
    }

    #endregion

    #region Helper Methods

    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int steps)
    {
        int batchSize = input.Shape[0], length = input.Shape[1], features = input.Shape[2];
        var newInput = new Tensor<T>(new[] { batchSize, length, features });
        int keepLen = length - steps;
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < keepLen; t++)
                for (int f = 0; f < features; f++) newInput[b, t, f] = input[b, t + steps, f];
            for (int t = 0; t < steps; t++)
                for (int f = 0; f < features; f++) newInput[b, keepLen + t, f] = predictions[b, t, f];
        }
        return newInput;
    }

    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps, _numFeatures });
        int batchSize = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batchSize, totalSteps, _numFeatures });
        int currentStep = 0;
        foreach (var pred in predictions)
        {
            int stepsToAdd = Math.Min(pred.Shape[1], totalSteps - currentStep);
            for (int b = 0; b < batchSize; b++)
                for (int t = 0; t < stepsToAdd; t++)
                    for (int f = 0; f < _numFeatures; f++) result[b, currentStep + t, f] = pred[b, t, f];
            currentStep += stepsToAdd;
        }
        return result;
    }

    #endregion

    #region IDisposable

    protected override void Dispose(bool disposing)
    {
        if (disposing) OnnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
