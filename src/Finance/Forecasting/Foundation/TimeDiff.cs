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
/// TimeDiff — Non-autoregressive Conditional Diffusion Models for Time Series Prediction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeDiff extends DDPM with future-mixup training augmentation and autoregressive initialization
/// at inference for high-quality non-autoregressive time series forecasting.
/// </para>
/// <para><b>For Beginners:</b> TimeDiff improves diffusion-based forecasting with two clever
/// tricks. During training, it mixes future values into the input (future-mixup) to help the
/// model learn what comes next. During prediction, it uses an initial rough forecast to guide
/// the diffusion process, producing all future values at once rather than one at a time, which
/// is both faster and more consistent.</para>
/// <para>
/// <b>Reference:</b> Shen &amp; Kwok, "Non-autoregressive Conditional Diffusion Models for Time Series Prediction", ICML 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TimeDiff non-autoregressive conditional diffusion model
/// // Uses future-mixup training and autoregressive initialization for consistent forecasts
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with future-mixup augmentation
/// var model = new TimeDiff&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TimeDiff&lt;double&gt;(architecture, "timediff.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Non-autoregressive Conditional Diffusion Models for Time Series Prediction", "https://arxiv.org/abs/2306.05043", Year = 2023, Authors = "Lifeng Shen, James Kwok")]
public class TimeDiff<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _transformerLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeDiffOptions<T> _options;

    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _diffusionSteps;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;
    private bool _useFutureMixup;
    private bool _useAutoregressiveInit;

    // DDPM noise schedule (precomputed as generic vectors)
    private Vector<T> _betas = Vector<T>.Empty();
    private Vector<T> _alphas = Vector<T>.Empty();
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();

    #endregion

    #region Properties

    public override int SequenceLength => _contextLength;
    public override int PredictionHorizon => _forecastHorizon;
    public override int NumFeatures => 1;
    public override int PatchSize => 1;
    public override int Stride => 1;
    public override bool IsChannelIndependent => true;
    public override bool UseNativeMode => _useNativeMode;
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    public override int MaxContextLength => _contextLength;
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    public TimeDiff(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        TimeDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeDiffOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
    }

    public TimeDiff(NeuralNetworkArchitecture<T> architecture,
        TimeDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeDiffOptions<T>();
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

    private void CopyOptionsToFields(TimeDiffOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _diffusionSteps = options.DiffusionSteps;
        _dropout = options.DropoutRate;
        _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd;
        _useFutureMixup = options.UseFutureMixup;
        _useAutoregressiveInit = options.UseAutoregressiveInit;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        if (_diffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(_diffusionSteps), "DiffusionSteps must be positive.");

        _betas = new Vector<T>(_diffusionSteps);
        _alphas = new Vector<T>(_diffusionSteps);
        _alphasCumprod = new Vector<T>(_diffusionSteps);
        _sqrtAlphasCumprod = new Vector<T>(_diffusionSteps);
        _sqrtOneMinusAlphasCumprod = new Vector<T>(_diffusionSteps);
        T one = NumOps.One;
        T betaStartT = NumOps.FromDouble(_betaStart);
        T betaRangeT = NumOps.FromDouble(_betaEnd - _betaStart);
        T maxDenom = NumOps.FromDouble(Math.Max(1, _diffusionSteps - 1));
        for (int t = 0; t < _diffusionSteps; t++)
        {
            _betas[t] = NumOps.Add(betaStartT, NumOps.Divide(NumOps.Multiply(betaRangeT, NumOps.FromDouble(t)), maxDenom));
            _alphas[t] = NumOps.Subtract(one, _betas[t]);
        }
        _alphasCumprod[0] = _alphas[0];
        for (int t = 1; t < _diffusionSteps; t++)
            _alphasCumprod[t] = NumOps.Multiply(_alphasCumprod[t - 1], _alphas[t]);
        for (int t = 0; t < _diffusionSteps; t++)
        {
            _sqrtAlphasCumprod[t] = NumOps.Sqrt(_alphasCumprod[t]);
            _sqrtOneMinusAlphasCumprod[t] = NumOps.Sqrt(NumOps.Subtract(one, _alphasCumprod[t]));
        }
    }

    private T SampleStandardNormal(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    #endregion

    #region Initialization

    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); ExtractLayerReferences(); }
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultTimeDiffLayers(Architecture, _contextLength, _forecastHorizon, _hiddenDimension, _numLayers, _numHeads, _dropout)); ExtractLayerReferences(); }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count) _inputProjection = Layers[idx++];
        _transformerLayers.Clear();
        while (idx < Layers.Count - 1) _transformerLayers.Add(Layers[idx++]);
        if (idx < Layers.Count) _outputProjection = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;
    public override Tensor<T> Predict(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is only supported in native mode.");
        SetTrainingMode(true);
        try
        {
            var rand = RandomHelper.CreateSecureRandom();
            int t = rand.Next(_diffusionSteps);

            // DDPM training: add noise to target at timestep t
            var noise = new Tensor<T>(target.Shape._dims);
            for (int i = 0; i < target.Length; i++)
                noise.Data.Span[i] = SampleStandardNormal(rand);

            T sqrtAlphaBar = _sqrtAlphasCumprod[t];
            T sqrtOneMinusAlphaBar = _sqrtOneMinusAlphasCumprod[t];
            var noisyTarget = new Tensor<T>(target.Shape._dims);
            for (int i = 0; i < target.Length; i++)
                noisyTarget.Data.Span[i] = NumOps.Add(
                    NumOps.Multiply(sqrtAlphaBar, target[i]),
                    NumOps.Multiply(sqrtOneMinusAlphaBar, noise[i]));

            // Future-mixup augmentation: blend future ground truth into noisy target
            if (_useFutureMixup)
            {
                T mixRatio = NumOps.FromDouble(rand.NextDouble() * 0.5);
                T oneMinusMix = NumOps.Subtract(NumOps.One, mixRatio);
                for (int i = 0; i < noisyTarget.Length && i < target.Length; i++)
                    noisyTarget.Data.Span[i] = NumOps.Add(
                        NumOps.Multiply(oneMinusMix, noisyTarget[i]),
                        NumOps.Multiply(mixRatio, target[i]));
            }

            var predictedNoise = ForwardTraining(input, noisyTarget, t);
            LastLoss = _lossFunction.CalculateLoss(predictedNoise.ToVector(), noise.ToVector());
            var gradient = _lossFunction.CalculateDerivative(predictedNoise.ToVector(), noise.ToVector());
            BackwardNative(Tensor<T>.FromVector(gradient, predictedNoise.Shape._dims));
            _optimizer.UpdateParameters(Layers);
        }
        finally { SetTrainingMode(false); }
    }

    /// <summary>
    /// Training forward pass: predict noise from noisy target conditioned on input at timestep t.
    /// </summary>
    private Tensor<T> ForwardTraining(Tensor<T> input, Tensor<T> noisyTarget, int t)
    {
        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1) conditioned = conditioned.Reshape(new[] { 1, conditioned.Length });

        Tensor<T> condHidden;
        if (_inputProjection is not null)
            condHidden = _inputProjection.Forward(conditioned);
        else
            condHidden = conditioned;

        int targetLen = noisyTarget.Length;
        int condLen = Math.Min(condHidden.Length, _hiddenDimension);
        var denoisingInput = new Tensor<T>(new[] { 1, targetLen + condLen + 1 });
        for (int i = 0; i < targetLen; i++) denoisingInput.Data.Span[i] = noisyTarget[i];
        for (int i = 0; i < condLen; i++) denoisingInput.Data.Span[targetLen + i] = condHidden[i];
        denoisingInput.Data.Span[targetLen + condLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _diffusionSteps - 1)));

        var eps = denoisingInput;
        foreach (var layer in _transformerLayers) eps = layer.Forward(eps);
        if (_outputProjection is not null) eps = _outputProjection.Forward(eps);

        var result = new Tensor<T>(new[] { _forecastHorizon });
        for (int i = 0; i < _forecastHorizon && i < eps.Length; i++)
            result.Data.Span[i] = eps[i];
        return result;
    }

    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "TimeDiff" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "DiffusionSteps", _diffusionSteps }, { "UseFutureMixup", _useFutureMixup }, { "UseAutoregressiveInit", _useAutoregressiveInit }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new TimeDiff<T>(Architecture, new TimeDiffOptions<T> { ContextLength = _contextLength, ForecastHorizon = _forecastHorizon, HiddenDimension = _hiddenDimension, NumLayers = _numLayers, NumHeads = _numHeads, DiffusionSteps = _diffusionSteps, DropoutRate = _dropout, BetaStart = _betaStart, BetaEnd = _betaEnd, UseFutureMixup = _useFutureMixup, UseAutoregressiveInit = _useAutoregressiveInit });

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_contextLength); writer.Write(_forecastHorizon); writer.Write(_hiddenDimension); writer.Write(_numLayers); writer.Write(_numHeads); writer.Write(_diffusionSteps); writer.Write(_dropout); writer.Write(_betaStart); writer.Write(_betaEnd); writer.Write(_useFutureMixup); writer.Write(_useAutoregressiveInit); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _contextLength = reader.ReadInt32(); _forecastHorizon = reader.ReadInt32(); _hiddenDimension = reader.ReadInt32(); _numLayers = reader.ReadInt32(); _numHeads = reader.ReadInt32(); _diffusionSteps = reader.ReadInt32(); _dropout = reader.ReadDouble(); _betaStart = reader.ReadDouble(); _betaEnd = reader.ReadDouble(); _useFutureMixup = reader.ReadBoolean(); _useAutoregressiveInit = reader.ReadBoolean(); ComputeNoiseSchedule(); }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) => _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // TimeDiff is non-autoregressive — truncate output to requested steps
        var fullForecast = Forecast(input, null);
        if (steps >= fullForecast.Length) return fullForecast;
        var result = new Tensor<T>(new[] { steps });
        for (int i = 0; i < steps; i++)
            result.Data.Span[i] = fullForecast[i];
        return result;
    }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) { int batchSize = input.Shape[0]; int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length; var result = new Tensor<T>(input.Shape._dims); for (int b = 0; b < batchSize; b++) { T mean = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) mean = NumOps.Add(mean, input[idx]); } mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen)); T variance = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) { var diff = NumOps.Subtract(input[idx], mean); variance = NumOps.Add(variance, NumOps.Multiply(diff, diff)); } } variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen)); T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5))); for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length && idx < result.Length) result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std); } } return result; }

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["DiffusionSteps"] = NumOps.FromDouble(_diffusionSteps), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// DDPM reverse process with optional autoregressive initialization.
    /// TimeDiff starts from AR-initialized noise (instead of pure noise) when UseAutoregressiveInit is enabled,
    /// which improves convergence by providing a structured starting point.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        Tensor<T> condHidden;
        if (_inputProjection is not null)
            condHidden = _inputProjection.Forward(conditioned);
        else
            condHidden = conditioned;

        int outputLen = _forecastHorizon;
        var rand = RandomHelper.CreateSecureRandom();

        // Autoregressive initialization: use last observed values extrapolated as starting point
        // then add noise at the last diffusion timestep level
        var xt = new Tensor<T>(new[] { 1, outputLen });
        if (_useAutoregressiveInit && conditioned.Length > 0)
        {
            // Simple AR(1) initialization: extrapolate from last observed values
            T lastVal = conditioned[conditioned.Length - 1];
            T secondLast = conditioned.Length > 1 ? conditioned[conditioned.Length - 2] : lastVal;
            T trend = NumOps.Subtract(lastVal, secondLast);
            int lastStep = _diffusionSteps - 1;
            T sqrtAlphaBarT = _sqrtAlphasCumprod[lastStep];
            T sqrtOneMinusAlphaBarT = _sqrtOneMinusAlphasCumprod[lastStep];
            for (int i = 0; i < outputLen; i++)
            {
                T arInit = NumOps.Add(lastVal, NumOps.Multiply(trend, NumOps.FromDouble(i + 1)));
                T epsNoise = SampleStandardNormal(rand);
                xt.Data.Span[i] = NumOps.Add(NumOps.Multiply(sqrtAlphaBarT, arInit), NumOps.Multiply(sqrtOneMinusAlphaBarT, epsNoise));
            }
        }
        else
        {
            for (int i = 0; i < outputLen; i++)
                xt.Data.Span[i] = SampleStandardNormal(rand);
        }

        // Iterative DDPM reverse process: t = T-1 ... 0
        T eps10 = NumOps.FromDouble(1e-10);
        for (int t = _diffusionSteps - 1; t >= 0; t--)
        {
            int xtLen = Math.Min(xt.Length, outputLen);
            int condLen = Math.Min(condHidden.Length, _hiddenDimension);
            var denoisingInput = new Tensor<T>(new[] { 1, xtLen + condLen + 1 });
            for (int i = 0; i < xtLen; i++) denoisingInput.Data.Span[i] = xt[i];
            for (int i = 0; i < condLen; i++) denoisingInput.Data.Span[xtLen + i] = condHidden[i];
            denoisingInput.Data.Span[xtLen + condLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _diffusionSteps - 1)));

            var eps = denoisingInput;
            foreach (var layer in _transformerLayers) eps = layer.Forward(eps);
            if (_outputProjection is not null) eps = _outputProjection.Forward(eps);

            // DDPM reverse step
            T alphaT = _alphas[t];
            T betaT = _betas[t];
            T sqrtOneMinusAlphaBarT = NumOps.Sqrt(NumOps.Subtract(NumOps.One, _alphasCumprod[t]));
            T noiseCoeffT = NumOps.Divide(betaT, NumOps.Add(sqrtOneMinusAlphaBarT, eps10));
            T sqrtAlphaT = NumOps.Sqrt(alphaT);
            T sigmaT = t > 0 ? NumOps.Sqrt(betaT) : NumOps.Zero;

            for (int i = 0; i < outputLen && i < xt.Length; i++)
            {
                T epsVal = i < eps.Length ? eps[i] : NumOps.Zero;
                T meanT = NumOps.Divide(NumOps.Subtract(xt[i], NumOps.Multiply(noiseCoeffT, epsVal)), NumOps.Add(sqrtAlphaT, eps10));
                T z = t > 0 ? SampleStandardNormal(rand) : NumOps.Zero;
                xt.Data.Span[i] = NumOps.Add(meanT, NumOps.Multiply(sigmaT, z));
            }
        }

        if (addedBatchDim && xt.Rank == 2 && xt.Shape[0] == 1) xt = xt.Reshape(new[] { xt.Shape[1] });
        return xt;
    }
    private Tensor<T> BackwardNative(Tensor<T> gradOutput) { var current = gradOutput; bool addedBatchDim = false; if (current.Rank == 1) { current = current.Reshape(new[] { 1, current.Length }); addedBatchDim = true; } if (_outputProjection is not null) current = _outputProjection.Backward(current); for (int i = _transformerLayers.Count - 1; i >= 0; i--) current = _transformerLayers[i].Backward(current); if (_inputProjection is not null) current = _inputProjection.Backward(current); if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1) current = current.Reshape(new[] { current.Shape[1] }); return current; }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Shape[0]; int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length; int features = input.Shape.Length > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion
}
