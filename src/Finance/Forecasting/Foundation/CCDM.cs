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
/// CCDM — Conditional Continuous Diffusion Model for Time Series.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CCDM extends continuous diffusion models for conditional time series generation,
/// operating in continuous space with a score-matching objective for high-quality
/// probabilistic forecasting.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class CCDM<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _denoisingLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly CCDMOptions<T> _options;

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
    private double _sigmaMin;
    private double _sigmaMax;

    // DDPM noise schedule (precomputed) - CCDM also uses beta schedule for discrete steps
    private Vector<T> _betas = Vector<T>.Empty();
    private Vector<T> _alphas = Vector<T>.Empty();
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();
    // Continuous sigma schedule (geometric interpolation from sigmaMax to sigmaMin)
    private Vector<T> _sigmas = Vector<T>.Empty();

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

    public CCDM(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        CCDMOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath)) throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath)) throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");
        options ??= new CCDMOptions<T>(); _options = options; Options = _options;
        _useNativeMode = false; OnnxModelPath = onnxModelPath; OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public CCDM(NeuralNetworkArchitecture<T> architecture,
        CCDMOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new CCDMOptions<T>(); _options = options; Options = _options;
        _useNativeMode = true; OnnxSession = null; OnnxModelPath = null;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options); InitializeLayers();
    }

    private void CopyOptionsToFields(CCDMOptions<T> options)
    {
        _contextLength = options.ContextLength; _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension; _numLayers = options.NumLayers;
        _numHeads = options.NumHeads; _diffusionSteps = options.DiffusionSteps;
        _dropout = options.DropoutRate; _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd; _sigmaMin = options.SigmaMin;
        _sigmaMax = options.SigmaMax;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        if (_diffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(_diffusionSteps), "DiffusionSteps must be positive.");

        // Standard DDPM beta schedule
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

        // Continuous sigma schedule: geometric interpolation from sigmaMax to sigmaMin
        _sigmas = new Vector<T>(_diffusionSteps + 1);
        for (int t = 0; t <= _diffusionSteps; t++)
        {
            double frac = (double)t / Math.Max(1, _diffusionSteps);
            _sigmas[t] = NumOps.FromDouble(_sigmaMax * Math.Pow(_sigmaMin / Math.Max(1e-10, _sigmaMax), frac));
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
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultCCDMLayers(Architecture, _contextLength, _forecastHorizon, _hiddenDimension, _numLayers, _numHeads, _dropout)); ExtractLayerReferences(); }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count) _inputProjection = Layers[idx++];
        _denoisingLayers.Clear();
        while (idx < Layers.Count - 1) _denoisingLayers.Add(Layers[idx++]);
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

            // Score matching training: perturb target with noise at sigma level
            T sigmaT = _sigmas[t];
            T eps10 = NumOps.FromDouble(1e-10);
            var noise = new Tensor<T>(target.Shape);
            for (int i = 0; i < target.Length; i++)
                noise.Data.Span[i] = SampleStandardNormal(rand);

            // x_noisy = x_0 + sigma * eps
            var noisyTarget = new Tensor<T>(target.Shape);
            for (int i = 0; i < target.Length; i++)
                noisyTarget.Data.Span[i] = NumOps.Add(target[i], NumOps.Multiply(sigmaT, noise[i]));

            // Score target: -eps/sigma (the gradient of log p_sigma(x))
            T negSigmaInv = NumOps.Negate(NumOps.Divide(NumOps.One, NumOps.Add(sigmaT, eps10)));
            var scoreTarget = new Tensor<T>(target.Shape);
            for (int i = 0; i < target.Length; i++)
                scoreTarget.Data.Span[i] = NumOps.Multiply(negSigmaInv, noise[i]);

            var predictedScore = ForwardTraining(input, noisyTarget, t);

            // Weighted score matching loss: sigma^2 * ||score_pred - score_target||^2
            T sigmaSquared = NumOps.Multiply(sigmaT, sigmaT);
            T weightedLoss = NumOps.Zero;
            int count = Math.Min(predictedScore.Length, scoreTarget.Length);
            for (int i = 0; i < count; i++)
            {
                T diff = NumOps.Subtract(predictedScore[i], scoreTarget[i]);
                weightedLoss = NumOps.Add(weightedLoss, NumOps.Multiply(sigmaSquared, NumOps.Multiply(diff, diff)));
            }
            LastLoss = count > 0 ? NumOps.Divide(weightedLoss, NumOps.FromDouble(count)) : NumOps.Zero;

            var gradient = _lossFunction.CalculateDerivative(predictedScore.ToVector(), scoreTarget.ToVector());
            // Scale gradient by sigmaSquared to match the weighted loss objective
            for (int i = 0; i < gradient.Length; i++)
                gradient[i] = NumOps.Multiply(sigmaSquared, gradient[i]);
            BackwardNative(Tensor<T>.FromVector(gradient, predictedScore.Shape));
            _optimizer.UpdateParameters(Layers);
        }
        finally { SetTrainingMode(false); }
    }

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
        // Encode sigma level as log(sigma) for continuous conditioning
        denoisingInput.Data.Span[targetLen + condLen] = NumOps.FromDouble(Math.Log(Math.Max(1e-10, NumOps.ToDouble(_sigmas[t]))));

        var eps = denoisingInput;
        foreach (var layer in _denoisingLayers) eps = layer.Forward(eps);
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
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "CCDM" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "DiffusionSteps", _diffusionSteps }, { "SigmaMin", _sigmaMin }, { "SigmaMax", _sigmaMax }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new CCDM<T>(Architecture, new CCDMOptions<T> { ContextLength = _contextLength, ForecastHorizon = _forecastHorizon, HiddenDimension = _hiddenDimension, NumLayers = _numLayers, NumHeads = _numHeads, DiffusionSteps = _diffusionSteps, DropoutRate = _dropout, BetaStart = _betaStart, BetaEnd = _betaEnd, SigmaMin = _sigmaMin, SigmaMax = _sigmaMax });

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_contextLength); writer.Write(_forecastHorizon); writer.Write(_hiddenDimension); writer.Write(_numLayers); writer.Write(_numHeads); writer.Write(_diffusionSteps); writer.Write(_dropout); writer.Write(_betaStart); writer.Write(_betaEnd); writer.Write(_sigmaMin); writer.Write(_sigmaMax); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _contextLength = reader.ReadInt32(); _forecastHorizon = reader.ReadInt32(); _hiddenDimension = reader.ReadInt32(); _numLayers = reader.ReadInt32(); _numHeads = reader.ReadInt32(); _diffusionSteps = reader.ReadInt32(); _dropout = reader.ReadDouble(); _betaStart = reader.ReadDouble(); _betaEnd = reader.ReadDouble(); _sigmaMin = reader.ReadDouble(); _sigmaMax = reader.ReadDouble(); ComputeNoiseSchedule(); }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) { if (quantiles is not null && quantiles.Length > 0) throw new NotSupportedException("CCDM does not support quantile forecasting. Pass null for point forecasts."); return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData); }
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) { int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; var result = new Tensor<T>(input.Shape); for (int b = 0; b < batchSize; b++) { T mean = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) mean = NumOps.Add(mean, input[idx]); } mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen)); T variance = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) { var diff = NumOps.Subtract(input[idx], mean); variance = NumOps.Add(variance, NumOps.Multiply(diff, diff)); } } variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen)); T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5))); for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length && idx < result.Length) result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std); } } return result; }

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["SigmaMin"] = NumOps.FromDouble(_sigmaMin), ["SigmaMax"] = NumOps.FromDouble(_sigmaMax), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Continuous diffusion reverse process using annealed Langevin dynamics.
    /// CCDM operates in continuous noise space: at each step, the model predicts the score
    /// (gradient of log probability) and uses it to iteratively denoise from high sigma to low sigma.
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

        // Start from noise at highest sigma level
        var xt = new Tensor<T>(new[] { 1, outputLen });
        for (int i = 0; i < outputLen; i++)
            xt.Data.Span[i] = NumOps.Multiply(_sigmas[0], SampleStandardNormal(rand));

        // Annealed Langevin dynamics: iterate from high sigma to low sigma
        for (int t = 0; t < _diffusionSteps; t++)
        {
            T sigmaT = _sigmas[t];
            T sigmaNext = _sigmas[t + 1];

            // Build score network input: [x_t | condHidden | log(sigma)]
            int xtLen = Math.Min(xt.Length, outputLen);
            int condLen = Math.Min(condHidden.Length, _hiddenDimension);
            var scoreInput = new Tensor<T>(new[] { 1, xtLen + condLen + 1 });
            for (int i = 0; i < xtLen; i++) scoreInput.Data.Span[i] = xt[i];
            for (int i = 0; i < condLen; i++) scoreInput.Data.Span[xtLen + i] = condHidden[i];
            scoreInput.Data.Span[xtLen + condLen] = NumOps.FromDouble(Math.Log(Math.Max(1e-10, NumOps.ToDouble(sigmaT))));

            // Predict score: s_theta(x_t, sigma_t)
            var score = scoreInput;
            foreach (var layer in _denoisingLayers) score = layer.Forward(score);
            if (_outputProjection is not null) score = _outputProjection.Forward(score);

            // Langevin step: x_{t+1} = x_t + (sigma_t^2 - sigma_next^2) * score + sqrt(sigma_t^2 - sigma_next^2) * z
            T stepSizeT = NumOps.Subtract(NumOps.Multiply(sigmaT, sigmaT), NumOps.Multiply(sigmaNext, sigmaNext));
            T noiseScaleT = NumOps.Sqrt(NumOps.Add(stepSizeT, NumOps.FromDouble(1e-30))); // clamp to non-negative

            for (int i = 0; i < outputLen && i < xt.Length; i++)
            {
                T scoreVal = i < score.Length ? score[i] : NumOps.Zero;
                T z = (t < _diffusionSteps - 1) ? SampleStandardNormal(rand) : NumOps.Zero;
                xt.Data.Span[i] = NumOps.Add(NumOps.Add(xt[i], NumOps.Multiply(stepSizeT, scoreVal)), NumOps.Multiply(noiseScaleT, z));
            }
        }

        if (addedBatchDim && xt.Rank == 2 && xt.Shape[0] == 1) xt = xt.Reshape(new[] { xt.Shape[1] });
        return xt;
    }
    private Tensor<T> BackwardNative(Tensor<T> gradOutput) { var current = gradOutput; bool addedBatchDim = false; if (current.Rank == 1) { current = current.Reshape(new[] { 1, current.Length }); addedBatchDim = true; } if (_outputProjection is not null) current = _outputProjection.Backward(current); for (int i = _denoisingLayers.Count - 1; i >= 0; i--) current = _denoisingLayers[i].Backward(current); if (_inputProjection is not null) current = _inputProjection.Backward(current); if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1) current = current.Reshape(new[] { current.Shape[1] }); return current; }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input"; var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion
}
