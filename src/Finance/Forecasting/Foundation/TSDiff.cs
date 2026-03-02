using System.IO;
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
/// TSDiff — Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TSDiff uses unconditional denoising diffusion as a self-supervised pretraining objective
/// with self-guided refinement for high-quality probabilistic forecasting.
/// </para>
/// <para>
/// <b>Reference:</b> Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", NeurIPS 2023.
/// </para>
/// </remarks>
public class TSDiff<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _residualLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TSDiffOptions<T> _options;

    public override ModelOptions GetOptions() => _options;

    private int _sequenceLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numResidualBlocks;
    private int _numDiffusionSteps;
    private int _numAttentionHeads;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;
    private double _guidanceScale;

    // DDPM noise schedule (precomputed as generic vectors)
    private Vector<T> _betas = Vector<T>.Empty();
    private Vector<T> _alphas = Vector<T>.Empty();
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();

    #endregion

    #region Properties

    public override int SequenceLength => _sequenceLength;
    public override int PredictionHorizon => _forecastHorizon;
    public override int NumFeatures => 1;
    public override int PatchSize => 1;
    public override int Stride => 1;
    public override bool IsChannelIndependent => true;
    public override bool UseNativeMode => _useNativeMode;
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    public override int MaxContextLength => _sequenceLength;
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    public TSDiff(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        TSDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath)) throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath)) throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");
        options ??= new TSDiffOptions<T>(); _options = options; Options = _options;
        _useNativeMode = false; OnnxModelPath = onnxModelPath; OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public TSDiff(NeuralNetworkArchitecture<T> architecture,
        TSDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TSDiffOptions<T>(); _options = options; Options = _options;
        _useNativeMode = true; OnnxSession = null; OnnxModelPath = null;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options); InitializeLayers();
    }

    private void CopyOptionsToFields(TSDiffOptions<T> options)
    {
        _sequenceLength = options.SequenceLength; _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension; _numResidualBlocks = options.NumResidualBlocks;
        _numDiffusionSteps = options.NumDiffusionSteps; _numAttentionHeads = options.NumAttentionHeads;
        _dropout = options.DropoutRate; _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd; _guidanceScale = options.GuidanceScale;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        _betas = new Vector<T>(_numDiffusionSteps);
        _alphas = new Vector<T>(_numDiffusionSteps);
        _alphasCumprod = new Vector<T>(_numDiffusionSteps);
        _sqrtAlphasCumprod = new Vector<T>(_numDiffusionSteps);
        _sqrtOneMinusAlphasCumprod = new Vector<T>(_numDiffusionSteps);
        T one = NumOps.One;
        T betaStartT = NumOps.FromDouble(_betaStart);
        T betaRangeT = NumOps.FromDouble(_betaEnd - _betaStart);
        T maxDenom = NumOps.FromDouble(Math.Max(1, _numDiffusionSteps - 1));
        for (int t = 0; t < _numDiffusionSteps; t++)
        {
            _betas[t] = NumOps.Add(betaStartT, NumOps.Divide(NumOps.Multiply(betaRangeT, NumOps.FromDouble(t)), maxDenom));
            _alphas[t] = NumOps.Subtract(one, _betas[t]);
        }
        _alphasCumprod[0] = _alphas[0];
        for (int t = 1; t < _numDiffusionSteps; t++)
            _alphasCumprod[t] = NumOps.Multiply(_alphasCumprod[t - 1], _alphas[t]);
        for (int t = 0; t < _numDiffusionSteps; t++)
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
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultTSDiffLayers(Architecture, _sequenceLength, _forecastHorizon, _hiddenDimension, _numResidualBlocks, _numAttentionHeads, _dropout)); ExtractLayerReferences(); }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count) _inputProjection = Layers[idx++];
        _residualLayers.Clear();
        while (idx < Layers.Count - 1) _residualLayers.Add(Layers[idx++]);
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
            // DDPM training: sample random timestep, add noise, predict noise
            var rand = RandomHelper.CreateSecureRandom();
            int t = rand.Next(_numDiffusionSteps);

            // Add noise to target: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
            var noise = new Tensor<T>(target.Shape);
            for (int i = 0; i < target.Length; i++)
                noise.Data.Span[i] = SampleStandardNormal(rand);

            T sqrtAlphaBar = _sqrtAlphasCumprod[t];
            T sqrtOneMinusAlphaBar = _sqrtOneMinusAlphasCumprod[t];
            var noisyTarget = new Tensor<T>(target.Shape);
            for (int i = 0; i < target.Length; i++)
                noisyTarget.Data.Span[i] = NumOps.Add(
                    NumOps.Multiply(sqrtAlphaBar, target[i]),
                    NumOps.Multiply(sqrtOneMinusAlphaBar, noise[i]));

            // Forward pass predicts noise from noisy target
            var predictedNoise = ForwardTraining(input, noisyTarget, t);

            // Loss: MSE between predicted and actual noise
            LastLoss = _lossFunction.CalculateLoss(predictedNoise.ToVector(), noise.ToVector());
            var gradient = _lossFunction.CalculateDerivative(predictedNoise.ToVector(), noise.ToVector());
            BackwardNative(Tensor<T>.FromVector(gradient, predictedNoise.Shape));
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
        denoisingInput.Data.Span[targetLen + condLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));

        var eps = denoisingInput;
        foreach (var layer in _residualLayers) eps = layer.Forward(eps);
        if (_outputProjection is not null) eps = _outputProjection.Forward(eps);

        // Extract forecast-length output
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
        ModelType = ModelType.NeuralNetwork,
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "TSDiff" }, { "SequenceLength", _sequenceLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "NumDiffusionSteps", _numDiffusionSteps }, { "GuidanceScale", _guidanceScale }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new TSDiff<T>(Architecture, new TSDiffOptions<T> { SequenceLength = _sequenceLength, ForecastHorizon = _forecastHorizon, HiddenDimension = _hiddenDimension, NumResidualBlocks = _numResidualBlocks, NumDiffusionSteps = _numDiffusionSteps, NumAttentionHeads = _numAttentionHeads, DropoutRate = _dropout, BetaStart = _betaStart, BetaEnd = _betaEnd, GuidanceScale = _guidanceScale });

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_sequenceLength); writer.Write(_forecastHorizon); writer.Write(_hiddenDimension); writer.Write(_numResidualBlocks); writer.Write(_numDiffusionSteps); writer.Write(_numAttentionHeads); writer.Write(_dropout); writer.Write(_betaStart); writer.Write(_betaEnd); writer.Write(_guidanceScale); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _sequenceLength = reader.ReadInt32(); _forecastHorizon = reader.ReadInt32(); _hiddenDimension = reader.ReadInt32(); _numResidualBlocks = reader.ReadInt32(); _numDiffusionSteps = reader.ReadInt32(); _numAttentionHeads = reader.ReadInt32(); _dropout = reader.ReadDouble(); _betaStart = reader.ReadDouble(); _betaEnd = reader.ReadDouble(); _guidanceScale = reader.ReadDouble(); ComputeNoiseSchedule(); }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) => _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) { int batchSize = input.Shape[0]; int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length; var result = new Tensor<T>(input.Shape); for (int b = 0; b < batchSize; b++) { T mean = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) mean = NumOps.Add(mean, input[idx]); } mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen)); T variance = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) { var diff = NumOps.Subtract(input[idx], mean); variance = NumOps.Add(variance, NumOps.Multiply(diff, diff)); } } variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen)); T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5))); for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length && idx < result.Length) result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std); } } return result; }

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["SequenceLength"] = NumOps.FromDouble(_sequenceLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["GuidanceScale"] = NumOps.FromDouble(_guidanceScale), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// DDPM reverse process with self-guided diffusion refinement.
    /// TSDiff uses unconditional denoising as a pretraining objective, then refines
    /// predictions at inference by blending unconditional and conditioned noise estimates.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        // Encode conditioning context through input projection
        Tensor<T> condHidden;
        if (_inputProjection is not null)
            condHidden = _inputProjection.Forward(conditioned);
        else
            condHidden = conditioned;

        int outputLen = _forecastHorizon;
        var rand = RandomHelper.CreateSecureRandom();

        // Start from pure Gaussian noise
        var xt = new Tensor<T>(new[] { 1, outputLen });
        for (int i = 0; i < outputLen; i++)
            xt.Data.Span[i] = SampleStandardNormal(rand);

        // Iterative DDPM reverse process: t = T-1 ... 0
        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            // Build denoising input: [x_t | condHidden | timestep_embedding]
            int xtLen = Math.Min(xt.Length, outputLen);
            int condLen = Math.Min(condHidden.Length, _hiddenDimension);
            var denoisingInput = new Tensor<T>(new[] { 1, xtLen + condLen + 1 });
            for (int i = 0; i < xtLen; i++) denoisingInput.Data.Span[i] = xt[i];
            for (int i = 0; i < condLen; i++) denoisingInput.Data.Span[xtLen + i] = condHidden[i];
            denoisingInput.Data.Span[xtLen + condLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));

            // Predict conditioned noise estimate eps_cond
            var epsCond = denoisingInput;
            foreach (var layer in _residualLayers) epsCond = layer.Forward(epsCond);
            if (_outputProjection is not null) epsCond = _outputProjection.Forward(epsCond);

            // Self-guided diffusion: compute unconditional estimate (no conditioning)
            // eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            // When guidance_scale > 1, this amplifies the conditioned signal
            Tensor<T> epsGuided;
            if (Math.Abs(_guidanceScale - 1.0) > 1e-6)
            {
                var uncondInput = new Tensor<T>(new[] { 1, xtLen + condLen + 1 });
                for (int i = 0; i < xtLen; i++) uncondInput.Data.Span[i] = xt[i];
                // Zero conditioning for unconditional estimate
                uncondInput.Data.Span[xtLen + condLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));

                var epsUncond = uncondInput;
                foreach (var layer in _residualLayers) epsUncond = layer.Forward(epsUncond);
                if (_outputProjection is not null) epsUncond = _outputProjection.Forward(epsUncond);

                // Classifier-free guidance blend: eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
                T guidanceT = NumOps.FromDouble(_guidanceScale);
                epsGuided = new Tensor<T>(new[] { 1, outputLen });
                for (int i = 0; i < outputLen; i++)
                {
                    T uc = i < epsUncond.Length ? epsUncond[i] : NumOps.Zero;
                    T cd = i < epsCond.Length ? epsCond[i] : NumOps.Zero;
                    epsGuided.Data.Span[i] = NumOps.Add(uc, NumOps.Multiply(guidanceT, NumOps.Subtract(cd, uc)));
                }
            }
            else
            {
                epsGuided = epsCond;
            }

            // DDPM reverse step: x_{t-1} = (x_t - beta_t/sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_t) + sigma_t * z
            T alphaT = _alphas[t];
            T betaT = _betas[t];
            T eps10 = NumOps.FromDouble(1e-10);
            T sqrtOneMinusAlphaBarT = NumOps.Sqrt(NumOps.Subtract(NumOps.One, _alphasCumprod[t]));
            T noiseCoeffT = NumOps.Divide(betaT, NumOps.Add(sqrtOneMinusAlphaBarT, eps10));
            T sqrtAlphaT = NumOps.Sqrt(alphaT);
            T sigmaT = t > 0 ? NumOps.Sqrt(betaT) : NumOps.Zero;

            for (int i = 0; i < outputLen && i < xt.Length; i++)
            {
                T epsVal = i < epsGuided.Length ? epsGuided[i] : NumOps.Zero;
                T meanT = NumOps.Divide(NumOps.Subtract(xt[i], NumOps.Multiply(noiseCoeffT, epsVal)), NumOps.Add(sqrtAlphaT, eps10));
                T z = t > 0 ? SampleStandardNormal(rand) : NumOps.Zero;
                xt.Data.Span[i] = NumOps.Add(meanT, NumOps.Multiply(sigmaT, z));
            }
        }

        if (addedBatchDim && xt.Rank == 2 && xt.Shape[0] == 1) xt = xt.Reshape(new[] { xt.Shape[1] });
        return xt;
    }
    private Tensor<T> BackwardNative(Tensor<T> gradOutput) { var current = gradOutput; bool addedBatchDim = false; if (current.Rank == 1) { current = current.Reshape(new[] { 1, current.Length }); addedBatchDim = true; } if (_outputProjection is not null) current = _outputProjection.Backward(current); for (int i = _residualLayers.Count - 1; i >= 0; i--) current = _residualLayers[i].Backward(current); if (_inputProjection is not null) current = _inputProjection.Backward(current); if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1) current = current.Reshape(new[] { current.Shape[1] }); return current; }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Shape[0]; int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length; int features = input.Shape.Length > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion
}
