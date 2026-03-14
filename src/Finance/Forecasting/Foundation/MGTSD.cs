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
/// MG-TSD — Multi-Granularity Time Series Diffusion Model with Guided Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MG-TSD captures temporal patterns at multiple granularities using a coarse-to-fine
/// guidance mechanism where predictions at coarser levels guide fine-grained diffusion.
/// </para>
/// <para><b>For Beginners:</b> MG-TSD forecasts at multiple zoom levels simultaneously.
/// It first makes a rough forecast (like predicting monthly trends), then uses that to
/// guide a more detailed forecast (like daily values). This coarse-to-fine approach is
/// similar to how an artist first sketches the broad outlines before adding fine details,
/// resulting in more coherent and accurate probabilistic predictions.</para>
/// <para>
/// <b>Reference:</b> Fan et al., "MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", ICLR 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an MG-TSD multi-granularity time series diffusion model
/// // Coarse-to-fine guidance: rough forecasts at monthly level guide daily predictions
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with multi-granularity guided diffusion
/// var model = new MGTSD&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new MGTSD&lt;double&gt;(architecture, "mgtsd.onnx");
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
[ModelPaper("MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", "https://arxiv.org/abs/2403.05751", Year = 2024, Authors = "Xinyao Fan, Yueying Wu, Chang Xu, Yuhao Huang, Weiqing Liu, Jiang Bian")]
public class MGTSD<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _denoisingLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly MGTSDOptions<T> _options;

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
    private int _numGranularities;
    private double _guidanceWeight;

    // DDPM noise schedule (precomputed)
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

    public MGTSD(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        MGTSDOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath)) throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath)) throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");
        options ??= new MGTSDOptions<T>(); _options = options; Options = _options;
        _useNativeMode = false; OnnxModelPath = onnxModelPath; OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public MGTSD(NeuralNetworkArchitecture<T> architecture,
        MGTSDOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new MGTSDOptions<T>();
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

    private void CopyOptionsToFields(MGTSDOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumLayers, nameof(options.NumLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));
        Guard.Positive(options.DiffusionSteps, nameof(options.DiffusionSteps));
        Guard.Positive(options.NumGranularities, nameof(options.NumGranularities));

        if (options.BetaStart <= 0 || options.BetaEnd <= 0 || options.BetaEnd <= options.BetaStart)
            throw new ArgumentOutOfRangeException(nameof(options), "BetaStart and BetaEnd must be positive, and BetaEnd must be greater than BetaStart.");

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _diffusionSteps = options.DiffusionSteps;
        _dropout = options.DropoutRate;
        _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd;
        _numGranularities = options.NumGranularities;
        _guidanceWeight = options.GuidanceWeight;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
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
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultMGTSDLayers(Architecture, _contextLength, _forecastHorizon, _hiddenDimension, _numLayers, _numHeads, _numGranularities, _dropout)); ExtractLayerReferences(); }
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

        if (target.Length != _forecastHorizon)
            throw new ArgumentException(
                $"Target length ({target.Length}) must match ForecastHorizon ({_forecastHorizon}).",
                nameof(target));

        SetTrainingMode(true);
        try
        {
            var rand = RandomHelper.CreateSecureRandom();
            int t = rand.Next(_diffusionSteps);

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

            var predictedNoise = ForwardTraining(input, noisyTarget, t);
            LastLoss = _lossFunction.CalculateLoss(predictedNoise.ToVector(), noise.ToVector());
            var gradient = _lossFunction.CalculateDerivative(predictedNoise.ToVector(), noise.ToVector());
            BackwardNative(Tensor<T>.FromVector(gradient, predictedNoise.Shape));
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
        denoisingInput.Data.Span[targetLen + condLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _diffusionSteps - 1)));

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
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "MGTSD" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "DiffusionSteps", _diffusionSteps }, { "NumGranularities", _numGranularities }, { "GuidanceWeight", _guidanceWeight }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var opts = new MGTSDOptions<T> { ContextLength = _contextLength, ForecastHorizon = _forecastHorizon, HiddenDimension = _hiddenDimension, NumLayers = _numLayers, NumHeads = _numHeads, DiffusionSteps = _diffusionSteps, DropoutRate = _dropout, BetaStart = _betaStart, BetaEnd = _betaEnd, NumGranularities = _numGranularities, GuidanceWeight = _guidanceWeight };
        if (!_useNativeMode && OnnxModelPath is not null) return new MGTSD<T>(Architecture, OnnxModelPath, opts);
        return new MGTSD<T>(Architecture, opts);
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_contextLength); writer.Write(_forecastHorizon); writer.Write(_hiddenDimension); writer.Write(_numLayers); writer.Write(_numHeads); writer.Write(_diffusionSteps); writer.Write(_dropout); writer.Write(_betaStart); writer.Write(_betaEnd); writer.Write(_numGranularities); writer.Write(_guidanceWeight); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _contextLength = reader.ReadInt32(); _forecastHorizon = reader.ReadInt32(); _hiddenDimension = reader.ReadInt32(); _numLayers = reader.ReadInt32(); _numHeads = reader.ReadInt32(); _diffusionSteps = reader.ReadInt32(); _dropout = reader.ReadDouble(); _betaStart = reader.ReadDouble(); _betaEnd = reader.ReadDouble(); _numGranularities = reader.ReadInt32(); _guidanceWeight = reader.ReadDouble(); ComputeNoiseSchedule(); }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) { if (quantiles is not null && quantiles.Length > 0) throw new NotSupportedException("MGTSD does not support quantile forecasting. Pass null for point forecasts."); return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData); }
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) { int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; var result = new Tensor<T>(input.Shape); for (int b = 0; b < batchSize; b++) { T mean = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) mean = NumOps.Add(mean, input[idx]); } mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen)); T variance = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) { var diff = NumOps.Subtract(input[idx], mean); variance = NumOps.Add(variance, NumOps.Multiply(diff, diff)); } } variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen)); T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5))); for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length && idx < result.Length) result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std); } } return result; }

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["NumGranularities"] = NumOps.FromDouble(_numGranularities), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Multi-granularity DDPM reverse process with coarse-to-fine guided denoising.
    /// MG-TSD generates predictions at multiple temporal granularities (coarse → fine) and
    /// uses coarser predictions to guide finer-grained diffusion via the guidance weight.
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

        // Multi-granularity: generate coarse predictions first, then refine
        // Granularity levels: g=0 (coarsest, len/2^(G-1)) to g=G-1 (finest, full length)
        Tensor<T>? coarseGuidance = null;

        for (int g = 0; g < _numGranularities; g++)
        {
            // Compute granularity-specific output length
            int granScale = 1 << (_numGranularities - 1 - g); // coarsest has largest scale
            int granLen = Math.Max(1, outputLen / granScale);

            // Start from noise
            var xt = new Tensor<T>(new[] { 1, granLen });
            for (int i = 0; i < granLen; i++)
                xt.Data.Span[i] = SampleStandardNormal(rand);

            // DDPM reverse process at this granularity
            for (int t = _diffusionSteps - 1; t >= 0; t--)
            {
                int xtLen = Math.Min(xt.Length, granLen);
                int condLen = Math.Min(condHidden.Length, _hiddenDimension);
                int guideLen = coarseGuidance is not null ? Math.Min(coarseGuidance.Length, granLen) : 0;
                var denoisingInput = new Tensor<T>(new[] { 1, xtLen + condLen + guideLen + 1 });

                // Pack: [x_t | condHidden | coarseGuidance | timestep]
                int offset = 0;
                for (int i = 0; i < xtLen; i++) denoisingInput.Data.Span[offset + i] = xt[i];
                offset += xtLen;
                for (int i = 0; i < condLen; i++) denoisingInput.Data.Span[offset + i] = condHidden[i];
                offset += condLen;

                // Inject coarse guidance (upsampled) weighted by guidanceWeight
                if (coarseGuidance is not null)
                {
                    T guidanceWeightT = NumOps.FromDouble(_guidanceWeight);
                    for (int i = 0; i < guideLen; i++)
                    {
                        int coarseIdx = Math.Min(i * coarseGuidance.Length / Math.Max(1, granLen), coarseGuidance.Length - 1);
                        denoisingInput.Data.Span[offset + i] = NumOps.Multiply(coarseGuidance[coarseIdx], guidanceWeightT);
                    }
                    offset += guideLen;
                }

                denoisingInput.Data.Span[offset] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _diffusionSteps - 1)));

                var eps = denoisingInput;
                foreach (var layer in _denoisingLayers) eps = layer.Forward(eps);
                if (_outputProjection is not null) eps = _outputProjection.Forward(eps);

                T alphaT = _alphas[t];
                T betaT = _betas[t];
                T eps10 = NumOps.FromDouble(1e-10);
                T sqrtOneMinusAlphaBarT = NumOps.Sqrt(NumOps.Subtract(NumOps.One, _alphasCumprod[t]));
                T noiseCoeffT = NumOps.Divide(betaT, NumOps.Add(sqrtOneMinusAlphaBarT, eps10));
                T sqrtAlphaT = NumOps.Sqrt(alphaT);
                T sigmaT = t > 0 ? NumOps.Sqrt(betaT) : NumOps.Zero;

                for (int i = 0; i < granLen && i < xt.Length; i++)
                {
                    T epsVal = i < eps.Length ? eps[i] : NumOps.Zero;
                    T meanT = NumOps.Divide(NumOps.Subtract(xt[i], NumOps.Multiply(noiseCoeffT, epsVal)), NumOps.Add(sqrtAlphaT, eps10));
                    T z = t > 0 ? SampleStandardNormal(rand) : NumOps.Zero;
                    xt.Data.Span[i] = NumOps.Add(meanT, NumOps.Multiply(sigmaT, z));
                }
            }

            coarseGuidance = xt; // This granularity's output guides the next finer level
        }

        // Final output from finest granularity
        var result = coarseGuidance ?? new Tensor<T>(new[] { 1, outputLen });
        if (addedBatchDim && result.Rank == 2 && result.Shape[0] == 1) result = result.Reshape(new[] { result.Shape[1] });
        return result;
    }
    private Tensor<T> BackwardNative(Tensor<T> gradOutput) { var current = gradOutput; bool addedBatchDim = false; if (current.Rank == 1) { current = current.Reshape(new[] { 1, current.Length }); addedBatchDim = true; } if (_outputProjection is not null) current = _outputProjection.Backward(current); for (int i = _denoisingLayers.Count - 1; i >= 0; i--) current = _denoisingLayers[i].Backward(current); if (_inputProjection is not null) current = _inputProjection.Backward(current); if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1) current = current.Reshape(new[] { current.Shape[1] }); return current; }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input"; var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion
}
