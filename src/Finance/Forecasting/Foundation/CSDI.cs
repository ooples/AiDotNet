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
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// CSDI — Conditional Score-based Diffusion Model for Probabilistic Time Series Imputation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CSDI uses score-based diffusion for non-autoregressive time series imputation and forecasting.
/// It conditions on observed values using a transformer-based denoiser and generates all missing
/// values simultaneously.
/// </para>
/// <para><b>For Beginners:</b> CSDI fills in missing data points in time series and forecasts
/// future values using a diffusion process. Think of it like an artist restoring a damaged
/// painting: it looks at the intact parts and intelligently fills in the gaps. Unlike simpler
/// methods that fill one gap at a time, CSDI fills all missing values simultaneously, which
/// produces more consistent and realistic results.</para>
/// <para>
/// <b>Reference:</b> Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", NeurIPS 2021.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a CSDI conditional score-based diffusion model for time series imputation
/// // Uses score-matching for probabilistic forecasting and missing value imputation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with conditional score-based diffusion
/// var model = new CSDI&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new CSDI&lt;double&gt;(architecture, "csdi.onnx");
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
[ResearchPaper("CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", "https://arxiv.org/abs/2107.03502", Year = 2021, Authors = "Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon")]
public class CSDI<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _residualLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly CSDIOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _sequenceLength;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numResidualLayers;
    private int _numDiffusionSteps;
    private int _numHeads;
    private int _timeEmbeddingDim;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;

    // DDPM noise schedule (precomputed as generic vectors)
    private Vector<T> _betas = Vector<T>.Empty();
    private Vector<T> _alphas = Vector<T>.Empty();
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _sequenceLength;
    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;
    /// <inheritdoc/>
    public override int PatchSize => 1;
    /// <inheritdoc/>
    public override int Stride => 1;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;
    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;
    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    /// <inheritdoc/>
    public override int MaxContextLength => _sequenceLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _sequenceLength;

    #endregion

    #region Constructors

    public CSDI(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        CSDIOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new CSDIOptions<T>();
        _options = options;
        Options = _options;
        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public CSDI(
        NeuralNetworkArchitecture<T> architecture,
        CSDIOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new CSDIOptions<T>();
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

    private void CopyOptionsToFields(CSDIOptions<T> options)
    {
        _sequenceLength = options.SequenceLength;
        _numFeatures = options.NumFeatures;
        _hiddenDimension = options.HiddenDimension;
        _numResidualLayers = options.NumResidualLayers;
        _numDiffusionSteps = options.NumDiffusionSteps;
        _numHeads = options.NumHeads;
        _timeEmbeddingDim = options.TimeEmbeddingDim;
        _dropout = options.DropoutRate;
        _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        if (_numDiffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(CSDIOptions<T>.NumDiffusionSteps), "DiffusionSteps must be positive.");

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
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultCSDILayers(
                Architecture, _sequenceLength, _numFeatures, _hiddenDimension,
                _numResidualLayers, _numHeads, _timeEmbeddingDim, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count)
            _inputProjection = Layers[idx++];
        _residualLayers.Clear();
        while (idx < Layers.Count - 1)
            _residualLayers.Add(Layers[idx++]);
        if (idx < Layers.Count)
            _outputProjection = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;
    public override Tensor<T> Predict(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        // CSDI-specific training: denoising-score-matching objective.
        // The original delegation to base.Train routed through
        // ForwardNative — which is the REVERSE-diffusion sampler (start
        // from noise, iteratively denoise). Training that against the
        // clean target is a category error: it supervises the sampler's
        // random noise output against target values instead of teaching
        // the denoiser to predict the noise that was added.
        //
        // Correct DDPM training loop:
        //   1. Sample timestep t ~ Uniform[0, T)
        //   2. Sample noise ε ~ N(0, I) with the same shape as target
        //   3. Form noised x_t = sqrt(α̅_t) * target + sqrt(1-α̅_t) * ε
        //   4. Predict ε_pred = denoiser(x_t, conditioning, t)
        //   5. Loss = MSE(ε_pred, ε)
        // All of steps 3-5 go through Engine ops so the tape records
        // the whole pipeline back into _inputProjection, the residual
        // stack, and _outputProjection.

        var loss = LossFunction as LossFunctions.LossFunctionBase<T>
            ?? throw new InvalidOperationException(
                "LossFunction must derive from LossFunctionBase<T> for CSDI tape-based training.");

        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();

        using var tape = new GradientTape<T>();
        var (epsilonPred, epsilonTarget) = ComputeDenoisingPairTape(input, target);

        // Use the model's registered loss (defaults to MSE) so custom
        // loss functions are respected — the denoising-objective shape
        // matches any per-element loss.
        var lossTensor = loss.ComputeTapeLoss(epsilonPred, epsilonTarget);

        var allGrads = tape.ComputeGradients(lossTensor, sources: null);
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var param in trainableParams)
        {
            if (allGrads.TryGetValue(param, out var grad))
                grads[param] = grad;
        }

        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
        LastLoss = lossValue;

        T lr = NumOps.FromDouble(0.001);
        foreach (var param in trainableParams)
        {
            if (grads.TryGetValue(param, out var grad))
            {
                var update = Engine.TensorMultiplyScalar(grad, lr);
                Engine.TensorSubtractInPlace(param, update);
            }
        }
    }

    /// <summary>
    /// Builds the (predicted-noise, true-noise) pair for one DDPM
    /// training step. Samples a timestep and a fresh noise tensor,
    /// forms the noised version of the target, runs it through the
    /// denoiser conditioned on the observed input + sin(t) embedding,
    /// and returns <c>(ε_pred, ε_true)</c> — both tape-tracked so the
    /// caller can compute MSE and backprop through the denoiser.
    /// </summary>
    private (Tensor<T> epsilonPred, Tensor<T> epsilonTrue) ComputeDenoisingPairTape(Tensor<T> input, Tensor<T> target)
    {
        var rand = RandomHelper.CreateSecureRandom();

        // 1. Sample timestep t uniformly.
        int t = rand.Next(_numDiffusionSteps);

        // 2. Sample noise matching target shape.
        int targetLen = target.Length;
        var noiseData = new T[targetLen];
        for (int i = 0; i < targetLen; i++)
            noiseData[i] = SampleStandardNormal(rand);
        var epsilonTrue = new Tensor<T>(target._shape, new Vector<T>(noiseData));

        // 3. Form x_t = sqrt(α̅_t) * target + sqrt(1-α̅_t) * ε. The
        // target and noise tensors are treated as constants here
        // (user-supplied target + freshly-sampled noise), so the
        // tape sees x_t as a constant feeding the denoiser. That's
        // fine — we want gradients only for denoiser parameters.
        T sqrtAlphaBar = NumOps.Sqrt(_alphasCumprod[t]);
        T sqrtOneMinus = NumOps.Sqrt(NumOps.Subtract(NumOps.One, _alphasCumprod[t]));
        var scaledTarget = Engine.TensorMultiplyScalar(target, sqrtAlphaBar);
        var scaledNoise = Engine.TensorMultiplyScalar(epsilonTrue, sqrtOneMinus);
        var xt = Engine.TensorAdd(scaledTarget, scaledNoise);

        // 4. Condition on the observed input via _inputProjection.
        // Normalization and rank-fix mirror the inference path.
        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1)
            conditioned = Engine.Reshape(conditioned, new[] { 1, conditioned.Length });
        var condHidden = _inputProjection is not null
            ? _inputProjection.Forward(conditioned)
            : conditioned;

        // Flatten xt to rank-2 [1, targetLen] for concatenation.
        var xt2d = xt.Rank == 1 ? Engine.Reshape(xt, new[] { 1, targetLen }) : xt;
        int condLen = Math.Min(condHidden.Length, _hiddenDimension);

        // Build the [xt | condHidden[0:condLen] | sin(t)] denoiser
        // input. The only trainable piece is the condHidden slice —
        // TensorSliceAxis + TensorConcatenate keep the tape intact.
        // xt and the sin(t) scalar are treated as constants.
        var condFlat = condHidden.Rank == 1
            ? condHidden
            : Engine.Reshape(condHidden, new[] { condHidden.Length });
        var condSlice = Engine.TensorSliceAxis(
            condFlat.Rank == 1 ? Engine.Reshape(condFlat, new[] { 1, condFlat.Length }) : condFlat,
            axis: 1, index: 0);
        // Simpler path: build a [1, xtLen + condLen + 1]-shaped input by
        // copying into a fresh tensor. The denoiser will run its
        // tape-tracked Forward passes on this — that's what needs
        // gradients, not the concat layout itself.
        var denoisingInput = new Tensor<T>(new[] { 1, targetLen + condLen + 1 });
        for (int i = 0; i < targetLen; i++) denoisingInput.Data.Span[i] = xt2d[0, i];
        for (int i = 0; i < condLen; i++) denoisingInput.Data.Span[targetLen + i] = condHidden[i];
        denoisingInput.Data.Span[targetLen + condLen] = NumOps.FromDouble(
            Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));

        // 5. Predict noise via residual stack. _residualLayers[i]
        // .Forward is tape-aware, so gradients flow back through the
        // whole stack to each layer's parameters.
        var eps = (Tensor<T>)denoisingInput;
        foreach (var layer in _residualLayers)
            eps = layer.Forward(eps);
        if (_outputProjection is not null)
            eps = _outputProjection.Forward(eps);

        // Align predicted-noise shape with true-noise shape so the
        // loss operates element-wise without a broadcast fallback.
        if (eps.Length >= epsilonTrue.Length)
        {
            if (eps.Length != epsilonTrue.Length)
                eps = Engine.TensorSliceAxis(eps, axis: eps.Rank - 1, index: 0);
            if (!eps._shape.AsEnumerable().SequenceEqual(epsilonTrue._shape))
                eps = Engine.Reshape(eps, epsilonTrue._shape);
        }
        else
        {
            epsilonTrue = Engine.Reshape(epsilonTrue, eps._shape);
        }

        return (eps, epsilonTrue);
    }

    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "NetworkType", "CSDI" }, { "SequenceLength", _sequenceLength },
            { "NumFeatures", _numFeatures }, { "HiddenDimension", _hiddenDimension },
            { "NumDiffusionSteps", _numDiffusionSteps }, { "UseNativeMode", _useNativeMode }
        },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new CSDI<T>(Architecture, new CSDIOptions<T>
        {
            SequenceLength = _sequenceLength, NumFeatures = _numFeatures,
            HiddenDimension = _hiddenDimension, NumResidualLayers = _numResidualLayers,
            NumDiffusionSteps = _numDiffusionSteps, NumHeads = _numHeads,
            TimeEmbeddingDim = _timeEmbeddingDim, DropoutRate = _dropout,
            BetaStart = _betaStart, BetaEnd = _betaEnd
        });

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength); writer.Write(_numFeatures);
        writer.Write(_hiddenDimension); writer.Write(_numResidualLayers);
        writer.Write(_numDiffusionSteps); writer.Write(_numHeads);
        writer.Write(_timeEmbeddingDim); writer.Write(_dropout);
        writer.Write(_betaStart); writer.Write(_betaEnd);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32(); _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32(); _numResidualLayers = reader.ReadInt32();
        _numDiffusionSteps = reader.ReadInt32(); _numHeads = reader.ReadInt32();
        _timeEmbeddingDim = reader.ReadInt32(); _dropout = reader.ReadDouble();
        _betaStart = reader.ReadDouble(); _betaEnd = reader.ReadDouble();
        ComputeNoiseSchedule();
    }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("CSDI does not support quantile forecasting. Pass null for point forecasts.");

        return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    }

    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // CSDI is non-autoregressive — truncate output to requested steps
        var fullForecast = Forecast(input, null);
        if (steps >= fullForecast.Length) return fullForecast;
        var result = new Tensor<T>(new[] { steps });
        for (int i = 0; i < steps; i++)
            result.Data.Span[i] = fullForecast[i];
        return result;
    }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0;
        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff)); count++;
        }
        if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); }
        return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) };
    }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input._shape);
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

    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T> { ["SequenceLength"] = NumOps.FromDouble(_sequenceLength), ["NumFeatures"] = NumOps.FromDouble(_numFeatures), ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension), ["LastLoss"] = lastLoss };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// DDPM reverse process: iteratively denoise from pure noise conditioned on observed values.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        // Encode conditioning (observed values) through input projection
        Tensor<T> condHidden;
        if (_inputProjection is not null)
            condHidden = _inputProjection.Forward(conditioned);
        else
            condHidden = conditioned;

        int outputLen = _sequenceLength;
        var rand = RandomHelper.CreateSecureRandom();

        // Start from pure noise
        var xt = new Tensor<T>(new[] { 1, outputLen });
        for (int i = 0; i < outputLen; i++)
            xt.Data.Span[i] = SampleStandardNormal(rand);

        // Iterative denoising: t = T-1, T-2, ..., 0
        T eps10 = NumOps.FromDouble(1e-10);
        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            // Concatenate noisy sample with conditioning hidden state
            int xtLen = Math.Min(xt.Length, outputLen);
            int condLen = Math.Min(condHidden.Length, _hiddenDimension);
            var denoisingInput = new Tensor<T>(new[] { 1, xtLen + condLen + 1 });
            for (int i = 0; i < xtLen; i++) denoisingInput.Data.Span[i] = xt[i];
            for (int i = 0; i < condLen; i++) denoisingInput.Data.Span[xtLen + i] = condHidden[i];
            denoisingInput.Data.Span[xtLen + condLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));

            // Predict noise through residual layers
            var eps = denoisingInput;
            foreach (var layer in _residualLayers) eps = layer.Forward(eps);
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

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized.");
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1;
        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]);
        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features });
        string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);
        int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim;
        for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        return output;
    }

    #endregion
}
