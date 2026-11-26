using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-DDPM meta-learns a denoising diffusion model that generates task-specific model
/// weights conditioned on support set embeddings. It uses the full DDPM framework with
/// a linear noise schedule and EMA parameter averaging for stable generation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// DDPM Schedule: β_t linear from β_start to β_end, α_t = 1-β_t, ᾱ_t = Πα_s
///
/// Training (noise prediction objective):
///   1. For task τ: get target weights w_0 = base_params + gradient_adapted_delta
///   2. Compute task condition: c = TaskEncoder(avg(model(x_support)))
///   3. Sample t ~ U(1,T), ε ~ N(0,I)
///   4. Noised: w_t = √ᾱ_t · w_0 + √(1-ᾱ_t) · ε
///   5. Predicted noise: ε̂ = UNet(w_t, c, t)
///   6. L = ||ε - ε̂||²
///   7. EMA update denoiser params: φ_ema = decay * φ_ema + (1-decay) * φ
///
/// Generation (DDPM sampling):
///   1. w_T ~ N(0,I)
///   2. For t = T,...,1:
///      μ_θ = (1/√α_t)(w_t - β_t/√(1-ᾱ_t) · ε_θ(w_t,c,t))
///      w_{t-1} = μ_θ + σ_t · z,  z ~ N(0,I), σ_t = √β_t
///   3. θ_adapted = θ_base + w_0
/// </code>
/// </para>
/// <para><b>Key difference from MetaDiff:</b> Meta-DDPM uses the standard DDPM sampling
/// algorithm with full variance (σ_t = √β_t), while MetaDiff uses the deterministic
/// DDIM-style denoising. Meta-DDPM also includes EMA for denoiser parameter stability.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Denoising Diffusion Probabilistic Models",
    "https://arxiv.org/abs/2006.11239",
    Year = 2020,
    Authors = "Jonathan Ho, Ajay Jain, Pieter Abbeel")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaDDPMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MetaDDPMOptions<T, TInput, TOutput> _algoOptions;

    private readonly Vector<T> _betas;
    private readonly Vector<T> _alphas;
    private readonly Vector<T> _alphasCumprod;
    private readonly Vector<T> _sqrtAlphasCumprod;
    private readonly Vector<T> _sqrtOneMinusAlphasCumprod;
    private readonly Vector<T> _posteriorVariance;

    /// <summary>Denoiser (noise predictor) parameters.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _denoiserParams;

    /// <summary>EMA copy of denoiser parameters for stable generation.</summary>
    [Buffer]
    private Vector<T> _denoiserEma;

    /// <summary>Task encoder parameters.</summary>
    private Vector<T> _taskEncoderParams;

    private readonly int _paramDim;
    private readonly int _condDim;
    private readonly int _compressedDim;
    private readonly int _numTimesteps;

    private const int MaxCompressedDim = 128;
    private const int MaxFeatureDim = 64;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaDDPM;

    public MetaDDPMAlgorithm(MetaDDPMOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.NumTimesteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumTimesteps must be positive.");
        if (options.TaskConditionDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "TaskConditionDim must be positive.");
        if (options.BetaStart <= 0 || options.BetaEnd <= 0 || options.BetaStart >= options.BetaEnd)
            throw new ArgumentOutOfRangeException(nameof(options), "BetaStart and BetaEnd must be positive with BetaStart < BetaEnd.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _condDim = options.TaskConditionDim;
        _compressedDim = Math.Min(_paramDim, MaxCompressedDim);
        _numTimesteps = options.NumTimesteps;

        // Precompute DDPM noise schedule quantities
        // beta_t = start + (end - start) * t / max(T - 1, 1), alpha_t = 1 - beta_t, alphaBar_t = prod_{s<=t} alpha_s
        var betas = Engine.TensorAddScalar(
            Engine.TensorDivideScalar(
                Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, _numTimesteps, 1), NumOps.FromDouble(options.BetaEnd - options.BetaStart)),
                NumOps.FromDouble(Math.Max(_numTimesteps - 1, 1))),
            NumOps.FromDouble(options.BetaStart));
        var ones = Engine.TensorOnesLike(betas);
        var alphas = Engine.TensorSubtract(ones, betas);
        var alphasCumprod = Engine.TensorCumProd(alphas, 0);
        var oneMinusCumprod = Engine.TensorSubtract(ones, alphasCumprod);
        _betas = betas.ToVector();
        _alphas = alphas.ToVector();
        _alphasCumprod = alphasCumprod.ToVector();
        _sqrtAlphasCumprod = Engine.TensorSqrt(alphasCumprod).ToVector();
        _sqrtOneMinusAlphasCumprod = Engine.TensorSqrt(oneMinusCumprod).ToVector();

        // Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t), and 0 at t = 0 (where 1 - ᾱ_{-1} is taken as 0)
        var oneMinusPrevious = _numTimesteps == 1
            ? new Vector<T>(1)
            : Vector<T>.Concatenate(new Vector<T>(1), oneMinusCumprod.ToVector().GetSubVector(0, _numTimesteps - 1));
        _posteriorVariance = Engine.Divide(
            Engine.Multiply(_betas, oneMinusPrevious),
            Engine.Add(oneMinusCumprod.ToVector(), Engine.Fill(_numTimesteps, NumOps.FromDouble(1e-10))));

        // Task encoder: featureDim → condDim
        int featureDim = Math.Min(_paramDim, MaxFeatureDim);
        int encoderSize = featureDim * _condDim + _condDim;
        _taskEncoderParams = new Vector<T>(encoderSize);
        InitGaussian(_taskEncoderParams, 1.0 / Math.Sqrt(featureDim));

        // Denoiser MLP: (compressedDim + condDim + 1) → hidden → compressedDim
        int denoiserInput = _compressedDim + _condDim + 1;
        int hidden = Math.Max(64, _compressedDim);
        int denoiserSize = denoiserInput * hidden + hidden + hidden * _compressedDim + _compressedDim;
        _denoiserParams = new Vector<T>(denoiserSize);
        InitGaussian(_denoiserParams, 1.0 / Math.Sqrt(denoiserInput));

        // Initialize EMA as copy of denoiser
        _denoiserEma = _denoiserParams.Clone();
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(baseParams);

            // Get target: gradient-adapted weights (compressed delta)
            var targetDelta = ComputeTargetDelta(baseParams, task);

            // Task conditioning
            var condition = ComputeTaskCondition(task.SupportInput);

            // Sample timestep and noise
            int t = RandomGenerator.Next(_numTimesteps);
            var epsilon = SampleGaussian(_compressedDim);

            // Forward diffusion: w_t = √ᾱ_t * w_0 + √(1-ᾱ_t) * ε
            var wt = AddNoise(targetDelta, epsilon, t);

            // Predict noise using denoiser
            var predictedNoise = PredictNoise(wt, condition, t, useEma: false);

            // Noise prediction loss
            T noiseLoss = MeanSquaredDifference(epsilon, predictedNoise);

            // Also evaluate actual task performance
            var generated = SampleDDPM(condition, Math.Min(_algoOptions.SamplingSteps, 5));
            var adaptedParams = ApplyCompressedDelta(baseParams, generated);
            ParamModel.SetParameters(adaptedParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(NumOps.Add(queryLoss, noiseLoss));

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update base params
        ParamModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            ParamModel.SetParameters(ApplyOuterGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update denoiser and task encoder via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _denoiserParams, _algoOptions.OuterLearningRate, ComputeDDPMLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _taskEncoderParams, _algoOptions.OuterLearningRate, ComputeDDPMLoss);

        // EMA update of denoiser
        double decay = _algoOptions.EmaDecay;
        var ema = Engine.Add(
            Engine.Multiply(_denoiserEma, NumOps.FromDouble(decay)),
            Engine.Multiply(_denoiserParams.GetSubVector(0, _denoiserEma.Length), NumOps.FromDouble(1.0 - decay)));
        ema.CopyTo(_denoiserEma.AsWritableSpan());

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = ParamModel.GetParameters();
        ParamModel.SetParameters(baseParams);

        var condition = ComputeTaskCondition(task.SupportInput);
        var generated = SampleDDPM(condition, _algoOptions.SamplingSteps);
        var adaptedParams = ApplyCompressedDelta(baseParams, generated);

        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Full DDPM sampling (reverse process) with standard variance.
    /// </summary>
    private Vector<T> SampleDDPM(Vector<T> condition, int numSteps)
    {
        var wt = SampleGaussian(_compressedDim);
        int stepSize = Math.Max(1, _numTimesteps / Math.Max(numSteps, 1));

        for (int step = _numTimesteps - 1; step >= 0; step -= stepSize)
        {
            int t = step;
            var predictedNoise = PredictNoise(wt, condition, t, useEma: true);
            wt = PosteriorStep(wt, predictedNoise, t);
        }

        return wt;
    }

    /// <summary>
    /// The posterior mean μ_θ = (1/√α_t)(w_t - β_t/(√(1-ᾱ_t) + 1e-10) * ε_θ), plus √β̃_t z for t &gt; 0.
    /// </summary>
    private Vector<T> PosteriorStep(Vector<T> wt, Vector<T> predictedNoise, int t)
    {
        T sqrtAlpha = NumOps.Sqrt(_alphas[t]);
        T betaOverSqrt = NumOps.Divide(_betas[t], NumOps.Add(_sqrtOneMinusAlphasCumprod[t], NumOps.FromDouble(1e-10)));

        var next = Engine.Divide(Engine.Subtract(wt, Engine.Multiply(predictedNoise, betaOverSqrt)), sqrtAlpha);

        // Add noise with posterior variance (except at t=0)
        if (t > 0)
        {
            T sigma = NumOps.Sqrt(_posteriorVariance[t]);
            var z = SampleGaussian(_compressedDim);
            next = Engine.Add(next, Engine.Multiply(z, sigma));
        }

        return next;
    }

    private Vector<T> ComputeTargetDelta(Vector<T> baseParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var current = baseParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(current);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            current = ApplyInnerGradients(current, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        // delta_i = (theta' - theta)[i * stride]; i * stride < paramDim since compressedDim <= paramDim
        int stride = Math.Max(1, _paramDim / _compressedDim);
        var full = Engine.Subtract(current.GetSubVector(0, _compressedDim * stride), baseParams.GetSubVector(0, _compressedDim * stride));
        var delta = Engine.TensorSlice(Tensor<T>.FromVector(full, new[] { _compressedDim, stride }),
            new[] { 0, 0 }, new[] { _compressedDim, 1 }).ToVector();
        return delta;
    }

    private Vector<T> ComputeTaskCondition(TInput supportInput)
    {
        var features = ConvertToVector(MetaModel.Predict(supportInput));
        int featureDim = Math.Min(_paramDim, MaxFeatureDim);
        var condition = new Vector<T>(_condDim);
        if (features == null) return condition;

        // tanh(W f + b), W stored [condDim, featureDim], reading the first min(|f|, featureDim) features
        int biasOffset = featureDim * _condDim;
        var bias = _taskEncoderParams.GetSubVector(biasOffset, _condDim);
        int fLen = Math.Min(features.Length, featureDim);
        if (fLen == 0) return VectorTanh(bias);

        var weights = Engine.TensorSlice(
            Tensor<T>.FromVector(_taskEncoderParams.GetSubVector(0, biasOffset), new[] { _condDim, featureDim }),
            new[] { 0, 0 }, new[] { _condDim, fLen });
        var input = Tensor<T>.FromVector(features.Length == fLen ? features : features.GetSubVector(0, fLen), new[] { fLen, 1 });
        condition = Engine.Add(Engine.TensorMatMul(weights, input).ToVector(), bias);
        return VectorTanh(condition);
    }

    /// <summary>
    /// The denoiser MLP: relu(W1 [w_t, c, t / T] + b1) -> W2 h + b2, parameters laid out [W1 | b1 | W2 | b2]
    /// with W1 [hidden, inputDim] and W2 [compressedDim, hidden].
    /// </summary>
    private Vector<T> PredictNoise(Vector<T> wt, Vector<T> condition, int timestep, bool useEma)
    {
        var denoiser = useEma ? _denoiserEma : _denoiserParams;
        int inputDim = _compressedDim + _condDim + 1;
        int hidden = Math.Max(64, _compressedDim);

        var input = Vector<T>.Concatenate(
            wt.GetSubVector(0, _compressedDim),
            condition.GetSubVector(0, _condDim),
            new Vector<T>(new[] { NumOps.FromDouble((double)timestep / _numTimesteps) }));

        // Hidden layer (ReLU)
        int b1Off = inputDim * hidden;
        var w1 = Tensor<T>.FromVector(denoiser.GetSubVector(0, b1Off), new[] { hidden, inputDim });
        var h = Engine.ReLU(Engine.Add(
            Engine.TensorMatMul(w1, Tensor<T>.FromVector(input, new[] { inputDim, 1 })).ToVector(),
            denoiser.GetSubVector(b1Off, hidden)));

        // Output layer
        int w2Off = b1Off + hidden;
        int b2Off = w2Off + hidden * _compressedDim;
        var w2 = Tensor<T>.FromVector(denoiser.GetSubVector(w2Off, hidden * _compressedDim), new[] { _compressedDim, hidden });
        return Engine.Add(
            Engine.TensorMatMul(w2, Tensor<T>.FromVector(h, new[] { hidden, 1 })).ToVector(),
            denoiser.GetSubVector(b2Off, _compressedDim));
    }

    /// <summary>√ᾱ_t * w_0 + √(1-ᾱ_t) * ε.</summary>
    private Vector<T> AddNoise(Vector<T> clean, Vector<T> noise, int t)
        => Engine.Add(
            Engine.Multiply(clean.GetSubVector(0, _compressedDim), _sqrtAlphasCumprod[t]),
            Engine.Multiply(noise, _sqrtOneMinusAlphasCumprod[t]));

    /// <summary>mean_i (a_i - b_i)^2.</summary>
    private T MeanSquaredDifference(Vector<T> a, Vector<T> b)
    {
        var diff = Engine.Subtract(a, b);
        return NumOps.Divide(Engine.DotProduct(diff, diff), NumOps.FromDouble(diff.Length));
    }

    /// <summary>
    /// theta + delta_i over [i * stride, (i + 1) * stride) for each compressed entry; the tail beyond
    /// compressedDim * stride is unchanged.
    /// </summary>
    private Vector<T> ApplyCompressedDelta(Vector<T> baseParams, Vector<T> delta)
    {
        int stride = Math.Max(1, _paramDim / _compressedDim);
        int covered = _compressedDim * stride;
        var expanded = Engine.Reshape(Engine.TensorBroadcastTo(
            Tensor<T>.FromVector(delta.GetSubVector(0, _compressedDim), new[] { _compressedDim, 1 }),
            new[] { _compressedDim, stride }), new[] { covered }).ToVector();
        if (covered < _paramDim)
            expanded = Vector<T>.Concatenate(expanded, new Vector<T>(_paramDim - covered));
        return Engine.Add(baseParams.GetSubVector(0, _paramDim), expanded);
    }

    private double ComputeDDPMLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = ParamModel.GetParameters();
        var losses = new List<T>(taskBatch.Tasks.Length);

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(baseParams);
            var target = ComputeTargetDelta(baseParams, task);
            int t = RandomGenerator.Next(_numTimesteps);
            var eps = SampleGaussian(_compressedDim);

            var noised = AddNoise(target, eps, t);
            var cond = ComputeTaskCondition(task.SupportInput);
            var pred = PredictNoise(noised, cond, t, useEma: false);
            losses.Add(MeanSquaredDifference(eps, pred));
        }

        ParamModel.SetParameters(baseParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }

    /// <summary>Standard-normal draws, one SampleNormal per entry in order (the seeded stream is unchanged).</summary>
    private Vector<T> SampleGaussian(int length)
    {
        // SampleNormal() * 1.0 is exactly SampleNormal(), so the draws match master's one per entry in order.
        var arr = new Vector<T>(length);
        InitGaussian(arr, 1.0);
        return arr;
    }

    private void InitGaussian(Vector<T> v, double stdDev)
    {
        for (int i = 0; i < v.Length; i++)
            v[i] = NumOps.FromDouble(SampleNormal() * stdDev);
    }
}
