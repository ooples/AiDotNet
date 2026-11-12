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
/// Implementation of MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning
/// (Zhang et al., AAAI 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// MetaDiff replaces the gradient-based inner loop of MAML with a learned diffusion-based
/// denoising process. Starting from Gaussian noise, a task-conditional denoising network
/// (TCUNet) iteratively produces task-specific weight parameters. The key insight is that
/// gradient descent (random init → optimal weights) is analogous to the diffusion reverse
/// process (noise → clean signal).
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Noise schedule: β_t linearly from β_start to β_end, α_t = 1 - β_t, ᾱ_t = Π α_s
///
/// Training (outer loop):
///   1. For task τ: compute "target weights" w_0 via support-set gradient adaptation
///   2. Sample t ~ Uniform(1, T), ε ~ N(0, I)
///   3. Noised weights: w_t = √ᾱ_t * w_0 + √(1-ᾱ_t) * ε
///   4. Task condition: c = TaskEncoder(support_features)
///   5. Predicted noise: ε̂ = Denoiser(w_t, c, t)
///   6. Loss = ||ε - ε̂||²
///
/// Inference (adaptation):
///   1. Start from w_T ~ N(0, I)
///   2. For t = T, T-1, ..., 1:
///      w_{t-1} = (1/√α_t)(w_t - β_t/√(1-ᾱ_t) * Denoiser(w_t, c, t)) + σ_t * z
///   3. Adapted params: θ' = θ_base + w_0
/// </code>
/// </para>
/// <para><b>Key advantage:</b> GPU memory is constant regardless of adaptation steps
/// (unlike MAML which scales linearly). The denoising process can be run for as many
/// steps as desired without backpropagating through the full chain.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning",
    "https://arxiv.org/abs/2307.16424",
    Year = 2024,
    Authors = "Baoquan Zhang, Chuyao Luo, Demin Yu, Xutao Li, Huiwei Lin, Yunming Ye, Bowen Zhang")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaDiffAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MetaDiffOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>Noise schedule: β_t values.</summary>
    private readonly Vector<T> _betas;

    /// <summary>α_t = 1 - β_t.</summary>
    private readonly Vector<T> _alphas;

    /// <summary>ᾱ_t = cumulative product of α.</summary>
    private readonly Vector<T> _alphasCumprod;

    /// <summary>Denoiser network parameters (task-conditional noise predictor).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _denoiserParams;

    /// <summary>Task encoder parameters: maps support features to conditioning vector.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _taskEncoderParams;

    private readonly int _paramDim;
    private readonly int _condDim;
    private readonly int _compressedDim;
    private readonly int _diffusionSteps;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaDiff;

    public MetaDiffAlgorithm(MetaDiffOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.BetaStart <= 0 || options.BetaEnd <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "BetaStart and BetaEnd must be positive.");
        if (options.BetaStart >= 1.0 || options.BetaEnd >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(options), "BetaStart and BetaEnd must be less than 1.0.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _condDim = Math.Max(1, options.TaskConditionDim);
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));
        _compressedDim = Math.Min(_paramDim, 128);
        _diffusionSteps = Math.Max(1, options.DiffusionSteps);

        // Compute linear noise schedule
        // beta_t = start + (end - start) * t / max(T - 1, 1), alpha_t = 1 - beta_t, alphaBar_t = prod_{s<=t} alpha_s
        var betas = Engine.TensorAddScalar(
            Engine.TensorDivideScalar(
                Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, _diffusionSteps, 1), NumOps.FromDouble(options.BetaEnd - options.BetaStart)),
                NumOps.FromDouble(Math.Max(_diffusionSteps - 1, 1))),
            NumOps.FromDouble(options.BetaStart));
        var alphas = Engine.TensorSubtract(Engine.TensorOnesLike(betas), betas);
        _betas = betas.ToVector();
        _alphas = alphas.ToVector();
        _alphasCumprod = Engine.TensorCumProd(alphas, 0).ToVector();

        // Task encoder: feature_dim → condDim (linear + tanh)
        int featureDim = Math.Min(_paramDim, 64);
        int encoderSize = featureDim * _condDim + _condDim;
        _taskEncoderParams = new Vector<T>(encoderSize);
        InitGaussian(_taskEncoderParams, 1.0 / Math.Sqrt(featureDim));

        // Denoiser: (compressedDim + condDim + 1_timestep) → compressedDim
        // Simple MLP: input → hidden → output
        int denoiserInput = _compressedDim + _condDim + 1;
        int hidden = Math.Max(32, _compressedDim);
        int denoiserSize = denoiserInput * hidden + hidden + hidden * _compressedDim + _compressedDim;
        _denoiserParams = new Vector<T>(denoiserSize);
        InitGaussian(_denoiserParams, 1.0 / Math.Sqrt(denoiserInput));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(baseParams);

            // Step 1: Compute "target weights" w_0 via gradient adaptation on support set
            var targetDelta = ComputeTargetDelta(baseParams, task);

            // Step 2: Sample random timestep and noise
            int t = RandomGenerator.Next(_diffusionSteps);
            var noise = SampleGaussianArray(_compressedDim);

            // Step 3: Add noise to target delta: w_t = √ᾱ_t * w_0 + √(1-ᾱ_t) * ε
            var noisedDelta = AddNoise(targetDelta, noise, t);

            // Step 4: Compute task conditioning
            var taskCondition = ComputeTaskCondition(task.SupportInput);

            // Step 5: Predict noise
            var predictedNoise = PredictNoise(noisedDelta, taskCondition, t);

            // Step 6: Noise prediction loss (MSE)
            T noiseLoss = MeanSquaredDifference(noise, predictedNoise);

            // Also evaluate task performance for meta-gradient on base params
            var denoised = RunDenoisingInference(taskCondition, Math.Min(_algoOptions.SamplingSteps, 5));
            var adaptedParams = ApplyCompressedDelta(baseParams, denoised);
            ParamModel.SetParameters(adaptedParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            var totalLoss = NumOps.Add(queryLoss, noiseLoss);
            losses.Add(totalLoss);

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update base params
        ParamModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            ParamModel.SetParameters(ApplyOuterGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update denoiser and task encoder via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _denoiserParams, _algoOptions.OuterLearningRate, ComputeDiffusionLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _taskEncoderParams, _algoOptions.OuterLearningRate, ComputeDiffusionLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var baseParams = ParamModel.GetParameters();

        // Compute task conditioning from support set
        var taskCondition = ComputeTaskCondition(task.SupportInput);

        // Run full denoising inference
        var denoisedDelta = RunDenoisingInference(taskCondition, _algoOptions.SamplingSteps);
        var adaptedParams = ApplyCompressedDelta(baseParams, denoisedDelta);

        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes target parameter delta by gradient adaptation on support set (compressed).
    /// </summary>
    private Vector<T> ComputeTargetDelta(Vector<T> baseParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = baseParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(currentParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            currentParams = ApplyInnerGradients(currentParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        // delta_i = (theta' - theta)[i * stride]; i * stride < paramDim since compressedDim <= paramDim
        int stride = Math.Max(1, _paramDim / _compressedDim);
        var full = Engine.Subtract(currentParams.GetSubVector(0, _compressedDim * stride), baseParams.GetSubVector(0, _compressedDim * stride));
        var delta = Engine.TensorSlice(Tensor<T>.FromVector(full, new[] { _compressedDim, stride }),
            new[] { 0, 0 }, new[] { _compressedDim, 1 }).ToVector();
        return delta;
    }

    /// <summary>
    /// Computes task conditioning vector from support features.
    /// </summary>
    private Vector<T> ComputeTaskCondition(TInput supportInput)
    {
        var features = ConvertToVector(MetaModel.Predict(supportInput));
        int featureDim = Math.Min(_paramDim, 64);
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
    /// Denoiser MLP: predicts noise given noised weights, task condition, and timestep.
    /// Architecture: concat(w_t, c, t/T) → hidden(ReLU) → output.
    /// </summary>
    /// <summary>
    /// The denoiser MLP: relu(W1 [w_t, c, t / T] + b1) -> W2 h + b2, parameters laid out [W1 | b1 | W2 | b2]
    /// with W1 [hidden, inputDim] and W2 [compressedDim, hidden].
    /// </summary>
    private Vector<T> PredictNoise(Vector<T> noisedDelta, Vector<T> taskCondition, int timestep)
    {
        int inputDim = _compressedDim + _condDim + 1;
        int hidden = Math.Max(32, _compressedDim);

        // Build input vector
        var input = Vector<T>.Concatenate(
            noisedDelta.GetSubVector(0, _compressedDim),
            taskCondition.GetSubVector(0, _condDim),
            new Vector<T>(new[] { NumOps.FromDouble((double)timestep / _diffusionSteps) })); // Normalized timestep

        // Layer 1: input → hidden (ReLU)
        int b1Offset = inputDim * hidden;
        var w1 = Tensor<T>.FromVector(_denoiserParams.GetSubVector(0, b1Offset), new[] { hidden, inputDim });
        var h = Engine.ReLU(Engine.Add(
            Engine.TensorMatMul(w1, Tensor<T>.FromVector(input, new[] { inputDim, 1 })).ToVector(),
            _denoiserParams.GetSubVector(b1Offset, hidden)));

        // Layer 2: hidden → output
        int w2Offset = b1Offset + hidden;
        int b2Offset = w2Offset + hidden * _compressedDim;
        var w2 = Tensor<T>.FromVector(_denoiserParams.GetSubVector(w2Offset, hidden * _compressedDim), new[] { _compressedDim, hidden });
        return Engine.Add(
            Engine.TensorMatMul(w2, Tensor<T>.FromVector(h, new[] { hidden, 1 })).ToVector(),
            _denoiserParams.GetSubVector(b2Offset, _compressedDim));
    }

    /// <summary>√ᾱ_t * w_0 + √(1-ᾱ_t) * ε.</summary>
    private Vector<T> AddNoise(Vector<T> clean, Vector<T> noise, int t)
    {
        T alphaBar = _alphasCumprod[t];
        return Engine.Add(
            Engine.Multiply(clean.GetSubVector(0, _compressedDim), NumOps.Sqrt(alphaBar)),
            Engine.Multiply(noise, NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaBar))));
    }

    /// <summary>mean_i (a_i - b_i)^2.</summary>
    private T MeanSquaredDifference(Vector<T> a, Vector<T> b)
    {
        var diff = Engine.Subtract(a, b);
        return NumOps.Divide(Engine.DotProduct(diff, diff), NumOps.FromDouble(diff.Length));
    }

    private Vector<T> RunDenoisingInference(Vector<T> taskCondition, int numSteps)
    {
        // Start from pure noise
        var wt = SampleGaussianArray(_compressedDim);
        int stepSize = Math.Max(1, _diffusionSteps / Math.Max(numSteps, 1));

        for (int step = _diffusionSteps - 1; step >= 0; step -= stepSize)
        {
            int t = step;
            var predictedNoise = PredictNoise(wt, taskCondition, t);

            T sqrtAlpha = NumOps.Sqrt(_alphas[t]);
            T betaOverSqrt = NumOps.Divide(_betas[t],
                NumOps.Sqrt(NumOps.Add(NumOps.Subtract(NumOps.One, _alphasCumprod[t]), NumOps.FromDouble(1e-10))));

            wt = Engine.Divide(Engine.Subtract(wt, Engine.Multiply(predictedNoise, betaOverSqrt)), sqrtAlpha);

            // Add noise (except for t=0)
            if (t > 0)
            {
                T sigma = NumOps.Sqrt(_betas[t]);
                var z = SampleGaussianArray(_compressedDim);
                wt = Engine.Add(wt, Engine.Multiply(z, sigma));
            }
        }

        return wt;
    }

    /// <summary>
    /// theta + delta_i over [i * stride, (i + 1) * stride) for each compressed entry; the tail beyond
    /// compressedDim * stride is unchanged.
    /// </summary>
    private Vector<T> ApplyCompressedDelta(Vector<T> baseParams, Vector<T> compressedDelta)
    {
        int stride = Math.Max(1, _paramDim / _compressedDim);
        int covered = _compressedDim * stride;
        var expanded = Engine.Reshape(Engine.TensorBroadcastTo(
            Tensor<T>.FromVector(compressedDelta.GetSubVector(0, _compressedDim), new[] { _compressedDim, 1 }),
            new[] { _compressedDim, stride }), new[] { covered }).ToVector();
        if (covered < _paramDim)
            expanded = Vector<T>.Concatenate(expanded, new Vector<T>(_paramDim - covered));
        return Engine.Add(baseParams.GetSubVector(0, _paramDim), expanded);
    }

    /// <summary>Standard-normal draws, one SampleNormal per entry in order (the seeded stream is unchanged).</summary>
    private Vector<T> SampleGaussianArray(int length)
    {
        var arr = new Vector<T>(length);
        for (int i = 0; i < length; i++)
            arr[i] = NumOps.FromDouble(SampleNormal());
        return arr;
    }

    private double ComputeDiffusionLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = ParamModel.GetParameters();
        var losses = new List<T>(taskBatch.Tasks.Length);

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(baseParams);
            var targetDelta = ComputeTargetDelta(baseParams, task);
            int t = RandomGenerator.Next(_diffusionSteps);
            var noise = SampleGaussianArray(_compressedDim);

            var noised = AddNoise(targetDelta, noise, t);
            var cond = ComputeTaskCondition(task.SupportInput);
            var pred = PredictNoise(noised, cond, t);
            losses.Add(MeanSquaredDifference(noise, pred));
        }

        ParamModel.SetParameters(baseParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }

    private void InitGaussian(Vector<T> v, double stdDev)
    {
        for (int i = 0; i < v.Length; i++)
            v[i] = NumOps.FromDouble(SampleNormal() * stdDev);
    }
}
