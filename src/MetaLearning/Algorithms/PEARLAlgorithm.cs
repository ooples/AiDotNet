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
/// Implementation of PEARL: Probabilistic Embeddings for Actor-critic RL
/// (Rakelly et al., ICML 2019).
/// </summary>
/// <remarks>
/// <para>
/// PEARL infers a latent task variable z from context (support data) using a probabilistic
/// encoder q(z|c). The posterior is permutation invariant: a product of independent Gaussian
/// factors, one per context example (paper Eq. 4), and z is sampled via reparameterization.
/// The model is conditioned on z through parameter modulation (a generic model has no input slot
/// for z). At test time, z is inferred from support data without gradient updates.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Context encoder: q(z|c) = Π_n N(μ_n, σ_n²)   (product of per-example factors, Eq. 4)
///   c_n = features of support pair n (its loss gradient, first EncoderHiddenDim entries)
///   μ_n, log(σ_n²) = encoder MLP(c_n)
///   σ² = 1 / Σ_n σ_n⁻²,  μ = σ² Σ_n μ_n / σ_n²
/// Task inference:
///   1. z ~ q(z|support)
///   2. Modulate params: θ_z = θ_0 + W_z * z  (linear projection of z)
///   (no gradient steps by default; AdaptationSteps &gt; 0 adds optional fine-tuning)
/// Outer loop:
///   L = E_z[L_query(θ_z)] + β * KL(q(z|c) || N(0,I))
///   Update θ_0 by gradient; encoder and W_z by SPSA on the same objective
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables",
    "https://arxiv.org/abs/1903.08254",
    Year = 2019,
    Authors = "Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, Deirdre Quillen")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class PEARLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly PEARLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _latentDim;
    private readonly int _compressedDim;

    private const double SpsaLearningRateMultiplier = 0.1;

    /// <summary>Encoder parameters: maps compressed gradient → (μ, log_σ²) of size 2*latentDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _encoderParams;

    /// <summary>Projection matrix W_z: maps z (latentDim) → parameter delta (compressedDim).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _projectionParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.PEARL;

    public PEARLAlgorithm(PEARLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LatentDim must be positive.");
        if (options.EncoderHiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "EncoderHiddenDim must be positive.");
        if (options.NumPosteriorSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumPosteriorSamples must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _latentDim = options.LatentDim;
        _compressedDim = Math.Min(_paramDim, options.EncoderHiddenDim);

        // Encoder: compressed context → hidden → (μ, log_σ²)
        // N(0, 0.01^2) draws: one SampleNormal per weight, in order; the loops only store the draws.
        int encoderSize = _compressedDim * _algoOptions.EncoderHiddenDim + _algoOptions.EncoderHiddenDim * 2 * _latentDim;
        var encoderDraws = new Vector<T>(encoderSize);
        for (int i = 0; i < encoderSize; i++)
            encoderDraws[i] = NumOps.FromDouble(SampleNormal());
        _encoderParams = Engine.Multiply(encoderDraws, NumOps.FromDouble(0.01));

        // Projection: z → parameter delta
        var projectionDraws = new Vector<T>(_latentDim * _compressedDim);
        for (int i = 0; i < projectionDraws.Length; i++)
            projectionDraws[i] = NumOps.FromDouble(SampleNormal());
        _projectionParams = Engine.Multiply(projectionDraws, NumOps.FromDouble(0.01));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Encode context from support set
            var (mu, logVar) = Encode(ComputeContext(task.SupportInput, task.SupportOutput, initParams));

            var sampleLosses = new List<T>();
            var sampleGrads = new List<Vector<T>>();
            for (int s = 0; s < _algoOptions.NumPosteriorSamples; s++)
            {
                // epsilon ~ N(0, I): one SampleNormal per latent dimension, in order; the loop only stores draws.
                var eps = new Vector<T>(_latentDim);
                for (int i = 0; i < _latentDim; i++)
                    eps[i] = NumOps.FromDouble(SampleNormal());
                var modulatedParams = Infer(initParams, task, Reparameterize(mu, logVar, eps));
                ParamModel.SetParameters(modulatedParams);
                sampleLosses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
                sampleGrads.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }

            losses.Add(WithKL(ComputeMean(sampleLosses), mu, logVar));
            metaGradients.Add(AverageVectors(sampleGrads));
        }
        // Outer loop
        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        // Update encoder and projection via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputePEARLLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _projectionParams, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputePEARLLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var (mu, _) = Encode(ComputeContext(task.SupportInput, task.SupportOutput, initParams));

        // Use posterior mean (no sampling for deterministic adaptation)
        var modulatedParams = Infer(initParams, task, mu);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, modulatedParams);
    }

    /// <summary>
    /// theta_z = theta_0 + W_z z (the compressed delta tiled over the parameters), then the optional
    /// <c>AdaptationSteps</c> fine-tuning steps on the support set (none in the paper).
    /// </summary>
    private Vector<T> Infer(Vector<T> initParams, IMetaLearningTask<T, TInput, TOutput> task, Vector<T> z)
    {
        var delta = ProjectZ(z);
        int copies = (_paramDim + _compressedDim - 1) / _compressedDim;
        var modulatedParams = Engine.Add(initParams,
            Vector<T>.Concatenate(Enumerable.Repeat(delta, copies).ToArray()).GetSubVector(0, _paramDim));

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(modulatedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            modulatedParams = ApplyInnerGradients(modulatedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        return modulatedParams;
    }

    /// <summary>
    /// Per-example context features c_n as the columns of a [compressedDim, N] tensor: the loss gradient of each
    /// support pair at <paramref name="currentParams"/>, truncated to the encoder's input width.
    /// </summary>
    private Tensor<T> ComputeContext(TInput supportInput, TOutput supportOutput, Vector<T> currentParams)
    {
        ParamModel.SetParameters(currentParams);
        int n = InputHelper<T, TInput>.GetBatchSize(supportInput);
        var columns = new Tensor<T>[Math.Max(n, 1)];
        if (n == 0)
        {
            columns[0] = Tensor<T>.FromVector(Truncate(ComputeGradients(MetaModel, supportInput, supportOutput)));
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                var index = new[] { i };
                var grad = ComputeGradients(MetaModel,
                    InputHelper<T, TInput>.GetBatch(supportInput, index),
                    InputHelper<T, TOutput>.GetBatch(supportOutput, index));
                columns[i] = Tensor<T>.FromVector(Truncate(grad));
            }
        }

        return Engine.TensorStack(columns, 1);
    }

    /// <summary>The first compressedDim entries, zero-padded if the gradient is shorter.</summary>
    private Vector<T> Truncate(Vector<T> grad)
        => grad.Length >= _compressedDim
            ? grad.GetSubVector(0, _compressedDim)
            : Vector<T>.Concatenate(new[] { grad, new Vector<T>(_compressedDim - grad.Length) });

    /// <summary>
    /// The product-of-Gaussians posterior (Eq. 4): every context column goes through the encoder MLP to a factor
    /// N(μ_n, σ_n²), and the factors are multiplied: σ² = 1 / Σ σ_n⁻², μ = σ² Σ μ_n σ_n⁻². Returns μ and log σ².
    /// </summary>
    internal (Vector<T> Mu, Vector<T> LogVar) Encode(Tensor<T> contexts)
    {
        int hiddenDim = _algoOptions.EncoderHiddenDim;
        int l1 = _compressedDim * hiddenDim;
        int head = _latentDim * hiddenDim;
        var w1 = Tensor<T>.FromVector(_encoderParams.GetSubVector(0, l1), new[] { hiddenDim, _compressedDim });
        var wMu = Tensor<T>.FromVector(_encoderParams.GetSubVector(l1, head), new[] { _latentDim, hiddenDim });
        var wLogVar = Tensor<T>.FromVector(_encoderParams.GetSubVector(l1 + head, head), new[] { _latentDim, hiddenDim });

        // Layer 1: context → hidden (ReLU); layer 2: hidden → (μ_n, log σ_n²), log σ_n² clamped to [-10, 2].
        var hidden = Engine.TensorReLU(Engine.TensorMatMul(w1, contexts));
        var muN = Engine.TensorMatMul(wMu, hidden);
        var logVarN = Engine.TensorClamp(Engine.TensorMatMul(wLogVar, hidden), NumOps.FromDouble(-10), NumOps.FromDouble(2));

        // Precisions are summed in the log domain: with m = max_n(-log σ_n²) the factors are exp(-log σ_n² - m) <= 1,
        // so Σ σ_n⁻² cannot overflow a narrow T (each raw factor reaches e^10 ≈ 22026, past Half's 65504 at n = 3).
        // log σ² = -(m + log Σ exp(-log σ_n² - m)); μ = Σ μ_n w_n / Σ w_n is unchanged by the common factor e^m.
        var negLogVarN = Engine.TensorNegate(logVarN);
        var shift = Engine.ReduceMax(negLogVarN, new[] { 1 }, keepDims: true);
        var weights = Engine.TensorExp(Engine.TensorSubtract(negLogVarN, Engine.TensorBroadcastTo(shift, new[] { _latentDim, negLogVarN.Shape[1] })));
        var weightSum = Engine.ReduceSum(weights, new[] { 1 }, keepDims: false);
        var weightedMu = Engine.ReduceSum(Engine.TensorMultiply(muN, weights), new[] { 1 }, keepDims: false);
        var mu = Engine.TensorDivide(weightedMu, weightSum);
        var logVar = Engine.TensorNegate(Engine.TensorAdd(shift.Reshape(new[] { _latentDim }), Engine.TensorLog(weightSum)));
        return (mu.ToVector(), logVar.ToVector());
    }

    /// <summary>loss + β KL(q(z|c) || N(0, I)).</summary>
    private T WithKL(T loss, Vector<T> mu, Vector<T> logVar)
        => NumOps.Add(loss, NumOps.Multiply(NumOps.FromDouble(_algoOptions.KLWeight), KLToStandardNormal(mu, logVar)));

    /// <summary>KL(N(μ, σ²) || N(0, I)) = ½ Σ (σ² + μ² − 1 − log σ²).</summary>
    private T KLToStandardNormal(Vector<T> mu, Vector<T> logVar)
    {
        var terms = Engine.Subtract(
            Engine.Add(Engine.Exp(logVar), Engine.Multiply(mu, mu)),
            Engine.Add(logVar, Engine.Fill(_latentDim, NumOps.One)));
        return NumOps.Multiply(NumOps.FromDouble(0.5), Engine.Sum(terms));
    }

    /// <summary>z = μ + exp(½ log σ²) ε for standard-normal draws ε.</summary>
    private Vector<T> Reparameterize(Vector<T> mu, Vector<T> logVar, Vector<T> eps)
    {
        return Engine.Add(mu, Engine.Multiply(Engine.Exp(Engine.Multiply(logVar, NumOps.FromDouble(0.5))), eps));
    }

    /// <summary>W_z z, with W_z stored [compressedDim, latentDim] row-major.</summary>
    private Vector<T> ProjectZ(Vector<T> z)
    {
        var w = Tensor<T>.FromVector(_projectionParams, new[] { _compressedDim, _latentDim });
        return Engine.TensorMatMul(w, Tensor<T>.FromVector(z, new[] { _latentDim, 1 })).ToVector();
    }

    private double ComputePEARLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var (mu, logVar) = Encode(ComputeContext(task.SupportInput, task.SupportOutput, initParams));
            ParamModel.SetParameters(Infer(initParams, task, mu));
            var loss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(WithKL(loss, mu, logVar));
        }

        ParamModel.SetParameters(initParams);
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
