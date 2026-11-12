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
/// Implementation of PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters
/// (Rothfuss et al., ICLR 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// PACOH meta-learns a Gaussian prior distribution N(μ_prior, diag(σ²_prior)) over model
/// parameters such that the PAC-Bayesian generalization bound is minimized. The key insight
/// is that optimizing the prior (not just the posterior) yields tighter generalization
/// guarantees for meta-learning.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: prior mean μ_P, prior log-variance log(σ²_P)
///
/// Inner loop (MAP estimation with learned prior):
///   For each task τ:
///     1. Initialize θ_τ = μ_P (from prior mean)
///     2. For each step:
///        grad_task = ∂L(θ_τ; D^s_τ)/∂θ_τ
///        grad_prior = (θ_τ - μ_P) / σ²_P   (prior regularization)
///        θ_τ ← θ_τ - η_inner * (grad_task + λ_KL * grad_prior)
///     3. Loss_τ = E_{θ ~ N(θ_τ, σ²_P)}[L(θ; D^q_τ)], averaged over NumPosteriorSamples draws
///
/// Outer loop (PAC-Bayes bound optimization):
///   L_PACOH = (1/T) Σ_τ Loss_τ + (KL_coeff / T) * (KL(Q_avg || P) + ln(2√T / δ))
///   Where KL(Q_avg || P) = Σ_i [(μ_Q_i - μ_P_i)² / σ²_P_i + σ²_Q_i / σ²_P_i - 1 - ln(σ²_Q_i / σ²_P_i)]
///   Update μ_P, log(σ²_P) via gradient descent on L_PACOH
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
[ResearchPaper("PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees",
    "https://arxiv.org/abs/2002.05551",
    Year = 2021,
    Authors = "Jonas Rothfuss, Vincent Fortuin, Martin Josifoski, Andreas Krause")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class PACOHAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly PACOHOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>Prior mean (meta-learned). Same as MetaModel parameters.</summary>
    private Vector<T> _priorMean;

    /// <summary>Prior log-variance (meta-learned). Per-parameter log(σ²).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _priorLogVar;

    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.PACOH;

    public PACOHAlgorithm(PACOHOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.InnerAndOuter)
    {
        if (options.Delta <= 0 || options.Delta >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Delta must lie in (0, 1).");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;

        _priorMean = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters();
        _priorLogVar = new Vector<T>(_paramDim);
        _priorLogVar.Fill(NumOps.FromDouble(options.InitialLogVariance));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var adaptedParamsList = new List<Vector<T>>();
        var metaGradients = new List<Vector<T>>();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: MAP estimation with prior regularization, from the prior mean
            var theta = AdaptToTask(task, clipTaskGradients: true);
            adaptedParamsList.Add(theta);

            // Gibbs risk of the task posterior on the query set, with its location gradient
            var (queryLoss, gradient) = PosteriorRisk(task, theta, withGradient: true);
            losses.Add(queryLoss);
            if (gradient is not null) metaGradients.Add(gradient);
        }

        // Compute PAC-Bayesian bound terms
        var pacBayesLoss = BoundPenalty(ComputeAggregateKL(adaptedParamsList), taskBatch.Tasks.Length);

        // Update prior mean via meta-gradients
        ParamModel.SetParameters(_priorMean);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            _priorMean = ApplyOuterGradients(_priorMean, avgGrad, _algoOptions.OuterLearningRate);
            ParamModel.SetParameters(_priorMean);
        }

        // Update prior log-variance via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _priorLogVar, _algoOptions.OuterLearningRate * 0.1, ComputePACOHLoss);

        // Add PAC-Bayes regularization to reported loss
        return NumOps.Add(ComputeMean(losses), pacBayesLoss);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var theta = AdaptToTask(task, clipTaskGradients: true);
        ParamModel.SetParameters(_priorMean);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, theta);
    }

    /// <summary>
    /// MAP adaptation from the prior mean: each step descends the support loss plus
    /// KLCoefficient * (theta - mu_P) / sigma_P^2, the gradient of the prior's negative log-density.
    /// </summary>
    private Vector<T> AdaptToTask(IMetaLearningTask<T, TInput, TOutput> task, bool clipTaskGradients)
    {
        var theta = _priorMean.Clone();
        var priorVariance = Engine.Exp(_priorLogVar) + NumOps.FromDouble(1e-10);
        var klCoefficient = NumOps.FromDouble(_algoOptions.KLCoefficient);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(theta);
            var taskGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            if (clipTaskGradients) taskGrad = ClipGradients(taskGrad);

            // Prior gradient: (θ - μ_P) / σ²_P, then combined gradient: task + KL regularization
            var priorGrad = Engine.Divide(Engine.Subtract(theta, _priorMean), priorVariance);
            var stepGradient = Engine.Add(taskGrad, Engine.Multiply(priorGrad, klCoefficient));
            theta = ApplyInnerGradients(theta, stepGradient, _algoOptions.InnerLearningRate, step == 0);
        }

        return theta;
    }

    /// <summary>
    /// Monte Carlo estimate of the Gibbs risk E_{θ ~ Q}[L(θ; D^q)] of the task posterior
    /// Q = N(θ_MAP, σ²_P) from <see cref="PACOHOptions{T, TInput, TOutput}.NumPosteriorSamples"/> draws
    /// (Rothfuss et al. 2021, Eq. 1), and the matching reparameterised gradient with respect to the location.
    /// </summary>
    private (T Loss, Vector<T>? Gradient) PosteriorRisk(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> theta, bool withGradient)
    {
        int samples = Math.Max(1, _algoOptions.NumPosteriorSamples);
        var location = Tensor<T>.FromVector(theta);
        var scale = Tensor<T>.FromVector(Engine.Exp(Engine.Multiply(_priorLogVar, NumOps.FromDouble(0.5))));
        var losses = new List<T>(samples);
        var gradients = new List<Vector<T>>(samples);

        for (int s = 0; s < samples; s++)
        {
            ParamModel.SetParameters(Engine.TensorNormal(location, scale, RandomGenerator.Next()).ToVector());
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            if (withGradient)
                gradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        return (ComputeMean(losses), withGradient ? AverageVectors(gradients) : null);
    }

    /// <summary>
    /// The PAC-Bayesian complexity term KLCoefficient * (KL(Q || P) + ln(2 sqrt(n) / δ)) / n over n tasks
    /// (Rothfuss et al. 2021, Thm. 1, with the confidence term ln(2 sqrt(n) / δ) of the
    /// <see cref="PACOHOptions{T, TInput, TOutput}.KLCoefficient"/> derivation).
    /// </summary>
    private T BoundPenalty(T kl, int taskCount)
    {
        int n = Math.Max(taskCount, 1);
        double confidence = Math.Log(2.0 * Math.Sqrt(n) / _algoOptions.Delta);
        return NumOps.Multiply(
            NumOps.FromDouble(_algoOptions.KLCoefficient / n), NumOps.Add(kl, NumOps.FromDouble(confidence)));
    }

    /// <summary>
    /// KL(N(μ_Q, σ²_Q) || N(μ_P, σ²_P)) where μ_Q and σ²_Q are the empirical mean and (population) variance of the
    /// adapted parameters across tasks.
    /// </summary>
    private T ComputeAggregateKL(List<Vector<T>> adaptedParams)
    {
        if (adaptedParams.Count == 0) return NumOps.Zero;

        var (posteriorMean, posteriorVar) = EmpiricalMoments(adaptedParams, _paramDim);
        posteriorVar = posteriorVar + NumOps.FromDouble(1e-10);

        // 0.5 Σ [((μ_Q - μ_P)² + σ²_Q) / σ²_P - 1 + ln(σ²_P / σ²_Q)]
        var priorVar = Engine.Exp(_priorLogVar) + NumOps.FromDouble(1e-10);
        var diff = Engine.Subtract(posteriorMean, _priorMean);
        var terms = Engine.Add(
            Engine.Divide(Engine.Add(Engine.Multiply(diff, diff), posteriorVar), priorVar),
            VectorLog(Engine.Divide(priorVar, posteriorVar)));
        return NumOps.Multiply(
            NumOps.FromDouble(0.5), NumOps.Subtract(Engine.Sum(terms), NumOps.FromDouble(_paramDim)));
    }

    /// <summary>The per-coordinate mean and population variance of <paramref name="vectors"/>.</summary>
    internal static (Vector<T> Mean, Vector<T> Variance) EmpiricalMoments(List<Vector<T>> vectors, int dimension)
    {
        var engine = AiDotNetEngine.Current;
        var stacked = engine.TensorStack(vectors.Select(v => Tensor<T>.FromVector(v)).ToArray(), 0);
        var mean = engine.ReduceMean(stacked, new[] { 0 }, true);
        var centered = engine.TensorSubtract(stacked, engine.TensorBroadcastTo(mean, new[] { vectors.Count, dimension }));
        var variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 0 }, false);
        return (mean.Reshape(dimension).ToVector(), variance.ToVector());
    }

    private double ComputePACOHLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var adapted = new List<Vector<T>>();

        foreach (var task in taskBatch.Tasks)
        {
            var theta = AdaptToTask(task, clipTaskGradients: false);
            adapted.Add(theta);
            losses.Add(PosteriorRisk(task, theta, withGradient: false).Loss);
        }

        var bound = losses.Count == 0
            ? NumOps.Zero
            : NumOps.Add(ComputeMean(losses), BoundPenalty(ComputeAggregateKL(adapted), taskBatch.Tasks.Length));
        ParamModel.SetParameters(_priorMean);
        return NumOps.ToDouble(bound);
    }
}
