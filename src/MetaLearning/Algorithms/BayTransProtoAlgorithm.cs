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
/// Implementation of BayTransProto: Bayesian Transductive Prototypical Networks.
/// </summary>
/// <remarks>
/// <para>
/// BayTransProto combines Bayesian posterior sampling with transductive refinement.
/// After standard inner-loop adaptation, the algorithm samples multiple parameter vectors
/// from a Gaussian posterior centered on the adapted params. Transductive refinement
/// then uses query-set gradients to iteratively update the posterior mean, leveraging
/// unlabeled query data to improve adaptation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Inner loop: θ_adapted = MAML adaptation on support
///
/// Posterior: q(θ) = N(θ_adapted, exp(log_var) * I)
/// Samples: θ_s ~ q(θ) for s = 1..S
///
/// Transductive refinement (T steps):
///   For each step:
///     Evaluate ensemble on query → ensemble loss
///     θ_adapted ← θ_adapted - TransductiveLR * ∇_query(ensemble_loss)
///
/// KL regularization: KL(q || N(θ_init, I))
/// L_meta = mean(ensemble_loss) + KLWeight * KL
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
[ResearchPaper("Transductive Information Maximization for Few-Shot Learning",
    "https://arxiv.org/abs/2008.11297",
    Year = 2020,
    Authors = "Malik Boudiaf, Ziko Imtiaz Masud, Jerome Rony, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class BayTransProtoAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly BayTransProtoOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Learned log-variance for the posterior.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _logVar;

    // The SPSA estimate of the log-variance gradient reuses these per-task posterior means and standard-normal
    // draws for the base and every perturbed loss (common random numbers), so each finite difference measures
    // only the change in _logVar through the reparameterized samples mean + exp(logVar / 2) * noise and the KL.
    private Vector<T>[] _spsaMeans = Array.Empty<Vector<T>>();
    private Vector<T>[] _spsaNoise = Array.Empty<Vector<T>>();

    /// <summary>The learned posterior log-variance, one entry per model parameter.</summary>
    internal Vector<T> PosteriorLogVariance => _logVar;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BayTransProto;

    public BayTransProtoAlgorithm(BayTransProtoOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _logVar = Engine.Fill(_paramDim, NumOps.FromDouble(options.InitialLogVar));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        // The setter is public, so re-check here: a nonpositive count would otherwise surface only after the
        // outer update had already changed the model (negative) or freeze the log-variance (zero).
        if (_algoOptions.NumPosteriorSamples <= 0)
        {
            throw new InvalidOperationException(
                $"NumPosteriorSamples must be positive, but is {_algoOptions.NumPosteriorSamples}.");
        }

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        T transductiveLR = NumOps.FromDouble(_algoOptions.TransductiveLR);
        T klScale = NumOps.FromDouble(_algoOptions.KLWeight / _paramDim);

        foreach (var task in taskBatch.Tasks)
        {
            // Phases 1 and 2: support adaptation, then transductive refinement of the posterior mean
            var adaptedMean = AdaptPosteriorMean(task, initParams, transductiveLR);

            // Final evaluation: ensemble loss
            var finalLosses = new List<T>();
            for (int s = 0; s < _algoOptions.NumPosteriorSamples; s++)
            {
                var sampled = SamplePosterior(adaptedMean);
                ParamModel.SetParameters(sampled);
                finalLosses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            }
            var ensembleLoss = ComputeMean(finalLosses);

            // KL divergence: KL(N(adapted, σ²) || N(init, I))
            T kl = DiagonalGaussianKL(adaptedMean, initParams);

            var totalLoss = NumOps.Add(ensembleLoss, NumOps.Multiply(klScale, kl));
            losses.Add(totalLoss);

            ParamModel.SetParameters(adaptedMean);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        // The posterior means do not depend on _logVar, so they are adapted once from the updated
        // initialization; the noise draws are fixed for the whole SPSA estimate.
        var updatedParams = ParamModel.GetParameters();
        _spsaMeans = taskBatch.Tasks.Select(task => AdaptPosteriorMean(task, updatedParams, transductiveLR)).ToArray();
        _spsaNoise = Enumerable.Range(0, _algoOptions.NumPosteriorSamples)
            .Select(_ => Engine.GenerateGaussianNoise(_paramDim, NumOps.Zero, NumOps.One, RandomGenerator.Next()))
            .ToArray();
        ParamModel.SetParameters(updatedParams);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _logVar, _algoOptions.OuterLearningRate * 0.01, ComputeBayTransLoss);

        _spsaMeans = Array.Empty<Vector<T>>();
        _spsaNoise = Array.Empty<Vector<T>>();
        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts the posterior mean to a task: inner-loop steps on the support set, then transductive steps on the
    /// query set. Leaves the model holding the last parameters it evaluated.
    /// </summary>
    private Vector<T> AdaptPosteriorMean(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams, T transductiveLR)
    {
        var adaptedMean = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedMean);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedMean = ApplyInnerGradients(adaptedMean, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        for (int tStep = 0; tStep < _algoOptions.TransductiveSteps; tStep++)
        {
            // Use query gradients to refine posterior mean
            ParamModel.SetParameters(adaptedMean);
            var queryGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));
            adaptedMean = Engine.Subtract(adaptedMean, Engine.Multiply(queryGrad, transductiveLR));
        }

        return adaptedMean;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedMean = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedMean);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedMean = ApplyInnerGradients(adaptedMean, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedMean);
    }

    private Vector<T> SamplePosterior(Vector<T> mean)
    {
        var sample = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            double std = Math.Exp(0.5 * NumOps.ToDouble(_logVar[d]));
            sample[d] = NumOps.Add(mean[d], NumOps.FromDouble(std * SampleNormal()));
        }
        return sample;
    }

    /// <summary>KL(N(mean, diag exp(logVar)) || N(prior, I)) = 0.5 * sum_d [v_d + (mean_d - prior_d)^2 - 1 - log v_d].</summary>
    private T DiagonalGaussianKL(Vector<T> mean, Vector<T> prior)
    {
        // Per dimension v_d - 1 - log v_d = expm1(log v_d) - log v_d is nonnegative and near zero when log v_d is,
        // so it is summed first; adding the squared mean shift to that small sum (rather than to a sum near D)
        // keeps a tiny shift visible when T is float.
        var diff = Engine.Subtract(mean, prior);
        T varianceTerm = Engine.Sum(Engine.Subtract(Engine.ExpM1(_logVar), _logVar));
        T kl = NumOps.Add(varianceTerm, Engine.DotProduct(diff, diff));
        return NumOps.Multiply(NumOps.FromDouble(0.5), kl);
    }

    /// <summary>
    /// The variational objective the posterior log-variance minimizes, averaged over the batch: the expected query
    /// loss under q(theta) = N(mean, diag exp(logVar)), estimated with the fixed reparameterized draws, plus
    /// KLWeight / D * KL(q || N(theta_init, I)) - the same ensemble-plus-KL loss meta-training reports.
    /// </summary>
    private double ComputeBayTransLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        int count = Math.Min(taskBatch.Tasks.Length, _spsaMeans.Length);
        if (count == 0 || _spsaNoise.Length == 0) return 0;

        var initParams = ParamModel.GetParameters();
        var objectives = new List<T>(count);
        for (int t = 0; t < count; t++)
        {
            objectives.Add(PosteriorObjective(taskBatch.Tasks[t], _spsaMeans[t], initParams));
        }

        ParamModel.SetParameters(initParams);
        return NumOps.ToDouble(ComputeMean(objectives));
    }

    /// <summary>Ensemble query loss over the reparameterized posterior samples plus the scaled KL term.</summary>
    private T PosteriorObjective(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> mean, Vector<T> initParams)
    {
        var std = Engine.Exp(Engine.Multiply(_logVar, NumOps.FromDouble(0.5)));
        var losses = new List<T>(_spsaNoise.Length);
        foreach (var noise in _spsaNoise)
        {
            ParamModel.SetParameters(Engine.Add(mean, Engine.Multiply(std, noise)));
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        T klScale = NumOps.FromDouble(_algoOptions.KLWeight / _paramDim);
        return NumOps.Add(ComputeMean(losses), NumOps.Multiply(klScale, DiagonalGaussianKL(mean, initParams)));
    }
}
