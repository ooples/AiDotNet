using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning with
/// data-dependent prior construction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Flex-PAC-Bayes constructs a data-dependent prior by partially adapting the meta-parameters
/// on the support set (first K_prior steps), then continues adaptation with prior regularization
/// (K_bound steps). The "flex" parameter λ interpolates between PAC-Bayes (λ=1) and ERM (λ→0).
/// The prior data fraction f controls how much of the adaptation budget is allocated to prior
/// construction vs. posterior refinement.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Phase 1 — Construct data-dependent prior (f fraction of steps):
///   θ_prior = θ_meta
///   For K_prior = round(f * K) steps:
///     θ_prior ← θ_prior - η * ∇L(θ_prior; D_support)
///
/// Phase 2 — Adapt posterior with prior regularization ((1-f) fraction of steps):
///   θ_post = θ_prior
///   For K_bound steps:
///     grad_task = ∇L(θ_post; D_support)
///     grad_prior = (λ / (σ² * n)) * (θ_post - θ_prior)
///     θ_post ← θ_post - η * (grad_task + grad_prior)
///
/// Flex-PAC-Bayes loss:
///   L = (1-f) * L(θ_post; D_query) + f * L(θ_prior; D_query) + (λ * KL(θ_post || θ_prior)) / (2n)
///   KL = 0.5 * Σ_d (θ_post_d - θ_prior_d)² / σ²_d
/// </code>
/// </para>
/// </remarks>
public class FlexPACBayesAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly FlexPACBayesOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Meta-learned prior log-variance (per-parameter).</summary>
    private Vector<T> _priorLogVar;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FlexPACBayes;

    public FlexPACBayesAlgorithm(FlexPACBayesOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        _priorLogVar = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            _priorLogVar[d] = NumOps.FromDouble(options.InitialLogVariance);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var metaParams = MetaModel.GetParameters();

        // Split adaptation steps into prior and bound phases
        int totalSteps = _algoOptions.AdaptationSteps;
        int priorSteps = Math.Max(1, (int)(totalSteps * _algoOptions.PriorDataFraction));
        int boundSteps = Math.Max(1, totalSteps - priorSteps);

        foreach (var task in taskBatch.Tasks)
        {
            // Phase 1: Construct data-dependent prior via partial adaptation
            var thetaPrior = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) thetaPrior[d] = metaParams[d];

            for (int step = 0; step < priorSteps; step++)
            {
                MetaModel.SetParameters(thetaPrior);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                thetaPrior = ApplyGradients(thetaPrior, grad, _algoOptions.InnerLearningRate);
            }

            // Phase 2: Adapt posterior from prior with KL regularization
            var thetaPost = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) thetaPost[d] = thetaPrior[d];

            for (int step = 0; step < boundSteps; step++)
            {
                MetaModel.SetParameters(thetaPost);
                var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                for (int d = 0; d < _paramDim; d++)
                {
                    double diff = NumOps.ToDouble(thetaPost[d]) - NumOps.ToDouble(thetaPrior[d]);
                    double var_d = Math.Exp(NumOps.ToDouble(_priorLogVar[d])) + 1e-10;
                    double priorGrad = _algoOptions.FlexParameter * diff / var_d;
                    double combined = NumOps.ToDouble(taskGrad[d]) + priorGrad;
                    thetaPost[d] = NumOps.Subtract(thetaPost[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            // Compute flex loss on query set
            double f = _algoOptions.PriorDataFraction;

            MetaModel.SetParameters(thetaPost);
            double postLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));

            MetaModel.SetParameters(thetaPrior);
            double priorLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));

            // Weighted flex combination
            double flexLoss = (1.0 - f) * postLoss + f * priorLoss;

            // KL penalty: KL(θ_post || θ_prior) with learned variance
            double kl = ComputePointKL(thetaPost, thetaPrior);
            flexLoss += _algoOptions.KLCoefficient * kl / 2.0;

            losses.Add(NumOps.FromDouble(flexLoss));

            MetaModel.SetParameters(thetaPost);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update meta-parameters
        MetaModel.SetParameters(metaParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newParams = ApplyGradients(metaParams, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(newParams);
        }

        // Update prior log-variance via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _priorLogVar, _algoOptions.OuterLearningRate * 0.1, ComputeFlexPACBayesLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var metaParams = MetaModel.GetParameters();

        int totalSteps = _algoOptions.AdaptationSteps;
        int priorSteps = Math.Max(1, (int)(totalSteps * _algoOptions.PriorDataFraction));
        int boundSteps = Math.Max(1, totalSteps - priorSteps);

        // Phase 1: Construct data-dependent prior
        var thetaPrior = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) thetaPrior[d] = metaParams[d];

        for (int step = 0; step < priorSteps; step++)
        {
            MetaModel.SetParameters(thetaPrior);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            thetaPrior = ApplyGradients(thetaPrior, grad, _algoOptions.InnerLearningRate);
        }

        // Phase 2: Adapt posterior with prior regularization
        var thetaPost = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) thetaPost[d] = thetaPrior[d];

        for (int step = 0; step < boundSteps; step++)
        {
            MetaModel.SetParameters(thetaPost);
            var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int d = 0; d < _paramDim; d++)
            {
                double diff = NumOps.ToDouble(thetaPost[d]) - NumOps.ToDouble(thetaPrior[d]);
                double var_d = Math.Exp(NumOps.ToDouble(_priorLogVar[d])) + 1e-10;
                double priorGrad = _algoOptions.FlexParameter * diff / var_d;
                double combined = NumOps.ToDouble(taskGrad[d]) + priorGrad;
                thetaPost[d] = NumOps.Subtract(thetaPost[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
            }
        }

        MetaModel.SetParameters(metaParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, thetaPost);
    }

    /// <summary>
    /// Computes KL(Q || P) for point posteriors using the learned prior variance:
    /// KL = 0.5 * Σ_d (θ_post_d - θ_prior_d)² / σ²_d
    /// </summary>
    private double ComputePointKL(Vector<T> posterior, Vector<T> prior)
    {
        double kl = 0;
        for (int d = 0; d < _paramDim; d++)
        {
            double diff = NumOps.ToDouble(posterior[d]) - NumOps.ToDouble(prior[d]);
            double var_d = Math.Exp(NumOps.ToDouble(_priorLogVar[d])) + 1e-10;
            kl += diff * diff / var_d;
        }
        return 0.5 * kl;
    }

    private double ComputeFlexPACBayesLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var metaParams = MetaModel.GetParameters();

        int totalSteps = _algoOptions.AdaptationSteps;
        int priorSteps = Math.Max(1, (int)(totalSteps * _algoOptions.PriorDataFraction));
        int boundSteps = Math.Max(1, totalSteps - priorSteps);

        foreach (var task in taskBatch.Tasks)
        {
            var thetaPrior = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) thetaPrior[d] = metaParams[d];

            for (int step = 0; step < priorSteps; step++)
            {
                MetaModel.SetParameters(thetaPrior);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                thetaPrior = ApplyGradients(thetaPrior, grad, _algoOptions.InnerLearningRate);
            }

            var thetaPost = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) thetaPost[d] = thetaPrior[d];

            for (int step = 0; step < boundSteps; step++)
            {
                MetaModel.SetParameters(thetaPost);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                for (int d = 0; d < _paramDim; d++)
                {
                    double diff = NumOps.ToDouble(thetaPost[d]) - NumOps.ToDouble(thetaPrior[d]);
                    double var_d = Math.Exp(NumOps.ToDouble(_priorLogVar[d])) + 1e-10;
                    double combined = NumOps.ToDouble(grad[d]) + _algoOptions.FlexParameter * diff / var_d;
                    thetaPost[d] = NumOps.Subtract(thetaPost[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            double f = _algoOptions.PriorDataFraction;
            MetaModel.SetParameters(thetaPost);
            double postLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            MetaModel.SetParameters(thetaPrior);
            double priorLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));

            double kl = ComputePointKL(thetaPost, thetaPrior);
            totalLoss += (1.0 - f) * postLoss + f * priorLoss + _algoOptions.KLCoefficient * kl / 2.0;
        }

        MetaModel.SetParameters(metaParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
