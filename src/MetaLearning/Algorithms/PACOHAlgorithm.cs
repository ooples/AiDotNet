using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

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
///     3. Loss_τ = L(θ_τ; D^q_τ)
///
/// Outer loop (PAC-Bayes bound optimization):
///   L_PACOH = (1/T) Σ_τ Loss_τ + (KL_coeff / T) * KL(Q_avg || P)
///   Where KL(Q_avg || P) = Σ_i [(μ_Q_i - μ_P_i)² / σ²_P_i + σ²_Q_i / σ²_P_i - 1 - ln(σ²_Q_i / σ²_P_i)]
///   Update μ_P, log(σ²_P) via gradient descent on L_PACOH
/// </code>
/// </para>
/// </remarks>
public class PACOHAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly PACOHOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>Prior mean (meta-learned). Same as MetaModel parameters.</summary>
    private Vector<T> _priorMean;

    /// <summary>Prior log-variance (meta-learned). Per-parameter log(σ²).</summary>
    private Vector<T> _priorLogVar;

    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.PACOH;

    public PACOHAlgorithm(PACOHOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        _priorMean = options.MetaModel.GetParameters();
        _priorLogVar = new Vector<T>(_paramDim);
        for (int i = 0; i < _paramDim; i++)
            _priorLogVar[i] = NumOps.FromDouble(options.InitialLogVariance);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var adaptedParamsList = new List<Vector<T>>();
        var metaGradients = new List<Vector<T>>();

        foreach (var task in taskBatch.Tasks)
        {
            // Initialize from prior mean
            var theta = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) theta[d] = _priorMean[d];

            // Inner loop: MAP estimation with prior regularization
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(theta);
                var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Prior gradient: (θ - μ_P) / σ²_P
                var priorGrad = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++)
                {
                    double diff = NumOps.ToDouble(theta[d]) - NumOps.ToDouble(_priorMean[d]);
                    double var_d = Math.Exp(NumOps.ToDouble(_priorLogVar[d]));
                    priorGrad[d] = NumOps.FromDouble(diff / (var_d + 1e-10));
                }

                // Combined gradient: task + KL regularization
                for (int d = 0; d < _paramDim; d++)
                {
                    double combined = NumOps.ToDouble(taskGrad[d]) + _algoOptions.KLCoefficient * NumOps.ToDouble(priorGrad[d]);
                    theta[d] = NumOps.Subtract(theta[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            adaptedParamsList.Add(theta);
            MetaModel.SetParameters(theta);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Compute PAC-Bayesian bound terms
        double klTerm = ComputeAggregateKL(adaptedParamsList);
        double pacBayesLoss = _algoOptions.KLCoefficient * klTerm / Math.Max(taskBatch.Tasks.Length, 1);

        // Update prior mean via meta-gradients
        MetaModel.SetParameters(_priorMean);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            _priorMean = ApplyGradients(_priorMean, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(_priorMean);
        }

        // Update prior log-variance via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _priorLogVar, _algoOptions.OuterLearningRate * 0.1, ComputePACOHLoss);

        // Add PAC-Bayes regularization to reported loss
        var totalLoss = NumOps.Add(ComputeMean(losses), NumOps.FromDouble(pacBayesLoss));
        return totalLoss;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var theta = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) theta[d] = _priorMean[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(theta);
            var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            var priorGrad = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
            {
                double diff = NumOps.ToDouble(theta[d]) - NumOps.ToDouble(_priorMean[d]);
                double var_d = Math.Exp(NumOps.ToDouble(_priorLogVar[d]));
                priorGrad[d] = NumOps.FromDouble(diff / (var_d + 1e-10));
            }

            for (int d = 0; d < _paramDim; d++)
            {
                double combined = NumOps.ToDouble(taskGrad[d]) + _algoOptions.KLCoefficient * NumOps.ToDouble(priorGrad[d]);
                theta[d] = NumOps.Subtract(theta[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
            }
        }

        MetaModel.SetParameters(_priorMean);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, theta);
    }

    /// <summary>
    /// Computes KL divergence between the empirical posterior (average of adapted params)
    /// and the prior: KL(Q || P) for diagonal Gaussians.
    /// </summary>
    private double ComputeAggregateKL(List<Vector<T>> adaptedParams)
    {
        if (adaptedParams.Count == 0) return 0;

        // Compute empirical posterior mean and variance
        var posteriorMean = new double[_paramDim];
        var posteriorVar = new double[_paramDim];

        foreach (var theta in adaptedParams)
            for (int d = 0; d < _paramDim; d++)
                posteriorMean[d] += NumOps.ToDouble(theta[d]);
        for (int d = 0; d < _paramDim; d++)
            posteriorMean[d] /= adaptedParams.Count;

        foreach (var theta in adaptedParams)
            for (int d = 0; d < _paramDim; d++)
            {
                double diff = NumOps.ToDouble(theta[d]) - posteriorMean[d];
                posteriorVar[d] += diff * diff;
            }
        for (int d = 0; d < _paramDim; d++)
            posteriorVar[d] = posteriorVar[d] / adaptedParams.Count + 1e-10;

        // KL(N(μ_Q, σ²_Q) || N(μ_P, σ²_P))
        double kl = 0;
        for (int d = 0; d < _paramDim; d++)
        {
            double priorVar = Math.Exp(NumOps.ToDouble(_priorLogVar[d])) + 1e-10;
            double diff = posteriorMean[d] - NumOps.ToDouble(_priorMean[d]);
            kl += (diff * diff + posteriorVar[d]) / priorVar - 1.0 + Math.Log(priorVar / posteriorVar[d]);
        }
        return 0.5 * kl;
    }

    private double ComputePACOHLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var adapted = new List<Vector<T>>();

        foreach (var task in taskBatch.Tasks)
        {
            var theta = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) theta[d] = _priorMean[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(theta);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                for (int d = 0; d < _paramDim; d++)
                {
                    double diff = NumOps.ToDouble(theta[d]) - NumOps.ToDouble(_priorMean[d]);
                    double var_d = Math.Exp(NumOps.ToDouble(_priorLogVar[d])) + 1e-10;
                    double combined = NumOps.ToDouble(grad[d]) + _algoOptions.KLCoefficient * diff / var_d;
                    theta[d] = NumOps.Subtract(theta[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            adapted.Add(theta);
            MetaModel.SetParameters(theta);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        double klPenalty = _algoOptions.KLCoefficient * ComputeAggregateKL(adapted) / Math.Max(taskBatch.Tasks.Length, 1);
        MetaModel.SetParameters(_priorMean);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1) + klPenalty;
    }
}
