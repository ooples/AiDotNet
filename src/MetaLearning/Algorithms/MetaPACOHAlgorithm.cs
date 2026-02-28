using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning with per-group
/// prior variances.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-PACOH extends PACOH by partitioning the parameter space into G groups and
/// learning independent prior log-variances {log(σ²_g)} for each group. This allows
/// different parts of the network (e.g., early vs late layers) to have different levels
/// of flexibility during task adaptation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: prior mean μ_P, per-group log-variances {log(σ²_g)}
///
/// Inner loop (MAP with group-specific regularization):
///   For each task τ:
///     θ_τ = μ_P
///     For each step:
///       grad_task = ∂L/∂θ
///       grad_prior_d = (θ_d - μ_P_d) / σ²_{g(d)}  (g(d) = group of dim d)
///       θ_τ ← θ_τ - η * (grad_task + λ * grad_prior)
///
/// Outer loop (hierarchical PAC-Bayes bound):
///   L = (1/T)Σ Loss_τ + (λ/T) * Σ_g KL_g(Q_g || P_g) + KL(P || P_0)
///   Update μ_P, {log(σ²_g)} via gradient descent
/// </code>
/// </para>
/// </remarks>
public class MetaPACOHAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaPACOHOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>Prior mean (meta-learned).</summary>
    private Vector<T> _priorMean;

    /// <summary>Per-group prior log-variances. Length = NumPriorGroups.</summary>
    private Vector<T> _groupLogVars;

    /// <summary>Group assignments: _groupOf[d] = group index for parameter d.</summary>
    private readonly int[] _groupOf;

    /// <summary>Number of parameters per group.</summary>
    private readonly int[] _groupSize;

    private readonly int _paramDim;
    private readonly int _numGroups;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaPACOH;

    public MetaPACOHAlgorithm(MetaPACOHOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _numGroups = Math.Min(options.NumPriorGroups, _paramDim);

        _priorMean = options.MetaModel.GetParameters();

        // Initialize per-group log-variances
        _groupLogVars = new Vector<T>(_numGroups);
        for (int g = 0; g < _numGroups; g++)
            _groupLogVars[g] = NumOps.FromDouble(options.InitialLogVariance);

        // Assign parameters to groups (even partition)
        _groupOf = new int[_paramDim];
        _groupSize = new int[_numGroups];
        for (int d = 0; d < _paramDim; d++)
        {
            int g = (int)((long)d * _numGroups / _paramDim);
            _groupOf[d] = g;
            _groupSize[g]++;
        }
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

            // Inner loop: MAP estimation with group-specific prior regularization
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(theta);
                var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Prior gradient with per-group variances
                for (int d = 0; d < _paramDim; d++)
                {
                    int g = _groupOf[d];
                    double diff = NumOps.ToDouble(theta[d]) - NumOps.ToDouble(_priorMean[d]);
                    double var_g = Math.Exp(NumOps.ToDouble(_groupLogVars[g])) + 1e-10;
                    double priorGrad = diff / var_g;
                    double combined = NumOps.ToDouble(taskGrad[d]) + _algoOptions.KLCoefficient * priorGrad;
                    theta[d] = NumOps.Subtract(theta[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            adaptedParamsList.Add(theta);
            MetaModel.SetParameters(theta);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Compute hierarchical PAC-Bayesian bound: per-group KL + hyper-prior KL
        double totalKL = ComputeHierarchicalKL(adaptedParamsList);
        double hyperPriorKL = ComputeHyperPriorKL();
        int numTasks = Math.Max(taskBatch.Tasks.Length, 1);
        double pacBayesLoss = _algoOptions.KLCoefficient * (totalKL + hyperPriorKL) / numTasks;

        // Update prior mean via meta-gradients
        MetaModel.SetParameters(_priorMean);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            _priorMean = ApplyGradients(_priorMean, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(_priorMean);
        }

        // Update per-group log-variances via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _groupLogVars, _algoOptions.OuterLearningRate * 0.1, ComputeMetaPACOHLoss);

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

            for (int d = 0; d < _paramDim; d++)
            {
                int g = _groupOf[d];
                double diff = NumOps.ToDouble(theta[d]) - NumOps.ToDouble(_priorMean[d]);
                double var_g = Math.Exp(NumOps.ToDouble(_groupLogVars[g])) + 1e-10;
                double priorGrad = diff / var_g;
                double combined = NumOps.ToDouble(taskGrad[d]) + _algoOptions.KLCoefficient * priorGrad;
                theta[d] = NumOps.Subtract(theta[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
            }
        }

        MetaModel.SetParameters(_priorMean);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, theta);
    }

    /// <summary>
    /// Computes per-group KL divergences between empirical posteriors and group priors.
    /// </summary>
    private double ComputeHierarchicalKL(List<Vector<T>> adaptedParams)
    {
        if (adaptedParams.Count == 0) return 0;

        // Compute empirical posterior mean and variance per group
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

        // Sum KL per dimension using group-specific prior variance
        double kl = 0;
        for (int d = 0; d < _paramDim; d++)
        {
            int g = _groupOf[d];
            double priorVar = Math.Exp(NumOps.ToDouble(_groupLogVars[g])) + 1e-10;
            double diff = posteriorMean[d] - NumOps.ToDouble(_priorMean[d]);
            kl += (diff * diff + posteriorVar[d]) / priorVar - 1.0 + Math.Log(priorVar / posteriorVar[d]);
        }
        return 0.5 * kl;
    }

    /// <summary>
    /// Computes KL divergence from the group log-variances to the hyper-prior.
    /// KL(N(log_σ²_g, ε) || N(hyperPriorLogVar, σ²_hyper)) summed over groups.
    /// </summary>
    private double ComputeHyperPriorKL()
    {
        double hyperVar = Math.Exp(_algoOptions.HyperPriorLogVar) + 1e-10;
        double kl = 0;
        for (int g = 0; g < _numGroups; g++)
        {
            double logVarG = NumOps.ToDouble(_groupLogVars[g]);
            double diff = logVarG - _algoOptions.HyperPriorLogVar;
            kl += diff * diff / (2.0 * hyperVar);
        }
        return kl;
    }

    private double ComputeMetaPACOHLoss(TaskBatch<T, TInput, TOutput> taskBatch)
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
                    int g = _groupOf[d];
                    double diff = NumOps.ToDouble(theta[d]) - NumOps.ToDouble(_priorMean[d]);
                    double var_g = Math.Exp(NumOps.ToDouble(_groupLogVars[g])) + 1e-10;
                    double combined = NumOps.ToDouble(grad[d]) + _algoOptions.KLCoefficient * diff / var_g;
                    theta[d] = NumOps.Subtract(theta[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            adapted.Add(theta);
            MetaModel.SetParameters(theta);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        int numTasks = Math.Max(taskBatch.Tasks.Length, 1);
        double klPenalty = _algoOptions.KLCoefficient * (ComputeHierarchicalKL(adapted) + ComputeHyperPriorKL()) / numTasks;
        MetaModel.SetParameters(_priorMean);
        return totalLoss / numTasks + klPenalty;
    }
}
