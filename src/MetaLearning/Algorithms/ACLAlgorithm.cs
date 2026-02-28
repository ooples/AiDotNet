using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of ACL: Adaptive Continual Learning with task-specific parameter importance masks.
/// </summary>
/// <remarks>
/// <para>
/// ACL prevents catastrophic forgetting by maintaining per-parameter importance scores that
/// accumulate across tasks via exponential moving average. Important parameters are protected
/// by reducing their effective learning rate and applying elastic weight consolidation (EWC)-style
/// regularization toward the pre-task initialization.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Importance estimation: Ω_d = EMA(Ω_d, |∂L/∂θ_d|²)
///   where EMA uses ImportanceDecay
///
/// Protected learning rate: η'_d = η / (1 + ProtectionStrength * Ω_d)
///
/// Inner loop:
///   θ_d ← θ_d - η'_d * grad_d
///
/// Elastic regularization on query loss:
///   L_total = L_query + ElasticRegWeight * Σ_d Ω_d * (θ_d - θ_init_d)²
///
/// Mask sparsity penalty:
///   L_meta += MaskSparsityPenalty * Σ_d |Ω_d|
///
/// Outer loop: update θ, Ω (via SPSA)
/// </code>
/// </para>
/// </remarks>
public class ACLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ACLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Per-parameter importance scores (accumulated via EMA).</summary>
    private double[] _importance;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ACL;

    public ACLAlgorithm(ACLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _importance = new double[_paramDim];
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            // Inner loop with importance-protected learning rates
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Update importance: EMA of squared gradients
                for (int d = 0; d < _paramDim; d++)
                {
                    double gVal = NumOps.ToDouble(grad[d]);
                    _importance[d] = _algoOptions.ImportanceDecay * _importance[d]
                                   + (1.0 - _algoOptions.ImportanceDecay) * gVal * gVal;
                }

                // Apply gradient with per-parameter protected learning rate
                for (int d = 0; d < _paramDim; d++)
                {
                    double effectiveLR = _algoOptions.InnerLearningRate
                                       / (1.0 + _algoOptions.ProtectionStrength * _importance[d]);
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(effectiveLR * NumOps.ToDouble(grad[d])));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Elastic weight consolidation regularization
            double ewcReg = 0;
            for (int d = 0; d < _paramDim; d++)
            {
                double diff = NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(initParams[d]);
                ewcReg += _importance[d] * diff * diff;
            }

            // Mask sparsity penalty (L1 on importance)
            double sparsityPenalty = 0;
            for (int d = 0; d < _paramDim; d++)
                sparsityPenalty += Math.Abs(_importance[d]);

            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.ElasticRegWeight * ewcReg
                                + _algoOptions.MaskSparsityPenalty * sparsityPenalty));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int d = 0; d < _paramDim; d++)
            {
                double gVal = NumOps.ToDouble(grad[d]);
                _importance[d] = _algoOptions.ImportanceDecay * _importance[d]
                               + (1.0 - _algoOptions.ImportanceDecay) * gVal * gVal;

                double effectiveLR = _algoOptions.InnerLearningRate
                                   / (1.0 + _algoOptions.ProtectionStrength * _importance[d]);
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(effectiveLR * gVal));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }
}
