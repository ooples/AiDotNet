using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of MPTS: Meta-learning with Progressive Task-Specific adaptation.
/// </summary>
/// <remarks>
/// <para>
/// MPTS divides model parameters into groups and progressively unfreezes them during the
/// inner loop. High-priority groups (typically the classifier head) are adapted from the
/// first step, while lower-priority groups (backbone) are gradually unfrozen as adaptation
/// progresses. Learned priority scores determine the unfreezing schedule.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Parameter groups: G_1..G_N (equal-sized blocks)
/// Learned priorities: p_1..p_N (via SPSA), normalized to [0,1]
///
/// Group activation at step k:
///   active_g = sigmoid((k/K - (1 - p_g)) * 10 / PriorityDecayRate)
///
/// Inner loop:
///   θ_d ← θ_d - η * active[group(d)] * grad_d
///
/// Group coherence regularization:
///   L_reg = GroupRegWeight * Σ_g var(θ_g - θ_g_init)
///
/// Outer loop: update θ, priorities (via SPSA)
/// </code>
/// </para>
/// </remarks>
public class MPTSAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MPTSOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numGroups;
    private readonly int _groupSize;

    /// <summary>Learned priority scores for each group (higher = adapted earlier).</summary>
    private Vector<T> _priorityScores;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MPTS;

    public MPTSAlgorithm(MPTSOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _numGroups = Math.Max(1, options.NumParamGroups);
        _groupSize = (_paramDim + _numGroups - 1) / _numGroups;

        // Initialize priorities: last group gets highest priority (classifier head pattern)
        _priorityScores = new Vector<T>(_numGroups);
        for (int g = 0; g < _numGroups; g++)
            _priorityScores[g] = NumOps.FromDouble((double)(g + 1) / _numGroups);
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

            int K = _algoOptions.AdaptationSteps;
            for (int step = 0; step < K; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                double progress = (double)step / Math.Max(K - 1, 1);

                for (int d = 0; d < _paramDim; d++)
                {
                    int g = Math.Min(d / _groupSize, _numGroups - 1);
                    double priority = NumOps.ToDouble(_priorityScores[g]);

                    // Sigmoid activation: higher priority → active earlier
                    double activation = 1.0 / (1.0 + Math.Exp(-(progress - (1.0 - priority)) * 10.0 / _algoOptions.PriorityDecayRate));

                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * activation * NumOps.ToDouble(grad[d])));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Group coherence regularization
            double groupReg = 0;
            for (int g = 0; g < _numGroups; g++)
            {
                double sumDiff = 0, sumDiffSq = 0;
                int count = 0;
                int start = g * _groupSize;
                for (int d = start; d < start + _groupSize && d < _paramDim; d++)
                {
                    double diff = NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(initParams[d]);
                    sumDiff += diff;
                    sumDiffSq += diff * diff;
                    count++;
                }
                if (count > 1)
                {
                    double mean = sumDiff / count;
                    groupReg += (sumDiffSq / count - mean * mean);
                }
            }

            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.GroupRegWeight * groupReg));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _priorityScores, _algoOptions.OuterLearningRate * 0.1, ComputeMPTSLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        int K = _algoOptions.AdaptationSteps;
        for (int step = 0; step < K; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            double progress = (double)step / Math.Max(K - 1, 1);

            for (int d = 0; d < _paramDim; d++)
            {
                int g = Math.Min(d / _groupSize, _numGroups - 1);
                double priority = NumOps.ToDouble(_priorityScores[g]);
                double activation = 1.0 / (1.0 + Math.Exp(-(progress - (1.0 - priority)) * 10.0 / _algoOptions.PriorityDecayRate));
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * activation * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double ComputeMPTSLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        int K = _algoOptions.AdaptationSteps;
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            for (int step = 0; step < K; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                double progress = (double)step / Math.Max(K - 1, 1);
                for (int d = 0; d < _paramDim; d++)
                {
                    int grp = Math.Min(d / _groupSize, _numGroups - 1);
                    double pr = NumOps.ToDouble(_priorityScores[grp]);
                    double act = 1.0 / (1.0 + Math.Exp(-(progress - (1.0 - pr)) * 10.0 / _algoOptions.PriorityDecayRate));
                    ap[d] = NumOps.Subtract(ap[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * act * NumOps.ToDouble(g[d])));
                }
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
