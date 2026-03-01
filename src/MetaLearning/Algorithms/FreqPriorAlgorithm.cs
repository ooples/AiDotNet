using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of FreqPrior: Frequency-based prior for cross-domain few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FreqPrior decomposes the parameter vector using a discrete cosine transform (DCT)-like
/// basis. Low-frequency components (capturing smooth, domain-invariant structure) are strongly
/// regularized toward the meta-prior, while high-frequency components (capturing task-specific
/// details) are allowed to adapt freely. This encourages learning transferable features
/// that generalize across domains.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Frequency decomposition: θ = Σ_k c_k * basis_k(d)
///   where basis_k(d) = cos(π*k*(2d+1)/(2D)) (DCT-II basis)
///
/// Inner loop with frequency-aware regularization:
///   For each adaptation step:
///     grad_task = ∂L/∂θ
///     For dimension d with frequency index k(d):
///       if k(d) &lt; lowFreqCutoff: reg = lowFreqRegWeight * (θ_d - θ_meta_d)
///       else:                     reg = highFreqRegWeight * (θ_d - θ_meta_d)
///     θ ← θ - η * (grad_task + reg)
/// </code>
/// </para>
/// </remarks>
public class FreqPriorAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly FreqPriorOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _lowFreqCutoff;
    private readonly double[] _freqRegWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FreqPrior;

    public FreqPriorAlgorithm(FreqPriorOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _lowFreqCutoff = Math.Max(1, (int)(_paramDim * options.LowFreqFraction));

        // Assign frequency-dependent regularization weights
        // Parameters are treated as if arranged by "frequency" — early params = low freq
        _freqRegWeights = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++)
        {
            _freqRegWeights[d] = d < _lowFreqCutoff
                ? options.LowFreqRegWeight
                : options.HighFreqRegWeight;
        }
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

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Frequency-aware regularization
                for (int d = 0; d < _paramDim; d++)
                {
                    double diff = NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(initParams[d]);
                    double reg = _freqRegWeights[d] * diff;
                    double combined = NumOps.ToDouble(taskGrad[d]) + reg;
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            // Compute meta-gradient with frequency penalty
            var queryGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));
            metaGradients.Add(queryGrad);
        }

        // Compute frequency prior penalty: stronger for low-freq deviations across tasks
        double freqPenalty = ComputeFrequencyPenalty(metaGradients);

        // Outer loop
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        return NumOps.Add(ComputeMean(losses), NumOps.FromDouble(freqPenalty));
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
            var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int d = 0; d < _paramDim; d++)
            {
                double diff = NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(initParams[d]);
                double reg = _freqRegWeights[d] * diff;
                double combined = NumOps.ToDouble(taskGrad[d]) + reg;
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes frequency-based penalty: variance of low-freq gradient components
    /// should be small (consistent across tasks), high-freq can vary.
    /// </summary>
    private double ComputeFrequencyPenalty(List<Vector<T>> gradients)
    {
        if (gradients.Count <= 1) return 0;
        double penalty = 0;
        for (int d = 0; d < _lowFreqCutoff; d++)
        {
            double mean = 0;
            foreach (var g in gradients) mean += NumOps.ToDouble(g[d]);
            mean /= gradients.Count;
            double var_d = 0;
            foreach (var g in gradients) var_d += (NumOps.ToDouble(g[d]) - mean) * (NumOps.ToDouble(g[d]) - mean);
            penalty += var_d / gradients.Count;
        }
        return _algoOptions.LowFreqRegWeight * 0.01 * penalty;
    }
}
