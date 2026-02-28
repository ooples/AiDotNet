using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of ActiveTransFSL: Active Transductive Few-Shot Learning.
/// </summary>
/// <remarks>
/// <para>
/// ActiveTransFSL performs standard inductive adaptation on support data, then applies
/// transductive refinement using query gradients. The refinement is focused on the most
/// uncertain parameter dimensions (measured by gradient magnitude), implementing an
/// active learning strategy in parameter space for transductive inference.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Phase 1 (Inductive): Standard MAML adaptation on support
///   θ_inductive = θ - η * Σ ∇L_support
///
/// Phase 2 (Transductive): Selective refinement using query gradients
///   For each refinement step:
///     grad_query = ∇L_query(θ_current)
///     uncertainty_d = |grad_query_d|
///     Select top-f fraction by uncertainty
///     θ_d ← θ_d - TransductiveLR * grad_query_d  (only for selected d)
///
/// L_meta = (1-TransductiveWeight) * L_inductive + TransductiveWeight * L_transductive
/// </code>
/// </para>
/// </remarks>
public class ActiveTransFSLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ActiveTransFSLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ActiveTransFSL;

    public ActiveTransFSLAlgorithm(ActiveTransFSLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Phase 1: Inductive adaptation on support
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            // Inductive loss
            MetaModel.SetParameters(adaptedParams);
            var inductiveLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Phase 2: Active transductive refinement
            var transParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) transParams[d] = adaptedParams[d];

            for (int rStep = 0; rStep < _algoOptions.TransductiveRefinementSteps; rStep++)
            {
                MetaModel.SetParameters(transParams);
                var queryGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));

                // Compute uncertainty per parameter
                var uncertainty = new double[_paramDim];
                for (int d = 0; d < _paramDim; d++)
                    uncertainty[d] = Math.Abs(NumOps.ToDouble(queryGrad[d]));

                // Find threshold for top-f fraction
                var sorted = new double[_paramDim];
                Array.Copy(uncertainty, sorted, _paramDim);
                Array.Sort(sorted);
                int threshIdx = (int)((1.0 - _algoOptions.SelectionFraction) * _paramDim);
                if (threshIdx >= _paramDim) threshIdx = _paramDim - 1;
                double threshold = sorted[threshIdx];

                // Selectively update only uncertain parameters
                for (int d = 0; d < _paramDim; d++)
                {
                    if (uncertainty[d] >= threshold)
                    {
                        transParams[d] = NumOps.Subtract(transParams[d],
                            NumOps.FromDouble(_algoOptions.TransductiveLR * NumOps.ToDouble(queryGrad[d])));
                    }
                }
            }

            MetaModel.SetParameters(transParams);
            var transductiveLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Combined loss
            double w = _algoOptions.TransductiveWeight;
            var totalLoss = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(1.0 - w), inductiveLoss),
                NumOps.Multiply(NumOps.FromDouble(w), transductiveLoss));

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
            adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }
}
