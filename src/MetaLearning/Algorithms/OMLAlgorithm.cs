using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of OML: Online Meta-Learning (Javed &amp; White, 2019).
/// </summary>
/// <remarks>
/// <para>
/// OML partitions model parameters into a Representation Learning Network (RLN) and a
/// Prediction Learning Network (PLN). The RLN (first 1-f fraction of parameters) is only
/// updated in the outer loop, while the PLN (last f fraction) is adapted in the inner loop.
/// This division encourages the RLN to learn sparse, non-interfering representations that
/// support continual learning — the PLN can be quickly adapted without forgetting.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Parameter partition:
///   PLN indices: [P*(1-f), P)  (last f fraction)
///   RLN indices: [0, P*(1-f))  (first 1-f fraction)
///
/// Inner loop (only PLN params):
///   θ_PLN ← θ_PLN - η * ∇_PLN L_support
///   θ_RLN stays frozen
///
/// Outer loop (both RLN + PLN):
///   L_meta = L_query + SparsityPenalty * ||θ_RLN||₁
///                     + RepresentationRegWeight * ||θ_RLN - θ_RLN_init||²
///   Update all θ
/// </code>
/// </para>
/// </remarks>
public class OMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly OMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _plnStart;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.OML;

    public OMLAlgorithm(OMLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _plnStart = (int)(_paramDim * (1.0 - options.PlnFraction));
        if (_plnStart < 0) _plnStart = 0;
        if (_plnStart >= _paramDim) _plnStart = _paramDim - 1;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: only adapt PLN parameters
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Only update PLN parameters (RLN stays frozen in inner loop)
                for (int d = _plnStart; d < _paramDim; d++)
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(grad[d])));
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // L1 sparsity on RLN parameters
            double sparsity = 0;
            for (int d = 0; d < _plnStart; d++)
                sparsity += Math.Abs(NumOps.ToDouble(adaptedParams[d]));

            // L2 regularization on RLN change
            double rlnReg = 0;
            for (int d = 0; d < _plnStart; d++)
            {
                double diff = NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(initParams[d]);
                rlnReg += diff * diff;
            }

            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.SparsityPenalty * sparsity
                                + _algoOptions.RepresentationRegWeight * rlnReg));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update ALL parameters (both RLN and PLN)
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

        // Only adapt PLN parameters
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int d = _plnStart; d < _paramDim; d++)
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(grad[d])));
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }
}
