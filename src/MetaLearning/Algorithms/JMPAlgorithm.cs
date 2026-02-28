using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of JMP: Joint Multi-Phase meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// JMP divides the inner loop into two phases with distinct optimization characteristics.
/// Phase 1 (coarse) uses a higher learning rate for rapid, broad adaptation. Phase 2 (fine)
/// uses a lower learning rate with L2 regularization toward the Phase 1 result, enabling
/// careful refinement without deviating too far from the coarse solution.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Phase 1 (coarse): K₁ = ⌊K * Phase1Fraction⌋ steps
///   η₁ = InnerLearningRate * Phase1LRMultiplier
///   θ₁ = θ₀ - η₁ * Σ grad
///
/// Phase 2 (fine): K₂ = K - K₁ steps
///   η₂ = InnerLearningRate * Phase2LRMultiplier
///   θ₂ = θ₁ - η₂ * (grad + PhaseRegWeight * (θ - θ₁))
///
/// L_meta = L_query(θ₂)
/// </code>
/// </para>
/// </remarks>
public class JMPAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly JMPOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.JMP;

    public JMPAlgorithm(JMPOptions<T, TInput, TOutput> options)
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

        int K = _algoOptions.AdaptationSteps;
        int K1 = (int)(K * _algoOptions.Phase1Fraction);
        if (K1 < 1) K1 = 1;
        if (K1 >= K) K1 = K - 1;

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            // Phase 1: Coarse adaptation (high LR)
            double phase1LR = _algoOptions.InnerLearningRate * _algoOptions.Phase1LRMultiplier;
            for (int step = 0; step < K1; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, phase1LR);
            }

            // Save Phase 1 result as anchor for Phase 2 regularization
            var phase1Params = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) phase1Params[d] = adaptedParams[d];

            // Phase 2: Fine adaptation (low LR + regularization toward Phase 1)
            double phase2LR = _algoOptions.InnerLearningRate * _algoOptions.Phase2LRMultiplier;
            for (int step = K1; step < K; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                for (int d = 0; d < _paramDim; d++)
                {
                    double gradVal = NumOps.ToDouble(grad[d]);
                    double regGrad = _algoOptions.PhaseRegWeight
                                   * (NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(phase1Params[d]));
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(phase2LR * (gradVal + regGrad)));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
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

        int K = _algoOptions.AdaptationSteps;
        int K1 = (int)(K * _algoOptions.Phase1Fraction);
        if (K1 < 1) K1 = 1;
        if (K1 >= K) K1 = K - 1;

        double phase1LR = _algoOptions.InnerLearningRate * _algoOptions.Phase1LRMultiplier;
        for (int step = 0; step < K1; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyGradients(adaptedParams, grad, phase1LR);
        }

        var phase1Params = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) phase1Params[d] = adaptedParams[d];

        double phase2LR = _algoOptions.InnerLearningRate * _algoOptions.Phase2LRMultiplier;
        for (int step = K1; step < K; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            for (int d = 0; d < _paramDim; d++)
            {
                double regGrad = _algoOptions.PhaseRegWeight
                               * (NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(phase1Params[d]));
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(phase2LR * (NumOps.ToDouble(grad[d]) + regGrad)));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }
}
