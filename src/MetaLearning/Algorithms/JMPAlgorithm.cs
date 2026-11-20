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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
    "https://arxiv.org/abs/1703.03400",
    Year = 2017,
    Authors = "Chelsea Finn, Pieter Abbeel, Sergey Levine")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class JMPAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly JMPOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.JMP;

    public JMPAlgorithm(JMPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        RequirePlainSgdInnerSteps(true, "a two-phase inner learning-rate schedule");
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        int K = Math.Max(2, _algoOptions.AdaptationSteps); // JMP requires at least 2 steps for both phases
        int K1 = Math.Min(Math.Max((int)(K * _algoOptions.Phase1Fraction), 1), K - 1);

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = initParams.Clone();

            // Phase 1: Coarse adaptation (high LR)
            double phase1LR = _algoOptions.InnerLearningRate * _algoOptions.Phase1LRMultiplier;
            for (int step = 0; step < K1; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyInnerGradients(adaptedParams, grad, phase1LR, step == 0);
            }

            // Save Phase 1 result as anchor for Phase 2 regularization
            var phase1Params = adaptedParams.Clone();

            // Phase 2: Fine adaptation (low LR + regularization toward Phase 1)
            double phase2LR = _algoOptions.InnerLearningRate * _algoOptions.Phase2LRMultiplier;
            for (int step = K1; step < K; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = Phase2Step(adaptedParams, grad, phase1Params, phase2LR);
            }

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams.Clone();

        int K = Math.Max(2, _algoOptions.AdaptationSteps); // JMP requires at least 2 steps for both phases
        int K1 = Math.Min(Math.Max((int)(K * _algoOptions.Phase1Fraction), 1), K - 1);

        double phase1LR = _algoOptions.InnerLearningRate * _algoOptions.Phase1LRMultiplier;
        for (int step = 0; step < K1; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, phase1LR, step == 0);
        }

        var phase1Params = adaptedParams.Clone();

        double phase2LR = _algoOptions.InnerLearningRate * _algoOptions.Phase2LRMultiplier;
        for (int step = K1; step < K; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = Phase2Step(adaptedParams, grad, phase1Params, phase2LR);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Phase-2 step: theta - lr * (g + lambda * (theta - theta_phase1)), pulling toward the phase-1 anchor.
    /// </summary>
    private Vector<T> Phase2Step(Vector<T> parameters, Vector<T> gradient, Vector<T> anchor, double learningRate)
    {
        var regularized = Engine.Add(gradient,
            Engine.Multiply(Engine.Subtract(parameters, anchor), NumOps.FromDouble(_algoOptions.PhaseRegWeight)));
        return Engine.Subtract(parameters, Engine.Multiply(regularized, NumOps.FromDouble(learningRate)));
    }
}
