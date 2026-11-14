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
/// Implementation of DREAM: Directed REward Augmented Meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// DREAM meta-learns a reward shaping function that transforms the raw task loss into a more
/// informative signal for the inner loop. The reward shaper is a small MLP that takes
/// (loss, gradient_norm, step/K) as input and outputs a scalar multiplier for the gradient.
/// This enables curriculum-like adaptation where early steps may be scaled differently
/// than later steps.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Reward shaper: R(loss, ||grad||, step/K) → scaling factor
///   R = MLP with hidden_dim, tanh activation, sigmoid output
///
/// Inner loop:
///   For step k:
///     grad = ∂L/∂θ
///     scale = R(L, ||grad||, k/K)
///     θ ← θ - η * scale * grad + discount^k * shaped_bonus
///
/// Outer loop:
///   L_meta = L_query + w_shape * Σ_k discount^k * |1 - scale_k|
///   Update θ_0 and R jointly
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning via Learned Loss",
    "https://arxiv.org/abs/1906.05374",
    Year = 2021,
    Authors = "Sarah Bechtle, Artem Molchanov, Yevgen Chebotar, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class DREAMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly DREAMOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Reward shaper parameters: 3-input → hidden → 1-output MLP.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _shaperParams;

    private readonly int _hiddenDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DREAM;

    public DREAMAlgorithm(DREAMOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        if (options.RewardShaperHiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "RewardShaperHiddenDim must be positive.");
        if (options.ShapingDiscount < 0 || options.ShapingDiscount > 1)
            throw new ArgumentOutOfRangeException(nameof(options), "ShapingDiscount must be in [0, 1].");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _hiddenDim = options.RewardShaperHiddenDim;

        // Shaper MLP: input(3) → hidden → output(1)
        int shaperSize = 3 * _hiddenDim + _hiddenDim; // weights + output weights
        _shaperParams = new Vector<T>(shaperSize);
        for (int i = 0; i < shaperSize; i++)
            _shaperParams[i] = NumOps.FromDouble(0.1 * SampleNormal());
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0)
            return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = AdaptWithShaping(task, initParams, clip: true, out T shapingPenalty);

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            var totalLoss = NumOps.Add(queryLoss,
                NumOps.Multiply(NumOps.FromDouble(_algoOptions.RewardShapingWeight), shapingPenalty));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _shaperParams, _algoOptions.OuterLearningRate * 0.1, ComputeDREAMLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptWithShaping(task, initParams, clip: true, out _);

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The shaped inner loop: each step scales the support gradient by the shaper's factor for
    /// (support loss, gradient norm, progress) and accumulates the discounted shaping penalty
    /// sum_k discount^k |1 - scale_k|.
    /// </summary>
    private Vector<T> AdaptWithShaping(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams, bool clip, out T shapingPenalty)
    {
        var adaptedParams = initParams.Clone();
        shapingPenalty = NumOps.Zero;
        int K = _algoOptions.AdaptationSteps;

        for (int step = 0; step < K; step++)
        {
            adaptedParams = ShapedStep(task, adaptedParams, clip, step, K, ref shapingPenalty);
        }

        return adaptedParams;
    }

    /// <summary>
    /// One shaped step: theta - lr * scale * g for the shaper's scale at (support loss, |g|, step / max(K - 1, 1)),
    /// adding discount^step * |1 - scale| to the shaping penalty.
    /// </summary>
    private Vector<T> ShapedStep(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> adaptedParams, bool clip, int step, int K, ref T shapingPenalty)
    {
        ParamModel.SetParameters(adaptedParams);
        var raw = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
        var grad = (clip ? ClipGradients(raw) : raw).GetSubVector(0, _paramDim);
        T loss = ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput);

        // Compute gradient norm
        T gradNorm = NumOps.Sqrt(Engine.DotProduct(grad, grad));

        // Reward shaper: scale factor
        T progress = NumOps.FromDouble((double)step / Math.Max(K - 1, 1));
        T scale = ComputeRewardScale(loss, gradNorm, progress);

        // Shaping penalty: deviation from unit scale
        T discountFactor = NumOps.FromDouble(Math.Pow(_algoOptions.ShapingDiscount, step));
        shapingPenalty = NumOps.Add(shapingPenalty,
            NumOps.Multiply(discountFactor, NumOps.Abs(NumOps.Subtract(NumOps.One, scale))));

        // Apply scaled gradient
        T learningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);
        return Engine.Subtract(adaptedParams, Engine.Multiply(grad, NumOps.Multiply(learningRate, scale)));
    }

    /// <summary>
    /// Computes the reward scaling factor using the shaper MLP:
    /// input = (loss, grad_norm, progress) -> hidden(tanh) -> 2 * sigmoid(output), in [0, 2] so it can
    /// both dampen and amplify.
    /// </summary>
    private T ComputeRewardScale(T loss, T gradNorm, T progress)
    {
        var input = Tensor<T>.FromVector(new Vector<T>(new[] { loss, gradNorm, progress }), new[] { 3, 1 });

        // Layer 1: input(3) -> hidden (tanh)
        var layer1 = Tensor<T>.FromVector(_shaperParams.GetSubVector(0, 3 * _hiddenDim), new[] { _hiddenDim, 3 });
        var hidden = Engine.Tanh(Engine.TensorMatMul(layer1, input)).ToVector();

        // Layer 2: hidden -> scalar
        T output = Engine.DotProduct(hidden, _shaperParams.GetSubVector(3 * _hiddenDim, _hiddenDim));
        return NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(output)))));
    }

    private double ComputeDREAMLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adapted = AdaptWithShaping(task, initParams, clip: false, out _);

            ParamModel.SetParameters(adapted);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
