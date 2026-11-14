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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("How to Train Your MAML to Excel in Few-Shot Classification",
    "https://arxiv.org/abs/2106.16245",
    Year = 2022,
    Authors = "Han-Jia Ye, Lu Ming, De-Chuan Zhan, Wei-Lun Chao")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MPTSAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MPTSOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numGroups;
    private readonly int _groupSize;

    /// <summary>Learned priority scores for each group (higher = adapted earlier).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _priorityScores;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MPTS;

    public MPTSAlgorithm(MPTSOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));
        if (options.PriorityDecayRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "PriorityDecayRate must be positive.");
        _numGroups = Math.Max(1, options.NumParamGroups);
        _groupSize = (_paramDim + _numGroups - 1) / _numGroups;

        // Initialize priorities: last group gets highest priority (classifier head pattern)
        _priorityScores = Engine.TensorDivideScalar(
            Engine.TensorArange<T>(1, _numGroups + 1, 1), NumOps.FromDouble(_numGroups)).ToVector();
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Leaves the adapted parameters on the model for the query gradient below
            losses.Add(TaskObjective(task, initParams));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _priorityScores, _algoOptions.OuterLearningRate * 0.1, ComputeMPTSLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptWithPriorities(task, initParams);

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The prioritized inner loop: at step k of K every parameter of group g moves by
    /// lr * sigmoid((progress - (1 - priority_g)) * 10 / decay) * g_d, with progress = k / max(K - 1, 1).
    /// </summary>
    private Vector<T> AdaptWithPriorities(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var adaptedParams = initParams.Clone();
        var priorities = PerParameterPriorities();
        T slope = NumOps.FromDouble(10.0 / _algoOptions.PriorityDecayRate);
        T learningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);

        int K = _algoOptions.AdaptationSteps;
        for (int step = 0; step < K; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var activation = StepActivation(priorities, slope, step, K);
            adaptedParams = Engine.Subtract(adaptedParams,
                Engine.Multiply(Engine.Multiply(activation, grad.GetSubVector(0, _paramDim)), learningRate));
        }

        return adaptedParams;
    }

    /// <summary>
    /// sigmoid((progress - (1 - p_d)) * slope) per parameter, progress = step / max(K - 1, 1): higher priority
    /// activates earlier. progress - (1 - p) = p + (progress - 1).
    /// </summary>
    private Vector<T> StepActivation(Vector<T> priorities, T slope, int step, int K)
    {
        T progress = NumOps.FromDouble((double)step / Math.Max(K - 1, 1));
        var shift = Engine.Add(priorities, Engine.Fill(_paramDim, NumOps.Subtract(progress, NumOps.One)));
        return Engine.Sigmoid(Engine.Multiply(shift, slope));
    }

    /// <summary>Each parameter's group priority: group g covers [g * groupSize, (g + 1) * groupSize).</summary>
    private Vector<T> PerParameterPriorities()
    {
        var repeated = Engine.TensorBroadcastTo(
            Tensor<T>.FromVector(_priorityScores, new[] { _numGroups, 1 }), new[] { _numGroups, _groupSize });
        return Engine.Reshape(repeated, new[] { _numGroups * _groupSize }).ToVector().GetSubVector(0, _paramDim);
    }

    /// <summary>
    /// Sum over groups with more than one parameter of the population variance of theta' - theta in the group.
    /// </summary>
    private T GroupCoherence(Vector<T> adaptedParams, Vector<T> initParams)
    {
        int padded = _numGroups * _groupSize;
        var diff = Engine.Subtract(adaptedParams.GetSubVector(0, _paramDim), initParams.GetSubVector(0, _paramDim));
        var present = Engine.Fill(_paramDim, NumOps.One);
        if (padded > _paramDim)
        {
            diff = Vector<T>.Concatenate(diff, new Vector<T>(padded - _paramDim));
            present = Vector<T>.Concatenate(present, new Vector<T>(padded - _paramDim));
        }

        var shape = new[] { _numGroups, _groupSize };
        var rows = Tensor<T>.FromVector(diff, shape);
        var count = Engine.ReduceSum(Tensor<T>.FromVector(present, shape), new[] { 1 }, keepDims: false);
        var safeCount = Engine.TensorMax(count, NumOps.One);
        var mean = Engine.TensorDivide(Engine.ReduceSum(rows, new[] { 1 }, keepDims: false), safeCount);
        var meanSquare = Engine.TensorDivide(Engine.ReduceSum(Engine.TensorSquare(rows), new[] { 1 }, keepDims: false), safeCount);
        var variance = Engine.TensorSubtract(meanSquare, Engine.TensorSquare(mean));
        return Engine.TensorSum(Engine.TensorMultiply(variance, Engine.TensorGreaterThan(count, NumOps.One)));
    }

    /// <summary>
    /// One task's trained objective: the query loss after prioritized adaptation plus GroupRegWeight times the
    /// group coherence of the adaptation. The adapted parameters are left on the model.
    /// </summary>
    private T TaskObjective(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var adaptedParams = AdaptWithPriorities(task, initParams);

        ParamModel.SetParameters(adaptedParams);
        var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

        // Group coherence regularization
        T groupReg = GroupCoherence(adaptedParams, initParams);
        return NumOps.Add(queryLoss, NumOps.Multiply(NumOps.FromDouble(_algoOptions.GroupRegWeight), groupReg));
    }

    /// <summary>
    /// The batch mean of the trained objective (query loss plus the group regularizer) at the current
    /// parameters and priorities, restoring the model's parameters. This is what MetaTrain reports and what the
    /// SPSA priority update must estimate; it used to drop the regularizer, so SPSA tuned the priorities against
    /// a different objective from the one being trained.
    /// </summary>
    internal T ComputeMPTSObjective(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
            losses.Add(TaskObjective(task, initParams));
        ParamModel.SetParameters(initParams);
        return ComputeMean(losses);
    }

    // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
    private double ComputeMPTSLoss(TaskBatch<T, TInput, TOutput> taskBatch)
        => NumOps.ToDouble(ComputeMPTSObjective(taskBatch));
}
