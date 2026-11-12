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
/// Implementation of Meta-FDMixup: Feature-Distribution Mixup for cross-domain
/// few-shot learning (Xu et al., CVPR 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-FDMixup performs gradient-level mixup between tasks in a meta-batch to improve
/// cross-domain robustness. For each task, with probability p, its inner-loop gradient
/// is mixed with a randomly selected other task's gradient using a Beta-distributed
/// coefficient λ. The outer loop also applies feature distribution alignment by
/// penalizing the variance of per-task gradient directions.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// For each meta-batch:
///   1. Compute per-task gradients g_τ on support sets
///   2. With probability p, mix gradients:
///      g'_τ = λ * g_τ + (1-λ) * g_{τ'}, where λ ~ Beta(α, α)
///   3. Inner loop: θ_τ ← θ_0 - η * g'_τ (for each step)
///   4. Outer loop: θ_0 ← θ_0 - η_outer * [mean(∇L_query) + w_align * Var(g_τ)]
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation",
    "https://arxiv.org/abs/2001.08735",
    Year = 2021,
    Authors = "Jiechao Guan, Zhiwu Lu, Tao Xiang, Ji-Rong Wen")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaFDMixupAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MetaFDMixupOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaFDMixup;

    public MetaFDMixupAlgorithm(MetaFDMixupOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        var tasks = taskBatch.Tasks;

        // Step 1: Compute initial task gradients for mixup pool
        var taskGradPool = new List<Vector<T>>();
        foreach (var task in tasks)
        {
            ParamModel.SetParameters(initParams);
            taskGradPool.Add(ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput)));
        }

        // Step 2: Inner loop with gradient mixup
        for (int tIdx = 0; tIdx < tasks.Length; tIdx++)
        {
            var task = tasks[tIdx];
            var adaptedParams = initParams.Clone();

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Apply gradient mixup with probability p
                if (RandomGenerator.NextDouble() < _algoOptions.MixupProbability && tasks.Length > 1)
                {
                    // Sample mixup partner (different task)
                    int partner = tIdx;
                    while (partner == tIdx)
                        partner = RandomGenerator.Next(tasks.Length);

                    // Sample λ ~ Beta(α, α) using Gamma trick
                    double lambda = SampleBeta(_algoOptions.MixupAlpha, _algoOptions.MixupAlpha);

                    // Mix gradients: g' = λ*g_task + (1-λ)*g_partner
                    var partnerGrad = taskGradPool[partner];
                    taskGrad = Engine.Add(
                        Engine.Multiply(taskGrad, NumOps.FromDouble(lambda)),
                        Engine.Multiply(partnerGrad, NumOps.FromDouble(1.0 - lambda)));
                }

                adaptedParams = ApplyInnerGradients(adaptedParams, taskGrad, _algoOptions.InnerLearningRate, step == 0);
            }

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Compute feature distribution alignment penalty (gradient direction variance)
        T alignmentPenalty = ComputeGradientAlignmentPenalty(taskGradPool);

        // Outer loop
        ParamModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            // Add alignment gradient: penalize high variance in gradient directions
            // Weighted consistently with the alignment penalty added to the loss
            var alignGrad = ComputeAlignmentGradient(taskGradPool);
            if (alignGrad is not null)
                avgGrad = Engine.Add(avgGrad, Engine.Multiply(alignGrad, NumOps.FromDouble(_algoOptions.AlignmentWeight)));
            ParamModel.SetParameters(ApplyOuterGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        var totalLoss = NumOps.Add(ComputeMean(losses), alignmentPenalty);
        return totalLoss;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();
        var adaptedParams = currentParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(currentParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes variance of gradient directions as alignment penalty.
    /// </summary>
    private T ComputeGradientAlignmentPenalty(List<Vector<T>> gradients)
    {
        if (gradients.Count <= 1) return NumOps.Zero;

        // Normalize gradients to unit vectors (g / (||g|| + 1e-10)) and compute variance of directions
        var rows = new Tensor<T>[gradients.Count];
        for (int t = 0; t < rows.Length; t++)
        {
            var g = gradients[t].Length == _paramDim ? gradients[t] : gradients[t].GetSubVector(0, _paramDim);
            T norm = NumOps.Add(NumOps.Sqrt(Engine.DotProduct(g, g)), NumOps.FromDouble(1e-10));
            rows[t] = Tensor<T>.FromVector(Engine.Divide(g, norm));
        }

        // Sum over dimensions of the per-dimension population variance across tasks
        var centered = CenterRows(Engine.TensorStack(rows, 0));
        T totalVar = Engine.TensorSum(Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, keepDims: false));

        return NumOps.Multiply(NumOps.FromDouble(_algoOptions.AlignmentWeight), totalVar);
    }

    /// <summary>
    /// Per dimension d: sum over tasks of 2 (g_t[d] - mean_d) / T, or null for fewer than two tasks.
    /// </summary>
    private Vector<T>? ComputeAlignmentGradient(List<Vector<T>> gradients)
    {
        if (gradients.Count <= 1) return null;
        var rows = new Tensor<T>[gradients.Count];
        for (int t = 0; t < rows.Length; t++)
            rows[t] = Tensor<T>.FromVector(gradients[t].Length == _paramDim ? gradients[t] : gradients[t].GetSubVector(0, _paramDim));
        var centered = CenterRows(Engine.TensorStack(rows, 0));
        var summed = Engine.ReduceSum(centered, new[] { 0 }, keepDims: false).ToVector();
        return Engine.Multiply(summed, NumOps.FromDouble(2.0 / gradients.Count));
    }

    /// <summary>Subtracts each column's mean from a [tasks, dims] stack.</summary>
    private Tensor<T> CenterRows(Tensor<T> stacked)
        => Engine.TensorSubtract(stacked, Engine.TensorBroadcastTo(
            Engine.ReduceMean(stacked, new[] { 0 }, keepDims: true), stacked.Shape.ToArray()));
}
