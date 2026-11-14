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
        var rawGradPool = new List<Vector<T>>();
        foreach (var task in tasks)
        {
            ParamModel.SetParameters(initParams);
            var raw = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            rawGradPool.Add(raw);
            taskGradPool.Add(ClipGradients(raw));
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
            var alignGrad = ComputeAlignmentGradient(rawGradPool, tasks, initParams);
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
    /// Computes variance of gradient directions as alignment penalty: w_align * L_align.
    /// </summary>
    private T ComputeGradientAlignmentPenalty(List<Vector<T>> gradients)
        => NumOps.Multiply(NumOps.FromDouble(_algoOptions.AlignmentWeight), AlignmentLoss(gradients));

    /// <summary>
    /// The feature-distribution alignment loss L_align = sum_d Var_t(u_t[d]): the population variance across
    /// tasks of the unit support-gradient directions u_t = g_t / (||g_t|| + 1e-10); 0 for fewer than two tasks.
    /// </summary>
    internal T AlignmentLoss(List<Vector<T>> gradients)
    {
        if (gradients.Count <= 1) return NumOps.Zero;
        var centered = CenterRows(StackDirections(gradients));
        return Engine.TensorSum(Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, keepDims: false));
    }

    /// <summary>L_align measured at the given parameters from each task's unclipped support gradient.</summary>
    internal T AlignmentLossAt(IReadOnlyList<IMetaLearningTask<T, TInput, TOutput>> tasks, Vector<T> parameters)
        => AlignmentLoss(SupportGradientsAt(tasks, parameters));

    /// <summary>Each task's unclipped support gradient at the given parameters; the model's parameters are restored.</summary>
    internal List<Vector<T>> SupportGradientsAt(IReadOnlyList<IMetaLearningTask<T, TInput, TOutput>> tasks, Vector<T> parameters)
    {
        var resume = ParamModel.GetParameters();
        var gradients = new List<Vector<T>>(tasks.Count);
        foreach (var task in tasks)
            gradients.Add(SupportGradientAt(task, parameters));
        ParamModel.SetParameters(resume);
        return gradients;
    }

    /// <summary>
    /// dL_align/dtheta, or null for fewer than two tasks.
    /// </summary>
    /// <remarks>
    /// With c_t = (2 / T)(u_t - mean_s u_s) = dL/du_t, the chain rule through u = g / (n + eps), n = ||g||, gives
    /// v_t = dL/dg_t = c_t / (n + eps) - g (g . c_t) / (n (n + eps)^2), and dL/dtheta = sum_t H_t v_t with H_t the
    /// Hessian of task t's support loss. The Hessian-vector products are central differences of support gradients.
    /// Master returned sum_t 2 (g_t - mean) / T, which is identically zero (deviations from a mean sum to zero), so
    /// the alignment term never moved the parameters. Global-norm clipping rescales g without changing u, so the
    /// unclipped gradients are the right inputs here.
    /// </remarks>
    internal Vector<T>? ComputeAlignmentGradient(
        List<Vector<T>> gradients, IReadOnlyList<IMetaLearningTask<T, TInput, TOutput>> tasks, Vector<T> parameters)
    {
        if (gradients.Count <= 1) return null;

        var c = Engine.TensorMultiplyScalar(CenterRows(StackDirections(gradients)), NumOps.FromDouble(2.0 / gradients.Count));

        var resume = ParamModel.GetParameters();
        var total = new Vector<T>(_paramDim);
        for (int t = 0; t < gradients.Count; t++)
        {
            var ct = Engine.TensorSlice(c, new[] { t, 0 }, new[] { 1, _paramDim }).ToVector();
            total = Engine.Add(total, HessianVectorProduct(tasks[t], parameters, DirectionChainRule(gradients[t], ct)));
        }

        ParamModel.SetParameters(resume);
        return total;
    }

    /// <summary>v = c / (n + eps) - g (g . c) / (n (n + eps)^2) for n = ||g||; the second term is 0 when g = 0.</summary>
    private Vector<T> DirectionChainRule(Vector<T> gradient, Vector<T> c)
    {
        var g = Fit(gradient);
        T n = NumOps.Sqrt(Engine.DotProduct(g, g));
        T nEps = NumOps.Add(n, NumOps.FromDouble(1e-10));
        var v = Engine.Divide(c, nEps);
        if (!NumOps.GreaterThan(n, NumOps.Zero)) return v;
        T coefficient = NumOps.Divide(Engine.DotProduct(g, c), NumOps.Multiply(n, NumOps.Multiply(nEps, nEps)));
        return Engine.Subtract(v, Engine.Multiply(g, coefficient));
    }

    /// <summary>
    /// H v for the task's support loss: (grad(theta + h v^) - grad(theta - h v^)) ||v|| / (2h), v^ = v / ||v||,
    /// h = 1e-4; zero for v = 0.
    /// </summary>
    private Vector<T> HessianVectorProduct(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> parameters, Vector<T> v)
    {
        T norm = NumOps.Sqrt(Engine.DotProduct(v, v));
        if (!NumOps.GreaterThan(norm, NumOps.Zero)) return new Vector<T>(_paramDim);

        T h = NumOps.FromDouble(1e-4);
        var offset = Engine.Multiply(v, NumOps.Divide(h, norm));
        var plus = SupportGradientAt(task, Engine.Add(Fit(parameters), offset));
        var minus = SupportGradientAt(task, Engine.Subtract(Fit(parameters), offset));
        return Engine.Multiply(Engine.Subtract(plus, minus), NumOps.Divide(norm, NumOps.Multiply(NumOps.FromDouble(2.0), h)));
    }

    /// <summary>The task's unclipped support gradient at the given parameters (first paramDim entries).</summary>
    private Vector<T> SupportGradientAt(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> parameters)
    {
        ParamModel.SetParameters(parameters);
        return Fit(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
    }

    /// <summary>[tasks, paramDim] rows of u_t = g_t / (||g_t|| + 1e-10).</summary>
    private Tensor<T> StackDirections(List<Vector<T>> gradients)
    {
        var rows = new Tensor<T>[gradients.Count];
        for (int t = 0; t < rows.Length; t++)
            rows[t] = Tensor<T>.FromVector(UnitDirection(gradients[t]));
        return Engine.TensorStack(rows, 0);
    }

    private Vector<T> UnitDirection(Vector<T> gradient)
    {
        var g = Fit(gradient);
        T norm = NumOps.Add(NumOps.Sqrt(Engine.DotProduct(g, g)), NumOps.FromDouble(1e-10));
        return Engine.Divide(g, norm);
    }

    private Vector<T> Fit(Vector<T> vector)
        => vector.Length == _paramDim ? vector : vector.GetSubVector(0, _paramDim);

    /// <summary>Subtracts each column's mean from a [tasks, dims] stack.</summary>
    private Tensor<T> CenterRows(Tensor<T> stacked)
        => Engine.TensorSubtract(stacked, Engine.TensorBroadcastTo(
            Engine.ReduceMean(stacked, new[] { 0 }, keepDims: true), stacked.Shape.ToArray()));
}
