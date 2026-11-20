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
/// Implementation of HyperCLIP: Contrastive Learning for Hypernetwork-based Meta-Learning.
/// </summary>
/// <remarks>
/// <para>
/// HyperCLIP uses contrastive alignment between task embeddings (from support gradients)
/// and parameter embeddings (from adapted parameter deltas). An InfoNCE contrastive loss
/// aligns each task's gradient signature with its adapted parameter fingerprint, learning
/// a shared projection space. This cross-modal alignment improves adaptation quality.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// For each task in batch:
///   Task embedding: t_i = project_task(compress(grad_support_i))
///   Adapt: θ_i via MAML
///   Param embedding: p_i = project_param(compress(θ_i - θ_init))
///
/// Contrastive loss (InfoNCE across batch):
///   sim(i,j) = cosine(t_i, p_j) / τ
///   L_NCE = -Σ_i log(exp(sim(i,i)) / Σ_j exp(sim(i,j)))
///
/// L_meta = Σ L_query_i + ContrastiveWeight * L_NCE
/// Outer: update θ_init, update projections via SPSA
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Learning to Learn with Compound HD Models",
    "https://arxiv.org/abs/2106.04989",
    Year = 2021,
    Authors = "Baifeng Shi, Dinghuai Zhang, Qi Dai, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class HyperCLIPAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly HyperCLIPOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    private const double SpsaLearningRateMultiplier = 0.1;

    /// <summary>Projection weights: task projection (embDim × projDim) + param projection (embDim × projDim).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _projectionWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperCLIP;

    public HyperCLIPAlgorithm(HyperCLIPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.EmbeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "EmbeddingDim must be positive.");
        if (options.ProjectionDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ProjectionDim must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;

        int embDim = options.EmbeddingDim;
        int projDim = options.ProjectionDim;
        // Task projection + param projection
        int totalWeights = 2 * embDim * projDim;
        _projectionWeights = new Vector<T>(totalWeights);
        double scale = 1.0 / Math.Sqrt(projDim);
        for (int i = 0; i < totalWeights; i++)
            _projectionWeights[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        int embDim = _algoOptions.EmbeddingDim;
        int projDim = _algoOptions.ProjectionDim;
        int numTasks = taskBatch.Tasks.Length;

        // Collect task and param embeddings for contrastive loss
        var taskEmbeddings = new Vector<T>[numTasks];
        var paramEmbeddings = new Vector<T>[numTasks];

        for (int i = 0; i < numTasks; i++)
        {
            var task = taskBatch.Tasks[i];

            // Task embedding from support gradient
            ParamModel.SetParameters(initParams);
            var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var gradFeatures = CompressVector(supportGrad, embDim);
            taskEmbeddings[i] = ProjectTask(gradFeatures);

            // Standard MAML adaptation
            var adaptedParams = initParams.Clone();

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
            }

            // Param embedding from parameter delta
            var paramDelta = CompressParamDelta(initParams, adaptedParams, embDim);
            paramEmbeddings[i] = ProjectParam(paramDelta);

            // Query loss
            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // InfoNCE contrastive loss
        T contrastiveLoss = ComputeInfoNCE(taskEmbeddings, paramEmbeddings, numTasks, projDim);

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _projectionWeights, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeCLIPLoss);

        // The contrastive loss is added to every task loss, so it adds once to their mean
        var meanLoss = ComputeMean(losses);
        return numTasks > 1
            ? NumOps.Add(meanLoss, NumOps.Multiply(NumOps.FromDouble(_algoOptions.ContrastiveWeight), contrastiveLoss))
            : meanLoss;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// tanh of the bucket means of theta' - theta: bucket e sums the delta over [e * b, e * b + b) within the
    /// parameters and divides by the full bucket size b = max(1, paramDim / targetDim).
    /// </summary>
    private Vector<T> CompressParamDelta(Vector<T> initParams, Vector<T> adaptedParams, int targetDim)
    {
        int bucketSize = Math.Max(1, _paramDim / targetDim);
        int covered = targetDim * bucketSize;
        var delta = Engine.Subtract(adaptedParams.GetSubVector(0, _paramDim), initParams.GetSubVector(0, _paramDim));
        var padded = covered <= _paramDim
            ? delta.GetSubVector(0, covered)
            : Vector<T>.Concatenate(delta, new Vector<T>(covered - _paramDim));
        var sums = Engine.ReduceSum(Tensor<T>.FromVector(padded, new[] { targetDim, bucketSize }), new[] { 1 }, keepDims: false);
        return Engine.Tanh(Engine.Divide(sums.ToVector(), NumOps.FromDouble(bucketSize)));
    }

    /// <summary>L2-normalized W f for the projection block starting at <paramref name="offset"/>.</summary>
    private Vector<T> Project(Vector<T> features, int offset)
    {
        int embDim = _algoOptions.EmbeddingDim;
        int projDim = _algoOptions.ProjectionDim;
        var weights = Tensor<T>.FromVector(_projectionWeights.GetSubVector(offset, embDim * projDim), new[] { projDim, embDim });
        var input = Tensor<T>.FromVector(features.Length == embDim ? features : features.GetSubVector(0, embDim), new[] { embDim, 1 });
        return L2Normalize(Engine.TensorMatMul(weights, input).ToVector());
    }

    // Task projection: first embDim * projDim weights
    private Vector<T> ProjectTask(Vector<T> features) => Project(features, 0);

    // Param projection: the block after the task projection weights
    private Vector<T> ProjectParam(Vector<T> features)
        => Project(features, _algoOptions.EmbeddingDim * _algoOptions.ProjectionDim);

    /// <summary>
    /// Mean over tasks i of -(s_ii - max_j s_ij - log(sum_j exp(s_ij - max_j s_ij) + 1e-10)) with
    /// s_ij = &lt;t_i, p_j&gt; / tau.
    /// </summary>
    private T ComputeInfoNCE(Vector<T>[] taskEmb, Vector<T>[] paramEmb, int n, int dim)
    {
        if (n <= 1) return NumOps.Zero;

        var taskRows = new Tensor<T>[n];
        var paramRows = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            taskRows[i] = Tensor<T>.FromVector(taskEmb[i].GetSubVector(0, dim));
            paramRows[i] = Tensor<T>.FromVector(paramEmb[i].GetSubVector(0, dim));
        }

        var square = new[] { n, n };
        var sims = Engine.TensorDivideScalar(
            Engine.TensorMatMulTransposed(Engine.TensorStack(taskRows, 0), Engine.TensorStack(paramRows, 0)),
            NumOps.FromDouble(_algoOptions.ContrastiveTemperature));
        var shifted = Engine.TensorSubtract(sims, Engine.TensorBroadcastTo(Engine.ReduceMax(sims, new[] { 1 }, keepDims: true), square));

        // Log-sum-exp for numerical stability
        var logSum = Engine.TensorLog(Engine.TensorAddScalar(
            Engine.ReduceSum(Engine.TensorExp(shifted), new[] { 1 }, keepDims: false), NumOps.FromDouble(1e-10)));
        var positive = Engine.ReduceSum(Engine.TensorMultiply(shifted, Engine.TensorEye<T>(n)), new[] { 1 }, keepDims: false);
        return Engine.TensorMean(Engine.TensorSubtract(logSum, positive));
    }

    private double ComputeCLIPLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        int embDim = _algoOptions.EmbeddingDim;
        int projDim = _algoOptions.ProjectionDim;
        int numTasks = taskBatch.Tasks.Length;

        var taskEmbs = new Vector<T>[numTasks];
        var paramEmbs = new Vector<T>[numTasks];

        for (int i = 0; i < numTasks; i++)
        {
            var task = taskBatch.Tasks[i];

            // Task embedding from support gradient
            ParamModel.SetParameters(initParams);
            var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            taskEmbs[i] = ProjectTask(CompressVector(supportGrad, embDim));

            // Adapt
            var ap = initParams.Clone();
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(ap);
                var g = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                ap = ApplyInnerGradients(ap, g, _algoOptions.InnerLearningRate, step == 0);
            }

            // Param embedding from delta
            paramEmbs[i] = ProjectParam(CompressParamDelta(initParams, ap, embDim));

            ParamModel.SetParameters(ap);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        // Include contrastive loss so projections are optimized for alignment: (sum + w * c * n) / n
        T contrastive = ComputeInfoNCE(taskEmbs, paramEmbs, numTasks, projDim);

        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the objective is reduced in T and converted once.
        return NumOps.ToDouble(NumOps.Add(ComputeMean(losses),
            NumOps.Multiply(NumOps.FromDouble(_algoOptions.ContrastiveWeight), contrastive)));
    }
}
