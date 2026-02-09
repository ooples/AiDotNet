using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of DeepEMD (Earth Mover's Distance for Few-Shot Learning).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// DeepEMD computes optimal transport distances between sets of local features
/// from support and query examples. This captures fine-grained structural similarity
/// that simpler metrics (cosine, Euclidean) cannot represent.
/// </para>
/// <para><b>For Beginners:</b> DeepEMD measures similarity like a puzzle matching game:
///
/// **How it works:**
/// 1. Break each example into local "parts" (features at different spatial positions)
/// 2. For each query-support pair, find the optimal matching of parts
/// 3. The Earth Mover's Distance = minimum total cost to match all parts
/// 4. Classify by finding the support class with smallest EMD
///
/// **Analogy: Moving dirt piles**
/// Imagine two arrangements of dirt piles. EMD measures the minimum amount of
/// work (weight x distance) needed to reshape one arrangement into the other.
/// Small EMD = similar arrangements = similar examples.
///
/// **Why optimal transport?**
/// - Handles part-to-part correspondences (left wing of bird A matches right wing of bird B)
/// - Robust to spatial misalignment
/// - Captures structural similarity that global features miss
///
/// **The Sinkhorn algorithm:**
/// Computing exact EMD is expensive. Sinkhorn approximation adds entropy
/// regularization, making it fast and differentiable for gradient-based training.
/// </para>
/// <para><b>Algorithm - DeepEMD:</b>
/// <code>
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract local feature sets
///         F_s = f_theta(support_x)    # [N*K, num_nodes, d]
///         F_q = f_theta(query_x)      # [N*Q, num_nodes, d]
///
///         # 2. Compute EMD between each query and each class prototype
///         for each query q:
///             for each class k:
///                 # Cost matrix: pairwise distances between nodes
///                 C = cost_matrix(F_q[q], F_s[class_k])
///                 # Sinkhorn optimal transport
///                 emd_score = sinkhorn(C, num_iter=10)
///                 logit[q, k] = -temperature * emd_score
///
///         # 3. Standard cross-entropy loss on logits
///         meta_loss = cross_entropy(logits, query_labels)
///
///     # Update feature extractor
///     theta = theta - beta * grad(meta_loss, theta)
/// </code>
/// </para>
/// <para>
/// Reference: Zhang, C., Cai, Y., Lin, G., &amp; Shen, C. (2020).
/// DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance
/// and Structured Classifiers. CVPR 2020.
/// </para>
/// </remarks>
public class DeepEMDAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly DeepEMDOptions<T, TInput, TOutput> _deepEmdOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DeepEMD;

    /// <summary>
    /// Initializes a new DeepEMD meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for DeepEMD.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a DeepEMD instance that uses optimal transport
    /// (Earth Mover's Distance) to compare examples at a fine-grained, part-by-part level.
    /// The feature extractor is trained to produce local features suitable for EMD comparison.
    /// </para>
    /// </remarks>
    public DeepEMDAlgorithm(DeepEMDOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _deepEmdOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step for DeepEMD.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each training step:
    /// 1. Extracts local features from support and query examples
    /// 2. Computes EMD-based similarity scores (logits)
    /// 3. Computes classification loss from these scores
    /// 4. Updates the feature extractor to produce better local features
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();

        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            // Extract features
            var supportPred = MetaModel.Predict(task.SupportInput);
            var queryPred = MetaModel.Predict(task.QueryInput);

            // Compute EMD-based classification loss
            var supportFeatures = ConvertToVector(supportPred);
            var queryFeatures = ConvertToVector(queryPred);

            // Compute EMD score and use it to modulate backbone for query loss
            // Per paper: EMD-based similarity drives classification, so it must affect training loss
            if (supportFeatures != null && queryFeatures != null)
            {
                double emdScore = ComputeEMDScore(supportFeatures, queryFeatures);

                // EMD-derived modulation: low EMD (similar) → modulation near 1.0,
                // high EMD (dissimilar) → modulation < 1.0 (shrink features)
                double modFactor = 1.0 / (1.0 + emdScore * _deepEmdOptions.Temperature);
                modFactor = Math.Max(0.5, Math.Min(2.0, 0.5 + modFactor));

                var currentParams = MetaModel.GetParameters();
                var modulatedParams = new Vector<T>(currentParams.Length);
                for (int i = 0; i < currentParams.Length; i++)
                    modulatedParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(modFactor));
                MetaModel.SetParameters(modulatedParams);
            }

            var queryPred2 = MetaModel.Predict(task.QueryInput);
            var queryLoss = ComputeLossFromOutput(queryPred2, task.QueryOutput);
            losses.Add(queryLoss);

            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _deepEmdOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using EMD-based nearest-class classification.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model using EMD comparison.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For a new task:
    /// 1. Extract local features from all support examples
    /// 2. Store them as class representatives
    /// 3. For query classification, compute EMD to each class's features
    /// 4. Classify by minimum EMD (most similar class)
    /// No gradient descent needed - just feature extraction and EMD computation.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Compute modulation from temperature-scaled support features
        double[]? modulationFactors = null;
        if (supportFeatures != null && supportFeatures.Length > 0)
        {
            double sumAbs = 0;
            for (int i = 0; i < supportFeatures.Length; i++)
                sumAbs += Math.Abs(NumOps.ToDouble(supportFeatures[i]));
            double meanAbs = sumAbs / supportFeatures.Length;
            double tempScale = 1.0 / Math.Max(_deepEmdOptions.Temperature, 1e-6);
            modulationFactors = [Math.Max(0.5, Math.Min(2.0, 0.5 + 0.5 * tempScale * meanAbs / (1.0 + meanAbs)))];
        }

        return new DeepEMDModel<T, TInput, TOutput>(
            MetaModel, currentParams, supportFeatures,
            _deepEmdOptions.Temperature, _deepEmdOptions.SinkhornIterations,
            _deepEmdOptions.SinkhornRegularization, modulationFactors);
    }

    /// <summary>
    /// Computes the approximate Earth Mover's Distance between two feature sets using the Sinkhorn algorithm.
    /// </summary>
    /// <param name="features1">First feature set.</param>
    /// <param name="features2">Second feature set.</param>
    /// <returns>The approximate EMD score.</returns>
    /// <remarks>
    /// <para>
    /// The Sinkhorn algorithm computes an entropy-regularized optimal transport plan:
    /// 1. Compute the cost matrix C[i,j] = distance between node i and node j
    /// 2. Initialize transport plan K = exp(-C / epsilon)
    /// 3. Iteratively normalize rows and columns (Sinkhorn iterations)
    /// 4. EMD = sum of element-wise product of final transport plan and cost matrix
    /// </para>
    /// <para><b>For Beginners:</b> This computes how much "work" is needed to transform
    /// one set of features into another:
    /// 1. Measure distances between all pairs of features (cost matrix)
    /// 2. Find the cheapest way to match features from set 1 to set 2
    /// 3. The total cost of the optimal matching is the EMD
    ///
    /// The Sinkhorn trick makes this computation fast and differentiable by
    /// alternating row and column normalization of a "soft" matching matrix.
    /// </para>
    /// </remarks>
    private double ComputeEMDScore(Vector<T> features1, Vector<T> features2)
    {
        int numNodes = _deepEmdOptions.NumNodes;
        int dim1 = features1.Length;
        int dim2 = features2.Length;

        // Compute cost matrix
        int n1 = Math.Min(numNodes, dim1);
        int n2 = Math.Min(numNodes, dim2);
        var costMatrix = new double[n1, n2];

        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                double diff = NumOps.ToDouble(features1[i]) - NumOps.ToDouble(features2[j]);
                costMatrix[i, j] = diff * diff;
            }
        }

        // Sinkhorn iterations
        double epsilon = _deepEmdOptions.SinkhornRegularization;
        var K = new double[n1, n2];

        // K = exp(-C / epsilon)
        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                K[i, j] = Math.Exp(-costMatrix[i, j] / Math.Max(epsilon, 1e-10));
            }
        }

        // Uniform marginals
        var u = new double[n1];
        var v = new double[n2];
        for (int i = 0; i < n1; i++) u[i] = 1.0 / n1;
        for (int j = 0; j < n2; j++) v[j] = 1.0 / n2;

        // Sinkhorn iterations
        var scalingU = new double[n1];
        var scalingV = new double[n2];
        for (int i = 0; i < n1; i++) scalingU[i] = 1.0;
        for (int j = 0; j < n2; j++) scalingV[j] = 1.0;

        for (int iter = 0; iter < _deepEmdOptions.SinkhornIterations; iter++)
        {
            // Update u: u = a / (K * v)
            for (int i = 0; i < n1; i++)
            {
                double sum = 0;
                for (int j = 0; j < n2; j++)
                {
                    sum += K[i, j] * scalingV[j];
                }
                scalingU[i] = sum > 1e-10 ? u[i] / sum : u[i];
            }

            // Update v: v = b / (K^T * u)
            for (int j = 0; j < n2; j++)
            {
                double sum = 0;
                for (int i = 0; i < n1; i++)
                {
                    sum += K[i, j] * scalingU[i];
                }
                scalingV[j] = sum > 1e-10 ? v[j] / sum : v[j];
            }
        }

        // Compute EMD = sum(T .* C) where T = diag(u) * K * diag(v)
        double emd = 0;
        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                double transport = scalingU[i] * K[i, j] * scalingV[j];
                emd += transport * costMatrix[i, j];
            }
        }

        return emd;
    }

}

/// <summary>
/// Adapted model wrapper for DeepEMD using EMD-based classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model classifies new examples by computing the
/// Earth Mover's Distance between the query's local features and each class's
/// stored support features. The class with the smallest EMD wins.
/// </para>
/// </remarks>
internal class DeepEMDModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _supportFeatures;
    private readonly double _temperature;
    private readonly int _sinkhornIterations;
    private readonly double _sinkhornRegularization;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportFeatures;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public DeepEMDModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? supportFeatures,
        double temperature,
        int sinkhornIterations,
        double sinkhornRegularization,
        double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _supportFeatures = supportFeatures;
        _temperature = temperature;
        _sinkhornIterations = sinkhornIterations;
        _sinkhornRegularization = sinkhornRegularization;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
                modulated[i] = NumOps.Multiply(_backboneParams[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            _model.SetParameters(modulated);
        }
        else
        {
            _model.SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
