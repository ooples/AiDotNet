using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of LaplacianShot (Laplacian Regularized Few-Shot Learning).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// LaplacianShot augments nearest-centroid classification with Laplacian regularization
/// over a kNN graph of query features. This propagates labels from confident predictions
/// to uncertain ones based on feature similarity.
/// </para>
/// <para><b>For Beginners:</b> LaplacianShot adds graph-based label propagation:
///
/// **Step 1: Initial classification (like SimpleShot)**
/// Classify each query by distance to support centroids.
///
/// **Step 2: Build similarity graph**
/// Connect each query to its k nearest neighbors (in feature space).
///
/// **Step 3: Label propagation**
/// Iteratively smooth predictions across the graph:
/// - If your neighbors are confidently "cat", you should also lean toward "cat"
/// - The Laplacian matrix encodes this smoothness constraint mathematically
///
/// **Why it helps:**
/// Query examples near the decision boundary get "pulled" toward the correct class
/// by their more confident neighbors, reducing boundary errors.
/// </para>
/// <para><b>Algorithm - LaplacianShot:</b>
/// <code>
/// # Given: prototypes p_k, query features z_q
///
/// # 1. Initial logits from nearest centroid
/// logits = -||z_q - p_k||^2              # [Q, K]
///
/// # 2. Build kNN graph over query features
/// W[i,j] = exp(-||z_q[i] - z_q[j]||^2 / sigma)  if j in kNN(i)
/// D = diag(sum(W, axis=1))               # Degree matrix
/// L = D - W                               # Graph Laplacian
///
/// # 3. Laplacian-regularized classification
/// # Solve: min_Y ||Y - logits||^2 + lambda * trace(Y^T L Y)
/// # Solution: Y = (I + lambda * L)^-1 * logits
/// predictions = (I + lambda * L)^-1 @ logits
/// </code>
/// </para>
/// <para>
/// Reference: Ziko, I., Dolz, J., Granger, E., &amp; Ben Ayed, I. (2020).
/// Laplacian Regularized Few-Shot Learning. ICML 2020.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Laplacian Regularized Few-Shot Learning",
    "https://arxiv.org/abs/2006.15486",
    Year = 2020,
    Authors = "Imtiaz Masud Ziko, Jose Dolz, Eric Granger, Ismail Ben Ayed")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class LaplacianShotAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly LaplacianShotOptions<T, TInput, TOutput> _lapShotOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LaplacianShot;

    /// <summary>
    /// Initializes a new LaplacianShot meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for LaplacianShot.</param>
    public LaplacianShotAlgorithm(LaplacianShotOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _lapShotOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step for LaplacianShot.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training is standard episodic training. The Laplacian
    /// regularization only applies during adaptation (inference), so training is the same
    /// as training any metric-based method.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            var queryLoss = ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        ApplyOuterUpdate(initParams, metaGradients, _lapShotOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using Laplacian-regularized nearest-centroid classification.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model with Laplacian-refined predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adaptation is:
    /// 1. Compute support centroids (class means)
    /// 2. Compute initial predictions for all queries
    /// 3. Build a kNN graph over query features
    /// 4. Smooth predictions using the graph Laplacian
    /// 5. Return model with smoothed decision boundaries
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Perform Laplacian refinement
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);
        var refinedWeights = LaplacianRefine(supportFeatures, queryFeatures);

        // Compute modulation factors from refined vs raw support features
        Vector<T>? modulationFactors = null;
        if (supportFeatures != null && refinedWeights != null)
        {
            modulationFactors = ParameterModulation.MeanClampedRatio(Engine, supportFeatures, refinedWeights, 0.5, 2.0, 1e-10);
        }

        return new LaplacianShotModel<T, TInput, TOutput>(MetaModel, currentParams, refinedWeights, modulationFactors);
    }

    /// <summary>
    /// Applies Laplacian regularization to refine predictions using the query graph.
    /// </summary>
    /// <param name="supportFeatures">Support set features (centroids).</param>
    /// <param name="queryFeatures">Query set features for graph construction.</param>
    /// <returns>Refined classification weights after Laplacian smoothing.</returns>
    /// <remarks>
    /// <para>
    /// Constructs a kNN graph over query features, computes the graph Laplacian,
    /// and solves the regularized classification problem iteratively:
    /// Y_{t+1} = Y_t - alpha * (lambda * L @ Y_t + (Y_t - logits_0))
    /// </para>
    /// <para><b>For Beginners:</b> This smooths predictions over the query graph:
    /// 1. Connect similar queries (kNN graph)
    /// 2. Start with initial predictions (nearest centroid)
    /// 3. Iteratively: each query adjusts its prediction based on neighbors
    /// 4. Result: smooth predictions where nearby queries agree
    /// </para>
    /// </remarks>
    private Vector<T>? LaplacianRefine(Vector<T>? supportFeatures, Vector<T>? queryFeatures)
    {
        if (supportFeatures == null || queryFeatures == null || queryFeatures.Length == 0)
        {
            return supportFeatures;
        }

        // Cap the number of graph nodes to prevent O(n^2) memory explosion.
        // ConvertToVector flattens all predictions into a single 1D vector, so
        // queryFeatures.Length is the total scalar count, not the number of examples.
        // The kNN graph and adjacency matrix scale quadratically with node count.
        const int MaxGraphNodes = 256;
        int numQuery = Math.Min(queryFeatures.Length, MaxGraphNodes);
        int k = Math.Min(_lapShotOptions.KNearestNeighbors, numQuery - 1);
        var square = new[] { numQuery, numQuery };

        // Build kNN graph: squared distances between query nodes, each node linked to its k nearest other nodes
        // (distance, then index), symmetrized, with weights exp(-d / sigma)
        var queries = numQuery == queryFeatures.Length ? queryFeatures : queryFeatures.GetSubVector(0, numQuery);
        var column = Engine.Reshape(Tensor<T>.FromVector(queries), new[] { numQuery, 1 });
        var distances = Engine.TensorSquare(Engine.TensorSubtract(
            Engine.TensorBroadcastTo(column, square),
            Engine.TensorBroadcastTo(Engine.Reshape(column, new[] { 1, numQuery }), square)));
        var kernel = Engine.TensorExp(Engine.TensorMultiplyScalar(
            distances, NumOps.FromDouble(-1.0 / Math.Max(_lapShotOptions.KernelBandwidth, 1e-10))));
        var weights = Engine.TensorMultiply(NearestNeighborGraph.SymmetricNearestMask(Engine, distances, k), kernel);

        // Compute degree and Laplacian L = D - W
        var degree = Engine.ReduceSum(weights, new[] { 1 }, false).ToVector();

        // Initial logits from support-query similarity: q * sum(support)
        var logits = Engine.Multiply(queries, Engine.Sum(supportFeatures));

        // Iterative Laplacian smoothing: y <- y - alpha * (lambda * (D y - W y) + (y - logits))
        var refined = logits.Clone();
        T alpha = NumOps.FromDouble(_lapShotOptions.StepSize);
        T lambda = NumOps.FromDouble(_lapShotOptions.LaplacianWeight);
        for (int iter = 0; iter < _lapShotOptions.PropagationIterations; iter++)
        {
            var neighbors = Engine.Reshape(
                Engine.TensorMatMul(weights, Engine.Reshape(Tensor<T>.FromVector(refined), new[] { numQuery, 1 })),
                new[] { numQuery }).ToVector();
            var lapTerm = Engine.Subtract(Engine.Multiply(degree, refined), neighbors);
            var step = Engine.Add(Engine.Multiply(lapTerm, lambda), Engine.Subtract(refined, logits));
            refined = Engine.Subtract(refined, Engine.Multiply(step, alpha));
        }

        // Convert back: support feature i scaled by sigmoid(refined[i]), or 0.5 past the query nodes
        int length = supportFeatures.Length;
        int scored = Math.Min(length, numQuery);
        var scores = Engine.Sigmoid(scored == numQuery ? refined : refined.GetSubVector(0, scored));
        var scale = scored == length
            ? scores
            : Vector<T>.Concatenate(scores, Engine.Fill(length - scored, NumOps.FromDouble(0.5)));
        var result = Engine.Multiply(supportFeatures, scale);

        return result;
    }

}

/// <summary>
/// Adapted model wrapper for LaplacianShot with graph-refined predictions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses predictions that have been refined
/// using the graph Laplacian, where similar query examples receive similar predictions.
/// </para>
/// </remarks>
internal class LaplacianShotModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _refinedWeights;
    private readonly Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _refinedWeights;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public LaplacianShotModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams,
        Vector<T>? refinedWeights, Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
        _backboneParams = backboneParams;
        _refinedWeights = refinedWeights;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = ParameterModulation.Apply(AiDotNetEngine.Current, _backboneParams, _modulationFactors);
            InterfaceGuard.Parameterizable(_model).SetParameters(modulated);
        }
        else
        {
            InterfaceGuard.Parameterizable(_model).SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
