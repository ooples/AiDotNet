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
/// Implementation of EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// EPNet refines embeddings through label propagation on a nearest-neighbor graph.
/// By propagating feature information between similar examples, embeddings become
/// smoother and more discriminative for few-shot classification.
/// </para>
/// <para><b>For Beginners:</b> EPNet makes features better by sharing information:
///
/// **The insight:**
/// Features extracted by a neural network are good but noisy. If two examples are
/// similar, they should have similar features. EPNet enforces this by propagating
/// features through a similarity graph.
///
/// **How it works:**
/// 1. Extract features for all examples (support + query)
/// 2. Build a k-nearest-neighbor graph based on feature similarity
/// 3. Propagate features through the graph (like heat diffusion)
///    - Each node averages its neighbors' features (weighted by similarity)
///    - Repeat for several iterations
/// 4. The propagated features are smoother and more consistent
/// 5. Classify using the refined features
///
/// **Why propagation helps:**
/// - Noisy features get smoothed out (noise reduction)
/// - Cluster structure becomes clearer (better separation)
/// - Query examples near support clusters get pulled toward them
/// - Works transductively: ALL queries benefit from each other
/// </para>
/// <para><b>Algorithm - EPNet:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor
///
/// # Embedding Propagation
/// for each task:
///     z = f_theta(all_examples)                 # Extract features
///
///     # Build kNN graph
///     W = knn_graph(z, k=num_neighbors)         # Adjacency matrix
///     D = diag(sum(W, axis=1))                  # Degree matrix
///     S = D^(-1/2) @ W @ D^(-1/2)              # Normalized graph
///
///     # Propagate embeddings
///     z_prop = z
///     for iter in range(propagation_iterations):
///         z_prop = alpha * S @ z_prop + (1-alpha) * z    # Diffusion + anchor
///
///     # Classify with propagated features
///     prototypes = mean(z_prop[support], per_class)
///     logits = -distance(z_prop[query], prototypes)
///     loss = cross_entropy(logits, query_labels)
/// </code>
/// </para>
/// <para>
/// Reference: Rodriguez, P., Laradji, I., Drouin, A., &amp; Lacoste, A. (2020).
/// Embedding Propagation: Smoother Manifold for Few-Shot Classification. CVPR 2020.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation URL corrected: the correct arXiv id for this title is 2003.04151 (ECCV 2020). The recorded
// 2003.04557 is "Spin detection via parametric frequency conversion in a membrane resonator", an
// unrelated physics paper. Title, authors and year were already right.
[ResearchPaper("Embedding Propagation: Smoother Manifold for Few-Shot Classification",
    "https://arxiv.org/abs/2003.04151",
    Year = 2020,
    Authors = "Pau Rodriguez, Issam Laradji, Alexandre Drouin, Alexandre Lacoste")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class EPNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly EPNetOptions<T, TInput, TOutput> _epnetOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.EPNet;

    /// <summary>Initializes a new EPNet meta-learner.</summary>
    /// <param name="options">Configuration options for EPNet.</param>
    public EPNetAlgorithm(EPNetOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _epnetOptions = options;
    }

    /// <summary>
    /// Performs embedding propagation on a feature vector using a proper kNN similarity graph.
    /// Builds pairwise distance matrix, selects k-nearest neighbors by feature similarity,
    /// constructs a normalized adjacency matrix, and iteratively diffuses features.
    /// </summary>
    /// <param name="features">Concatenated features for all examples.</param>
    /// <returns>Propagated (smoothed) features.</returns>
    private Vector<T> PropagateEmbeddings(Vector<T> features)
    {
        int n = features.Length;
        if (n < 2) return features;

        int k = Math.Min(_epnetOptions.NumNeighbors, n - 1);
        T alpha = NumOps.FromDouble(_epnetOptions.PropagationAlpha);
        var square = new[] { n, n };

        // Step 1: Compute pairwise squared distances between all feature elements
        var column = Engine.Reshape(Tensor<T>.FromVector(features), new[] { n, 1 });
        var distances = Engine.TensorSquare(Engine.TensorSubtract(
            Engine.TensorBroadcastTo(column, square),
            Engine.TensorBroadcastTo(Engine.Reshape(column, new[] { 1, n }), square)));

        // Step 2: Build kNN adjacency matrix W using feature similarity, with the median distance over distinct
        // pairs (i < j) as the adaptive RBF bandwidth
        var upper = new List<int>(n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                upper.Add(i * n + j);
        var pairDistances = Engine.TensorIndexSelect(
            Engine.Reshape(distances, new[] { n * n }), new Tensor<int>(upper.ToArray(), new[] { upper.Count }), 0);
        T medianDist = Engine.TensorSort(pairDistances, 0, false).Values[upper.Count / 2];
        T sigma2 = NumOps.GreaterThan(medianDist, NumOps.FromDouble(1e-10)) ? medianDist : NumOps.FromDouble(1e-10);

        // Each node links to its k nearest other nodes (distance, then index), symmetrized, with RBF weights
        var rbf = Engine.TensorExp(Engine.TensorMultiplyScalar(
            distances, NumOps.Negate(NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), sigma2)))));
        var W = Engine.TensorMultiply(NearestNeighborGraph.SymmetricNearestMask(Engine, distances, k), rbf);

        // Step 3: Compute degree matrix D and normalized adjacency S = D^(-1/2) W D^(-1/2)
        var degree = Engine.ReduceSum(W, new[] { 1 }, true);
        var connected = Engine.TensorGreaterThan(degree, NumOps.FromDouble(1e-10));
        var invSqrtDegree = Engine.TensorWhere(connected,
            Engine.TensorDivide(Engine.TensorOnesLike(degree),
                Engine.TensorSqrt(Engine.TensorWhere(connected, degree, Engine.TensorOnesLike(degree)))),
            Engine.TensorZerosLike(degree));
        var S = Engine.TensorMultiply(Engine.TensorMultiply(
            Engine.TensorBroadcastTo(invSqrtDegree, square), W),
            Engine.TensorBroadcastTo(Engine.Reshape(invSqrtDegree, new[] { 1, n }), square));

        // Step 4: Iterative feature propagation: z_prop = alpha * S @ z_prop + (1-alpha) * z_original
        var anchor = Engine.Multiply(features, NumOps.Subtract(NumOps.One, alpha));
        var propagated = features.Clone();
        for (int iter = 0; iter < _epnetOptions.PropagationIterations; iter++)
        {
            var aggregated = Engine.Reshape(
                Engine.TensorMatMul(S, Engine.Reshape(Tensor<T>.FromVector(propagated), new[] { n, 1 })), new[] { n });
            propagated = Engine.Add(Engine.Multiply(aggregated.ToVector(), alpha), anchor);
        }

        return propagated;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            // Extract support and query features, propagate on joint graph
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));

            int supportLen = supportFeatures?.Length ?? 0;
            int queryLen = queryFeatures?.Length ?? 0;

            // Run embedding propagation to compute modulation factor
            if (supportFeatures is not null && queryFeatures is not null && supportLen > 0 && queryLen > 0)
            {
                var combined = Vector<T>.Concatenate(supportFeatures, queryFeatures);

                var propagated = PropagateEmbeddings(combined);

                // Compute modulation from propagation effect on features
                ParameterModulation.ScaleByMeanClampedRatio(Engine, ParamModel, combined, propagated, 0.5, 2.0, 1e-10);
            }

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _epnetOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();

        // Extract support and query features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);

        // Transductive propagation: concatenate support + query onto a single graph,
        // propagate features jointly, then split back. This is the core EPNet insight:
        // query features benefit from support features and vice versa.
        Vector<T>? propagatedSupport = null;
        Vector<T>? propagatedQuery = null;

        int supportLen = supportFeatures?.Length ?? 0;
        int queryLen = queryFeatures?.Length ?? 0;

        if (supportLen > 0 || queryLen > 0)
        {
            // Concatenate support + query into a single feature vector (a missing set contributes nothing)
            var combined = Vector<T>.Concatenate(
                supportFeatures ?? new Vector<T>(0), queryFeatures ?? new Vector<T>(0));

            // Propagate on the joint graph
            var propagated = PropagateEmbeddings(combined);

            // Split back into support and query portions
            if (supportLen > 0)
            {
                propagatedSupport = propagated.GetSubVector(0, supportLen);
            }
            if (queryLen > 0)
            {
                propagatedQuery = propagated.GetSubVector(supportLen, queryLen);
            }
        }

        // Compute feature-delta modulation: how much propagation changed support features
        // Use this to create a task-specific parameter perturbation
        Vector<T>? modulationFactors = null;
        if (propagatedSupport != null && supportFeatures != null && supportLen > 0)
        {
            // Compute relative change: propagated / raw (clamped to avoid extreme values)
            modulationFactors = ParameterModulation.MeanClampedRatio(Engine, supportFeatures, propagatedSupport, 0.5, 2.0, 1e-10)
                ?? new Vector<T>(new[] { NumOps.One });
        }

        return new EPNetModel<T, TInput, TOutput>(
            MetaModel, currentParams, propagatedSupport, propagatedQuery, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for EPNet with embedding propagation.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses features that have been smoothed
/// by propagation across a similarity graph, where nearby examples share
/// information to produce more consistent and discriminative embeddings.
/// </para>
/// </remarks>
internal class EPNetModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _propagatedSupport;
    private readonly Vector<T>? _propagatedQuery;
    // Not readonly: a readonly Vector<T> field is not persisted by the generated model state, and these
    // factors belong in the adapted model's checkpoint as the double[] they replace did.
    private Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _propagatedSupport;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    public EPNetModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? propagatedSupport,
        Vector<T>? propagatedQuery,
        Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
        _backboneParams = backboneParams;
        _propagatedSupport = propagatedSupport;
        _propagatedQuery = propagatedQuery;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            // Apply propagation-derived parameter modulation
            var modulatedParams = ParameterModulation.Apply(AiDotNetEngine.Current, _backboneParams, _modulationFactors);
            InterfaceGuard.Parameterizable(_model).SetParameters(modulatedParams);
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
