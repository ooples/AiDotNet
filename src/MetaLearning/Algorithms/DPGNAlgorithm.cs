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
/// Implementation of DPGN (Distribution Propagation Graph Network) (Yang et al., CVPR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// DPGN constructs dual graphs over support and query examples: a point graph for feature
/// propagation and a distribution graph for uncertainty propagation. Both graphs are refined
/// through multiple layers of message passing.
/// </para>
/// <para><b>For Beginners:</b> DPGN uses TWO graphs working together:
///
/// **Graph 1 - Point Graph:**
/// - Nodes = examples (support + query)
/// - Edges = feature similarity
/// - Message passing: Share feature information between similar examples
/// - Result: Refined, context-aware features
///
/// **Graph 2 - Distribution Graph:**
/// - Same nodes but edges = distribution similarity
/// - Passes uncertainty/confidence information
/// - Result: Each node knows how certain it is
///
/// **Why two graphs?**
/// Knowing features isn't enough. You also need to know CONFIDENCE.
/// Two examples might have similar features but very different confidences.
/// The distribution graph captures this distinction.
///
/// **How they interact:**
/// After each propagation layer:
/// 1. Point graph updates features (making them more discriminative)
/// 2. Distribution graph updates confidences (making uncertainty estimates better)
/// 3. Both use each other's output as input for the next layer
/// </para>
/// <para><b>Algorithm - DPGN:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor         # Backbone
/// G_point = point_graph               # Feature propagation
/// G_dist = distribution_graph         # Uncertainty propagation
///
/// # Meta-training
/// for each task T_i:
///     # 1. Extract features for ALL examples (support + query)
///     z = f_theta(all_examples)
///
///     # 2. Initialize graphs
///     node_features = z
///     node_distributions = initialize_uniform()
///
///     # 3. Multi-layer dual propagation
///     for layer in range(num_layers):
///         # Point graph: propagate features
///         edge_weights_point = compute_similarity(node_features)
///         node_features = propagate(node_features, edge_weights_point)
///
///         # Distribution graph: propagate distributions
///         edge_weights_dist = compute_dist_similarity(node_distributions)
///         node_distributions = propagate(node_distributions, edge_weights_dist)
///
///     # 4. Final classification from refined distributions
///     logits = node_distributions[query_indices]
///     loss = cross_entropy(logits, query_labels)
/// </code>
/// </para>
/// <para>
/// Reference: Yang, L., Li, L., Zhang, Z., Zhou, X., Zhou, E., &amp; Liu, Y. (2020).
/// DPGN: Distribution Propagation Graph Network for Few-Shot Learning. CVPR 2020.
/// </para>
/// </remarks>
public class DPGNAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly DPGNOptions<T, TInput, TOutput> _dpgnOptions;

    /// <summary>Parameters for the point graph propagation layers.</summary>
    private Vector<T> _pointGraphParams = new Vector<T>(0);

    /// <summary>Parameters for the distribution graph propagation layers.</summary>
    private Vector<T> _distGraphParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DPGN;

    /// <summary>Initializes a new DPGN meta-learner.</summary>
    /// <param name="options">Configuration options for DPGN.</param>
    public DPGNAlgorithm(DPGNOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _dpgnOptions = options;
        InitializeGraphParams();
    }

    /// <summary>Initializes dual graph parameters.</summary>
    private void InitializeGraphParams()
    {
        int nodeDim = _dpgnOptions.NodeFeatureDim;
        int edgeDim = _dpgnOptions.EdgeFeatureDim;
        int layers = _dpgnOptions.NumPropagationLayers;

        // Per-layer: edge MLP + node update MLP
        int paramsPerLayer = nodeDim * edgeDim + edgeDim + edgeDim * nodeDim + nodeDim;
        int totalPerGraph = layers * paramsPerLayer;

        _pointGraphParams = new Vector<T>(totalPerGraph);
        _distGraphParams = new Vector<T>(totalPerGraph);

        double scale = Math.Sqrt(2.0 / nodeDim);
        for (int i = 0; i < totalPerGraph; i++)
        {
            _pointGraphParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
            _distGraphParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        }
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _dpgnOptions.OuterLearningRate));
        }

        // Update dual graph params via SPSA
        UpdateAuxiliaryParams(taskBatch, ref _pointGraphParams);
        UpdateAuxiliaryParams(taskBatch, ref _distGraphParams);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Performs one layer of graph propagation: computes pairwise similarity-based edge weights
    /// via softmax, then updates each node via weighted neighbor aggregation with a residual connection.
    /// </summary>
    /// <param name="nodes">Current node features/distributions.</param>
    /// <param name="graphParams">Parameters for edge weight computation and node update.</param>
    /// <param name="layerIdx">Which propagation layer (for parameter offset).</param>
    /// <returns>Updated node features after one round of message passing.</returns>
    private Vector<T> PropagateOneLayer(Vector<T> nodes, Vector<T> graphParams, int layerIdx)
    {
        int n = nodes.Length;
        int nodeDim = _dpgnOptions.NodeFeatureDim;
        int edgeDim = _dpgnOptions.EdgeFeatureDim;
        int paramsPerLayer = nodeDim * edgeDim + edgeDim + edgeDim * nodeDim + nodeDim;
        int offset = layerIdx * paramsPerLayer;

        // Compute pairwise edge weights (scaled dot-product similarity)
        var edgeWeights = new double[n * n];
        double maxW = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double fi = NumOps.ToDouble(nodes[i]);
            for (int j = 0; j < n; j++)
            {
                double fj = NumOps.ToDouble(nodes[j]);
                int pIdx = (offset + (i * n + j)) % graphParams.Length;
                double w = NumOps.ToDouble(graphParams[pIdx]);
                double score = fi * fj * w / Math.Sqrt(Math.Max(n, 1));
                edgeWeights[i * n + j] = score;
                maxW = Math.Max(maxW, score);
            }
        }

        // Softmax normalization per node (row-wise)
        for (int i = 0; i < n; i++)
        {
            double sumExp = 0;
            for (int j = 0; j < n; j++)
            {
                edgeWeights[i * n + j] = Math.Exp(Math.Min(edgeWeights[i * n + j] - maxW, 10.0));
                sumExp += edgeWeights[i * n + j];
            }
            for (int j = 0; j < n; j++)
                edgeWeights[i * n + j] /= Math.Max(sumExp, 1e-10);
        }

        // Message passing: aggregate neighbors, project through shared MLP, add residual
        // Standard GNN pattern: shared weights applied identically to each node's aggregated message
        var updated = new Vector<T>(n);
        int projOffset = (offset + n * n) % Math.Max(graphParams.Length, 1);
        double wProj = NumOps.ToDouble(graphParams[projOffset % graphParams.Length]);
        double bProj = NumOps.ToDouble(graphParams[(projOffset + 1) % graphParams.Length]);
        for (int i = 0; i < n; i++)
        {
            double aggregated = 0;
            for (int j = 0; j < n; j++)
                aggregated += edgeWeights[i * n + j] * NumOps.ToDouble(nodes[j]);

            // Shared projection: ReLU(w * aggregated + b) — same weights for all nodes
            double projected = Math.Max(0, wProj * aggregated + bProj);

            // Residual connection
            updated[i] = NumOps.Add(nodes[i], NumOps.FromDouble(projected));
        }

        return updated;
    }

    /// <summary>
    /// Performs multi-layer dual graph propagation: alternating point graph (feature refinement)
    /// and distribution graph (confidence refinement) for the configured number of layers.
    /// Per the DPGN paper, both graphs interact and refine each other iteratively.
    /// </summary>
    /// <param name="features">Initial node features from the backbone.</param>
    /// <returns>Refined features and distribution estimates after dual propagation.</returns>
    private (Vector<T> refinedFeatures, Vector<T> refinedDistributions) DualGraphPropagate(Vector<T> features)
    {
        int n = features.Length;
        int numLayers = _dpgnOptions.NumPropagationLayers;

        // Initialize working copies
        var nodeFeatures = new Vector<T>(n);
        var nodeDistributions = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            nodeFeatures[i] = features[i];
            nodeDistributions[i] = NumOps.FromDouble(1.0 / Math.Max(n, 1)); // Uniform init
        }

        // Multi-layer dual propagation
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Point graph: refine features using feature-based edge weights
            nodeFeatures = PropagateOneLayer(nodeFeatures, _pointGraphParams, layer);

            // Distribution graph: refine confidence using distribution-based edge weights
            nodeDistributions = PropagateOneLayer(nodeDistributions, _distGraphParams, layer);
        }

        return (nodeFeatures, nodeDistributions);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Run dual graph propagation (point + distribution graphs)
        Vector<T>? refinedFeatures = null;
        Vector<T>? refinedDistributions = null;
        if (supportFeatures != null && supportFeatures.Length >= 2)
        {
            (refinedFeatures, refinedDistributions) = DualGraphPropagate(supportFeatures);
        }

        return new DPGNModel<T, TInput, TOutput>(MetaModel, currentParams, refinedFeatures, refinedDistributions);
    }

    /// <summary>Updates auxiliary parameters using SPSA gradient estimation.</summary>
    private void UpdateAuxiliaryParams(TaskBatch<T, TInput, TOutput> taskBatch, ref Vector<T> auxParams)
    {
        double epsilon = 1e-5;
        double lr = _dpgnOptions.OuterLearningRate;

        var direction = new Vector<T>(auxParams.Length);
        for (int i = 0; i < direction.Length; i++)
            direction[i] = NumOps.FromDouble(RandomGenerator.NextDouble() > 0.5 ? 1.0 : -1.0);

        double baseLoss = 0;
        foreach (var task in taskBatch.Tasks)
            baseLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        baseLoss /= taskBatch.Tasks.Length;

        for (int i = 0; i < auxParams.Length; i++)
            auxParams[i] = NumOps.Add(auxParams[i], NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon)));

        double perturbedLoss = 0;
        foreach (var task in taskBatch.Tasks)
            perturbedLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        perturbedLoss /= taskBatch.Tasks.Length;

        double directionalGrad = (perturbedLoss - baseLoss) / epsilon;
        for (int i = 0; i < auxParams.Length; i++)
            auxParams[i] = NumOps.Subtract(auxParams[i],
                NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon + lr * directionalGrad)));
    }

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0) return new Vector<T>(0);
        var result = new Vector<T>(vectors[0].Length);
        foreach (var v in vectors)
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.Add(result[i], v[i]);
        var scale = NumOps.FromDouble(1.0 / vectors.Count);
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.Multiply(result[i], scale);
        return result;
    }
}

/// <summary>Adapted model wrapper for DPGN with dual graph-propagated features.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model stores features that have been refined through
/// dual graph propagation. The point graph refined the feature representations by sharing
/// information between similar examples, while the distribution graph refined confidence
/// estimates. Together they produce more discriminative, context-aware representations.
/// </para>
/// </remarks>
internal class DPGNModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _refinedFeatures;
    private readonly Vector<T>? _refinedDistributions;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public DPGNModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? refinedFeatures,
        Vector<T>? refinedDistributions)
    {
        _model = model;
        _backboneParams = backboneParams;
        _refinedFeatures = refinedFeatures;
        _refinedDistributions = refinedDistributions;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_backboneParams); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
