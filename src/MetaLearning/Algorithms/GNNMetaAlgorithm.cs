using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Graph Neural Network-based Meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// GNN-based meta-learning models tasks and examples as nodes in a graph,
/// with edges representing relationships between them. The graph neural network
/// learns to propagate information across the task structure to improve learning.
/// </para>
/// <para>
/// <b>Key Innovation:</b> Instead of treating tasks independently, GNN Meta-learning:
/// 1. Builds a graph where nodes represent tasks or examples
/// 2. Edges connect similar or related tasks
/// 3. Message passing propagates useful information between tasks
/// 4. The aggregated graph information guides adaptation
/// </para>
/// <para>
/// <b>For Beginners:</b> GNN Meta-learning is like studying with a study group:
/// </para>
/// <para>
/// - MAML: Each student learns alone but starts with good study habits
/// - GNN Meta: Students share notes and help each other learn faster
///
/// When learning a new subject (task), you can benefit from what others
/// who studied similar subjects (similar tasks) have learned. The graph
/// network learns which tasks are helpful for each other.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// - <b>Node Embeddings:</b> Each task gets a vector representation
/// - <b>Edge Weights:</b> Learned weights showing task relationships
/// - <b>Message Passing:</b> Information flows between related tasks
/// - <b>Graph Aggregation:</b> Combines all node information for prediction
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <code>
/// For each task batch:
///   1. Build task graph from current batch
///   2. Compute node embeddings from task data
///   3. Perform K rounds of message passing
///   4. Aggregate graph information for each task
///   5. Use graph context to guide adaptation
///   6. Compute meta-gradients and update all components
/// </code>
/// </para>
/// <para>
/// Reference: Garcia, V., &amp; Bruna, J. (2018). Few-shot learning with graph neural networks.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Few-Shot Learning with Graph Neural Networks",
    "https://arxiv.org/abs/1711.04043",
    Year = 2018,
    Authors = "Victor Garcia, Joan Bruna")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class GNNMetaAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly GNNMetaOptions<T, TInput, TOutput> _gnnOptions;

    // GNN parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _messagePassingWeights;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _aggregationWeights;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _edgeWeights;

    // Task graph state
    [Scratch]
    private Matrix<T>? _currentAdjacencyMatrix;
    [Scratch]
    private List<Vector<T>>? _currentNodeEmbeddings;

    /// <summary>
    /// Initializes a new instance of the GNNMetaAlgorithm class.
    /// </summary>
    /// <param name="options">GNN Meta configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 4, outputSize: 2);
    /// var myNeuralNetwork = new NeuralNetwork&lt;double&gt;(architecture);
    /// // Create GNN Meta with minimal configuration
    /// var options = new GNNMetaOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myNeuralNetwork);
    /// var gnnMeta = new GNNMetaAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    ///
    /// // Create GNN Meta with custom configuration
    /// var options2 = new GNNMetaOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myNeuralNetwork)
    /// {
    ///     NumMessagePassingLayers = 5,
    ///     NodeEmbeddingDimension = 256,
    ///     AggregationType = GNNAggregationType.Attention,
    ///     SimilarityMetric = TaskSimilarityMetric.GradientSimilarity
    /// };
    /// var gnnMeta2 = new GNNMetaAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options2);
    /// </code>
    /// </example>
    public GNNMetaAlgorithm(GNNMetaOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer,
            // The task-graph context modulates each gradient coordinate by a learned scale, which an adaptive inner
            // optimizer divides back out; the inner steps stay plain SGD so the context keeps its effect.
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _gnnOptions = options;

        // Initialize GNN weights
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        int hidDim = _gnnOptions.GNNHiddenDimension;

        // Message passing weights: for each layer, we need embeddings -> hidden -> embeddings
        int mpWeightsSize = _gnnOptions.NumMessagePassingLayers * (embDim * hidDim + hidDim * embDim);
        _messagePassingWeights = InitializeWeights(mpWeightsSize);

        // Aggregation weights: use a compact projection with fixed dimensions
        // This avoids O(embDim * numParams) memory which could be very large for big models
        // Project: embDim -> hidDim -> hidDim (fixed-size bottleneck for graph context)
        int aggWeightsSize = embDim * hidDim + hidDim * hidDim;
        _aggregationWeights = InitializeWeights(aggWeightsSize);

        // Edge weights: if learning edge weights
        if (_gnnOptions.LearnEdgeWeights)
        {
            int edgeWeightsSize = embDim * _gnnOptions.EdgeFeatureDimension;
            _edgeWeights = InitializeWeights(edgeWeightsSize);
        }
        else
        {
            _edgeWeights = new Vector<T>(1);
            _edgeWeights[0] = NumOps.One;
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.GNNMeta"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as GNN Meta-learning,
    /// which uses graph neural networks to model task relationships
    /// and propagate information between related tasks.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.GNNMeta;

    /// <summary>
    /// Performs one meta-training step using GNN-based task relationship modeling.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// GNN Meta-training leverages task relationships through graph structure:
    /// </para>
    /// <para>
    /// <b>Graph Construction:</b>
    /// 1. Each task in the batch becomes a node in the graph
    /// 2. Edges are created based on task similarity (configurable metric)
    /// 3. Edge weights are either fixed or learned during training
    /// </para>
    /// <para>
    /// <b>Message Passing:</b>
    /// 1. Node embeddings are initialized from task representations
    /// 2. For each layer, nodes aggregate information from neighbors
    /// 3. Updated embeddings capture multi-hop task relationships
    /// </para>
    /// <para>
    /// <b>Adaptation with Graph Context:</b>
    /// 1. Graph embeddings provide context for each task
    /// 2. Context guides the adaptation process
    /// 3. Similar tasks benefit from shared information
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During training, the GNN learns:
    /// - Which tasks are similar and should share information
    /// - How to combine information from related tasks
    /// - How to use this combined information for better adaptation
    ///
    /// It's like learning which study partners are most helpful and
    /// how to best combine their notes with your own understanding.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Step 1: Build task graph
        var (adjacencyMatrix, nodeEmbeddings) = BuildTaskGraph(taskBatch);
        _currentAdjacencyMatrix = adjacencyMatrix;
        _currentNodeEmbeddings = nodeEmbeddings;

        // Step 2: Perform message passing
        var updatedEmbeddings = MessagePassing(nodeEmbeddings, adjacencyMatrix);

        // Step 3: Aggregate graph information for each task
        var graphContexts = AggregateGraphInformation(updatedEmbeddings, adjacencyMatrix);

        // Step 4: Train each task with graph context
        Vector<T>? accumulatedMetaGradients = null;
        T totalLoss = NumOps.Zero;

        for (int taskIdx = 0; taskIdx < taskBatch.Tasks.Length; taskIdx++)
        {
            var task = taskBatch.Tasks[taskIdx];
            var graphContext = graphContexts[taskIdx];

            // Clone and adapt model with graph context
            var taskModel = CloneModel();
            var adaptedParams = AdaptWithGraphContext(taskModel, task, graphContext);
            InterfaceGuard.Parameterizable(taskModel).SetParameters(adaptedParams);

            // Evaluate on query set
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T taskLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);
            totalLoss = NumOps.Add(totalLoss, taskLoss);

            // Compute meta-gradients
            var metaGradients = ComputeGradients(taskModel, task.QueryInput, task.QueryOutput);

            if (accumulatedMetaGradients == null)
            {
                accumulatedMetaGradients = metaGradients;
            }
            else
            {
                accumulatedMetaGradients = Engine.Add(accumulatedMetaGradients, metaGradients);
            }
        }

        if (accumulatedMetaGradients == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average gradients
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        accumulatedMetaGradients = Engine.Divide(accumulatedMetaGradients, batchSizeT);

        // Clip gradients if configured
        if (_gnnOptions.GradientClipThreshold.HasValue && _gnnOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedMetaGradients = ClipGradients(accumulatedMetaGradients, _gnnOptions.GradientClipThreshold.Value);
        }

        // Update meta-model parameters
        var currentParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        var updatedParams = ApplyOuterGradients(currentParams, accumulatedMetaGradients, _gnnOptions.OuterLearningRate);
        InterfaceGuard.Parameterizable(MetaModel).SetParameters(updatedParams);

        // Update GNN weights using finite differences
        UpdateGNNWeights(taskBatch, totalLoss);

        return NumOps.Divide(totalLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using graph-informed adaptation.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been adapted to the given task.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// GNN adaptation uses learned graph structure for context:
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// 1. Compute task embedding from support set
    /// 2. If previous tasks exist, use learned relationships
    /// 3. Apply graph context to guide adaptation
    /// 4. Perform gradient-based fine-tuning
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When adapting to a new task, GNN Meta uses what
    /// it learned about task relationships. If the new task is similar to
    /// tasks it's seen before, it can leverage that knowledge.
    ///
    /// It's like starting a new subject - if you remember that it's similar
    /// to something you studied before, you can apply relevant techniques.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone the meta model
        var adaptedModel = CloneModel();

        // Compute task embedding
        var taskEmbedding = ComputeTaskEmbedding(task);

        // Use cached graph context if available, otherwise adapt normally
        Vector<T> graphContext;
        if (_currentNodeEmbeddings != null && _currentNodeEmbeddings.Count > 0)
        {
            // Find most similar cached task and use its context
            graphContext = FindMostSimilarContext(taskEmbedding);
        }
        else
        {
            // No graph context available, use task embedding directly
            graphContext = taskEmbedding;
        }

        // Adapt with graph context
        var adaptedParams = AdaptWithGraphContext(adaptedModel, task, graphContext);
        InterfaceGuard.Parameterizable(adaptedModel).SetParameters(adaptedParams);

        return adaptedModel;
    }

    /// <summary>
    /// Builds a task graph from the current batch of tasks.
    /// </summary>
    private (Matrix<T> adjacencyMatrix, List<Vector<T>> nodeEmbeddings) BuildTaskGraph(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        int numTasks = taskBatch.Tasks.Length;
        var adjacencyMatrix = new Matrix<T>(numTasks, numTasks);
        var nodeEmbeddings = new List<Vector<T>>();

        // Compute node embeddings for each task
        foreach (var task in taskBatch.Tasks)
        {
            var embedding = ComputeTaskEmbedding(task);
            nodeEmbeddings.Add(embedding);
        }

        // Compute adjacency matrix based on similarity: self-loops with weight 1, other edges kept when the
        // graph is fully connected or the similarity reaches the threshold (NaN never does)
        if (numTasks > 0)
        {
            var similarity = SimilarityMatrix(StackRows(nodeEmbeddings));
            var kept = _gnnOptions.UseFullyConnectedGraph
                ? Engine.TensorOnesLike(similarity)
                : Engine.TensorAdd(
                    Engine.TensorGreaterThan(similarity, NumOps.FromDouble(_gnnOptions.EdgeThreshold)),
                    Engine.TensorEquals(similarity, NumOps.FromDouble(_gnnOptions.EdgeThreshold)));
            var edges = Engine.TensorWhere(kept, similarity, Engine.TensorZerosLike(similarity));
            var eye = Engine.TensorEye<T>(numTasks);
            adjacencyMatrix = Engine.TensorWhere(eye, eye, edges).ToMatrix();
        }

        return (adjacencyMatrix, nodeEmbeddings);
    }

    /// <summary>
    /// Computes an embedding for a task from its support set.
    /// </summary>
    private Vector<T> ComputeTaskEmbedding(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var embedding = new Vector<T>(_gnnOptions.NodeEmbeddingDimension);

        // Clone model and adapt to get task-specific parameters
        var taskModel = CloneModel();
        var adaptedParams = InnerLoopAdaptation(taskModel, task);

        // Use adapted parameters as embedding (or a projection of them)
        int embeddingDim = _gnnOptions.NodeEmbeddingDimension;
        int numToUse = Math.Min(adaptedParams.Length, embeddingDim);
        var head = numToUse > 0 ? adaptedParams.GetSubVector(0, numToUse) : new Vector<T>(0);

        // Fill remaining dimensions
        if (numToUse < embeddingDim)
        {
            var fill = Engine.Fill(embeddingDim - numToUse, ComputeMeanPartial(head, numToUse));
            embedding = numToUse > 0 ? Vector<T>.Concatenate(head, fill) : fill;
        }
        else
        {
            embedding = head;
        }

        return embedding;
    }

    /// <summary>Stacks equal-length vectors as the rows of a [count, length] tensor.</summary>
    private Tensor<T> StackRows(List<Vector<T>> rows)
    {
        var tensors = new Tensor<T>[rows.Count];
        for (int i = 0; i < rows.Count; i++) tensors[i] = Tensor<T>.FromVector(rows[i]);
        return Engine.TensorStack(tensors, 0);
    }

    /// <summary>
    /// Pairwise task similarity of the rows under the configured metric: the learned metric is
    /// sigmoid(sum_k w_k a_k b_k) over the first min(D, |w|) dimensions, every other metric is cosine.
    /// </summary>
    private Tensor<T> SimilarityMatrix(Tensor<T> rows)
    {
        if (_gnnOptions.SimilarityMetric != TaskSimilarityMetric.Learned)
            return CosineMatrix(rows, rows);

        // Simple learned similarity: weighted dot product
        int count = rows._shape[0];
        int minLen = Math.Min(rows._shape[1], _edgeWeights.Length);
        if (minLen == 0) return Engine.TensorAddScalar(new Tensor<T>(new[] { count, count }), NumOps.FromDouble(0.5));

        var used = minLen == rows._shape[1] ? rows : Engine.TensorSlice(rows, new[] { 0, 0 }, new[] { count, minLen });
        var weights = Engine.TensorBroadcastTo(
            Tensor<T>.FromVector(_edgeWeights.GetSubVector(0, minLen), new[] { 1, minLen }), new[] { count, minLen });

        // Apply sigmoid to get 0-1 range
        return Engine.TensorSigmoid(Engine.TensorMatMulTransposed(Engine.TensorMultiply(used, weights), used));
    }

    /// <summary>
    /// Cosine similarity of every row of <paramref name="queries"/> with every row of <paramref name="keys"/>,
    /// matching VectorHelper.CosineSimilarity: 0 when sqrt(|q|^2 |k|^2) &lt;= 1e-10, otherwise the ratio clamped
    /// to [-1, 1] (NaN passes through).
    /// </summary>
    private Tensor<T> CosineMatrix(Tensor<T> queries, Tensor<T> keys)
    {
        var dots = Engine.TensorMatMulTransposed(queries, keys);
        var queryNorms = Engine.ReduceSum(Engine.TensorSquare(queries), new[] { 1 }, keepDims: false);
        var keyNorms = Engine.ReduceSum(Engine.TensorSquare(keys), new[] { 1 }, keepDims: false);
        var denom = Engine.TensorSqrt(Engine.TensorOuter(queryNorms, keyNorms));

        var ratio = Engine.TensorDivide(dots, denom);
        var one = Engine.TensorOnesLike(ratio);
        ratio = Engine.TensorWhere(Engine.TensorGreaterThan(ratio, NumOps.One), one, ratio);
        ratio = Engine.TensorWhere(Engine.TensorLessThan(ratio, NumOps.FromDouble(-1.0)), Engine.TensorNegate(one), ratio);
        return Engine.TensorWhere(Engine.TensorGreaterThan(denom, NumOps.FromDouble(1e-10)), ratio, Engine.TensorZerosLike(ratio));
    }

    /// <summary>
    /// Performs message passing on the task graph.
    /// </summary>
    private List<Vector<T>> MessagePassing(List<Vector<T>> nodeEmbeddings, Matrix<T> adjacencyMatrix)
    {
        var currentEmbeddings = new List<Vector<T>>(nodeEmbeddings);
        int numNodes = nodeEmbeddings.Count;

        for (int layer = 0; layer < _gnnOptions.NumMessagePassingLayers; layer++)
        {
            if (numNodes == 0) break;

            // Aggregate messages from neighbors
            var aggregated = AggregateNeighborMessages(StackRows(currentEmbeddings), adjacencyMatrix);

            // Transform and combine with self embedding
            var transformed = TransformEmbedding(aggregated, layer);

            // Residual connection if enabled
            if (_gnnOptions.UseResidualConnections)
            {
                transformed = Engine.TensorAdd(transformed, StackRows(currentEmbeddings));
            }

            // Apply activation
            var activated = Engine.TensorTanh(transformed);

            var newEmbeddings = new List<Vector<T>>(numNodes);
            for (int i = 0; i < numNodes; i++)
                newEmbeddings.Add(Engine.TensorSlice(activated, new[] { i, 0 }, new[] { 1, activated._shape[1] }).ToVector());

            currentEmbeddings = newEmbeddings;
        }

        return currentEmbeddings;
    }

    /// <summary>
    /// Every node's positively-weighted neighbour mean: rows of (A+ E), each divided by its total positive
    /// edge weight when that exceeds 1e-8, with A+ the adjacency with non-positive (and NaN) weights dropped.
    /// </summary>
    private Tensor<T> AggregateNeighborMessages(Tensor<T> embeddings, Matrix<T> adjacencyMatrix)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        int numNodes = embeddings._shape[0];
        var used = embeddings._shape[1] == embDim
            ? embeddings
            : Engine.TensorSlice(embeddings, new[] { 0, 0 }, new[] { numNodes, embDim });

        var positive = PositiveEdges(adjacencyMatrix);
        var aggregated = Engine.TensorMatMul(positive, used);

        // Normalize by total weight
        var totalWeight = Engine.TensorBroadcastTo(
            Engine.ReduceSum(positive, new[] { 1 }, keepDims: true), new[] { numNodes, embDim });
        return Engine.TensorWhere(
            Engine.TensorGreaterThan(totalWeight, NumOps.FromDouble(1e-8)),
            Engine.TensorDivide(aggregated, totalWeight),
            aggregated);
    }

    /// <summary>The adjacency with every weight that is not &gt; 0 (including NaN) replaced by 0.</summary>
    private Tensor<T> PositiveEdges(Matrix<T> adjacencyMatrix)
    {
        var adjacency = Tensor<T>.FromMatrix(adjacencyMatrix);
        return Engine.TensorWhere(Engine.TensorGreaterThan(adjacency, NumOps.Zero), adjacency, Engine.TensorZerosLike(adjacency));
    }

    /// <summary>
    /// The layer's two-layer transform of every row: tanh(x W1^T) W2^T, with W1 [hidDim, embDim] and
    /// W2 [embDim, hidDim] read from the layer's block of the message passing weights.
    /// </summary>
    private Tensor<T> TransformEmbedding(Tensor<T> embeddings, int layer)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        int hidDim = _gnnOptions.GNNHiddenDimension;

        // Two-layer transformation: embDim → hidDim → embDim
        int layerOffset = layer * (embDim * hidDim + hidDim * embDim);
        var first = Tensor<T>.FromVector(_messagePassingWeights.GetSubVector(layerOffset, hidDim * embDim), new[] { hidDim, embDim });
        var second = Tensor<T>.FromVector(
            _messagePassingWeights.GetSubVector(layerOffset + embDim * hidDim, embDim * hidDim), new[] { embDim, hidDim });

        // First layer: embDim → hidDim, with tanh nonlinearity
        var hidden = Engine.TensorTanh(Engine.TensorMatMulTransposed(embeddings, first));

        // Second layer: hidDim → embDim
        return Engine.TensorMatMulTransposed(hidden, second);
    }

    /// <summary>
    /// Aggregates graph information to provide context for each task.
    /// </summary>
    private List<Vector<T>> AggregateGraphInformation(List<Vector<T>> nodeEmbeddings, Matrix<T> adjacencyMatrix)
    {
        var graphContexts = new List<Vector<T>>();

        switch (_gnnOptions.AggregationType)
        {
            case GNNAggregationType.Mean:
                for (int nodeIdx = 0; nodeIdx < nodeEmbeddings.Count; nodeIdx++)
                {
                    graphContexts.Add(ComputeMeanContext(nodeEmbeddings, nodeIdx, adjacencyMatrix));
                }
                break;

            case GNNAggregationType.Attention:
                for (int nodeIdx = 0; nodeIdx < nodeEmbeddings.Count; nodeIdx++)
                {
                    graphContexts.Add(ComputeAttentionContext(nodeEmbeddings, nodeEmbeddings[nodeIdx]));
                }
                break;

            case GNNAggregationType.Sum:
            case GNNAggregationType.Max:
            case GNNAggregationType.Set2Set:
            default:
                // Default to mean aggregation
                for (int nodeIdx = 0; nodeIdx < nodeEmbeddings.Count; nodeIdx++)
                {
                    graphContexts.Add(ComputeMeanContext(nodeEmbeddings, nodeIdx, adjacencyMatrix));
                }
                break;
        }

        return graphContexts;
    }

    /// <summary>
    /// The edge-weighted sum of the node's positively-connected neighbours divided by their count, or the
    /// node's own embedding when it has none.
    /// </summary>
    private Vector<T> ComputeMeanContext(List<Vector<T>> embeddings, int nodeIndex, Matrix<T> adjacencyMatrix)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        int numNodes = embeddings.Count;

        // Aggregate embeddings of neighbors based on adjacency matrix
        var row = Engine.TensorSlice(PositiveEdges(adjacencyMatrix), new[] { nodeIndex, 0 }, new[] { 1, numNodes });
        var neighborCount = Engine.TensorSum(Engine.TensorGreaterThan(row, NumOps.Zero));

        // If no neighbors, use the node's own embedding
        if (!NumOps.GreaterThan(neighborCount, NumOps.Zero))
            return embeddings[nodeIndex].GetSubVector(0, embDim);

        var rows = StackRows(embeddings);
        var used = rows._shape[1] == embDim ? rows : Engine.TensorSlice(rows, new[] { 0, 0 }, new[] { numNodes, embDim });

        // Normalize by neighbor count
        return Engine.TensorDivideScalar(Engine.TensorMatMul(row, used), neighborCount).ToVector();
    }

    /// <summary>
    /// sum_i (cos(q, e_i) / max(sum_j cos(q, e_j), 1e-8)) e_i.
    /// </summary>
    private Vector<T> ComputeAttentionContext(List<Vector<T>> embeddings, Vector<T> query)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        if (embeddings.Count == 0) return new Vector<T>(embDim);

        // Compute attention scores
        var rows = StackRows(embeddings);
        var scores = CosineMatrix(Tensor<T>.FromVector(query, new[] { 1, query.Length }), rows);
        T totalScore = Engine.TensorSum(scores);

        // Normalize and apply (Math.Max keeps a NaN total)
        T floor = NumOps.FromDouble(1e-8);
        T total = NumOps.IsNaN(totalScore) || NumOps.GreaterThan(totalScore, floor) ? totalScore : floor;
        var used = rows._shape[1] == embDim ? rows : Engine.TensorSlice(rows, new[] { 0, 0 }, new[] { embeddings.Count, embDim });
        return Engine.TensorMatMul(Engine.TensorDivideScalar(scores, total), used).ToVector();
    }

    /// <summary>
    /// Adapts model using graph context.
    /// </summary>
    private Vector<T> AdaptWithGraphContext(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task, Vector<T> graphContext)
    {
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();

        // Apply graph context as a modulation to the learning process
        // This is a simplified version - could be more sophisticated
        for (int step = 0; step < _gnnOptions.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Modulate gradients with graph context
            var modulatedGradients = ModulateGradientsWithContext(gradients, graphContext);

            // Apply gradients
            parameters = ApplyInnerGradients(parameters, modulatedGradients, _gnnOptions.InnerLearningRate, step == 0);
            InterfaceGuard.Parameterizable(model).SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>g_i * (1 + 0.1 * tanh(context[i mod |context|])), a soft modulation.</summary>
    private Vector<T> ModulateGradientsWithContext(Vector<T> gradients, Vector<T> context)
    {
        var scale = Engine.Add(Engine.Fill(context.Length, NumOps.One), Engine.Multiply(Engine.Tanh(context), NumOps.FromDouble(0.1)));
        return ParameterModulation.Apply(Engine, gradients, scale);
    }

    /// <summary>
    /// The cached node embedding with the highest cosine similarity to the task (the first on ties, NaN never
    /// selected), or the task embedding itself when nothing is cached or every similarity is NaN.
    /// </summary>
    private Vector<T> FindMostSimilarContext(Vector<T> taskEmbedding)
    {
        if (_currentNodeEmbeddings == null || _currentNodeEmbeddings.Count == 0)
        {
            return taskEmbedding;
        }

        int count = _currentNodeEmbeddings.Count;
        var similarity = CosineMatrix(Tensor<T>.FromVector(taskEmbedding, new[] { 1, taskEmbedding.Length }), StackRows(_currentNodeEmbeddings));

        // NaN compares unequal to itself; cosine lies in [-1, 1], so -3 ranks below every real score
        var valid = Engine.TensorEquals(similarity, similarity);
        if (!NumOps.GreaterThan(Engine.TensorSum(valid), NumOps.Zero)) return taskEmbedding;
        var keyed = Engine.TensorWhere(valid, similarity, Engine.TensorAddScalar(Engine.TensorZerosLike(similarity), NumOps.FromDouble(-3.0)));

        // First maximum: among the entries equal to the max, the largest (count - index) wins
        var best = Engine.TensorEquals(keyed, Engine.TensorMaxValue(keyed));
        var reversedIndex = Engine.TensorSubtract(
            Engine.TensorAddScalar(Engine.TensorZerosLike(keyed), NumOps.FromDouble(count)),
            Engine.Reshape(Engine.TensorArange<T>(0, count, 1), new[] { 1, count }));
        int index = count - (int)Math.Round(NumOps.ToDouble(Engine.TensorMaxValue(Engine.TensorMultiply(best, reversedIndex))));
        return _currentNodeEmbeddings[index];
    }

    /// <summary>
    /// Performs inner loop adaptation on a task.
    /// </summary>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();

        for (int step = 0; step < _gnnOptions.AdaptationSteps; step++)
        {
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);
            parameters = ApplyInnerGradients(parameters, gradients, _gnnOptions.InnerLearningRate, step == 0);
            InterfaceGuard.Parameterizable(model).SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Updates GNN weights using finite differences with scaled gradient estimation.
    /// </summary>
    private void UpdateGNNWeights(TaskBatch<T, TInput, TOutput> taskBatch, T currentLoss)
    {
        double epsilon = 1e-5;
        T epsilonT = NumOps.FromDouble(epsilon);

        // Update message passing weights (sample a subset for efficiency)
        int sampleCount = Math.Min(_messagePassingWeights.Length, 50);
        // Scale factor for unbiased gradient estimation when subsampling
        double scaleFactor = (double)_messagePassingWeights.Length / sampleCount;
        T stepSize = NumOps.FromDouble(_gnnOptions.OuterLearningRate * scaleFactor);

        for (int i = 0; i < sampleCount; i++)
        {
            int idx = (i * _messagePassingWeights.Length / sampleCount) % _messagePassingWeights.Length;

            T original = _messagePassingWeights[idx];
            _messagePassingWeights[idx] = NumOps.Add(original, epsilonT);

            // Recompute loss with perturbed weight
            T newLoss = ComputeBatchLoss(taskBatch);
            T grad = NumOps.Divide(NumOps.Subtract(newLoss, currentLoss), epsilonT);

            // Apply scaled gradient update
            _messagePassingWeights[idx] = NumOps.Subtract(original, NumOps.Multiply(stepSize, grad));
        }
    }

    /// <summary>
    /// Computes loss on a batch for gradient estimation.
    /// </summary>
    private T ComputeBatchLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            var taskModel = CloneModel();
            var adaptedParams = InnerLoopAdaptation(taskModel, task);
            InterfaceGuard.Parameterizable(taskModel).SetParameters(adaptedParams);

            var predictions = taskModel.Predict(task.QueryInput);
            T taskLoss = ComputeLossFromOutput(predictions, task.QueryOutput);
            totalLoss = NumOps.Add(totalLoss, taskLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <summary>
    /// Initializes weights using Xavier initialization.
    /// </summary>
    private Vector<T> InitializeWeights(int size)
    {
        var weights = new Vector<T>(size);
        double scale = Math.Sqrt(2.0 / size);

        for (int i = 0; i < size; i++)
        {
            double value = (RandomGenerator.NextDouble() * 2 - 1) * scale;
            weights[i] = NumOps.FromDouble(value);
        }

        return weights;
    }

    /// <summary>
    /// Computes mean of first n elements.
    /// </summary>
    private T ComputeMeanPartial(Vector<T> vec, int n)
    {
        if (n <= 0) return NumOps.Zero;

        T sum = Engine.Sum(vec.Length == n ? vec : vec.GetSubVector(0, n));
        return NumOps.Divide(sum, NumOps.FromDouble(n));
    }
}
