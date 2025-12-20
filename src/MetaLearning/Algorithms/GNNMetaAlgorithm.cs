using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;

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
public class GNNMetaAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly GNNMetaOptions<T, TInput, TOutput> _gnnOptions;

    // GNN parameters
    private Vector<T> _messagePassingWeights;
    private Vector<T> _aggregationWeights;
    private Vector<T> _edgeWeights;

    // Task graph state
    private Matrix<T>? _currentAdjacencyMatrix;
    private List<Vector<T>>? _currentNodeEmbeddings;

    /// <summary>
    /// Initializes a new instance of the GNNMetaAlgorithm class.
    /// </summary>
    /// <param name="options">GNN Meta configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create GNN Meta with minimal configuration
    /// var options = new GNNMetaOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var gnnMeta = new GNNMetaAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create GNN Meta with custom configuration
    /// var options = new GNNMetaOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     NumMessagePassingLayers = 5,
    ///     NodeEmbeddingDimension = 256,
    ///     AggregationType = GNNAggregationType.Attention,
    ///     SimilarityMetric = TaskSimilarityMetric.GradientSimilarity
    /// };
    /// var gnnMeta = new GNNMetaAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public GNNMetaAlgorithm(GNNMetaOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
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
            taskModel.SetParameters(adaptedParams);

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
                for (int i = 0; i < accumulatedMetaGradients.Length; i++)
                {
                    accumulatedMetaGradients[i] = NumOps.Add(accumulatedMetaGradients[i], metaGradients[i]);
                }
            }
        }

        if (accumulatedMetaGradients == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average gradients
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < accumulatedMetaGradients.Length; i++)
        {
            accumulatedMetaGradients[i] = NumOps.Divide(accumulatedMetaGradients[i], batchSizeT);
        }

        // Clip gradients if configured
        if (_gnnOptions.GradientClipThreshold.HasValue && _gnnOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedMetaGradients = ClipGradients(accumulatedMetaGradients, _gnnOptions.GradientClipThreshold.Value);
        }

        // Update meta-model parameters
        var currentParams = MetaModel.GetParameters();
        var updatedParams = ApplyGradients(currentParams, accumulatedMetaGradients, _gnnOptions.OuterLearningRate);
        MetaModel.SetParameters(updatedParams);

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
        adaptedModel.SetParameters(adaptedParams);

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

        // Compute adjacency matrix based on similarity
        for (int i = 0; i < numTasks; i++)
        {
            for (int j = 0; j < numTasks; j++)
            {
                if (i == j)
                {
                    // Self-loops with weight 1
                    adjacencyMatrix[i, j] = NumOps.One;
                }
                else
                {
                    // Compute similarity
                    T similarity = ComputeTaskSimilarity(nodeEmbeddings[i], nodeEmbeddings[j]);
                    double simValue = NumOps.ToDouble(similarity);

                    if (_gnnOptions.UseFullyConnectedGraph || simValue >= _gnnOptions.EdgeThreshold)
                    {
                        adjacencyMatrix[i, j] = similarity;
                    }
                    else
                    {
                        adjacencyMatrix[i, j] = NumOps.Zero;
                    }
                }
            }
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
        int numToUse = Math.Min(adaptedParams.Length, _gnnOptions.NodeEmbeddingDimension);
        for (int i = 0; i < numToUse; i++)
        {
            embedding[i] = adaptedParams[i % adaptedParams.Length];
        }

        // Fill remaining dimensions
        if (numToUse < _gnnOptions.NodeEmbeddingDimension)
        {
            T mean = ComputeMeanPartial(embedding, numToUse);
            for (int i = numToUse; i < _gnnOptions.NodeEmbeddingDimension; i++)
            {
                embedding[i] = mean;
            }
        }

        return embedding;
    }

    /// <summary>
    /// Computes similarity between two task embeddings.
    /// </summary>
    private T ComputeTaskSimilarity(Vector<T> embedding1, Vector<T> embedding2)
    {
        switch (_gnnOptions.SimilarityMetric)
        {
            case TaskSimilarityMetric.ParameterDistance:
                return ComputeCosineSimilarity(embedding1, embedding2);

            case TaskSimilarityMetric.GradientSimilarity:
                return ComputeCosineSimilarity(embedding1, embedding2);

            case TaskSimilarityMetric.DataDistribution:
                return ComputeCosineSimilarity(embedding1, embedding2);

            case TaskSimilarityMetric.Learned:
                return ComputeLearnedSimilarity(embedding1, embedding2);

            default:
                return ComputeCosineSimilarity(embedding1, embedding2);
        }
    }

    /// <summary>
    /// Computes cosine similarity between two vectors.
    /// </summary>
    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dotProduct = NumOps.Zero;
        T normA = NumOps.Zero;
        T normB = NumOps.Zero;

        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
            normA = NumOps.Add(normA, NumOps.Multiply(a[i], a[i]));
            normB = NumOps.Add(normB, NumOps.Multiply(b[i], b[i]));
        }

        double normAVal = Math.Sqrt(Math.Max(NumOps.ToDouble(normA), 1e-8));
        double normBVal = Math.Sqrt(Math.Max(NumOps.ToDouble(normB), 1e-8));
        double similarity = NumOps.ToDouble(dotProduct) / (normAVal * normBVal);

        // Convert to 0-1 range
        similarity = (similarity + 1.0) / 2.0;
        return NumOps.FromDouble(similarity);
    }

    /// <summary>
    /// Computes learned similarity using edge weights.
    /// </summary>
    private T ComputeLearnedSimilarity(Vector<T> a, Vector<T> b)
    {
        // Simple learned similarity: weighted dot product
        T similarity = NumOps.Zero;
        int minLen = Math.Min(Math.Min(a.Length, b.Length), _edgeWeights.Length);

        for (int i = 0; i < minLen; i++)
        {
            T product = NumOps.Multiply(a[i], b[i]);
            similarity = NumOps.Add(similarity, NumOps.Multiply(_edgeWeights[i], product));
        }

        // Apply sigmoid to get 0-1 range
        double val = NumOps.ToDouble(similarity);
        double sigmoid = 1.0 / (1.0 + Math.Exp(-val));
        return NumOps.FromDouble(sigmoid);
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
            var newEmbeddings = new List<Vector<T>>();

            for (int i = 0; i < numNodes; i++)
            {
                // Aggregate messages from neighbors
                var aggregatedMessage = AggregateNeighborMessages(i, currentEmbeddings, adjacencyMatrix);

                // Transform and combine with self embedding
                var transformed = TransformEmbedding(aggregatedMessage, layer);

                // Residual connection if enabled
                if (_gnnOptions.UseResidualConnections)
                {
                    for (int j = 0; j < transformed.Length; j++)
                    {
                        transformed[j] = NumOps.Add(transformed[j], currentEmbeddings[i][j]);
                    }
                }

                // Apply activation
                for (int j = 0; j < transformed.Length; j++)
                {
                    double val = NumOps.ToDouble(transformed[j]);
                    transformed[j] = NumOps.FromDouble(Math.Tanh(val));
                }

                newEmbeddings.Add(transformed);
            }

            currentEmbeddings = newEmbeddings;
        }

        return currentEmbeddings;
    }

    /// <summary>
    /// Aggregates messages from neighboring nodes.
    /// </summary>
    private Vector<T> AggregateNeighborMessages(int nodeIdx, List<Vector<T>> embeddings, Matrix<T> adjacencyMatrix)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        var aggregated = new Vector<T>(embDim);
        T totalWeight = NumOps.Zero;

        for (int j = 0; j < embeddings.Count; j++)
        {
            T edgeWeight = adjacencyMatrix[nodeIdx, j];
            double weightVal = NumOps.ToDouble(edgeWeight);

            if (weightVal > 0)
            {
                for (int k = 0; k < embDim; k++)
                {
                    T weighted = NumOps.Multiply(edgeWeight, embeddings[j][k]);
                    aggregated[k] = NumOps.Add(aggregated[k], weighted);
                }
                totalWeight = NumOps.Add(totalWeight, edgeWeight);
            }
        }

        // Normalize by total weight
        double totalWeightVal = NumOps.ToDouble(totalWeight);
        if (totalWeightVal > 1e-8)
        {
            for (int k = 0; k < embDim; k++)
            {
                aggregated[k] = NumOps.Divide(aggregated[k], totalWeight);
            }
        }

        return aggregated;
    }

    /// <summary>
    /// Transforms embedding using message passing weights with a two-layer projection.
    /// Uses the allocated embDim → hidDim → embDim weight structure.
    /// </summary>
    private Vector<T> TransformEmbedding(Vector<T> embedding, int layer)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        int hidDim = _gnnOptions.GNNHiddenDimension;
        var hidden = new Vector<T>(hidDim);
        var output = new Vector<T>(embDim);

        // Two-layer transformation: embDim → hidDim → embDim
        int layerOffset = layer * (embDim * hidDim + hidDim * embDim);

        // First layer: embDim → hidDim
        for (int i = 0; i < hidDim; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < Math.Min(embDim, embedding.Length); j++)
            {
                int weightIdx = layerOffset + i * embDim + j;
                if (weightIdx < _messagePassingWeights.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_messagePassingWeights[weightIdx], embedding[j]));
                }
            }
            // Apply tanh nonlinearity
            hidden[i] = NumOps.FromDouble(Math.Tanh(NumOps.ToDouble(sum)));
        }

        // Second layer: hidDim → embDim
        int secondLayerOffset = layerOffset + embDim * hidDim;
        for (int i = 0; i < embDim; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < hidDim; j++)
            {
                int weightIdx = secondLayerOffset + i * hidDim + j;
                if (weightIdx < _messagePassingWeights.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_messagePassingWeights[weightIdx], hidden[j]));
                }
            }
            output[i] = sum;
        }

        return output;
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
    /// Computes mean context from neighbor embeddings using the adjacency matrix.
    /// </summary>
    private Vector<T> ComputeMeanContext(List<Vector<T>> embeddings, int nodeIndex, Matrix<T> adjacencyMatrix)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        var context = new Vector<T>(embDim);
        int neighborCount = 0;

        // Aggregate embeddings of neighbors based on adjacency matrix
        for (int j = 0; j < embeddings.Count; j++)
        {
            // Check if there's an edge from nodeIndex to j (non-zero adjacency)
            T edgeWeight = adjacencyMatrix[nodeIndex, j];
            if (NumOps.ToDouble(edgeWeight) > 0)
            {
                for (int i = 0; i < embDim; i++)
                {
                    // Weight the embedding by the edge weight
                    context[i] = NumOps.Add(context[i], NumOps.Multiply(edgeWeight, embeddings[j][i]));
                }
                neighborCount++;
            }
        }

        // Normalize by neighbor count (if any neighbors exist)
        if (neighborCount > 0)
        {
            T count = NumOps.FromDouble(neighborCount);
            for (int i = 0; i < embDim; i++)
            {
                context[i] = NumOps.Divide(context[i], count);
            }
        }
        else
        {
            // If no neighbors, use the node's own embedding
            var selfEmbedding = embeddings[nodeIndex];
            for (int i = 0; i < embDim; i++)
            {
                context[i] = selfEmbedding[i];
            }
        }

        return context;
    }

    /// <summary>
    /// Computes attention-weighted context from node embeddings.
    /// </summary>
    private Vector<T> ComputeAttentionContext(List<Vector<T>> embeddings, Vector<T> query)
    {
        int embDim = _gnnOptions.NodeEmbeddingDimension;
        var context = new Vector<T>(embDim);
        var attentionWeights = new T[embeddings.Count];

        // Compute attention scores
        T totalScore = NumOps.Zero;
        for (int i = 0; i < embeddings.Count; i++)
        {
            T score = ComputeCosineSimilarity(query, embeddings[i]);
            attentionWeights[i] = score;
            totalScore = NumOps.Add(totalScore, score);
        }

        // Normalize and apply
        double totalVal = Math.Max(NumOps.ToDouble(totalScore), 1e-8);
        for (int i = 0; i < embeddings.Count; i++)
        {
            T weight = NumOps.Divide(attentionWeights[i], NumOps.FromDouble(totalVal));
            for (int j = 0; j < embDim; j++)
            {
                T contribution = NumOps.Multiply(weight, embeddings[i][j]);
                context[j] = NumOps.Add(context[j], contribution);
            }
        }

        return context;
    }

    /// <summary>
    /// Adapts model using graph context.
    /// </summary>
    private Vector<T> AdaptWithGraphContext(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task, Vector<T> graphContext)
    {
        var parameters = model.GetParameters();

        // Apply graph context as a modulation to the learning process
        // This is a simplified version - could be more sophisticated
        for (int step = 0; step < _gnnOptions.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Modulate gradients with graph context
            var modulatedGradients = ModulateGradientsWithContext(gradients, graphContext);

            // Apply gradients
            parameters = ApplyGradients(parameters, modulatedGradients, _gnnOptions.InnerLearningRate);
            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Modulates gradients using graph context.
    /// </summary>
    private Vector<T> ModulateGradientsWithContext(Vector<T> gradients, Vector<T> context)
    {
        var modulated = new Vector<T>(gradients.Length);

        for (int i = 0; i < gradients.Length; i++)
        {
            // Use context to scale gradients
            int contextIdx = i % context.Length;
            double contextVal = NumOps.ToDouble(context[contextIdx]);
            double scale = 1.0 + 0.1 * Math.Tanh(contextVal);  // Soft modulation

            modulated[i] = NumOps.Multiply(NumOps.FromDouble(scale), gradients[i]);
        }

        return modulated;
    }

    /// <summary>
    /// Finds the most similar cached context for a new task.
    /// </summary>
    private Vector<T> FindMostSimilarContext(Vector<T> taskEmbedding)
    {
        if (_currentNodeEmbeddings == null || _currentNodeEmbeddings.Count == 0)
        {
            return taskEmbedding;
        }

        Vector<T>? bestMatch = null;
        double bestSimilarity = double.MinValue;

        foreach (var cachedEmbedding in _currentNodeEmbeddings)
        {
            T similarity = ComputeCosineSimilarity(taskEmbedding, cachedEmbedding);
            double simVal = NumOps.ToDouble(similarity);

            if (simVal > bestSimilarity)
            {
                bestSimilarity = simVal;
                bestMatch = cachedEmbedding;
            }
        }

        return bestMatch ?? taskEmbedding;
    }

    /// <summary>
    /// Performs inner loop adaptation on a task.
    /// </summary>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        for (int step = 0; step < _gnnOptions.AdaptationSteps; step++)
        {
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);
            parameters = ApplyGradients(parameters, gradients, _gnnOptions.InnerLearningRate);
            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Updates GNN weights using finite differences with scaled gradient estimation.
    /// </summary>
    private void UpdateGNNWeights(TaskBatch<T, TInput, TOutput> taskBatch, T currentLoss)
    {
        double epsilon = 1e-5;
        double currentLossVal = NumOps.ToDouble(currentLoss);

        // Update message passing weights (sample a subset for efficiency)
        int sampleCount = Math.Min(_messagePassingWeights.Length, 50);
        // Scale factor for unbiased gradient estimation when subsampling
        double scaleFactor = (double)_messagePassingWeights.Length / sampleCount;

        for (int i = 0; i < sampleCount; i++)
        {
            int idx = (i * _messagePassingWeights.Length / sampleCount) % _messagePassingWeights.Length;

            T original = _messagePassingWeights[idx];
            _messagePassingWeights[idx] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            // Recompute loss with perturbed weight
            T newLoss = ComputeBatchLoss(taskBatch);
            double grad = (NumOps.ToDouble(newLoss) - currentLossVal) / epsilon;

            // Apply scaled gradient update
            _messagePassingWeights[idx] = NumOps.Subtract(original,
                NumOps.FromDouble(_gnnOptions.OuterLearningRate * grad * scaleFactor));
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
            taskModel.SetParameters(adaptedParams);

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

        T sum = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            sum = NumOps.Add(sum, vec[i]);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(n));
    }
}
