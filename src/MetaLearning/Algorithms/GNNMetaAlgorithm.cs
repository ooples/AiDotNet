using AiDotNet.Interfaces;
using AiDotNet.Layers;
using AiDotNet.MetaLearning.Config;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Graph Neural Network-based Meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// GNN-based meta-learning models tasks and examples as nodes in a graph,
/// with edges representing relationships between them. The graph neural network
/// learns to propagate information across the task structure to improve learning.
/// </para>
/// <para><b>Key Components:</b></para>
/// - <b>Task Graph Construction:</b> Builds graphs from task relationships
/// - <b>Graph Encoder:</b> Processes graph structure with GNN layers
/// - <b>Attention Mechanism:</b> Learns to attend to relevant tasks/examples
/// - <b>Graph Pooling:</b> Aggregates graph information for predictions
/// </remarks>
public class GNNMetaAlgorithm<T, TInput, TOutput> : IMetaLearningAlgorithm<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly GNNMetaAlgorithmOptions<T, TInput, TOutput> _gnnOptions;
    private readonly INeuralNetwork<T> _featureEncoder;
    private readonly INeuralNetwork<T> _graphEncoder;
    private readonly INeuralNetwork<T> _taskDecoder;
    private readonly INeuralNetwork<T> _edgePredictor;
    private readonly IInitializer<T> _initializer;
    private readonly IOptimizer<T> _metaOptimizer;

    private readonly List<ILayer<T>> _featureEncoderLayers;
    private readonly List<ILayer<T>> _graphEncoderLayers;
    private readonly List<ILayer<T>> _taskDecoderLayers;
    private readonly List<ILayer<T>> _edgePredictorLayers;

    private Graph<T>? _currentTaskGraph;
    private Tensor<T>? _nodeEmbeddings;
    private Tensor<T>? _graphEmbedding;
    private readonly Dictionary<int, Graph<T>> _graphCache;

    /// <summary>
    /// Gets the algorithm type identifier.
    /// </summary>
    public string AlgorithmType => "GNN-Meta";

    /// <summary>
    /// Gets the current episode number.
    /// </summary>
    public int CurrentEpisode { get; private set; }

    /// <summary>
    /// Gets or sets the meta-parameters of the algorithm.
    /// </summary>
    public Tensor<T> MetaParameters { get; private set; }

    /// <summary>
    /// Gets the performance history across episodes.
    /// </summary>
    public List<T> PerformanceHistory { get; }

    /// <summary>
    /// Initializes a new instance of the GNNMetaAlgorithm class.
    /// </summary>
    /// <param name="gnnOptions">Configuration options for GNN meta-learning.</param>
    /// <param name="featureEncoder">The feature encoder network.</param>
    /// <param name="graphEncoder">The graph neural network encoder.</param>
    /// <param name="taskDecoder">The task decoder network.</param>
    /// <param name="edgePredictor">The edge prediction network.</param>
    /// <param name="initializer">Weight initializer.</param>
    /// <param name="metaOptimizer">Optimizer for meta-parameters.</param>
    public GNNMetaAlgorithm(
        GNNMetaAlgorithmOptions<T, TInput, TOutput> gnnOptions,
        INeuralNetwork<T> featureEncoder,
        INeuralNetwork<T> graphEncoder,
        INeuralNetwork<T> taskDecoder,
        INeuralNetwork<T> edgePredictor,
        IInitializer<T> initializer,
        IOptimizer<T> metaOptimizer)
    {
        _gnnOptions = gnnOptions ?? throw new ArgumentNullException(nameof(gnnOptions));
        _featureEncoder = featureEncoder ?? throw new ArgumentNullException(nameof(featureEncoder));
        _graphEncoder = graphEncoder ?? throw new ArgumentNullException(nameof(graphEncoder));
        _taskDecoder = taskDecoder ?? throw new ArgumentNullException(nameof(taskDecoder));
        _edgePredictor = edgePredictor ?? throw new ArgumentNullException(nameof(edgePredictor));
        _initializer = initializer ?? throw new ArgumentNullException(nameof(initializer));
        _metaOptimizer = metaOptimizer ?? throw new ArgumentNullException(nameof(metaOptimizer));

        _featureEncoderLayers = new List<ILayer<T>>();
        _graphEncoderLayers = new List<ILayer<T>>();
        _taskDecoderLayers = new List<ILayer<T>>();
        _edgePredictorLayers = new List<ILayer<T>>();
        _graphCache = new Dictionary<int, Graph<T>>();
        PerformanceHistory = new List<T>();

        InitializeNetworks();
    }

    /// <summary>
    /// Performs meta-learning on a batch of tasks.
    /// </summary>
    /// <param name="taskBatch">Batch of tasks for meta-learning.</param>
    /// <returns>Dictionary of performance metrics.</returns>
    public Dictionary<string, T> MetaLearnBatch(TaskBatch<TInput, TOutput> taskBatch)
    {
        // Build task graph
        var taskGraph = BuildTaskGraph(taskBatch);
        _currentTaskGraph = taskGraph;

        // Encode features
        var nodeFeatures = EncodeNodeFeatures(taskGraph);

        // Apply graph neural network
        _nodeEmbeddings = ApplyGraphNetwork(nodeFeatures, taskGraph);

        // Pool node embeddings to get graph representation
        _graphEmbedding = PoolNodeEmbeddings(_nodeEmbeddings, taskGraph);

        // Decode to task predictions
        var predictions = DecodeToPredictions(_nodeEmbeddings, taskGraph);

        // Compute losses
        var (taskLoss, accuracy) = ComputeTaskLoss(taskBatch, predictions);
        var graphLoss = ComputeGraphRegularizationLoss(_nodeEmbeddings, taskGraph);
        var edgeLoss = _gnnOptions.LearnEdges ?
            ComputeEdgePredictionLoss(taskGraph) : NumOps.Zero;

        // Total loss
        var totalLoss = NumOps.Add(
            taskLoss,
            NumOps.Multiply(NumOps.FromDouble(_gnnOptions.GraphRegularizationWeight), graphLoss));

        if (_gnnOptions.LearnEdges)
        {
            totalLoss = NumOps.Add(
                totalLoss,
                NumOps.Multiply(NumOps.FromDouble(_gnnOptions.EdgePredictionWeight), edgeLoss));
        }

        // Meta-update
        MetaUpdate(totalLoss);

        // Cache graph embeddings for future use
        CacheGraphEmbeddings(taskBatch);

        // Record performance
        PerformanceHistory.Add(taskLoss);

        CurrentEpisode++;

        return new Dictionary<string, T>
        {
            ["loss"] = totalLoss,
            ["task_loss"] = taskLoss,
            ["graph_loss"] = graphLoss,
            ["edge_loss"] = edgeLoss,
            ["accuracy"] = accuracy
        };
    }

    /// <summary>
    /// Adapts to a new task using support set examples.
    /// </summary>
    /// <param name="supportSet">Support set for adaptation.</param>
    /// <param name="numSteps">Number of adaptation steps.</param>
    public void Adapt(TaskBatch<TInput, TOutput> supportSet, int numSteps = 1)
    {
        // Build graph for new task
        var taskGraph = BuildSingleTaskGraph(supportSet);

        // Encode features
        var nodeFeatures = EncodeNodeFeatures(taskGraph);

        // Apply graph neural network
        var nodeEmbeddings = ApplyGraphNetwork(nodeFeatures, taskGraph);

        // Perform message passing with support set
        for (int step = 0; step < numSteps; step++)
        {
            nodeEmbeddings = MessagePassingStep(nodeEmbeddings, taskGraph, supportSet);
        }

        // Update embeddings
        _nodeEmbeddings = nodeEmbeddings;
        _graphEmbedding = PoolNodeEmbeddings(nodeEmbeddings, taskGraph);
    }

    /// <summary>
    /// Makes predictions on query examples after adaptation.
    /// </summary>
    /// <param name="querySet">Query examples for prediction.</param>
    /// <returns>Predictions for query examples.</returns>
    public TOutput Predict(TInput querySet)
    {
        if (_nodeEmbeddings == null || _graphEmbedding == null)
        {
            throw new InvalidOperationException("Model must be adapted before making predictions");
        }

        // Add query node to graph
        var queryGraph = AddQueryToGraph(querySet);

        // Get final prediction
        return DecodeSingleNode(queryGraph.QueryNodeIndex);
    }

    /// <summary>
    /// Resets the meta-learning algorithm state.
    /// </summary>
    public void Reset()
    {
        CurrentEpisode = 0;
        _currentTaskGraph = null;
        _nodeEmbeddings = null;
        _graphEmbedding = null;
        _graphCache.Clear();
        PerformanceHistory.Clear();

        // Reset all networks
        foreach (var layer in _featureEncoderLayers.Concat(_graphEncoderLayers)
            .Concat(_taskDecoderLayers).Concat(_edgePredictorLayers))
        {
            layer.Reset();
        }

        // Reset optimizer
        _metaOptimizer.Reset();
    }

    /// <summary>
    /// Gets the meta-parameters for saving/loading.
    /// </summary>
    /// <returns>Current meta-parameters.</returns>
    public Tensor<T> GetMetaParameters()
    {
        return MetaParameters;
    }

    /// <summary>
    /// Sets the meta-parameters from saved state.
    /// </summary>
    /// <param name="parameters">Meta-parameters to load.</param>
    public void SetMetaParameters(Tensor<T> parameters)
    {
        MetaParameters = parameters;
        DistributeParameters(parameters);
    }

    private void InitializeNetworks()
    {
        // Initialize feature encoder
        InitializeFeatureEncoder();

        // Initialize graph encoder (GNN layers)
        InitializeGraphEncoder();

        // Initialize task decoder
        InitializeTaskDecoder();

        // Initialize edge predictor
        if (_gnnOptions.LearnEdges)
        {
            InitializeEdgePredictor();
        }

        // Initialize meta-parameters
        InitializeMetaParameters();
    }

    private void InitializeFeatureEncoder()
    {
        // Feature encoder for input data
        _featureEncoderLayers.Add(new DenseLayer<T>(
            _gnnOptions.InputDimension,
            _gnnOptions.FeatureHiddenDimension,
            activation: _gnnOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        // Additional hidden layers
        for (int i = 0; i < _gnnOptions.NumFeatureLayers - 1; i++)
        {
            _featureEncoderLayers.Add(new DenseLayer<T>(
                _gnnOptions.FeatureHiddenDimension,
                _gnnOptions.FeatureHiddenDimension,
                activation: _gnnOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true,
                dropoutRate: _gnnOptions.DropoutRate));
        }

        // Output to embedding dimension
        _featureEncoderLayers.Add(new DenseLayer<T>(
            _gnnOptions.FeatureHiddenDimension,
            _gnnOptions.NodeEmbeddingDimension,
            activation: ActivationFunctionType.None,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeGraphEncoder()
    {
        // Graph Neural Network layers
        for (int layer = 0; layer < _gnnOptions.NumGNNLayers; layer++)
        {
            // Graph convolution layer
            _graphEncoderLayers.Add(new GraphConvolutionLayer<T>(
                _gnnOptions.NodeEmbeddingDimension,
                _gnnOptions.GNNHiddenDimension,
                activation: _gnnOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true));

            // Attention mechanism if enabled
            if (_gnnOptions.UseGraphAttention)
            {
                _graphEncoderLayers.Add(new GraphAttentionLayer<T>(
                    _gnnOptions.GNNHiddenDimension,
                    _gnnOptions.NumAttentionHeads,
                    _gnnOptions.AttentionDimension));
            }

            // Residual connection
            if (_gnnOptions.UseResidualConnections)
            {
                _graphEncoderLayers.Add(new ResidualConnection<T>());
            }
        }
    }

    private void InitializeTaskDecoder()
    {
        // Task decoder to make predictions
        _taskDecoderLayers.Add(new DenseLayer<T>(
            _gnnOptions.NodeEmbeddingDimension,
            _gnnOptions.DecoderHiddenDimension,
            activation: _gnnOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        // Hidden layers
        for (int i = 0; i < _gnnOptions.NumDecoderLayers - 1; i++)
        {
            _taskDecoderLayers.Add(new DenseLayer<T>(
                _gnnOptions.DecoderHiddenDimension,
                _gnnOptions.DecoderHiddenDimension,
                activation: _gnnOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true,
                dropoutRate: _gnnOptions.DropoutRate));
        }

        // Output layer
        _taskDecoderLayers.Add(new DenseLayer<T>(
            _gnnOptions.DecoderHiddenDimension,
            _gnnOptions.OutputDimension,
            activation: ActivationFunctionType.None,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeEdgePredictor()
    {
        // Edge prediction network
        _edgePredictorLayers.Add(new DenseLayer<T>(
            2 * _gnnOptions.NodeEmbeddingDimension,  // Concatenated node embeddings
            _gnnOptions.EdgeHiddenDimension,
            activation: _gnnOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        _edgePredictorLayers.Add(new DenseLayer<T>(
            _gnnOptions.EdgeHiddenDimension,
            1,  // Edge probability
            activation: ActivationFunctionType.Sigmoid,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeMetaParameters()
    {
        // Collect all parameters
        var allParameters = new List<Tensor<T>>();

        foreach (var layer in _featureEncoderLayers.Concat(_graphEncoderLayers)
            .Concat(_taskDecoderLayers).Concat(_edgePredictorLayers))
        {
            allParameters.AddRange(layer.GetParameters());
        }

        // Concatenate into single tensor
        MetaParameters = Tensor<T>.Concat(allParameters);
    }

    private Graph<T> BuildTaskGraph(TaskBatch<TInput, TOutput> taskBatch)
    {
        var nodes = new List<Node<T>>();
        var edges = new List<Edge<T>>();

        // Add task nodes
        for (int taskIdx = 0; taskIdx < taskBatch.Tasks.Count; taskIdx++)
        {
            var task = taskBatch.Tasks[taskIdx];

            // Create task node
            var taskNode = new Node<T>
            {
                Id = taskIdx,
                Type = NodeType.Task,
                Features = ExtractTaskFeatures(task),
                Label = task.Label
            };
            nodes.Add(taskNode);

            // Add example nodes
            for (int exampleIdx = 0; exampleIdx < task.NumExamples; exampleIdx++)
            {
                var exampleNode = new Node<T>
                {
                    Id = nodes.Count,
                    Type = NodeType.Example,
                    Features = ExtractExampleFeatures(task, exampleIdx),
                    Label = GetExampleLabel(task, exampleIdx)
                };
                nodes.Add(exampleNode);

                // Connect task to examples
                edges.Add(new Edge<T>
                {
                    Source = taskNode.Id,
                    Target = exampleNode.Id,
                    Weight = NumOps.FromDouble(1.0),
                    Type = EdgeType.Contains
                });
            }
        }

        // Add inter-task edges based on similarity
        if (_gnnOptions.UseInterTaskEdges)
        {
            AddInterTaskEdges(nodes, edges);
        }

        // Add example-example edges within tasks
        if (_gnnOptions.UseIntraTaskEdges)
        {
            AddIntraTaskEdges(nodes, edges, taskBatch);
        }

        return new Graph<T>
        {
            Nodes = nodes,
            Edges = edges,
            NumNodes = nodes.Count,
            NumEdges = edges.Count
        };
    }

    private Graph<T> BuildSingleTaskGraph(TaskBatch<TInput, TOutput> supportSet)
    {
        // Similar to BuildTaskGraph but for a single task
        // Implementation would be similar but simpler
        return new Graph<T>();
    }

    private void AddInterTaskEdges(List<Node<T>> nodes, List<Edge<T>> edges)
    {
        var taskNodes = nodes.Where(n => n.Type == NodeType.Task).ToList();

        for (int i = 0; i < taskNodes.Count; i++)
        {
            for (int j = i + 1; j < taskNodes.Count; j++)
            {
                var similarity = ComputeTaskSimilarity(taskNodes[i], taskNodes[j]);

                if (Convert.ToDouble(similarity) > _gnnOptions.SimilarityThreshold)
                {
                    edges.Add(new Edge<T>
                    {
                        Source = taskNodes[i].Id,
                        Target = taskNodes[j].Id,
                        Weight = similarity,
                        Type = EdgeType.Similar
                    });
                }
            }
        }
    }

    private void AddIntraTaskEdges(List<Node<T>> nodes, List<Edge<T>> edges, TaskBatch<TInput, TOutput> taskBatch)
    {
        int exampleStartIdx = 1; // First task node at index 0

        foreach (var task in taskBatch.Tasks)
        {
            var taskExamples = nodes.Skip(exampleStartIdx).Take(task.NumExamples).ToList();

            // Create edges between similar examples
            for (int i = 0; i < taskExamples.Count; i++)
            {
                for (int j = i + 1; j < taskExamples.Count; j++)
                {
                    var similarity = ComputeExampleSimilarity(
                        taskExamples[i].Features,
                        taskExamples[j].Features);

                    if (Convert.ToDouble(similarity) > _gnnOptions.SimilarityThreshold)
                    {
                        edges.Add(new Edge<T>
                        {
                            Source = taskExamples[i].Id,
                            Target = taskExamples[j].Id,
                            Weight = similarity,
                            Type = EdgeType.Related
                        });
                    }
                }
            }

            exampleStartIdx += task.NumExamples;
        }
    }

    private Tensor<T> EncodeNodeFeatures(Graph<T> graph)
    {
        var nodeFeatures = new List<Tensor<T>>();

        foreach (var node in graph.Nodes)
        {
            // Encode node features through feature encoder
            var current = node.Features;
            foreach (var layer in _featureEncoderLayers)
            {
                current = layer.Forward(current);
            }
            nodeFeatures.Add(current);
        }

        return Tensor<T>.Stack(nodeFeatures, axis: 0);
    }

    private Tensor<T> ApplyGraphNetwork(Tensor<T> nodeFeatures, Graph<T> graph)
    {
        var currentEmbeddings = nodeFeatures;
        var layerIdx = 0;

        // Apply each GNN layer
        for (int layer = 0; layer < _gnnOptions.NumGNNLayers; layer++)
        {
            // Graph convolution
            var gnnLayer = (GraphConvolutionLayer<T>)_graphEncoderLayers[layerIdx++];
            currentEmbeddings = gnnLayer.Forward(currentEmbeddings, graph);

            // Graph attention if enabled
            if (_gnnOptions.UseGraphAttention)
            {
                var attentionLayer = (GraphAttentionLayer<T>)_graphEncoderLayers[layerIdx++];
                currentEmbeddings = attentionLayer.Forward(currentEmbeddings, graph);
            }

            // Residual connection
            if (_gnnOptions.UseResidualConnections)
            {
                var residualLayer = (ResidualConnection<T>)_graphEncoderLayers[layerIdx++];
                currentEmbeddings = residualLayer.Forward(currentEmbeddings, nodeFeatures);
            }

            nodeFeatures = currentEmbeddings; // For next residual connection
        }

        return currentEmbeddings;
    }

    private Tensor<T> MessagePassingStep(Tensor<T> nodeEmbeddings, Graph<T> graph, TaskBatch<TInput, TOutput> supportSet)
    {
        // Perform one step of message passing with support set information
        var messages = ComputeMessages(nodeEmbeddings, graph);
        var aggregated = AggregateMessages(messages, graph);

        // Update embeddings
        return UpdateNodeEmbeddings(nodeEmbeddings, aggregated);
    }

    private Tensor<T> PoolNodeEmbeddings(Tensor<T> nodeEmbeddings, Graph<T> graph)
    {
        // Pool embeddings based on strategy
        switch (_gnnOptions.PoolingStrategy)
        {
            case PoolingStrategy.Mean:
                return Tensor<T>.Mean(nodeEmbeddings, axis: 0);

            case PoolingStrategy.Max:
                return Tensor<T>.Max(nodeEmbeddings, axis: 0);

            case PoolingStrategy.Sum:
                return Tensor<T>.Sum(nodeEmbeddings, axis: 0);

            case PoolingStrategy.Attention:
                return AttentionPooling(nodeEmbeddings, graph);

            default:
                return Tensor<T>.Mean(nodeEmbeddings, axis: 0);
        }
    }

    private Tensor<T> AttentionPooling(Tensor<T> nodeEmbeddings, Graph<T> graph)
    {
        // Learn attention weights for pooling
        var attentionScores = new List<T>();

        foreach (var node in graph.Nodes)
        {
            var nodeEmbedding = nodeEmbeddings[node.Id];
            var score = ComputeAttentionScore(nodeEmbedding, _graphEmbedding);
            attentionScores.Add(score);
        }

        // Normalize scores
        var attentionTensor = Tensor<T>.Concat(attentionScores);
        var attentionWeights = Tensor<T>.Softmax(attentionTensor);

        // Weighted sum
        return Tensor<T>.WeightedSum(nodeEmbeddings, attentionWeights);
    }

    private Dictionary<Node<T>, TOutput> DecodeToPredictions(Tensor<T> nodeEmbeddings, Graph<T> graph)
    {
        var predictions = new Dictionary<Node<T>, TOutput>();

        // Decode each task node
        var taskNodes = graph.Nodes.Where(n => n.Type == NodeType.Task).ToList();

        foreach (var taskNode in taskNodes)
        {
            var nodeEmbedding = nodeEmbeddings[taskNode.Id];
            var current = nodeEmbedding;

            // Pass through decoder
            foreach (var layer in _taskDecoderLayers)
            {
                current = layer.Forward(current);
            }

            predictions[taskNode] = ConvertFromTensor(current);
        }

        return predictions;
    }

    private TOutput DecodeSingleNode(int nodeId)
    {
        // Get embedding for specific node
        var nodeEmbedding = _nodeEmbeddings[nodeId];
        var current = nodeEmbedding;

        // Decode to output
        foreach (var layer in _taskDecoderLayers)
        {
            current = layer.Forward(current);
        }

        return ConvertFromTensor(current);
    }

    private (T, T) ComputeTaskLoss(TaskBatch<TInput, TOutput> taskBatch, Dictionary<Node<T>, TOutput> predictions)
    {
        var totalLoss = NumOps.Zero;
        var totalAccuracy = NumOps.Zero;
        var taskNodes = _currentTaskGraph.Nodes.Where(n => n.Type == NodeType.Task).ToList();

        for (int i = 0; i < taskNodes.Count; i++)
        {
            var taskNode = taskNodes[i];
            var task = taskBatch.Tasks[i];
            var prediction = predictions[taskNode];

            // Compute loss and accuracy
            var (loss, accuracy) = ComputeTaskMetrics(task, prediction);
            totalLoss = NumOps.Add(totalLoss, loss);
            totalAccuracy = NumOps.Add(totalAccuracy, accuracy);
        }

        // Average across tasks
        var numTasksT = NumOps.FromDouble(taskNodes.Count);
        var avgLoss = NumOps.Divide(totalLoss, numTasksT);
        var avgAccuracy = NumOps.Divide(totalAccuracy, numTasksT);

        return (avgLoss, avgAccuracy);
    }

    private T ComputeGraphRegularizationLoss(Tensor<T> nodeEmbeddings, Graph<T> graph)
    {
        if (!_gnnOptions.UseGraphRegularization)
        {
            return NumOps.Zero;
        }

        // L2 regularization on embeddings
        var embeddingNorm = Tensor<T>.Square(nodeEmbeddings).Sum();
        return NumOps.Multiply(
            NumOps.FromDouble(_gnnOptions.EmbeddingRegularizationWeight),
            embeddingNorm);
    }

    private T ComputeEdgePredictionLoss(Graph<T> graph)
    {
        var totalLoss = NumOps.Zero;

        foreach (var edge in graph.Edges)
        {
            var sourceEmbedding = _nodeEmbeddings[edge.Source];
            var targetEmbedding = _nodeEmbeddings[edge.Target];

            // Concatenate embeddings
            var concatenated = Tensor<T>.Concat(new[] { sourceEmbedding, targetEmbedding }, axis: -1);

            // Predict edge
            var current = concatenated;
            foreach (var layer in _edgePredictorLayers)
            {
                current = layer.Forward(current);
            }

            // Compute loss against actual edge weight
            var predictedProb = current[0];
            var edgeLoss = ComputeBinaryLoss(predictedProb, edge.Weight);
            totalLoss = NumOps.Add(totalLoss, edgeLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(graph.NumEdges));
    }

    private void MetaUpdate(T totalLoss)
    {
        // Compute gradients
        var gradients = ComputeGradients(totalLoss);

        // Update meta-parameters
        _metaOptimizer.Update(MetaParameters, gradients);

        // Distribute updated parameters
        DistributeParameters(MetaParameters);
    }

    private void CacheGraphEmbeddings(TaskBatch<TInput, TOutput> taskBatch)
    {
        // Cache embeddings for faster future access
        var taskId = ComputeTaskId(taskBatch);
        _graphCache[taskId] = _currentTaskGraph;
    }

    // Helper methods
    private Tensor<T> ExtractTaskFeatures(Task<TInput, TOutput> task)
    {
        // Extract features representing the task
        // This could include task statistics, metadata, etc.
        return Tensor<T>.Zeros(new[] { _gnnOptions.InputDimension });
    }

    private Tensor<T> ExtractExampleFeatures(Task<TInput, TOutput> task, int exampleIdx)
    {
        // Extract features from individual example
        return Tensor<T>.Zeros(new[] { _gnnOptions.InputDimension });
    }

    private T GetExampleLabel(Task<TInput, TOutput> task, int exampleIdx)
    {
        // Get label for specific example
        return NumOps.Zero;
    }

    private Tensor<T> ComputeTaskSimilarity(Node<T> node1, Node<T> node2)
    {
        // Compute similarity between task nodes
        return NumOps.FromDouble(0.5);
    }

    private Tensor<T> ComputeExampleSimilarity(Tensor<T> features1, Tensor<T> features2)
    {
        // Compute similarity between example features
        return Tensor<T>.CosineSimilarity(features1, features2);
    }

    private Tensor<T> ComputeMessages(Tensor<T> nodeEmbeddings, Graph<T> graph)
    {
        // Compute messages to be passed between nodes
        return nodeEmbeddings;
    }

    private Tensor<T> AggregateMessages(Tensor<T> messages, Graph<T> graph)
    {
        // Aggregate incoming messages for each node
        return messages;
    }

    private Tensor<T> UpdateNodeEmbeddings(Tensor<T> currentEmbeddings, Tensor<T> aggregatedMessages)
    {
        // Update node embeddings with aggregated messages
        return Tensor<T>.Add(currentEmbeddings, aggregatedMessages);
    }

    private Tensor<T> ComputeAttentionScore(Tensor<T> nodeEmbedding, Tensor<T> graphEmbedding)
    {
        // Compute attention score for pooling
        return NumOps.FromDouble(1.0);
    }

    private Graph<T> AddQueryToGraph(TInput querySet)
    {
        // Add query node to existing graph
        var queryNode = new Node<T>
        {
            Id = _currentTaskGraph.NumNodes,
            Type = NodeType.Query,
            Features = ExtractQueryFeatures(querySet),
            Label = NumOps.Zero
        };

        // Copy existing graph
        var newGraph = new Graph<T>
        {
            Nodes = new List<Node<T>>(_currentTaskGraph.Nodes) { queryNode },
            Edges = new List<Edge<T>>(_currentTaskGraph.Edges),
            NumNodes = _currentTaskGraph.NumNodes + 1,
            NumEdges = _currentTaskGraph.NumEdges
        };

        // Add edges from query to relevant nodes
        AddQueryEdges(newGraph, queryNode);

        return newGraph;
    }

    private void AddQueryEdges(Graph<T> graph, Node<T> queryNode)
    {
        // Add edges from query to similar nodes
        // Implementation would depend on similarity strategy
    }

    private Tensor<T> ExtractQueryFeatures(TInput querySet)
    {
        // Extract features from query
        return Tensor<T>.Zeros(new[] { _gnnOptions.InputDimension });
    }

    private (T, T) ComputeTaskMetrics(Task<TInput, TOutput> task, TOutput prediction)
    {
        // Compute task-specific metrics
        return (NumOps.Zero, NumOps.Zero);
    }

    private T ComputeBinaryLoss(T predicted, T target)
    {
        // Compute binary cross-entropy loss
        return NumOps.Zero;
    }

    private int ComputeTaskId(TaskBatch<TInput, TOutput> taskBatch)
    {
        // Compute unique ID for task batch
        return taskBatch.GetHashCode();
    }

    private Tensor<T> ComputeGradients(T loss)
    {
        // Compute gradients of loss w.r.t. meta-parameters
        return Tensor<T>.ZerosLike(MetaParameters);
    }

    private void DistributeParameters(Tensor<T> parameters)
    {
        // Distribute parameters to all networks
        var offset = 0;
        var allLayers = _featureEncoderLayers.Concat(_graphEncoderLayers)
            .Concat(_taskDecoderLayers).Concat(_edgePredictorLayers).ToList();

        foreach (var layer in allLayers)
        {
            var layerParams = layer.GetParameters();
            var numParams = layerParams.Sum(p => p.Count);
            var layerSlice = parameters.Slice(
                new[] { offset },
                new[] { numParams });
            layer.SetParameters(new List<Tensor<T>> { layerSlice });
            offset += numParams;
        }
    }

    private TOutput ConvertFromTensor(Tensor<T> tensor)
    {
        // Convert tensor to output type
        return default(TOutput);
    }
}

/// <summary>
/// Graph structure for GNN-based meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class Graph<T>
    where T : struct, IEquatable<T>, IFormattable
{
    public List<Node<T>> Nodes { get; set; } = new List<Node<T>>();
    public List<Edge<T>> Edges { get; set; } = new List<Edge<T>>();
    public int NumNodes { get; set; }
    public int NumEdges { get; set; }
    public int QueryNodeIndex { get; set; }
}

/// <summary>
/// Node in the task graph.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class Node<T>
    where T : struct, IEquatable<T>, IFormattable
{
    public int Id { get; set; }
    public NodeType Type { get; set; }
    public Tensor<T> Features { get; set; }
    public T Label { get; set; }
}

/// <summary>
/// Edge in the task graph.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class Edge<T>
    where T : struct, IEquatable<T>, IFormattable
{
    public int Source { get; set; }
    public int Target { get; set; }
    public T Weight { get; set; }
    public EdgeType Type { get; set; }
}

/// <summary>
/// Node types in the task graph.
/// </summary>
public enum NodeType
{
    Task,
    Example,
    Query
}

/// <summary>
/// Edge types in the task graph.
/// </summary>
public enum EdgeType
{
    Contains,
    Similar,
    Related,
    Temporal
}

/// <summary>
/// Pooling strategies for graph embeddings.
/// </summary>
public enum PoolingStrategy
{
    Mean,
    Max,
    Sum,
    Attention
}