using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.Graph;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Implements a complete model for graph classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Graph classification assigns labels to entire graphs based on their structure and features.
/// The model consists of:
/// 1. Node-level processing (GNN layers)
/// 2. Graph-level pooling (aggregate node embeddings)
/// 3. Classification head (fully connected layers)
/// </para>
/// <para><b>For Beginners:</b> This model classifies whole graphs.
///
/// **Architecture pipeline:**
///
/// ```
/// Step 1: Node Encoding
/// Input: Graph with node features
/// Process: Stack of GNN layers
/// Output: Node embeddings [num_nodes, hidden_dim]
///
/// Step 2: Graph Pooling (KEY STEP!)
/// Input: Node embeddings from variable-sized graph
/// Process: Aggregate to fixed-size representation
/// Output: Graph embedding [hidden_dim]
///
/// Step 3: Classification
/// Input: Graph embedding [hidden_dim]
/// Process: Fully connected layers
/// Output: Class probabilities [num_classes]
/// ```
///
/// **Why pooling is crucial:**
/// - Graphs have variable sizes (10 nodes vs 100 nodes)
/// - Need fixed-size representation for classification
/// - Like summarizing a book (any length) into a fixed review (200 words)
///
/// **Example: Molecular toxicity prediction**
/// ```
/// Molecule (graph) → GNN layers → Molecule embedding → Classifier → Toxic? (Yes/No)
///
/// Small molecule (10 atoms):
///   10 nodes → GNN → 10 embeddings → Pool → 1 graph embedding → Classify
///
/// Large molecule (50 atoms):
///   50 nodes → GNN → 50 embeddings → Pool → 1 graph embedding → Classify
///
/// Both produce same-sized graph embedding despite different input sizes!
/// ```
/// </para>
/// </remarks>
public class GraphClassificationModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly List<ILayer<T>> _gnnLayers;
    private readonly List<ILayer<T>> _classifierLayers;
    private readonly GraphPooling _poolingType;
    private Tensor<T>? _nodeEmbeddings;
    private Tensor<T>? _graphEmbedding;
    private bool _isTrainingMode;

    /// <summary>
    /// Graph pooling methods for aggregating node embeddings.
    /// </summary>
    public enum GraphPooling
    {
        /// <summary>Mean pooling: Average all node embeddings.</summary>
        Mean,

        /// <summary>Max pooling: Take max across all node embeddings.</summary>
        Max,

        /// <summary>Sum pooling: Sum all node embeddings.</summary>
        Sum,

        /// <summary>Attention pooling: Weighted average with learned attention.</summary>
        Attention
    }

    /// <summary>
    /// Gets the graph embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; private set; }

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphClassificationModel{T}"/> class.
    /// </summary>
    /// <param name="gnnLayers">GNN layers for node-level processing.</param>
    /// <param name="classifierLayers">Fully connected layers for graph-level classification.</param>
    /// <param name="embeddingDim">Dimension of graph embedding after pooling.</param>
    /// <param name="numClasses">Number of classification classes.</param>
    /// <param name="poolingType">Method for pooling node embeddings to graph embedding.</param>
    /// <remarks>
    /// <para>
    /// Typical architecture:
    /// ```
    /// GNN Layers:
    ///   - GraphConv(in_features, 64)
    ///   - ReLU
    ///   - GraphConv(64, 128)
    ///   - ReLU
    ///   - GraphConv(128, embedding_dim)
    ///
    /// Pooling: Sum/Mean/Max → [embedding_dim]
    ///
    /// Classifier Layers:
    ///   - Linear(embedding_dim, 64)
    ///   - ReLU
    ///   - Dropout(0.5)
    ///   - Linear(64, num_classes)
    /// ```
    /// </para>
    /// <para><b>For Beginners:</b> Choosing pooling strategy:
    ///
    /// **Mean Pooling:**
    /// - Average all node features
    /// - Good for: General purpose, stable gradients
    /// - Example: "What's the average property across atoms?"
    ///
    /// **Max Pooling:**
    /// - Take maximum value per feature dimension
    /// - Good for: Capturing extreme/important features
    /// - Example: "Is there ANY atom with this critical property?"
    ///
    /// **Sum Pooling:**
    /// - Sum all node features
    /// - Good for: Size-dependent properties
    /// - Example: "Total molecular weight" (bigger molecules = larger sum)
    ///
    /// **Attention Pooling:**
    /// - Learned weighted average (important nodes weighted higher)
    /// - Good for: Complex patterns, best accuracy
    /// - Example: "Which atoms matter most for toxicity?"
    /// - Trade-off: More parameters, slower training
    /// </para>
    /// </remarks>
    public GraphClassificationModel(
        List<ILayer<T>> gnnLayers,
        List<ILayer<T>> classifierLayers,
        int embeddingDim,
        int numClasses,
        GraphPooling poolingType = GraphPooling.Mean)
    {
        _gnnLayers = gnnLayers ?? throw new ArgumentNullException(nameof(gnnLayers));
        _classifierLayers = classifierLayers ?? throw new ArgumentNullException(nameof(classifierLayers));
        EmbeddingDim = embeddingDim;
        NumClasses = numClasses;
        _poolingType = poolingType;
    }

    /// <summary>
    /// Sets the adjacency matrix for a single graph.
    /// </summary>
    /// <param name="adjacencyMatrix">The graph adjacency matrix.</param>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        foreach (var layer in _gnnLayers.OfType<IGraphConvolutionLayer<T>>())
        {
            layer.SetAdjacencyMatrix(adjacencyMatrix);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Step 1: Node-level processing through GNN layers
        var current = input;
        foreach (var layer in _gnnLayers)
        {
            current = layer.Forward(current);
        }
        _nodeEmbeddings = current;

        // Step 2: Pool node embeddings to graph embedding
        _graphEmbedding = PoolGraph(_nodeEmbeddings);

        // Step 3: Graph-level classification
        current = _graphEmbedding;
        foreach (var layer in _classifierLayers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Pools node embeddings into a single graph-level embedding.
    /// </summary>
    /// <param name="nodeEmbeddings">Node embeddings of shape [batch_size, num_nodes, embedding_dim].</param>
    /// <returns>Graph embedding of shape [batch_size, embedding_dim].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pooling converts variable-sized node sets to fixed size.
    ///
    /// Think of it like summarizing reviews:
    /// - **Input**: 10 movie reviews (variable number of words each)
    /// - **Pooling**: Extract key sentiment (fixed-size summary)
    /// - **Output**: Overall rating (fixed size)
    ///
    /// For graphs:
    /// - **Input**: Variable number of nodes with embeddings
    /// - **Pooling**: Aggregate into single vector
    /// - **Output**: One embedding representing entire graph
    /// </para>
    /// </remarks>
    private Tensor<T> PoolGraph(Tensor<T> nodeEmbeddings)
    {
        int batchSize = nodeEmbeddings.Shape[0];
        int numNodes = nodeEmbeddings.Shape[1];
        int embDim = nodeEmbeddings.Shape[2];

        var graphEmb = new Tensor<T>([batchSize, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            switch (_poolingType)
            {
                case GraphPooling.Mean:
                    // Average pooling
                    for (int d = 0; d < embDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int n = 0; n < numNodes; n++)
                        {
                            sum = NumOps.Add(sum, nodeEmbeddings[b, n, d]);
                        }
                        graphEmb[b, d] = NumOps.Divide(sum, NumOps.FromDouble(numNodes));
                    }
                    break;

                case GraphPooling.Max:
                    // Max pooling
                    for (int d = 0; d < embDim; d++)
                    {
                        T maxVal = nodeEmbeddings[b, 0, d];
                        for (int n = 1; n < numNodes; n++)
                        {
                            if (NumOps.GreaterThan(nodeEmbeddings[b, n, d], maxVal))
                            {
                                maxVal = nodeEmbeddings[b, n, d];
                            }
                        }
                        graphEmb[b, d] = maxVal;
                    }
                    break;

                case GraphPooling.Sum:
                    // Sum pooling
                    for (int d = 0; d < embDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int n = 0; n < numNodes; n++)
                        {
                            sum = NumOps.Add(sum, nodeEmbeddings[b, n, d]);
                        }
                        graphEmb[b, d] = sum;
                    }
                    break;

                case GraphPooling.Attention:
                    // Simplified attention pooling (full version would learn attention weights)
                    // For now, use uniform attention (equivalent to mean)
                    for (int d = 0; d < embDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int n = 0; n < numNodes; n++)
                        {
                            sum = NumOps.Add(sum, nodeEmbeddings[b, n, d]);
                        }
                        graphEmb[b, d] = NumOps.Divide(sum, NumOps.FromDouble(numNodes));
                    }
                    break;
            }
        }

        return graphEmb;
    }

    /// <inheritdoc/>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backprop through classifier
        var currentGradient = outputGradient;
        for (int i = _classifierLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _classifierLayers[i].Backward(currentGradient);
        }

        // Backprop through pooling (distribute gradient to all nodes)
        currentGradient = BackpropPooling(currentGradient);

        // Backprop through GNN layers
        for (int i = _gnnLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _gnnLayers[i].Backward(currentGradient);
        }

        return currentGradient;
    }

    private Tensor<T> BackpropPooling(Tensor<T> gradGraphEmb)
    {
        if (_nodeEmbeddings == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward.");
        }

        int batchSize = _nodeEmbeddings.Shape[0];
        int numNodes = _nodeEmbeddings.Shape[1];
        int embDim = _nodeEmbeddings.Shape[2];

        var gradNodeEmb = new Tensor<T>([batchSize, numNodes, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            switch (_poolingType)
            {
                case GraphPooling.Mean:
                    // Gradient distributed equally to all nodes
                    for (int n = 0; n < numNodes; n++)
                    {
                        for (int d = 0; d < embDim; d++)
                        {
                            gradNodeEmb[b, n, d] = NumOps.Divide(
                                gradGraphEmb[b, d],
                                NumOps.FromDouble(numNodes));
                        }
                    }
                    break;

                case GraphPooling.Max:
                    // Gradient goes only to node that had max value
                    for (int d = 0; d < embDim; d++)
                    {
                        int maxIdx = 0;
                        T maxVal = _nodeEmbeddings[b, 0, d];
                        for (int n = 1; n < numNodes; n++)
                        {
                            if (NumOps.GreaterThan(_nodeEmbeddings[b, n, d], maxVal))
                            {
                                maxVal = _nodeEmbeddings[b, n, d];
                                maxIdx = n;
                            }
                        }
                        gradNodeEmb[b, maxIdx, d] = gradGraphEmb[b, d];
                    }
                    break;

                case GraphPooling.Sum:
                case GraphPooling.Attention:
                    // Full gradient to all nodes
                    for (int n = 0; n < numNodes; n++)
                    {
                        for (int d = 0; d < embDim; d++)
                        {
                            gradNodeEmb[b, n, d] = gradGraphEmb[b, d];
                        }
                    }
                    break;
            }
        }

        return gradNodeEmb;
    }

    /// <inheritdoc/>
    public void UpdateParameters(T learningRate)
    {
        foreach (var layer in _gnnLayers)
        {
            layer.UpdateParameters(learningRate);
        }
        foreach (var layer in _classifierLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <inheritdoc/>
    public void SetTrainingMode(bool isTraining)
    {
        _isTrainingMode = isTraining;
        foreach (var layer in _gnnLayers)
        {
            layer.SetTrainingMode(isTraining);
        }
        foreach (var layer in _classifierLayers)
        {
            layer.SetTrainingMode(isTraining);
        }
    }

    /// <summary>
    /// Trains the model on a graph classification task.
    /// </summary>
    /// <param name="task">The graph classification task with training/validation/test graphs.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <param name="batchSize">Number of graphs per batch.</param>
    /// <returns>Training history with loss and accuracy per epoch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training on batches of graphs:
    ///
    /// **Challenge:** Graphs have different sizes
    /// - Graph 1: 10 nodes, 15 edges
    /// - Graph 2: 25 nodes, 40 edges
    /// - Graph 3: 8 nodes, 12 edges
    ///
    /// **Solution: Process one at a time or batch similar sizes**
    ///
    /// Training loop:
    /// ```
    /// For each epoch:
    ///   For each graph in training set:
    ///     1. Set graph's adjacency matrix
    ///     2. Forward pass: nodes → GNN → pool → classify
    ///     3. Compute loss with true label
    ///     4. Backward pass
    ///     5. Update parameters
    ///   Evaluate on validation set
    /// ```
    ///
    /// Unlike node classification (semi-supervised on one graph),
    /// graph classification is supervised learning on a dataset of graphs.
    /// </para>
    /// </remarks>
    public Dictionary<string, List<double>> Train(
        GraphClassificationTask<T> task,
        int epochs,
        T learningRate,
        int batchSize = 1)
    {
        SetTrainingMode(true);

        var history = new Dictionary<string, List<double>>
        {
            ["train_loss"] = new List<double>(),
            ["train_accuracy"] = new List<double>(),
            ["val_accuracy"] = new List<double>()
        };

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double epochLoss = 0.0;
            int correctTrain = 0;

            // Training loop
            for (int i = 0; i < task.TrainGraphs.Count; i++)
            {
                var graph = task.TrainGraphs[i];
                if (graph.AdjacencyMatrix == null)
                {
                    throw new ArgumentException($"Training graph {i} must have an adjacency matrix.");
                }

                SetAdjacencyMatrix(graph.AdjacencyMatrix);
                var logits = Forward(graph.NodeFeatures);

                // Compute loss
                double loss = 0.0;
                for (int c = 0; c < NumClasses; c++)
                {
                    var logit = NumOps.ToDouble(logits[0, c]);
                    var label = NumOps.ToDouble(task.TrainLabels[i, c]);
                    loss -= label * Math.Log(Math.Max(logit, 1e-10));
                }
                epochLoss += loss;

                // Accuracy
                int predictedClass = GetPredictedClass(logits);
                int trueClass = GetTrueClass(task.TrainLabels, i, NumClasses);
                if (predictedClass == trueClass) correctTrain++;

                // Backward and update
                var gradient = ComputeGradient(logits, task.TrainLabels, i, NumClasses);
                Backward(gradient);
                UpdateParameters(learningRate);
            }

            double avgLoss = epochLoss / task.TrainGraphs.Count;
            double trainAcc = (double)correctTrain / task.TrainGraphs.Count;

            // Validation accuracy
            double valAcc = EvaluateGraphs(task.ValGraphs, task.ValLabels, NumClasses);

            history["train_loss"].Add(avgLoss);
            history["train_accuracy"].Add(trainAcc);
            history["val_accuracy"].Add(valAcc);
        }

        SetTrainingMode(false);
        return history;
    }

    /// <summary>
    /// Evaluates the model on test graphs.
    /// </summary>
    public double Evaluate(GraphClassificationTask<T> task)
    {
        return EvaluateGraphs(task.TestGraphs, task.TestLabels, NumClasses);
    }

    private double EvaluateGraphs(List<GraphData<T>> graphs, Tensor<T> labels, int numClasses)
    {
        SetTrainingMode(false);
        int correct = 0;

        for (int i = 0; i < graphs.Count; i++)
        {
            var graph = graphs[i];
            if (graph.AdjacencyMatrix != null)
            {
                SetAdjacencyMatrix(graph.AdjacencyMatrix);
            }

            var logits = Forward(graph.NodeFeatures);
            int predictedClass = GetPredictedClass(logits);
            int trueClass = GetTrueClass(labels, i, numClasses);

            if (predictedClass == trueClass) correct++;
        }

        return (double)correct / graphs.Count;
    }

    private int GetPredictedClass(Tensor<T> logits)
    {
        int maxClass = 0;
        T maxValue = logits[0, 0];
        for (int c = 1; c < NumClasses; c++)
        {
            if (NumOps.GreaterThan(logits[0, c], maxValue))
            {
                maxValue = logits[0, c];
                maxClass = c;
            }
        }
        return maxClass;
    }

    private int GetTrueClass(Tensor<T> labels, int graphIdx, int numClasses)
    {
        for (int c = 0; c < numClasses; c++)
        {
            if (!NumOps.Equals(labels[graphIdx, c], NumOps.Zero))
                return c;
        }
        return 0;
    }

    private Tensor<T> ComputeGradient(Tensor<T> logits, Tensor<T> labels, int graphIdx, int numClasses)
    {
        var gradient = new Tensor<T>([1, numClasses]);
        for (int c = 0; c < numClasses; c++)
        {
            gradient[0, c] = NumOps.Subtract(logits[0, c], labels[graphIdx, c]);
        }
        return gradient;
    }
}
