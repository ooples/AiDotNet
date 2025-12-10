using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Implements a complete model for node classification tasks on graphs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Node classification predicts labels for individual nodes in a graph using:
/// - Node features
/// - Graph structure (adjacency information)
/// - Semi-supervised learning (only some nodes have labels)
/// </para>
/// <para><b>For Beginners:</b> This model classifies nodes in a graph.
///
/// **How it works:**
///
/// 1. **Input**: Graph with node features and structure
/// 2. **Processing**: Stack of graph convolutional layers
///    - Each layer aggregates information from neighbors
///    - Features become more "context-aware" at each layer
///    - After k layers, each node knows about its k-hop neighborhood
/// 3. **Output**: Class predictions for each node
///
/// **Example architecture:**
/// ```
/// Input: [num_nodes, input_features]
///   ↓
/// GCN Layer 1: [num_nodes, hidden_dim]
///   ↓
/// Activation (ReLU)
///   ↓
/// Dropout
///   ↓
/// GCN Layer 2: [num_nodes, num_classes]
///   ↓
/// Softmax: [num_nodes, num_classes] (probabilities)
/// ```
///
/// **Training:**
/// - Use labeled nodes for computing loss
/// - Unlabeled nodes still participate in message passing
/// - Graph structure helps propagate label information
/// </para>
/// </remarks>
public class NodeClassificationModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly List<ILayer<T>> _layers;
    private readonly IGraphConvolutionLayer<T> _firstGraphLayer;
    private Tensor<T>? _adjacencyMatrix;
    private bool _isTrainingMode;

    /// <summary>
    /// Gets the number of input features per node.
    /// </summary>
    public int InputFeatures { get; private set; }

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="NodeClassificationModel{T}"/> class.
    /// </summary>
    /// <param name="layers">List of layers including graph convolutional layers.</param>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="numClasses">Number of classification classes.</param>
    /// <remarks>
    /// <para>
    /// Typical layer configuration:
    /// 1. Graph convolutional layer (GCN, GAT, GraphSAGE, etc.)
    /// 2. Activation function (ReLU, LeakyReLU)
    /// 3. Dropout (for regularization)
    /// 4. Additional graph conv layers as needed
    /// 5. Final layer projects to num_classes dimensions
    /// </para>
    /// </remarks>
    public NodeClassificationModel(
        List<ILayer<T>> layers,
        int inputFeatures,
        int numClasses)
    {
        _layers = layers ?? throw new ArgumentNullException(nameof(layers));
        InputFeatures = inputFeatures;
        NumClasses = numClasses;

        // Find first graph layer to set adjacency matrix
        _firstGraphLayer = layers.OfType<IGraphConvolutionLayer<T>>().FirstOrDefault()
            ?? throw new ArgumentException("Model must contain at least one graph convolutional layer.");
    }

    /// <summary>
    /// Sets the adjacency matrix for all graph layers in the model.
    /// </summary>
    /// <param name="adjacencyMatrix">The graph adjacency matrix.</param>
    /// <remarks>
    /// <para>
    /// Call this before training or inference to provide the graph structure.
    /// All graph convolutional layers in the model will use this adjacency matrix.
    /// </para>
    /// <para><b>For Beginners:</b> The adjacency matrix tells the model which nodes are connected.
    ///
    /// For a graph with 4 nodes:
    /// ```
    /// Node connections:
    /// 0 -- 1
    /// |    |
    /// 2 -- 3
    ///
    /// Adjacency matrix:
    /// [0 1 1 0]
    /// [1 0 0 1]
    /// [1 0 0 1]
    /// [0 1 1 0]
    /// ```
    /// Where A[i,j] = 1 means nodes i and j are connected.
    /// </para>
    /// </remarks>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;

        // Set adjacency matrix for all graph layers
        foreach (var layer in _layers.OfType<IGraphConvolutionLayer<T>>())
        {
            layer.SetAdjacencyMatrix(adjacencyMatrix);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set before forward pass. Call SetAdjacencyMatrix() first.");
        }

        var current = input;
        foreach (var layer in _layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <inheritdoc/>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var currentGradient = outputGradient;
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            currentGradient = _layers[i].Backward(currentGradient);
        }
        return currentGradient;
    }

    /// <inheritdoc/>
    public void UpdateParameters(T learningRate)
    {
        foreach (var layer in _layers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <inheritdoc/>
    public void SetTrainingMode(bool isTraining)
    {
        _isTrainingMode = isTraining;
        foreach (var layer in _layers)
        {
            layer.SetTrainingMode(isTraining);
        }
    }

    /// <summary>
    /// Trains the model on a node classification task.
    /// </summary>
    /// <param name="task">The node classification task with graph data and labels.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <returns>Training history with loss and accuracy per epoch.</returns>
    /// <remarks>
    /// <para>
    /// Training procedure:
    /// 1. Set adjacency matrix from task graph
    /// 2. For each epoch:
    ///    - Forward pass through all nodes
    ///    - Compute loss only on training nodes
    ///    - Backward pass
    ///    - Update parameters
    ///    - Evaluate on validation nodes
    /// </para>
    /// <para><b>For Beginners:</b> Semi-supervised training is special:
    ///
    /// - **All nodes participate in message passing**
    ///   Even unlabeled test nodes help propagate information
    ///
    /// - **Loss computed only on labeled training nodes**
    ///   We only update weights based on nodes where we know the answer
    ///
    /// - **Test nodes benefit from training nodes**
    ///   Graph structure lets label information flow through the network
    ///
    /// This is like learning in school:
    /// - Some students get answers (training nodes)
    /// - They help friends (neighbors) understand
    /// - Friends share with their friends (message passing)
    /// - Eventually everyone learns (test nodes get correct labels)
    /// </para>
    /// </remarks>
    public Dictionary<string, List<double>> Train(
        NodeClassificationTask<T> task,
        int epochs,
        T learningRate)
    {
        if (task.Graph.AdjacencyMatrix == null)
        {
            throw new ArgumentException("Task graph must have an adjacency matrix.");
        }

        SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);
        SetTrainingMode(true);

        var history = new Dictionary<string, List<double>>
        {
            ["train_loss"] = new List<double>(),
            ["train_accuracy"] = new List<double>(),
            ["val_accuracy"] = new List<double>()
        };

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Forward pass on all nodes
            var logits = Forward(task.Graph.NodeFeatures);

            // Compute loss on training nodes only
            double totalLoss = 0.0;
            int correct = 0;

            foreach (var nodeIdx in task.TrainIndices)
            {
                // Cross-entropy loss for this node
                for (int c = 0; c < task.NumClasses; c++)
                {
                    var logit = NumOps.ToDouble(logits[nodeIdx, c]);
                    var label = NumOps.ToDouble(task.Labels[nodeIdx, c]);
                    totalLoss -= label * Math.Log(Math.Max(logit, 1e-10));
                }

                // Accuracy
                int predictedClass = GetPredictedClass(logits, nodeIdx, task.NumClasses);
                int trueClass = GetTrueClass(task.Labels, nodeIdx, task.NumClasses);
                if (predictedClass == trueClass) correct++;
            }

            double avgLoss = totalLoss / task.TrainIndices.Length;
            double trainAcc = (double)correct / task.TrainIndices.Length;

            // Validation accuracy
            double valAcc = EvaluateAccuracy(logits, task.Labels, task.ValIndices, task.NumClasses);

            history["train_loss"].Add(avgLoss);
            history["train_accuracy"].Add(trainAcc);
            history["val_accuracy"].Add(valAcc);

            // Backward pass and update
            var gradient = ComputeGradient(logits, task.Labels, task.TrainIndices, task.NumClasses);
            Backward(gradient);
            UpdateParameters(learningRate);
        }

        SetTrainingMode(false);
        return history;
    }

    /// <summary>
    /// Evaluates the model on test nodes.
    /// </summary>
    /// <param name="task">The node classification task.</param>
    /// <returns>Test accuracy.</returns>
    public double Evaluate(NodeClassificationTask<T> task)
    {
        if (task.Graph.AdjacencyMatrix != null)
        {
            SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);
        }

        SetTrainingMode(false);
        var logits = Forward(task.Graph.NodeFeatures);
        return EvaluateAccuracy(logits, task.Labels, task.TestIndices, task.NumClasses);
    }

    private double EvaluateAccuracy(Tensor<T> logits, Tensor<T> labels, int[] indices, int numClasses)
    {
        int correct = 0;
        foreach (var nodeIdx in indices)
        {
            int predictedClass = GetPredictedClass(logits, nodeIdx, numClasses);
            int trueClass = GetTrueClass(labels, nodeIdx, numClasses);
            if (predictedClass == trueClass) correct++;
        }
        return (double)correct / indices.Length;
    }

    private int GetPredictedClass(Tensor<T> logits, int nodeIdx, int numClasses)
    {
        int maxClass = 0;
        T maxValue = logits[nodeIdx, 0];
        for (int c = 1; c < numClasses; c++)
        {
            if (NumOps.GreaterThan(logits[nodeIdx, c], maxValue))
            {
                maxValue = logits[nodeIdx, c];
                maxClass = c;
            }
        }
        return maxClass;
    }

    private int GetTrueClass(Tensor<T> labels, int nodeIdx, int numClasses)
    {
        for (int c = 0; c < numClasses; c++)
        {
            if (!NumOps.Equals(labels[nodeIdx, c], NumOps.Zero))
                return c;
        }
        return 0;
    }

    private Tensor<T> ComputeGradient(Tensor<T> logits, Tensor<T> labels, int[] trainIndices, int numClasses)
    {
        var gradient = new Tensor<T>(logits.Shape);

        foreach (var nodeIdx in trainIndices)
        {
            for (int c = 0; c < numClasses; c++)
            {
                gradient[nodeIdx, c] = NumOps.Subtract(logits[nodeIdx, c], labels[nodeIdx, c]);
            }
        }

        return gradient;
    }
}
