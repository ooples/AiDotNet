using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Implements a complete neural network model for node classification tasks on graphs.
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
///   |
/// GCN Layer 1: [num_nodes, hidden_dim]
///   |
/// Activation (ReLU)
///   |
/// Dropout
///   |
/// GCN Layer 2: [num_nodes, num_classes]
///   |
/// Softmax: [num_nodes, num_classes] (probabilities)
/// ```
///
/// **Training:**
/// - Use labeled nodes for computing loss
/// - Unlabeled nodes still participate in message passing
/// - Graph structure helps propagate label information
/// </para>
/// </remarks>
public class NodeClassificationModel<T> : NeuralNetworkBase<T>
{
    private static new readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private Tensor<T>? _cachedAdjacencyMatrix;

    /// <summary>
    /// Gets the number of input features per node.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses { get; }

    /// <summary>
    /// Gets the hidden dimension size.
    /// </summary>
    public int HiddenDim { get; }

    /// <summary>
    /// Gets the number of graph layers.
    /// </summary>
    public int NumLayers { get; }

    /// <summary>
    /// Gets the dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="NodeClassificationModel{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output sizes and layers.</param>
    /// <param name="hiddenDim">Hidden dimension for intermediate layers (default: 64).</param>
    /// <param name="numLayers">Number of graph convolutional layers (default: 2).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.5).</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a node classification model:
    ///
    /// ```csharp
    /// // Create architecture for Cora citation network
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional,
    ///     NeuralNetworkTaskType.MultiClassClassification,
    ///     NetworkComplexity.Simple,
    ///     inputSize: 1433,    // Cora has 1433 word features
    ///     outputSize: 7);     // 7 paper categories
    ///
    /// // Create model with default layers
    /// var model = new NodeClassificationModel&lt;double&gt;(architecture);
    ///
    /// // Train on node classification task
    /// var history = model.TrainOnTask(task, epochs: 200, learningRate: 0.01);
    /// ```
    /// </para>
    /// </remarks>
    public NodeClassificationModel(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 64,
        int numLayers = 2,
        double dropoutRate = 0.5,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), maxGradNorm)
    {
        InputFeatures = architecture.InputSize;
        NumClasses = architecture.OutputSize;
        HiddenDim = hiddenDim;
        NumLayers = numLayers;
        DropoutRate = dropoutRate;

        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Create default node classification layers using LayerHelper
            Layers.AddRange(LayerHelper<T>.CreateDefaultNodeClassificationLayers(
                Architecture, HiddenDim, NumLayers, DropoutRate));
        }
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
    /// </remarks>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;

        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
        }
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <returns>Output predictions for all nodes.</returns>
    public Tensor<T> Forward(Tensor<T> nodeFeatures)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set before forward pass. Call SetAdjacencyMatrix() first.");
        }

        var current = nodeFeatures;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Performs a backward pass through the network.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var currentGradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }
        return currentGradient;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParamCount = layer.ParameterCount;
            if (layerParamCount > 0)
            {
                var layerParams = parameters.SubVector(index, layerParamCount);
                layer.SetParameters(layerParams);
                index += layerParamCount;
            }
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
    /// </para>
    /// </remarks>
    public Dictionary<string, List<double>> TrainOnTask(
        NodeClassificationTask<T> task,
        int epochs,
        double learningRate = 0.01)
    {
        if (task.Graph.AdjacencyMatrix is null)
        {
            throw new ArgumentException("Task graph must have an adjacency matrix.");
        }

        SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);

        var history = new Dictionary<string, List<double>>
        {
            ["train_loss"] = new List<double>(),
            ["train_accuracy"] = new List<double>(),
            ["val_accuracy"] = new List<double>()
        };

        var lr = NumOps.FromDouble(learningRate);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Set all layers to training mode
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            // Forward pass on all nodes
            var logits = Forward(task.Graph.NodeFeatures);

            // Compute loss and accuracy on training nodes
            var (loss, trainAcc) = ComputeLossAndAccuracy(logits, task.Labels, task.TrainIndices, task.NumClasses);

            // Validation accuracy
            double valAcc = EvaluateAccuracy(logits, task.Labels, task.ValIndices, task.NumClasses);

            history["train_loss"].Add(loss);
            history["train_accuracy"].Add(trainAcc);
            history["val_accuracy"].Add(valAcc);

            // Compute gradient and backward pass
            var gradient = ComputeGradient(logits, task.Labels, task.TrainIndices, task.NumClasses);
            Backward(gradient);

            // Update parameters
            foreach (var layer in Layers)
            {
                layer.UpdateParameters(lr);
            }
        }

        // Set layers back to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return history;
    }

    /// <summary>
    /// Evaluates the model on test nodes.
    /// </summary>
    /// <param name="task">The node classification task.</param>
    /// <returns>Test accuracy.</returns>
    public double EvaluateOnTask(NodeClassificationTask<T> task)
    {
        if (task.Graph.AdjacencyMatrix is not null)
        {
            SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        var logits = Forward(task.Graph.NodeFeatures);
        return EvaluateAccuracy(logits, task.Labels, task.TestIndices, task.NumClasses);
    }

    private (double loss, double accuracy) ComputeLossAndAccuracy(
        Tensor<T> logits, Tensor<T> labels, int[] indices, int numClasses)
    {
        if (indices.Length == 0)
        {
            return (0.0, 0.0);
        }

        // Gather logits and labels for the subset of nodes
        var subsetLogits = new Tensor<T>([indices.Length, numClasses]);
        var subsetLabels = new Tensor<T>([indices.Length, numClasses]);
        for (int i = 0; i < indices.Length; i++)
        {
            int nodeIdx = indices[i];
            for (int c = 0; c < numClasses; c++)
            {
                subsetLogits[i, c] = logits[nodeIdx, c];
                subsetLabels[i, c] = labels[nodeIdx, c];
            }
        }

        // Vectorized cross-entropy loss: -sum(labels * log(clamp(logits, epsilon, 1)))
        T epsilon = NumOps.FromDouble(1e-10);
        T one = NumOps.One;
        var clampedLogits = Engine.TensorClamp(subsetLogits, epsilon, one);
        var logLogits = Engine.TensorLog(clampedLogits);
        var labelLogProduct = Engine.TensorMultiply(subsetLabels, logLogits);
        T negSumLoss = Engine.TensorSum(labelLogProduct);
        double totalLoss = -NumOps.ToDouble(negSumLoss);
        double avgLoss = totalLoss / indices.Length;

        // Compute accuracy: compare argmax of logits vs argmax of labels
        int correct = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            int predictedClass = GetPredictedClassFromSubset(subsetLogits, i, numClasses);
            int trueClass = GetTrueClassFromSubset(subsetLabels, i, numClasses);
            if (predictedClass == trueClass) correct++;
        }
        double accuracy = (double)correct / indices.Length;

        return (avgLoss, accuracy);
    }

    private int GetPredictedClassFromSubset(Tensor<T> logits, int rowIdx, int numClasses)
    {
        int maxClass = 0;
        T maxValue = logits[rowIdx, 0];
        for (int c = 1; c < numClasses; c++)
        {
            if (NumOps.GreaterThan(logits[rowIdx, c], maxValue))
            {
                maxValue = logits[rowIdx, c];
                maxClass = c;
            }
        }
        return maxClass;
    }

    private int GetTrueClassFromSubset(Tensor<T> labels, int rowIdx, int numClasses)
    {
        for (int c = 0; c < numClasses; c++)
        {
            if (!NumOps.Equals(labels[rowIdx, c], NumOps.Zero))
                return c;
        }
        return 0;
    }

    private double EvaluateAccuracy(Tensor<T> logits, Tensor<T> labels, int[] indices, int numClasses)
    {
        if (indices.Length == 0) return 0.0;

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
        // Initialize gradient tensor (zeros)
        var gradient = new Tensor<T>(logits.Shape);
        Engine.TensorFill(gradient, NumOps.Zero);

        if (trainIndices.Length == 0)
        {
            return gradient;
        }

        // Gather logits and labels for the subset of training nodes
        var subsetLogits = new Tensor<T>([trainIndices.Length, numClasses]);
        var subsetLabels = new Tensor<T>([trainIndices.Length, numClasses]);
        for (int i = 0; i < trainIndices.Length; i++)
        {
            int nodeIdx = trainIndices[i];
            for (int c = 0; c < numClasses; c++)
            {
                subsetLogits[i, c] = logits[nodeIdx, c];
                subsetLabels[i, c] = labels[nodeIdx, c];
            }
        }

        // Vectorized gradient computation: (logits - labels) / n
        var diff = Engine.TensorSubtract<T>(subsetLogits, subsetLabels);
        var scaledDiff = Engine.TensorDivideScalar(diff, NumOps.FromDouble(trainIndices.Length));

        // Scatter gradients back to the full gradient tensor
        for (int i = 0; i < trainIndices.Length; i++)
        {
            int nodeIdx = trainIndices[i];
            for (int c = 0; c < numClasses; c++)
            {
                gradient[nodeIdx, c] = scaledDiff[i, c];
            }
        }

        return gradient;
    }

    /// <summary>
    /// Gets all parameters as a vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                allParams.Add(layerParams[i]);
            }
        }
        return new Vector<T>([.. allParams]);
    }

    #region Abstract Method Implementations

    /// <summary>
    /// Makes a prediction using the trained network.
    /// </summary>
    /// <param name="input">The input tensor containing node features.</param>
    /// <returns>The prediction tensor with class probabilities for each node.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Predict.");
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return Forward(input);
    }

    /// <summary>
    /// Trains the network on a single batch of data.
    /// </summary>
    /// <param name="input">The input node features.</param>
    /// <param name="expectedOutput">The expected output (labels).</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Train.");
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(true);
        }

        var predictions = Forward(input);

        var flattenedPredictions = predictions.ToVector();
        var flattenedExpected = expectedOutput.ToVector();

        LastLoss = _lossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        var outputGradients = _lossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);
        var gradOutput = Tensor<T>.FromVector(outputGradients);

        if (gradOutput.Shape.Length == 1 && predictions.Shape.Length > 1)
        {
            gradOutput = gradOutput.Reshape(predictions.Shape);
        }

        Backward(gradOutput);

        Vector<T> parameterGradients = GetParameterGradients();
        Vector<T> currentParameters = GetParameters();
        Vector<T> updatedParameters = _optimizer.UpdateParameters(currentParameters, parameterGradients);

        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Gets metadata about this model for serialization and identification.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GraphNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NetworkType"] = "NodeClassificationModel",
                ["InputFeatures"] = InputFeatures,
                ["NumClasses"] = NumClasses,
                ["HiddenDim"] = HiddenDim,
                ["NumLayers"] = NumLayers,
                ["DropoutRate"] = DropoutRate
            }
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(InputFeatures);
        writer.Write(NumClasses);
        writer.Write(HiddenDim);
        writer.Write(NumLayers);
        writer.Write(DropoutRate);

        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
    }

    /// <summary>
    /// Deserializes network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // InputFeatures
        _ = reader.ReadInt32(); // NumClasses
        _ = reader.ReadInt32(); // HiddenDim
        _ = reader.ReadInt32(); // NumLayers
        _ = reader.ReadDouble(); // DropoutRate

        _ = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        _ = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);
    }

    /// <summary>
    /// Creates a new instance of this network type for cloning or deserialization.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NodeClassificationModel<T>(
            architecture: Architecture,
            hiddenDim: HiddenDim,
            numLayers: NumLayers,
            dropoutRate: DropoutRate);
    }

    #endregion
}
