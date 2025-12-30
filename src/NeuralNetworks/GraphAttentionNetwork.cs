using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Graph Attention Network (GAT) that uses attention mechanisms to process graph-structured data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Graph Attention Networks introduce attention mechanisms to graph neural networks, allowing the model
/// to learn which neighbors are most important for each node. Unlike GCN which treats all neighbors equally,
/// GAT learns attention weights that determine how much each neighbor contributes to a node's representation.
/// </para>
/// <para><b>For Beginners:</b> GAT is like having a smart filter for your social network.
///
/// **How it works:**
/// - Each node looks at its neighbors and decides which ones are most important
/// - Important neighbors get more "attention" (higher weights)
/// - Less relevant neighbors get less attention
///
/// **Example - Movie Recommendations:**
/// - You're a node connected to movies you've watched
/// - Some movies better represent your taste than others
/// - GAT learns to pay more attention to movies that define your preferences
/// - Result: Better recommendations by focusing on what matters most
///
/// **Key Features:**
/// - **Multi-head attention**: Multiple attention "perspectives" for robustness
/// - **Dynamic weights**: Attention weights are learned, not fixed
/// - **Dropout support**: Prevents overfitting during training
/// - **Configurable heads**: Adjust number of attention heads for your task
///
/// **Architecture:**
/// The standard GAT architecture consists of:
/// 1. Multiple GAT layers with attention mechanisms
/// 2. Optional dropout between layers
/// 3. Final classification or regression head
///
/// **When to use GAT:**
/// - When some neighbors are more informative than others
/// - When you need interpretable importance scores
/// - For heterogeneous graphs where relationships vary in importance
/// - Citation networks, social networks, knowledge graphs
/// </para>
/// </remarks>
public class GraphAttentionNetwork<T> : NeuralNetworkBase<T>
{
    private static new readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Gets the number of attention heads used in each GAT layer.
    /// </summary>
    public int NumHeads { get; }

    /// <summary>
    /// Gets the dropout rate applied to attention coefficients during training.
    /// </summary>
    public double DropoutRate { get; }

    /// <summary>
    /// Gets the hidden dimension size for each layer.
    /// </summary>
    public int HiddenDim { get; }

    /// <summary>
    /// Gets the number of GAT layers in the network.
    /// </summary>
    public int NumLayers { get; }

    /// <summary>
    /// Cached adjacency matrix for forward/backward passes.
    /// </summary>
    private Tensor<T>? _cachedAdjacencyMatrix;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphAttentionNetwork{T}"/> class with specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="numHeads">Number of attention heads per layer (default: 8). Used only when creating default layers.</param>
    /// <param name="numLayers">Number of GAT layers (default: 2). Used only when creating default layers.</param>
    /// <param name="dropoutRate">Dropout rate for attention coefficients (default: 0.6).</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a GAT network:
    ///
    /// ```csharp
    /// // Create architecture for node classification
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional,
    ///     NeuralNetworkTaskType.MultiClassClassification,
    ///     NetworkComplexity.Simple,
    ///     inputSize: 1433,    // Cora has 1433 word features
    ///     outputSize: 7);     // 7 paper categories
    ///
    /// // Create GAT with default layers
    /// var gat = new GraphAttentionNetwork&lt;double&gt;(architecture);
    ///
    /// // Or create with custom layers by adding them to architecture
    /// var gatCustom = new GraphAttentionNetwork&lt;double&gt;(architectureWithCustomLayers);
    ///
    /// // Train on graph data
    /// gat.TrainOnGraph(nodeFeatures, adjacencyMatrix, labels, epochs: 200);
    /// ```
    /// </para>
    /// </remarks>
    public GraphAttentionNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int numHeads = 8,
        int numLayers = 2,
        double dropoutRate = 0.6,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture,
               lossFunction ?? new CrossEntropyLoss<T>(),
               maxGradNorm)
    {
        NumHeads = numHeads;
        DropoutRate = dropoutRate;
        HiddenDim = 64; // Default hidden dimension
        NumLayers = numLayers;

        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraphAttentionLayers(
                Architecture, NumHeads, NumLayers, DropoutRate));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network with node features and adjacency matrix.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor of shape [batchSize, numNodes, inputFeatures] or [numNodes, inputFeatures].</param>
    /// <param name="adjacencyMatrix">Adjacency matrix of shape [batchSize, numNodes, numNodes] or [numNodes, numNodes].</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    public Tensor<T> Forward(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;

        // Set adjacency matrix on all graph layers
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
        }

        // Forward through all layers
        Tensor<T> output = nodeFeatures;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Performs a backward pass through the network to calculate gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the network's output.</param>
    /// <returns>The gradient of the loss with respect to the network's input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            outputGradient = Layers[i].Backward(outputGradient);
        }

        return outputGradient;
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
    /// Trains the GAT network on graph-structured data.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor of shape [numNodes, inputFeatures].</param>
    /// <param name="adjacencyMatrix">Adjacency matrix of shape [numNodes, numNodes].</param>
    /// <param name="labels">Label tensor for supervised learning.</param>
    /// <param name="trainMask">Optional boolean mask indicating which nodes to train on.</param>
    /// <param name="epochs">Number of training epochs (default: 200).</param>
    /// <param name="learningRate">Learning rate for optimization (default: 0.005).</param>
    public void TrainOnGraph(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        Tensor<T> labels,
        bool[]? trainMask = null,
        int epochs = 200,
        double learningRate = 0.005)
    {
        var lr = NumOps.FromDouble(learningRate);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Set all layers to training mode
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            // Forward pass
            var output = Forward(nodeFeatures, adjacencyMatrix);

            // Compute loss gradient
            var gradOutput = ComputeLossGradient(output, labels, trainMask);

            // Backward pass
            Backward(gradOutput);

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
    }

    /// <summary>
    /// Computes the gradient of the cross-entropy loss.
    /// </summary>
    private Tensor<T> ComputeLossGradient(Tensor<T> predictions, Tensor<T> labels, bool[]? mask)
    {
        var gradient = new Tensor<T>(predictions.Shape);
        int numNodes = predictions.Shape[0];
        int numClasses = predictions.Shape[1];
        int count = 0;

        // Count training nodes
        for (int i = 0; i < numNodes; i++)
        {
            if (mask == null || mask[i]) count++;
        }

        if (count == 0) return gradient;

        var scale = NumOps.Divide(NumOps.One, NumOps.FromDouble(count));

        for (int i = 0; i < numNodes; i++)
        {
            if (mask != null && !mask[i]) continue;

            // Compute softmax probabilities
            var maxLogit = NumOps.MinValue;
            for (int c = 0; c < numClasses; c++)
            {
                if (NumOps.GreaterThan(predictions[i, c], maxLogit))
                {
                    maxLogit = predictions[i, c];
                }
            }

            var sumExp = NumOps.Zero;
            var probs = new T[numClasses];
            for (int c = 0; c < numClasses; c++)
            {
                probs[c] = NumOps.Exp(NumOps.Subtract(predictions[i, c], maxLogit));
                sumExp = NumOps.Add(sumExp, probs[c]);
            }

            // Gradient = (softmax - label) / count
            for (int c = 0; c < numClasses; c++)
            {
                var prob = NumOps.Divide(probs[c], sumExp);
                gradient[i, c] = NumOps.Multiply(scale, NumOps.Subtract(prob, labels[i, c]));
            }
        }

        return gradient;
    }

    /// <summary>
    /// Gets attention weights from all GAT layers for interpretability.
    /// </summary>
    /// <returns>List of attention weight tensors (currently returns nulls as implementation is pending).</returns>
    /// <remarks>
    /// <para><b>Note:</b> This method is a placeholder. Full attention coefficient retrieval
    /// requires exposing internal state from GraphAttentionLayer, which will be added in a future update.</para>
    /// </remarks>
    public List<Tensor<T>?> GetAttentionWeights()
    {
        var attentions = new List<Tensor<T>?>();

        foreach (var layer in Layers)
        {
            if (layer is GraphAttentionLayer<T>)
            {
                // Note: GraphAttentionLayer stores attention coefficients internally
                // but does not expose them via a public method yet
                attentions.Add(null);
            }
        }

        return attentions;
    }

    /// <summary>
    /// Evaluates the model on test data and returns accuracy.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <param name="adjacencyMatrix">Adjacency matrix.</param>
    /// <param name="labels">Ground truth labels.</param>
    /// <param name="testMask">Boolean mask for test nodes.</param>
    /// <returns>Classification accuracy on test nodes.</returns>
    public double Evaluate(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        Tensor<T> labels,
        bool[] testMask)
    {
        // Set to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        var predictions = Forward(nodeFeatures, adjacencyMatrix);
        int correct = 0;
        int total = 0;

        int numNodes = predictions.Shape[0];
        int numClasses = predictions.Shape[1];

        for (int i = 0; i < numNodes; i++)
        {
            if (!testMask[i]) continue;

            // Find predicted class
            int predClass = 0;
            var maxProb = predictions[i, 0];
            for (int c = 1; c < numClasses; c++)
            {
                if (NumOps.GreaterThan(predictions[i, c], maxProb))
                {
                    maxProb = predictions[i, c];
                    predClass = c;
                }
            }

            // Find true class
            int trueClass = 0;
            for (int c = 0; c < numClasses; c++)
            {
                if (NumOps.GreaterThan(labels[i, c], NumOps.Zero))
                {
                    trueClass = c;
                    break;
                }
            }

            if (predClass == trueClass) correct++;
            total++;
        }

        return total > 0 ? (double)correct / total : 0.0;
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the network.
    /// </summary>
    public new int GetParameterCount()
    {
        int count = 0;
        foreach (var layer in Layers)
        {
            count += layer.ParameterCount;
        }
        return count;
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

    #region LoRA Fine-Tuning Support

    /// <summary>
    /// Gets whether LoRA fine-tuning is currently enabled.
    /// </summary>
    public bool IsLoRAEnabled { get; private set; }

    /// <summary>
    /// Gets the LoRA rank when LoRA is enabled.
    /// </summary>
    public int LoRARank { get; private set; }

    /// <summary>
    /// Enables LoRA (Low-Rank Adaptation) fine-tuning for parameter-efficient training.
    /// </summary>
    /// <param name="rank">The rank of the LoRA decomposition (default: 8).</param>
    /// <param name="alpha">The LoRA scaling factor (default: same as rank).</param>
    /// <param name="freezeBaseLayers">Whether to freeze base layer parameters (default: true).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> LoRA allows you to fine-tune the GAT network with far fewer
    /// trainable parameters:
    ///
    /// ```csharp
    /// // Create and pre-train a GAT network
    /// var gat = new GraphAttentionNetwork&lt;double&gt;(128, 64, 7, numHeads: 8);
    /// gat.TrainOnGraph(features, adjacency, labels, epochs: 200);
    ///
    /// // Enable LoRA for efficient fine-tuning on new task
    /// gat.EnableLoRAFineTuning(rank: 8, alpha: 16);
    ///
    /// // Now only ~4% of parameters are trainable!
    /// Console.WriteLine($"LoRA parameters: {gat.GetLoRAParameterCount()}");
    /// Console.WriteLine($"Total parameters: {gat.GetParameterCount()}");
    ///
    /// // Fine-tune on new data
    /// gat.TrainOnGraph(newFeatures, newAdjacency, newLabels, epochs: 50);
    ///
    /// // Optionally merge LoRA weights for deployment
    /// gat.MergeLoRAWeights();
    /// ```
    /// </para>
    /// </remarks>
    public void EnableLoRAFineTuning(int rank = 8, double alpha = -1, bool freezeBaseLayers = true)
    {
        if (IsLoRAEnabled)
        {
            throw new InvalidOperationException(
                "LoRA is already enabled. Call DisableLoRA() first to reconfigure.");
        }

        LoRARank = rank;
        var newLayers = new List<ILayer<T>>();

        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];

            // Wrap graph layers with LoRA adapters
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                var loraAdapter = new GraphConvolutionalLoRAAdapter<T>(
                    layer, rank, alpha, freezeBaseLayers);
                newLayers.Add(loraAdapter);
            }
            else
            {
                // Keep non-graph layers as-is
                newLayers.Add(layer);
            }
        }

        Layers.Clear();
        Layers.AddRange(newLayers);
        IsLoRAEnabled = true;
    }

    /// <summary>
    /// Disables LoRA fine-tuning and restores original layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This removes the LoRA adapters and restores the original base layers.
    /// Any LoRA adaptations that were not merged will be lost.
    /// </para>
    /// </remarks>
    public void DisableLoRA()
    {
        if (!IsLoRAEnabled)
        {
            return;
        }

        var newLayers = new List<ILayer<T>>();

        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLoRAAdapter<T> loraAdapter)
            {
                newLayers.Add(loraAdapter.BaseLayer);
            }
            else
            {
                newLayers.Add(layer);
            }
        }

        Layers.Clear();
        Layers.AddRange(newLayers);
        IsLoRAEnabled = false;
        LoRARank = 0;
    }

    /// <summary>
    /// Merges LoRA weights into the base layers and disables LoRA mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After fine-tuning with LoRA, you can "bake in" the learned
    /// adaptations to create a standard network for deployment:
    ///
    /// - Before merge: Forward pass requires computing both base and LoRA outputs
    /// - After merge: Single forward pass through merged layers (faster)
    ///
    /// This is useful when deploying the fine-tuned model to production where you want
    /// maximum inference speed and don't need to track LoRA parameters separately.
    /// </para>
    /// </remarks>
    public void MergeLoRAWeights()
    {
        if (!IsLoRAEnabled)
        {
            throw new InvalidOperationException(
                "LoRA is not enabled. Nothing to merge.");
        }

        var newLayers = new List<ILayer<T>>();

        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLoRAAdapter<T> loraAdapter)
            {
                var mergedLayer = loraAdapter.MergeToOriginalLayer();
                newLayers.Add(mergedLayer);
            }
            else
            {
                newLayers.Add(layer);
            }
        }

        Layers.Clear();
        Layers.AddRange(newLayers);
        IsLoRAEnabled = false;
        LoRARank = 0;
    }

    /// <summary>
    /// Gets the number of trainable LoRA parameters when LoRA is enabled.
    /// </summary>
    /// <returns>The count of LoRA parameters, or 0 if LoRA is not enabled.</returns>
    public int GetLoRAParameterCount()
    {
        if (!IsLoRAEnabled)
        {
            return 0;
        }

        int count = 0;
        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLoRAAdapter<T> loraAdapter)
            {
                count += loraAdapter.LoRALayer.ParameterCount;
            }
        }
        return count;
    }

    /// <summary>
    /// Gets the percentage of parameters that are trainable when using LoRA.
    /// </summary>
    /// <returns>The percentage of trainable parameters (0-100).</returns>
    public double GetLoRATrainablePercentage()
    {
        if (!IsLoRAEnabled)
        {
            return 100.0;
        }

        int loraParams = GetLoRAParameterCount();
        int totalParams = GetParameterCount();

        return totalParams > 0 ? (100.0 * loraParams / totalParams) : 0.0;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Makes a prediction using the trained network.
    /// </summary>
    /// <param name="input">The input tensor containing node features.</param>
    /// <returns>The prediction tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method for using a trained GAT network.
    /// Pass in node features and get predictions back. For classification, the output
    /// will be class probabilities for each node.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Auto-create adjacency matrix if not set (assumes fully-connected graph)
        // This supports flexible input without requiring explicit graph structure
        if (_cachedAdjacencyMatrix == null)
        {
            // Determine number of nodes from input shape
            // Input is typically [numNodes, featureDim] or [batch, numNodes, featureDim]
            int numNodes = input.Rank >= 2 ? input.Shape[input.Rank - 2] : input.Shape[0];

            // Create fully-connected adjacency matrix (all nodes connected to all)
            _cachedAdjacencyMatrix = new Tensor<T>([numNodes, numNodes]);
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    _cachedAdjacencyMatrix.SetFlat(i * numNodes + j, NumOps.One);
                }
            }
        }

        // Set all layers to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return Forward(input, _cachedAdjacencyMatrix);
    }

    /// <summary>
    /// Sets the adjacency matrix for graph operations.
    /// </summary>
    /// <param name="adjacencyMatrix">The adjacency matrix defining graph structure.</param>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;
    }

    /// <summary>
    /// Trains the network on a single batch of data.
    /// </summary>
    /// <param name="input">The input node features.</param>
    /// <param name="expectedOutput">The expected output (labels).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method performs one training step.
    /// For full training, call TrainOnGraph which handles multiple epochs and
    /// adjacency matrix setup.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_cachedAdjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Train.");
        }

        // Set all layers to training mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(true);
        }

        // Forward pass
        var predictions = Forward(input, _cachedAdjacencyMatrix);

        // Flatten tensors for loss function (which works on vectors)
        var flattenedPredictions = predictions.ToVector();
        var flattenedExpected = expectedOutput.ToVector();

        // Compute loss
        LastLoss = _lossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        // Compute loss gradient
        var outputGradients = _lossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);
        var gradOutput = Tensor<T>.FromVector(outputGradients);

        // Reshape gradient back to tensor shape if needed
        if (gradOutput.Shape.Length == 1 && predictions.Shape.Length > 1)
        {
            gradOutput = gradOutput.Reshape(predictions.Shape);
        }

        // Backward pass through all layers
        Backward(gradOutput);

        // Get parameter gradients for all trainable layers and update
        Vector<T> parameterGradients = GetParameterGradients();

        // Get current parameters
        Vector<T> currentParameters = GetParameters();

        // Update parameters using the optimizer
        Vector<T> updatedParameters = _optimizer.UpdateParameters(currentParameters, parameterGradients);

        // Apply updated parameters
        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Gets metadata about this model for serialization and identification.
    /// </summary>
    /// <returns>Model metadata including type and configuration.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GraphNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NetworkType"] = "GraphAttentionNetwork",
                ["NumHeads"] = NumHeads,
                ["HiddenDim"] = HiddenDim,
                ["NumLayers"] = NumLayers,
                ["DropoutRate"] = DropoutRate,
                ["IsLoRAEnabled"] = IsLoRAEnabled,
                ["LoRARank"] = LoRARank
            }
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Serialize GAT-specific configuration
        writer.Write(NumHeads);
        writer.Write(HiddenDim);
        writer.Write(NumLayers);
        writer.Write(DropoutRate);
        writer.Write(IsLoRAEnabled);
        writer.Write(LoRARank);

        // Serialize loss function and optimizer
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
    }

    /// <summary>
    /// Deserializes network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Note: The readonly fields are set in constructor, so we just read and discard
        // to maintain stream position. For full deserialization, use Load method.
        var numHeads = reader.ReadInt32();
        var hiddenDim = reader.ReadInt32();
        var numLayers = reader.ReadInt32();
        var dropoutRate = reader.ReadDouble();
        var isLoRAEnabled = reader.ReadBoolean();
        var loraRank = reader.ReadInt32();

        // Deserialize loss function and optimizer
        _ = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        _ = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);
    }

    /// <summary>
    /// Creates a new instance of this network type for cloning or deserialization.
    /// </summary>
    /// <returns>A new GraphAttentionNetwork instance.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GraphAttentionNetwork<T>(
            architecture: Architecture,
            numHeads: NumHeads,
            numLayers: NumLayers,
            dropoutRate: DropoutRate);
    }

    #endregion
}
