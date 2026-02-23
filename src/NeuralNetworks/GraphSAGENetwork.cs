using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a GraphSAGE (Graph Sample and Aggregate) Network for inductive learning on graphs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// GraphSAGE, introduced by Hamilton et al. (2017), is designed for inductive learning on graphs.
/// Unlike transductive methods that require all nodes during training, GraphSAGE learns aggregator
/// functions that can generalize to completely unseen nodes and even new graphs.
/// </para>
/// <para><b>For Beginners:</b> GraphSAGE learns how to combine neighbor information.
///
/// **How it works:**
/// - For each node, sample its neighbors
/// - Aggregate neighbor features using a learnable function
/// - Combine with node's own features
/// - Result: new representation that captures local structure
///
/// **Example - Social Network Recommendations:**
/// - New user joins the platform (unseen during training)
/// - GraphSAGE can still make recommendations by:
///   1. Looking at the new user's connections
///   2. Aggregating features from those connections
///   3. Generating a representation for the new user
///
/// **Key Features:**
/// - **Inductive**: Can generalize to new, unseen nodes
/// - **Scalable**: Uses sampling, not full neighborhoods
/// - **Flexible aggregators**: Mean, MaxPool, or Sum
/// - **L2 normalization**: Optional for stable training
///
/// **Aggregator Types:**
/// - **Mean**: Average of neighbor features (most common)
/// - **MaxPool**: Element-wise max (captures salient features)
/// - **Sum**: Sum of neighbor features (preserves structure)
///
/// **Architecture:**
/// 1. Multiple GraphSAGE layers with different aggregators
/// 2. Optional L2 normalization between layers
/// 3. Final classification or regression head
///
/// **When to use GraphSAGE:**
/// - When new nodes appear frequently (evolving graphs)
/// - When you need to generalize to new graphs
/// - For large-scale graphs where full-batch training is infeasible
/// - Social networks, recommendation systems, dynamic graphs
/// </para>
/// </remarks>
public class GraphSAGENetwork<T> : NeuralNetworkBase<T>
{
    private readonly GraphSAGEOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// Gets the aggregator type used in GraphSAGE layers.
    /// </summary>
    public SAGEAggregatorType AggregatorType { get; }

    /// <summary>
    /// Gets whether L2 normalization is applied after each layer.
    /// </summary>
    public bool Normalize { get; }

    /// <summary>
    /// Gets the hidden dimension size for each layer.
    /// </summary>
    public int HiddenDim { get; }

    /// <summary>
    /// Gets the number of GraphSAGE layers in the network.
    /// </summary>
    public int NumLayers { get; }

    /// <summary>
    /// Gets the dropout rate applied during training.
    /// </summary>
    public double DropoutRate { get; }

    /// <summary>
    /// Cached adjacency matrix for forward/backward passes.
    /// </summary>
    private Tensor<T>? _cachedAdjacencyMatrix;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphSAGENetwork{T}"/> class with specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="aggregatorType">Type of aggregation function (default: Mean). Used only when creating default layers.</param>
    /// <param name="numLayers">Number of GraphSAGE layers (default: 2). Used only when creating default layers.</param>
    /// <param name="normalize">Whether to apply L2 normalization (default: true). Used only when creating default layers.</param>
    /// <param name="dropoutRate">Dropout rate applied during training (default: 0.0).</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a GraphSAGE network:
    ///
    /// ```csharp
    /// // Create architecture for node classification
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional,
    ///     NeuralNetworkTaskType.MultiClassClassification,
    ///     NetworkComplexity.Simple,
    ///     inputSize: 128,     // User profile features
    ///     outputSize: 5);     // 5 user categories
    ///
    /// // Create GraphSAGE with default layers
    /// var sage = new GraphSAGENetwork&lt;double&gt;(architecture);
    ///
    /// // Or create with custom layers by adding them to architecture:
    /// architecture.Layers.Add(new GraphSAGELayer&lt;double&gt;(...));
    /// var sageCustom = new GraphSAGENetwork&lt;double&gt;(architecture);
    ///
    /// // Train on graph data
    /// sage.TrainOnGraph(nodeFeatures, adjacencyMatrix, labels, epochs: 200);
    /// ```
    /// </para>
    /// </remarks>
    public GraphSAGENetwork(
        NeuralNetworkArchitecture<T> architecture,
        SAGEAggregatorType aggregatorType = SAGEAggregatorType.Mean,
        int numLayers = 2,
        bool normalize = true,
        double dropoutRate = 0.0,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        GraphSAGEOptions? options = null)
        : base(architecture,
               lossFunction ?? new CrossEntropyLoss<T>(),
               maxGradNorm)
    {
        _options = options ?? new GraphSAGEOptions();
        Options = _options;
        AggregatorType = aggregatorType;
        Normalize = normalize;
        HiddenDim = 64; // Default hidden dimension
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
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraphSAGELayers(
                Architecture, AggregatorType, NumLayers, Normalize));
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
    /// Trains the GraphSAGE network on graph-structured data.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor of shape [numNodes, inputFeatures].</param>
    /// <param name="adjacencyMatrix">Adjacency matrix of shape [numNodes, numNodes].</param>
    /// <param name="labels">Label tensor for supervised learning.</param>
    /// <param name="trainMask">Optional boolean mask indicating which nodes to train on.</param>
    /// <param name="epochs">Number of training epochs (default: 200).</param>
    /// <param name="learningRate">Learning rate for optimization (default: 0.01).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training GraphSAGE on graph data:
    ///
    /// GraphSAGE learns to aggregate neighbor information through training.
    /// The aggregation functions become better at combining relevant features
    /// to produce informative node representations.
    ///
    /// **Training tips:**
    /// - Use Mean aggregator for most tasks (stable, effective)
    /// - Use MaxPool for graphs where individual features are important
    /// - L2 normalization helps with training stability
    /// - Higher learning rates (0.01) often work well for GraphSAGE
    /// </para>
    /// </remarks>
    public void TrainOnGraph(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        Tensor<T> labels,
        bool[]? trainMask = null,
        int epochs = 200,
        double learningRate = 0.01)
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
    /// Performs mini-batch training using neighbor sampling for scalability.
    /// </summary>
    /// <param name="nodeFeatures">Full node feature tensor.</param>
    /// <param name="adjacencyMatrix">Full adjacency matrix.</param>
    /// <param name="labels">Label tensor for supervised learning.</param>
    /// <param name="trainIndices">Indices of training nodes.</param>
    /// <param name="batchSize">Number of nodes per batch (default: 512).</param>
    /// <param name="epochs">Number of training epochs (default: 200).</param>
    /// <param name="learningRate">Learning rate (default: 0.01).</param>
    /// <param name="numSamples">Number of neighbors to sample per layer (default: 25).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Mini-batch training for large graphs:
    ///
    /// For very large graphs, training on all nodes at once is infeasible.
    /// Mini-batch training with neighbor sampling makes GraphSAGE scalable:
    ///
    /// **How it works:**
    /// 1. Sample a batch of target nodes
    /// 2. For each target, sample a subset of neighbors
    /// 3. Compute representations only for sampled subgraph
    /// 4. Update model parameters
    ///
    /// **Benefits:**
    /// - Constant memory usage regardless of graph size
    /// - Can train on graphs with millions of nodes
    /// - Provides regularization through random sampling
    /// </para>
    /// </remarks>
    public void TrainMiniBatch(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        Tensor<T> labels,
        int[] trainIndices,
        int batchSize = 512,
        int epochs = 200,
        double learningRate = 0.01,
        int numSamples = 25)
    {
        var lr = NumOps.FromDouble(learningRate);
        var random = RandomHelper.CreateSeededRandom(42);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Shuffle training indices
            var shuffled = trainIndices.OrderBy(_ => random.Next()).ToArray();

            for (int batchStart = 0; batchStart < shuffled.Length; batchStart += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, shuffled.Length - batchStart);
                var batchIndices = shuffled.Skip(batchStart).Take(actualBatchSize).ToArray();

                // Sample subgraph for this batch
                var (sampledFeatures, sampledAdj, sampledLabels) = SampleSubgraph(
                    nodeFeatures, adjacencyMatrix, labels, batchIndices, numSamples, random);

                // Set all layers to training mode
                foreach (var layer in Layers)
                {
                    layer.SetTrainingMode(true);
                }

                // Forward pass on sampled subgraph
                var output = Forward(sampledFeatures, sampledAdj);

                // Compute loss gradient
                var gradOutput = ComputeLossGradient(output, sampledLabels, null);

                // Backward pass
                Backward(gradOutput);

                // Update parameters
                foreach (var layer in Layers)
                {
                    layer.UpdateParameters(lr);
                }
            }
        }

        // Set layers back to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Samples a subgraph around target nodes for mini-batch training.
    /// </summary>
    private (Tensor<T> features, Tensor<T> adj, Tensor<T> labels) SampleSubgraph(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        Tensor<T> labels,
        int[] targetIndices,
        int numSamples,
        Random random)
    {
        int numFeatures = nodeFeatures.Shape[1];
        int numClasses = labels.Shape[1];
        int numNodes = nodeFeatures.Shape[0];

        // Collect all nodes needed (targets + sampled neighbors)
        var nodesNeeded = new HashSet<int>(targetIndices);

        // Sample neighbors for each layer
        foreach (var target in targetIndices)
        {
            var neighbors = new List<int>();
            for (int j = 0; j < numNodes; j++)
            {
                if (NumOps.GreaterThan(adjacencyMatrix[target, j], NumOps.Zero))
                {
                    neighbors.Add(j);
                }
            }

            // Random sample of neighbors
            var sampled = neighbors.OrderBy(_ => random.Next()).Take(numSamples);
            foreach (var n in sampled)
            {
                nodesNeeded.Add(n);
            }
        }

        var nodeList = nodesNeeded.ToList();
        var nodeMap = new Dictionary<int, int>();
        for (int i = 0; i < nodeList.Count; i++)
        {
            nodeMap[nodeList[i]] = i;
        }

        int subgraphSize = nodeList.Count;

        // Extract features for subgraph
        var sampledFeatures = new Tensor<T>([subgraphSize, numFeatures]);
        for (int i = 0; i < subgraphSize; i++)
        {
            int originalIdx = nodeList[i];
            for (int f = 0; f < numFeatures; f++)
            {
                sampledFeatures[i, f] = nodeFeatures[originalIdx, f];
            }
        }

        // Extract adjacency for subgraph
        var sampledAdj = new Tensor<T>([subgraphSize, subgraphSize]);
        for (int i = 0; i < subgraphSize; i++)
        {
            for (int j = 0; j < subgraphSize; j++)
            {
                int origI = nodeList[i];
                int origJ = nodeList[j];
                sampledAdj[i, j] = adjacencyMatrix[origI, origJ];
            }
        }

        // Extract labels (only for target nodes at the beginning of the list)
        var sampledLabels = new Tensor<T>([targetIndices.Length, numClasses]);
        for (int i = 0; i < targetIndices.Length; i++)
        {
            int mappedIdx = nodeMap[targetIndices[i]];
            for (int c = 0; c < numClasses; c++)
            {
                sampledLabels[i, c] = labels[targetIndices[i], c];
            }
        }

        return (sampledFeatures, sampledAdj, sampledLabels);
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
    /// Generates node embeddings using the trained network.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <param name="adjacencyMatrix">Adjacency matrix.</param>
    /// <returns>Node embedding tensor from the second-to-last layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Node embeddings are useful for:
    ///
    /// - **Clustering**: Group similar nodes together
    /// - **Visualization**: Plot nodes in 2D/3D using t-SNE or UMAP
    /// - **Transfer learning**: Use embeddings as features for other tasks
    /// - **Similarity search**: Find similar nodes efficiently
    /// </para>
    /// </remarks>
    public Tensor<T> GetNodeEmbeddings(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
    {
        // Set to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        // Set adjacency matrix on all graph layers
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
        }

        // Forward through all but the last layer to get embeddings
        Tensor<T> output = nodeFeatures;
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            output = Layers[i].Forward(output);
        }

        return output;
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
    /// Enables LoRA fine-tuning for parameter-efficient training.
    /// </summary>
    public void EnableLoRAFineTuning(int rank = 8, double alpha = -1, bool freezeBaseLayers = true)
    {
        if (IsLoRAEnabled)
        {
            throw new InvalidOperationException("LoRA is already enabled. Call DisableLoRA() first to reconfigure.");
        }

        LoRARank = rank;
        var newLayers = new List<ILayer<T>>();

        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T>)
            {
                var loraAdapter = new GraphConvolutionalLoRAAdapter<T>(layer, rank, alpha, freezeBaseLayers);
                newLayers.Add(loraAdapter);
            }
            else
            {
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
    public void DisableLoRA()
    {
        if (!IsLoRAEnabled) return;

        var newLayers = new List<ILayer<T>>();
        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLoRAAdapter<T> loraAdapter)
                newLayers.Add(loraAdapter.BaseLayer);
            else
                newLayers.Add(layer);
        }

        Layers.Clear();
        Layers.AddRange(newLayers);
        IsLoRAEnabled = false;
        LoRARank = 0;
    }

    /// <summary>
    /// Merges LoRA weights into the base layers and disables LoRA mode.
    /// </summary>
    public void MergeLoRAWeights()
    {
        if (!IsLoRAEnabled)
            throw new InvalidOperationException("LoRA is not enabled. Nothing to merge.");

        var newLayers = new List<ILayer<T>>();
        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLoRAAdapter<T> loraAdapter)
                newLayers.Add(loraAdapter.MergeToOriginalLayer());
            else
                newLayers.Add(layer);
        }

        Layers.Clear();
        Layers.AddRange(newLayers);
        IsLoRAEnabled = false;
        LoRARank = 0;
    }

    /// <summary>
    /// Gets the number of trainable LoRA parameters.
    /// </summary>
    public int GetLoRAParameterCount()
    {
        if (!IsLoRAEnabled) return 0;

        int count = 0;
        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLoRAAdapter<T> loraAdapter)
                count += loraAdapter.LoRALayer.ParameterCount;
        }
        return count;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Makes a prediction using the trained network.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        if (_cachedAdjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Predict.");
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return Forward(input, _cachedAdjacencyMatrix);
    }

    /// <summary>
    /// Sets the adjacency matrix for graph operations.
    /// </summary>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;
    }

    /// <summary>
    /// Trains the network on a single batch of data.
    /// </summary>
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
    /// Gets metadata about this model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GraphNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NetworkType"] = "GraphSAGENetwork",
                ["HiddenDim"] = HiddenDim,
                ["NumLayers"] = NumLayers,
                ["AggregatorType"] = AggregatorType.ToString(),
                ["DropoutRate"] = DropoutRate,
                ["IsLoRAEnabled"] = IsLoRAEnabled,
                ["LoRARank"] = LoRARank
            }
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(HiddenDim);
        writer.Write(NumLayers);
        writer.Write((int)AggregatorType);
        writer.Write(DropoutRate);
        writer.Write(IsLoRAEnabled);
        writer.Write(LoRARank);
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
    }

    /// <summary>
    /// Deserializes network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // HiddenDim
        _ = reader.ReadInt32(); // NumLayers
        _ = (SAGEAggregatorType)reader.ReadInt32();
        _ = reader.ReadDouble(); // DropoutRate
        _ = reader.ReadBoolean(); // IsLoRAEnabled
        _ = reader.ReadInt32(); // LoRARank
        _ = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        _ = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);
    }

    /// <summary>
    /// Creates a new instance of this network type.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GraphSAGENetwork<T>(
            architecture: Architecture,
            aggregatorType: AggregatorType,
            numLayers: NumLayers,
            normalize: Normalize,
            dropoutRate: DropoutRate);
    }

    #endregion
}
