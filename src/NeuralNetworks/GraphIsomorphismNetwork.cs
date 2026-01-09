using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Graph Isomorphism Network (GIN) for powerful graph representation learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Graph Isomorphism Networks (GIN), introduced by Xu et al. (2019), are provably as powerful as
/// the Weisfeiler-Lehman (WL) graph isomorphism test for distinguishing graph structures. GIN uses
/// sum aggregation with a learnable epsilon parameter and applies a multi-layer perceptron (MLP)
/// for powerful feature transformation.
/// </para>
/// <para><b>For Beginners:</b> GIN is optimal for structural graph understanding.
///
/// **How it works:**
/// - Sum neighbor features (preserves multiset information)
/// - Combine with self features using learnable weighting (1 + epsilon)
/// - Transform through a 2-layer MLP
/// - Result: maximally expressive graph representation
///
/// **Example - Chemical Structure Analysis:**
/// - Distinguishing molecules with subtle structural differences
/// - GIN can tell apart molecules that simpler GNNs confuse
/// - Critical for drug discovery where small differences matter
///
/// **Key Features:**
/// - **Provably powerful**: As expressive as WL test
/// - **Learnable epsilon**: Optimizes self vs neighbor weighting
/// - **Two-layer MLP**: Provides non-linear transformation capacity
/// - **Sum aggregation**: Preserves structural information
///
/// **Why GIN is powerful:**
/// - Mean/max pooling loses information (e.g., can't distinguish {1,1,1} from {1})
/// - Sum aggregation preserves multiset: {1,1,1} != {1}
/// - MLP can approximate complex functions
/// - Learnable epsilon finds optimal self-weighting
///
/// **Architecture:**
/// 1. Multiple GIN layers with sum aggregation
/// 2. Each layer has learnable epsilon and 2-layer MLP
/// 3. Optional graph-level readout for classification
///
/// **When to use GIN:**
/// - When structural differentiation is critical
/// - Molecular property prediction
/// - Chemical compound classification
/// - Any task where graph structure similarity matters
/// </para>
/// </remarks>
public class GraphIsomorphismNetwork<T> : NeuralNetworkBase<T>
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
    /// Gets whether epsilon is learnable in GIN layers.
    /// </summary>
    public bool LearnEpsilon { get; }

    /// <summary>
    /// Gets the initial epsilon value for GIN layers.
    /// </summary>
    public double InitialEpsilon { get; }

    /// <summary>
    /// Gets the hidden dimension size for MLP in each layer.
    /// </summary>
    public int MlpHiddenDim { get; }

    /// <summary>
    /// Gets the number of GIN layers in the network.
    /// </summary>
    public int NumLayers { get; }

    /// <summary>
    /// Cached adjacency matrix for forward/backward passes.
    /// </summary>
    private Tensor<T>? _cachedAdjacencyMatrix;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphIsomorphismNetwork{T}"/> class with specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="mlpHiddenDim">Hidden dimension for MLP within GIN layers (default: 64). Used only when creating default layers.</param>
    /// <param name="numLayers">Number of GIN layers (default: 5). Used only when creating default layers.</param>
    /// <param name="learnEpsilon">Whether to learn epsilon parameter (default: true). Used only when creating default layers.</param>
    /// <param name="initialEpsilon">Initial value for epsilon (default: 0.0). Used only when creating default layers.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a GIN network:
    ///
    /// ```csharp
    /// // Create architecture for molecular property prediction
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional,
    ///     NeuralNetworkTaskType.MultiClassClassification,
    ///     NetworkComplexity.Simple,
    ///     inputSize: 9,        // Atom features
    ///     outputSize: 2);      // Binary classification
    ///
    /// // Create GIN with default layers
    /// var gin = new GraphIsomorphismNetwork&lt;double&gt;(architecture);
    ///
    /// // Or create with custom layers by adding them to architecture
    /// var ginCustom = new GraphIsomorphismNetwork&lt;double&gt;(architectureWithCustomLayers);
    ///
    /// // Train on molecular graphs
    /// gin.TrainOnGraphs(molecules, adjacencyMatrices, labels, epochs: 100);
    /// ```
    /// </para>
    /// </remarks>
    public GraphIsomorphismNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int mlpHiddenDim = 64,
        int numLayers = 5,
        bool learnEpsilon = true,
        double initialEpsilon = 0.0,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture,
               lossFunction ?? new CrossEntropyLoss<T>(),
               maxGradNorm)
    {
        LearnEpsilon = learnEpsilon;
        InitialEpsilon = initialEpsilon;
        MlpHiddenDim = mlpHiddenDim;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraphIsomorphismLayers(
                Architecture, MlpHiddenDim, NumLayers, LearnEpsilon, InitialEpsilon));
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
    /// Trains the GIN network on a single graph with node classification.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor of shape [numNodes, inputFeatures].</param>
    /// <param name="adjacencyMatrix">Adjacency matrix of shape [numNodes, numNodes].</param>
    /// <param name="labels">Label tensor for supervised learning.</param>
    /// <param name="trainMask">Optional boolean mask indicating which nodes to train on.</param>
    /// <param name="epochs">Number of training epochs (default: 200).</param>
    /// <param name="learningRate">Learning rate for optimization (default: 0.01).</param>
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
    /// Trains the GIN network on multiple graphs for graph classification.
    /// </summary>
    /// <param name="graphs">List of graph node feature tensors.</param>
    /// <param name="adjacencyMatrices">List of adjacency matrices.</param>
    /// <param name="graphLabels">Labels for each graph.</param>
    /// <param name="epochs">Number of training epochs (default: 100).</param>
    /// <param name="learningRate">Learning rate for optimization (default: 0.01).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Graph classification with GIN:
    ///
    /// GIN is particularly effective for graph-level tasks like:
    /// - Molecular property prediction (e.g., toxicity, activity)
    /// - Social network classification
    /// - Document classification based on citation graphs
    ///
    /// **How graph classification works:**
    /// 1. Process each graph through GIN layers
    /// 2. Aggregate node features to get graph-level representation
    /// 3. Classify using the aggregated representation
    /// </para>
    /// </remarks>
    public void TrainOnGraphs(
        List<Tensor<T>> graphs,
        List<Tensor<T>> adjacencyMatrices,
        Tensor<T> graphLabels,
        int epochs = 100,
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

            // Train on each graph
            for (int g = 0; g < graphs.Count; g++)
            {
                var nodeFeatures = graphs[g];
                var adjMatrix = adjacencyMatrices[g];

                // Forward pass
                var nodeOutput = Forward(nodeFeatures, adjMatrix);

                // Graph-level readout (sum pooling)
                var graphRepresentation = SumReadout(nodeOutput);

                // Get label for this graph
                int numClasses = graphLabels.Shape[1];
                var graphLabel = new Tensor<T>([1, numClasses]);
                for (int c = 0; c < numClasses; c++)
                {
                    graphLabel[0, c] = graphLabels[g, c];
                }

                // Compute loss gradient
                var gradOutput = ComputeGraphLossGradient(graphRepresentation, graphLabel);

                // Distribute gradient back to nodes (reverse of sum readout)
                var nodeGradient = DistributeGradient(gradOutput, nodeOutput.Shape[0]);

                // Backward pass
                Backward(nodeGradient);

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
    /// Sum readout for graph-level representation.
    /// </summary>
    private Tensor<T> SumReadout(Tensor<T> nodeFeatures)
    {
        int numNodes = nodeFeatures.Shape[0];
        int numFeatures = nodeFeatures.Shape[1];

        var graphRep = new Tensor<T>([1, numFeatures]);
        for (int f = 0; f < numFeatures; f++)
        {
            var sum = NumOps.Zero;
            for (int n = 0; n < numNodes; n++)
            {
                sum = NumOps.Add(sum, nodeFeatures[n, f]);
            }
            graphRep[0, f] = sum;
        }

        return graphRep;
    }

    /// <summary>
    /// Distributes gradient from graph-level back to nodes.
    /// </summary>
    private Tensor<T> DistributeGradient(Tensor<T> graphGradient, int numNodes)
    {
        int numFeatures = graphGradient.Shape[1];
        var nodeGradient = new Tensor<T>([numNodes, numFeatures]);

        for (int n = 0; n < numNodes; n++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                nodeGradient[n, f] = graphGradient[0, f];
            }
        }

        return nodeGradient;
    }

    /// <summary>
    /// Computes the gradient of the cross-entropy loss for node classification.
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
    /// Computes the gradient of the cross-entropy loss for graph classification.
    /// </summary>
    private Tensor<T> ComputeGraphLossGradient(Tensor<T> predictions, Tensor<T> labels)
    {
        var gradient = new Tensor<T>(predictions.Shape);
        int numClasses = predictions.Shape[1];

        // Compute softmax probabilities
        var maxLogit = NumOps.MinValue;
        for (int c = 0; c < numClasses; c++)
        {
            if (NumOps.GreaterThan(predictions[0, c], maxLogit))
            {
                maxLogit = predictions[0, c];
            }
        }

        var sumExp = NumOps.Zero;
        var probs = new T[numClasses];
        for (int c = 0; c < numClasses; c++)
        {
            probs[c] = NumOps.Exp(NumOps.Subtract(predictions[0, c], maxLogit));
            sumExp = NumOps.Add(sumExp, probs[c]);
        }

        // Gradient = softmax - label
        for (int c = 0; c < numClasses; c++)
        {
            var prob = NumOps.Divide(probs[c], sumExp);
            gradient[0, c] = NumOps.Subtract(prob, labels[0, c]);
        }

        return gradient;
    }

    /// <summary>
    /// Evaluates the model on test data and returns accuracy for node classification.
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
    /// Evaluates the model on test graphs and returns accuracy for graph classification.
    /// </summary>
    /// <param name="graphs">List of graph node feature tensors.</param>
    /// <param name="adjacencyMatrices">List of adjacency matrices.</param>
    /// <param name="graphLabels">Ground truth labels for each graph.</param>
    /// <returns>Classification accuracy on test graphs.</returns>
    public double EvaluateGraphs(
        List<Tensor<T>> graphs,
        List<Tensor<T>> adjacencyMatrices,
        Tensor<T> graphLabels)
    {
        // Set to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        int correct = 0;
        int total = graphs.Count;
        int numClasses = graphLabels.Shape[1];

        for (int g = 0; g < graphs.Count; g++)
        {
            var nodeOutput = Forward(graphs[g], adjacencyMatrices[g]);
            var graphRep = SumReadout(nodeOutput);

            // Find predicted class
            int predClass = 0;
            var maxProb = graphRep[0, 0];
            for (int c = 1; c < numClasses; c++)
            {
                if (NumOps.GreaterThan(graphRep[0, c], maxProb))
                {
                    maxProb = graphRep[0, c];
                    predClass = c;
                }
            }

            // Find true class
            int trueClass = 0;
            for (int c = 0; c < numClasses; c++)
            {
                if (NumOps.GreaterThan(graphLabels[g, c], NumOps.Zero))
                {
                    trueClass = c;
                    break;
                }
            }

            if (predClass == trueClass) correct++;
        }

        return total > 0 ? (double)correct / total : 0.0;
    }

    /// <summary>
    /// Gets graph-level representations using sum, mean, and max pooling combined.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <param name="adjacencyMatrix">Adjacency matrix.</param>
    /// <returns>Graph-level representation combining multiple readout strategies.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Hierarchical graph representations:
    ///
    /// This method creates rich graph-level embeddings by:
    /// 1. Processing through all GIN layers
    /// 2. At each layer, computing sum, mean, and max of node features
    /// 3. Concatenating all layer representations
    ///
    /// This captures both local (early layers) and global (later layers) structure.
    /// </para>
    /// </remarks>
    public Tensor<T> GetGraphRepresentation(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
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

        // Collect representations from all layers
        var layerRepresentations = new List<Tensor<T>>();
        Tensor<T> current = nodeFeatures;

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
            layerRepresentations.Add(SumReadout(current));
        }

        // Concatenate all layer representations
        int totalFeatures = 0;
        foreach (var rep in layerRepresentations)
        {
            totalFeatures += rep.Shape[1];
        }

        var graphRep = new Tensor<T>([1, totalFeatures]);
        int offset = 0;
        foreach (var rep in layerRepresentations)
        {
            for (int f = 0; f < rep.Shape[1]; f++)
            {
                graphRep[0, offset + f] = rep[0, f];
            }
            offset += rep.Shape[1];
        }

        return graphRep;
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
                ["NetworkType"] = "GraphIsomorphismNetwork",
                ["MlpHiddenDim"] = MlpHiddenDim,
                ["NumLayers"] = NumLayers,
                ["InitialEpsilon"] = InitialEpsilon,
                ["LearnEpsilon"] = LearnEpsilon,
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
        writer.Write(MlpHiddenDim);
        writer.Write(NumLayers);
        writer.Write(InitialEpsilon);
        writer.Write(LearnEpsilon);
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
        _ = reader.ReadInt32(); // MlpHiddenDim
        _ = reader.ReadInt32(); // NumLayers
        _ = reader.ReadDouble(); // InitialEpsilon
        _ = reader.ReadBoolean(); // LearnEpsilon
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
        return new GraphIsomorphismNetwork<T>(
            architecture: Architecture,
            mlpHiddenDim: MlpHiddenDim,
            numLayers: NumLayers,
            learnEpsilon: LearnEpsilon,
            initialEpsilon: InitialEpsilon);
    }

    #endregion
}
