using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LoRA.Adapters;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;

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
/// <example>
/// <code>
/// var options = new GraphAttentionNetworkOptions { NodeFeatureSize = 16, HiddenSize = 64, NumHeads = 8 };
/// var model = new GraphAttentionNetwork&lt;float&gt;(options);
/// var nodeFeatures = Tensor&lt;float&gt;.Random(new[] { 10, 16 });
/// var output = model.Predict(nodeFeatures);
/// </code>
/// </example>
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Graph Attention Networks", "https://arxiv.org/abs/1710.10903", Year = 2018, Authors = "Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio")]
public class GraphAttentionNetwork<T> : NeuralNetworkBase<T>
{
    private readonly GraphAttentionNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    [Scratch]
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
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public GraphAttentionNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 128,
            outputSize: 7))
    {
    }

    public GraphAttentionNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int numHeads = 8,
        int numLayers = 2,
        double dropoutRate = 0.6,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        ILearningRateScheduler? learningRateScheduler = null,
        GraphAttentionNetworkOptions? options = null)
        : base(architecture,
               lossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
               maxGradNorm)
    {
        _options = options ?? new GraphAttentionNetworkOptions();
        Options = _options;
        NumHeads = numHeads;
        DropoutRate = dropoutRate;
        HiddenDim = 64; // Default hidden dimension
        NumLayers = numLayers;

        // The graph-aware layer builder intentionally leaves the per-node prediction
        // head as logits (a global ActivationLayer would normalize nodes together).
        // Fuse the paper's final softmax with cross-entropy so normalization remains
        // per node and numerically stable.
        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
        var adamOpts = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            // Veličković et al. train the Cora/Citeseer GAT with Adam at 0.005.
            // Their early stopping is validation-driven; an unconditional per-batch
            // exponential decay is not part of the paper and nearly freezes the
            // optimizer by the end of a 200-step fit.
            InitialLearningRate = 0.005,
            LearningRateScheduler = learningRateScheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        };
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, adamOpts);

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
            // Graph networks need per-node activation, not global softmax.
            // Filter out trailing SoftmaxActivation — it applies softmax over all
            // (nodes × classes) elements, which is incorrect for multi-node graph output.
            foreach (var layer in LayerHelper<T>.CreateDefaultGraphAttentionLayers(
                Architecture, NumHeads, NumLayers, DropoutRate))
            {
                if (layer is ActivationLayer<T>)
                    continue;
                Layers.Add(layer);
            }
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

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
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
        if (learningRate <= 0 || double.IsNaN(learningRate) || double.IsInfinity(learningRate))
            throw new ArgumentOutOfRangeException(nameof(learningRate));

        SetAdjacencyMatrix(adjacencyMatrix);

        // The mask is applied to the LOSS, not to the labels.
        //
        // Zeroing a held-out node's one-hot row does not hold it out. Under this network's default
        // CrossEntropyWithLogitsLoss the gradient of an all-zero target row is
        // softmax(logits) - 0 = softmax(logits), which is non-zero: it drives every logit of every
        // held-out node downward on every step. That is worse than no signal, because it is a
        // consistent wrong one. MaskedRowLoss instead selects the training rows, so an excluded node
        // is not on the tape at all and its gradient is exactly zero -- matching what the older
        // hand-written ComputeLossGradient achieved with its `continue`.
        //
        // The forward pass still runs over the whole graph, which is the point of transductive
        // training: attention over a held-out node's neighbourhood is how the training nodes see it.
        Tensor<T> trainingLabels = labels;
        LossFunctionBase<T>? maskedLoss = null;
        if (trainMask is not null)
        {
            if (labels.Rank < 2 || trainMask.Length != labels.Shape[0])
                throw new ArgumentException("The training mask must contain one entry per labeled node.", nameof(trainMask));

            if (_lossFunction is not LossFunctionBase<T> maskableLoss)
            {
                throw new InvalidOperationException(
                    "Masked graph training needs a loss derived from LossFunctionBase<T> so the held-out " +
                    $"rows can be excluded from the tape; '{_lossFunction.GetType().Name}' does not derive " +
                    "from it. Train without a mask, or supply a loss that does.");
            }

            maskedLoss = new MaskedRowLoss<T>(maskableLoss, trainMask);
        }

        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> trainingOptimizer =
            Math.Abs(learningRate - 0.005) < 1e-15
                ? _optimizer
                : new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                    this,
                    new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                    {
                        InitialLearningRate = learningRate,
                        UseAMSGrad = false,
                    });

        // Swapped for the duration of the run and restored unconditionally: TrainWithTape reads the
        // base LossFunction field, and leaving a mask-bound loss installed afterwards would silently
        // apply this call's node split to every later Predict-time loss evaluation and to any
        // subsequent training on a different mask.
        var originalLoss = LossFunction;
        if (maskedLoss is not null)
        {
            LossFunction = maskedLoss;
        }

        try
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                SetTrainingMode(true);
                try
                {
                    TrainWithTape(nodeFeatures, trainingLabels, trainingOptimizer);
                }
                finally
                {
                    SetTrainingMode(false);
                }
            }
        }
        finally
        {
            LossFunction = originalLoss;
        }
    }

    /// <summary>
    /// Computes the gradient of the cross-entropy loss.
    /// </summary>
    private Tensor<T> ComputeLossGradient(Tensor<T> predictions, Tensor<T> labels, bool[]? mask)
    {
        var gradient = new Tensor<T>(predictions._shape);
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
    public new long GetParameterCount()
    {
        int count = 0;
        foreach (var layer in Layers)
        {
            count += (int)layer.ParameterCount;
        }
        return count;
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
    /// // Result is available in the returned value
    /// // Result is available in the returned value
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
                count += (int)(loraAdapter.LoRALayer.ParameterCount);
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
        int totalParams = (int)GetParameterCount();

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
    /// will be class probabilities for each node. If no adjacency matrix has been set,
    /// a self-loop-only matrix is generated as the neutral fallback; call
    /// <see cref="SetAdjacencyMatrix"/> to supply the true graph.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> normalizedInput = NormalizeSingleNodeInput(input, out bool wasReshaped);
        var adjacencyMatrix = EnsureAdjacencyMatrix(normalizedInput);

        // Set all layers to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        var output = Forward(normalizedInput, adjacencyMatrix);
        return wasReshaped ? output.Reshape([output.Shape[^1]]) : output;
    }

    /// <summary>
    /// Sets the adjacency matrix for graph operations.
    /// </summary>
    /// <param name="adjacencyMatrix">The adjacency matrix defining graph structure (shape [numNodes, numNodes]).</param>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;
    }

    private Tensor<T> EnsureAdjacencyMatrix(Tensor<T> input)
    {
        if (input.Rank < 2)
        {
            throw new ArgumentException(
                $"Input must be at least rank 2 ([numNodes, featureDim] or [batch, numNodes, featureDim]). Got rank {input.Rank}.",
                nameof(input));
        }

        int numNodes = input.Shape[input.Rank - 2];

        if (_cachedAdjacencyMatrix != null)
        {
            if (_cachedAdjacencyMatrix.Shape.Length != 2 ||
                _cachedAdjacencyMatrix.Shape[0] != numNodes ||
                _cachedAdjacencyMatrix.Shape[1] != numNodes)
            {
                throw new ArgumentException(
                    $"Adjacency matrix shape [{string.Join(", ", _cachedAdjacencyMatrix._shape)}] does not match node count {numNodes}.",
                    nameof(_cachedAdjacencyMatrix));
            }

            return _cachedAdjacencyMatrix;
        }

        // A feature tensor contains no topology. Preserve each node through the
        // mandatory GAT self-loop without inventing edges between unrelated samples.
        // The previous all-ones fallback silently turned ordinary batched training
        // into one fully connected graph and mixed targets across nodes.
        var adjacencyMatrix = new Tensor<T>([numNodes, numNodes]);
        for (int i = 0; i < numNodes; i++)
            adjacencyMatrix.SetFlat(i * numNodes + i, NumOps.One);

        _cachedAdjacencyMatrix = adjacencyMatrix;
        return adjacencyMatrix;
    }

    private static Tensor<T> NormalizeSingleNodeInput(Tensor<T> input, out bool wasReshaped)
    {
        if (input.Rank == 1)
        {
            wasReshaped = true;
            return input.Reshape([1, input.Shape[0]]);
        }

        wasReshaped = false;
        return input;
    }

    /// <summary>
    /// Trains the network on a single batch of data.
    /// </summary>
    /// <param name="input">The input node features.</param>
    /// <param name="expectedOutput">The expected output (labels).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method performs one training step.
    /// For full training, call TrainOnGraph which handles multiple epochs and
    /// adjacency matrix setup. If no adjacency matrix has been set, a self-loop-only
    /// matrix is generated; provide an explicit matrix to train with graph edges.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Route through TrainWithTape so the gradient flows back through
        // the layer stack via GradientTape (the unified post-#1060
        // training path). The previous implementation called
        // GetParameterGradients() WITHOUT first running a backward pass
        // through layers — since Backward() was removed from ILayer<T>
        // by #b92c4d0dd, those gradients are always whatever stale value
        // the layer's _kernelsGradient field holds (zero on a fresh
        // network), which is why Training_ShouldChangeParameters and
        // GradientFlow_ShouldBeNonZeroAndFinite reported "no parameters
        // changed". ForwardForTraining is overridden below to set the
        // adjacency matrix on graph layers before the tape-recorded
        // forward pass.
        SetTrainingMode(true);
        try
        {
            Tensor<T> normalizedInput = NormalizeSingleNodeInput(input, out _);
            Tensor<T> normalizedExpected = NormalizeSingleNodeInput(expectedOutput, out _);
            TrainWithTape(normalizedInput, normalizedExpected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Computes the adjacency matrix and pushes it into every IGraphConvolutionLayer
    /// before the tape-recorded forward pass. Without this, graph layers see
    /// stale or empty adjacency from a previous unrelated call (or none at
    /// all), which makes their attention weights collapse and produces zero
    /// loss-gradient → zero parameter updates.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var adjacencyMatrix = EnsureAdjacencyMatrix(input);
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
        }
        return base.ForwardForTraining(input);
    }

    /// <summary>
    /// Gets the intermediate activations from each layer, ensuring adjacency is set for graph layers.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        Tensor<T> normalizedInput = NormalizeSingleNodeInput(input, out _);
        var adjacencyMatrix = EnsureAdjacencyMatrix(normalizedInput);

        // Set adjacency on all graph layers before calling Forward on each
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
        }

        var activations = new Dictionary<string, Tensor<T>>();
        var current = normalizedInput;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <summary>
    /// Gets metadata about this model for serialization and identification.
    /// </summary>
    /// <returns>Model metadata including type and configuration.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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
