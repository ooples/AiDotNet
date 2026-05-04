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
/// <example>
/// <code>
/// var options = new GraphIsomorphismNetworkOptions { NodeFeatureSize = 16, HiddenSize = 64, NumLayers = 5 };
/// var model = new GraphIsomorphismNetwork&lt;float&gt;(options);
/// var nodeFeatures = Tensor&lt;float&gt;.Random(new[] { 20, 16 });
/// var output = model.Predict(nodeFeatures);
/// </code>
/// </example>
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("How Powerful are Graph Neural Networks?", "https://arxiv.org/abs/1810.00826", Year = 2019, Authors = "Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka")]
public class GraphIsomorphismNetwork<T> : NeuralNetworkBase<T>
{
    private readonly GraphIsomorphismNetworkOptions _options;

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
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public GraphIsomorphismNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 128,
            outputSize: 7))
    {
    }

    public GraphIsomorphismNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int mlpHiddenDim = 64,
        int numLayers = 5,
        bool learnEpsilon = true,
        double initialEpsilon = 0.0,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        ILearningRateScheduler? learningRateScheduler = null,
        GraphIsomorphismNetworkOptions? options = null)
        : base(architecture,
               lossFunction ?? new MeanSquaredErrorLoss<T>(),
               maxGradNorm)
    {
        _options = options ?? new GraphIsomorphismNetworkOptions();
        Options = _options;
        LearnEpsilon = learnEpsilon;
        InitialEpsilon = initialEpsilon;
        MlpHiddenDim = mlpHiddenDim;
        NumLayers = numLayers;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        var adamOpts = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = 0.001,
            LearningRateScheduler = learningRateScheduler ?? new ExponentialLRScheduler(
                baseLearningRate: 0.001, gamma: 0.99),
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
            // Filter out trailing SoftmaxActivation.
            foreach (var layer in LayerHelper<T>.CreateDefaultGraphIsomorphismLayers(
                Architecture, MlpHiddenDim, NumLayers, LearnEpsilon, InitialEpsilon))
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

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParamCount = (int)layer.ParameterCount;
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
    /// <param name="learningRate">
    /// [Obsolete] Per-call learning rate. Ignored on the tape-based path
    /// because training always uses the instance's <c>_optimizer</c>
    /// configuration (set in the constructor). Kept on the signature as
    /// a non-breaking transition; pass the rate via the optimizer instead
    /// (e.g. <c>new GraphIsomorphismNetwork&lt;T&gt;(arch, optimizer:
    /// new AdamOptimizer&lt;T,Tensor&lt;T&gt;,Tensor&lt;T&gt;&gt;(this, new AdamOptions { LearningRate = 0.01 }))</c>).
    /// </param>
    public void TrainOnGraph(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        Tensor<T> labels,
        bool[]? trainMask = null,
        int epochs = 200,
        double learningRate = 0.01)
    {
        // The pre-#1219 implementation called ComputeLossGradient and then
        // UpdateParameters(lr) WITHOUT a backward pass — silent no-op. Route
        // through TrainWithTape via the shared helper so gradients actually
        // flow. trainMask + per-mask gradient zeroing is a separate concern
        // that the tape path doesn't yet expose; reject it explicitly so
        // semi-supervised callers see the limitation instead of a silently
        // mistrained network.
        if (trainMask is not null)
        {
            throw new NotSupportedException(
                "TrainOnGraph(trainMask) is not yet supported on the tape-based " +
                "training path (issue follow-up). For semi-supervised node " +
                "classification, pre-mask the labels (set masked-out node labels " +
                "equal to model predictions so their loss contribution is zero) " +
                "and call TrainOnGraph without a mask.");
        }
        // learningRate is kept for non-breaking signature compat but the
        // tape path always uses _optimizer's configured rate. Discard
        // explicitly so the analyzer doesn't flag it as dead-with-no-reason.
        _ = learningRate;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            TrainStepWithAdjacency(nodeFeatures, adjacencyMatrix, labels);
        }
    }

    /// <summary>
    /// Shared tape-backed training step that pins the supplied adjacency
    /// matrix on the network and routes through <see cref="NeuralNetworkBase{T}.TrainWithTape"/>.
    /// Both <see cref="TrainOnGraph"/> and the per-graph step inside
    /// <see cref="TrainOnGraphs"/> call this helper so neither has its own
    /// hand-rolled (and previously broken) backward path.
    /// </summary>
    private void TrainStepWithAdjacency(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        Tensor<T> expected)
    {
        // Pin adjacency on the network so ForwardForTraining's
        // EnsureAdjacencyMatrix returns this graph's matrix instead
        // of generating a fully-connected fallback.
        SetAdjacencyMatrix(adjacencyMatrix);
        // ForwardForTraining(input) re-pushes adjacency to every
        // IGraphConvolutionLayer before the tape-recorded forward, so
        // the gradient flows through the GIN aggregation correctly.
        SetTrainingMode(true);
        try
        {
            TrainWithTape(nodeFeatures, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Trains the GIN network on multiple graphs for graph classification.
    /// </summary>
    /// <param name="graphs">List of graph node feature tensors.</param>
    /// <param name="adjacencyMatrices">List of adjacency matrices.</param>
    /// <param name="graphLabels">Labels for each graph.</param>
    /// <param name="epochs">Number of training epochs (default: 100).</param>
    /// <param name="learningRate">
    /// [Obsolete] Per-call learning rate. Ignored on the tape-based path
    /// because training always uses the instance's <c>_optimizer</c>
    /// configuration. Kept on the signature for non-breaking transition;
    /// pass the rate via the optimizer instead.
    /// </param>
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
        // Graph-level classification = node-level forward + sum readout
        // + per-graph cross-entropy. The pre-#1219 implementation called
        // ComputeGraphLossGradient + DistributeGradient + UpdateParameters
        // with NO backward pass — silent no-op. Routing through TrainWithTape
        // requires the network's output to ALREADY be at graph level
        // ([1, numClasses]) before the loss layer sees it, which means the
        // architecture must contain a SumReadout-style pooling layer at the
        // end. Reject the call explicitly so users add the readout to their
        // architecture instead of silently training nothing.

        // Up-front argument validation: catch mismatched-list and shape
        // bugs at the boundary so callers see a clear error instead of an
        // IndexOutOfRangeException emitted mid-training.
        if (graphs is null) throw new ArgumentNullException(nameof(graphs));
        if (adjacencyMatrices is null) throw new ArgumentNullException(nameof(adjacencyMatrices));
        if (graphLabels is null) throw new ArgumentNullException(nameof(graphLabels));
        if (graphs.Count == 0)
            throw new ArgumentException("graphs list must not be empty.", nameof(graphs));
        if (adjacencyMatrices.Count != graphs.Count)
        {
            throw new ArgumentException(
                $"adjacencyMatrices.Count ({adjacencyMatrices.Count}) must equal graphs.Count " +
                $"({graphs.Count}); each graph needs exactly one adjacency matrix.",
                nameof(adjacencyMatrices));
        }
        if (graphLabels.Rank != 2)
        {
            throw new ArgumentException(
                $"graphLabels must be rank-2 [numGraphs, numClasses]; got rank {graphLabels.Rank} " +
                $"(shape [{string.Join(",", graphLabels.Shape)}]).",
                nameof(graphLabels));
        }
        if (graphLabels.Shape[0] != graphs.Count)
        {
            throw new ArgumentException(
                $"graphLabels.Shape[0] ({graphLabels.Shape[0]}) must equal graphs.Count " +
                $"({graphs.Count}); one label row per graph.",
                nameof(graphLabels));
        }
        // learningRate is kept for non-breaking signature compat but the
        // tape path always uses _optimizer's configured rate.
        _ = learningRate;

        var firstNodeFeatures = graphs[0];
        var firstAdjacency = adjacencyMatrices[0];
        SetAdjacencyMatrix(firstAdjacency);
        var probeOutput = Forward(firstNodeFeatures, firstAdjacency);
        int graphLabelClasses = graphLabels.Shape[1];

        // Network must end in a graph-level pooling layer producing
        // [1, numClasses]; otherwise the per-node output ([numNodes, hidden])
        // can't be matched against a single graph label without manual
        // pooling outside the tape.
        if (probeOutput.Rank != 2 || probeOutput.Shape[0] != 1
            || probeOutput.Shape[1] != graphLabelClasses)
        {
            throw new NotSupportedException(
                $"TrainOnGraphs requires the architecture to end in a graph-level " +
                $"pooling layer that emits shape [1, numClasses]. Got network output " +
                $"shape [{string.Join(",", probeOutput.Shape)}] for graphLabels with " +
                $"{graphLabelClasses} classes. Add a SumReadout / mean-pool layer at " +
                $"the end of your GIN architecture (or inline pooling into a custom " +
                $"final layer), then call TrainOnGraphs again.");
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int g = 0; g < graphs.Count; g++)
            {
                // Slice this graph's label row out as [1, numClasses].
                var graphLabel = new Tensor<T>([1, graphLabelClasses]);
                for (int c = 0; c < graphLabelClasses; c++)
                    graphLabel[0, c] = graphLabels[g, c];

                TrainStepWithAdjacency(graphs[g], adjacencyMatrices[g], graphLabel);
            }
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
    /// Computes the gradient of the cross-entropy loss for graph classification.
    /// </summary>
    private Tensor<T> ComputeGraphLossGradient(Tensor<T> predictions, Tensor<T> labels)
    {
        var gradient = new Tensor<T>(predictions._shape);
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
    public new long GetParameterCount()
    {
        int count = 0;
        foreach (var layer in Layers)
        {
            count += (int)((int)layer.ParameterCount);
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
                count += (int)(loraAdapter.LoRALayer.ParameterCount);
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

        // Ensure 2D input [numNodes, features]
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        var adjacencyMatrix = EnsureAdjacencyMatrix(input);

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return Forward(input, adjacencyMatrix);
    }

    private Tensor<T> EnsureAdjacencyMatrix(Tensor<T> input)
    {
        int numNodes = input.Shape[0];

        if (_cachedAdjacencyMatrix != null &&
            _cachedAdjacencyMatrix.Shape[0] == numNodes &&
            _cachedAdjacencyMatrix.Shape[1] == numNodes)
        {
            return _cachedAdjacencyMatrix;
        }

        var adj = new Tensor<T>([numNodes, numNodes]);
        for (int i = 0; i < numNodes; i++)
            for (int j = 0; j < numNodes; j++)
                adj[i, j] = NumOps.One;

        _cachedAdjacencyMatrix = adj;
        return adj;
    }

    /// <summary>
    /// Sets the adjacency matrix for graph operations.
    /// </summary>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;
    }

    /// <summary>
    /// Trains the network on a single batch of data via tape-based autodiff.
    /// </summary>
    /// <remarks>
    /// Routes through <see cref="NeuralNetworkBase{T}.TrainWithTape"/> so the
    /// gradient flows back through the layer stack via GradientTape (the
    /// unified post-#1060 training path). The previous implementation called
    /// <c>GetParameterGradients()</c> WITHOUT first running a backward pass —
    /// since manual <c>Backward()</c> was removed from <c>ILayer&lt;T&gt;</c>
    /// in #1219, those gradients were always whatever stale value the layer's
    /// gradient field held (zero on a fresh network), which is why
    /// <c>Training_ShouldChangeParameters</c> reported "no parameters changed".
    /// <see cref="ForwardForTraining"/> is overridden below to set the
    /// adjacency matrix on graph layers before the tape-recorded forward pass.
    /// Mirrors <c>GraphAttentionNetwork.Train</c> (line 788) which solved the
    /// identical problem post-#1060.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Ensure 2D input [numNodes, features]
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Computes the adjacency matrix and pushes it into every
    /// <see cref="IGraphConvolutionLayer{T}"/> before the tape-recorded
    /// forward pass. Without this, graph layers see stale or empty
    /// adjacency from a previous call (or none at all on a fresh network),
    /// which makes their aggregation collapse and produces zero loss-
    /// gradient → zero parameter updates. Mirrors
    /// <c>GraphAttentionNetwork.ForwardForTraining</c> (line 823).
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        var adjacencyMatrix = EnsureAdjacencyMatrix(input);
        _cachedAdjacencyMatrix = adjacencyMatrix;
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
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        var adjacencyMatrix = EnsureAdjacencyMatrix(input);

        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
        }

        var activations = new Dictionary<string, Tensor<T>>();
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <summary>
    /// Gets metadata about this model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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
