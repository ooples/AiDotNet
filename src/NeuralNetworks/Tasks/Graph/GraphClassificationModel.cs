using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Implements a complete neural network model for graph classification tasks.
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
/// Molecule (graph) -> GNN layers -> Molecule embedding -> Classifier -> Toxic? (Yes/No)
///
/// Small molecule (10 atoms):
///   10 nodes -> GNN -> 10 embeddings -> Pool -> 1 graph embedding -> Classify
///
/// Large molecule (50 atoms):
///   50 nodes -> GNN -> 50 embeddings -> Pool -> 1 graph embedding -> Classify
///
/// Both produce same-sized graph embedding despite different input sizes!
/// ```
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a graph classification model for molecular property prediction
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputSize: 16,   // node feature dimension
///     outputSize: 2,   // number of classes (e.g., toxic / non-toxic)
///     hiddenSizes: new[] { 64, 32 });
/// var model = new GraphClassificationModel&lt;float&gt;(architecture);
///
/// // Prepare graph data (adjacency + node features as tensors)
/// var adjacency = new Tensor&lt;float&gt;(new[] { 10, 10 }); // 10-node graph
/// var nodeFeatures = new Tensor&lt;float&gt;(new[] { 10, 16 });
///
/// // Classify the entire graph
/// var prediction = model.Predict(nodeFeatures);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Semi-Supervised Classification with Graph Convolutional Networks",
    "https://arxiv.org/abs/1609.02907",
    Year = 2017,
    Authors = "Thomas N. Kipf, Max Welling")]
public class GraphClassificationModel<T> : NeuralNetworkBase<T>
{
    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly GraphPooling _poolingType;
    private Tensor<T>? _cachedAdjacencyMatrix;
    // Opt-in (EnableImplicitIdentityAdjacency): mirrors the GraphConvolutionalLayer
    // implicitIdentityWhenUnset ctor flag at the model level. Default is strict (throw on a
    // missing graph); when enabled, Predict/Train build an identity sized to the input.
    private bool _implicitIdentityWhenUnset;
    // Tracks whether _cachedAdjacencyMatrix came from EnsureDefaultAdjacencyForInput
    // (auto-inferred identity) vs an explicit SetAdjacencyMatrix call. Explicit
    // matrices are sticky — caller knows the graph; auto-inferred ones must be
    // regenerated whenever the input node count changes so a later Predict /
    // Train on a different-sized graph doesn't run against a stale identity.
    private bool _usesFallbackAdjacency;
    private Tensor<T>? _nodeEmbeddings;
    private Tensor<T>? _graphEmbedding;
    private int[]? _maxPoolingIndices; // Cached indices for max pooling backward pass

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
    /// Gets the graph embedding dimension after pooling.
    /// </summary>
    public int EmbeddingDim { get; }

    /// <summary>
    /// Gets the number of GNN layers.
    /// </summary>
    public int NumGnnLayers { get; }

    /// <summary>
    /// Gets the dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphClassificationModel{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output sizes and layers.</param>
    /// <param name="hiddenDim">Hidden dimension for intermediate layers (default: 64).</param>
    /// <param name="embeddingDim">Dimension of graph embedding after pooling (default: 128).</param>
    /// <param name="numGnnLayers">Number of graph convolutional layers (default: 3).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.5).</param>
    /// <param name="poolingType">Method for pooling node embeddings to graph embedding.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a graph classification model:
    ///
    /// ```csharp
    /// // Create architecture for molecular property prediction
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional,
    ///     NeuralNetworkTaskType.MultiClassClassification,
    ///     NetworkComplexity.Simple,
    ///     inputSize: 9,      // Atom features
    ///     outputSize: 2);    // Binary classification (toxic/not toxic)
    ///
    /// // Create model with default layers
    /// var model = new GraphClassificationModel&lt;double&gt;(architecture);
    ///
    /// // Train on graph classification task
    /// var history = model.TrainOnTask(task, epochs: 100, learningRate: 0.001);
    /// ```
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public GraphClassificationModel()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 128,
            outputSize: 7))
    {
    }

    public GraphClassificationModel(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 64,
        int embeddingDim = 128,
        int numGnnLayers = 3,
        double dropoutRate = 0.5,
        GraphPooling poolingType = GraphPooling.Mean,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        // The classification head emits RAW LOGITS (PredictCore returns Forward(input) with no
        // Softmax), so the numerically-stable, industry-standard pairing is CrossEntropyWithLogitsLoss
        // — it applies LogSoftmax internally (matching PyTorch nn.CrossEntropyLoss, which also takes
        // logits). Train feeds raw logits straight to the loss with NO explicit Softmax (a separate
        // Softmax would double-apply the activation and produce wrong gradients). Exposing it as the
        // model's loss also lets the test harness's MeasureLoss measure the model's own cross-entropy
        // objective (which DECREASES as training sharpens the correct-class logits) instead of
        // MSE-against-a-random-target (which GROWS as logits sharpen, mis-reading healthy training as
        // "loss increased"). This is the logits variant the prior comment said callers should adopt.
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), maxGradNorm)
    {
        InputFeatures = architecture.InputSize;
        NumClasses = architecture.OutputSize;
        HiddenDim = hiddenDim;
        EmbeddingDim = embeddingDim;
        NumGnnLayers = numGnnLayers;
        DropoutRate = dropoutRate;
        _poolingType = poolingType;

        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
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
            // Create default graph classification layers using LayerHelper
            // Note: Pooling is handled separately in Forward, not as a layer
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraphClassificationLayers(
                Architecture, HiddenDim, EmbeddingDim, NumGnnLayers, DropoutRate));
        }
    }

    /// <summary>
    /// Sets the adjacency matrix for all graph layers in the model.
    /// </summary>
    /// <param name="adjacencyMatrix">The graph adjacency matrix.</param>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;
        _usesFallbackAdjacency = false;

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
    /// <param name="nodeFeatures">Node feature tensor [num_nodes, input_features].</param>
    /// <returns>Output predictions [num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> nodeFeatures)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set before forward pass. Call SetAdjacencyMatrix() first.");
        }

        // Step 1: Node-level processing through GNN layers
        var current = nodeFeatures;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        _nodeEmbeddings = current;

        // Step 2: Pool node embeddings to graph embedding
        _graphEmbedding = PoolGraph(_nodeEmbeddings);

        // Step 3: For classification, apply softmax (done in loss computation)
        return _graphEmbedding;
    }

    /// <summary>
    /// Pools node embeddings into a single graph-level embedding.
    /// </summary>
    /// <param name="nodeEmbeddings">Node embeddings of shape [num_nodes, embedding_dim].</param>
    /// <returns>Graph embedding of shape [1, embedding_dim].</returns>
    private Tensor<T> PoolGraph(Tensor<T> nodeEmbeddings)
    {
        int numNodes = nodeEmbeddings.Shape[0];

        // Vectorized pooling using Engine operations
        switch (_poolingType)
        {
            case GraphPooling.Mean:
                // Vectorized mean pooling: reduce along node dimension and divide by count
                var sum = Engine.ReduceSum(nodeEmbeddings, [0], keepDims: true);
                _maxPoolingIndices = null; // Not needed for mean pooling
                return Engine.TensorDivideScalar(sum, NumOps.FromDouble(numNodes));

            case GraphPooling.Max:
                // Vectorized max pooling: reduce max along node dimension
                // Store indices for backward pass
                var maxResult = Engine.ReduceMax(nodeEmbeddings, [0], keepDims: true, out int[] maxIndices);
                _maxPoolingIndices = maxIndices;
                return maxResult;

            case GraphPooling.Sum:
                // Vectorized sum pooling: reduce along node dimension
                _maxPoolingIndices = null; // Not needed for sum pooling
                return Engine.ReduceSum(nodeEmbeddings, [0], keepDims: true);

            case GraphPooling.Attention:
                // Simplified attention pooling (uniform weights = mean)
                var attSum = Engine.ReduceSum(nodeEmbeddings, [0], keepDims: true);
                _maxPoolingIndices = null; // Not needed for attention pooling
                return Engine.TensorDivideScalar(attSum, NumOps.FromDouble(numNodes));

            default:
                // Fallback to mean pooling
                var defaultSum = Engine.ReduceSum(nodeEmbeddings, [0], keepDims: true);
                _maxPoolingIndices = null;
                return Engine.TensorDivideScalar(defaultSum, NumOps.FromDouble(numNodes));
        }
    }

    private Tensor<T> BackpropPooling(Tensor<T> gradGraphEmb)
    {
        if (_nodeEmbeddings is null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward.");
        }

        int numNodes = _nodeEmbeddings.Shape[0];
        int embDim = _nodeEmbeddings.Shape[1];
        int[] inputShape = [numNodes, embDim];

        switch (_poolingType)
        {
            case GraphPooling.Mean:
                // Vectorized: gradient divided by numNodes then tiled to all nodes
                var scaledGrad = Engine.TensorDivideScalar(gradGraphEmb, NumOps.FromDouble(numNodes));
                // Tile [1, embDim] -> [numNodes, embDim] by repeating numNodes times along axis 0
                return Engine.TensorTile(scaledGrad, [numNodes, 1]);

            case GraphPooling.Max:
                // Max pooling backward: use ReduceMaxBackward with cached indices
                if (_maxPoolingIndices is null)
                {
                    throw new InvalidOperationException("Max pooling indices not cached from forward pass.");
                }
                return Engine.ReduceMaxBackward(gradGraphEmb, _maxPoolingIndices, inputShape);

            case GraphPooling.Sum:
                // Sum pooling backward: gradient is copied to all nodes
                return Engine.TensorTile(gradGraphEmb, [numNodes, 1]);

            case GraphPooling.Attention:
            default:
                // For simplified attention (uniform weights = mean), same as mean backward
                var attScaledGrad = Engine.TensorDivideScalar(gradGraphEmb, NumOps.FromDouble(numNodes));
                return Engine.TensorTile(attScaledGrad, [numNodes, 1]);
        }
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
            int layerParamCount = checked((int)layer.ParameterCount);
            if (layerParamCount > 0)
            {
                var layerParams = parameters.SubVector(index, layerParamCount);
                layer.SetParameters(layerParams);
                index += layerParamCount;
            }
        }
    }

    /// <summary>
    /// Trains the model on a graph classification task.
    /// </summary>
    /// <param name="task">The graph classification task with training/validation/test graphs.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <returns>Training history with loss and accuracy per epoch.</returns>
    public Dictionary<string, List<double>> TrainOnTask(
        GraphClassificationTask<T> task,
        int epochs,
        double learningRate = 0.001)
    {
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

            double epochLoss = 0.0;
            int correctTrain = 0;

            // Training loop - process each graph
            for (int i = 0; i < task.TrainGraphs.Count; i++)
            {
                var graph = task.TrainGraphs[i];
                if (graph.AdjacencyMatrix is null)
                {
                    throw new ArgumentException($"Training graph {i} must have an adjacency matrix.");
                }

                SetAdjacencyMatrix(graph.AdjacencyMatrix);
                var logits = Forward(graph.NodeFeatures);

                // Apply softmax to get probabilities
                var probs = Softmax(logits);

                // Compute cross-entropy loss
                double loss = 0.0;
                for (int c = 0; c < NumClasses; c++)
                {
                    var prob = NumOps.ToDouble(probs[0, c]);
                    var label = NumOps.ToDouble(task.TrainLabels[i, c]);
                    loss -= label * Math.Log(Math.Max(prob, 1e-10));
                }
                epochLoss += loss;

                // Accuracy
                int predictedClass = GetPredictedClass(logits);
                int trueClass = GetTrueClass(task.TrainLabels, i);
                if (predictedClass == trueClass) correctTrain++;

                // Backward pass - gradient of cross-entropy with softmax is (prob - label)
                var gradient = ComputeGradient(probs, task.TrainLabels, i);

                // Update parameters
                foreach (var layer in Layers)
                {
                    layer.UpdateParameters(lr);
                }
            }

            double avgLoss = task.TrainGraphs.Count > 0 ? epochLoss / task.TrainGraphs.Count : 0.0;
            double trainAcc = task.TrainGraphs.Count > 0 ? (double)correctTrain / task.TrainGraphs.Count : 0.0;

            // Validation accuracy
            double valAcc = EvaluateGraphs(task.ValGraphs, task.ValLabels);

            history["train_loss"].Add(avgLoss);
            history["train_accuracy"].Add(trainAcc);
            history["val_accuracy"].Add(valAcc);
        }

        // Set layers back to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return history;
    }

    /// <summary>
    /// Evaluates the model on test graphs.
    /// </summary>
    public double EvaluateOnTask(GraphClassificationTask<T> task)
    {
        return EvaluateGraphs(task.TestGraphs, task.TestLabels);
    }

    private double EvaluateGraphs(List<GraphData<T>> graphs, Tensor<T> labels)
    {
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        if (graphs.Count == 0)
        {
            return 0.0;
        }

        int correct = 0;

        for (int i = 0; i < graphs.Count; i++)
        {
            var graph = graphs[i];
            if (graph.AdjacencyMatrix is null) continue;

            SetAdjacencyMatrix(graph.AdjacencyMatrix);
            var logits = Forward(graph.NodeFeatures);
            int predictedClass = GetPredictedClass(logits);
            int trueClass = GetTrueClass(labels, i);

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

    private int GetTrueClass(Tensor<T> labels, int graphIdx)
    {
        for (int c = 0; c < NumClasses; c++)
        {
            if (!NumOps.Equals(labels[graphIdx, c], NumOps.Zero))
                return c;
        }
        return 0;
    }

    private Tensor<T> ComputeGradient(Tensor<T> probs, Tensor<T> labels, int graphIdx)
    {
        // Vectorized: gradient of cross-entropy with softmax is (prob - label)
        // Extract the label row for this graph using Engine.GetRow equivalent
        var labelRow = new Tensor<T>([1, NumClasses]);
        for (int c = 0; c < NumClasses; c++)
        {
            labelRow[0, c] = labels[graphIdx, c];
        }
        return Engine.TensorSubtract<T>(probs, labelRow);
    }

    private Tensor<T> Softmax(Tensor<T> logits)
    {
        // Vectorized softmax using Engine operations
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        //
        // Both the max-subtraction and the sum-normalisation involve a
        // reduce-with-keepDims → broadcast back to logits shape. Strict
        // TensorSubtract / TensorDivide reject shape mismatch, so use the
        // broadcast-aware variants — the reduced tensors are [..., 1] and
        // need to broadcast across the last dim.

        // Find max for numerical stability
        var maxLogit = Engine.ReduceMax(logits, [1], keepDims: true, out _);

        // Subtract max for stability: logits - max
        var shifted = Engine.TensorBroadcastSubtract<T>(logits, maxLogit);

        // Compute exp(shifted)
        var expValues = Engine.TensorExp(shifted);

        // Sum the exp values
        var sumExp = Engine.ReduceSum(expValues, [1], keepDims: true);

        // Normalize: exp / sum using element-wise division with broadcast
        return Engine.TensorBroadcastDivide<T>(expValues, sumExp);
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
    /// <returns>The prediction tensor with class probabilities.</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        EnsureDefaultAdjacencyForInput(input);

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return Forward(input);
    }
    /// <summary>
    /// Enables implicit self-loops-only (identity) tolerance for this model and its graph layers —
    /// the model-level analogue of GraphConvolutionalLayer's implicitIdentityWhenUnset ctor flag.
    /// Default is strict (Predict/Train throw on a missing graph); this is an explicit opt-in for
    /// scaffold / clone / deserialize paths, propagated to the GCN layers so direct-layer paths
    /// (e.g. GetNamedLayerActivations) tolerate too.
    /// </summary>
    public void EnableImplicitIdentityAdjacency()
    {
        _implicitIdentityWhenUnset = true;
        foreach (var layer in Layers)
        {
            if (layer is NeuralNetworks.Layers.GraphConvolutionalLayer<T> gcn)
            {
                gcn.EnableImplicitIdentityAdjacency();
            }
        }
    }


    /// <summary>
    /// Auto-creates an identity adjacency matrix when none has been set,
    /// sized to the input's first dimension. Lets the test scaffold's
    /// generic <c>Predict</c> / <c>Train</c> invariant tests run on graph
    /// models without each scaffold knowing to call <see cref="SetAdjacencyMatrix"/>
    /// first — every invariant the scaffold checks (gradient flow, loss
    /// reduction, determinism, etc.) is well-defined on an isolated-nodes
    /// graph (Kipf &amp; Welling 2017 §2: with <c>A = I</c>, the GCN layer
    /// degenerates to a per-node dense transform). Production callers
    /// SHOULD still call <see cref="SetAdjacencyMatrix"/> explicitly with
    /// the real graph structure; this is a default-of-last-resort, not a
    /// recommended training mode.
    /// </summary>
    private void EnsureDefaultAdjacencyForInput(Tensor<T> input)
    {
        if (_cachedAdjacencyMatrix is null && !_implicitIdentityWhenUnset)
        {
            // PyTorch-Geometric contract: the graph is REQUIRED — call SetAdjacencyMatrix() with
            // the real structure before Predict/Train. Implicit-identity tolerance is opt-in
            // (EnableImplicitIdentityAdjacency / the layer's implicitIdentityWhenUnset), never a
            // silent default that would let a GCN ignore all graph structure unnoticed.
            throw new InvalidOperationException(
                "Adjacency matrix must be set before Predict/Train: call SetAdjacencyMatrix() with " +
                "the graph structure. Graph neural networks have no meaningful default graph.");
        }

        if (input.Rank < 1) return; // can't infer; let Forward throw with a clear shape error
        int numNodes = input.Shape[0];

        // Explicit matrices (from SetAdjacencyMatrix / FitFromGraph) are sticky.
        // Auto-inferred ones must be regenerated when the input node count changes
        // — otherwise the first inferred identity becomes stuck model state and
        // a later differently-sized input runs against the wrong graph structure.
        if (_cachedAdjacencyMatrix is not null
            && (!_usesFallbackAdjacency || _cachedAdjacencyMatrix.Shape[0] == numNodes))
        {
            return;
        }

        var identity = new Tensor<T>(new[] { numNodes, numNodes });
        for (int i = 0; i < numNodes; i++)
            identity[i, i] = NumOps.One;
        SetAdjacencyMatrix(identity);
        _usesFallbackAdjacency = true;
    }

    /// <summary>
    /// Trains the network on a single batch of data.
    /// </summary>
    /// <param name="input">The input node features.</param>
    /// <param name="expectedOutput">The expected output (labels).</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        EnsureDefaultAdjacencyForInput(input);

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(true);
        }

        // ILayer.Backward was removed in favour of GradientTape autodiff. The previous code computed a
        // loss derivative but NEVER ran a backward pass — it read GetParameterGradients() (stale zeros)
        // straight away — so the optimizer applied a zero step and training was a silent no-op: the loss
        // stayed flat (2.4452 -> 2.4452) and not a single parameter moved. Record the GNN + pooling
        // forward (raw logits) and the cross-entropy-with-logits loss on the tape, compute real gradients for every
        // trainable parameter, then drive the update through the model's configured optimizer
        // (default Adam) — NOT a hardcoded SGD step — so the loss actually converges on a memorization
        // task. The loss takes raw logits (it applies LogSoftmax internally) and aligns the target
        // shape itself, so there is no explicit Softmax (which would double-apply it) and no reshape.
        var tapeLoss = _lossFunction as LossFunctionBase<T> ?? new CrossEntropyWithLogitsLoss<T>();
        using (var tape = new GradientTape<T>())
        {
            var logits = Forward(input);
            var lossTensor = tapeLoss.ComputeTapeLoss(logits, expectedOutput);
            var trainableParameters = Training.TapeTrainingStep<T>.CollectParameters(Layers, LayerStructureVersion);
            if (trainableParameters.Count > 0)
            {
                var gradients = tape.ComputeGradients(lossTensor, trainableParameters);
                T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
                var context = new TapeStepContext<T>(trainableParameters, gradients, lossValue);
                _optimizer.Step(context);
                LastLoss = lossValue;
            }
        }
    }

    /// <summary>
    /// Gets metadata about this model for serialization and identification.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NetworkType"] = "GraphClassificationModel",
                ["InputFeatures"] = InputFeatures,
                ["NumClasses"] = NumClasses,
                ["HiddenDim"] = HiddenDim,
                ["EmbeddingDim"] = EmbeddingDim,
                ["NumGnnLayers"] = NumGnnLayers,
                ["DropoutRate"] = DropoutRate,
                ["PoolingType"] = _poolingType.ToString()
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
        writer.Write(EmbeddingDim);
        writer.Write(NumGnnLayers);
        writer.Write(DropoutRate);
        writer.Write((int)_poolingType);

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
        _ = reader.ReadInt32(); // EmbeddingDim
        _ = reader.ReadInt32(); // NumGnnLayers
        _ = reader.ReadDouble(); // DropoutRate
        _ = reader.ReadInt32(); // PoolingType

        _ = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        _ = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);
    }

    /// <summary>
    /// Creates a new instance of this network type for cloning or deserialization.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var clone = new GraphClassificationModel<T>(
            architecture: Architecture,
            hiddenDim: HiddenDim,
            embeddingDim: EmbeddingDim,
            numGnnLayers: NumGnnLayers,
            dropoutRate: DropoutRate,
            poolingType: _poolingType);
        // Graph STATE is model state in this stateful-adjacency design, so a clone of a configured
        // model must stay usable: carry the implicit-identity opt-in and any explicit adjacency.
        if (_implicitIdentityWhenUnset)
        {
            clone.EnableImplicitIdentityAdjacency();
        }
        else if (_cachedAdjacencyMatrix is not null)
        {
            clone.SetAdjacencyMatrix(_cachedAdjacencyMatrix);
        }

        return clone;
    }

    #endregion
}
