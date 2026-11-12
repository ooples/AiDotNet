using AiDotNet.Optimizers;
using AiDotNet.LearningRateSchedulers;
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
/// Implements a complete neural network model for link prediction tasks on graphs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Link prediction predicts whether edges should exist between node pairs using:
/// - Node features
/// - Graph structure
/// - Learned node embeddings
/// </para>
/// <para><b>For Beginners:</b> This model predicts connections between nodes.
///
/// **How it works:**
///
/// 1. **Encode**: Learn embeddings for all nodes using GNN layers
///    ```
///    Input: Node features + Graph structure
///    Process: Stack of graph conv layers
///    Output: Node embeddings [num_nodes, embedding_dim]
///    ```
///
/// 2. **Decode**: Score node pairs to predict edges
///    ```
///    Input: Node pair (i, j)
///    Compute: score = f(embedding[i], embedding[j])
///    Common functions:
///    - Dot product: z_i * z_j
///    - Concatenation + MLP: MLP([z_i || z_j])
///    - Distance-based: -||z_i - z_j||^2
///    ```
///
/// 3. **Train**: Learn to score existing edges high, non-existing edges low
///
/// **Example:**
/// ```
/// Friend recommendation:
/// - Encode users as embeddings using friend network
/// - For user pair (Alice, Bob):
///   * Compute score from their embeddings
///   * High score -> Likely to be friends
///   * Low score -> Unlikely to be friends
/// ```
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a link prediction model for graph edge prediction
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.MultiClassClassification,
///     inputSize: 16,   // node feature dimension
///     outputSize: 2);  // number of classes (e.g. toxic / non-toxic)
///
/// // Prepare graph data
/// var adjacency = new Tensor&lt;float&gt;(new[] { 100, 100 }); // 100-node graph
/// var nodeFeatures = new Tensor&lt;float&gt;(new[] { 100, 16 });
///
/// // Predict edge likelihood between node pairs
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new LinkPredictionModel&lt;float&gt;(architecture))
///     .Build(trainX, trainY);
/// var scores = result.Predict(nodeFeatures);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
// LinkPredictionModel IS a graph neural network (it requires an adjacency matrix), so it must carry the
// GraphNetwork category like its siblings GraphClassificationModel / NodeClassificationModel. Without it the
// test scaffold classified it as a generic NN and exercised it through the non-graph test base, which never
// enables the implicit-identity adjacency — so every Predict/Train threw "Adjacency matrix must be set".
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Variational Graph Auto-Encoders",
    "https://arxiv.org/abs/1611.07308",
    Year = 2016,
    Authors = "Thomas N. Kipf, Max Welling")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.01,
                Source = "Kipf and Welling 2016 (VGAE), Sec. 2 (Experiments on link prediction): Adam, lr 0.01, 200 iterations. Verified against the paper.")]
public partial class LinkPredictionModel<T> : GraphModelLayoutBase<T>
{
    private readonly LinkPredictionOptions _options;

    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly LinkPredictionDecoder _decoderType;
    // [Scratch], matching GraphClassificationModel and NodeClassificationModel, which hold the
    // identically named field for the identical purpose. As [Buffer] it was nullable with no
    // initializer, which the generator reads as fit-produced and gives ParameterAvailability.Fit --
    // so the registry refused to report parameters at all until the model was "fit", and the graph's
    // own [numNodes, numNodes] width landed in GetParameters() while ParameterCount omitted it.
    // Neither is true of this member: it caches the adjacency the CALLER supplied, and #1593 made
    // supplying it the model's contract (the strict PyTorch-Geometric behaviour), so there is
    // nothing here for a checkpoint to carry.
    [Scratch]
    private Tensor<T>? _cachedAdjacencyMatrix;
    // Opt-in (EnableImplicitIdentityAdjacency): mirrors the GraphConvolutionalLayer
    // implicitIdentityWhenUnset ctor flag at the model level. Default is strict (throw on a
    // missing graph); when enabled, Predict/Train build an identity sized to the input.
    private bool _implicitIdentityWhenUnset;
    // Tracks whether _cachedAdjacencyMatrix came from EnsureDefaultAdjacencyForInput
    // (auto-inferred identity) vs an explicit SetAdjacencyMatrix call. Explicit
    // matrices are sticky — caller knows the graph; auto-inferred ones must be
    // regenerated whenever the input node count changes (same contract as
    // GraphClassificationModel / NodeClassificationModel).
    private bool _usesFallbackAdjacency;
    [Scratch]
    private Tensor<T>? _nodeEmbeddings;


    /// <summary>
    /// Gets the number of input features per node.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; }

    /// <summary>
    /// Gets the hidden dimension size.
    /// </summary>
    public int HiddenDim { get; }

    /// <summary>
    /// Gets the number of GNN layers.
    /// </summary>
    public int NumLayers { get; }

    /// <summary>
    /// Gets the dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="LinkPredictionModel{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output sizes and layers.</param>
    /// <param name="hiddenDim">Hidden dimension for intermediate layers (default: 64).</param>
    /// <param name="embeddingDim">Dimension of node embeddings (default: 32).</param>
    /// <param name="numLayers">Number of graph convolutional layers (default: 2).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.5).</param>
    /// <param name="decoderType">Method for combining node embeddings into edge scores.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a link prediction model:
    ///
    /// ```csharp
    /// // Create architecture for friend prediction
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional,
    ///     NeuralNetworkTaskType.BinaryClassification,
    ///     NetworkComplexity.Simple,
    ///     // User features
    ///     outputSize: 1);    // Edge score
    ///
    /// // Create model with default layers
    /// var model = new LinkPredictionModel&lt;double&gt;(architecture);
    ///
    /// // Train on link prediction task
    /// var history = model.TrainOnTask(task, epochs: 100, learningRate: 0.01);
    /// ```
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public LinkPredictionModel()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.BinaryClassification,
            inputSize: 128,
            outputSize: 1))
    {
    }

    public LinkPredictionModel(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LinkPredictionOptions? options = null)
        : this(options ?? new LinkPredictionOptions(), architecture, optimizer, lossFunction)
    {
    }

    /// <summary>
    /// Initializes the model from an already-resolved options instance.
    /// </summary>
    private LinkPredictionModel(
        LinkPredictionOptions options,
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(), options.MaxGradNorm)
    {
        options.Validate();
        _options = options;
        Options = _options;
        InputFeatures = architecture.InputSize;
        EmbeddingDim = options.EmbeddingDim;
        HiddenDim = options.HiddenDim;
        NumLayers = options.NumLayers;
        DropoutRate = options.DropoutRate;
        _decoderType = options.DecoderType;

        // The outputs are raw scores (the graph autoencoder scores an edge as sigmoid(z_i . z_j)), so the default
        // takes logits. Probability-space BinaryCrossEntropyLoss clamps its input to [eps, 1 - eps]: whenever every
        // output fell outside (0, 1), as one random initialisation in about twelve did, its gradient was exactly zero
        // and training never moved the weights.
        _lossFunction = lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
            // Create default link prediction encoder layers using LayerHelper
            Layers.AddRange(LayerHelper<T>.CreateDefaultLinkPredictionLayers(
                Architecture, HiddenDim, EmbeddingDim, NumLayers, DropoutRate));
        }
    }

    /// <summary>
    /// Sets the adjacency matrix for all graph layers in the model.
    /// </summary>
    /// <param name="adjacencyMatrix">The graph adjacency matrix.</param>
    // Points every graph layer at an adjacency without changing the model's own: a training step's packed copy, then
    // the model's again.
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
    /// Default-of-last-resort adjacency: when no explicit matrix was set, fall back to the
    /// IDENTITY graph so scaffold invariants stay well-defined — per Kipf &amp; Welling 2017 §2,
    /// with <c>A = I</c> the GCN encoder degenerates to a per-node dense transform. Production
    /// callers SHOULD still call <see cref="SetAdjacencyMatrix"/> with the real graph structure.
    /// Brings this model in line with the documented contract its siblings
    /// (GraphClassificationModel / NodeClassificationModel) already implement.
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

        // Explicit matrices (from SetAdjacencyMatrix) are sticky. Auto-inferred ones must be
        // regenerated when the input node count changes — otherwise the first inferred identity
        // becomes stuck model state for a later differently-sized graph.
        if (_cachedAdjacencyMatrix is not null
            && (!_usesFallbackAdjacency || _cachedAdjacencyMatrix.Shape[0] == numNodes))
        {
            return;
        }

        SetAdjacencyMatrix(Engine.TensorEye<T>(numNodes));
        _usesFallbackAdjacency = true;
    }

    /// <summary>
    /// Performs a forward pass through the encoder network.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor [num_nodes, input_features].</param>
    /// <returns>Node embeddings [num_nodes, embedding_dim].</returns>
    public Tensor<T> Forward(Tensor<T> nodeFeatures)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set before forward pass. Call SetAdjacencyMatrix() first.");
        }

        // Encode: Pass through GNN layers to get node embeddings
        var current = nodeFeatures;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        _nodeEmbeddings = current;
        return current;
    }

    /// <summary>
    /// Computes edge scores for given node pairs.
    /// </summary>
    /// <param name="edges">Edge tensor of shape [num_edges, 2] where each row is [source, target].</param>
    /// <returns>Edge scores of shape [num_edges].</returns>
    public Tensor<T> PredictEdges(Tensor<T> edges)
    {
        if (_nodeEmbeddings is null)
        {
            throw new InvalidOperationException(
                "Must call Forward() to compute node embeddings before predicting edges.");
        }

        int numEdges = edges.Shape[0];
        if (numEdges == 0)
        {
            return new Tensor<T>([0]);
        }

        var sourceEmb = GatherEmbeddings(EdgeEndpoints(edges, 0));
        var targetEmb = GatherEmbeddings(EdgeEndpoints(edges, 1));

        return _decoderType switch
        {
            LinkPredictionDecoder.CosineSimilarity => CosineSimilarityRows(sourceEmb, targetEmb),
            LinkPredictionDecoder.Distance => Engine.TensorNegate(RowSums(SquaredDifference(sourceEmb, targetEmb))),
            // Hadamard decoding sums the element-wise product, which is the dot product
            _ => RowSums(Engine.TensorMultiply(sourceEmb, targetEmb))
        };
    }

    /// <summary>
    /// Node indices in one column of an edge tensor, rounded to the nearest integer.
    /// </summary>
    private Tensor<int> EdgeEndpoints(Tensor<T> edges, int column)
    {
        int numEdges = edges.Shape[0];
        return Engine.TensorRound(Engine.TensorNarrow(edges, 1, column, 1)).Reshape(numEdges).Cast<int>();
    }

    /// <summary>
    /// Embedding rows [count, EmbeddingDim] of the given nodes.
    /// </summary>
    private Tensor<T> GatherEmbeddings(Tensor<int> nodes)
    {
        if (_nodeEmbeddings is null) throw new InvalidOperationException("Embeddings not computed.");

        var table = Engine.TensorNarrow(_nodeEmbeddings, 1, 0, EmbeddingDim);
        return Engine.TensorIndexSelect(table, nodes, 0);
    }

    private Tensor<T> RowSums(Tensor<T> rows) => Engine.ReduceSum(rows, [1], keepDims: false);

    private Tensor<T> SquaredDifference(Tensor<T> a, Tensor<T> b)
    {
        var diff = Engine.TensorSubtract(a, b);
        return Engine.TensorMultiply(diff, diff);
    }

    /// <summary>
    /// Row-wise cosine similarity clamped to [-1, 1], or 0 where ||a|| * ||b|| is not above 1e-10.
    /// </summary>
    private Tensor<T> CosineSimilarityRows(Tensor<T> a, Tensor<T> b)
    {
        var dot = RowSums(Engine.TensorMultiply(a, b));
        var denom = Engine.TensorMultiply(
            Engine.TensorSqrt(RowSums(Engine.TensorMultiply(a, a))),
            Engine.TensorSqrt(RowSums(Engine.TensorMultiply(b, b))));
        var cosine = Engine.TensorClamp(Engine.TensorDivide(dot, denom), NumOps.FromDouble(-1.0), NumOps.One);
        return Engine.TensorWhere(
            Engine.TensorGreaterThan(denom, NumOps.FromDouble(1e-10)), cosine, Engine.TensorZerosLike(dot));
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Trains the model on a link prediction task.
    /// </summary>
    /// <param name="task">The link prediction task with graph data and edge splits.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <returns>Training history with loss and metrics per epoch.</returns>
    public Dictionary<string, List<double>> TrainOnTask(
        LinkPredictionTask<T> task,
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
            ["val_auc"] = new List<double>()
        };

        var lr = NumOps.FromDouble(learningRate);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Set all layers to training mode
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            // Encode nodes
            Forward(task.Graph.NodeFeatures);

            // Score training edges
            var posScores = PredictEdges(task.TrainPosEdges);
            var negScores = PredictEdges(task.TrainNegEdges);

            // Compute binary cross-entropy loss
            double loss = ComputeBCELoss(posScores, negScores);
            history["train_loss"].Add(loss);

            // Validation AUC
            if (task.ValPosEdges.Shape[0] > 0)
            {
                var valPosScores = PredictEdges(task.ValPosEdges);
                var valNegScores = PredictEdges(task.ValNegEdges);
                double auc = ComputeAUC(valPosScores, valNegScores);
                history["val_auc"].Add(auc);
            }

            // Compute gradients and backpropagate
            var gradient = ComputeBCEGradients(posScores, negScores, task.TrainPosEdges, task.TrainNegEdges);

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
    /// Evaluates the model on test edges.
    /// </summary>
    public double EvaluateOnTask(LinkPredictionTask<T> task)
    {
        if (task.Graph.AdjacencyMatrix is not null)
        {
            SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        Forward(task.Graph.NodeFeatures);

        var testPosScores = PredictEdges(task.TestPosEdges);
        var testNegScores = PredictEdges(task.TestNegEdges);

        return ComputeAUC(testPosScores, testNegScores);
    }

    private Tensor<T> ComputeBCEGradients(
        Tensor<T> posScores,
        Tensor<T> negScores,
        Tensor<T> posEdges,
        Tensor<T> negEdges)
    {
        if (_nodeEmbeddings is null)
        {
            throw new InvalidOperationException("Node embeddings not computed.");
        }

        int numPos = posEdges.Shape[0];
        int numNeg = negEdges.Shape[0];
        var indexParts = new List<Tensor<int>>(2);
        var updateParts = new List<Tensor<T>>(2);

        // Positive edges (target = 1, grad = sigmoid(score) - 1)
        if (numPos > 0)
        {
            var posGrad = Engine.TensorAddScalar(
                Engine.Sigmoid(Engine.TensorNarrow(posScores, 0, 0, numPos)), NumOps.FromDouble(-1.0));
            AddEdgeGradients(posEdges, posGrad, indexParts, updateParts);
        }

        // Negative edges (target = 0, grad = sigmoid(score))
        if (numNeg > 0)
        {
            var negGrad = Engine.Sigmoid(Engine.TensorNarrow(negScores, 0, 0, numNeg));
            AddEdgeGradients(negEdges, negGrad, indexParts, updateParts);
        }

        var gradient = new Tensor<T>(_nodeEmbeddings._shape);
        if (indexParts.Count == 0)
        {
            return gradient;
        }

        // Accumulate edge by edge: source row, then target row, positive edges before negative ones
        var indices = indexParts.Count == 1 ? indexParts[0] : Engine.TensorConcatenate(indexParts.ToArray(), 0);
        var updates = updateParts.Count == 1 ? updateParts[0] : Engine.TensorConcatenate(updateParts.ToArray(), 0);
        var accumulated = Engine.TensorIndexAdd(Engine.TensorNarrow(gradient, 1, 0, EmbeddingDim), 0, indices, updates);
        if (gradient.Shape[1] == EmbeddingDim)
        {
            return accumulated;
        }

        return Engine.TensorConcatenate(
            new[] { accumulated, Engine.TensorNarrow(gradient, 1, EmbeddingDim, gradient.Shape[1] - EmbeddingDim) }, 1);
    }

    /// <summary>
    /// Appends the interleaved (source, target) row indices of each edge and their updates
    /// lossGrad * targetEmbedding and lossGrad * sourceEmbedding.
    /// </summary>
    private void AddEdgeGradients(
        Tensor<T> edges, Tensor<T> lossGrad, List<Tensor<int>> indexParts, List<Tensor<T>> updateParts)
    {
        int numEdges = edges.Shape[0];
        var srcIdx = EdgeEndpoints(edges, 0);
        var tgtIdx = EdgeEndpoints(edges, 1);
        var scale = lossGrad.Reshape(numEdges, 1);
        var srcGrad = Engine.TensorMultiply(GatherEmbeddings(tgtIdx), scale);
        var tgtGrad = Engine.TensorMultiply(GatherEmbeddings(srcIdx), scale);

        indexParts.Add(Engine.TensorConcatenate(
            new[] { srcIdx.Reshape(numEdges, 1), tgtIdx.Reshape(numEdges, 1) }, 1).Reshape(2 * numEdges));
        updateParts.Add(Engine.TensorConcatenate(
            new[] { srcGrad.Reshape(numEdges, 1, EmbeddingDim), tgtGrad.Reshape(numEdges, 1, EmbeddingDim) }, 1)
            .Reshape(2 * numEdges, EmbeddingDim));
    }

    private double ComputeBCELoss(Tensor<T> posScores, Tensor<T> negScores)
    {
        int numPos = posScores.Shape[0];
        int numNeg = negScores.Shape[0];
        double loss = 0.0;
        T one = NumOps.One;
        T epsilon = NumOps.FromDouble(1e-10);

        // Vectorized computation for positive scores: loss = -log(sigmoid(score))
        if (numPos > 0)
        {
            // sigmoid(x) = 1 / (1 + exp(-x))
            // -log(sigmoid(x)) = log(1 + exp(-x))
            var negPosScores = Engine.TensorMultiplyScalar(posScores, NumOps.FromDouble(-1.0));
            var expNegPos = Engine.TensorExp(negPosScores);
            var onePlusExp = Engine.TensorAddScalar(expNegPos, one);
            var clamped = Engine.TensorClamp(onePlusExp, epsilon, NumOps.FromDouble(1e10));
            var logValues = Engine.TensorLog(clamped);
            T posLoss = Engine.TensorSum(logValues);
            loss += NumOps.ToDouble(posLoss);
        }

        // Vectorized computation for negative scores: loss = -log(1 - sigmoid(score))
        if (numNeg > 0)
        {
            // 1 - sigmoid(x) = exp(-x) / (1 + exp(-x)) = 1 / (1 + exp(x))
            // -log(1 - sigmoid(x)) = log(1 + exp(x))
            var expNeg = Engine.TensorExp(negScores);
            var onePlusExp = Engine.TensorAddScalar(expNeg, one);
            var clamped = Engine.TensorClamp(onePlusExp, epsilon, NumOps.FromDouble(1e10));
            var logValues = Engine.TensorLog(clamped);
            T negLoss = Engine.TensorSum(logValues);
            loss += NumOps.ToDouble(negLoss);
        }

        return (numPos + numNeg) > 0 ? loss / (numPos + numNeg) : 0.0;
    }

    private double ComputeAUC(Tensor<T> posScores, Tensor<T> negScores)
    {
        int numPos = posScores.Shape[0];
        int numNeg = negScores.Shape[0];
        int totalPairs = numPos * numNeg;
        if (totalPairs == 0)
        {
            return 0.5;
        }

        // Count (positive, negative) pairs where the positive edge scores strictly higher
        var pairShape = new[] { numPos, numNeg };
        var pos = Engine.TensorBroadcastTo(posScores.Reshape(numPos, 1), pairShape);
        var neg = Engine.TensorBroadcastTo(negScores.Reshape(1, numNeg), pairShape);
        int correctRankings = Engine.TensorCountNonzero(Engine.TensorGreaterThan(pos, neg));

        return (double)correctRankings / totalPairs;
    }

    #region Abstract Method Implementations

    /// <summary>
    /// Makes a prediction using the trained network.
    /// </summary>
    /// <param name="input">The input tensor containing node features.</param>
    /// <returns>The node embeddings tensor.</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        EnsureDefaultAdjacencyForInput(input);

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return Forward(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Trains the node embeddings this generic contract outputs (see <c>PredictCore</c>; edge scoring is the separate
    /// <c>PredictEdges</c> decode) over the input's graph. It needs a tape-differentiable loss: substituting one would
    /// train a different objective than the caller configured.
    /// </remarks>
    protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> target)
    {
        EnsureDefaultAdjacencyForInput(input);
        if (_lossFunction is not LossFunctionBase<T> tapeLoss)
            throw new InvalidOperationException(
                $"LinkPredictionModel tape-based training requires a LossFunctionBase<T> (one that "
                + $"implements ComputeTapeLoss); the configured loss '{_lossFunction.GetType().Name}' is "
                + "not tape-differentiable. Supply a LossFunctionBase<T>-derived loss such as BinaryCrossEntropyWithLogitsLoss.");
        var adjacency = _cachedAdjacencyMatrix ?? throw new InvalidOperationException(
            "Adjacency matrix must be set before training. Call SetAdjacencyMatrix() first.");
        return GraphTrainingObjective(input, adjacency, target, Forward, tapeLoss);
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
                ["NetworkType"] = "LinkPredictionModel",
                ["InputFeatures"] = InputFeatures,
                ["EmbeddingDim"] = EmbeddingDim,
                ["HiddenDim"] = HiddenDim,
                ["NumLayers"] = NumLayers,
                ["DropoutRate"] = DropoutRate,
                ["DecoderType"] = _decoderType.ToString()
            }
        };
    }

    #endregion
}
