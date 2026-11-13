using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.Document.GraphBased;

/// <summary>
/// LayoutGraph: graph-based document layout analysis after GLAM (Wang et al. 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LayoutGraph constructs and analyzes graphs from document layouts, where nodes
/// represent document elements and edges encode spatial relationships: as in GLAM, a node-feature input
/// <c>[N, F]</c> whose first four columns are each element's normalized box is joined into a k-nearest-centre
/// page graph, graph convolutions propagate over it, and each node is classified with a softmax. It excels
/// at understanding hierarchical document structures.
/// </para>
/// <para>
/// <b>For Beginners:</b> LayoutGraph analyzes how document parts relate:
/// 1. Builds a graph from document structure
/// 2. Models reading order and containment
/// 3. Learns hierarchical relationships
/// 4. Predicts document element types and groupings
///
/// Key features:
/// - Hierarchical graph construction
/// - Spatial relationship modeling
/// - Reading order prediction
/// - Multi-level layout understanding
///
/// Example usage:
/// <code>
/// var model = new LayoutGraph&lt;float&gt;(architecture);
/// var result = model.DetectLayout(documentImage);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Graphical Approach to Document Layout Analysis", "https://doi.org/10.48550/arXiv.2308.02051", Year = 2023, Authors = "Jilin Wang, Michael Krumdick, Baojia Tong, Hamima Halim, Maxim Sokolov, Vadym Barda, Delphine Vendryes, Chris Tanner")]
public partial class LayoutGraph<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IReadingOrderDetector<T>
{
    private readonly LayoutGraphOptions _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _nodeDim;
    private int _edgeDim;
    private int _graphLayers;
    private int _numClasses;
    private int _maxNodes;

    // Native mode layers
    private readonly List<ILayer<T>> _nodeEncoderLayers = [];
    private readonly List<ILayer<T>> _edgeEncoderLayers = [];
    private readonly List<ILayer<T>> _graphLayersList = [];
    private readonly List<ILayer<T>> _outputLayers = [];

    // Embeddings

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the node dimension.
    /// </summary>
    public int NodeDim => _nodeDim;

    /// <inheritdoc/>
    public IReadOnlyList<LayoutElementType> SupportedElementTypes { get; } =
    [
        LayoutElementType.Text,
        LayoutElementType.Title,
        LayoutElementType.List,
        LayoutElementType.Table,
        LayoutElementType.Figure,
        LayoutElementType.Caption,
        LayoutElementType.Header,
        LayoutElementType.Footer,
        LayoutElementType.FormField
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a LayoutGraph model using a pre-trained ONNX model for inference.
    /// </summary>
    public LayoutGraph(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        LayoutGraphOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutGraphOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        // Validated after the path checks so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        _nodeDim = _options.NodeDim;
        _edgeDim = _options.EdgeDim;
        _graphLayers = _options.GraphLayers;
        _numClasses = _options.NumClasses;
        _maxNodes = _options.MaxNodes;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a LayoutGraph model using native layers for training and inference.
    /// </summary>
    public LayoutGraph(
        NeuralNetworkArchitecture<T> architecture,
        LayoutGraphOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutGraphOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _nodeDim = _options.NodeDim;
        _edgeDim = _options.EdgeDim;
        _graphLayers = _options.GraphLayers;
        _numClasses = _options.NumClasses;
        _maxNodes = _options.MaxNodes;
        // Honor the model's configured LearningRate (the bare AdamOptimizer(this) ignored it and ran at Adam's
        // 0.001) and enable gradient clipping so graph-conv training does not drift upward over more iterations
        // (MoreData saw 200-iter loss 2.40 -> 2.82). Fully user-overridable via the optimizer parameter and
        // LayoutGraphOptions.LearningRate. (#1789)
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate, EnableGradientClipping = true, MaxGradientNorm = 1.0 });

        // Route base tape training through the configured optimizer. Previously _optimizer was stored but
        // never used — TrainWithTape resolved the default base optimizer, so a caller-supplied optimizer
        // was silently ignored. Install it as the base-train optimizer (matches SVTR<T>).
        SetBaseTrainOptimizer(_optimizer);

        InitializeLayers();
        InitializeEmbeddings();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultLayoutGraphLayers(
            inputDim: _nodeDim,
            hiddenDim: _edgeDim,
            numGraphLayers: _graphLayers,
            numClasses: _numClasses,
            maxNodes: _maxNodes));
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

    }

    #endregion

    #region ILayoutDetector Implementation

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage)
    {
        return DetectLayout(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var regions = ParseLayoutOutput(output, confidenceThreshold);

        return new DocumentLayoutResult<T>
        {
            Regions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private List<LayoutRegion<T>> ParseLayoutOutput(Tensor<T> output, double threshold)
        => DocumentTensorOps.GraphLayoutRegions(Engine, output, _maxNodes, _numClasses, ImageSize, threshold);

    #endregion

    #region IReadingOrderDetector Implementation

    /// <inheritdoc/>
    public ReadingOrderResult<T> DetectReadingOrder(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var orderedElements = PredictReadingOrder(output);

        return new ReadingOrderResult<T>
        {
            OrderedElements = orderedElements,
            Confidence = DocumentTensorOps.Mean(Engine, orderedElements.Select(e => e.Confidence).ToArray()),
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public ReadingOrderResult<T> DetectReadingOrder(DocumentLayoutResult<T> layoutResult)
    {
        var orderedElements = layoutResult.Regions
            .OrderBy(r => r.Index)
            .Select((r, idx) => new OrderedElement<T>
            {
                ElementIndex = r.Index,
                ReadingOrderPosition = idx,
                Confidence = r.Confidence
            })
            .ToList();

        return new ReadingOrderResult<T>
        {
            OrderedElements = orderedElements,
            Confidence = DocumentTensorOps.Mean(Engine, orderedElements.Select(e => e.Confidence).ToArray()),
            ProcessingTimeMs = 0
        };
    }

    internal List<OrderedElement<T>> PredictReadingOrder(Tensor<T> output)
    {
        // Each node row's confidence is its softmax probability of the predicted class; the order is the node order.
        int width = output.Shape.Length > 1 ? output.Shape[output.Shape.Length - 1] : output.Length;
        int rows = width == 0 ? 0 : output.Length / width;
        int numNodes = Math.Min(rows, _maxNodes);
        int classes = Math.Min(width, _numClasses);
        if (numNodes <= 0 || classes <= 0)
            return new List<OrderedElement<T>>();

        var logits = Engine.TensorSlice(Engine.Reshape(output, new[] { rows, width }), new[] { 0, 0 }, new[] { numNodes, classes });
        T[] confidences = DocumentTensorOps.RowMax(Engine, Engine.Softmax(logits, 1), classes).Max.ToArray();
        return Enumerable.Range(0, numNodes).Select(i => new OrderedElement<T>
        {
            ElementIndex = i,
            ReadingOrderPosition = i,
            Confidence = confidences[i]
        }).ToList();
    }

    /// <summary>
    /// GLAM's page graph (Wang et al. 2023, Sec. 3.1): every node is joined to its
    /// <paramref name="neighbours"/> nearest nodes by box-centre distance (ties at the k-th distance included),
    /// made symmetric, with self-loops, and normalized as <c>D^-1/2 (A + I) D^-1/2</c> for the graph
    /// convolutions. <paramref name="boxes"/> is <c>[N, 4]</c> normalized <c>(x1, y1, x2, y2)</c>.
    /// </summary>
    internal static Tensor<T> SpatialAdjacency(IEngine engine, Tensor<T> boxes, int neighbours)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = boxes.Shape[0];
        var centers = engine.TensorMultiplyScalar(engine.TensorAdd(
            engine.TensorSlice(boxes, new[] { 0, 0 }, new[] { n, 2 }),
            engine.TensorSlice(boxes, new[] { 0, 2 }, new[] { n, 2 })), numOps.FromDouble(0.5));
        var diff = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(centers, new[] { n, 1, 2 }), new[] { n, n, 2 }),
            engine.TensorBroadcastTo(engine.Reshape(centers, new[] { 1, n, 2 }), new[] { n, n, 2 }));
        var distance = engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 2 }, keepDims: false);

        int k = Math.Max(0, Math.Min(neighbours, n - 1));
        var sorted = engine.TensorSort(distance, 1, false).Values;
        var radius = engine.TensorBroadcastTo(engine.TensorSlice(sorted, new[] { 0, k }, new[] { n, 1 }), new[] { n, n });
        var near = engine.TensorGreaterEqual(radius, distance);
        var adjacency = engine.TensorMax(near, engine.TensorTranspose(near));
        var inverseRoot = engine.TensorRsqrt(engine.ReduceSum(adjacency, new[] { 1 }, keepDims: true));
        return engine.TensorMultiply(
            engine.TensorMultiply(adjacency, engine.TensorBroadcastTo(inverseRoot, new[] { n, n })),
            engine.TensorBroadcastTo(engine.Reshape(inverseRoot, new[] { 1, n }), new[] { n, n }));
    }

    /// <summary>Spatial neighbours per node in the page graph.</summary>
    internal const int SpatialNeighbours = 4;

    /// <summary>
    /// Wires the page graph into every graph convolution before a pass: a node-feature input <c>[N, F]</c> whose
    /// first four columns are the node's normalized box gets <see cref="SpatialAdjacency"/>; any other input
    /// (a page image) has no node boxes, so its rows propagate through self-loops only.
    /// </summary>
    private void WireSpatialGraph(Tensor<T> input)
    {
        var graphLayers = Layers.OfType<GraphConvolutionalLayer<T>>().ToList();
        if (graphLayers.Count == 0)
            return;
        var adjacency = input.Rank == 2 && input.Shape[0] > 0 && input.Shape[1] >= 4
            ? SpatialAdjacency(Engine, Engine.StopGradient(Engine.TensorSlice(input, new[] { 0, 0 }, new[] { input.Shape[0], 4 })), SpatialNeighbours)
            : Engine.TensorEye<T>(input.Rank >= 2 ? input.Shape[input.Rank - 2] : 1);
        foreach (var layer in graphLayers)
            layer.SetAdjacencyMatrix(adjacency);
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("LayoutGraph Model Summary");
        sb.AppendLine("=========================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: GLAM-style spatial graph convolution network");
        sb.AppendLine($"Node Dimension: {_nodeDim}");
        sb.AppendLine($"Edge Dimension: {_edgeDim}");
        sb.AppendLine($"Graph Layers: {_graphLayers}");
        sb.AppendLine($"Max Nodes: {_maxNodes}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Reading Order: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies LayoutGraph's industry-standard preprocessing: simple normalization to [0,1].
    /// </summary>
    /// <remarks>
    /// LayoutGraph uses basic normalization (divide by 255) since the focus is on graph-based layout analysis.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
        => Engine.TensorDivideScalar(EnsureBatchDimension(rawImage), NumOps.FromDouble(255.0));

    /// <summary>
    /// Applies LayoutGraph's industry-standard postprocessing: pass-through (graph node classifications are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "LayoutGraph",
            Description = "LayoutGraph for hierarchical document layout analysis",
            FeatureCount = _nodeDim,
            Complexity = _graphLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "node_dim", _nodeDim },
                { "edge_dim", _edgeDim },
                { "graph_layers", _graphLayers },
                { "num_classes", _numClasses },
                { "max_nodes", _maxNodes },
                { "use_native_mode", _useNativeMode }
            },
            ModelDataProvider = () => SafeSerialize()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <summary>
    /// Inference forward: runs the graph layer stack and returns the per-node class logits
    /// [numNodes, numClasses] UNCHANGED. DetectLayout, DetectReadingOrder, and ParseLayoutOutput index
    /// this rank-2 output per node (output[node, class]); pooling it to a rank-1 [numClasses] vector here
    /// would collapse every node and break their two-dimensional indexing. The document-level pooling
    /// needed to align with the classification target happens only on the training path
    /// (<see cref="ForwardForTraining"/>).
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        WireSpatialGraph(input);
        // Unchanged when no node types are supplied -- the plain walk below IS the previous
        // behaviour, and keeping it byte-identical matters: routing the default path through a
        // hand-written walk instead of the base one made the analytic and finite-difference
        // gradients disagree on every sampled parameter, because the base path owns dropout,
        // seed wiring and checkpointing that a bare Layers[i].Forward loop does not reproduce.
        if (AuxiliaryInput is null || AuxiliaryInput.Length == 0)
        {
            var plain = input;
            foreach (var layer in Layers) plain = layer.Forward(plain);
            return plain;
        }

        return RunWithNodeTypes(input);
    }

    /// <summary>
    /// Training forward: runs the base training forward (which wires layer seeds and applies gradient
    /// checkpointing / weight streaming over the graph layers), then mean-pools every axis but the class
    /// axis so the per-node logits [numNodes, numClasses] reduce to the document-level [numClasses] logit
    /// vector the classification target expects. Without this pooling the rank-2 tensor cannot align to
    /// the rank-1 [numClasses] target and CrossEntropyWithLogits over-indexes ClassIndicesToOneHot and
    /// throws. All ops are tape-aware, so training back-propagates through the pool into the graph layers.
    /// Inference (PredictCore → Forward) deliberately skips this pooling to keep the per-node output.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        WireSpatialGraph(input);
        // Default path stays on base.ForwardForTraining, which owns seed wiring, gradient
        // checkpointing and weight streaming. Only a call that actually supplies node types diverts,
        // and that one wires seeds itself per EnsureLayerRandomSeedsWired's contract.
        Tensor<T> output;
        if (AuxiliaryInput is null || AuxiliaryInput.Length == 0)
        {
            output = base.ForwardForTraining(input);
        }
        else
        {
            EnsureLayerRandomSeedsWired();
            output = RunWithNodeTypes(input);
        }

        if (output.Shape.Length >= 2)
        {
            int classAxis = output.Shape.Length - 1;
            var poolAxes = new int[classAxis];
            for (int a = 0; a < classAxis; a++) poolAxes[a] = a;
            output = Engine.ReduceMean(output, poolAxes, keepDims: false); // → [numClasses]
        }
        return output;
    }

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.


    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    /// <remarks>
    /// Replaces a hand-written throw that used to sit inside UpdateParameters. The base checks this
    /// on every mutating entry point rather than the one member the throw happened to guard, and
    /// reading -- ParameterCount and GetParameters -- stays available either way.
    /// </remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion

    #region Node-type fusion

    /// <summary>
    /// The per-node TYPE embedding. Held OUTSIDE Layers on purpose: it is not a step in the
    /// sequential chain, and putting it there fed an index lookup a graph hidden state. It reaches
    /// the parameter surface and the optimizer through GetExtraTrainableLayers instead, which is the
    /// base's existing hook for exactly this -- a trainable layer that the chain does not walk.
    /// </summary>
    private readonly EmbeddingLayer<T> _nodeTypeEmbedding = new(NodeTypeCount, NodeTypeDim);

    private const int NodeTypeCount = 64;
    private const int NodeTypeDim = 256;

    /// <summary>
    /// Runs the graph stack, adding a learned per-node TYPE vector when the caller supplies type ids
    /// through the auxiliary input.
    /// </summary>
    /// <remarks>
    /// _nodeTypeEmbeddings used to be a model field nothing read. Type is not derivable from the node
    /// features -- it is a label the layout parser assigns ("title", "caption", "table") -- so unlike
    /// node order it needs an input, which is what the base's auxiliary slot provides. Both forwards
    /// route here, so the type embedding is on the gradient tape. With no type ids the model behaves
    /// exactly as before.
    /// </remarks>
    private Tensor<T> RunWithNodeTypes(Tensor<T> input)
    {
        // Project to the graph hidden width first, so the type vector is added in that space rather
        // than to the raw node features.
        var hidden = Layers[0].Forward(input);

        var types = AuxiliaryInput;
        if (types is not null && types.Length > 0)
        {
            var typeVectors = _nodeTypeEmbedding.Forward(types);
            if (typeVectors.Rank == hidden.Rank && typeVectors.Length == hidden.Length)
            {
                hidden = Engine.TensorAdd(hidden, typeVectors);
            }
        }

        for (int i = 1; i < Layers.Count; i++)
        {
            hidden = Layers[i].Forward(hidden);
        }

        return hidden;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The base walks Layers as a chain and would hand the appended type table a graph hidden state.
    /// This is the fourth site of that same pattern in this family (LiLT, SVTR, DocOwl were the
    /// others), so the walk is reused rather than re-derived.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

        var activations = new Dictionary<string, Tensor<T>>();
        var hidden = Layers[0].Forward(input);
        activations[$"0_{Layers[0].GetType().Name}"] = hidden;

        var types = AuxiliaryInput;
        if (types is not null && types.Length > 0)
        {
            var typeVectors = _nodeTypeEmbedding.Forward(types);
            activations["node_type_embedding"] = typeVectors;
            if (typeVectors.Rank == hidden.Rank && typeVectors.Length == hidden.Length)
            {
                hidden = Engine.TensorAdd(hidden, typeVectors);
            }
        }

        for (int i = 1; i < Layers.Count; i++)
        {
            hidden = Layers[i].Forward(hidden);
            activations[$"{i}_{Layers[i].GetType().Name}"] = hidden;
        }

        return activations;
    }

    #endregion
}
