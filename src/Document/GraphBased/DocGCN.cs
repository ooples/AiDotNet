using AiDotNet.Document.Interfaces;
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
/// DocGCN (Document Graph Convolutional Network) for document understanding using graph neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DocGCN represents documents as graphs where nodes are text blocks and edges represent
/// spatial and semantic relationships. Graph convolutional layers propagate information
/// to understand document structure.
/// </para>
/// <para>
/// <b>For Beginners:</b> DocGCN views documents as networks:
/// 1. Each text block becomes a node in a graph
/// 2. Nearby blocks are connected by edges
/// 3. Graph convolutions learn relationships
/// 4. Can classify, extract, or understand document structure
///
/// Key features:
/// - Graph-based document representation
/// - Spatial relationship modeling
/// - Multi-hop reasoning through graph layers
/// - Entity and relation extraction
///
/// Example usage:
/// <code>
/// var model = new DocGCN&lt;float&gt;(architecture);
/// var result = model.DetectLayout(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> Based on graph neural network approaches for document understanding.
/// </para>
/// </remarks>
public class DocGCN<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _nodeDim;
    private int _edgeDim;
    private int _gcnLayers;
    private int _numClasses;
    private int _maxNodes;

    // Native mode layers
    private readonly List<ILayer<T>> _nodeEncoderLayers = [];
    private readonly List<ILayer<T>> _gcnLayersList = [];
    private readonly List<ILayer<T>> _classifierLayers = [];

    // Node embeddings
    private Tensor<T>? _nodeEmbeddings;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the node feature dimension.
    /// </summary>
    public int NodeDim => _nodeDim;

    /// <summary>
    /// Gets the number of GCN layers.
    /// </summary>
    public int NumGCNLayers => _gcnLayers;

    /// <summary>
    /// Gets the maximum number of nodes.
    /// </summary>
    public int MaxNodes => _maxNodes;

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
    /// Creates a DocGCN model using a pre-trained ONNX model for inference.
    /// </summary>
    public DocGCN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int nodeDim = 256,
        int edgeDim = 64,
        int gcnLayers = 3,
        int numClasses = 9,
        int maxNodes = 512,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _nodeDim = nodeDim;
        _edgeDim = edgeDim;
        _gcnLayers = gcnLayers;
        _numClasses = numClasses;
        _maxNodes = maxNodes;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DocGCN model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration:</b>
    /// - Node feature encoder (text + spatial)
    /// - Multiple GCN layers with message passing
    /// - Edge-aware attention mechanism
    /// - Node classification head
    /// </para>
    /// </remarks>
    public DocGCN(
        NeuralNetworkArchitecture<T> architecture,
        int nodeDim = 256,
        int edgeDim = 64,
        int gcnLayers = 3,
        int numClasses = 9,
        int maxNodes = 512,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _nodeDim = nodeDim;
        _edgeDim = edgeDim;
        _gcnLayers = gcnLayers;
        _numClasses = numClasses;
        _maxNodes = maxNodes;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        Layers.AddRange(LayerHelper<T>.CreateDefaultDocGCNLayers(
            inputDim: _nodeDim,
            hiddenDim: _edgeDim,
            numGCNLayers: _gcnLayers,
            numClasses: _numClasses));
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        _nodeEmbeddings = Tensor<T>.CreateDefault([_maxNodes, _nodeDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_nodeEmbeddings, random, 0.02);
    }

    private void InitializeWithSmallRandomValues(Tensor<T> tensor, Random random, double stdDev)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            tensor.Data[i] = NumOps.FromDouble(randStdNormal * stdDev);
        }
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
    {
        var regions = new List<LayoutRegion<T>>();
        int numNodes = Math.Min(output.Shape[0], _maxNodes);
        int hiddenDim = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;
        int numClasses = Math.Min(hiddenDim - 4, _numClasses); // Reserve 4 for bbox
        bool hasBbox = hiddenDim > _numClasses;

        for (int i = 0; i < numNodes; i++)
        {
            double maxConf = 0;
            int maxClass = 0;
            for (int c = 0; c < numClasses; c++)
            {
                double conf = NumOps.ToDouble(output[i, c]);
                if (conf > maxConf) { maxConf = conf; maxClass = c; }
            }

            if (maxConf >= threshold && maxClass > 0)
            {
                // Extract bounding box from last 4 values (normalized coordinates)
                Vector<T> bbox;
                if (hasBbox && hiddenDim >= 4)
                {
                    int bboxStart = hiddenDim - 4;
                    double x1 = NumOps.ToDouble(output[i, bboxStart]) * ImageSize;
                    double y1 = NumOps.ToDouble(output[i, bboxStart + 1]) * ImageSize;
                    double x2 = NumOps.ToDouble(output[i, bboxStart + 2]) * ImageSize;
                    double y2 = NumOps.ToDouble(output[i, bboxStart + 3]) * ImageSize;

                    bbox = new Vector<T>([
                        NumOps.FromDouble(Math.Max(0, x1)),
                        NumOps.FromDouble(Math.Max(0, y1)),
                        NumOps.FromDouble(Math.Min(ImageSize, x2)),
                        NumOps.FromDouble(Math.Min(ImageSize, y2))
                    ]);
                }
                else
                {
                    // Grid-based fallback for node index
                    int gridSize = (int)Math.Sqrt(numNodes);
                    int cellSize = ImageSize / Math.Max(1, gridSize);
                    int row = i / gridSize;
                    int col = i % gridSize;

                    bbox = new Vector<T>([
                        NumOps.FromDouble(col * cellSize),
                        NumOps.FromDouble(row * cellSize),
                        NumOps.FromDouble((col + 1) * cellSize),
                        NumOps.FromDouble((row + 1) * cellSize)
                    ]);
                }

                regions.Add(new LayoutRegion<T>
                {
                    ElementType = (LayoutElementType)Math.Min(maxClass, (int)LayoutElementType.Other),
                    Confidence = NumOps.FromDouble(maxConf),
                    ConfidenceValue = maxConf,
                    Index = i,
                    BoundingBox = bbox
                });
            }
        }

        return regions;
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
        sb.AppendLine("DocGCN Model Summary");
        sb.AppendLine("====================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Graph Convolutional Network");
        sb.AppendLine($"Node Dimension: {_nodeDim}");
        sb.AppendLine($"Edge Dimension: {_edgeDim}");
        sb.AppendLine($"GCN Layers: {_gcnLayers}");
        sb.AppendLine($"Max Nodes: {_maxNodes}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Graph-Based: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies DocGCN's industry-standard preprocessing: simple normalization to [0,1].
    /// </summary>
    /// <remarks>
    /// DocGCN uses basic normalization (divide by 255) since the focus is on graph-based processing.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);
        for (int i = 0; i < image.Data.Length; i++)
        {
            normalized.Data[i] = NumOps.FromDouble(NumOps.ToDouble(image.Data[i]) / 255.0);
        }
        return normalized;
    }

    /// <summary>
    /// Applies DocGCN's industry-standard postprocessing: pass-through (node classifications are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DocGCN",
            ModelType = ModelType.NeuralNetwork,
            Description = "DocGCN for graph-based document understanding",
            FeatureCount = _nodeDim,
            Complexity = _gcnLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "node_dim", _nodeDim },
                { "edge_dim", _edgeDim },
                { "gcn_layers", _gcnLayers },
                { "num_classes", _numClasses },
                { "max_nodes", _maxNodes },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_nodeDim);
        writer.Write(_edgeDim);
        writer.Write(_gcnLayers);
        writer.Write(_numClasses);
        writer.Write(_maxNodes);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _nodeDim = reader.ReadInt32();
        _edgeDim = reader.ReadInt32();
        _gcnLayers = reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        _maxNodes = reader.ReadInt32();
        _ = reader.ReadBoolean(); // useNativeMode - already set by constructor
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DocGCN<T>(Architecture, _nodeDim, _edgeDim, _gcnLayers, _numClasses, _maxNodes);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var gradient = Tensor<T>.FromVector(
            LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector()));

        for (int i = Layers.Count - 1; i >= 0; i--)
            gradient = Layers[i].Backward(gradient);

        UpdateParameters(CollectGradients());
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        var currentParams = GetParameters();
        T lr = NumOps.FromDouble(0.001);
        for (int i = 0; i < currentParams.Length; i++)
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(lr, gradients[i]));
        SetParameters(currentParams);
    }

    private Vector<T> CollectGradients()
    {
        var grads = new List<T>();
        foreach (var layer in Layers)
            grads.AddRange(layer.GetParameterGradients());
        return new Vector<T>([.. grads]);
    }

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
}
