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
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.Document.Analysis.TableDetection;

/// <summary>
/// TableTransformer for table detection and structure recognition using DETR-style architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TableTransformer is based on the DETR (DEtection TRansformer) architecture, adapted for
/// table detection and table structure recognition. It can detect tables in documents and
/// identify their internal structure (rows, columns, cells, headers).
/// </para>
/// <para>
/// <b>For Beginners:</b> TableTransformer helps computers understand tables in documents.
/// It can:
/// 1. Find where tables are located in a page (table detection)
/// 2. Identify the structure within tables - rows, columns, and cells (structure recognition)
/// 3. Handle both bordered and borderless tables
///
/// Example usage:
/// <code>
/// var tableModel = new TableTransformer&lt;float&gt;(architecture);
/// var tables = tableModel.DetectTables(documentImage);
/// foreach (var table in tables)
/// {
///     var structure = tableModel.RecognizeStructure(table.Image);
///     Console.WriteLine($"Table has {structure.NumRows} rows and {structure.NumColumns} columns");
/// }
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "PubTables-1M: Towards Comprehensive Table Extraction from Unstructured Documents" (CVPR 2022)
/// https://arxiv.org/abs/2110.00061
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PubTables-1M: Towards Comprehensive Table Extraction from Unstructured Documents", "https://doi.org/10.48550/arXiv.2110.00061", Year = 2022, Authors = "Brandon Smock, Rohith Pesala, Robin Abraham")]
public partial class TableTransformer<T> : DocumentNeuralNetworkBase<T>, ITableExtractor<T>
{
    private readonly TableTransformerOptions _options;

    #region Fields

    private bool _useNativeMode;
    private InferenceSession? _onnxDetectionSession;
    private InferenceSession? _onnxStructureSession;
    private string? _onnxDetectionModelPath;
    private string? _onnxStructureModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _hiddenDim;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _numHeads;
    private int _numQueries;
    private int _numTableClasses;
    private int _numStructureClasses;

    // Native mode layers
    private readonly List<ILayer<T>> _backboneLayers = [];
    private readonly List<ILayer<T>> _encoderLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];
    private readonly List<ILayer<T>> _detectionHead = [];
    private readonly List<ILayer<T>> _structureHead = [];

    // Learnable object queries
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _objectQueries;

    // Task mode - tracks whether we're doing detection or structure recognition
#pragma warning disable CS0414 // Field is assigned but its value is never used - kept for future use in task-specific processing
    private TableTransformerTask _currentTask = TableTransformerTask.Detection;
#pragma warning restore CS0414

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public bool SupportsBorderedTables => true;

    /// <inheritdoc/>
    public bool SupportsBorderlessTables => true;

    /// <inheritdoc/>
    public bool SupportsMergedCells => true;

    /// <summary>
    /// Gets the number of object queries used in DETR decoder.
    /// </summary>
    public int NumQueries => _numQueries;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TableTransformer model using pre-trained ONNX models for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="detectionModelPath">Path to the table detection ONNX model.</param>
    /// <param name="structureModelPath">Path to the structure recognition ONNX model.</param>
    /// <param name="imageSize">Expected input image size (default: 800).</param>
    /// <param name="hiddenDim">Transformer hidden dimension (default: 256).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 6).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="numQueries">Number of object queries (default: 100).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <exception cref="ArgumentNullException">Thrown if model paths are null.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX model files don't exist.</exception>
    public TableTransformer(
        NeuralNetworkArchitecture<T> architecture,
        string detectionModelPath,
        string structureModelPath,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        TableTransformerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new TableTransformerOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(detectionModelPath))
            throw new ArgumentNullException(nameof(detectionModelPath));
        if (string.IsNullOrWhiteSpace(structureModelPath))
            throw new ArgumentNullException(nameof(structureModelPath));
        if (!File.Exists(detectionModelPath))
            throw new FileNotFoundException($"Detection model not found: {detectionModelPath}", detectionModelPath);
        if (!File.Exists(structureModelPath))
            throw new FileNotFoundException($"Structure model not found: {structureModelPath}", structureModelPath);

        _useNativeMode = false;
        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _numHeads = _options.NumHeads;
        _numQueries = _options.NumQueries;
        // Previously the literals 2 and 7 with their meanings only in trailing comments. Both
        // size a classifier head, so they are part of the model's shape and belong on the options.
        _numTableClasses = _options.NumTableClasses;
        _numStructureClasses = _options.NumStructureClasses;
        // DETR fine-tunes well below Adam's generic 1e-3, with gradient-norm clipping. The rate
        // now comes from the options; the clipping stays, being a stability measure rather than a
        // tunable the paper specifies.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                EnableGradientClipping = true,
                MaxGradientNorm = 1.0,
            });

        ImageSize = _options.ImageSize;

        _onnxDetectionModelPath = detectionModelPath;
        _onnxStructureModelPath = structureModelPath;
        InitializeOnnxSessions();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a TableTransformer model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="imageSize">Expected input image size (default: 800).</param>
    /// <param name="hiddenDim">Transformer hidden dimension (default: 256).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 6).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="numQueries">Number of object queries (default: 100).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (from CVPR 2022 paper):</b>
    /// - Backbone: ResNet-18 (for detection) or ResNet-50 (for structure)
    /// - Transformer: 6 encoder layers, 6 decoder layers, 256 hidden dim
    /// - Object queries: 100
    /// - Image size: 800
    /// </para>
    /// </remarks>
    public TableTransformer(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        TableTransformerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new TableTransformerOptions();
        Options = _options;

        _useNativeMode = true;
        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _numHeads = _options.NumHeads;
        _numQueries = _options.NumQueries;
        _numTableClasses = _options.NumTableClasses;
        _numStructureClasses = _options.NumStructureClasses;
        // TableTransformer is a DETR-based detector (Smock et al. 2022). DETR fine-tunes at 1e-4 with
        // gradient-norm clipping at 0.1-1.0; built bare, the optimizer ran on framework defaults.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                EnableGradientClipping = true,
                MaxGradientNorm = 1.0,
            });

        ImageSize = _options.ImageSize;

        InitializeLayers();
        InitializeObjectQueries();
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

        // Check if user provided custom layers
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        // Use LayerHelper to create default TableTransformer layers
        Layers.AddRange(LayerHelper<T>.CreateDefaultTableTransformerLayers(
            imageSize: ImageSize,
            hiddenDim: _hiddenDim,
            numEncoderLayers: _numEncoderLayers,
            numDecoderLayers: _numDecoderLayers,
            numHeads: _numHeads,
            numQueries: _numQueries,
            numStructureClasses: _numStructureClasses));
    }

    private void InitializeObjectQueries()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Xavier-scaled Box-Muller normals, drawn in the same (u1, u2) order per element as before.
        double scale = Math.Sqrt(2.0 / (_numQueries + _hiddenDim));
        var normals = new double[_numQueries * _hiddenDim];
        for (int i = 0; i < normals.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            normals[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2) * scale;
        }

        _objectQueries = new Tensor<double>(normals, new[] { _numQueries, _hiddenDim }).Cast<T>();
    }

    private void InitializeOnnxSessions()
    {
        if (string.IsNullOrWhiteSpace(_onnxDetectionModelPath))
            throw new InvalidOperationException("Detection ONNX model path is not set.");
        if (string.IsNullOrWhiteSpace(_onnxStructureModelPath))
            throw new InvalidOperationException("Structure ONNX model path is not set.");
        if (!File.Exists(_onnxDetectionModelPath))
            throw new FileNotFoundException($"Detection model not found: {_onnxDetectionModelPath}", _onnxDetectionModelPath);
        if (!File.Exists(_onnxStructureModelPath))
            throw new FileNotFoundException($"Structure model not found: {_onnxStructureModelPath}", _onnxStructureModelPath);

        _onnxDetectionSession?.Dispose();
        _onnxStructureSession?.Dispose();
        _onnxDetectionSession = new InferenceSession(_onnxDetectionModelPath);
        _onnxStructureSession = new InferenceSession(_onnxStructureModelPath);
    }

    #endregion

    #region ITableExtractor Implementation

    /// <inheritdoc/>
    public IEnumerable<TableRegion<T>> DetectTables(Tensor<T> documentImage)
    {
        return DetectTables(documentImage, 0.5);
    }

    /// <summary>
    /// Detects tables with a custom confidence threshold.
    /// </summary>
    public IEnumerable<TableRegion<T>> DetectTables(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);

        _currentTask = TableTransformerTask.Detection;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode
            ? Forward(preprocessed)
            : RunDetectionOnnx(preprocessed);

        return ParseTableDetections(output, documentImage, confidenceThreshold);
    }

    /// <inheritdoc/>
    public TableStructureResult<T> RecognizeStructure(Tensor<T> tableImage)
    {
        return RecognizeStructure(tableImage, 0.5);
    }

    /// <summary>
    /// Recognizes table structure with a custom confidence threshold.
    /// </summary>
    public TableStructureResult<T> RecognizeStructure(Tensor<T> tableImage, double confidenceThreshold)
    {
        ValidateImageShape(tableImage);

        _currentTask = TableTransformerTask.Structure;

        var preprocessed = PreprocessDocument(tableImage);
        var output = _useNativeMode
            ? Forward(preprocessed)
            : RunStructureOnnx(preprocessed);

        return ParseStructureOutput(output, tableImage, confidenceThreshold);
    }

    /// <inheritdoc/>
    public IEnumerable<List<List<string>>> ExtractTableContent(Tensor<T> documentImage)
    {
        var tables = DetectTables(documentImage);

        foreach (var table in tables)
        {
            if (table.Image is not null)
            {
                var structure = RecognizeStructure(table.Image);
                yield return structure.ToStringGrid();
            }
        }
    }

    /// <inheritdoc/>
    public string ExportTables(Tensor<T> documentImage, TableExportFormat format)
    {
        var tables = ExtractTableContent(documentImage).ToList();

        return format switch
        {
            TableExportFormat.CSV => ExportToCSV(tables),
            TableExportFormat.JSON => ExportToJSON(tables),
            TableExportFormat.HTML => ExportToHTML(tables),
            TableExportFormat.Markdown => ExportToMarkdown(tables),
            _ => ExportToCSV(tables)
        };
    }

    private Tensor<T> RunDetectionOnnx(Tensor<T> input)
    {
        if (_onnxDetectionSession is null)
            throw new InvalidOperationException("Detection ONNX session not initialized.");
        return RunOnnxInferenceWithSession(_onnxDetectionSession, input);
    }

    private Tensor<T> RunStructureOnnx(Tensor<T> input)
    {
        if (_onnxStructureSession is null)
            throw new InvalidOperationException("Structure ONNX session not initialized.");
        return RunOnnxInferenceWithSession(_onnxStructureSession, input);
    }

    private static Tensor<T> RunOnnxInferenceWithSession(InferenceSession session, Tensor<T> input)
    {
        if (session.InputMetadata.Count == 0)
            throw new InvalidOperationException("ONNX session has no inputs.");
        if (session.OutputMetadata.Count == 0)
            throw new InvalidOperationException("ONNX session has no outputs.");

        var inputName = session.InputMetadata.Keys.First();
        var elementType = session.InputMetadata[inputName].ElementType.ToString();
        var inputValue = OnnxTensorConverter.ToOnnxValue(inputName, input, elementType);
        using var results = session.Run([inputValue]);
        if (results.Count == 0)
            throw new InvalidOperationException("ONNX session returned no outputs.");

        if (results.Count == 1)
        {
            var result = results.FirstOrDefault()
                ?? throw new InvalidOperationException("ONNX session returned no outputs.");
            return OnnxTensorConverter.FromOnnxValue<T>(result)
                ?? throw new InvalidOperationException("Failed to convert ONNX output tensor.");
        }

        var outputs = results
            .Select(result => new OutputTensor(result.Name,
                OnnxTensorConverter.FromOnnxValue<T>(result)
                ?? throw new InvalidOperationException($"Failed to convert ONNX output tensor '{result.Name}'.")))
            .ToList();

        var boxesOutput = outputs.FirstOrDefault(o => NameMatches(o.Name, "pred_boxes", "boxes", "bbox"))
            ?? outputs.FirstOrDefault(o => o.Tensor.Rank >= 2 && o.Tensor.Shape[^1] == 4);
        var logitsOutput = outputs.FirstOrDefault(o => NameMatches(o.Name, "pred_logits", "logits", "class"))
            ?? outputs.FirstOrDefault(o => o.Tensor.Rank >= 2 && o.Tensor.Shape[^1] > 4);

        if (boxesOutput is null || logitsOutput is null)
        {
            var outputNames = string.Join(", ", outputs.Select(o => o.Name));
            throw new InvalidOperationException(
                $"ONNX session returned multiple outputs but pred_boxes/pred_logits were not found. Outputs: {outputNames}");
        }

        return CombineDetrOutputs(boxesOutput.Tensor, logitsOutput.Tensor);
    }

    private sealed class OutputTensor
    {
        public OutputTensor(string name, Tensor<T> tensor)
        {
            Name = name;
            Tensor = tensor;
        }

        public string Name { get; }
        public Tensor<T> Tensor { get; }
    }

    private static bool NameMatches(string name, params string[] tokens)
    {
        foreach (var token in tokens)
        {
            if (name.Contains(token, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    private static Tensor<T> CombineDetrOutputs(Tensor<T> boxes, Tensor<T> logits)
    {
        var boxesTensor = SqueezeBatch(boxes, "pred_boxes");
        var logitsTensor = SqueezeBatch(logits, "pred_logits");

        if (boxesTensor.Rank != 2 || logitsTensor.Rank != 2)
            throw new InvalidOperationException("DETR outputs must be 2D tensors after removing batch dimension.");
        if (boxesTensor.Shape[0] != logitsTensor.Shape[0])
            throw new InvalidOperationException("pred_boxes and pred_logits must have the same number of queries.");

        return AiDotNetEngine.Current.TensorConcatenate(new[] { boxesTensor, logitsTensor }, 1);
    }

    private static Tensor<T> SqueezeBatch(Tensor<T> tensor, string outputName)
    {
        if (tensor.Rank == 2)
            return tensor;
        if (tensor.Rank == 3 && tensor.Shape[0] == 1)
            return tensor.Slice(0);
        if (tensor.Rank == 3)
            throw new InvalidOperationException($"{outputName} output has batch size {tensor.Shape[0]}; only batch size 1 is supported.");

        throw new InvalidOperationException($"{outputName} output must be rank 2 or 3.");
    }

    private static (int Height, int Width) GetImageDimensions(Tensor<T> image)
    {
        if (image.Rank < 3)
            throw new ArgumentException("Expected image tensor with rank 3 or 4.", nameof(image));

        return (image.Shape[^2], image.Shape[^1]);
    }

    private static int Clamp(int value, int min, int max)
    {
        if (value < min)
            return min;
        if (value > max)
            return max;
        return value;
    }

    internal IEnumerable<TableRegion<T>> ParseTableDetections(Tensor<T> output, Tensor<T> originalImage, double threshold)
    {
        // Every class except DETR's trailing "no object" is a table (official weights: table, table rotated).
        var objects = DetrObjects(output, _numTableClasses, originalImage, threshold);
        return objects.Scores.Select((score, i) => new TableRegion<T>
        {
            BoundingBox = new Vector<T>(objects.Boxes[i]),
            Confidence = score,
            TableIndex = i,
            Image = CropTableImage(originalImage, objects.Pixels[i][0], objects.Pixels[i][1], objects.Pixels[i][2], objects.Pixels[i][3])
        }).ToList();
    }

    /// <summary>
    /// DETR's post-processing (Carion et al. 2020, as in Table Transformer's <c>postprocess</c>): each query row is
    /// <c>[cx, cy, w, h, class logits...]</c> whose LAST class is "no object". The class probabilities are the
    /// softmax over all logits; a query's label and score are the best class other than "no object". Boxes are
    /// normalized centre/size (the native head emits them as logits, so they get the head's sigmoid), converted to
    /// pixel corners and clipped to the image. Queries scoring below <paramref name="threshold"/> or with an empty
    /// box are dropped.
    /// </summary>
    internal (int[] Labels, T[] Scores, T[][] Boxes, double[][] Pixels) DetrObjects(Tensor<T> output, int numClasses, Tensor<T> image, double threshold)
    {
        var (imageHeight, imageWidth) = GetImageDimensions(image);
        int width = output.Shape.Length > 1 ? output.Shape[^1] : 4 + numClasses;
        int classes = width - 4;
        int count = classes > 1 ? output.Length / width : 0;
        if (count == 0)
            return (Array.Empty<int>(), Array.Empty<T>(), Array.Empty<T[]>(), Array.Empty<double[]>());

        var flat = Engine.Reshape(output, new[] { output.Length });
        if (flat.Length != count * width)
            flat = Engine.TensorSlice(flat, new[] { 0 }, new[] { count * width });
        var queries = Engine.Reshape(flat, new[] { count, width });
        var boxes = Engine.TensorSlice(queries, new[] { 0, 0 }, new[] { count, 4 });
        if (_useNativeMode)
            boxes = Engine.Sigmoid(boxes);
        var probabilities = Engine.Softmax(Engine.TensorSlice(queries, new[] { 0, 4 }, new[] { count, classes }), 1);
        var (labels, scores) = DocumentTensorOps.RowMax(Engine, probabilities, classes - 1);

        Tensor<T> Column(int c) => Engine.TensorSlice(boxes, new[] { 0, c }, new[] { count, 1 });
        T half = NumOps.FromDouble(0.5), pixelWidth = NumOps.FromDouble(imageWidth), pixelHeight = NumOps.FromDouble(imageHeight);
        var halfWidth = Engine.TensorMultiplyScalar(Column(2), half);
        var halfHeight = Engine.TensorMultiplyScalar(Column(3), half);
        Tensor<T> ToPixels(Tensor<T> t, T size) => Engine.TensorClamp(Engine.TensorMultiplyScalar(t, size), NumOps.Zero, size);
        var x1 = ToPixels(Engine.TensorSubtract(Column(0), halfWidth), pixelWidth);
        var y1 = ToPixels(Engine.TensorSubtract(Column(1), halfHeight), pixelHeight);
        var x2 = ToPixels(Engine.TensorAdd(Column(0), halfWidth), pixelWidth);
        var y2 = ToPixels(Engine.TensorAdd(Column(1), halfHeight), pixelHeight);

        var keep = Engine.TensorMultiply(
            Engine.TensorMultiply(Engine.TensorGreaterThan(x2, x1), Engine.TensorGreaterThan(y2, y1)),
            Engine.Reshape(DocumentTensorOps.AtLeast(Engine, scores, NumOps.FromDouble(threshold)), new[] { count, 1 }));
        int[] kept = DocumentTensorOps.Hits(Engine, Engine.Reshape(keep, new[] { count }));
        var corners = Engine.TensorConcatenate(new[] { x1, y1, x2, y2 }, 1);
        T[] values = corners.ToArray();
        double[] pixels = corners.Cast<double>().ToArray();
        return (kept.Select(i => labels[i]).ToArray(),
            DocumentTensorOps.Pick(Engine, scores, kept),
            kept.Select(i => values.Skip(4 * i).Take(4).ToArray()).ToArray(),
            kept.Select(i => pixels.Skip(4 * i).Take(4).ToArray()).ToArray());
    }

    /// <summary>Table Transformer's structure classes (the official order; "no object" is the last logit).</summary>
    private const int StructureColumn = 1, StructureRow = 2, StructureColumnHeader = 3, StructureSpanningCell = 5;

    internal TableStructureResult<T> ParseStructureOutput(Tensor<T> output, Tensor<T> tableImage, double threshold)
    {
        // Table Transformer's objects-to-cells: the table's rows and columns are the detected row and column
        // objects ordered top-to-bottom and left-to-right; each row/column intersection is a cell, a row is a header
        // row when its centre lies in a column-header object, and a spanning-cell object merges the grid cells whose
        // row and column centres it contains into one cell.
        var (labels, scores, boxes, pixels) = DetrObjects(output, _numStructureClasses, tableImage, threshold);
        int[] Of(int label) => Enumerable.Range(0, labels.Length).Where(i => labels[i] == label).ToArray();
        double Center(int i, int axis) => (pixels[i][axis] + pixels[i][axis + 2]) / 2;
        bool Inside(double v, int obj, int axis) => v >= pixels[obj][axis] && v <= pixels[obj][axis + 2];

        int[] rows = Of(StructureRow).OrderBy(i => Center(i, 1)).ToArray();
        int[] columns = Of(StructureColumn).OrderBy(i => Center(i, 0)).ToArray();
        int[] headers = Of(StructureColumnHeader);
        var headerRows = Enumerable.Range(0, rows.Length).Where(r => headers.Any(h => Inside(Center(rows[r], 1), h, 1))).ToList();
        var cells = new List<TableCell<T>>();
        if (rows.Length > 0 && columns.Length > 0)
        {
            // A grid cell's confidence is the mean of the scores of its row and its column.
            var all = new Tensor<T>(scores, new[] { scores.Length });
            int[] grid = { rows.Length, columns.Length };
            T[] gridScores = Engine.TensorMultiplyScalar(Engine.TensorAdd(
                Engine.TensorBroadcastTo(new Tensor<T>(DocumentTensorOps.Pick(Engine, all, rows), new[] { rows.Length, 1 }), grid),
                Engine.TensorBroadcastTo(new Tensor<T>(DocumentTensorOps.Pick(Engine, all, columns), new[] { 1, columns.Length }), grid)),
                NumOps.FromDouble(0.5)).ToArray();
            Vector<T> Box(int firstRow, int firstColumn, int lastRow, int lastColumn) => new(new[]
            {
                boxes[columns[firstColumn]][0], boxes[rows[firstRow]][1], boxes[columns[lastColumn]][2], boxes[rows[lastRow]][3]
            });

            var covered = new bool[rows.Length * columns.Length];
            foreach (int span in Of(StructureSpanningCell))
            {
                int[] r = Enumerable.Range(0, rows.Length).Where(i => Inside(Center(rows[i], 1), span, 1)).ToArray();
                int[] c = Enumerable.Range(0, columns.Length).Where(i => Inside(Center(columns[i], 0), span, 0)).ToArray();
                if (r.Length * c.Length < 2 || r.Any(i => c.Any(j => covered[(i * columns.Length) + j])))
                    continue;
                foreach (int i in r)
                    foreach (int j in c)
                        covered[(i * columns.Length) + j] = true;
                cells.Add(new TableCell<T>
                {
                    Row = r[0],
                    Column = c[0],
                    RowSpan = r[r.Length - 1] - r[0] + 1,
                    ColSpan = c[c.Length - 1] - c[0] + 1,
                    BoundingBox = Box(r[0], c[0], r[r.Length - 1], c[c.Length - 1]),
                    IsHeader = headerRows.Contains(r[0]),
                    Confidence = scores[span]
                });
            }

            for (int i = 0; i < rows.Length; i++)
                for (int j = 0; j < columns.Length; j++)
                    if (!covered[(i * columns.Length) + j])
                        cells.Add(new TableCell<T>
                        {
                            Row = i,
                            Column = j,
                            BoundingBox = Box(i, j, i, j),
                            IsHeader = headerRows.Contains(i),
                            Confidence = gridScores[(i * columns.Length) + j]
                        });
        }

        var ordered = cells.OrderBy(c => c.Row).ThenBy(c => c.Column).ToList();
        return new TableStructureResult<T>
        {
            NumRows = rows.Length,
            NumColumns = columns.Length,
            Cells = ordered,
            HeaderRows = headerRows,
            HasBorders = true,
            Confidence = DocumentTensorOps.Mean(Engine, ordered.Select(c => c.Confidence).ToArray())
        };
    }

    private Tensor<T>? CropTableImage(Tensor<T> image, double x1, double y1, double x2, double y2)
    {
        var (imageHeight, imageWidth) = GetImageDimensions(image);
        int startX = Clamp((int)Math.Floor(x1), 0, imageWidth);
        int startY = Clamp((int)Math.Floor(y1), 0, imageHeight);
        int endX = Clamp((int)Math.Ceiling(x2), 0, imageWidth);
        int endY = Clamp((int)Math.Ceiling(y2), 0, imageHeight);

        int cropWidth = endX - startX;
        int cropHeight = endY - startY;
        if (cropWidth <= 0 || cropHeight <= 0 || (image.Rank != 3 && image.Rank != 4))
            return null;

        int lead = image.Rank - 2;
        var start = Enumerable.Repeat(0, lead).Concat(new[] { startY, startX }).ToArray();
        var length = image.Shape.ToArray().Take(lead).Concat(new[] { cropHeight, cropWidth }).ToArray();
        return Engine.TensorSlice(image, start, length).Clone();
    }

    #region Export Methods

    private static string ExportToCSV(List<List<List<string>>> tables)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var table in tables)
        {
            foreach (var row in table)
            {
                sb.AppendLine(string.Join(",", row.Select(c => $"\"{c.Replace("\"", "\"\"")}\"")));
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string ExportToJSON(List<List<List<string>>> tables)
    {
        var payload = tables.Select(table => new { rows = table }).ToList();
        var options = new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        };

        return System.Text.Json.JsonSerializer.Serialize(payload, options);
    }

    private static string ExportToHTML(List<List<List<string>>> tables)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var table in tables)
        {
            sb.AppendLine("<table border=\"1\">");
            foreach (var row in table)
            {
                sb.AppendLine("  <tr>");
                foreach (var cell in row)
                {
                    sb.AppendLine($"    <td>{System.Net.WebUtility.HtmlEncode(cell)}</td>");
                }
                sb.AppendLine("  </tr>");
            }
            sb.AppendLine("</table>");
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string ExportToMarkdown(List<List<List<string>>> tables)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var table in tables)
        {
            if (table.Count == 0) continue;

            // Header row
            if (table.Count > 0)
            {
                var header = table[0].Select(EscapeMarkdownCell).ToList();
                sb.AppendLine("| " + string.Join(" | ", header) + " |");
                sb.AppendLine("| " + string.Join(" | ", header.Select(_ => "---")) + " |");
            }

            // Data rows
            for (int r = 1; r < table.Count; r++)
            {
                var row = table[r].Select(EscapeMarkdownCell);
                sb.AppendLine("| " + string.Join(" | ", row) + " |");
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string EscapeMarkdownCell(string value)
    {
        return value?.Replace("|", "\\|") ?? string.Empty;
    }

    #endregion

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunDetectionOnnx(preprocessed);
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
        sb.AppendLine("TableTransformer Model Summary");
        sb.AppendLine("==============================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: DETR-style (Detection Transformer)");
        sb.AppendLine();
        sb.AppendLine("Transformer Configuration:");
        sb.AppendLine($"  Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"  Encoder Layers: {_numEncoderLayers}");
        sb.AppendLine($"  Decoder Layers: {_numDecoderLayers}");
        sb.AppendLine($"  Attention Heads: {_numHeads}");
        sb.AppendLine($"  Object Queries: {_numQueries}");
        sb.AppendLine();
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Table Classes: {_numTableClasses}");
        sb.AppendLine($"Structure Classes: {_numStructureClasses}");
        sb.AppendLine($"Supports Bordered Tables: {SupportsBorderedTables}");
        sb.AppendLine($"Supports Borderless Tables: {SupportsBorderlessTables}");
        sb.AppendLine($"Supports Merged Cells: {SupportsMergedCells}");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies TableTransformer's industry-standard preprocessing: COCO/ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// TableTransformer uses COCO-style normalization with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    /// From the PubTables-1M paper (CVPR 2022).
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
        => DocumentTensorOps.NormalizeChannels(Engine,
            Engine.TensorClamp(EnsureBatchDimension(rawImage), NumOps.Zero, NumOps.One),
            DocumentTensorOps.ImageNetMean, DocumentTensorOps.ImageNetStd);

    /// <summary>
    /// Applies TableTransformer's industry-standard postprocessing: pass-through (DETR outputs are already in final format).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput)
    {
        // DETR-style outputs are already in final format (bbox + class logits)
        return modelOutput;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "TableTransformer",
            Description = "DETR-style table detection and structure recognition (CVPR 2022)",
            FeatureCount = _hiddenDim,
            Complexity = _numEncoderLayers + _numDecoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_encoder_layers", _numEncoderLayers },
                { "num_decoder_layers", _numDecoderLayers },
                { "num_heads", _numHeads },
                { "num_queries", _numQueries },
                { "num_table_classes", _numTableClasses },
                { "num_structure_classes", _numStructureClasses },
                { "image_size", ImageSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelDataProvider = () => SafeSerialize()
        };
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunDetectionOnnx(preprocessed);
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
    private Vector<T> CollectParameterGradients()
    {
        var gradients = new List<T>();

        foreach (var layer in Layers)
        {
            var layerGradients = layer.GetParameterGradients();
            gradients.AddRange(layerGradients);
        }

        return new Vector<T>([.. gradients]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxDetectionSession?.Dispose();
            _onnxStructureSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}

/// <summary>
/// Task modes for TableTransformer.
/// </summary>
internal enum TableTransformerTask
{
    Detection,
    Structure
}
