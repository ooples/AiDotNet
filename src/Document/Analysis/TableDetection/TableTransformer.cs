using AiDotNet.Document.Interfaces;
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
public class TableTransformer<T> : DocumentNeuralNetworkBase<T>, ITableExtractor<T>
{
    #region Fields

    private bool _useNativeMode;
    private InferenceSession? _onnxDetectionSession;
    private InferenceSession? _onnxStructureSession;
    private string? _onnxDetectionModelPath;
    private string? _onnxStructureModelPath;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
        int imageSize = 800,
        int hiddenDim = 256,
        int numEncoderLayers = 6,
        int numDecoderLayers = 6,
        int numHeads = 8,
        int numQueries = 100,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(detectionModelPath))
            throw new ArgumentNullException(nameof(detectionModelPath));
        if (string.IsNullOrWhiteSpace(structureModelPath))
            throw new ArgumentNullException(nameof(structureModelPath));
        if (!File.Exists(detectionModelPath))
            throw new FileNotFoundException($"Detection model not found: {detectionModelPath}", detectionModelPath);
        if (!File.Exists(structureModelPath))
            throw new FileNotFoundException($"Structure model not found: {structureModelPath}", structureModelPath);

        _useNativeMode = false;
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _numQueries = numQueries;
        _numTableClasses = 2;       // background, table
        _numStructureClasses = 7;   // background, table, column, row, column header, projected row header, spanning cell
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

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
        int imageSize = 800,
        int hiddenDim = 256,
        int numEncoderLayers = 6,
        int numDecoderLayers = 6,
        int numHeads = 8,
        int numQueries = 100,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _numQueries = numQueries;
        _numTableClasses = 2;
        _numStructureClasses = 7;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

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
        _objectQueries = Tensor<T>.CreateDefault([_numQueries, _hiddenDim], NumOps.Zero);

        // Xavier initialization for object queries
        double scale = Math.Sqrt(2.0 / (_numQueries + _hiddenDim));
        for (int i = 0; i < _objectQueries.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            _objectQueries.Data.Span[i] = NumOps.FromDouble(randStdNormal * scale);
        }
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

        int numQueries = boxesTensor.Shape[0];
        int boxDim = boxesTensor.Shape[1];
        int classDim = logitsTensor.Shape[1];
        var merged = new Tensor<T>([numQueries, boxDim + classDim]);

        for (int i = 0; i < numQueries; i++)
        {
            for (int b = 0; b < boxDim; b++)
                merged[i, b] = boxesTensor[i, b];
            for (int c = 0; c < classDim; c++)
                merged[i, boxDim + c] = logitsTensor[i, c];
        }

        return merged;
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

    private static double Clamp(double value, double min, double max)
    {
        if (value < min)
            return min;
        if (value > max)
            return max;
        return value;
    }

    private static int Clamp(int value, int min, int max)
    {
        if (value < min)
            return min;
        if (value > max)
            return max;
        return value;
    }

    private IEnumerable<TableRegion<T>> ParseTableDetections(Tensor<T> output, Tensor<T> originalImage, double threshold)
    {
        var regions = new List<TableRegion<T>>();
        var (imageHeight, imageWidth) = GetImageDimensions(originalImage);

        // DETR output: [num_queries, 4 + num_classes] (bbox + class logits)
        bool is1D = output.Shape.Length == 1;
        int outputDim = output.Shape.Length > 1 ? output.Shape[1] : 4 + _numTableClasses; // 4 bbox + classes
        int numDetections = is1D ? output.Length / outputDim : output.Shape[0];

        for (int i = 0; i < numDetections; i++)
        {
            int baseIndex = is1D ? i * outputDim : 0;
            if (is1D && baseIndex + outputDim > output.Length)
                break;

            // Get class probabilities
            double tableProb = 0;
            if (outputDim >= 6)
            {
                // Softmax over class logits
                if (is1D && baseIndex + 5 >= output.Length)
                    continue;

                double bg = is1D
                    ? NumOps.ToDouble(output[baseIndex + 4])
                    : NumOps.ToDouble(output[i, 4]);
                double table = is1D
                    ? NumOps.ToDouble(output[baseIndex + 5])
                    : NumOps.ToDouble(output[i, 5]);
                double maxLogit = Math.Max(bg, table);
                double sumExp = Math.Exp(bg - maxLogit) + Math.Exp(table - maxLogit);
                tableProb = Math.Exp(table - maxLogit) / sumExp;
            }

            if (tableProb >= threshold)
            {
                if (is1D && baseIndex + 3 >= output.Length)
                    continue;

                // Get bounding box (DETR uses center_x, center_y, width, height format normalized to [0,1])
                double cx = is1D
                    ? NumOps.ToDouble(output[baseIndex + 0])
                    : NumOps.ToDouble(output[i, 0]);
                double cy = is1D
                    ? NumOps.ToDouble(output[baseIndex + 1])
                    : NumOps.ToDouble(output[i, 1]);
                double w = is1D
                    ? NumOps.ToDouble(output[baseIndex + 2])
                    : NumOps.ToDouble(output[i, 2]);
                double h = is1D
                    ? NumOps.ToDouble(output[baseIndex + 3])
                    : NumOps.ToDouble(output[i, 3]);

                // Convert to [x1, y1, x2, y2] format
                double x1 = (cx - w / 2) * imageWidth;
                double y1 = (cy - h / 2) * imageHeight;
                double x2 = (cx + w / 2) * imageWidth;
                double y2 = (cy + h / 2) * imageHeight;

                x1 = Clamp(x1, 0, imageWidth);
                x2 = Clamp(x2, 0, imageWidth);
                y1 = Clamp(y1, 0, imageHeight);
                y2 = Clamp(y2, 0, imageHeight);

                if (x2 <= x1 || y2 <= y1)
                    continue;

                var region = new TableRegion<T>
                {
                    BoundingBox = new Vector<T>([
                        NumOps.FromDouble(x1),
                        NumOps.FromDouble(y1),
                        NumOps.FromDouble(x2),
                        NumOps.FromDouble(y2)
                    ]),
                    Confidence = NumOps.FromDouble(tableProb),
                    TableIndex = regions.Count,
                    Image = CropTableImage(originalImage, x1, y1, x2, y2)
                };

                regions.Add(region);
            }
        }

        return regions;
    }

    private TableStructureResult<T> ParseStructureOutput(Tensor<T> output, Tensor<T> tableImage, double threshold)
    {
        var cells = new List<TableCell<T>>();
        var rows = new HashSet<int>();
        var columns = new HashSet<int>();
        var (imageHeight, imageWidth) = GetImageDimensions(tableImage);

        bool is1D = output.Shape.Length == 1;
        int outputDim = output.Shape.Length > 1 ? output.Shape[1] : 4 + _numStructureClasses; // 4 bbox + classes
        int numDetections = is1D ? output.Length / outputDim : output.Shape[0];

        for (int i = 0; i < numDetections; i++)
        {
            int baseIndex = is1D ? i * outputDim : 0;
            if (is1D && baseIndex + outputDim > output.Length)
                break;

            // Find the class with highest probability
            double maxProb = 0;
            int maxClass = 0;
            for (int c = 0; c < _numStructureClasses && (4 + c) < outputDim; c++)
            {
                if (is1D)
                {
                    int flatIndex = baseIndex + 4 + c;
                    if (flatIndex < 0 || flatIndex >= output.Length)
                        break;
                    double prob = NumOps.ToDouble(output[flatIndex]);
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxClass = c;
                    }
                }
                else
                {
                    double prob = NumOps.ToDouble(output[i, 4 + c]);
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxClass = c;
                    }
                }
            }

            if (maxProb >= threshold && maxClass > 0) // Skip background class
            {
                if (is1D && baseIndex + 3 >= output.Length)
                    continue;

                double cx = is1D
                    ? NumOps.ToDouble(output[baseIndex + 0])
                    : NumOps.ToDouble(output[i, 0]);
                double cy = is1D
                    ? NumOps.ToDouble(output[baseIndex + 1])
                    : NumOps.ToDouble(output[i, 1]);
                double w = is1D
                    ? NumOps.ToDouble(output[baseIndex + 2])
                    : NumOps.ToDouble(output[i, 2]);
                double h = is1D
                    ? NumOps.ToDouble(output[baseIndex + 3])
                    : NumOps.ToDouble(output[i, 3]);

                // Convert to pixel coordinates
                double x1 = (cx - w / 2) * imageWidth;
                double y1 = (cy - h / 2) * imageHeight;
                double x2 = (cx + w / 2) * imageWidth;
                double y2 = (cy + h / 2) * imageHeight;

                x1 = Clamp(x1, 0, imageWidth);
                x2 = Clamp(x2, 0, imageWidth);
                y1 = Clamp(y1, 0, imageHeight);
                y2 = Clamp(y2, 0, imageHeight);

                if (x2 <= x1 || y2 <= y1)
                    continue;

                // Determine row/column based on position
                int rowIdx = EstimateRowIndex(y1, y2);
                int colIdx = EstimateColumnIndex(x1, x2);
                rows.Add(rowIdx);
                columns.Add(colIdx);

                bool isHeader = maxClass == 4; // column header class
                int rowSpan = maxClass == 6 ? 2 : 1; // spanning cell
                int colSpan = maxClass == 6 ? 2 : 1;

                cells.Add(new TableCell<T>
                {
                    Row = rowIdx,
                    Column = colIdx,
                    RowSpan = rowSpan,
                    ColSpan = colSpan,
                    BoundingBox = new Vector<T>([
                        NumOps.FromDouble(x1),
                        NumOps.FromDouble(y1),
                        NumOps.FromDouble(x2),
                        NumOps.FromDouble(y2)
                    ]),
                    IsHeader = isHeader,
                    Confidence = NumOps.FromDouble(maxProb),
                    Text = "" // Would be filled by OCR
                });
            }
        }

        return new TableStructureResult<T>
        {
            NumRows = rows.Count > 0 ? rows.Max() + 1 : 0,
            NumColumns = columns.Count > 0 ? columns.Max() + 1 : 0,
            Cells = cells,
            HeaderRows = cells.Any(c => c.IsHeader) ? [0] : [],
            HasBorders = true,
            Confidence = cells.Count > 0
                ? NumOps.FromDouble(cells.Average(c => NumOps.ToDouble(c.Confidence)))
                : NumOps.Zero
        };
    }

    private int EstimateRowIndex(double y1, double y2)
    {
        // Simple heuristic: divide image into grid
        double centerY = (y1 + y2) / 2;
        return (int)(centerY / 50); // Assume ~50px per row
    }

    private int EstimateColumnIndex(double x1, double x2)
    {
        double centerX = (x1 + x2) / 2;
        return (int)(centerX / 100); // Assume ~100px per column
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
        if (cropWidth <= 0 || cropHeight <= 0)
            return null;

        if (image.Rank == 3)
        {
            int channels = image.Shape[0];
            var cropped = new Tensor<T>([channels, cropHeight, cropWidth]);
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < cropHeight; y++)
                {
                    int srcY = startY + y;
                    for (int x = 0; x < cropWidth; x++)
                    {
                        int srcX = startX + x;
                        cropped[c, y, x] = image[c, srcY, srcX];
                    }
                }
            }
            return cropped;
        }

        if (image.Rank == 4)
        {
            int batch = image.Shape[0];
            int channels = image.Shape[1];
            var cropped = new Tensor<T>([batch, channels, cropHeight, cropWidth]);
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int y = 0; y < cropHeight; y++)
                    {
                        int srcY = startY + y;
                        for (int x = 0; x < cropWidth; x++)
                        {
                            int srcX = startX + x;
                            cropped[b, c, y, x] = image[b, c, srcY, srcX];
                        }
                    }
                }
            }
            return cropped;
        }

        return null;
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
    {
        var image = EnsureBatchDimension(rawImage);

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);

        // COCO/ImageNet normalization (industry standard for TableTransformer)
        double[] means = [0.485, 0.456, 0.406];
        double[] stds = [0.229, 0.224, 0.225];

        double minValue = double.PositiveInfinity;
        double maxValue = double.NegativeInfinity;
        for (int i = 0; i < image.Data.Length; i++)
        {
            double value = NumOps.ToDouble(image.Data.Span[i]);
            if (value < minValue) minValue = value;
            if (value > maxValue) maxValue = value;
        }

        const double epsilon = 1e-6;
        if (minValue < -epsilon || maxValue > 1.0 + epsilon)
        {
            throw new ArgumentOutOfRangeException(
                nameof(rawImage),
                $"TableTransformer expects input values in [0,1] before COCO/ImageNet normalization. Got range [{minValue:F4}, {maxValue:F4}].");
        }

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double mean = c < means.Length ? means[c] : 0.5;
                double std = c < stds.Length ? stds[c] : 0.5;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        double value = NumOps.ToDouble(image.Data.Span[idx]);
                        normalized.Data.Span[idx] = NumOps.FromDouble((value - mean) / std);
                    }
                }
            }
        }

        return normalized;
    }

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
            ModelType = ModelType.NeuralNetwork,
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
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_numQueries);
        writer.Write(_numTableClasses);
        writer.Write(_numStructureClasses);
        writer.Write(ImageSize);
        writer.Write(_useNativeMode);
        writer.Write(_onnxDetectionModelPath ?? string.Empty);
        writer.Write(_onnxStructureModelPath ?? string.Empty);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int numQueries = reader.ReadInt32();
        int numTableClasses = reader.ReadInt32();
        int numStructureClasses = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();
        string? detectionModelPath = null;
        string? structureModelPath = null;
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            detectionModelPath = reader.ReadString();
            if (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                structureModelPath = reader.ReadString();
            }
        }

        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _numQueries = numQueries;
        _numTableClasses = numTableClasses;
        _numStructureClasses = numStructureClasses;
        _useNativeMode = useNativeMode;
        ImageSize = imageSize;
        if (!string.IsNullOrWhiteSpace(detectionModelPath))
        {
            _onnxDetectionModelPath = detectionModelPath;
        }
        else if (_useNativeMode)
        {
            _onnxDetectionModelPath = null;
        }

        if (!string.IsNullOrWhiteSpace(structureModelPath))
        {
            _onnxStructureModelPath = structureModelPath;
        }
        else if (_useNativeMode)
        {
            _onnxStructureModelPath = null;
        }

        if (_useNativeMode && _objectQueries is null)
        {
            InitializeObjectQueries();
        }

        if (!_useNativeMode)
        {
            if (!string.IsNullOrWhiteSpace(_onnxDetectionModelPath)
                && !string.IsNullOrWhiteSpace(_onnxStructureModelPath))
            {
                InitializeOnnxSessions();
            }
            else if (_onnxDetectionSession is null || _onnxStructureSession is null)
            {
                throw new InvalidOperationException(
                    "Missing ONNX model paths required to restore TableTransformer.");
            }
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new TableTransformer<T>(
                Architecture,
                ImageSize,
                _hiddenDim,
                _numEncoderLayers,
                _numDecoderLayers,
                _numHeads,
                _numQueries);
        }

        if (string.IsNullOrWhiteSpace(_onnxDetectionModelPath) || string.IsNullOrWhiteSpace(_onnxStructureModelPath))
        {
            throw new InvalidOperationException(
                "Missing ONNX model paths required to clone TableTransformer.");
        }

        var detectionModelPath = _onnxDetectionModelPath
            ?? throw new InvalidOperationException(
                "Missing ONNX detection model path required to clone TableTransformer.");
        var structureModelPath = _onnxStructureModelPath
            ?? throw new InvalidOperationException(
                "Missing ONNX structure model path required to clone TableTransformer.");

        return new TableTransformer<T>(
            Architecture,
            detectionModelPath,
            structureModelPath,
            ImageSize,
            _hiddenDim,
            _numEncoderLayers,
            _numDecoderLayers,
            _numHeads,
            _numQueries);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunDetectionOnnx(preprocessed);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Training is not supported in ONNX inference mode.");
        }

        SetTrainingMode(true);

        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var lossGradient = LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        var paramGradients = CollectParameterGradients();
        UpdateParameters(paramGradients);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Parameter updates are not supported in ONNX inference mode.");
        }

        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.0001);

        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(currentParams);
    }

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
