using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.ComputerVision.OCR.Recognition;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.OCR.EndToEnd;

/// <summary>
/// Document reader for OCR with layout analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DocumentReader is optimized for reading structured documents
/// like scanned papers, forms, and PDFs. Unlike scene text, documents have regular layouts
/// with clear reading order. This reader analyzes the document structure and extracts
/// text in logical reading order.</para>
///
/// <para>Key features:
/// - Layout analysis for document structure understanding
/// - Reading order detection
/// - Paragraph and line grouping
/// - Handles multi-column layouts
/// - Optimized for clean text on uniform backgrounds
/// </para>
/// </remarks>
public class DocumentReader<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly TextDetectorBase<T> _detector;
    private readonly OCRBase<T> _recognizer;
    private readonly OCROptions<T> _options;

    /// <summary>
    /// Name of this document reader.
    /// </summary>
    public string Name => $"DocumentReader-{_recognizer.Name}";

    /// <summary>
    /// Creates a new document reader.
    /// </summary>
    public DocumentReader(OCROptions<T> options)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _options = options;

        // Use DBNet for document text detection (works well with clean documents)
        var detectionOptions = new TextDetectionOptions<T>
        {
            Architecture = TextDetectionArchitecture.DBNet,
            ConfidenceThreshold = _numOps.FromDouble(0.3) // Lower threshold for documents
        };

        _detector = new DBNet<T>(detectionOptions);

        // Initialize recognizer
        _recognizer = options.RecognitionModel switch
        {
            TextRecognitionModel.TrOCR => new TrOCR<T>(options),
            _ => new CRNN<T>(options)
        };
    }

    /// <summary>
    /// Reads a document image and returns structured text.
    /// </summary>
    /// <param name="image">Document image tensor [batch, channels, height, width].</param>
    /// <returns>Document OCR result with layout information.</returns>
    public DocumentOCRResult<T> ReadDocument(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageWidth = image.Shape[3];
        int imageHeight = image.Shape[2];

        // Preprocess for document (enhance contrast, binarize if needed)
        var preprocessed = PreprocessDocument(image);

        // Detect text regions
        var detectionResult = _detector.Detect(preprocessed);

        // Analyze layout
        var layout = AnalyzeLayout(detectionResult.TextRegions, imageWidth, imageHeight);

        // Recognize text in each region
        var blocks = new List<DocumentBlock<T>>();

        foreach (var layoutBlock in layout.Blocks)
        {
            var lines = new List<DocumentLine<T>>();

            foreach (var lineRegions in layoutBlock.Lines)
            {
                var lineTexts = new List<RecognizedText<T>>();

                foreach (var region in lineRegions)
                {
                    var crop = CropRegion(preprocessed, region);
                    var (text, confidence) = _recognizer.RecognizeText(crop);

                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        lineTexts.Add(new RecognizedText<T>(text, confidence)
                        {
                            Box = region.Box,
                            Polygon = region.Polygon
                        });
                    }
                }

                if (lineTexts.Count > 0)
                {
                    lines.Add(new DocumentLine<T>
                    {
                        Words = lineTexts,
                        Text = string.Join(" ", lineTexts.Select(t => t.Text))
                    });
                }
            }

            if (lines.Count > 0)
            {
                blocks.Add(new DocumentBlock<T>
                {
                    BlockType = layoutBlock.Type,
                    Lines = lines,
                    Text = string.Join("\n", lines.Select(l => l.Text))
                });
            }
        }

        string fullText = string.Join("\n\n", blocks.Select(b => b.Text));

        return new DocumentOCRResult<T>
        {
            Blocks = blocks,
            FullText = fullText,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight,
            PageCount = 1
        };
    }

    private Tensor<T> PreprocessDocument(Tensor<T> image)
    {
        int batch = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var result = new Tensor<T>(image.Shape);

        // Convert to grayscale-like processing and enhance
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        double val = _numOps.ToDouble(image[b, c, h, w]);

                        // Simple contrast enhancement
                        val = (val - 128) * 1.2 + 128;
                        val = Math.Clamp(val, 0, 255);

                        result[b, c, h, w] = _numOps.FromDouble(val);
                    }
                }
            }
        }

        return result;
    }

    private DocumentLayout<T> AnalyzeLayout(
        List<TextRegion<T>> regions,
        int imageWidth,
        int imageHeight)
    {
        var layout = new DocumentLayout<T>();

        if (regions.Count == 0)
            return layout;

        // Sort regions by vertical position
        var sortedRegions = regions
            .OrderBy(r => _numOps.ToDouble(r.Box.Y1))
            .ToList();

        // Group into lines based on Y-overlap
        var lines = GroupIntoLines(sortedRegions);

        // Detect columns (for multi-column documents)
        var columns = DetectColumns(lines, imageWidth);

        // Create layout blocks
        foreach (var column in columns)
        {
            var block = new LayoutBlock<T>
            {
                Type = DocumentBlockType.Paragraph,
                Lines = column
            };

            layout.Blocks.Add(block);
        }

        return layout;
    }

    private List<List<TextRegion<T>>> GroupIntoLines(List<TextRegion<T>> regions)
    {
        var lines = new List<List<TextRegion<T>>>();

        foreach (var region in regions)
        {
            bool addedToLine = false;
            double regionTop = _numOps.ToDouble(region.Box.Y1);
            double regionBottom = _numOps.ToDouble(region.Box.Y2);
            double regionHeight = regionBottom - regionTop;

            foreach (var line in lines)
            {
                double lineTop = line.Min(r => _numOps.ToDouble(r.Box.Y1));
                double lineBottom = line.Max(r => _numOps.ToDouble(r.Box.Y2));

                // Check vertical overlap
                double overlap = Math.Min(regionBottom, lineBottom) - Math.Max(regionTop, lineTop);
                double overlapRatio = overlap / regionHeight;

                if (overlapRatio > 0.5)
                {
                    line.Add(region);
                    addedToLine = true;
                    break;
                }
            }

            if (!addedToLine)
            {
                lines.Add(new List<TextRegion<T>> { region });
            }
        }

        // Sort each line left-to-right
        foreach (var line in lines)
        {
            line.Sort((a, b) =>
                _numOps.ToDouble(a.Box.X1).CompareTo(_numOps.ToDouble(b.Box.X1)));
        }

        return lines;
    }

    private List<List<List<TextRegion<T>>>> DetectColumns(
        List<List<TextRegion<T>>> lines,
        int imageWidth)
    {
        // Simple column detection: check if there's a consistent vertical gap
        var columns = new List<List<List<TextRegion<T>>>>();

        double midPoint = imageWidth / 2.0;
        bool hasLeftColumn = false;
        bool hasRightColumn = false;

        foreach (var line in lines)
        {
            foreach (var region in line)
            {
                double x1 = _numOps.ToDouble(region.Box.X1);
                double x2 = _numOps.ToDouble(region.Box.X2);

                if (x2 < midPoint - imageWidth * 0.1)
                    hasLeftColumn = true;
                if (x1 > midPoint + imageWidth * 0.1)
                    hasRightColumn = true;
            }
        }

        if (hasLeftColumn && hasRightColumn)
        {
            // Two-column layout
            var leftLines = new List<List<TextRegion<T>>>();
            var rightLines = new List<List<TextRegion<T>>>();

            foreach (var line in lines)
            {
                var leftRegions = line.Where(r =>
                    _numOps.ToDouble(r.Box.X2) < midPoint + imageWidth * 0.05).ToList();

                var rightRegions = line.Where(r =>
                    _numOps.ToDouble(r.Box.X1) > midPoint - imageWidth * 0.05).ToList();

                if (leftRegions.Count > 0)
                    leftLines.Add(leftRegions);
                if (rightRegions.Count > 0)
                    rightLines.Add(rightRegions);
            }

            if (leftLines.Count > 0)
                columns.Add(leftLines);
            if (rightLines.Count > 0)
                columns.Add(rightLines);
        }
        else
        {
            // Single column layout
            columns.Add(lines);
        }

        return columns;
    }

    private Tensor<T> CropRegion(Tensor<T> image, TextRegion<T> region)
    {
        int batch = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        int x1 = Math.Max(0, (int)_numOps.ToDouble(region.Box.X1));
        int y1 = Math.Max(0, (int)_numOps.ToDouble(region.Box.Y1));
        int x2 = Math.Min(width, (int)_numOps.ToDouble(region.Box.X2));
        int y2 = Math.Min(height, (int)_numOps.ToDouble(region.Box.Y2));

        int cropW = Math.Max(1, x2 - x1);
        int cropH = Math.Max(1, y2 - y1);

        var crop = new Tensor<T>(new[] { batch, channels, cropH, cropW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < cropH; h++)
                {
                    for (int w = 0; w < cropW; w++)
                    {
                        crop[b, c, h, w] = image[b, c, y1 + h, x1 + w];
                    }
                }
            }
        }

        return crop;
    }

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long GetParameterCount()
    {
        return _detector.GetParameterCount() + _recognizer.GetParameterCount();
    }
}

/// <summary>
/// Result of document OCR.
/// </summary>
public class DocumentOCRResult<T>
{
    /// <summary>Document blocks (paragraphs, headers, etc.).</summary>
    public List<DocumentBlock<T>> Blocks { get; set; } = new();

    /// <summary>Full text of the document.</summary>
    public string FullText { get; set; } = string.Empty;

    /// <summary>Inference time.</summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>Image width.</summary>
    public int ImageWidth { get; set; }

    /// <summary>Image height.</summary>
    public int ImageHeight { get; set; }

    /// <summary>Number of pages (for multi-page documents).</summary>
    public int PageCount { get; set; } = 1;
}

/// <summary>
/// A block of text in a document (paragraph, header, etc.).
/// </summary>
public class DocumentBlock<T>
{
    /// <summary>Type of block.</summary>
    public DocumentBlockType BlockType { get; set; }

    /// <summary>Lines in this block.</summary>
    public List<DocumentLine<T>> Lines { get; set; } = new();

    /// <summary>Full text of this block.</summary>
    public string Text { get; set; } = string.Empty;
}

/// <summary>
/// A line of text in a document.
/// </summary>
public class DocumentLine<T>
{
    /// <summary>Words/regions in this line.</summary>
    public List<RecognizedText<T>> Words { get; set; } = new();

    /// <summary>Full text of this line.</summary>
    public string Text { get; set; } = string.Empty;
}

/// <summary>
/// Type of document block.
/// </summary>
public enum DocumentBlockType
{
    /// <summary>Regular paragraph.</summary>
    Paragraph,
    /// <summary>Header/title.</summary>
    Header,
    /// <summary>Table.</summary>
    Table,
    /// <summary>List item.</summary>
    ListItem,
    /// <summary>Caption.</summary>
    Caption,
    /// <summary>Footer.</summary>
    Footer
}

/// <summary>
/// Layout analysis result.
/// </summary>
internal class DocumentLayout<T>
{
    public List<LayoutBlock<T>> Blocks { get; set; } = new();
}

/// <summary>
/// A layout block before text recognition.
/// </summary>
internal class LayoutBlock<T>
{
    public DocumentBlockType Type { get; set; }
    public List<List<TextRegion<T>>> Lines { get; set; } = new();
}
