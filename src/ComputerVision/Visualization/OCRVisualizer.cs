using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.OCR;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Visualization;

/// <summary>
/// Visualizes OCR (Optical Character Recognition) results on images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class draws text region bounding boxes, polygons,
/// and recognized text on images to visualize OCR results.</para>
/// </remarks>
public class OCRVisualizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly VisualizationOptions _options;
    private readonly BitmapFont<T> _font;

    // Default colors
    private readonly (byte R, byte G, byte B) _textBoxColor = (0, 255, 0); // Green

    /// <summary>
    /// Creates a new OCR visualizer.
    /// </summary>
    public OCRVisualizer(VisualizationOptions? options = null)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _options = options ?? new VisualizationOptions();
        _font = new BitmapFont<T>();
    }

    /// <summary>
    /// Draws OCR results on an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <param name="result">OCR result to visualize.</param>
    /// <returns>Image with visualizations drawn.</returns>
    public Tensor<T> Visualize(Tensor<T> image, OCRResult<T> result)
    {
        var output = CloneImage(image);

        foreach (var region in result.TextRegions)
        {
            DrawRecognizedText(output, region);
        }

        return output;
    }

    /// <summary>
    /// Draws only text bounding boxes without text labels.
    /// </summary>
    public Tensor<T> VisualizeBoxesOnly(Tensor<T> image, OCRResult<T> result)
    {
        var output = CloneImage(image);

        foreach (var region in result.TextRegions)
        {
            if (region.Box != null)
            {
                DrawTextBox(output, region.Box, _textBoxColor);
            }

            if (_options.ShowPolygons && region.Polygon.Count >= 4)
            {
                DrawPolygon(output, region.Polygon);
            }
        }

        return output;
    }

    /// <summary>
    /// Creates a text overlay image with recognized text shown at original positions.
    /// </summary>
    public Tensor<T> CreateTextOverlay(Tensor<T> image, OCRResult<T> result)
    {
        var output = CloneImage(image);

        // Dim the background
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = _numOps.FromDouble(_numOps.ToDouble(output[i]) * 0.3);
        }

        // Draw text regions with white background
        foreach (var region in result.TextRegions)
        {
            DrawTextRegionHighlight(output, region);
        }

        return output;
    }

    /// <summary>
    /// Visualizes document layout analysis results.
    /// </summary>
    public Tensor<T> VisualizeDocumentLayout(Tensor<T> image, DocumentLayoutResult<T> layout)
    {
        var output = CloneImage(image);

        // Draw blocks with different colors
        foreach (var block in layout.Blocks)
        {
            var color = GetBlockTypeColor(block.Type);
            DrawTextBox(output, block.Box, color);

            if (_options.ShowLabels)
            {
                string label = block.Type.ToString();
                DrawLabel(output, block.Box, label, color);
            }
        }

        // Draw reading order arrows
        if (layout.ReadingOrder != null && layout.ReadingOrder.Count > 1)
        {
            DrawReadingOrder(output, layout);
        }

        return output;
    }

    private void DrawRecognizedText(Tensor<T> image, RecognizedText<T> region)
    {
        // Draw bounding box if available
        if (region.Box != null)
        {
            DrawTextBox(image, region.Box, _textBoxColor);
        }

        // Draw polygon if available
        if (_options.ShowPolygons && region.Polygon.Count >= 4)
        {
            DrawPolygon(image, region.Polygon);
        }

        // Draw text label
        if (_options.ShowLabels && !string.IsNullOrEmpty(region.Text) && region.Box != null)
        {
            string label = region.Text;
            if (_options.ShowConfidence)
            {
                double conf = _numOps.ToDouble(region.Confidence);
                label = $"{region.Text} ({conf:P0})";
            }
            DrawLabel(image, region.Box, label, _textBoxColor);
        }
    }

    private void DrawTextRegionHighlight(Tensor<T> image, RecognizedText<T> region)
    {
        if (region.Box == null)
            return;

        int height = image.Shape[2];
        int width = image.Shape[3];

        int x1 = Math.Max(0, (int)_numOps.ToDouble(region.Box.X1));
        int y1 = Math.Max(0, (int)_numOps.ToDouble(region.Box.Y1));
        int x2 = Math.Min(width - 1, (int)_numOps.ToDouble(region.Box.X2));
        int y2 = Math.Min(height - 1, (int)_numOps.ToDouble(region.Box.Y2));

        // Fill with white background
        for (int y = y1; y <= y2; y++)
        {
            for (int x = x1; x <= x2; x++)
            {
                SetPixel(image, 0, y, x, 0.95, 0.95, 0.95);
            }
        }

        // Draw border
        DrawRectangle(image, x1, y1, x2, y2, ((byte)50, (byte)50, (byte)50), 1);
    }

    private void DrawTextBox(Tensor<T> image, BoundingBox<T> box, (byte R, byte G, byte B) color)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        int x1 = Math.Max(0, (int)_numOps.ToDouble(box.X1));
        int y1 = Math.Max(0, (int)_numOps.ToDouble(box.Y1));
        int x2 = Math.Min(width - 1, (int)_numOps.ToDouble(box.X2));
        int y2 = Math.Min(height - 1, (int)_numOps.ToDouble(box.Y2));

        DrawRectangle(image, x1, y1, x2, y2, color, _options.BoxThickness);
    }

    private void DrawPolygon(Tensor<T> image, List<(T X, T Y)> polygon)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        var color = ((byte)255, (byte)0, (byte)255); // Magenta for polygon

        // Draw lines between consecutive points
        for (int i = 0; i < polygon.Count; i++)
        {
            int nextIdx = (i + 1) % polygon.Count;

            int x1 = (int)_numOps.ToDouble(polygon[i].X);
            int y1 = (int)_numOps.ToDouble(polygon[i].Y);
            int x2 = (int)_numOps.ToDouble(polygon[nextIdx].X);
            int y2 = (int)_numOps.ToDouble(polygon[nextIdx].Y);

            DrawLine(image, x1, y1, x2, y2, color);
        }

        // Draw corner points
        foreach (var point in polygon)
        {
            int px = (int)_numOps.ToDouble(point.X);
            int py = (int)_numOps.ToDouble(point.Y);
            DrawCircle(image, px, py, 3, color);
        }
    }

    private void DrawLine(Tensor<T> image, int x1, int y1, int x2, int y2,
        (byte R, byte G, byte B) color)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Bresenham's line algorithm
        int dx = Math.Abs(x2 - x1);
        int dy = Math.Abs(y2 - y1);
        int sx = x1 < x2 ? 1 : -1;
        int sy = y1 < y2 ? 1 : -1;
        int err = dx - dy;

        double r = color.R / 255.0;
        double g = color.G / 255.0;
        double b = color.B / 255.0;

        while (true)
        {
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
            {
                SetPixel(image, 0, y1, x1, r, g, b);
            }

            if (x1 == x2 && y1 == y2)
                break;

            int e2 = 2 * err;

            if (e2 > -dy)
            {
                err -= dy;
                x1 += sx;
            }

            if (e2 < dx)
            {
                err += dx;
                y1 += sy;
            }
        }
    }

    private void DrawCircle(Tensor<T> image, int cx, int cy, int radius,
        (byte R, byte G, byte B) color)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        double r = color.R / 255.0;
        double g = color.G / 255.0;
        double b = color.B / 255.0;

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx * dx + dy * dy <= radius * radius)
                {
                    int px = cx + dx;
                    int py = cy + dy;

                    if (px >= 0 && px < width && py >= 0 && py < height)
                    {
                        SetPixel(image, 0, py, px, r, g, b);
                    }
                }
            }
        }
    }

    private void DrawRectangle(Tensor<T> image, int x1, int y1, int x2, int y2,
        (byte R, byte G, byte B) color, int thickness)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        double r = color.R / 255.0;
        double g = color.G / 255.0;
        double b = color.B / 255.0;

        for (int t = 0; t < thickness; t++)
        {
            // Horizontal lines
            for (int x = x1; x <= x2; x++)
            {
                if (x >= 0 && x < width)
                {
                    if (y1 - t >= 0)
                        SetPixel(image, 0, y1 - t, x, r, g, b);
                    if (y2 + t < height)
                        SetPixel(image, 0, y2 + t, x, r, g, b);
                }
            }

            // Vertical lines
            for (int y = y1; y <= y2; y++)
            {
                if (y >= 0 && y < height)
                {
                    if (x1 - t >= 0)
                        SetPixel(image, 0, y, x1 - t, r, g, b);
                    if (x2 + t < width)
                        SetPixel(image, 0, y, x2 + t, r, g, b);
                }
            }
        }
    }

    private void DrawLabel(Tensor<T> image, BoundingBox<T> box, string label,
        (byte R, byte G, byte B) color)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        int x = Math.Max(0, (int)_numOps.ToDouble(box.X1));
        int y = Math.Max(0, (int)_numOps.ToDouble(box.Y1));

        // Calculate font scale based on options
        int scale = Math.Max(1, (int)_options.FontScale);

        // Calculate label dimensions
        int textWidth = _font.MeasureWidth(label) * scale;
        int textHeight = BitmapFont<T>.CharHeight * scale;
        int padding = 2 * scale;
        int labelHeight = textHeight + padding * 2;
        int labelWidth = Math.Min(textWidth + padding * 2, width - x);

        // Position label above the bounding box
        int labelY = Math.Max(0, y - labelHeight);
        int labelX = x;

        // Clamp to image bounds
        if (labelX + labelWidth > width)
            labelX = Math.Max(0, width - labelWidth);

        // Draw background
        double bgR = color.R / 255.0 * 0.8;
        double bgG = color.G / 255.0 * 0.8;
        double bgB = color.B / 255.0 * 0.8;

        for (int ly = labelY; ly < Math.Min(labelY + labelHeight, height); ly++)
        {
            for (int lx = labelX; lx < Math.Min(labelX + labelWidth, width); lx++)
            {
                SetPixel(image, 0, ly, lx, bgR, bgG, bgB);
            }
        }

        // Calculate text color (white or black for contrast)
        double luminance = 0.299 * bgR + 0.587 * bgG + 0.114 * bgB;
        var textColor = luminance > 0.5 ? (0.0, 0.0, 0.0) : (1.0, 1.0, 1.0);

        // Draw text using bitmap font
        _font.DrawText(image, label, labelX + padding, labelY + padding, textColor, scale);
    }

    private void DrawReadingOrder(Tensor<T> image, DocumentLayoutResult<T> layout)
    {
        if (layout.ReadingOrder == null || layout.ReadingOrder.Count < 2)
            return;

        var arrowColor = ((byte)255, (byte)255, (byte)0); // Yellow for reading order

        for (int i = 0; i < layout.ReadingOrder.Count - 1; i++)
        {
            int currentIdx = layout.ReadingOrder[i];
            int nextIdx = layout.ReadingOrder[i + 1];

            if (currentIdx < layout.Blocks.Count && nextIdx < layout.Blocks.Count)
            {
                var currentBlock = layout.Blocks[currentIdx];
                var nextBlock = layout.Blocks[nextIdx];

                // Get center points
                int cx1 = (int)((_numOps.ToDouble(currentBlock.Box.X1) + _numOps.ToDouble(currentBlock.Box.X2)) / 2);
                int cy1 = (int)((_numOps.ToDouble(currentBlock.Box.Y1) + _numOps.ToDouble(currentBlock.Box.Y2)) / 2);
                int cx2 = (int)((_numOps.ToDouble(nextBlock.Box.X1) + _numOps.ToDouble(nextBlock.Box.X2)) / 2);
                int cy2 = (int)((_numOps.ToDouble(nextBlock.Box.Y1) + _numOps.ToDouble(nextBlock.Box.Y2)) / 2);

                DrawLine(image, cx1, cy1, cx2, cy2, arrowColor);

                // Draw arrow head
                DrawArrowHead(image, cx1, cy1, cx2, cy2, arrowColor);
            }
        }
    }

    private void DrawArrowHead(Tensor<T> image, int x1, int y1, int x2, int y2,
        (byte R, byte G, byte B) color)
    {
        double angle = Math.Atan2(y2 - y1, x2 - x1);
        double arrowLength = 10;
        double arrowAngle = Math.PI / 6;

        int ax1 = (int)(x2 - arrowLength * Math.Cos(angle - arrowAngle));
        int ay1 = (int)(y2 - arrowLength * Math.Sin(angle - arrowAngle));
        int ax2 = (int)(x2 - arrowLength * Math.Cos(angle + arrowAngle));
        int ay2 = (int)(y2 - arrowLength * Math.Sin(angle + arrowAngle));

        DrawLine(image, x2, y2, ax1, ay1, color);
        DrawLine(image, x2, y2, ax2, ay2, color);
    }

    private (byte R, byte G, byte B) GetBlockTypeColor(DocumentBlockType blockType)
    {
        return blockType switch
        {
            DocumentBlockType.Title => ((byte)255, (byte)0, (byte)0),       // Red
            DocumentBlockType.Paragraph => ((byte)0, (byte)255, (byte)0),   // Green
            DocumentBlockType.List => ((byte)0, (byte)0, (byte)255),        // Blue
            DocumentBlockType.Table => ((byte)255, (byte)165, (byte)0),     // Orange
            DocumentBlockType.Figure => ((byte)128, (byte)0, (byte)128),    // Purple
            DocumentBlockType.Caption => ((byte)255, (byte)192, (byte)203), // Pink
            DocumentBlockType.Header => ((byte)0, (byte)128, (byte)128),    // Teal
            DocumentBlockType.Footer => ((byte)128, (byte)128, (byte)0),    // Olive
            DocumentBlockType.PageNumber => ((byte)169, (byte)169, (byte)169), // Gray
            _ => ((byte)100, (byte)100, (byte)100)
        };
    }

    private void SetPixel(Tensor<T> image, int batch, int y, int x, double r, double g, double b)
    {
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        if (x < 0 || x >= width || y < 0 || y >= height)
            return;

        if (channels >= 1)
            image[batch, 0, y, x] = _numOps.FromDouble(MathHelper.Clamp(r, 0, 1));
        if (channels >= 2)
            image[batch, 1, y, x] = _numOps.FromDouble(MathHelper.Clamp(g, 0, 1));
        if (channels >= 3)
            image[batch, 2, y, x] = _numOps.FromDouble(MathHelper.Clamp(b, 0, 1));
    }

    private Tensor<T> CloneImage(Tensor<T> image)
    {
        var clone = new Tensor<T>(image.Shape);
        for (int i = 0; i < image.Length; i++)
        {
            clone[i] = image[i];
        }
        return clone;
    }
}

/// <summary>
/// Result of document layout analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DocumentLayoutResult<T>
{
    /// <summary>
    /// Detected document blocks.
    /// </summary>
    public List<DocumentBlock<T>> Blocks { get; set; } = new();

    /// <summary>
    /// Reading order as indices into Blocks list.
    /// </summary>
    public List<int>? ReadingOrder { get; set; }

    /// <summary>
    /// Total inference time.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }
}

/// <summary>
/// A block in a document (paragraph, table, figure, etc.).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DocumentBlock<T>
{
    /// <summary>
    /// Bounding box of the block.
    /// </summary>
    public BoundingBox<T> Box { get; set; } = default!;

    /// <summary>
    /// Type of document block.
    /// </summary>
    public DocumentBlockType Type { get; set; }

    /// <summary>
    /// Confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Text content (if applicable).
    /// </summary>
    public string? Text { get; set; }

    /// <summary>
    /// Child text regions within this block.
    /// </summary>
    public List<RecognizedText<T>>? TextRegions { get; set; }
}

/// <summary>
/// Types of document blocks.
/// </summary>
public enum DocumentBlockType
{
    /// <summary>Document title.</summary>
    Title,
    /// <summary>Section heading.</summary>
    Heading,
    /// <summary>Paragraph of text.</summary>
    Paragraph,
    /// <summary>Bulleted or numbered list.</summary>
    List,
    /// <summary>Table.</summary>
    Table,
    /// <summary>Figure or image.</summary>
    Figure,
    /// <summary>Figure or table caption.</summary>
    Caption,
    /// <summary>Page header.</summary>
    Header,
    /// <summary>Page footer.</summary>
    Footer,
    /// <summary>Page number.</summary>
    PageNumber,
    /// <summary>Other/unknown block type.</summary>
    Other
}
