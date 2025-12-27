using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Visualization;

/// <summary>
/// Visualizes object detection results on images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class draws bounding boxes, class labels, and
/// confidence scores on images to visualize detection results.</para>
/// </remarks>
public class DetectionVisualizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly VisualizationOptions _options;
    private readonly Dictionary<int, (byte R, byte G, byte B)> _classColors;

    /// <summary>
    /// Creates a new detection visualizer.
    /// </summary>
    public DetectionVisualizer(VisualizationOptions? options = null)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _options = options ?? new VisualizationOptions();
        _classColors = new Dictionary<int, (byte R, byte G, byte B)>();
    }

    /// <summary>
    /// Draws detection results on an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <param name="result">Detection result to visualize.</param>
    /// <param name="classNames">Optional class name mapping.</param>
    /// <returns>Image with visualizations drawn.</returns>
    public Tensor<T> Visualize(Tensor<T> image, DetectionResult<T> result, string[]? classNames = null)
    {
        // Clone image for drawing
        var output = CloneImage(image);

        foreach (var detection in result.Detections)
        {
            DrawDetection(output, detection, classNames);
        }

        return output;
    }

    /// <summary>
    /// Draws a single detection on an image.
    /// </summary>
    public void DrawDetection(Tensor<T> image, Detection<T> detection, string[]? classNames = null)
    {
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Get bounding box coordinates
        int x1 = Math.Max(0, (int)_numOps.ToDouble(detection.Box.X1));
        int y1 = Math.Max(0, (int)_numOps.ToDouble(detection.Box.Y1));
        int x2 = Math.Min(width - 1, (int)_numOps.ToDouble(detection.Box.X2));
        int y2 = Math.Min(height - 1, (int)_numOps.ToDouble(detection.Box.Y2));

        // Get color for this class
        var color = GetClassColor(detection.ClassId);

        // Draw bounding box
        DrawRectangle(image, x1, y1, x2, y2, color, _options.BoxThickness);

        // Draw label background and text
        if (_options.ShowLabels)
        {
            string label = GetLabel(detection, classNames);
            DrawLabel(image, x1, y1, label, color);
        }
    }

    private string GetLabel(Detection<T> detection, string[]? classNames)
    {
        string className = detection.ClassName ?? string.Empty;
        if (string.IsNullOrEmpty(className) && classNames != null && detection.ClassId < classNames.Length)
        {
            className = classNames[detection.ClassId];
        }
        else if (string.IsNullOrEmpty(className))
        {
            className = $"Class {detection.ClassId}";
        }

        if (_options.ShowConfidence)
        {
            double conf = _numOps.ToDouble(detection.Confidence);
            return $"{className} {conf:P0}";
        }

        return className;
    }

    private void DrawRectangle(Tensor<T> image, int x1, int y1, int x2, int y2,
        (byte R, byte G, byte B) color, int thickness)
    {
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Normalize color to 0-1 range
        double r = color.R / 255.0;
        double g = color.G / 255.0;
        double b = color.B / 255.0;

        // Draw horizontal lines (top and bottom)
        for (int t = 0; t < thickness; t++)
        {
            // Top line
            int ty = y1 - t;
            if (ty >= 0 && ty < height)
            {
                for (int x = x1; x <= x2; x++)
                {
                    if (x >= 0 && x < width)
                    {
                        SetPixel(image, 0, ty, x, r, g, b);
                    }
                }
            }

            // Bottom line
            int by = y2 + t;
            if (by >= 0 && by < height)
            {
                for (int x = x1; x <= x2; x++)
                {
                    if (x >= 0 && x < width)
                    {
                        SetPixel(image, 0, by, x, r, g, b);
                    }
                }
            }
        }

        // Draw vertical lines (left and right)
        for (int t = 0; t < thickness; t++)
        {
            // Left line
            int lx = x1 - t;
            if (lx >= 0 && lx < width)
            {
                for (int y = y1; y <= y2; y++)
                {
                    if (y >= 0 && y < height)
                    {
                        SetPixel(image, 0, y, lx, r, g, b);
                    }
                }
            }

            // Right line
            int rx = x2 + t;
            if (rx >= 0 && rx < width)
            {
                for (int y = y1; y <= y2; y++)
                {
                    if (y >= 0 && y < height)
                    {
                        SetPixel(image, 0, y, rx, r, g, b);
                    }
                }
            }
        }
    }

    private void DrawLabel(Tensor<T> image, int x, int y, string label,
        (byte R, byte G, byte B) color)
    {
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Simple label background (8 pixels high per character, 6 wide)
        int labelHeight = 18;
        int labelWidth = label.Length * 8 + 4;

        int labelY = Math.Max(0, y - labelHeight);
        int labelX = Math.Max(0, x);

        // Draw background
        double r = color.R / 255.0;
        double g = color.G / 255.0;
        double b = color.B / 255.0;

        for (int ly = labelY; ly < Math.Min(labelY + labelHeight, height); ly++)
        {
            for (int lx = labelX; lx < Math.Min(labelX + labelWidth, width); lx++)
            {
                SetPixel(image, 0, ly, lx, r, g, b);
            }
        }

        // Draw text (simplified: just white dots for now)
        // In a real implementation, you'd use a bitmap font
        DrawSimpleText(image, label, labelX + 2, labelY + 2);
    }

    private void DrawSimpleText(Tensor<T> image, string text, int x, int y)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Simple 5x7 font representation (just draw white rectangles for each char)
        int charWidth = 6;
        int charHeight = 10;

        for (int i = 0; i < text.Length; i++)
        {
            int cx = x + i * charWidth;
            if (cx >= width)
                break;

            // Draw a simple character representation (white rectangle)
            for (int cy = y; cy < Math.Min(y + charHeight, height); cy++)
            {
                for (int cxp = cx; cxp < Math.Min(cx + charWidth - 1, width); cxp++)
                {
                    // Skip some pixels to make it look like text
                    if (cy == y || cy == y + charHeight - 1)
                    {
                        SetPixel(image, 0, cy, cxp, 1.0, 1.0, 1.0);
                    }
                }
            }
        }
    }

    private void SetPixel(Tensor<T> image, int batch, int y, int x, double r, double g, double b)
    {
        int channels = image.Shape[1];

        if (channels >= 1)
            image[batch, 0, y, x] = _numOps.FromDouble(r);
        if (channels >= 2)
            image[batch, 1, y, x] = _numOps.FromDouble(g);
        if (channels >= 3)
            image[batch, 2, y, x] = _numOps.FromDouble(b);
    }

    private (byte R, byte G, byte B) GetClassColor(int classId)
    {
        if (_classColors.TryGetValue(classId, out var color))
        {
            return color;
        }

        // Generate consistent color for class
        var newColor = GenerateColor(classId);
        _classColors[classId] = newColor;
        return newColor;
    }

    private (byte R, byte G, byte B) GenerateColor(int classId)
    {
        // Use golden ratio to generate well-distributed colors
        double hue = (classId * 0.618033988749895) % 1.0;
        double saturation = 0.8;
        double value = 0.9;

        return HsvToRgb(hue, saturation, value);
    }

    private (byte R, byte G, byte B) HsvToRgb(double h, double s, double v)
    {
        double r, g, b;

        int i = (int)(h * 6);
        double f = h * 6 - i;
        double p = v * (1 - s);
        double q = v * (1 - f * s);
        double t = v * (1 - (1 - f) * s);

        switch (i % 6)
        {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }

        return ((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
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
/// Options for visualization.
/// </summary>
public class VisualizationOptions
{
    /// <summary>
    /// Thickness of bounding boxes in pixels.
    /// </summary>
    public int BoxThickness { get; set; } = 2;

    /// <summary>
    /// Whether to show class labels.
    /// </summary>
    public bool ShowLabels { get; set; } = true;

    /// <summary>
    /// Whether to show confidence scores.
    /// </summary>
    public bool ShowConfidence { get; set; } = true;

    /// <summary>
    /// Opacity for mask overlays (0-1).
    /// </summary>
    public double MaskOpacity { get; set; } = 0.5;

    /// <summary>
    /// Font scale for text.
    /// </summary>
    public double FontScale { get; set; } = 1.0;

    /// <summary>
    /// Whether to show polygon contours for text regions.
    /// </summary>
    public bool ShowPolygons { get; set; } = true;

    /// <summary>
    /// Custom class colors (classId -> RGB).
    /// </summary>
    public Dictionary<int, (byte R, byte G, byte B)>? ClassColors { get; set; }
}
