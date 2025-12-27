using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Visualization;

/// <summary>
/// Visualizes instance segmentation results on images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class overlays colored masks and bounding boxes
/// on images to visualize instance segmentation results.</para>
/// </remarks>
public class MaskVisualizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly VisualizationOptions _options;
    private readonly DetectionVisualizer<T> _detectionVisualizer;
    private readonly Dictionary<int, (byte R, byte G, byte B)> _instanceColors;
    private readonly BitmapFont<T> _font;

    /// <summary>
    /// Creates a new mask visualizer.
    /// </summary>
    public MaskVisualizer(VisualizationOptions? options = null)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _options = options ?? new VisualizationOptions();
        _detectionVisualizer = new DetectionVisualizer<T>(_options);
        _instanceColors = new Dictionary<int, (byte R, byte G, byte B)>();
        _font = new BitmapFont<T>();
    }

    /// <summary>
    /// Draws instance segmentation results on an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <param name="result">Instance segmentation result to visualize.</param>
    /// <param name="classNames">Optional class name mapping.</param>
    /// <returns>Image with visualizations drawn.</returns>
    public Tensor<T> Visualize(Tensor<T> image, InstanceSegmentationResult<T> result, string[]? classNames = null)
    {
        // Clone image for drawing
        var output = CloneImage(image);

        // Draw masks first (so boxes appear on top)
        for (int i = 0; i < result.Instances.Count; i++)
        {
            var instance = result.Instances[i];
            DrawMaskOverlay(output, instance.Mask, i, instance.ClassId);
        }

        // Draw bounding boxes and labels
        for (int i = 0; i < result.Instances.Count; i++)
        {
            var instance = result.Instances[i];
            DrawBoundingBox(output, instance, classNames);
        }

        return output;
    }

    /// <summary>
    /// Draws only the mask overlay without bounding boxes.
    /// </summary>
    public Tensor<T> VisualizeMasksOnly(Tensor<T> image, InstanceSegmentationResult<T> result)
    {
        var output = CloneImage(image);

        for (int i = 0; i < result.Instances.Count; i++)
        {
            var instance = result.Instances[i];
            DrawMaskOverlay(output, instance.Mask, i, instance.ClassId);
        }

        return output;
    }

    /// <summary>
    /// Creates a combined semantic segmentation map from instances.
    /// </summary>
    public Tensor<T> CreateSemanticMap(InstanceSegmentationResult<T> result, int height, int width)
    {
        var semanticMap = new Tensor<T>(new[] { 1, 3, height, width });

        foreach (var instance in result.Instances)
        {
            var color = GetClassColor(instance.ClassId);

            for (int y = 0; y < Math.Min(instance.Mask.Shape[0], height); y++)
            {
                for (int x = 0; x < Math.Min(instance.Mask.Shape[1], width); x++)
                {
                    if (_numOps.ToDouble(instance.Mask[y, x]) > 0.5)
                    {
                        semanticMap[0, 0, y, x] = _numOps.FromDouble(color.R / 255.0);
                        semanticMap[0, 1, y, x] = _numOps.FromDouble(color.G / 255.0);
                        semanticMap[0, 2, y, x] = _numOps.FromDouble(color.B / 255.0);
                    }
                }
            }
        }

        return semanticMap;
    }

    /// <summary>
    /// Creates an instance ID map where each pixel contains the instance index.
    /// </summary>
    public int[,] CreateInstanceIdMap(InstanceSegmentationResult<T> result, int height, int width)
    {
        var instanceMap = new int[height, width];

        // Initialize with -1 (background)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                instanceMap[y, x] = -1;
            }
        }

        for (int i = 0; i < result.Instances.Count; i++)
        {
            var instance = result.Instances[i];

            for (int y = 0; y < Math.Min(instance.Mask.Shape[0], height); y++)
            {
                for (int x = 0; x < Math.Min(instance.Mask.Shape[1], width); x++)
                {
                    if (_numOps.ToDouble(instance.Mask[y, x]) > 0.5)
                    {
                        instanceMap[y, x] = i;
                    }
                }
            }
        }

        return instanceMap;
    }

    private void DrawMaskOverlay(Tensor<T> image, Tensor<T> mask, int instanceId, int classId)
    {
        int channels = image.Shape[1];
        int imgH = image.Shape[2];
        int imgW = image.Shape[3];

        var color = GetInstanceColor(instanceId, classId);
        double opacity = _options.MaskOpacity;

        int maskH = mask.Shape[0];
        int maskW = mask.Shape[1];

        for (int y = 0; y < Math.Min(maskH, imgH); y++)
        {
            for (int x = 0; x < Math.Min(maskW, imgW); x++)
            {
                double maskVal = _numOps.ToDouble(mask[y, x]);

                if (maskVal > 0.5)
                {
                    // Blend with original pixel
                    if (channels >= 3)
                    {
                        double origR = _numOps.ToDouble(image[0, 0, y, x]);
                        double origG = _numOps.ToDouble(image[0, 1, y, x]);
                        double origB = _numOps.ToDouble(image[0, 2, y, x]);

                        double newR = origR * (1 - opacity) + (color.R / 255.0) * opacity;
                        double newG = origG * (1 - opacity) + (color.G / 255.0) * opacity;
                        double newB = origB * (1 - opacity) + (color.B / 255.0) * opacity;

                        image[0, 0, y, x] = _numOps.FromDouble(newR);
                        image[0, 1, y, x] = _numOps.FromDouble(newG);
                        image[0, 2, y, x] = _numOps.FromDouble(newB);
                    }
                    else if (channels == 1)
                    {
                        double origVal = _numOps.ToDouble(image[0, 0, y, x]);
                        double colorGray = (color.R + color.G + color.B) / 3.0 / 255.0;
                        double newVal = origVal * (1 - opacity) + colorGray * opacity;
                        image[0, 0, y, x] = _numOps.FromDouble(newVal);
                    }
                }
            }
        }

        // Draw mask contour
        DrawMaskContour(image, mask, color);
    }

    private void DrawMaskContour(Tensor<T> image, Tensor<T> mask, (byte R, byte G, byte B) color)
    {
        int imgH = image.Shape[2];
        int imgW = image.Shape[3];
        int maskH = mask.Shape[0];
        int maskW = mask.Shape[1];

        // Find contour pixels (mask edge)
        for (int y = 1; y < Math.Min(maskH - 1, imgH - 1); y++)
        {
            for (int x = 1; x < Math.Min(maskW - 1, imgW - 1); x++)
            {
                double centerVal = _numOps.ToDouble(mask[y, x]);

                if (centerVal > 0.5)
                {
                    // Check if this is an edge pixel
                    bool isEdge = false;

                    if (_numOps.ToDouble(mask[y - 1, x]) <= 0.5 ||
                        _numOps.ToDouble(mask[y + 1, x]) <= 0.5 ||
                        _numOps.ToDouble(mask[y, x - 1]) <= 0.5 ||
                        _numOps.ToDouble(mask[y, x + 1]) <= 0.5)
                    {
                        isEdge = true;
                    }

                    if (isEdge)
                    {
                        // Draw contour pixel
                        SetPixel(image, 0, y, x, color.R / 255.0, color.G / 255.0, color.B / 255.0);
                    }
                }
            }
        }
    }

    private void DrawBoundingBox(Tensor<T> image, InstanceMask<T> instance, string[]? classNames)
    {
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var box = instance.Box;
        int x1 = Math.Max(0, (int)_numOps.ToDouble(box.X1));
        int y1 = Math.Max(0, (int)_numOps.ToDouble(box.Y1));
        int x2 = Math.Min(width - 1, (int)_numOps.ToDouble(box.X2));
        int y2 = Math.Min(height - 1, (int)_numOps.ToDouble(box.Y2));

        var color = GetClassColor(instance.ClassId);

        // Draw rectangle
        DrawRectangle(image, x1, y1, x2, y2, color, _options.BoxThickness);

        // Draw label
        if (_options.ShowLabels)
        {
            string label = GetLabel(instance, classNames);
            DrawLabel(image, x1, y1, label, color);
        }
    }

    private string GetLabel(InstanceMask<T> instance, string[]? classNames)
    {
        string className = classNames != null && instance.ClassId < classNames.Length
            ? classNames[instance.ClassId]
            : $"Class {instance.ClassId}";

        if (_options.ShowConfidence)
        {
            double conf = _numOps.ToDouble(instance.Confidence);
            return $"{className} {conf:P0}";
        }

        return className;
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
                if (y1 - t >= 0)
                    SetPixel(image, 0, y1 - t, x, r, g, b);
                if (y2 + t < height)
                    SetPixel(image, 0, y2 + t, x, r, g, b);
            }

            // Vertical lines
            for (int y = y1; y <= y2; y++)
            {
                if (x1 - t >= 0)
                    SetPixel(image, 0, y, x1 - t, r, g, b);
                if (x2 + t < width)
                    SetPixel(image, 0, y, x2 + t, r, g, b);
            }
        }
    }

    private void DrawLabel(Tensor<T> image, int x, int y, string label, (byte R, byte G, byte B) color)
    {
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Calculate font scale based on options
        int scale = Math.Max(1, (int)_options.FontScale);

        // Calculate label dimensions
        int textWidth = _font.MeasureWidth(label) * scale;
        int textHeight = BitmapFont<T>.CharHeight * scale;
        int padding = 2 * scale;
        int labelHeight = textHeight + padding * 2;
        int labelWidth = textWidth + padding * 2;

        // Position label above the bounding box
        int labelY = Math.Max(0, y - labelHeight);
        int labelX = Math.Max(0, x);

        // Clamp to image bounds
        if (labelX + labelWidth > width)
            labelX = Math.Max(0, width - labelWidth);

        // Draw background
        double bgR = color.R / 255.0;
        double bgG = color.G / 255.0;
        double bgB = color.B / 255.0;

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

    private void SetPixel(Tensor<T> image, int batch, int y, int x, double r, double g, double b)
    {
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        if (x < 0 || x >= width || y < 0 || y >= height)
            return;

        if (channels >= 1)
            image[batch, 0, y, x] = _numOps.FromDouble(r);
        if (channels >= 2)
            image[batch, 1, y, x] = _numOps.FromDouble(g);
        if (channels >= 3)
            image[batch, 2, y, x] = _numOps.FromDouble(b);
    }

    private (byte R, byte G, byte B) GetInstanceColor(int instanceId, int classId)
    {
        // Use instance-specific color for better distinction
        int key = instanceId * 1000 + classId;

        if (_instanceColors.TryGetValue(key, out var color))
        {
            return color;
        }

        var newColor = GenerateColor(key);
        _instanceColors[key] = newColor;
        return newColor;
    }

    private (byte R, byte G, byte B) GetClassColor(int classId)
    {
        if (_instanceColors.TryGetValue(classId + 100000, out var color))
        {
            return color;
        }

        var newColor = GenerateColor(classId);
        _instanceColors[classId + 100000] = newColor;
        return newColor;
    }

    private (byte R, byte G, byte B) GenerateColor(int seed)
    {
        double hue = (seed * 0.618033988749895) % 1.0;
        double saturation = 0.7 + (seed % 3) * 0.1;
        double value = 0.85 + (seed % 2) * 0.1;

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
