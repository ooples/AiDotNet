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
            output = DrawMaskOverlay(output, instance.Mask, i, instance.ClassId);
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
            output = DrawMaskOverlay(output, instance.Mask, i, instance.ClassId);
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
            semanticMap = PaintWhere(semanticMap, instance.Mask, 3, ColorTensor(color, 3));
        }

        return semanticMap;
    }

    /// <summary>
    /// Creates an instance ID map where each pixel contains the instance index.
    /// </summary>
    public int[,] CreateInstanceIdMap(InstanceSegmentationResult<T> result, int height, int width)
    {
        var engine = AiDotNetEngine.Current;

        // Initialize with -1 (background); later instances overwrite earlier ones.
        var instanceMap = new Tensor<T>(new[] { 1, 1, height, width });
        engine.TensorFill(instanceMap, _numOps.FromDouble(-1.0));

        for (int i = 0; i < result.Instances.Count; i++)
        {
            var id = new Tensor<T>(new[] { _numOps.FromDouble(i) }, new[] { 1, 1, 1, 1 });
            instanceMap = PaintWhere(instanceMap, result.Instances[i].Mask, 1, id);
        }

        var ids = new int[height, width];
        var flat = instanceMap.Cast<int>().ToArray();
        Buffer.BlockCopy(flat, 0, ids, 0, flat.Length * sizeof(int));
        return ids;
    }

    private Tensor<T> DrawMaskOverlay(Tensor<T> image, Tensor<T> mask, int instanceId, int classId)
    {
        var engine = AiDotNetEngine.Current;
        int channels = image.Shape[1];
        int imgH = image.Shape[2];
        int imgW = image.Shape[3];

        var color = GetInstanceColor(instanceId, classId);
        T opacity = _numOps.FromDouble(_options.MaskOpacity);
        T keep = _numOps.Subtract(_numOps.One, opacity);

        int h = Math.Min(mask.Shape[0], imgH);
        int w = Math.Min(mask.Shape[1], imgW);
        int blendChannels = channels >= 3 ? 3 : channels == 1 ? 1 : 0;

        if (blendChannels > 0 && h > 0 && w > 0)
        {
            // Blend with original pixel: orig * (1 - opacity) + color * opacity, inside the mask only.
            Tensor<T> tint = blendChannels == 3
                ? ColorTensor(color, 3)
                : new Tensor<T>(new[] { _numOps.FromDouble((color.R + color.G + color.B) / 3.0 / 255.0) }, new[] { 1, 1, 1, 1 });
            var region = engine.TensorSlice(image, new[] { 0, 0, 0, 0 }, new[] { 1, blendChannels, h, w });
            var blended = engine.TensorAdd(
                engine.TensorMultiplyScalar(region, keep),
                engine.TensorBroadcastTo(engine.TensorMultiplyScalar(tint, opacity), region.Shape.ToArray()));
            var inside = engine.TensorBroadcastTo(
                Reshape4D(MaskOn(engine.TensorSlice(mask, new[] { 0, 0 }, new[] { h, w }))), region.Shape.ToArray());
            image = engine.TensorSetSlice(image, engine.TensorWhere(inside, blended, region), new[] { 0, 0, 0, 0 });
        }

        // Draw mask contour
        return DrawMaskContour(image, mask, color);
    }

    private Tensor<T> DrawMaskContour(Tensor<T> image, Tensor<T> mask, (byte R, byte G, byte B) color)
    {
        var engine = AiDotNetEngine.Current;
        int channels = image.Shape[1];
        int imgH = image.Shape[2];
        int imgW = image.Shape[3];

        // Contour pixels are interior pixels y in [1, yEnd), x in [1, xEnd) that are on while at least
        // one 4-neighbour is off.
        int yEnd = Math.Min(mask.Shape[0] - 1, imgH - 1);
        int xEnd = Math.Min(mask.Shape[1] - 1, imgW - 1);
        int h = yEnd - 1;
        int w = xEnd - 1;
        int paintChannels = Math.Min(channels, 3);
        if (h <= 0 || w <= 0 || paintChannels <= 0)
        {
            return image;
        }

        var on = MaskOn(engine.TensorSlice(mask, new[] { 0, 0 }, new[] { h + 2, w + 2 }));
        var centre = engine.TensorSlice(on, new[] { 1, 1 }, new[] { h, w });
        var up = engine.TensorSlice(on, new[] { 0, 1 }, new[] { h, w });
        var down = engine.TensorSlice(on, new[] { 2, 1 }, new[] { h, w });
        var left = engine.TensorSlice(on, new[] { 1, 0 }, new[] { h, w });
        var right = engine.TensorSlice(on, new[] { 1, 2 }, new[] { h, w });
        var allNeighboursOn = engine.TensorMultiply(engine.TensorMultiply(up, down), engine.TensorMultiply(left, right));
        var edge = engine.TensorMultiply(centre, engine.ScalarMinusTensor(_numOps.One, allNeighboursOn));

        var region = engine.TensorSlice(image, new[] { 0, 0, 1, 1 }, new[] { 1, paintChannels, h, w });
        var edgeMask = engine.TensorBroadcastTo(Reshape4D(edge), region.Shape.ToArray());
        var paint = engine.TensorBroadcastTo(ColorTensor(color, paintChannels), region.Shape.ToArray());
        return engine.TensorSetSlice(image, engine.TensorWhere(edgeMask, paint, region), new[] { 0, 0, 1, 1 });
    }

    /// <summary>
    /// Writes <paramref name="value"/> (broadcast from <c>[1, channels, 1, 1]</c>) into
    /// the first batch item's first <paramref name="channels"/> channels wherever the mask is above 0.5,
    /// over the overlap of the mask and the target's spatial extent.
    /// </summary>
    private Tensor<T> PaintWhere(Tensor<T> target, Tensor<T> mask, int channels, Tensor<T> value)
    {
        var engine = AiDotNetEngine.Current;
        int h = Math.Min(mask.Shape[0], target.Shape[2]);
        int w = Math.Min(mask.Shape[1], target.Shape[3]);
        if (h <= 0 || w <= 0)
        {
            return target;
        }

        var start = new[] { 0, 0, 0, 0 };
        var region = engine.TensorSlice(target, start, new[] { 1, channels, h, w });
        var inside = engine.TensorBroadcastTo(
            Reshape4D(MaskOn(engine.TensorSlice(mask, new[] { 0, 0 }, new[] { h, w }))), region.Shape.ToArray());
        var paint = engine.TensorBroadcastTo(value, region.Shape.ToArray());
        return engine.TensorSetSlice(target, engine.TensorWhere(inside, paint, region), start);
    }

    /// <summary>1 where the mask is above 0.5, else 0.</summary>
    private Tensor<T> MaskOn(Tensor<T> mask)
        => AiDotNetEngine.Current.TensorGreaterThan(mask, _numOps.FromDouble(0.5));

    /// <summary>Views a <c>[h, w]</c> plane as <c>[1, 1, h, w]</c>.</summary>
    private static Tensor<T> Reshape4D(Tensor<T> plane)
        => AiDotNetEngine.Current.Reshape(plane, new[] { 1, 1, plane.Shape[0], plane.Shape[1] });

    /// <summary>The first <paramref name="channels"/> of (R, G, B) / 255 as <c>[1, channels, 1, 1]</c>.</summary>
    private Tensor<T> ColorTensor((byte R, byte G, byte B) color, int channels)
    {
        var rgb = new[]
        {
            _numOps.FromDouble(color.R / 255.0),
            _numOps.FromDouble(color.G / 255.0),
            _numOps.FromDouble(color.B / 255.0),
        };
        return new Tensor<T>(rgb.Take(channels).ToArray(), new[] { 1, channels, 1, 1 });
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

    private static Tensor<T> CloneImage(Tensor<T> image) => image.Clone();
}
