using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>
/// DBNet (Differentiable Binarization Network) text detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DBNet is a state-of-the-art text detector that uses
/// differentiable binarization to segment text regions. Unlike traditional methods
/// that use a fixed threshold, DBNet learns an adaptive threshold for each pixel,
/// making it more robust to varying text appearances.</para>
///
/// <para>Key features:
/// - Differentiable binarization for end-to-end training
/// - Adaptive thresholding per pixel
/// - Fast inference with single-pass architecture
/// - Works well for both regular and irregular text shapes
/// </para>
///
/// <para>Reference: Liao et al., "Real-time Scene Text Detection with Differentiable
/// Binarization", AAAI 2020</para>
/// </remarks>
public class DBNet<T> : TextDetectorBase<T>
{
    private readonly Conv2D<T> _inConv;
    private readonly Conv2D<T> _upConv1;
    private readonly Conv2D<T> _upConv2;
    private readonly Conv2D<T> _upConv3;
    private readonly Conv2D<T> _probHead;
    private readonly Conv2D<T> _threshHead;
    private readonly int _hiddenDim;
    private readonly double _k;

    /// <inheritdoc/>
    public override string Name => $"DBNet-{Options.Size}";

    /// <summary>
    /// Creates a new DBNet text detector.
    /// </summary>
    /// <param name="options">Text detection options.</param>
    /// <param name="k">Amplification factor for differentiable binarization (default 50).</param>
    public DBNet(TextDetectionOptions<T> options, double k = 50.0) : base(options)
    {
        _hiddenDim = GetHiddenDim(options.Size);
        _k = k;

        // ResNet backbone
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // Feature pyramid for multi-scale fusion
        int backboneChannels = Backbone.OutputChannels[^1];
        _inConv = new Conv2D<T>(backboneChannels, _hiddenDim, kernelSize: 1);
        _upConv1 = new Conv2D<T>(_hiddenDim * 2, _hiddenDim, kernelSize: 3, padding: 1);
        _upConv2 = new Conv2D<T>(_hiddenDim * 2, _hiddenDim, kernelSize: 3, padding: 1);
        _upConv3 = new Conv2D<T>(_hiddenDim, _hiddenDim / 2, kernelSize: 3, padding: 1);

        // Probability map head (text probability)
        _probHead = new Conv2D<T>(_hiddenDim / 2, 1, kernelSize: 1);

        // Threshold map head (adaptive threshold)
        _threshHead = new Conv2D<T>(_hiddenDim / 2, 1, kernelSize: 1);
    }

    private static int GetHiddenDim(ModelSize size) => size switch
    {
        ModelSize.Nano => 64,
        ModelSize.Small => 128,
        ModelSize.Medium => 256,
        ModelSize.Large => 384,
        ModelSize.XLarge => 512,
        _ => 256
    };

    /// <inheritdoc/>
    public override TextDetectionResult<T> Detect(Tensor<T> image)
    {
        return Detect(image, NumOps.ToDouble(Options.ConfidenceThreshold));
    }

    /// <inheritdoc/>
    public override TextDetectionResult<T> Detect(Tensor<T> image, double confidenceThreshold)
    {
        var startTime = DateTime.UtcNow;

        int originalHeight = image.Shape[2];
        int originalWidth = image.Shape[3];

        var input = Preprocess(image);
        var outputs = Forward(input);
        var textRegions = PostProcess(outputs, originalWidth, originalHeight, confidenceThreshold);

        return new TextDetectionResult<T>
        {
            TextRegions = textRegions,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = originalWidth,
            ImageHeight = originalHeight
        };
    }

    /// <inheritdoc/>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        // Extract multi-scale backbone features
        var features = Backbone!.ExtractFeatures(input);

        // Feature pyramid fusion
        var x = _inConv.Forward(features[^1]);
        x = ApplyReLU(x);

        if (features.Count > 1)
        {
            x = UpsampleAndConcat(x, features[^2]);
            x = _upConv1.Forward(x);
            x = ApplyReLU(x);
        }

        if (features.Count > 2)
        {
            x = UpsampleAndConcat(x, features[^3]);
            x = _upConv2.Forward(x);
            x = ApplyReLU(x);
        }

        x = _upConv3.Forward(x);
        x = ApplyReLU(x);

        // Predict probability and threshold maps
        var probMap = _probHead.Forward(x);
        probMap = ApplySigmoid(probMap);

        var threshMap = _threshHead.Forward(x);
        threshMap = ApplySigmoid(threshMap);

        // Apply differentiable binarization: DB = 1 / (1 + exp(-k * (P - T)))
        var binaryMap = ApplyDifferentiableBinarization(probMap, threshMap);

        return new List<Tensor<T>> { probMap, threshMap, binaryMap };
    }

    /// <inheritdoc/>
    protected override List<TextRegion<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold)
    {
        var probMap = outputs[0];
        var binaryMap = outputs[2];

        int mapH = binaryMap.Shape[2];
        int mapW = binaryMap.Shape[3];

        double scaleX = (double)imageWidth / mapW;
        double scaleY = (double)imageHeight / mapH;

        // Binarize the probability map
        double binThreshold = NumOps.ToDouble(Options.BinaryThreshold);
        var textMask = new bool[mapH, mapW];

        for (int h = 0; h < mapH; h++)
        {
            for (int w = 0; w < mapW; w++)
            {
                textMask[h, w] = NumOps.ToDouble(binaryMap[0, 0, h, w]) > binThreshold;
            }
        }

        // Find connected components
        var components = FindConnectedComponents(textMask, mapH, mapW);

        // Convert components to text regions
        var regions = new List<TextRegion<T>>();

        foreach (var component in components)
        {
            if (component.Count < 10)
                continue;

            // Compute bounding contour
            var contour = GetContour(component, textMask, mapH, mapW);

            if (contour.Count < 4)
                continue;

            // Compute average probability as confidence
            double avgProb = 0;
            foreach (var (h, w) in component)
            {
                avgProb += NumOps.ToDouble(probMap[0, 0, h, w]);
            }
            avgProb /= component.Count;

            if (avgProb < confidenceThreshold)
                continue;

            // Scale contour to original image coordinates
            var polygon = contour
                .Select(p => (X: p.W * scaleX, Y: p.H * scaleY))
                .ToList();

            // Simplify polygon
            polygon = SimplifyPolygon(polygon, Options.PolygonSimplificationEpsilon * Math.Max(scaleX, scaleY));

            if (polygon.Count >= 4)
            {
                var region = TextRegion<T>.FromPolygon(
                    polygon.Select(p => (NumOps.FromDouble(p.X), NumOps.FromDouble(p.Y))).ToList(),
                    NumOps.FromDouble(avgProb));

                region.RegionType = TextRegionType.Word;
                regions.Add(region);
            }
        }

        // Apply polygon NMS
        regions = ApplyPolygonNMS(regions, 0.3);

        // Limit to max detections
        if (regions.Count > Options.MaxDetections)
        {
            regions = regions
                .OrderByDescending(r => NumOps.ToDouble(r.Confidence))
                .Take(Options.MaxDetections)
                .ToList();
        }

        return regions;
    }

    private Tensor<T> ApplyDifferentiableBinarization(Tensor<T> prob, Tensor<T> thresh)
    {
        var result = new Tensor<T>(prob.Shape);

        for (int i = 0; i < prob.Length; i++)
        {
            double p = NumOps.ToDouble(prob[i]);
            double t = NumOps.ToDouble(thresh[i]);

            // DB formula: 1 / (1 + exp(-k * (P - T)))
            double db = 1.0 / (1.0 + Math.Exp(-_k * (p - t)));
            result[i] = NumOps.FromDouble(db);
        }

        return result;
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        return _inConv.GetParameterCount() +
               _upConv1.GetParameterCount() +
               _upConv2.GetParameterCount() +
               _upConv3.GetParameterCount() +
               _probHead.GetParameterCount() +
               _threshHead.GetParameterCount();
    }

    /// <inheritdoc/>
    public override async Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        byte[] data;
        if (pathOrUrl.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            pathOrUrl.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            using var client = new System.Net.Http.HttpClient();
            data = await client.GetByteArrayWithCancellationAsync(pathOrUrl, cancellationToken);
        }
        else
        {
            // Use Task.Run for net471 compatibility (ReadAllBytesAsync not available)
            data = await Task.Run(() => File.ReadAllBytes(pathOrUrl), cancellationToken);
        }

        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and verify header
        int magic = reader.ReadInt32();
        if (magic != 0x44424E54) // "DBNT" in ASCII
        {
            throw new InvalidDataException($"Invalid DBNet model file. Expected magic 0x44424E54, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported DBNet model version: {version}");
        }

        string name = reader.ReadString();
        int hiddenDim = reader.ReadInt32();
        double k = reader.ReadDouble();

        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidOperationException(
                $"DBNet configuration mismatch. Expected hiddenDim={_hiddenDim}, got hiddenDim={hiddenDim}");
        }

        // Read component weights
        Backbone!.ReadParameters(reader);
        _inConv.ReadParameters(reader);
        _upConv1.ReadParameters(reader);
        _upConv2.ReadParameters(reader);
        _upConv3.ReadParameters(reader);
        _probHead.ReadParameters(reader);
        _threshHead.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x44424E54); // "DBNT" in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_hiddenDim);
        writer.Write(_k);

        // Write component weights
        Backbone!.WriteParameters(writer);
        _inConv.WriteParameters(writer);
        _upConv1.WriteParameters(writer);
        _upConv2.WriteParameters(writer);
        _upConv3.WriteParameters(writer);
        _probHead.WriteParameters(writer);
        _threshHead.WriteParameters(writer);
    }

    private Tensor<T> ApplyReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(Math.Max(0, val));
        }
        return result;
    }

    private Tensor<T> ApplySigmoid(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
        }
        return result;
    }

    private Tensor<T> UpsampleAndConcat(Tensor<T> x, Tensor<T> skip)
    {
        int batch = x.Shape[0];
        int xChannels = x.Shape[1];
        int skipChannels = skip.Shape[1];
        int targetH = skip.Shape[2];
        int targetW = skip.Shape[3];

        var upsampled = BilinearUpsample(x, targetH, targetW);
        var result = new Tensor<T>(new[] { batch, xChannels + skipChannels, targetH, targetW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < xChannels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        result[b, c, h, w] = upsampled[b, c, h, w];
                    }
                }
            }

            for (int c = 0; c < skipChannels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        result[b, xChannels + c, h, w] = skip[b, c, h, w];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> BilinearUpsample(Tensor<T> x, int targetH, int targetW)
    {
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int srcH = x.Shape[2];
        int srcW = x.Shape[3];

        var result = new Tensor<T>(new[] { batch, channels, targetH, targetW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        double srcY = (double)h / targetH * srcH;
                        double srcX = (double)w / targetW * srcW;

                        int y0 = (int)Math.Floor(srcY);
                        int x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, srcH - 1);
                        int x1 = Math.Min(x0 + 1, srcW - 1);

                        double wy1 = srcY - y0;
                        double wy0 = 1.0 - wy1;
                        double wx1 = srcX - x0;
                        double wx0 = 1.0 - wx1;

                        double v00 = NumOps.ToDouble(x[b, c, y0, x0]);
                        double v01 = NumOps.ToDouble(x[b, c, y0, x1]);
                        double v10 = NumOps.ToDouble(x[b, c, y1, x0]);
                        double v11 = NumOps.ToDouble(x[b, c, y1, x1]);

                        double val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                        result[b, c, h, w] = NumOps.FromDouble(val);
                    }
                }
            }
        }

        return result;
    }

    private List<List<(int H, int W)>> FindConnectedComponents(bool[,] mask, int height, int width)
    {
        var components = new List<List<(int H, int W)>>();
        var visited = new bool[height, width];

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                if (mask[h, w] && !visited[h, w])
                {
                    var component = new List<(int H, int W)>();
                    FloodFill(mask, visited, h, w, height, width, component);
                    if (component.Count > 0)
                    {
                        components.Add(component);
                    }
                }
            }
        }

        return components;
    }

    private void FloodFill(
        bool[,] mask,
        bool[,] visited,
        int startH,
        int startW,
        int height,
        int width,
        List<(int H, int W)> component)
    {
        var stack = new Stack<(int H, int W)>();
        stack.Push((startH, startW));

        while (stack.Count > 0)
        {
            var (h, w) = stack.Pop();

            if (h < 0 || h >= height || w < 0 || w >= width)
                continue;

            if (visited[h, w] || !mask[h, w])
                continue;

            visited[h, w] = true;
            component.Add((h, w));

            stack.Push((h - 1, w));
            stack.Push((h + 1, w));
            stack.Push((h, w - 1));
            stack.Push((h, w + 1));
        }
    }

    private List<(int H, int W)> GetContour(
        List<(int H, int W)> component,
        bool[,] mask,
        int height,
        int width)
    {
        // Find boundary pixels (pixels with at least one non-text neighbor)
        var boundary = new HashSet<(int H, int W)>();

        foreach (var (h, w) in component)
        {
            bool isBoundary = false;

            // Check 4-connected neighbors
            if (h == 0 || !mask[h - 1, w]) isBoundary = true;
            if (h == height - 1 || !mask[h + 1, w]) isBoundary = true;
            if (w == 0 || !mask[h, w - 1]) isBoundary = true;
            if (w == width - 1 || !mask[h, w + 1]) isBoundary = true;

            if (isBoundary)
            {
                boundary.Add((h, w));
            }
        }

        // Order boundary points by angle from centroid
        double cx = boundary.Average(p => p.W);
        double cy = boundary.Average(p => p.H);

        return boundary
            .OrderBy(p => Math.Atan2(p.H - cy, p.W - cx))
            .ToList();
    }

    private List<TextRegion<T>> ApplyPolygonNMS(List<TextRegion<T>> regions, double iouThreshold)
    {
        if (regions.Count == 0)
            return regions;

        var sorted = regions.OrderByDescending(r => NumOps.ToDouble(r.Confidence)).ToList();
        var selected = new List<TextRegion<T>>();
        var used = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            if (used[i]) continue;

            selected.Add(sorted[i]);
            used[i] = true;

            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (used[j]) continue;

                double iou = ComputeBoxIoU(sorted[i].Box, sorted[j].Box);
                if (iou > iouThreshold)
                {
                    used[j] = true;
                }
            }
        }

        return selected;
    }

    private double ComputeBoxIoU(BoundingBox<T> a, BoundingBox<T> b)
    {
        double ax1 = NumOps.ToDouble(a.X1);
        double ay1 = NumOps.ToDouble(a.Y1);
        double ax2 = NumOps.ToDouble(a.X2);
        double ay2 = NumOps.ToDouble(a.Y2);

        double bx1 = NumOps.ToDouble(b.X1);
        double by1 = NumOps.ToDouble(b.Y1);
        double bx2 = NumOps.ToDouble(b.X2);
        double by2 = NumOps.ToDouble(b.Y2);

        double intersectX1 = Math.Max(ax1, bx1);
        double intersectY1 = Math.Max(ay1, by1);
        double intersectX2 = Math.Min(ax2, bx2);
        double intersectY2 = Math.Min(ay2, by2);

        double intersectW = Math.Max(0, intersectX2 - intersectX1);
        double intersectH = Math.Max(0, intersectY2 - intersectY1);
        double intersect = intersectW * intersectH;

        double areaA = (ax2 - ax1) * (ay2 - ay1);
        double areaB = (bx2 - bx1) * (by2 - by1);
        double union = areaA + areaB - intersect;

        return union > 0 ? intersect / union : 0;
    }
}
