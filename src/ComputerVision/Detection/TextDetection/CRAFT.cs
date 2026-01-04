using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>
/// CRAFT (Character Region Awareness for Text) detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CRAFT detects text by identifying individual characters
/// and the connections (affinity) between them. This approach works well for scene text
/// with arbitrary orientations and curved text.</para>
///
/// <para>Key features:
/// - Character region score map
/// - Affinity score map (character connections)
/// - Works with arbitrary shaped text
/// - Good for scene text detection
/// </para>
///
/// <para>Reference: Baek et al., "Character Region Awareness for Text Detection", CVPR 2019</para>
/// </remarks>
public class CRAFT<T> : TextDetectorBase<T>
{
    private readonly Conv2D<T> _upConv1;
    private readonly Conv2D<T> _upConv2;
    private readonly Conv2D<T> _upConv3;
    private readonly Conv2D<T> _upConv4;
    private readonly Conv2D<T> _regionHead;
    private readonly Conv2D<T> _affinityHead;
    private readonly int _hiddenDim;

    /// <inheritdoc/>
    public override string Name => $"CRAFT-{Options.Size}";

    /// <summary>
    /// Creates a new CRAFT text detector.
    /// </summary>
    public CRAFT(TextDetectionOptions<T> options) : base(options)
    {
        _hiddenDim = GetHiddenDim(options.Size);

        // VGG16-based backbone
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // Upsampling convolutions for feature fusion
        int backboneChannels = Backbone.OutputChannels[^1];
        _upConv1 = new Conv2D<T>(backboneChannels, _hiddenDim, kernelSize: 3, padding: 1);
        _upConv2 = new Conv2D<T>(_hiddenDim * 2, _hiddenDim, kernelSize: 3, padding: 1);
        _upConv3 = new Conv2D<T>(_hiddenDim * 2, _hiddenDim, kernelSize: 3, padding: 1);
        _upConv4 = new Conv2D<T>(_hiddenDim * 2, _hiddenDim, kernelSize: 3, padding: 1);

        // Prediction heads: region score and affinity score
        _regionHead = new Conv2D<T>(_hiddenDim, 1, kernelSize: 1);
        _affinityHead = new Conv2D<T>(_hiddenDim, 1, kernelSize: 1);
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

        // U-Net style upsampling with skip connections
        // Start from deepest features
        var x = _upConv1.Forward(features[^1]);
        x = ApplyReLU(x);

        // Upsample and concatenate with skip features
        if (features.Count > 1)
        {
            x = UpsampleAndConcat(x, features[^2]);
            x = _upConv2.Forward(x);
            x = ApplyReLU(x);
        }

        if (features.Count > 2)
        {
            x = UpsampleAndConcat(x, features[^3]);
            x = _upConv3.Forward(x);
            x = ApplyReLU(x);
        }

        if (features.Count > 3)
        {
            x = UpsampleAndConcat(x, features[^4]);
            x = _upConv4.Forward(x);
            x = ApplyReLU(x);
        }

        // Predict region and affinity scores
        var regionScore = _regionHead.Forward(x);
        regionScore = ApplySigmoid(regionScore);

        var affinityScore = _affinityHead.Forward(x);
        affinityScore = ApplySigmoid(affinityScore);

        return new List<Tensor<T>> { regionScore, affinityScore };
    }

    /// <inheritdoc/>
    protected override List<TextRegion<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold)
    {
        var regionScore = outputs[0];
        var affinityScore = outputs[1];

        int scoreH = regionScore.Shape[2];
        int scoreW = regionScore.Shape[3];

        // Scale factors from score map to original image
        double scaleX = (double)imageWidth / scoreW;
        double scaleY = (double)imageHeight / scoreH;

        // Find connected components in the combined score map
        var textMask = new bool[scoreH, scoreW];
        double threshold = NumOps.ToDouble(Options.BinaryThreshold);

        for (int h = 0; h < scoreH; h++)
        {
            for (int w = 0; w < scoreW; w++)
            {
                double region = NumOps.ToDouble(regionScore[0, 0, h, w]);
                double affinity = NumOps.ToDouble(affinityScore[0, 0, h, w]);

                // Text pixel if either region or affinity is high enough
                textMask[h, w] = region > threshold || affinity > threshold;
            }
        }

        // Find connected components
        var components = FindConnectedComponents(textMask, scoreH, scoreW);

        // Convert components to text regions
        var regions = new List<TextRegion<T>>();

        foreach (var component in components)
        {
            if (component.Count < 10) // Filter very small regions
                continue;

            // Get bounding box and confidence
            var (minX, minY, maxX, maxY, avgConfidence) = GetComponentStats(
                component, regionScore, scaleX, scaleY);

            if (avgConfidence < confidenceThreshold)
                continue;

            // Create polygon from component boundary
            var polygon = GetComponentBoundary(component, scaleX, scaleY);

            if (polygon.Count >= 4)
            {
                var region = TextRegion<T>.FromPolygon(
                    polygon.Select(p => (NumOps.FromDouble(p.X), NumOps.FromDouble(p.Y))).ToList(),
                    NumOps.FromDouble(avgConfidence));

                region.RegionType = TextRegionType.Word;
                regions.Add(region);
            }
        }

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

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        return _upConv1.GetParameterCount() +
               _upConv2.GetParameterCount() +
               _upConv3.GetParameterCount() +
               _upConv4.GetParameterCount() +
               _regionHead.GetParameterCount() +
               _affinityHead.GetParameterCount();
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
        if (magic != 0x43524146) // "CRAF" in ASCII
        {
            throw new InvalidDataException($"Invalid CRAFT model file. Expected magic 0x43524146, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported CRAFT model version: {version}");
        }

        string name = reader.ReadString();
        int hiddenDim = reader.ReadInt32();

        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidOperationException(
                $"CRAFT configuration mismatch. Expected hiddenDim={_hiddenDim}, got hiddenDim={hiddenDim}");
        }

        // Read component weights
        Backbone!.ReadParameters(reader);
        _upConv1.ReadParameters(reader);
        _upConv2.ReadParameters(reader);
        _upConv3.ReadParameters(reader);
        _upConv4.ReadParameters(reader);
        _regionHead.ReadParameters(reader);
        _affinityHead.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x43524146); // "CRAF" in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_hiddenDim);

        // Write component weights
        Backbone!.WriteParameters(writer);
        _upConv1.WriteParameters(writer);
        _upConv2.WriteParameters(writer);
        _upConv3.WriteParameters(writer);
        _upConv4.WriteParameters(writer);
        _regionHead.WriteParameters(writer);
        _affinityHead.WriteParameters(writer);
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

        // Upsample x to match skip spatial dimensions
        var upsampled = BilinearUpsample(x, targetH, targetW);

        // Concatenate along channel dimension
        var result = new Tensor<T>(new[] { batch, xChannels + skipChannels, targetH, targetW });

        for (int b = 0; b < batch; b++)
        {
            // Copy upsampled
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

            // Copy skip
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

            // 4-connected neighbors
            stack.Push((h - 1, w));
            stack.Push((h + 1, w));
            stack.Push((h, w - 1));
            stack.Push((h, w + 1));
        }
    }

    private (double minX, double minY, double maxX, double maxY, double avgConf) GetComponentStats(
        List<(int H, int W)> component,
        Tensor<T> regionScore,
        double scaleX,
        double scaleY)
    {
        double minX = double.MaxValue, minY = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue;
        double sumConf = 0;

        foreach (var (h, w) in component)
        {
            double x = w * scaleX;
            double y = h * scaleY;

            minX = Math.Min(minX, x);
            minY = Math.Min(minY, y);
            maxX = Math.Max(maxX, x);
            maxY = Math.Max(maxY, y);

            sumConf += NumOps.ToDouble(regionScore[0, 0, h, w]);
        }

        return (minX, minY, maxX, maxY, sumConf / component.Count);
    }

    private List<(double X, double Y)> GetComponentBoundary(
        List<(int H, int W)> component,
        double scaleX,
        double scaleY)
    {
        // Simple approach: get convex hull of component points
        var points = component.Select(p => (X: p.W * scaleX, Y: p.H * scaleY)).ToList();

        // Sort by angle from centroid
        double cx = points.Average(p => p.X);
        double cy = points.Average(p => p.Y);

        var boundary = points
            .OrderBy(p => Math.Atan2(p.Y - cy, p.X - cx))
            .ToList();

        // Simplify polygon
        return SimplifyPolygon(boundary, Options.PolygonSimplificationEpsilon * Math.Max(scaleX, scaleY));
    }
}
