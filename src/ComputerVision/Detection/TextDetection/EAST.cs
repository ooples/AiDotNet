using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Enums;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>
/// EAST (Efficient and Accurate Scene Text) detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> EAST is a fast and accurate text detector that directly
/// predicts word or text-line bounding boxes in one forward pass. It outputs a score map
/// showing where text is likely to be, plus geometry (box coordinates or rotated rectangles)
/// for each text region.</para>
///
/// <para>Key features:
/// - Single-shot detection (no region proposals)
/// - Supports both axis-aligned and rotated bounding boxes
/// - Fast inference suitable for real-time applications
/// - Works well for both horizontal and multi-oriented text
/// </para>
///
/// <para>Reference: Zhou et al., "EAST: An Efficient and Accurate Scene Text Detector", CVPR 2017</para>
/// </remarks>
public class EAST<T> : TextDetectorBase<T>
{
    private readonly Conv2D<T> _mergeConv1;
    private readonly Conv2D<T> _mergeConv2;
    private readonly Conv2D<T> _mergeConv3;
    private readonly Conv2D<T> _mergeConv4;
    private readonly Conv2D<T> _scoreHead;
    private readonly Conv2D<T> _geometryHead;
    private readonly int _hiddenDim;
    private readonly bool _useRotatedBoxes;

    /// <inheritdoc/>
    public override string Name => $"EAST-{Options.Size}";

    /// <summary>
    /// Creates a new EAST text detector.
    /// </summary>
    /// <param name="options">Text detection options.</param>
    /// <param name="useRotatedBoxes">Whether to predict rotated boxes (RBOX) or quadrilaterals (QUAD).</param>
    public EAST(TextDetectionOptions<T> options, bool useRotatedBoxes = true) : base(options)
    {
        _hiddenDim = GetHiddenDim(options.Size);
        _useRotatedBoxes = useRotatedBoxes;

        // ResNet backbone
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // Feature merging branch (U-Net style)
        int backboneChannels = Backbone.OutputChannels[^1];
        _mergeConv1 = new Conv2D<T>(backboneChannels, _hiddenDim, kernelSize: 1);
        _mergeConv2 = new Conv2D<T>(_hiddenDim * 2, _hiddenDim, kernelSize: 3, padding: 1);
        _mergeConv3 = new Conv2D<T>(_hiddenDim * 2, _hiddenDim, kernelSize: 3, padding: 1);
        _mergeConv4 = new Conv2D<T>(_hiddenDim, _hiddenDim / 2, kernelSize: 3, padding: 1);

        // Output heads
        _scoreHead = new Conv2D<T>(_hiddenDim / 2, 1, kernelSize: 1); // Text/non-text score

        // Geometry: 4 distances + 1 angle for RBOX, or 8 coordinates for QUAD
        int geometryChannels = useRotatedBoxes ? 5 : 8;
        _geometryHead = new Conv2D<T>(_hiddenDim / 2, geometryChannels, kernelSize: 1);
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

        // Feature merging (U-Net style)
        var x = _mergeConv1.Forward(features[^1]);
        x = ApplyBatchNormReLU(x);

        if (features.Count > 1)
        {
            x = UpsampleAndConcat(x, features[^2]);
            x = _mergeConv2.Forward(x);
            x = ApplyBatchNormReLU(x);
        }

        if (features.Count > 2)
        {
            x = UpsampleAndConcat(x, features[^3]);
            x = _mergeConv3.Forward(x);
            x = ApplyBatchNormReLU(x);
        }

        x = _mergeConv4.Forward(x);
        x = ApplyBatchNormReLU(x);

        // Predict score and geometry
        var score = _scoreHead.Forward(x);
        score = ApplySigmoid(score);

        var geometry = _geometryHead.Forward(x);

        return new List<Tensor<T>> { score, geometry };
    }

    /// <inheritdoc/>
    protected override List<TextRegion<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold)
    {
        var score = outputs[0];
        var geometry = outputs[1];

        int scoreH = score.Shape[2];
        int scoreW = score.Shape[3];

        double scaleX = (double)imageWidth / scoreW;
        double scaleY = (double)imageHeight / scoreH;

        var regions = new List<TextRegion<T>>();

        // Find text pixels and decode boxes
        for (int h = 0; h < scoreH; h++)
        {
            for (int w = 0; w < scoreW; w++)
            {
                double scoreVal = NumOps.ToDouble(score[0, 0, h, w]);

                if (scoreVal < confidenceThreshold)
                    continue;

                // Decode geometry
                var polygon = DecodeGeometry(geometry, h, w, scaleX, scaleY);

                if (polygon.Count >= 4)
                {
                    var region = TextRegion<T>.FromPolygon(
                        polygon.Select(p => (NumOps.FromDouble(p.X), NumOps.FromDouble(p.Y))).ToList(),
                        NumOps.FromDouble(scoreVal));

                    region.RegionType = TextRegionType.Word;

                    if (_useRotatedBoxes)
                    {
                        // Extract rotation angle
                        double angle = NumOps.ToDouble(geometry[0, 4, h, w]);
                        region.RotationAngle = angle * 180.0 / Math.PI;
                    }

                    regions.Add(region);
                }
            }
        }

        // Apply NMS to remove overlapping detections
        regions = ApplyTextNMS(regions, 0.2);

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

    private List<(double X, double Y)> DecodeGeometry(
        Tensor<T> geometry,
        int h,
        int w,
        double scaleX,
        double scaleY)
    {
        double centerX = (w + 0.5) * scaleX;
        double centerY = (h + 0.5) * scaleY;

        if (_useRotatedBoxes)
        {
            // RBOX format: 4 distances from pixel to box edges + angle
            double d0 = NumOps.ToDouble(geometry[0, 0, h, w]) * scaleY; // top
            double d1 = NumOps.ToDouble(geometry[0, 1, h, w]) * scaleX; // right
            double d2 = NumOps.ToDouble(geometry[0, 2, h, w]) * scaleY; // bottom
            double d3 = NumOps.ToDouble(geometry[0, 3, h, w]) * scaleX; // left
            double angle = NumOps.ToDouble(geometry[0, 4, h, w]);

            // Compute rotated rectangle corners
            double boxHeight = d0 + d2;
            double boxWidth = d1 + d3;

            double cos = Math.Cos(angle);
            double sin = Math.Sin(angle);

            // Box center offset from pixel
            double offsetX = (d1 - d3) / 2;
            double offsetY = (d2 - d0) / 2;

            double cx = centerX + offsetX * cos - offsetY * sin;
            double cy = centerY + offsetX * sin + offsetY * cos;

            // Compute 4 corners
            double hw = boxWidth / 2;
            double hh = boxHeight / 2;

            return new List<(double X, double Y)>
            {
                (cx - hw * cos + hh * sin, cy - hw * sin - hh * cos), // top-left
                (cx + hw * cos + hh * sin, cy + hw * sin - hh * cos), // top-right
                (cx + hw * cos - hh * sin, cy + hw * sin + hh * cos), // bottom-right
                (cx - hw * cos - hh * sin, cy - hw * sin + hh * cos)  // bottom-left
            };
        }
        else
        {
            // QUAD format: 8 offsets to 4 corners
            var points = new List<(double X, double Y)>();
            for (int i = 0; i < 4; i++)
            {
                double offsetX = NumOps.ToDouble(geometry[0, i * 2, h, w]) * scaleX;
                double offsetY = NumOps.ToDouble(geometry[0, i * 2 + 1, h, w]) * scaleY;
                points.Add((centerX + offsetX, centerY + offsetY));
            }
            return points;
        }
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        return _mergeConv1.GetParameterCount() +
               _mergeConv2.GetParameterCount() +
               _mergeConv3.GetParameterCount() +
               _mergeConv4.GetParameterCount() +
               _scoreHead.GetParameterCount() +
               _geometryHead.GetParameterCount();
    }

    /// <inheritdoc/>
    public override async Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        byte[] data;
        if (pathOrUrl.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            pathOrUrl.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            using var client = new System.Net.Http.HttpClient();
            data = await client.GetByteArrayAsync(pathOrUrl, cancellationToken);
        }
        else
        {
            data = await File.ReadAllBytesAsync(pathOrUrl, cancellationToken);
        }

        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and verify header
        int magic = reader.ReadInt32();
        if (magic != 0x45415354) // "EAST" in ASCII
        {
            throw new InvalidDataException($"Invalid EAST model file. Expected magic 0x45415354, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported EAST model version: {version}");
        }

        string name = reader.ReadString();
        bool useRotatedBoxes = reader.ReadBoolean();
        int hiddenDim = reader.ReadInt32();

        if (useRotatedBoxes != _useRotatedBoxes || hiddenDim != _hiddenDim)
        {
            throw new InvalidOperationException(
                $"EAST configuration mismatch. Expected useRotatedBoxes={_useRotatedBoxes}, hiddenDim={_hiddenDim}, " +
                $"got useRotatedBoxes={useRotatedBoxes}, hiddenDim={hiddenDim}");
        }

        // Read component weights
        Backbone!.ReadParameters(reader);
        _mergeConv1.ReadParameters(reader);
        _mergeConv2.ReadParameters(reader);
        _mergeConv3.ReadParameters(reader);
        _mergeConv4.ReadParameters(reader);
        _scoreHead.ReadParameters(reader);
        _geometryHead.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x45415354); // "EAST" in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_useRotatedBoxes);
        writer.Write(_hiddenDim);

        // Write component weights
        Backbone!.WriteParameters(writer);
        _mergeConv1.WriteParameters(writer);
        _mergeConv2.WriteParameters(writer);
        _mergeConv3.WriteParameters(writer);
        _mergeConv4.WriteParameters(writer);
        _scoreHead.WriteParameters(writer);
        _geometryHead.WriteParameters(writer);
    }

    private Tensor<T> ApplyBatchNormReLU(Tensor<T> x)
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

    private List<TextRegion<T>> ApplyTextNMS(List<TextRegion<T>> regions, double iouThreshold)
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
