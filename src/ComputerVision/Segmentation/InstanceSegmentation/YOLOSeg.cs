using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Extensions;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// YOLOv8-Seg for instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> YOLOv8-Seg extends YOLOv8 detection with a segmentation head.
/// It uses prototype masks and per-instance coefficients for efficient mask prediction.
/// This is much faster than Mask R-CNN while maintaining good quality.</para>
///
/// <para>Key features:
/// - Single-stage detection + segmentation
/// - Prototype-based mask assembly
/// - Shared backbone with detection head
/// - Real-time performance
/// </para>
/// </remarks>
public class YOLOSeg<T> : InstanceSegmenterBase<T>
{
    private readonly CSPDarknet<T> _backbone;
    private readonly PANet<T> _neck;
    private readonly YOLOHead<T> _detectionHead;
    private readonly PrototypeMaskHead<T> _protoHead;
    private readonly Dense<T> _coeffHead;
    private readonly int _numPrototypes;

    /// <inheritdoc/>
    public override string Name => "YOLOv8-Seg";

    /// <summary>
    /// Creates a new YOLOv8-Seg model.
    /// </summary>
    public YOLOSeg(InstanceSegmentationOptions<T> options) : base(options)
    {
        _numPrototypes = 32;

        // CSPDarknet backbone (depth=1.0, width=1.0)
        _backbone = new CSPDarknet<T>(depth: 1.0, widthMultiplier: 1.0);

        // PANet neck
        _neck = new PANet<T>(new[] { 64, 128, 256, 512 }, 256);

        // Detection head (YOLO-style) - takes int[] inputChannels, numClasses, numAnchors
        _detectionHead = new YOLOHead<T>(new[] { 256, 256, 256 }, options.NumClasses, 3);

        // Prototype head
        _protoHead = new PrototypeMaskHead<T>(256, _numPrototypes);

        // Coefficient prediction head (one set per detection)
        _coeffHead = new Dense<T>(256 * 5 * 5, _numPrototypes);
    }

    /// <inheritdoc/>
    public override InstanceSegmentationResult<T> Segment(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageHeight = image.Shape[2];
        int imageWidth = image.Shape[3];

        // Extract backbone features
        var backboneFeatures = _backbone.ExtractFeatures(image);

        // Apply neck
        var neckFeatures = _neck.Forward(backboneFeatures);

        // Generate prototypes from P3 (highest resolution)
        var prototypes = _protoHead.GeneratePrototypes(neckFeatures[0]);
        int protoH = prototypes.Shape[2];
        int protoW = prototypes.Shape[3];

        // Detection forward pass
        var (bboxes, scores, labels) = DetectObjects(neckFeatures, imageHeight, imageWidth);

        // Assemble instance masks
        var instances = new List<InstanceMask<T>>();

        for (int i = 0; i < bboxes.Count && i < Options.MaxDetections; i++)
        {
            var box = bboxes[i];
            int classId = labels[i];
            double confidence = scores[i];

            if (confidence < NumOps.ToDouble(Options.ConfidenceThreshold))
                continue;

            // Get RoI features for coefficient prediction
            var roiFeatures = ExtractRoIFeatures(neckFeatures[0], box);
            var flatFeatures = Flatten(roiFeatures);

            // Predict mask coefficients
            var coefficients = _coeffHead.Forward(flatFeatures);
            ApplySigmoid(coefficients);

            // Assemble mask from prototypes
            var coefVector = new Tensor<T>(new[] { _numPrototypes });
            for (int p = 0; p < _numPrototypes; p++)
            {
                coefVector[p] = coefficients[0, p];
            }

            var mask = _protoHead.AssembleMask(prototypes, coefVector);

            // Crop mask to bounding box and resize to image size
            var croppedMask = CropMaskToBoxProto(mask, box, protoH, protoW, imageHeight, imageWidth);
            var fullMask = new Tensor<T>(new[] { imageHeight, imageWidth });
            PasteMask(fullMask, croppedMask, box, imageHeight, imageWidth);

            // Binarize
            var binaryMask = BinarizeMask(fullMask, Options.MaskThreshold);

            instances.Add(new InstanceMask<T>(box, binaryMask, classId, NumOps.FromDouble(confidence)));
        }

        // Apply mask NMS
        instances = ApplyMaskNMS(instances, NumOps.ToDouble(Options.NmsThreshold));

        return new InstanceSegmentationResult<T>
        {
            Instances = instances,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight
        };
    }

    private (List<BoundingBox<T>> boxes, List<double> scores, List<int> labels) DetectObjects(
        List<Tensor<T>> neckFeatures, int imageHeight, int imageWidth)
    {
        var allBoxes = new List<(BoundingBox<T> box, double score, int label)>();

        int[] strides = { 8, 16, 32 };

        // Forward all features through detection head
        var allPreds = _detectionHead.Forward(neckFeatures);

        for (int level = 0; level < Math.Min(neckFeatures.Count, allPreds.Count); level++)
        {
            var feat = neckFeatures[level];
            var rawPreds = allPreds[level];
            int featH = feat.Shape[2];
            int featW = feat.Shape[3];
            int stride = strides[Math.Min(level, strides.Length - 1)];

            // Decode predictions
            for (int y = 0; y < featH; y++)
            {
                for (int x = 0; x < featW; x++)
                {
                    // Get objectness and class scores
                    double objectness = Sigmoid(GetTensorValue(rawPreds, 0, y, x));

                    if (objectness < 0.3)
                        continue;

                    // Find best class
                    int bestClass = 0;
                    double bestScore = 0;
                    for (int c = 0; c < Options.NumClasses; c++)
                    {
                        double classProb = Sigmoid(GetTensorValue(rawPreds, 5 + c, y, x));
                        double score = objectness * classProb;
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestClass = c;
                        }
                    }

                    if (bestScore < NumOps.ToDouble(Options.ConfidenceThreshold))
                        continue;

                    // Decode box
                    double cx = (x + Sigmoid(GetTensorValue(rawPreds, 1, y, x))) * stride;
                    double cy = (y + Sigmoid(GetTensorValue(rawPreds, 2, y, x))) * stride;
                    double w = Math.Exp(GetTensorValue(rawPreds, 3, y, x)) * stride;
                    double h = Math.Exp(GetTensorValue(rawPreds, 4, y, x)) * stride;

                    double x1 = MathHelper.Clamp(cx - w / 2, 0, imageWidth);
                    double y1 = MathHelper.Clamp(cy - h / 2, 0, imageHeight);
                    double x2 = MathHelper.Clamp(cx + w / 2, 0, imageWidth);
                    double y2 = MathHelper.Clamp(cy + h / 2, 0, imageHeight);

                    if (x2 > x1 && y2 > y1)
                    {
                        var box = new BoundingBox<T>(
                            NumOps.FromDouble(x1), NumOps.FromDouble(y1),
                            NumOps.FromDouble(x2), NumOps.FromDouble(y2),
                            BoundingBoxFormat.XYXY);

                        allBoxes.Add((box, bestScore, bestClass));
                    }
                }
            }
        }

        // NMS
        var nmsResult = ApplyNMS(allBoxes, NumOps.ToDouble(Options.NmsThreshold));

        var boxes = nmsResult.Select(r => r.box).ToList();
        var scores = nmsResult.Select(r => r.score).ToList();
        var labels = nmsResult.Select(r => r.label).ToList();

        return (boxes, scores, labels);
    }

    private List<(BoundingBox<T> box, double score, int label)> ApplyNMS(
        List<(BoundingBox<T> box, double score, int label)> detections, double iouThreshold)
    {
        var sorted = detections.OrderByDescending(d => d.score).ToList();
        var keep = new List<(BoundingBox<T> box, double score, int label)>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            keep.Add(best);
            sorted.RemoveAt(0);

            sorted = sorted.Where(d =>
            {
                if (d.label != best.label)
                    return true;

                double iou = ComputeBoxIoU(best.box, d.box);
                return iou < iouThreshold;
            }).ToList();
        }

        return keep;
    }

    private double ComputeBoxIoU(BoundingBox<T> a, BoundingBox<T> b)
    {
        double x1 = Math.Max(NumOps.ToDouble(a.X1), NumOps.ToDouble(b.X1));
        double y1 = Math.Max(NumOps.ToDouble(a.Y1), NumOps.ToDouble(b.Y1));
        double x2 = Math.Min(NumOps.ToDouble(a.X2), NumOps.ToDouble(b.X2));
        double y2 = Math.Min(NumOps.ToDouble(a.Y2), NumOps.ToDouble(b.Y2));

        double intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);

        double areaA = (NumOps.ToDouble(a.X2) - NumOps.ToDouble(a.X1)) *
                       (NumOps.ToDouble(a.Y2) - NumOps.ToDouble(a.Y1));
        double areaB = (NumOps.ToDouble(b.X2) - NumOps.ToDouble(b.X1)) *
                       (NumOps.ToDouble(b.Y2) - NumOps.ToDouble(b.Y1));

        double union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0;
    }

    private Tensor<T> ExtractRoIFeatures(Tensor<T> features, BoundingBox<T> box)
    {
        int channels = features.Shape[1];
        int featH = features.Shape[2];
        int featW = features.Shape[3];
        int poolSize = 5;

        var output = new Tensor<T>(new[] { 1, channels, poolSize, poolSize });

        // Map box to feature space
        double scaleH = (double)featH / Options.InputSize[0];
        double scaleW = (double)featW / Options.InputSize[1];

        double x1 = NumOps.ToDouble(box.X1) * scaleW;
        double y1 = NumOps.ToDouble(box.Y1) * scaleH;
        double x2 = NumOps.ToDouble(box.X2) * scaleW;
        double y2 = NumOps.ToDouble(box.Y2) * scaleH;

        double binH = (y2 - y1) / poolSize;
        double binW = (x2 - x1) / poolSize;

        for (int c = 0; c < channels; c++)
        {
            for (int ph = 0; ph < poolSize; ph++)
            {
                for (int pw = 0; pw < poolSize; pw++)
                {
                    double sampY = y1 + (ph + 0.5) * binH;
                    double sampX = x1 + (pw + 0.5) * binW;

                    int fy = MathHelper.Clamp((int)sampY, 0, featH - 1);
                    int fx = MathHelper.Clamp((int)sampX, 0, featW - 1);

                    output[0, c, ph, pw] = features[0, c, fy, fx];
                }
            }
        }

        return output;
    }

    private Tensor<T> Flatten(Tensor<T> input)
    {
        int total = input.Length;
        var output = new Tensor<T>(new[] { 1, total });

        for (int i = 0; i < total; i++)
        {
            output[0, i] = input[i];
        }

        return output;
    }

    private Tensor<T> CropMaskToBoxProto(Tensor<T> mask, BoundingBox<T> box,
        int protoH, int protoW, int imageH, int imageW)
    {
        // Map box to prototype space
        double scaleH = (double)protoH / imageH;
        double scaleW = (double)protoW / imageW;

        int x1 = Math.Max(0, (int)(NumOps.ToDouble(box.X1) * scaleW));
        int y1 = Math.Max(0, (int)(NumOps.ToDouble(box.Y1) * scaleH));
        int x2 = Math.Min(protoW, (int)(NumOps.ToDouble(box.X2) * scaleW));
        int y2 = Math.Min(protoH, (int)(NumOps.ToDouble(box.Y2) * scaleH));

        int cropW = Math.Max(1, x2 - x1);
        int cropH = Math.Max(1, y2 - y1);

        var cropped = new Tensor<T>(new[] { cropH, cropW });

        for (int h = 0; h < cropH; h++)
        {
            for (int w = 0; w < cropW; w++)
            {
                int srcH = y1 + h;
                int srcW = x1 + w;

                if (srcH < mask.Shape[0] && srcW < mask.Shape[1])
                {
                    cropped[h, w] = mask[srcH, srcW];
                }
            }
        }

        return cropped;
    }

    private void PasteMask(Tensor<T> fullMask, Tensor<T> croppedMask, BoundingBox<T> box,
        int imageH, int imageW)
    {
        int x1 = Math.Max(0, (int)NumOps.ToDouble(box.X1));
        int y1 = Math.Max(0, (int)NumOps.ToDouble(box.Y1));
        int x2 = Math.Min(imageW, (int)NumOps.ToDouble(box.X2));
        int y2 = Math.Min(imageH, (int)NumOps.ToDouble(box.Y2));

        int boxW = x2 - x1;
        int boxH = y2 - y1;

        if (boxW <= 0 || boxH <= 0)
            return;

        // Resize cropped mask to box size
        var resized = ResizeMask(croppedMask, boxH, boxW);

        for (int h = 0; h < boxH; h++)
        {
            for (int w = 0; w < boxW; w++)
            {
                fullMask[y1 + h, x1 + w] = resized[h, w];
            }
        }
    }

    private double GetTensorValue(Tensor<T> tensor, int channel, int y, int x)
    {
        // Handle different tensor shapes
        if (tensor.Shape.Length == 4)
        {
            // [batch, channels, height, width]
            if (channel < tensor.Shape[1] && y < tensor.Shape[2] && x < tensor.Shape[3])
            {
                return NumOps.ToDouble(tensor[0, channel, y, x]);
            }
        }
        else if (tensor.Shape.Length == 3)
        {
            // [channels, height, width]
            if (channel < tensor.Shape[0] && y < tensor.Shape[1] && x < tensor.Shape[2])
            {
                return NumOps.ToDouble(tensor[channel, y, x]);
            }
        }
        return 0;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        return _backbone.GetParameterCount() +
               _neck.GetParameterCount() +
               _detectionHead.GetParameterCount() +
               _protoHead.GetParameterCount() +
               _coeffHead.GetParameterCount();
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
        if (magic != 0x594F5347) // "YOSG" in ASCII
        {
            throw new InvalidDataException($"Invalid YOLOSeg model file. Expected magic 0x594F5347, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported YOLOSeg model version: {version}");
        }

        string name = reader.ReadString();
        int numPrototypes = reader.ReadInt32();

        if (numPrototypes != _numPrototypes)
        {
            throw new InvalidOperationException(
                $"YOLOSeg configuration mismatch. Expected numPrototypes={_numPrototypes}, got {numPrototypes}");
        }

        // Read component weights
        _backbone.ReadParameters(reader);
        _neck.ReadParameters(reader);
        _detectionHead.ReadParameters(reader);
        _protoHead.ReadParameters(reader);
        _coeffHead.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x594F5347); // "YOSG" in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_numPrototypes);

        // Write component weights
        _backbone.WriteParameters(writer);
        _neck.WriteParameters(writer);
        _detectionHead.WriteParameters(writer);
        _protoHead.WriteParameters(writer);
        _coeffHead.WriteParameters(writer);
    }
}
