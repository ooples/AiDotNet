using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// Mask R-CNN for instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Mask R-CNN extends Faster R-CNN by adding a mask
/// prediction branch parallel to the box classification and regression branches.
/// It's a two-stage detector that first proposes regions, then classifies them
/// and predicts masks.</para>
///
/// <para>Key features:
/// - Two-stage detection with RPN and RoI heads
/// - Parallel mask prediction branch
/// - RoIAlign for precise spatial alignment
/// - Decoupled mask and class prediction
/// </para>
///
/// <para>Reference: He et al., "Mask R-CNN", ICCV 2017</para>
/// </remarks>
public class MaskRCNN<T> : InstanceSegmenterBase<T>
{
    private readonly ResNet<T> _backbone;
    private readonly FPN<T> _fpn;
    private readonly RPN<T> _rpn;
    private readonly Dense<T> _boxHead;
    private readonly Dense<T> _classHead;
    private readonly MaskHead<T> _maskHead;
    private readonly int _roiPoolSize;

    /// <inheritdoc/>
    public override string Name => "MaskRCNN";

    /// <summary>
    /// Creates a new Mask R-CNN model.
    /// </summary>
    public MaskRCNN(InstanceSegmentationOptions<T> options) : base(options)
    {
        _roiPoolSize = 7;

        // Backbone
        _backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // Feature Pyramid Network
        _fpn = new FPN<T>(new[] { 256, 512, 1024, 2048 }, 256);

        // Region Proposal Network
        _rpn = new RPN<T>(256, 9); // 256 channels, 9 anchors per location

        // Box head (2 FC layers)
        int roiFeatureDim = 256 * _roiPoolSize * _roiPoolSize;
        _boxHead = new Dense<T>(roiFeatureDim, 1024);

        // Classification head
        _classHead = new Dense<T>(1024, options.NumClasses + 1); // +1 for background

        // Mask head
        _maskHead = new MaskHead<T>(256, options.NumClasses, options.MaskResolution);
    }

    /// <inheritdoc/>
    public override InstanceSegmentationResult<T> Segment(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageHeight = image.Shape[2];
        int imageWidth = image.Shape[3];

        // Extract backbone features
        var backboneFeatures = _backbone.ExtractFeatures(image);

        // Apply FPN
        var fpnFeatures = _fpn.Forward(backboneFeatures);

        // Generate proposals with RPN
        var p3Features = fpnFeatures[0]; // Use P3 level for RPN
        var (objectness, bboxDeltas, anchors) = _rpn.Forward(p3Features);

        // Decode proposals
        var proposals = DecodeProposals(objectness, bboxDeltas, anchors, imageHeight, imageWidth);

        // RoI pooling and classification
        var instances = new List<InstanceMask<T>>();

        foreach (var proposal in proposals.Take(Options.MaxDetections))
        {
            // Select appropriate FPN level based on proposal size (standard FPN assignment)
            var (fpnLevel, stride) = SelectFPNLevel(proposal, fpnFeatures.Count);
            var roiFeatures = RoIAlign(fpnFeatures[fpnLevel], proposal, _roiPoolSize, _roiPoolSize, stride);

            // Flatten for FC layers
            var flattened = Flatten(roiFeatures);

            // Box head
            var boxFeat = ApplyReLU(_boxHead.Forward(flattened));

            // Classification
            var classLogits = _classHead.Forward(boxFeat);
            var (classId, confidence) = GetPrediction(classLogits);

            // Skip background class
            if (classId == 0 || NumOps.ToDouble(confidence) < NumOps.ToDouble(Options.ConfidenceThreshold))
                continue;

            // Predict mask for this class
            var mask = _maskHead.PredictMask(roiFeatures, classId - 1); // Subtract 1 for background offset

            // Resize mask to full image size
            int boxWidth = (int)(NumOps.ToDouble(proposal.X2) - NumOps.ToDouble(proposal.X1));
            int boxHeight = (int)(NumOps.ToDouble(proposal.Y2) - NumOps.ToDouble(proposal.Y1));

            // Skip degenerate boxes
            if (boxWidth <= 0 || boxHeight <= 0)
                continue;

            var resizedMask = ResizeMask(mask, boxHeight, boxWidth);

            // Place mask in full image
            var fullMask = new Tensor<T>(new[] { imageHeight, imageWidth });
            PasteMask(fullMask, resizedMask, proposal);

            // Binarize mask
            var binaryMask = BinarizeMask(fullMask, Options.MaskThreshold);

            instances.Add(new InstanceMask<T>(proposal, binaryMask, classId - 1, confidence));
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

    private List<BoundingBox<T>> DecodeProposals(
        Tensor<T> objectness, Tensor<T> bboxDeltas, List<BoundingBox<T>> anchors,
        int imageHeight, int imageWidth)
    {
        var proposals = new List<(BoundingBox<T> box, double score)>();

        int numAnchors = anchors.Count;

        // Use actual feature map dimensions from objectness tensor
        int featureH = objectness.Shape[2];
        int featureW = objectness.Shape[3];

        // Guard against zero feature dimensions
        if (featureH <= 0 || featureW <= 0)
        {
            return new List<BoundingBox<T>>();
        }

        for (int i = 0; i < numAnchors && i < objectness.Length / 2; i++)
        {
            // Use actual feature map dimensions for indexing
            int h = i / featureW;
            int w = i % featureW;

            // Guard against out of bounds access
            if (h >= featureH || w >= featureW)
                continue;

            double score = NumOps.ToDouble(objectness[0, 1, h, w]);

            if (score > 0.3) // Pre-NMS threshold
            {
                var anchor = anchors[i % anchors.Count];

                // Apply deltas
                double anchorX = (NumOps.ToDouble(anchor.X1) + NumOps.ToDouble(anchor.X2)) / 2;
                double anchorY = (NumOps.ToDouble(anchor.Y1) + NumOps.ToDouble(anchor.Y2)) / 2;
                double anchorW = NumOps.ToDouble(anchor.X2) - NumOps.ToDouble(anchor.X1);
                double anchorH = NumOps.ToDouble(anchor.Y2) - NumOps.ToDouble(anchor.Y1);

                // Decode using standard box encoding
                double dx = 0, dy = 0, dw = 0, dh = 0;
                int idx = i * 4;
                if (idx + 3 < bboxDeltas.Length)
                {
                    dx = NumOps.ToDouble(bboxDeltas[idx]);
                    dy = NumOps.ToDouble(bboxDeltas[idx + 1]);
                    dw = NumOps.ToDouble(bboxDeltas[idx + 2]);
                    dh = NumOps.ToDouble(bboxDeltas[idx + 3]);
                }

                double predX = anchorX + dx * anchorW;
                double predY = anchorY + dy * anchorH;
                double predW = anchorW * Math.Exp(MathHelper.Clamp(dw, -4, 4));
                double predH = anchorH * Math.Exp(MathHelper.Clamp(dh, -4, 4));

                // Convert to corner format and clip
                double x1 = MathHelper.Clamp(predX - predW / 2, 0, imageWidth);
                double y1 = MathHelper.Clamp(predY - predH / 2, 0, imageHeight);
                double x2 = MathHelper.Clamp(predX + predW / 2, 0, imageWidth);
                double y2 = MathHelper.Clamp(predY + predH / 2, 0, imageHeight);

                if (x2 > x1 && y2 > y1)
                {
                    var box = new BoundingBox<T>(
                        NumOps.FromDouble(x1), NumOps.FromDouble(y1),
                        NumOps.FromDouble(x2), NumOps.FromDouble(y2),
                        BoundingBoxFormat.XYXY);

                    proposals.Add((box, score));
                }
            }
        }

        // Sort by score and take top proposals
        return proposals.OrderByDescending(p => p.score)
            .Take(1000)
            .Select(p => p.box)
            .ToList();
    }

    private Tensor<T> RoIAlign(Tensor<T> features, BoundingBox<T> box, int outputH, int outputW, double stride = 8.0)
    {
        int batch = 1;
        int channels = features.Shape[1];
        int featureH = features.Shape[2];
        int featureW = features.Shape[3];

        var output = new Tensor<T>(new[] { batch, channels, outputH, outputW });

        // Map box to feature space using the provided stride (P3=8, P4=16, P5=32)
        double x1 = NumOps.ToDouble(box.X1) / stride;
        double y1 = NumOps.ToDouble(box.Y1) / stride;
        double x2 = NumOps.ToDouble(box.X2) / stride;
        double y2 = NumOps.ToDouble(box.Y2) / stride;

        // Guard against zero or negative box dimensions
        double boxH = y2 - y1;
        double boxW = x2 - x1;
        if (boxH <= 0 || boxW <= 0)
        {
            // Return zero-filled tensor for degenerate boxes
            return output;
        }

        double binH = boxH / outputH;
        double binW = boxW / outputW;

        for (int c = 0; c < channels; c++)
        {
            for (int oh = 0; oh < outputH; oh++)
            {
                for (int ow = 0; ow < outputW; ow++)
                {
                    // Sample 4 points in each bin
                    double sumVal = 0;
                    int numSamples = 0;

                    for (int sy = 0; sy < 2; sy++)
                    {
                        for (int sx = 0; sx < 2; sx++)
                        {
                            double sampY = y1 + (oh + 0.25 + 0.5 * sy) * binH;
                            double sampX = x1 + (ow + 0.25 + 0.5 * sx) * binW;

                            if (sampY >= 0 && sampY < featureH && sampX >= 0 && sampX < featureW)
                            {
                                // Bilinear interpolation
                                int fy0 = (int)Math.Floor(sampY);
                                int fx0 = (int)Math.Floor(sampX);
                                int fy1 = Math.Min(fy0 + 1, featureH - 1);
                                int fx1 = Math.Min(fx0 + 1, featureW - 1);

                                double wy1 = sampY - fy0;
                                double wy0 = 1 - wy1;
                                double wx1 = sampX - fx0;
                                double wx0 = 1 - wx1;

                                double v00 = NumOps.ToDouble(features[0, c, fy0, fx0]);
                                double v01 = NumOps.ToDouble(features[0, c, fy0, fx1]);
                                double v10 = NumOps.ToDouble(features[0, c, fy1, fx0]);
                                double v11 = NumOps.ToDouble(features[0, c, fy1, fx1]);

                                sumVal += wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                                numSamples++;
                            }
                        }
                    }

                    output[0, c, oh, ow] = NumOps.FromDouble(numSamples > 0 ? sumVal / numSamples : 0);
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Selects the appropriate FPN level and stride based on proposal size.
    /// Uses the standard FPN level assignment formula: k = floor(k0 + log2(sqrt(area) / 224))
    /// </summary>
    private (int level, double stride) SelectFPNLevel(BoundingBox<T> proposal, int numLevels)
    {
        double w = NumOps.ToDouble(proposal.X2) - NumOps.ToDouble(proposal.X1);
        double h = NumOps.ToDouble(proposal.Y2) - NumOps.ToDouble(proposal.Y1);
        double area = Math.Max(1, w * h);

        // Standard FPN assignment: k0=4 corresponds to P4 (stride=16) for 224x224 proposals
        // k = floor(4 + log2(sqrt(area) / 224))
        const int k0 = 4;
        const double canonicalSize = 224.0;
        double k = k0 + MathHelper.Log2(Math.Sqrt(area) / canonicalSize);
        int level = (int)Math.Floor(k);

        // Clamp to valid FPN levels (P3=0, P4=1, P5=2, P6=3, etc.)
        // Map from P-levels to array indices: P3->0, P4->1, P5->2
        int fpnLevel = MathHelper.Clamp(level - 3, 0, numLevels - 1);

        // Strides for each FPN level: P3=8, P4=16, P5=32, P6=64
        double[] strides = { 8.0, 16.0, 32.0, 64.0 };
        double stride = strides[Math.Min(fpnLevel, strides.Length - 1)];

        return (fpnLevel, stride);
    }

    private Tensor<T> Flatten(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int total = input.Length / batch;

        var output = new Tensor<T>(new[] { batch, total });

        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            for (int i = b * total; i < (b + 1) * total; i++)
            {
                output[b, idx++] = input[i];
            }
        }

        return output;
    }

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            output[i] = NumOps.FromDouble(Math.Max(0, val));
        }

        return output;
    }

    private (int classId, T confidence) GetPrediction(Tensor<T> logits)
    {
        // Apply softmax and find argmax
        int numClasses = logits.Shape[1];
        double maxLogit = double.NegativeInfinity;
        int maxIdx = 0;

        for (int c = 0; c < numClasses; c++)
        {
            double val = NumOps.ToDouble(logits[0, c]);
            if (val > maxLogit)
            {
                maxLogit = val;
                maxIdx = c;
            }
        }

        // Compute softmax probability
        double sumExp = 0;
        for (int c = 0; c < numClasses; c++)
        {
            sumExp += Math.Exp(NumOps.ToDouble(logits[0, c]) - maxLogit);
        }

        double confidence = Math.Exp(0) / sumExp; // exp(maxLogit - maxLogit) = 1

        return (maxIdx, NumOps.FromDouble(confidence));
    }

    private void PasteMask(Tensor<T> fullMask, Tensor<T> mask, BoundingBox<T> box)
    {
        int x1 = Math.Max(0, (int)NumOps.ToDouble(box.X1));
        int y1 = Math.Max(0, (int)NumOps.ToDouble(box.Y1));
        int x2 = Math.Min(fullMask.Shape[1], (int)NumOps.ToDouble(box.X2));
        int y2 = Math.Min(fullMask.Shape[0], (int)NumOps.ToDouble(box.Y2));

        int maskH = mask.Shape[0];
        int maskW = mask.Shape[1];

        // Guard against degenerate boxes (zero height or width)
        int boxH = y2 - y1;
        int boxW = x2 - x1;
        if (boxH <= 0 || boxW <= 0)
        {
            return;
        }

        for (int y = y1; y < y2; y++)
        {
            for (int x = x1; x < x2; x++)
            {
                int my = (int)((double)(y - y1) / boxH * maskH);
                int mx = (int)((double)(x - x1) / boxW * maskW);

                my = Math.Min(my, maskH - 1);
                mx = Math.Min(mx, maskW - 1);

                fullMask[y, x] = mask[my, mx];
            }
        }
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        return _backbone.GetParameterCount() +
               _fpn.GetParameterCount() +
               _rpn.GetParameterCount() +
               _boxHead.GetParameterCount() +
               _classHead.GetParameterCount() +
               _maskHead.GetParameterCount();
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
        if (magic != 0x4D524E4E) // "MRNN" (Mask RCNN) in ASCII
        {
            throw new InvalidDataException($"Invalid MaskRCNN model file. Expected magic 0x4D524E4E, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported MaskRCNN model version: {version}");
        }

        string name = reader.ReadString();
        int roiPoolSize = reader.ReadInt32();

        if (roiPoolSize != _roiPoolSize)
        {
            throw new InvalidOperationException(
                $"MaskRCNN configuration mismatch. Expected roiPoolSize={_roiPoolSize}, got {roiPoolSize}");
        }

        // Read component weights
        _backbone.ReadParameters(reader);
        _fpn.ReadParameters(reader);
        _rpn.ReadParameters(reader);
        _boxHead.ReadParameters(reader);
        _classHead.ReadParameters(reader);
        _maskHead.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x4D524E4E); // "MRNN" (Mask RCNN) in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_roiPoolSize);

        // Write component weights
        _backbone.WriteParameters(writer);
        _fpn.WriteParameters(writer);
        _rpn.WriteParameters(writer);
        _boxHead.WriteParameters(writer);
        _classHead.WriteParameters(writer);
        _maskHead.WriteParameters(writer);
    }
}
