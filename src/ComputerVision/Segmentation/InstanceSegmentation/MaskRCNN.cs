using AiDotNet.Tensors.Engines;
using System.IO;
using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Mask R-CNN", "https://arxiv.org/abs/1703.06870", Year = 2017, Authors = "Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick")]
public class MaskRCNN<T> : InstanceSegmenterBase<T>
{
    private readonly ResNet<T> _backbone;
    private readonly FPN<T> _fpn;
    private readonly RPN<T> _rpn;
    private readonly RoIAlign<T> _roiAlign;
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
        _backbone = new ResNet<T>(options: new ResNetBackboneOptions { Variant = ResNetVariant.ResNet50 });

        // Feature Pyramid Network
        _fpn = new FPN<T>(new[] { 256, 512, 1024, 2048 }, 256);

        // Region Proposal Network: a shared head with the standard 256-wide 3x3 conv, run on P2-P6
        // with one anchor size per level (32-512) and 3 aspect ratios per location (Lin et al. 2017).
        _rpn = new RPN<T>(256, 256);

        // RoIAlign (He et al. 2017): 7x7 bins, 2x2 bilinear samples per bin
        _roiAlign = new RoIAlign<T>(_roiPoolSize, samplingRatio: 2);

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

        var (fpnFeatures, proposalBoxes, _, _) = ProposeRegions(image, Options.MaxDetections);
        int numProposals = proposalBoxes.Shape[0];

        // Each RoI is pooled from the pyramid level matching its size, at that level's stride.
        var pooled = numProposals == 0
            ? null
            : FpnRoIPooler<T>.Pool(_roiAlign, fpnFeatures, _backbone.Strides, proposalBoxes);

        // RoI classification and mask prediction
        var instances = new List<InstanceMask<T>>();

        for (int p = 0; p < numProposals; p++)
        {
            var proposal = new BoundingBox<T>(
                proposalBoxes[p, 0], proposalBoxes[p, 1], proposalBoxes[p, 2], proposalBoxes[p, 3],
                BoundingBoxFormat.XYXY);
            var roiFeatures = CvTensorOps<T>.Select(pooled!, new[] { p }, 0);

            // Flatten for FC layers
            var flattened = Flatten(roiFeatures);

            // Box head
            var boxFeat = ApplyReLU(_boxHead.Forward(flattened));

            // Classification
            var classLogits = _classHead.Forward(boxFeat);
            var (classId, confidence) = GetPrediction(classLogits);

            // Skip background class
            if (classId == 0 || NumOps.LessThan(confidence, Options.ConfidenceThreshold))
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

    /// <summary>
    /// Runs the backbone, FPN and RPN, returning the pyramid features and the highest-scoring proposals.
    /// </summary>
    /// <param name="image">Input image [1, 3, H, W].</param>
    /// <param name="maxProposals">Maximum number of proposals to return, best first.</param>
    /// <returns>
    /// The FPN levels P2-P5, proposal boxes [N, 4] in input pixels (XYXY), every RPN anchor in level
    /// order P2-P6, and the number of anchors on each level.
    /// </returns>
    internal (List<Tensor<T>> fpnFeatures, Tensor<T> proposalBoxes, List<BoundingBox<T>> anchors, int[] levelAnchorCounts)
        ProposeRegions(Tensor<T> image, int maxProposals)
    {
        int imageHeight = image.Shape[2];
        int imageWidth = image.Shape[3];

        // Extract backbone features
        var backboneFeatures = _backbone.ExtractFeatures(image);

        // Apply FPN
        var fpnFeatures = _fpn.Forward(backboneFeatures);

        // The shared RPN head runs on every level P2-P5 plus P6 (P5 subsampled by 2), each level's
        // anchors laid out at its own stride; proposals are the top 1000 per level after NMS, best
        // 1000 overall (Mask R-CNN on FPN; detectron2, torchvision).
        var rpnLevels = new List<Tensor<T>>(fpnFeatures) { CvTensorOps<T>.MaxPoolPadded(fpnFeatures[^1], 1, 2, 0) };
        var (objectness, bboxDeltas, anchors, levelAnchorCounts) = _rpn.ForwardLevels(rpnLevels);
        var proposalSets = _rpn.GenerateProposals(
            objectness, bboxDeltas, anchors,
            imageHeight, imageWidth,
            preNmsTopK: 1000,
            postNmsTopK: 1000,
            nmsThreshold: 0.7,
            levelAnchorCounts: levelAnchorCounts);

        var proposalBoxes = proposalSets.Count == 0 ? new Tensor<T>(new[] { 0, 4 }) : proposalSets[0].boxes;
        if (proposalBoxes.Shape[0] > maxProposals)
        {
            proposalBoxes = CvTensorOps<T>.Select(proposalBoxes, Enumerable.Range(0, maxProposals).ToArray(), 0);
        }

        return (fpnFeatures, proposalBoxes, anchors, levelAnchorCounts);
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

    /// <summary>
    /// Elementwise ReLU, delegated to the engine.
    /// </summary>
    /// <remarks>
    /// This was a scalar loop that read each element out to <c>double</c> and wrote a fresh
    /// tensor. Arithmetically identical, but it severed the autodiff tape: the gradient chain
    /// stopped here, so every trainable layer UPSTREAM of this call received no gradient and
    /// silently never trained. The engine op records itself on the tape.
    /// </remarks>
    private Tensor<T> ApplyReLU(Tensor<T> input) => AiDotNetEngine.Current.ReLU(input);

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

        if (name != Name)
        {
            throw new InvalidOperationException(
                $"MaskRCNN configuration mismatch. Expected name={Name}, got name={name}");
        }

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
