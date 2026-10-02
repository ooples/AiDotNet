using AiDotNet.Tensors.Engines;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.PostProcessing;
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
public partial class MaskRCNN<T> : InstanceSegmenterBase<T>
{
    // He et al. 2017 with FPN (Lin et al. 2017), as in detectron2's R50-FPN configuration.
    private const int BoxRoiSize = 7;
    private const int RepresentationSize = 1024;
    private const int ProposalsPerImage = 1000;
    private const int FpnChannels = 256;

    private readonly ResNet<T> _backbone;
    private readonly FPN<T> _fpn;
    private readonly RPN<T> _rpn;
    private readonly RoIAlign<T> _boxRoiAlign;
    private readonly RoIAlign<T> _maskRoiAlign;
    private readonly Dense<T> _boxFc1;
    private readonly Dense<T> _boxFc2;
    private readonly Dense<T> _classHead;
    private readonly Dense<T> _boxRegressor;
    private readonly MaskHead<T> _maskHead;
    private readonly TwoStageDetectionLoss<T> _detectionLoss;
    private readonly NMS<T> _nms;

    [AiDotNet.Attributes.Scratch]
    private Random? _trainingRandom;

    /// <inheritdoc/>
    public override string Name => "MaskRCNN";

    /// <summary>
    /// Creates a new Mask R-CNN model.
    /// </summary>
    public MaskRCNN(InstanceSegmentationOptions<T> options) : base(options)
    {
        if (options.NumClasses <= 0) throw new ArgumentOutOfRangeException(nameof(options), "Mask R-CNN needs at least one foreground class.");

        _backbone = new ResNet<T>(options: new ResNetBackboneOptions { Variant = ResNetVariant.ResNet50 });
        _fpn = new FPN<T>(new[] { 256, 512, 1024, 2048 }, FpnChannels);

        // Region Proposal Network: a shared head with the standard 256-wide 3x3 conv, run on P2-P6
        // with one anchor size per level (32-512) and 3 aspect ratios per location (Lin et al. 2017).
        _rpn = new RPN<T>(FpnChannels, 256);

        // RoIAlign (He et al. 2017): 7x7 bins for the box branch, 14x14 for the mask branch, 2x2 samples per bin.
        _boxRoiAlign = new RoIAlign<T>(BoxRoiSize, samplingRatio: 2);
        _maskHead = new MaskHead<T>(FpnChannels, options.NumClasses, options.MaskResolution);
        _maskRoiAlign = new RoIAlign<T>(_maskHead.RoiSize, samplingRatio: 2);

        // Box branch: two 1024-wide fully connected layers, then sibling classification (K + 1, with
        // background) and class-specific box regression ((K + 1) x 4) layers.
        _boxFc1 = new Dense<T>(FpnChannels * BoxRoiSize * BoxRoiSize, RepresentationSize);
        _boxFc2 = new Dense<T>(RepresentationSize, RepresentationSize);
        _classHead = new Dense<T>(RepresentationSize, options.NumClasses + 1);
        _boxRegressor = new Dense<T>(RepresentationSize, (options.NumClasses + 1) * 4);

        _nms = new NMS<T>();
        _detectionLoss = new TwoStageDetectionLoss<T>(options.NumClasses, 1,
            options.TwoStageLoss ?? new TwoStageDetectionLossOptions());
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Heads, in order: class logits [proposals, K + 1], class-specific box deltas
    /// [proposals, (K + 1) x 4], proposal boxes [proposals, 4], RPN objectness and RPN box deltas.
    /// The mask branch runs on detections at inference and on sampled foreground RoIs in training.
    /// </remarks>
    protected override List<Tensor<T>> Forward(Tensor<T> input) => ForwardDetection(input, null, out _, out _);

    /// <inheritdoc/>
    protected override void MaterializeAuxiliaryHeads()
        => _maskHead.Forward(new Tensor<T>(new[] { 1, FpnChannels, _maskHead.RoiSize, _maskHead.RoiSize }));

    /// <inheritdoc/>
    public override InstanceSegmentationResult<T> Segment(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var startTime = DateTime.UtcNow;

        int imageHeight = image.Shape[2];
        int imageWidth = image.Shape[3];

        var heads = ForwardDetection(image, null, out var fpnFeatures, out _);
        var detections = DecodeDetections(heads, imageHeight, imageWidth);

        var instances = new List<InstanceMask<T>>();
        if (detections.Count > 0)
        {
            // As in the paper's inference: the mask branch runs on the top detection boxes, and each
            // detection takes the mask of its predicted class.
            var boxes = new Tensor<T>(new[] { detections.Count, 4 });
            for (int d = 0; d < detections.Count; d++)
            {
                boxes[d, 0] = detections[d].Box.X1;
                boxes[d, 1] = detections[d].Box.Y1;
                boxes[d, 2] = detections[d].Box.X2;
                boxes[d, 3] = detections[d].Box.Y2;
            }

            var maskLogits = _maskHead.Forward(FpnRoIPooler<T>.Pool(_maskRoiAlign, fpnFeatures, _backbone.Strides, boxes));
            int side = _maskHead.MaskResolution;
            int classes = Options.NumClasses;
            var probabilities = Engine.Sigmoid(Engine.Reshape(maskLogits, new[] { detections.Count * classes, side * side }));

            for (int d = 0; d < detections.Count; d++)
            {
                var detection = detections[d];
                var mask = new Tensor<T>(new[] { side, side });
                int row = d * classes + detection.ClassId;
                for (int i = 0; i < side * side; i++) mask[i / side, i % side] = probabilities[row, i];

                int boxWidth = (int)Math.Ceiling(NumOps.ToDouble(detection.Box.X2) - NumOps.ToDouble(detection.Box.X1));
                int boxHeight = (int)Math.Ceiling(NumOps.ToDouble(detection.Box.Y2) - NumOps.ToDouble(detection.Box.Y1));
                if (boxWidth <= 0 || boxHeight <= 0) continue;

                var fullMask = new Tensor<T>(new[] { imageHeight, imageWidth });
                PasteMask(fullMask, ResizeMask(mask, boxHeight, boxWidth), detection.Box);
                var instance = new InstanceMask<T>(detection.Box, BinarizeMask(fullMask, Options.MaskThreshold),
                    detection.ClassId, detection.Confidence);
                instances.Add(instance);
            }
        }

        return new InstanceSegmentationResult<T>
        {
            Instances = instances,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight
        };
    }

    /// <summary>
    /// Trains every branch with the Mask R-CNN multi-task loss L = L_cls + L_box + L_mask, plus the
    /// RPN's proposal loss (He et al. 2017, section 3).
    /// </summary>
    /// <param name="input">One model-ready image [1, 3, H, W].</param>
    /// <param name="targets">The image's objects; each mask is [H, W] over the input.</param>
    /// <remarks>
    /// <para>
    /// The RPN and box branch use the Faster R-CNN objectives (Ren et al. 2015; Girshick 2015): sampled
    /// anchors and RoIs, softmax classification and smooth-L1 class-specific box regression, with the
    /// object boxes added to the proposals. The mask loss is defined only on the sampled foreground
    /// RoIs: for a RoI matched to an object of class k, the average per-pixel binary cross-entropy of
    /// the k-th sigmoid mask against the object's mask resampled to the RoI at 28x28. Other classes'
    /// masks do not contribute, which decouples mask and class prediction.
    /// </para>
    /// <para>The RoI heads classify one image's proposals per forward pass, so each step takes one image.</para>
    /// </remarks>
    public void TrainInstances(Tensor<T> input, IReadOnlyList<InstanceSegmentationTrainingTarget<T>> targets)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (input.Rank != 4 || input.Shape[0] != 1 || input.Shape[1] != 3 || input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new ArgumentException("Mask R-CNN trains on one three-channel NCHW image [1, 3, H, W] per step.", nameof(input));
        int height = input.Shape[2];
        int width = input.Shape[3];
        foreach (var target in targets)
        {
            if (target is null) throw new ArgumentException("Targets cannot contain null entries.", nameof(targets));
            if (target.ClassId >= Options.NumClasses)
                throw new ArgumentException($"Class {target.ClassId} is outside the model's {Options.NumClasses} classes.", nameof(targets));
            if (target.Mask.Shape[0] != height || target.Mask.Shape[1] != width)
                throw new ArgumentException($"Each mask must be [{height}, {width}], the input's size.", nameof(targets));
        }

        var gold = targets.Select(target => TwoStageTargets.PixelCorners(target.Box, width, height)).ToList();
        var goldClasses = targets.Select(target => target.ClassId).ToList();
        var random = _trainingRandom ??= Options.RandomSeed is int seed
            ? RandomHelper.CreateSeededRandom(seed)
            : RandomHelper.CreateSecureRandom();

        List<Tensor<T>> fpnFeatures = new();
        List<BoundingBox<T>> anchors = new();
        TrainWithTargets(input, targets,
            image => ForwardDetection(image, gold, out fpnFeatures, out anchors),
            (heads, objects) =>
            {
                var proposalLoss = _detectionLoss.ComputeProposalLoss(
                    TwoStageTargets.FirstImage(heads[3]), TwoStageTargets.FirstImage(heads[4]),
                    anchors, gold, _rpn.AnchorsPerLocation, random);
                var boxLoss = _detectionLoss.ComputeStageLoss(heads[0], heads[1], heads[2], gold, goldClasses, 0, random,
                    out var foreground);
                var loss = Engine.TensorAdd(proposalLoss, boxLoss);
                return foreground.Count == 0
                    ? loss
                    : Engine.TensorAdd(loss, MaskLoss(fpnFeatures, heads[2], foreground, objects));
            });
    }

    /// <summary>
    /// L_mask: the mean binary cross-entropy of each foreground RoI's class-k mask logits against its
    /// object's mask resampled to the RoI, computed stably as softplus(x) - x * y.
    /// </summary>
    private Tensor<T> MaskLoss(List<Tensor<T>> fpnFeatures, Tensor<T> proposals,
        IReadOnlyList<(int Row, int Gold)> foreground, IReadOnlyList<InstanceSegmentationTrainingTarget<T>> objects)
    {
        int count = foreground.Count;
        int side = _maskHead.MaskResolution;
        int classes = Options.NumClasses;

        var rows = foreground.Select(f => f.Row).ToArray();
        var boxes = CvTensorOps<T>.Select(proposals, rows, 0);
        var logits = _maskHead.Forward(FpnRoIPooler<T>.Pool(_maskRoiAlign, fpnFeatures, _backbone.Strides, boxes));
        var perClass = Engine.Reshape(logits, new[] { count * classes, side * side });
        var picked = CvTensorOps<T>.Select(perClass,
            foreground.Select((f, i) => i * classes + objects[f.Gold].ClassId).ToArray(), 0);

        var targets = new Tensor<T>(new[] { count, side * side });
        for (int i = 0; i < count; i++)
        {
            var box = new[]
            {
                NumOps.ToDouble(boxes[i, 0]), NumOps.ToDouble(boxes[i, 1]),
                NumOps.ToDouble(boxes[i, 2]), NumOps.ToDouble(boxes[i, 3])
            };
            var resampled = ResampleMask(objects[foreground[i].Gold].Mask, box, side);
            for (int k = 0; k < resampled.Length; k++) targets[i, k] = NumOps.FromDouble(resampled[k]);
        }

        var crossEntropy = Engine.TensorSubtract(Engine.Softplus(picked), Engine.TensorMultiply(picked, targets));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(crossEntropy, null),
            NumOps.FromDouble(1.0 / (count * side * side)));
    }

    /// <summary>
    /// An object's mask resampled to a RoI at side x side: bilinear at each bin centre (pixel centres
    /// at +0.5), then binarized at 0.5, as detectron2 builds Mask R-CNN's mask targets.
    /// </summary>
    internal static double[] ResampleMask(Tensor<T> mask, double[] box, int side)
    {
        var ops = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int height = mask.Shape[0];
        int width = mask.Shape[1];
        double binWidth = (box[2] - box[0]) / side;
        double binHeight = (box[3] - box[1]) / side;
        var result = new double[side * side];
        for (int i = 0; i < side; i++)
        {
            double y = box[1] + (i + 0.5) * binHeight - 0.5;
            for (int j = 0; j < side; j++)
            {
                double x = box[0] + (j + 0.5) * binWidth - 0.5;
                result[i * side + j] = Bilinear(mask, ops, height, width, y, x) >= 0.5 ? 1.0 : 0.0;
            }
        }

        return result;
    }

    private static double Bilinear(Tensor<T> mask, INumericOperations<T> ops, int height, int width, double y, double x)
    {
        if (y < -1 || y > height || x < -1 || x > width) return 0;
        y = Math.Max(y, 0);
        x = Math.Max(x, 0);
        int y0 = Math.Min((int)Math.Floor(y), height - 1);
        int x0 = Math.Min((int)Math.Floor(x), width - 1);
        int y1 = Math.Min(y0 + 1, height - 1);
        int x1 = Math.Min(x0 + 1, width - 1);
        double wy = y0 >= height - 1 ? 0 : y - y0;
        double wx = x0 >= width - 1 ? 0 : x - x0;
        return (1 - wy) * ((1 - wx) * ops.ToDouble(mask[y0, x0]) + wx * ops.ToDouble(mask[y0, x1]))
             + wy * ((1 - wx) * ops.ToDouble(mask[y1, x0]) + wx * ops.ToDouble(mask[y1, x1]));
    }

    /// <summary>The detection forward, optionally adding boxes to the proposals the RoI heads classify.</summary>
    private List<Tensor<T>> ForwardDetection(Tensor<T> input, IReadOnlyList<double[]>? extraProposals,
        out List<Tensor<T>> fpnFeatures, out List<BoundingBox<T>> anchors)
    {
        var (features, proposalBoxes, objectness, rpnDeltas, levelAnchors, _) = ProposeRegionsCore(input, ProposalsPerImage);
        fpnFeatures = features;
        anchors = levelAnchors;
        if (extraProposals is { Count: > 0 })
            proposalBoxes = TwoStageTargets.AppendBoxes(proposalBoxes, extraProposals);

        int width = Options.NumClasses + 1;
        int count = proposalBoxes.Shape[0];
        if (count == 0)
        {
            return new List<Tensor<T>>
            {
                new Tensor<T>(new[] { 0, width }), new Tensor<T>(new[] { 0, width * 4 }), proposalBoxes, objectness, rpnDeltas
            };
        }

        // Each RoI is pooled from the pyramid level matching its size, at that level's stride.
        var pooled = FpnRoIPooler<T>.Pool(_boxRoiAlign, features, _backbone.Strides, proposalBoxes);
        var flattened = Engine.Reshape(pooled, new[] { count, pooled.Shape[1] * pooled.Shape[2] * pooled.Shape[3] });
        var representation = Engine.ReLU(_boxFc2.Forward(Engine.ReLU(_boxFc1.Forward(flattened))));
        return new List<Tensor<T>>
        {
            _classHead.Forward(representation), _boxRegressor.Forward(representation), proposalBoxes, objectness, rpnDeltas
        };
    }

    /// <summary>
    /// Box inference (Girshick 2015; He et al. 2017): softmax class scores, each class's own box
    /// deltas applied to the proposal, a score threshold, class-aware NMS, then the top detections.
    /// </summary>
    private List<Detection<T>> DecodeDetections(List<Tensor<T>> heads, int imageHeight, int imageWidth)
    {
        var classLogits = heads[0];
        var boxDeltas = heads[1];
        var proposals = heads[2];
        int count = classLogits.Shape[0];
        int width = Options.NumClasses + 1;
        double threshold = NumOps.ToDouble(Options.ConfidenceThreshold);

        var candidates = new List<Detection<T>>();
        for (int r = 0; r < count; r++)
        {
            double maxLogit = double.NegativeInfinity;
            for (int c = 0; c < width; c++) maxLogit = Math.Max(maxLogit, NumOps.ToDouble(classLogits[r, c]));
            var probabilities = new double[width];
            double sum = 0;
            for (int c = 0; c < width; c++)
            {
                probabilities[c] = Math.Exp(NumOps.ToDouble(classLogits[r, c]) - maxLogit);
                sum += probabilities[c];
            }

            double px1 = NumOps.ToDouble(proposals[r, 0]);
            double py1 = NumOps.ToDouble(proposals[r, 1]);
            double pw = NumOps.ToDouble(proposals[r, 2]) - px1;
            double ph = NumOps.ToDouble(proposals[r, 3]) - py1;
            if (pw <= 0 || ph <= 0) continue;

            for (int c = 1; c < width; c++)
            {
                double score = probabilities[c] / sum;
                if (score < threshold) continue;

                double cx = px1 + pw / 2 + NumOps.ToDouble(boxDeltas[r, c * 4]) * pw;
                double cy = py1 + ph / 2 + NumOps.ToDouble(boxDeltas[r, c * 4 + 1]) * ph;
                double w = pw * Math.Exp(Math.Min(NumOps.ToDouble(boxDeltas[r, c * 4 + 2]), BoxDeltaClamp));
                double h = ph * Math.Exp(Math.Min(NumOps.ToDouble(boxDeltas[r, c * 4 + 3]), BoxDeltaClamp));
                double x1 = Math.Max(0, cx - w / 2);
                double y1 = Math.Max(0, cy - h / 2);
                double x2 = Math.Min(imageWidth, cx + w / 2);
                double y2 = Math.Min(imageHeight, cy + h / 2);
                if (x2 <= x1 || y2 <= y1) continue;

                candidates.Add(new Detection<T>(
                    new BoundingBox<T>(NumOps.FromDouble(x1), NumOps.FromDouble(y1), NumOps.FromDouble(x2), NumOps.FromDouble(y2),
                        BoundingBoxFormat.XYXY),
                    c - 1, NumOps.FromDouble(score)));
            }
        }

        var kept = _nms.ApplyClassAware(candidates, NumOps.ToDouble(Options.NmsThreshold));
        return kept.OrderByDescending(d => NumOps.ToDouble(d.Confidence)).Take(Options.MaxDetections).ToList();
    }

    // log(1000 / 16), the clamp detectron2 and torchvision apply to dw and dh before exponentiating.
    private static readonly double BoxDeltaClamp = Math.Log(1000.0 / 16);

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
        var (features, proposals, _, _, anchors, counts) = ProposeRegionsCore(image, maxProposals);
        return (features, proposals, anchors, counts);
    }

    private (List<Tensor<T>> fpnFeatures, Tensor<T> proposalBoxes, Tensor<T> objectness, Tensor<T> rpnDeltas,
        List<BoundingBox<T>> anchors, int[] levelAnchorCounts) ProposeRegionsCore(Tensor<T> image, int maxProposals)
    {
        int imageHeight = image.Shape[2];
        int imageWidth = image.Shape[3];

        var fpnFeatures = _fpn.Forward(_backbone.ExtractFeatures(image));

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

        return (fpnFeatures, proposalBoxes, objectness, bboxDeltas, anchors, levelAnchorCounts);
    }

    private void PasteMask(Tensor<T> fullMask, Tensor<T> mask, BoundingBox<T> box)
    {
        int x1 = Math.Max(0, (int)NumOps.ToDouble(box.X1));
        int y1 = Math.Max(0, (int)NumOps.ToDouble(box.Y1));
        int x2 = Math.Min(fullMask.Shape[1], (int)Math.Ceiling(NumOps.ToDouble(box.X2)));
        int y2 = Math.Min(fullMask.Shape[0], (int)Math.Ceiling(NumOps.ToDouble(box.Y2)));

        int maskH = mask.Shape[0];
        int maskW = mask.Shape[1];
        int boxH = y2 - y1;
        int boxW = x2 - x1;
        if (boxH <= 0 || boxW <= 0) return;

        for (int y = y1; y < y2; y++)
        {
            for (int x = x1; x < x2; x++)
            {
                int my = Math.Min((int)((double)(y - y1) / boxH * maskH), maskH - 1);
                int mx = Math.Min((int)((double)(x - x1) / boxW * maskW), maskW - 1);
                fullMask[y, x] = mask[my, mx];
            }
        }
    }

    /// <inheritdoc/>
    public override long GetParameterCount() => ParameterCount;

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

        int magic = reader.ReadInt32();
        if (magic != 0x4D524E4E) // "MRNN" (Mask RCNN) in ASCII
        {
            throw new InvalidDataException($"Invalid MaskRCNN model file. Expected magic 0x4D524E4E, got 0x{magic:X8}");
        }

        // Version 2: the box branch gained its second 1024-wide layer and class-specific box regression,
        // the RPN its 256-wide hidden layer and the mask head its stride-2 transposed convolution.
        // Version 1 files describe a different network and cannot be loaded into this one.
        int version = reader.ReadInt32();
        if (version != 2)
        {
            throw new InvalidDataException($"Unsupported MaskRCNN model version: {version} (expected 2).");
        }

        string name = reader.ReadString();
        if (name != Name)
        {
            throw new InvalidOperationException($"MaskRCNN configuration mismatch. Expected name={Name}, got name={name}");
        }

        _backbone.ReadParameters(reader);
        _fpn.ReadParameters(reader);
        _rpn.ReadParameters(reader);
        _boxFc1.ReadParameters(reader);
        _boxFc2.ReadParameters(reader);
        _classHead.ReadParameters(reader);
        _boxRegressor.ReadParameters(reader);
        _maskHead.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write(0x4D524E4E); // "MRNN" (Mask RCNN) in ASCII
        writer.Write(2);
        writer.Write(Name);

        _backbone.WriteParameters(writer);
        _fpn.WriteParameters(writer);
        _rpn.WriteParameters(writer);
        _boxFc1.WriteParameters(writer);
        _boxFc2.WriteParameters(writer);
        _classHead.WriteParameters(writer);
        _boxRegressor.WriteParameters(writer);
        _maskHead.WriteParameters(writer);
    }
}
