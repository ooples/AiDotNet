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
            var boxes = CvTensorOps<T>.FromRows(
                detections.Select(d => new[] { d.Box.X1, d.Box.Y1, d.Box.X2, d.Box.Y2 }).ToArray(), 4);
            var boxCoordinates = CvTensorOps<T>.ToDoubleArray(boxes);

            var maskLogits = _maskHead.Forward(FpnRoIPooler<T>.Pool(_maskRoiAlign, fpnFeatures, _backbone.Strides, boxes));
            int side = _maskHead.MaskResolution;
            int classes = Options.NumClasses;
            var probabilities = Engine.Sigmoid(Engine.Reshape(maskLogits, new[] { detections.Count * classes, side * side }));

            for (int d = 0; d < detections.Count; d++)
            {
                var detection = detections[d];
                int row = d * classes + detection.ClassId;
                var mask = Engine.Reshape(Engine.TensorSlice(probabilities, new[] { row, 0 }, new[] { 1, side * side }), new[] { side, side });

                double bx1 = boxCoordinates[4 * d], by1 = boxCoordinates[4 * d + 1];
                double bx2 = boxCoordinates[4 * d + 2], by2 = boxCoordinates[4 * d + 3];
                int boxWidth = (int)Math.Ceiling(bx2 - bx1);
                int boxHeight = (int)Math.Ceiling(by2 - by1);
                if (boxWidth <= 0 || boxHeight <= 0) continue;

                var fullMask = PasteMask(imageHeight, imageWidth, ResizeMask(mask, boxHeight, boxWidth), bx1, by1, bx2, by2);
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
            image => ForwardDetection(image, gold, out fpnFeatures, out anchors, random),
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

        var boxCoordinates = CvTensorOps<T>.ToDoubleArray(boxes);
        var targetRows = new Tensor<T>[count];
        for (int i = 0; i < count; i++)
        {
            var box = new[] { boxCoordinates[4 * i], boxCoordinates[4 * i + 1], boxCoordinates[4 * i + 2], boxCoordinates[4 * i + 3] };
            targetRows[i] = Engine.Reshape(ResampleMaskTensor(objects[foreground[i].Gold].Mask, box, side), new[] { 1, side * side });
        }

        var targets = Engine.Concat(targetRows, 0);

        var crossEntropy = Engine.TensorSubtract(Engine.Softplus(picked), Engine.TensorMultiply(picked, targets));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(crossEntropy, null),
            NumOps.FromDouble(1.0 / ((double)count * side * side)));
    }

    /// <summary>
    /// An object's mask resampled to a RoI at side x side: bilinear at each bin centre (pixel centres
    /// at +0.5), then binarized at 0.5, as detectron2 builds Mask R-CNN's mask targets.
    /// </summary>
    internal static double[] ResampleMask(Tensor<T> mask, double[] box, int side)
        => CvTensorOps<T>.ToDoubleArray(ResampleMaskTensor(mask, box, side));

    /// <summary>
    /// <see cref="ResampleMask"/> as a <c>[side, side]</c> tensor of <typeparamref name="T"/>. The sample grid is
    /// separable - row i sits at <c>y1 + (i + 0.5) * binHeight - 0.5</c>, column j at
    /// <c>x1 + (j + 0.5) * binWidth - 0.5</c> - so the bilinear sample is two one-axis interpolations, each
    /// clamping as torchvision's roi_align does: a coordinate below -1 or past the extent samples zero, one
    /// in (-1, 0] clamps to the first cell, and the last cell has no far neighbour.
    /// </summary>
    internal static Tensor<T> ResampleMaskTensor(Tensor<T> mask, double[] box, int side)
    {
        var engine = AiDotNetEngine.Current;
        var ops = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int height = mask.Shape[0];
        int width = mask.Shape[1];
        var (y0, y1, wy, validY) = AxisTaps(engine, ops, box[1], (box[3] - box[1]) / side, side, height);
        var (x0, x1, wx, validX) = AxisTaps(engine, ops, box[0], (box[2] - box[0]) / side, side, width);

        // Along x: [height, side]; then along y: [side, side].
        var wxRow = engine.TensorBroadcastTo(engine.Reshape(wx, new[] { 1, side }), new[] { height, side });
        var left = CvTensorOps<T>.Select(mask, x0, 1);
        var right = CvTensorOps<T>.Select(mask, x1, 1);
        var alongX = engine.TensorAdd(
            engine.TensorMultiply(engine.ScalarMinusTensor(ops.One, wxRow), left), engine.TensorMultiply(wxRow, right));
        var wyCol = engine.TensorBroadcastTo(engine.Reshape(wy, new[] { side, 1 }), new[] { side, side });
        var top = CvTensorOps<T>.Select(alongX, y0, 0);
        var bottom = CvTensorOps<T>.Select(alongX, y1, 0);
        var sampled = engine.TensorAdd(
            engine.TensorMultiply(engine.ScalarMinusTensor(ops.One, wyCol), top), engine.TensorMultiply(wyCol, bottom));

        // Out-of-range rows or columns sample zero; the target is the sample binarized at 0.5.
        var valid = engine.TensorMultiply(
            engine.TensorBroadcastTo(engine.Reshape(validY, new[] { side, 1 }), new[] { side, side }),
            engine.TensorBroadcastTo(engine.Reshape(validX, new[] { 1, side }), new[] { side, side }));
        var binary = CvTensorOps<T>.GreaterOrEqual(sampled, ops.FromDouble(0.5));
        return engine.TensorMultiply(binary, valid);
    }

    /// <summary>
    /// One axis of the RoI sample grid: for sample k at <c>start + (k + 0.5) * bin - 0.5</c>, the near and far
    /// cell, the far cell's weight and whether the sample lies inside <c>[-1, extent]</c>.
    /// </summary>
    private static (int[] Near, int[] Far, Tensor<T> FarWeight, Tensor<T> Valid) AxisTaps(
        IEngine engine, INumericOperations<T> ops, double start, double bin, int side, int extent)
    {
        var near = new int[side];
        var far = new int[side];
        var interior = new int[side];
        var inRange = new int[side];
        for (int k = 0; k < side; k++)
        {
            double position = start + (k + 0.5) * bin - 0.5;
            inRange[k] = position < -1 || position > extent ? 0 : 1;
            double clamped = Math.Max(position, 0);
            near[k] = Math.Min((int)Math.Floor(clamped), extent - 1);
            far[k] = Math.Min(near[k] + 1, extent - 1);
            interior[k] = near[k] >= extent - 1 ? 0 : 1;
        }

        // The far weight is the clamped position minus the near cell, zero at the last cell.
        var positions = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorArange<T>(0, side, 1), ops.FromDouble(0.5)), ops.FromDouble(bin)),
            ops.FromDouble(start - 0.5));
        var clampedPositions = engine.TensorMax(positions, ops.Zero);
        var farWeight = engine.TensorMultiply(
            engine.TensorSubtract(clampedPositions, CvTensorOps<T>.IndicesAsValues(near, extent)),
            CvTensorOps<T>.IndicesAsValues(interior, 2));
        return (near, far, farWeight, CvTensorOps<T>.IndicesAsValues(inRange, 2));
    }

    /// <summary>The detection forward, optionally adding boxes to the proposals the RoI heads classify.</summary>
    /// <remarks>
    /// With <paramref name="sampleRandom"/> (training) the box head runs only on the RoIs the stage loss trains on, as
    /// torchvision's RoIHeads does, instead of on every proposal before sampling (see FasterRCNN).
    /// </remarks>
    private List<Tensor<T>> ForwardDetection(Tensor<T> input, IReadOnlyList<double[]>? extraProposals,
        out List<Tensor<T>> fpnFeatures, out List<BoundingBox<T>> anchors, Random? sampleRandom = null)
    {
        var (features, proposalBoxes, objectness, rpnDeltas, levelAnchors, _) = ProposeRegionsCore(input, ProposalsPerImage);
        fpnFeatures = features;
        anchors = levelAnchors;
        if (extraProposals is { Count: > 0 })
            proposalBoxes = TwoStageTargets.AppendBoxes(proposalBoxes, extraProposals);

        if (sampleRandom is not null && extraProposals is not null && proposalBoxes.Shape[0] > 0)
        {
            var rows = _detectionLoss.SampleStageRois(proposalBoxes, extraProposals, 0, sampleRandom);
            proposalBoxes = rows.Length == 0
                ? new Tensor<T>(new[] { 0, 4 })
                : CvTensorOps<T>.Select(proposalBoxes, rows, 0);
        }

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
        if (count > 0 && width > 1)
        {
            // Every (RoI, foreground class) pair at once, RoI-major: pair m = r * (width - 1) + (c - 1).
            int foreground = width - 1;
            int pairs = count * foreground;
            var roiOf = new int[pairs];
            var cellOf = new int[pairs];
            for (int m = 0; m < pairs; m++)
            {
                roiOf[m] = m / foreground;
                cellOf[m] = roiOf[m] * width + m % foreground + 1;
            }

            var probabilities = TwoStageBoxDecoding<T>.ClassProbabilities(classLogits);
            var scores = CvTensorOps<T>.Select(Engine.Reshape(probabilities, new[] { count * width }), cellOf, 0);
            var deltas = CvTensorOps<T>.Select(Engine.Reshape(boxDeltas, new[] { count * width, 4 }), cellOf, 0);

            // The detection loss multiplies the stage's regression targets by these weights, so the predicted deltas are
            // divided by them here; unit weights (the default) leave them raw.
            var regressionWeights = _detectionLoss.StageBoxRegressionWeights(0);
            var boxes = TwoStageBoxDecoding<T>.Decode(CvTensorOps<T>.Select(proposals, roiOf, 0), deltas, regressionWeights,
                BoxDeltaClamp, 1.0, 1.0, imageWidth, imageHeight);

            var scoreValues = CvTensorOps<T>.ToDoubleArray(scores);
            var scoreItems = scores.ToArray();
            var proposalWidths = CvTensorOps<T>.ToDoubleArray(boxes.ProposalWidth);
            var proposalHeights = CvTensorOps<T>.ToDoubleArray(boxes.ProposalHeight);
            var x1 = boxes.X1.ToArray(); var y1 = boxes.Y1.ToArray(); var x2 = boxes.X2.ToArray(); var y2 = boxes.Y2.ToArray();
            var x1d = CvTensorOps<T>.ToDoubleArray(boxes.X1); var y1d = CvTensorOps<T>.ToDoubleArray(boxes.Y1);
            var x2d = CvTensorOps<T>.ToDoubleArray(boxes.X2); var y2d = CvTensorOps<T>.ToDoubleArray(boxes.Y2);
            for (int m = 0; m < pairs; m++)
            {
                if (proposalWidths[m] <= 0 || proposalHeights[m] <= 0) continue;
                if (scoreValues[m] < threshold) continue;
                if (x2d[m] <= x1d[m] || y2d[m] <= y1d[m]) continue;

                candidates.Add(new Detection<T>(
                    new BoundingBox<T>(x1[m], y1[m], x2[m], y2[m], BoundingBoxFormat.XYXY),
                    m % foreground, scoreItems[m]));
            }
        }

        var kept = _nms.ApplyClassAware(candidates, NumOps.ToDouble(Options.NmsThreshold));
        return CvTensorOps<T>.OrderByDescending(kept, d => d.Confidence).Take(Options.MaxDetections).ToList();
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

    /// <summary>
    /// An <c>[imageHeight, imageWidth]</c> map that is zero except inside the box, which takes the
    /// nearest-neighbour resample of <paramref name="mask"/> (cell <c>min(offset / boxSize * maskSize, maskSize - 1)</c>).
    /// </summary>
    private Tensor<T> PasteMask(int imageHeight, int imageWidth, Tensor<T> mask, double boxX1, double boxY1, double boxX2, double boxY2)
    {
        var fullMask = new Tensor<T>(new[] { imageHeight, imageWidth });
        int x1 = Math.Max(0, (int)boxX1);
        int y1 = Math.Max(0, (int)boxY1);
        int x2 = Math.Min(imageWidth, (int)Math.Ceiling(boxX2));
        int y2 = Math.Min(imageHeight, (int)Math.Ceiling(boxY2));

        int maskH = mask.Shape[0];
        int maskW = mask.Shape[1];
        int boxH = y2 - y1;
        int boxW = x2 - x1;
        if (boxH <= 0 || boxW <= 0) return fullMask;

        var rowIndex = new int[boxH];
        for (int y = 0; y < boxH; y++) rowIndex[y] = Math.Min((int)((double)y / boxH * maskH), maskH - 1);
        var colIndex = new int[boxW];
        for (int x = 0; x < boxW; x++) colIndex[x] = Math.Min((int)((double)x / boxW * maskW), maskW - 1);

        var patch = CvTensorOps<T>.Select(CvTensorOps<T>.Select(mask, rowIndex, 0), colIndex, 1);
        return Engine.TensorSetSlice(fullMask, patch, new[] { y1, x1 });
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
            using var download = AiDotNet.Helpers.SharedHttp.WithTimeout(AiDotNet.Helpers.SharedHttp.DefaultTimeout, cancellationToken);
            data = await AiDotNet.Helpers.SharedHttp.Downloads.GetByteArrayWithCancellationAsync(pathOrUrl, download.Token);
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
