using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// The instance branch of a Panoptic Cascade Mask R-CNN (Cai and Vasconcelos 2018, with the Mask R-CNN mask head of
/// He et al. 2017) on a shared FPN: an RPN over P2-P6, three cascade box stages trained at IoU 0.5, 0.6 and 0.7, and
/// a mask head on the last stage's boxes.
/// </summary>
/// <remarks>
/// <para>
/// This is the instance head CUPS trains on its panoptic pseudo labels (Hahn et al. 2025, Sec. 3.3: "Panoptic
/// Cascade Mask R-CNN"). It reads the FPN levels the owning model computes from its own backbone (strides 4, 8, 16
/// and 32, 256 channels), so the backbone is shared with the semantic branch as in the paper.
/// </para>
/// <para>
/// It reuses the library's two-stage pieces: <see cref="RPN{T}"/>, <see cref="CascadeStage{T}"/>, the cascade's box
/// refinement and stage losses (<see cref="TwoStageDetectionLoss{T}"/> with Cascade R-CNN's reference stage weights)
/// and <see cref="MaskHead{T}"/>. Detection scores are the cascade ensemble: each RoI's class probabilities averaged
/// over the three stages. Class 0 of the box heads is the background; foreground class <c>k</c> is column
/// <c>k + 1</c>.
/// </para>
/// </remarks>
internal sealed class CascadeMaskRCNNHead<T> : CvParameterModule<T>
{
    /// <summary>FPN width (Lin et al. 2017).</summary>
    internal const int Channels = 256;

    /// <summary>Strides of the FPN levels P2-P5 the RoIs are pooled from.</summary>
    internal static readonly int[] Strides = { 4, 8, 16, 32 };

    private const int NumStages = 3;
    private const int BoxRoiSize = 7;
    private const int MaskResolution = 28;
    // detectron2's ROI_HEADS.SCORE_THRESH_TEST, NMS_THRESH_TEST and TEST.DETECTIONS_PER_IMAGE.
    private const double ScoreThreshold = 0.05;
    private const double NmsThreshold = 0.5;
    private const int MaxDetections = 100;

    private readonly int _numClasses;
    private readonly RPN<T> _rpn;
    private readonly RoIAlign<T> _boxAlign;
    private readonly RoIAlign<T> _maskAlign;
    private readonly RoIAlign<T> _maskTargetAlign;
    private readonly CascadeStage<T>[] _stages;
    private readonly MaskHead<T> _maskHead;
    private readonly TwoStageDetectionLoss<T> _loss;
    private readonly NMS<T> _nms = new();

    internal CascadeMaskRCNNHead(int numClasses)
    {
        if (numClasses < 1) throw new ArgumentOutOfRangeException(nameof(numClasses));
        _numClasses = numClasses;
        _rpn = new RPN<T>(Channels, Channels);
        _boxAlign = new RoIAlign<T>(BoxRoiSize, samplingRatio: 2);
        _maskHead = new MaskHead<T>(Channels, numClasses, MaskResolution);
        _maskAlign = new RoIAlign<T>(_maskHead.RoiSize, samplingRatio: 2);
        _maskTargetAlign = new RoIAlign<T>(MaskResolution, samplingRatio: 2);
        _stages = new CascadeStage<T>[NumStages];
        for (int i = 0; i < NumStages; i++)
            _stages[i] = new CascadeStage<T>(Channels * BoxRoiSize * BoxRoiSize, 1024, numClasses + 1);
        _loss = new TwoStageDetectionLoss<T>(numClasses, NumStages, CascadeRCNN<T>.ReferenceLossOptions(NumStages));
    }

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _rpn;
        foreach (var stage in _stages) yield return stage;
        yield return _maskHead;
    }

    /// <summary>The RPN and cascade outputs for one image.</summary>
    internal sealed class Outputs
    {
        internal Outputs(Tensor<T> objectness, Tensor<T> deltas, List<BoundingBox<T>> anchors,
            List<(Tensor<T> ClassLogits, Tensor<T> BoxDeltas, Tensor<T> Boxes)> stages)
        {
            Objectness = objectness;
            Deltas = deltas;
            Anchors = anchors;
            Stages = stages;
        }

        internal Tensor<T> Objectness { get; }
        internal Tensor<T> Deltas { get; }
        internal List<BoundingBox<T>> Anchors { get; }
        internal List<(Tensor<T> ClassLogits, Tensor<T> BoxDeltas, Tensor<T> Boxes)> Stages { get; }
    }

    /// <summary>
    /// Runs the RPN over P2-P6 (P6 subsampled from P5) and the three cascade stages for one image.
    /// </summary>
    /// <param name="fpn">The FPN levels P2-P5 <c>[1, 256, h, w]</c>, finest first.</param>
    /// <param name="height">Input height in pixels.</param>
    /// <param name="width">Input width in pixels.</param>
    /// <param name="gold">Ground-truth boxes added to the proposals during training, as Cascade R-CNN does.</param>
    internal Outputs Run(IReadOnlyList<Tensor<T>> fpn, int height, int width, IReadOnlyList<double[]>? gold)
    {
        if (fpn is null) throw new ArgumentNullException(nameof(fpn));
        if (fpn.Count != Strides.Length)
            throw new ArgumentException($"The cascade head reads {Strides.Length} FPN levels (strides 4-32).", nameof(fpn));
        var rpnLevels = new List<Tensor<T>>(fpn) { CvTensorOps<T>.MaxPoolPadded(fpn[fpn.Count - 1], 1, 2, 0) };
        var (objectness, deltas, anchors, levelAnchorCounts) = _rpn.ForwardLevels(rpnLevels);
        var proposals = _rpn.GenerateProposals(objectness, deltas, anchors, height, width,
            preNmsTopK: 1000, postNmsTopK: FasterRCNN<T>.PostNmsTopK, nmsThreshold: 0.7, levelAnchorCounts: levelAnchorCounts);
        var boxes = proposals.Count == 0 ? new Tensor<T>(new[] { 0, 4 }) : proposals[0].boxes;
        if (gold is { Count: > 0 }) boxes = TwoStageTargets.AppendBoxes(boxes, gold);

        var stages = new List<(Tensor<T>, Tensor<T>, Tensor<T>)>(NumStages);
        if (boxes.Shape[0] == 0) return new Outputs(objectness, deltas, anchors, stages);
        for (int s = 0; s < NumStages; s++)
        {
            var pooled = FpnRoIPooler<T>.Pool(_boxAlign, fpn, Strides, boxes);
            var flattened = AiDotNetEngine.Current.Reshape(
                pooled, new[] { pooled.Shape[0], pooled.Shape[1] * pooled.Shape[2] * pooled.Shape[3] });
            var (classLogits, boxDeltas) = _stages[s].Forward(flattened);
            stages.Add((classLogits, boxDeltas, boxes));
            // The refined boxes are constants to the next stage's RoIAlign, as in Cascade R-CNN.
            if (s < NumStages - 1)
                boxes = CascadeRCNN<T>.RefineBoxes(boxes, boxDeltas, width, height, _loss.StageBoxRegressionWeights(s));
        }

        return new Outputs(objectness, deltas, anchors, stages);
    }

    /// <summary>
    /// Detected instances of one image: binary masks <c>[M, H*W]</c>, their classes and scores <c>[M]</c>.
    /// </summary>
    /// <remarks>
    /// Scores are the cascade ensemble (class probabilities averaged over the stages); the last stage's deltas refine
    /// its boxes; every (RoI, class) pair scoring above 0.05 goes through class-aware NMS at 0.5 and the best 100 are
    /// kept (detectron2's test settings). Each kept box's mask for its class is pasted into the image and thresholded at 0.5.
    /// </remarks>
    internal (Tensor<int> Masks, int[] Classes, Tensor<T> Scores) Detect(IReadOnlyList<Tensor<T>> fpn, int height, int width)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var empty = (new Tensor<int>(new[] { 0, height * width }), Array.Empty<int>(), new Tensor<T>(new[] { 0 }));
        var outputs = Run(fpn, height, width, null);
        if (outputs.Stages.Count == 0) return empty;

        var last = outputs.Stages[outputs.Stages.Count - 1];
        Tensor<T>? probabilitySum = null;
        foreach (var stage in outputs.Stages)
        {
            var probabilities = engine.TensorSoftmax(stage.ClassLogits, 1);
            probabilitySum = probabilitySum is null ? probabilities : engine.TensorAdd(probabilitySum, probabilities);
        }

        var ensemble = engine.TensorMultiplyScalar(
            probabilitySum ?? throw new InvalidOperationException("No cascade stage ran."), ops.FromDouble(1.0 / outputs.Stages.Count));
        var foreground = engine.TensorNarrow(ensemble, 1, 1, _numClasses);                        // [R, K]
        var boxes = CascadeRCNN<T>.RefineBoxes(last.Boxes, last.BoxDeltas, width, height,
            _loss.StageBoxRegressionWeights(outputs.Stages.Count - 1));
        var candidates = Candidates(foreground, boxes, ScoreThreshold);

        var kept = _nms.ApplyClassAware(candidates, NmsThreshold)
            .OrderByDescending(detection => detection.Confidence)
            .Take(MaxDetections)
            .ToList();
        int m = kept.Count;
        if (m == 0) return empty;

        var keptCorners = new T[4 * m];
        var keptScores = new T[m];
        var keptClasses = new int[m];
        for (int i = 0; i < m; i++)
        {
            keptCorners[4 * i] = kept[i].Box.X1;
            keptCorners[4 * i + 1] = kept[i].Box.Y1;
            keptCorners[4 * i + 2] = kept[i].Box.X2;
            keptCorners[4 * i + 3] = kept[i].Box.Y2;
            keptScores[i] = kept[i].Confidence;
            keptClasses[i] = kept[i].ClassId;
        }

        var keptBoxes = new Tensor<T>(keptCorners, new[] { m, 4 });
        var maskLogits = _maskHead.Forward(FpnRoIPooler<T>.Pool(_maskAlign, fpn, Strides, keptBoxes)); // [M, K, 28, 28]
        var classMasks = engine.TensorSigmoid(ClassChannels(maskLogits, keptClasses));                // [M, 28 * 28]
        var masks = PasteMasks(classMasks, keptBoxes, height, width);
        return (masks, keptClasses, new Tensor<T>(keptScores, new[] { m }));
    }

    /// <summary>
    /// The detection candidates of detectron2's <c>fast_rcnn_inference</c>: every (RoI, foreground class) pair whose
    /// score is strictly above <paramref name="threshold"/>, with the RoI's (class-agnostic) box. A RoI can therefore
    /// yield several candidates of different classes; class-aware NMS then runs per class.
    /// </summary>
    /// <param name="foreground">Foreground class scores <c>[R, K]</c>.</param>
    /// <param name="boxes">The RoI boxes <c>[R, 4]</c> (x1, y1, x2, y2).</param>
    /// <param name="threshold">The score threshold (0.05 at test time).</param>
    internal static List<Detection<T>> Candidates(Tensor<T> foreground, Tensor<T> boxes, double threshold)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int r = foreground.Shape[0], k = foreground.Shape[1];
        int[] passes = engine.TensorGreaterThan(
            foreground, Tensor<T>.CreateDefault(new[] { r, k }, ops.FromDouble(threshold))).Cast<int>().ToArray();
        T[] scores = foreground.ToArray();
        T[] corners = boxes.ToArray();
        var candidates = new List<Detection<T>>();
        for (int i = 0; i < r; i++)
        {
            for (int c = 0; c < k; c++)
            {
                if (passes[i * k + c] == 0) continue;
                var box = new BoundingBox<T>(corners[4 * i], corners[4 * i + 1], corners[4 * i + 2], corners[4 * i + 3]);
                candidates.Add(new Detection<T>(box, c, scores[i * k + c]));
            }
        }

        return candidates;
    }

    /// <summary>
    /// The instance branch's training loss for one image: the RPN loss, the three cascade stage losses (each stage
    /// samples its RoIs at its own IoU threshold) and the mask loss on the last stage's foreground RoIs.
    /// </summary>
    /// <param name="fpn">The FPN levels P2-P5, finest first.</param>
    /// <param name="height">Input height.</param>
    /// <param name="width">Input width.</param>
    /// <param name="gold">Instance boxes in pixels (x1, y1, x2, y2).</param>
    /// <param name="goldClasses">Instance classes.</param>
    /// <param name="goldMasks">Instance masks <c>[M, 1, H, W]</c>.</param>
    /// <param name="random">The RoI and anchor sampling draws.</param>
    internal Tensor<T> Loss(IReadOnlyList<Tensor<T>> fpn, int height, int width, IReadOnlyList<double[]> gold,
        IReadOnlyList<int> goldClasses, Tensor<T> goldMasks, Random random)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var outputs = Run(fpn, height, width, gold);
        var loss = _loss.ComputeProposalLoss(TwoStageTargets.FirstImage(outputs.Objectness),
            TwoStageTargets.FirstImage(outputs.Deltas), outputs.Anchors, gold, _rpn.AnchorsPerLocation, random);
        IReadOnlyList<(int Row, int Gold)> foreground = Array.Empty<(int Row, int Gold)>();
        for (int s = 0; s < outputs.Stages.Count; s++)
        {
            var stage = outputs.Stages[s];
            var stageLoss = s == outputs.Stages.Count - 1
                ? _loss.ComputeStageLoss(stage.ClassLogits, stage.BoxDeltas, stage.Boxes, gold, goldClasses, s, random, out foreground)
                : _loss.ComputeStageLoss(stage.ClassLogits, stage.BoxDeltas, stage.Boxes, gold, goldClasses, s, random);
            loss = engine.TensorAdd(loss, stageLoss);
        }

        if (foreground.Count == 0 || outputs.Stages.Count == 0) return loss;

        // Mask R-CNN's mask loss: the mean per-pixel binary cross-entropy of each foreground RoI's mask for its
        // object's class, against the object's mask resampled into the RoI (RoIAlign on the ground-truth mask).
        int count = foreground.Count;
        var rows = foreground.Select(f => f.Row).ToArray();
        var roiBoxes = CvTensorOps<T>.Select(outputs.Stages[outputs.Stages.Count - 1].Boxes, rows, 0);
        var logits = ClassChannels(
            _maskHead.Forward(FpnRoIPooler<T>.Pool(_maskAlign, fpn, Strides, roiBoxes)),
            foreground.Select(f => goldClasses[f.Gold]).ToArray());                                     // [F, 28 * 28]
        Tensor<T> targets;
        using (new NoGradScope<T>())
        {
            var resampled = _maskTargetAlign.Forward(goldMasks, roiBoxes, 1.0, foreground.Select(f => f.Gold).ToArray());
            targets = engine.Reshape(
                engine.TensorGreaterEqual(resampled, Tensor<T>.CreateDefault(resampled.Shape.ToArray(), ops.FromDouble(0.5))),
                new[] { count, MaskResolution * MaskResolution });
        }

        var crossEntropy = engine.TensorSubtract(engine.Softplus(logits), engine.TensorMultiply(logits, targets));
        var maskLoss = engine.TensorMultiplyScalar(engine.ReduceSum(crossEntropy, null),
            ops.FromDouble(1.0 / ((double)count * MaskResolution * MaskResolution)));
        return engine.TensorAdd(loss, maskLoss);
    }

    /// <summary>Each RoI's mask logits for its class: rows <c>i * K + class_i</c> of <c>[R, K, S, S]</c>.</summary>
    private Tensor<T> ClassChannels(Tensor<T> maskLogits, int[] classes)
    {
        var engine = AiDotNetEngine.Current;
        int r = maskLogits.Shape[0], side = maskLogits.Shape[2];
        var perClass = engine.Reshape(maskLogits, new[] { r * _numClasses, side * side });
        var rows = new int[r];
        for (int i = 0; i < r; i++) rows[i] = i * _numClasses + classes[i];
        return engine.TensorIndexSelect(perClass, new Tensor<int>(rows, new[] { r }), 0);
    }

    /// <summary>
    /// Pastes each RoI mask <c>[M, S*S]</c> into its box of an <paramref name="height"/> x <paramref name="width"/>
    /// image (bilinear resize to the box, zero outside) and thresholds it at 0.5.
    /// </summary>
    internal static Tensor<int> PasteMasks(Tensor<T> roiMasks, Tensor<T> boxes, int height, int width)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = roiMasks.Shape[0], side = (int)Math.Round(Math.Sqrt(roiMasks.Shape[1]));
        var half = Tensor<T>.CreateDefault(new[] { height, width }, ops.FromDouble(0.5));
        var low = engine.TensorFloor(boxes).Cast<int>().ToArray();
        var high = engine.TensorCeiling(boxes).Cast<int>().ToArray();
        var pasted = new Tensor<int>[m];
        for (int i = 0; i < m; i++)
        {
            int x1 = Math.Max(0, Math.Min(width - 1, low[4 * i])), y1 = Math.Max(0, Math.Min(height - 1, low[4 * i + 1]));
            int x2 = Math.Max(x1 + 1, Math.Min(width, high[4 * i + 2])), y2 = Math.Max(y1 + 1, Math.Min(height, high[4 * i + 3]));
            var mask = engine.Reshape(engine.TensorSlice(roiMasks, new[] { i, 0 }, new[] { 1, side * side }), new[] { 1, 1, side, side });
            var resized = engine.Reshape(
                engine.Interpolate(mask, new[] { y2 - y1, x2 - x1 }, InterpolateMode.Bilinear, false), new[] { y2 - y1, x2 - x1 });
            var canvas = engine.Pad(resized, y1, height - y2, x1, width - x2, ops.Zero);
            pasted[i] = engine.Reshape(engine.TensorGreaterEqual(canvas, half).Cast<int>(), new[] { 1, height * width });
        }

        return engine.TensorConcatenate(pasted, 0);
    }
}
