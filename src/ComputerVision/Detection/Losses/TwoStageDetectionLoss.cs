using AiDotNet.Augmentation.Image;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>Region proposal and region-of-interest losses for Faster R-CNN and Cascade R-CNN.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Region proposal network (Ren et al. 2015, Eq. 1): anchors are labeled by IoU with the objects (positive above the
/// positive threshold or as an object's best anchor, negative below the negative threshold, otherwise ignored),
/// a balanced sample is drawn, the two-way object/background cross-entropy is averaged over the sample, and the
/// smooth-L1 regression of positive anchors is weighted by lambda and divided by the number of anchor locations.
/// </para>
/// <para>
/// Region-of-interest head (Girshick 2015, Eq. 1-3): proposals with IoU of at least the stage threshold take their
/// object's class, proposals with IoU in [low, threshold) are background (class 0), a sample with a bounded
/// foreground fraction is drawn, cross-entropy is averaged over it, and foreground boxes regress the deltas of
/// their own class with smooth-L1, also averaged over the sample. Cascade R-CNN applies this per stage with rising
/// thresholds to the boxes that stage actually receives (Cai and Vasconcelos 2018, Eq. 8).
/// </para>
/// <para>
/// Box deltas use the R-CNN parameterization the detectors here decode: t_x = (g_x - p_x) / p_w,
/// t_y = (g_y - p_y) / p_h, t_w = log(g_w / p_w), t_h = log(g_h / p_h), with center coordinates. Assignment and
/// sampling use detached host values; the losses are built from engine operations on the live heads.
/// </para>
/// <para><b>For Beginners:</b> the first loss teaches the detector where objects might be; the second teaches it what
/// each proposed region contains and how to tighten its box.</para>
/// </remarks>
public sealed class TwoStageDetectionLoss<T>
{
    private static readonly INumericOperations<T> NumOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
    private readonly TwoStageDetectionLossOptions _options;
    private readonly int _foregroundClasses;

    /// <summary>Creates the objective for a detector with the given foreground classes and detection stages.</summary>
    /// <param name="foregroundClasses">Object classes; the heads add a background class at index 0.</param>
    /// <param name="stages">Detection stages (1 for Faster R-CNN).</param>
    /// <param name="options">Sampling thresholds and weights; copied on construction.</param>
    public TwoStageDetectionLoss(int foregroundClasses, int stages, TwoStageDetectionLossOptions options)
    {
        if (foregroundClasses < 1) throw new ArgumentOutOfRangeException(nameof(foregroundClasses));
        if (stages < 1) throw new ArgumentOutOfRangeException(nameof(stages));
        if (options is null) throw new ArgumentNullException(nameof(options));
        _options = options.Snapshot(stages);
        _foregroundClasses = foregroundClasses;
    }

    /// <summary>The copied sampling thresholds and weights.</summary>
    internal TwoStageDetectionLossOptions Options => _options;

    /// <summary>Builds the region proposal loss for one image.</summary>
    /// <param name="objectness">Raw two-way logits [anchors, 2]; index 1 is "object".</param>
    /// <param name="deltas">Raw box deltas [anchors, 4].</param>
    /// <param name="anchors">Anchors in input pixels (corner format), aligned with the logits.</param>
    /// <param name="gold">Object boxes in input pixels, as (x1, y1, x2, y2).</param>
    /// <param name="anchorsPerLocation">Anchor shapes per feature position, for the paper's N_reg.</param>
    /// <param name="random">The source of the sampling draws.</param>
    public Tensor<T> ComputeProposalLoss(Tensor<T> objectness, Tensor<T> deltas, IReadOnlyList<BoundingBox<T>> anchors,
        IReadOnlyList<double[]> gold, int anchorsPerLocation, Random random)
    {
        if (objectness is null) throw new ArgumentNullException(nameof(objectness));
        if (deltas is null) throw new ArgumentNullException(nameof(deltas));
        if (anchors is null) throw new ArgumentNullException(nameof(anchors));
        if (gold is null) throw new ArgumentNullException(nameof(gold));
        if (random is null) throw new ArgumentNullException(nameof(random));
        int count = anchors.Count;
        if (objectness.Rank != 2 || objectness.Shape[0] != count || objectness.Shape[1] != 2)
            throw new ArgumentException("Objectness must be [anchors, 2].", nameof(objectness));
        if (deltas.Rank != 2 || deltas.Shape[0] != count || deltas.Shape[1] != 4)
            throw new ArgumentException("Deltas must be [anchors, 4].", nameof(deltas));
        if (anchorsPerLocation < 1) throw new ArgumentOutOfRangeException(nameof(anchorsPerLocation));

        var anchorBoxes = anchors.Select(anchor => new[]
        {
            NumOps.ToDouble(anchor.X1), NumOps.ToDouble(anchor.Y1), NumOps.ToDouble(anchor.X2), NumOps.ToDouble(anchor.Y2)
        }).ToArray();
        var labels = new int[count]; // -1 ignored, 0 background, 1 object
        var match = new int[count];
        for (int a = 0; a < count; a++) labels[a] = -1;
        if (gold.Count == 0)
        {
            for (int a = 0; a < count; a++) labels[a] = 0;
        }
        else
        {
            var bestForGold = new double[gold.Count];
            for (int a = 0; a < count; a++)
            {
                double best = -1;
                for (int g = 0; g < gold.Count; g++)
                {
                    double iou = IoU(anchorBoxes[a], gold[g]);
                    if (iou > best) { best = iou; match[a] = g; }
                    bestForGold[g] = Math.Max(bestForGold[g], iou);
                }
                if (best < _options.RpnNegativeIoU) labels[a] = 0;
                if (best > _options.RpnPositiveIoU) labels[a] = 1;
            }
            // Ren et al.: the anchor(s) with the highest IoU for each object are positive even below the threshold.
            for (int a = 0; a < count; a++)
                for (int g = 0; g < gold.Count; g++)
                    if (bestForGold[g] > 0 && IoU(anchorBoxes[a], gold[g]) == bestForGold[g])
                    {
                        labels[a] = 1;
                        match[a] = g;
                    }
        }

        var positives = Sample(Enumerable.Range(0, count).Where(a => labels[a] == 1).ToList(),
            (int)(_options.RpnBatchSizePerImage * _options.RpnPositiveFraction), random);
        var negatives = Sample(Enumerable.Range(0, count).Where(a => labels[a] == 0).ToList(),
            _options.RpnBatchSizePerImage - positives.Count, random);
        var engine = AiDotNetEngine.Current;
        int sampled = positives.Count + negatives.Count;
        if (sampled == 0)
            return engine.TensorAdd(ZeroConnected(objectness), ZeroConnected(deltas));

        var rows = positives.Concat(negatives).ToArray();
        var classes = positives.Select(_ => 1).Concat(negatives.Select(_ => 0)).ToArray();
        var classification = engine.TensorMultiplyScalar(CrossEntropySum(objectness, rows, classes, 2),
            NumOps.FromDouble(1.0 / sampled));
        if (positives.Count == 0)
            return engine.TensorAdd(classification, ZeroConnected(deltas));

        var targets = new T[positives.Count * 4];
        for (int i = 0; i < positives.Count; i++)
            WriteDeltas(targets, i * 4, anchorBoxes[positives[i]], gold[match[positives[i]]]);
        double locations = Math.Max(1.0, count / (double)anchorsPerLocation);
        var regression = engine.TensorMultiplyScalar(
            SmoothL1Sum(CvTensorOps<T>.Select(deltas, positives.ToArray(), 0), new Tensor<T>(targets, new[] { positives.Count, 4 })),
            NumOps.FromDouble(_options.RpnRegressionWeight / locations));
        return engine.TensorAdd(classification, regression);
    }

    /// <summary>Builds one detection stage's region-of-interest loss for one image.</summary>
    /// <param name="classLogits">Raw logits [proposals, foreground classes + 1]; index 0 is background.</param>
    /// <param name="boxDeltas">Class-specific deltas [proposals, (classes + 1) * 4].</param>
    /// <param name="proposals">The boxes this stage received [proposals, 4] in input pixels (corner format).</param>
    /// <param name="gold">Object boxes in input pixels, as (x1, y1, x2, y2).</param>
    /// <param name="goldClasses">Each object's foreground class, aligned with <paramref name="gold"/>.</param>
    /// <param name="stage">Zero-based stage index selecting the IoU threshold and weight.</param>
    /// <param name="random">The source of the sampling draws.</param>
    public Tensor<T> ComputeStageLoss(Tensor<T> classLogits, Tensor<T> boxDeltas, Tensor<T> proposals,
        IReadOnlyList<double[]> gold, IReadOnlyList<int> goldClasses, int stage, Random random)
    {
        if (classLogits is null) throw new ArgumentNullException(nameof(classLogits));
        if (boxDeltas is null) throw new ArgumentNullException(nameof(boxDeltas));
        if (proposals is null) throw new ArgumentNullException(nameof(proposals));
        if (gold is null) throw new ArgumentNullException(nameof(gold));
        if (goldClasses is null || goldClasses.Count != gold.Count)
            throw new ArgumentException("Each object needs a class.", nameof(goldClasses));
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (stage < 0 || stage >= _options.StageForegroundIoU.Length)
            throw new ArgumentOutOfRangeException(nameof(stage));
        int width = _foregroundClasses + 1;
        int count = proposals.Rank == 2 ? proposals.Shape[0] : -1;
        if (count < 0 || proposals.Shape[1] != 4)
            throw new ArgumentException("Proposals must be [proposals, 4].", nameof(proposals));
        if (classLogits.Rank != 2 || classLogits.Shape[0] != count || classLogits.Shape[1] != width)
            throw new ArgumentException($"Class logits must be [proposals, {width}].", nameof(classLogits));
        if (boxDeltas.Rank != 2 || boxDeltas.Shape[0] != count || boxDeltas.Shape[1] != width * 4)
            throw new ArgumentException($"Box deltas must be [proposals, {width * 4}].", nameof(boxDeltas));
        foreach (int goldClass in goldClasses)
            if (goldClass < 0 || goldClass >= _foregroundClasses)
                throw new ArgumentException("Every object class must be a foreground class of the detector.", nameof(goldClasses));

        var engine = AiDotNetEngine.Current;
        var proposalValues = proposals.ToArray();
        var boxes = new double[count][];
        var bestIoU = new double[count];
        var match = new int[count];
        for (int r = 0; r < count; r++)
        {
            boxes[r] = new[]
            {
                NumOps.ToDouble(proposalValues[r * 4]), NumOps.ToDouble(proposalValues[r * 4 + 1]),
                NumOps.ToDouble(proposalValues[r * 4 + 2]), NumOps.ToDouble(proposalValues[r * 4 + 3])
            };
            for (int g = 0; g < gold.Count; g++)
            {
                double iou = IoU(boxes[r], gold[g]);
                if (iou > bestIoU[r]) { bestIoU[r] = iou; match[r] = g; }
            }
        }

        double threshold = _options.StageForegroundIoU[stage];
        var foreground = Sample(Enumerable.Range(0, count).Where(r => gold.Count > 0 && bestIoU[r] >= threshold).ToList(),
            (int)(_options.RoiBatchSizePerImage * _options.RoiForegroundFraction), random);
        var background = Sample(Enumerable.Range(0, count)
                .Where(r => (gold.Count == 0 || bestIoU[r] < threshold) && bestIoU[r] >= _options.RoiBackgroundIoULow).ToList(),
            _options.RoiBatchSizePerImage - foreground.Count, random);
        int sampled = foreground.Count + background.Count;
        if (sampled == 0)
            return engine.TensorAdd(ZeroConnected(classLogits), ZeroConnected(boxDeltas));

        var rows = foreground.Concat(background).ToArray();
        var classes = foreground.Select(r => goldClasses[match[r]] + 1).Concat(background.Select(_ => 0)).ToArray();
        var classification = engine.TensorMultiplyScalar(CrossEntropySum(classLogits, rows, classes, width),
            NumOps.FromDouble(1.0 / sampled));
        Tensor<T> loss = classification;
        if (foreground.Count > 0)
        {
            var deltaIndices = new int[foreground.Count * 4];
            var targets = new T[foreground.Count * 4];
            for (int i = 0; i < foreground.Count; i++)
            {
                int r = foreground[i];
                int column = (goldClasses[match[r]] + 1) * 4;
                for (int k = 0; k < 4; k++) deltaIndices[i * 4 + k] = r * width * 4 + column + k;
                WriteDeltas(targets, i * 4, boxes[r], gold[match[r]]);
            }
            var flat = engine.Reshape(boxDeltas, new[] { boxDeltas.Length });
            var predicted = engine.Reshape(CvTensorOps<T>.Select(flat, deltaIndices, 0), new[] { foreground.Count, 4 });
            var regression = engine.TensorMultiplyScalar(
                SmoothL1Sum(predicted, new Tensor<T>(targets, new[] { foreground.Count, 4 })),
                NumOps.FromDouble(_options.RoiRegressionWeight / sampled));
            loss = engine.TensorAdd(loss, regression);
        }
        else
        {
            loss = engine.TensorAdd(loss, ZeroConnected(boxDeltas));
        }
        return engine.TensorMultiplyScalar(loss, NumOps.FromDouble(_options.StageLossWeights[stage]));
    }

    /// <summary>R-CNN box deltas of <paramref name="gold"/> relative to <paramref name="reference"/>.</summary>
    internal static double[] EncodeDeltas(double[] reference, double[] gold)
    {
        double pw = reference[2] - reference[0];
        double ph = reference[3] - reference[1];
        double gw = gold[2] - gold[0];
        double gh = gold[3] - gold[1];
        if (pw <= 0 || ph <= 0 || gw <= 0 || gh <= 0)
            throw new ArgumentException("Boxes used for regression targets must have positive width and height.");
        return new[]
        {
            (gold[0] + gw / 2 - (reference[0] + pw / 2)) / pw,
            (gold[1] + gh / 2 - (reference[1] + ph / 2)) / ph,
            Math.Log(gw / pw),
            Math.Log(gh / ph)
        };
    }

    private static void WriteDeltas(T[] destination, int offset, double[] reference, double[] gold)
    {
        var delta = EncodeDeltas(reference, gold);
        for (int k = 0; k < 4; k++) destination[offset + k] = NumOps.FromDouble(delta[k]);
    }

    /// <summary>Sum over selected rows of -log softmax(logits)[class], gathered to avoid 0 * -inf.</summary>
    private static Tensor<T> CrossEntropySum(Tensor<T> logits, int[] rows, int[] classes, int width)
    {
        var engine = AiDotNetEngine.Current;
        var logProbabilities = engine.Reshape(engine.TensorLogSoftmax(logits, 1), new[] { logits.Length });
        var entries = new int[rows.Length];
        for (int i = 0; i < rows.Length; i++) entries[i] = rows[i] * width + classes[i];
        return engine.TensorNegate(engine.ReduceSum(CvTensorOps<T>.Select(logProbabilities, entries, 0), null));
    }

    /// <summary>Sum of smooth-L1 (Girshick 2015, Eq. 3): 0.5 x^2 where |x| &lt; 1, otherwise |x| - 0.5.</summary>
    /// <remarks>
    /// The branch is chosen per element from detached values and applied as a constant mask, so the derivative is
    /// x inside the quadratic region and sign(x) outside, using only differentiable elementwise operations.
    /// </remarks>
    private static Tensor<T> SmoothL1Sum(Tensor<T> predicted, Tensor<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var difference = engine.TensorSubtract(predicted, target);
        var values = difference.ToArray();
        var quadraticMask = new T[values.Length];
        var linearMask = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            bool quadratic = Math.Abs(NumOps.ToDouble(values[i])) < 1;
            quadraticMask[i] = quadratic ? NumOps.One : NumOps.Zero;
            linearMask[i] = quadratic ? NumOps.Zero : NumOps.One;
        }
        var shape = difference.Shape.ToArray();
        var quadraticPart = engine.TensorMultiply(new Tensor<T>(quadraticMask, shape),
            engine.TensorMultiplyScalar(engine.TensorMultiply(difference, difference), NumOps.FromDouble(0.5)));
        var linearPart = engine.TensorMultiply(new Tensor<T>(linearMask, shape),
            engine.TensorAddScalar(engine.TensorAbs(difference), NumOps.FromDouble(-0.5)));
        return engine.ReduceSum(engine.TensorAdd(quadraticPart, linearPart), null);
    }

    private static Tensor<T> ZeroConnected(Tensor<T> tensor)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMultiplyScalar(engine.ReduceSum(tensor, null), NumOps.Zero);
    }

    private static List<int> Sample(List<int> candidates, int limit, Random random)
    {
        if (limit <= 0) return new List<int>();
        if (candidates.Count <= limit) return candidates;
        // Partial Fisher-Yates: an unbiased sample without replacement.
        for (int i = 0; i < limit; i++)
        {
            int j = i + random.Next(candidates.Count - i);
            (candidates[i], candidates[j]) = (candidates[j], candidates[i]);
        }
        return candidates.GetRange(0, limit);
    }

    private static double IoU(double[] a, double[] b)
    {
        double width = Math.Max(0, Math.Min(a[2], b[2]) - Math.Max(a[0], b[0]));
        double height = Math.Max(0, Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]));
        double intersection = width * height;
        double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
        return union > 0 ? intersection / union : 0;
    }
}
