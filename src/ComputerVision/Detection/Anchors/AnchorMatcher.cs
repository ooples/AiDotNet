using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.PostProcessing;

namespace AiDotNet.ComputerVision.Detection.Anchors;

/// <summary>
/// Matches anchor boxes to ground truth boxes for training object detectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> During training, we need to assign each anchor box to either:
/// - A ground truth box (positive sample): The anchor should predict this object
/// - Background (negative sample): The anchor should predict "no object"
/// - Ignore: The anchor is borderline and excluded from loss calculation
///
/// This matching process determines which anchors learn to detect which objects.</para>
/// </remarks>
public class AnchorMatcher<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly NMS<T> _nms;

    /// <summary>
    /// IoU threshold above which an anchor is considered positive.
    /// </summary>
    public double PositiveThreshold { get; }

    /// <summary>
    /// IoU threshold below which an anchor is considered negative.
    /// </summary>
    public double NegativeThreshold { get; }

    /// <summary>
    /// Whether to allow low-quality matches (best anchor for each GT even if below threshold).
    /// </summary>
    public bool AllowLowQualityMatches { get; }

    /// <summary>
    /// Creates a new anchor matcher with default thresholds.
    /// </summary>
    public AnchorMatcher()
        : this(positiveThreshold: 0.5, negativeThreshold: 0.4, allowLowQualityMatches: true)
    {
    }

    /// <summary>
    /// Creates a new anchor matcher with custom thresholds.
    /// </summary>
    /// <param name="positiveThreshold">IoU threshold for positive matches.</param>
    /// <param name="negativeThreshold">IoU threshold for negative matches.</param>
    /// <param name="allowLowQualityMatches">Whether to match best anchor per GT regardless of threshold.</param>
    public AnchorMatcher(
        double positiveThreshold,
        double negativeThreshold,
        bool allowLowQualityMatches = true)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _nms = new NMS<T>();
        PositiveThreshold = positiveThreshold;
        NegativeThreshold = negativeThreshold;
        AllowLowQualityMatches = allowLowQualityMatches;
    }

    /// <summary>
    /// Matches anchors to ground truth boxes.
    /// </summary>
    /// <param name="anchors">List of anchor boxes.</param>
    /// <param name="groundTruth">List of ground truth boxes.</param>
    /// <returns>Match result containing assignments for each anchor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method looks at each anchor and decides:
    /// - If it overlaps enough with a ground truth box → match to that box
    /// - If it doesn't overlap much with any box → mark as background
    /// - If it's in between → ignore during training
    /// </para>
    /// </remarks>
    public AnchorMatchResult<T> Match(
        List<BoundingBox<T>> anchors,
        List<BoundingBox<T>> groundTruth)
    {
        int numAnchors = anchors.Count;
        int numGt = groundTruth.Count;

        var result = new AnchorMatchResult<T>
        {
            MatchedGtIndices = new int[numAnchors],
            MatchedIoUs = new double[numAnchors],
            Labels = new MatchLabel[numAnchors]
        };

        // Initialize all as negative
        for (int i = 0; i < numAnchors; i++)
        {
            result.MatchedGtIndices[i] = -1;
            result.MatchedIoUs[i] = 0;
            result.Labels[i] = MatchLabel.Negative;
        }

        if (numGt == 0)
        {
            return result;
        }

        // Compute IoU matrix [numAnchors x numGt]
        var iouMatrix = new double[numAnchors, numGt];
        for (int i = 0; i < numAnchors; i++)
        {
            for (int j = 0; j < numGt; j++)
            {
                iouMatrix[i, j] = _nms.ComputeIoU(anchors[i], groundTruth[j]);
            }
        }

        // For each anchor, find best matching GT
        for (int i = 0; i < numAnchors; i++)
        {
            double maxIoU = 0;
            int bestGt = -1;

            for (int j = 0; j < numGt; j++)
            {
                if (iouMatrix[i, j] > maxIoU)
                {
                    maxIoU = iouMatrix[i, j];
                    bestGt = j;
                }
            }

            result.MatchedIoUs[i] = maxIoU;

            if (maxIoU >= PositiveThreshold)
            {
                result.MatchedGtIndices[i] = bestGt;
                result.Labels[i] = MatchLabel.Positive;
            }
            else if (maxIoU < NegativeThreshold)
            {
                result.Labels[i] = MatchLabel.Negative;
            }
            else
            {
                result.Labels[i] = MatchLabel.Ignore;
            }
        }

        // Low-quality matching: ensure each GT has at least one positive anchor
        if (AllowLowQualityMatches)
        {
            for (int j = 0; j < numGt; j++)
            {
                double maxIoU = 0;
                int bestAnchor = -1;

                for (int i = 0; i < numAnchors; i++)
                {
                    if (iouMatrix[i, j] > maxIoU)
                    {
                        maxIoU = iouMatrix[i, j];
                        bestAnchor = i;
                    }
                }

                // Force this anchor to be positive for this GT
                if (bestAnchor >= 0 && maxIoU > 0)
                {
                    result.MatchedGtIndices[bestAnchor] = j;
                    result.MatchedIoUs[bestAnchor] = maxIoU;
                    result.Labels[bestAnchor] = MatchLabel.Positive;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Matches anchors using center-based assignment (used in FCOS, YOLOX).
    /// </summary>
    /// <param name="anchors">List of anchor points (centers).</param>
    /// <param name="groundTruth">List of ground truth boxes.</param>
    /// <param name="strides">Stride at each anchor's feature level.</param>
    /// <returns>Match result with center-based assignment.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Center-based matching assigns an anchor to a GT box
    /// if the anchor's center point falls inside the GT box. This is used by
    /// anchor-free detectors like FCOS and YOLOX.</para>
    /// </remarks>
    public AnchorMatchResult<T> MatchCenterBased(
        List<(double X, double Y)> anchorCenters,
        List<BoundingBox<T>> groundTruth,
        List<int> anchorStrides)
    {
        int numAnchors = anchorCenters.Count;
        int numGt = groundTruth.Count;

        var result = new AnchorMatchResult<T>
        {
            MatchedGtIndices = new int[numAnchors],
            MatchedIoUs = new double[numAnchors],
            Labels = new MatchLabel[numAnchors]
        };

        // Initialize all as negative
        for (int i = 0; i < numAnchors; i++)
        {
            result.MatchedGtIndices[i] = -1;
            result.Labels[i] = MatchLabel.Negative;
        }

        if (numGt == 0)
        {
            return result;
        }

        // For each anchor, check if its center is inside any GT box
        for (int i = 0; i < numAnchors; i++)
        {
            var (cx, cy) = anchorCenters[i];
            int stride = anchorStrides[i];

            double bestArea = double.MaxValue;
            int bestGt = -1;

            for (int j = 0; j < numGt; j++)
            {
                var gt = groundTruth[j];
                double x1 = _numOps.ToDouble(gt.X1);
                double y1 = _numOps.ToDouble(gt.Y1);
                double x2 = _numOps.ToDouble(gt.X2);
                double y2 = _numOps.ToDouble(gt.Y2);

                // Check if center is inside GT box
                if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2)
                {
                    double area = (x2 - x1) * (y2 - y1);

                    // Also check if GT size is appropriate for this stride
                    double gtSize = Math.Sqrt(area);
                    double minSize = stride * 0.5;
                    double maxSize = stride * 16;

                    if (gtSize >= minSize && gtSize <= maxSize && area < bestArea)
                    {
                        bestArea = area;
                        bestGt = j;
                    }
                }
            }

            if (bestGt >= 0)
            {
                result.MatchedGtIndices[i] = bestGt;
                result.Labels[i] = MatchLabel.Positive;
                result.MatchedIoUs[i] = 1.0; // Center-based doesn't use IoU
            }
        }

        return result;
    }

    /// <summary>
    /// Matches anchors using SimOTA (used in YOLOX).
    /// </summary>
    /// <param name="anchors">List of anchor boxes.</param>
    /// <param name="groundTruth">List of ground truth boxes.</param>
    /// <param name="predictions">Predicted boxes for cost calculation.</param>
    /// <param name="predScores">Predicted scores for cost calculation.</param>
    /// <returns>Match result using optimal transport assignment.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> SimOTA (Simplified Optimal Transport Assignment) is an
    /// advanced matching strategy that considers both location and predicted confidence
    /// to make better assignments. It dynamically decides how many anchors each GT gets.</para>
    /// </remarks>
    public AnchorMatchResult<T> MatchSimOTA(
        List<BoundingBox<T>> anchors,
        List<BoundingBox<T>> groundTruth,
        List<BoundingBox<T>> predictions,
        List<double> predScores)
    {
        int numAnchors = anchors.Count;
        int numGt = groundTruth.Count;

        var result = new AnchorMatchResult<T>
        {
            MatchedGtIndices = new int[numAnchors],
            MatchedIoUs = new double[numAnchors],
            Labels = new MatchLabel[numAnchors]
        };

        for (int i = 0; i < numAnchors; i++)
        {
            result.MatchedGtIndices[i] = -1;
            result.Labels[i] = MatchLabel.Negative;
        }

        if (numGt == 0 || predictions.Count != numAnchors)
        {
            return result;
        }

        // Step 1: Filter candidates by center distance
        var candidateIndices = new List<int>[numGt];
        for (int j = 0; j < numGt; j++)
        {
            candidateIndices[j] = new List<int>();
            var gt = groundTruth[j];
            double gtCx = (_numOps.ToDouble(gt.X1) + _numOps.ToDouble(gt.X2)) / 2;
            double gtCy = (_numOps.ToDouble(gt.Y1) + _numOps.ToDouble(gt.Y2)) / 2;
            double gtSize = Math.Sqrt(gt.Area());
            double radius = gtSize * 2.5; // Center prior radius

            for (int i = 0; i < numAnchors; i++)
            {
                var anchor = anchors[i];
                double anchorCx = (_numOps.ToDouble(anchor.X1) + _numOps.ToDouble(anchor.X2)) / 2;
                double anchorCy = (_numOps.ToDouble(anchor.Y1) + _numOps.ToDouble(anchor.Y2)) / 2;

                double dist = Math.Sqrt(Math.Pow(anchorCx - gtCx, 2) + Math.Pow(anchorCy - gtCy, 2));
                if (dist <= radius)
                {
                    candidateIndices[j].Add(i);
                }
            }
        }

        // Step 2: Compute cost matrix for candidates
        for (int j = 0; j < numGt; j++)
        {
            if (candidateIndices[j].Count == 0) continue;

            var costs = new List<(int AnchorIdx, double Cost, double IoU)>();

            foreach (int i in candidateIndices[j])
            {
                double iou = _nms.ComputeIoU(predictions[i], groundTruth[j]);
                double clsCost = -Math.Log(Math.Max(predScores[i], 1e-6));
                double regCost = -Math.Log(Math.Max(iou, 1e-6));
                double cost = clsCost + 3.0 * regCost;

                costs.Add((i, cost, iou));
            }

            // Step 3: Dynamic k selection based on IoU sum
            double iouSum = costs.Sum(c => c.IoU);
            int dynamicK = Math.Max(1, (int)Math.Round(iouSum));
            dynamicK = Math.Min(dynamicK, costs.Count);

            // Select top-k candidates with lowest cost
            var topK = costs.OrderBy(c => c.Cost).Take(dynamicK).ToList();

            foreach (var (anchorIdx, _, iou) in topK)
            {
                // If anchor already matched to a GT with higher IoU, skip
                if (result.Labels[anchorIdx] == MatchLabel.Positive &&
                    result.MatchedIoUs[anchorIdx] >= iou)
                {
                    continue;
                }

                result.MatchedGtIndices[anchorIdx] = j;
                result.MatchedIoUs[anchorIdx] = iou;
                result.Labels[anchorIdx] = MatchLabel.Positive;
            }
        }

        return result;
    }
}

/// <summary>
/// Result of anchor-to-ground-truth matching.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AnchorMatchResult<T>
{
    /// <summary>
    /// For each anchor, the index of the matched GT box (-1 if no match).
    /// </summary>
    public int[] MatchedGtIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// IoU between each anchor and its matched GT box.
    /// </summary>
    public double[] MatchedIoUs { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Label for each anchor (positive, negative, or ignore).
    /// </summary>
    public MatchLabel[] Labels { get; set; } = Array.Empty<MatchLabel>();

    /// <summary>
    /// Gets the number of positive matches.
    /// </summary>
    public int NumPositive => Labels.Count(l => l == MatchLabel.Positive);

    /// <summary>
    /// Gets the number of negative matches.
    /// </summary>
    public int NumNegative => Labels.Count(l => l == MatchLabel.Negative);

    /// <summary>
    /// Gets the number of ignored anchors.
    /// </summary>
    public int NumIgnored => Labels.Count(l => l == MatchLabel.Ignore);

    /// <summary>
    /// Gets indices of all positive anchors.
    /// </summary>
    public int[] GetPositiveIndices()
    {
        return Labels
            .Select((label, idx) => (label, idx))
            .Where(x => x.label == MatchLabel.Positive)
            .Select(x => x.idx)
            .ToArray();
    }

    /// <summary>
    /// Gets indices of all negative anchors.
    /// </summary>
    public int[] GetNegativeIndices()
    {
        return Labels
            .Select((label, idx) => (label, idx))
            .Where(x => x.label == MatchLabel.Negative)
            .Select(x => x.idx)
            .ToArray();
    }
}

/// <summary>
/// Label indicating how an anchor should be treated during training.
/// </summary>
public enum MatchLabel
{
    /// <summary>Anchor matched to a GT box - should predict the object.</summary>
    Positive,

    /// <summary>Anchor not matched - should predict background/no object.</summary>
    Negative,

    /// <summary>Anchor is borderline - excluded from loss calculation.</summary>
    Ignore
}
