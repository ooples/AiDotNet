using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Helpers;

namespace AiDotNet.Metrics;

/// <summary>
/// ICDAR-style evaluation metrics for text detection: polygon IoU, and the precision, recall and
/// H-mean triple that the ICDAR Robust Reading competitions report.
/// </summary>
/// <remarks>
/// <para>
/// Text detectors localise words or lines as quadrilaterals or polygons rather than axis-aligned
/// boxes, because printed and scene text is frequently rotated or curved. Evaluation therefore
/// works on polygon overlap, not box overlap.
/// </para>
/// <para><b>The ICDAR 2015 IoU protocol.</b>
/// Within each image, predictions are considered in descending confidence order and matched
/// one-to-one against ground-truth regions: a prediction claims the unmatched ground-truth region
/// it overlaps most, provided that overlap reaches the IoU threshold (0.5 by convention). Then
/// precision is matched / predicted, recall is matched / ground-truth, and H-mean is their
/// harmonic mean. H-mean is the number competitions rank on.
/// </para>
/// <para><b>Convexity.</b>
/// <see cref="PolygonIoU"/> computes the intersection by Sutherland-Hodgman clipping, which is
/// exact when the second polygon is convex. Detector output for word- and line-level text is
/// quadrilateral or near-convex, which is the case the ICDAR protocol is defined over. For a
/// strongly concave polygon (a curved-text detector emitting a banana-shaped region) the
/// intersection can be over-estimated, so treat such scores as approximate.
/// </para>
/// <para><b>For Beginners:</b> Precision asks "of the regions the model reported, how many were
/// real text?" Recall asks "of the real text, how much did the model find?" A model can score
/// perfectly on one by sacrificing the other - report everything for perfect recall, report only
/// the single most obvious word for perfect precision. H-mean combines them so that a model has to
/// do well at both: it is close to the smaller of the two, so one bad number drags it down.
/// </para>
/// <example>
/// <code>
/// var metrics = new TextDetectionMetrics&lt;double&gt;();
/// var (precision, recall, hmean) = metrics.Evaluate(predictedRegionsPerImage, groundTruthRegionsPerImage);
/// </code>
/// </example>
/// </remarks>
/// <typeparam name="T">The numeric type the detected regions are expressed in.</typeparam>
public class TextDetectionMetrics<T> where T : struct
{
    /// <summary>
    /// Coordinates closer together than this are treated as coincident when intersecting edges.
    /// </summary>
    private const double GeometricTolerance = 1e-12;

    /// <summary>
    /// The numeric operations provider for type <typeparamref name="T"/>.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextDetectionMetrics{T}"/> class.
    /// </summary>
    public TextDetectionMetrics()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes the area of a simple polygon using the shoelace formula.
    /// </summary>
    /// <param name="polygon">The polygon vertices, in order. Fewer than three vertices enclose no area.</param>
    /// <returns>The absolute area, so the result does not depend on winding direction.</returns>
    public static double PolygonArea(IReadOnlyList<(double X, double Y)> polygon)
        => Math.Abs(SignedArea(polygon));

    /// <summary>
    /// Computes intersection-over-union between two polygons.
    /// </summary>
    /// <param name="first">The first polygon vertices, in order.</param>
    /// <param name="second">The second polygon vertices, in order. This one is used as the clipping
    /// polygon, so the result is exact when it is convex (see the class remarks).</param>
    /// <returns>IoU in [0, 1]. Returns 0 when either polygon is degenerate or they do not overlap.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    public static double PolygonIoU(
        IReadOnlyList<(double X, double Y)> first,
        IReadOnlyList<(double X, double Y)> second)
    {
        if (first is null)
        {
            throw new ArgumentNullException(nameof(first));
        }

        if (second is null)
        {
            throw new ArgumentNullException(nameof(second));
        }

        double areaFirst = PolygonArea(first);
        double areaSecond = PolygonArea(second);
        if (areaFirst <= 0.0 || areaSecond <= 0.0)
        {
            return 0.0;
        }

        // Sutherland-Hodgman requires both polygons wound the same way and the clip polygon
        // counter-clockwise, so the left-of-edge test means inside.
        var subject = EnsureCounterClockwise(first);
        var clip = EnsureCounterClockwise(second);

        double intersection = PolygonArea(ClipToConvex(subject, clip));
        double union = areaFirst + areaSecond - intersection;

        return union > 0.0 ? intersection / union : 0.0;
    }

    /// <summary>
    /// Evaluates detected text regions against ground truth using the ICDAR IoU protocol.
    /// </summary>
    /// <param name="predictions">Detected regions, one list per image. Confidence drives the matching
    /// order within each image.</param>
    /// <param name="groundTruth">Ground-truth regions, one list per image, aligned with
    /// <paramref name="predictions"/>.</param>
    /// <param name="iouThreshold">Minimum polygon IoU for a match. ICDAR uses 0.5.</param>
    /// <returns>Precision, recall and their harmonic mean, each in [0, 1]. Precision is 1 when nothing
    /// was predicted, recall is 1 when there is nothing to find, and H-mean is 0 when precision and
    /// recall are both 0.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The two lists describe a different number of images.</exception>
    public (double Precision, double Recall, double HMean) Evaluate(
        IReadOnlyList<IReadOnlyList<TextRegion<T>>> predictions,
        IReadOnlyList<IReadOnlyList<TextRegion<T>>> groundTruth,
        double iouThreshold = 0.5)
    {
        if (predictions is null)
        {
            throw new ArgumentNullException(nameof(predictions));
        }

        if (groundTruth is null)
        {
            throw new ArgumentNullException(nameof(groundTruth));
        }

        if (predictions.Count != groundTruth.Count)
        {
            throw new ArgumentException(
                $"Predictions cover {predictions.Count} images but ground truth covers {groundTruth.Count}. "
                + "Both lists must be indexed by the same image order.",
                nameof(predictions));
        }

        int matched = 0;
        int predictedCount = 0;
        int truthCount = 0;

        for (int i = 0; i < predictions.Count; i++)
        {
            var truthPolygons = ToPolygons(groundTruth[i]);
            var claimed = new bool[truthPolygons.Count];
            truthCount += truthPolygons.Count;

            // Highest confidence first, so when two predictions both cover a region the better one
            // claims it. OrderByDescending is stable, keeping equal-confidence order reproducible.
            var ordered = OrderByConfidenceDescending(predictions[i]);
            predictedCount += ordered.Count;

            foreach (var region in ordered)
            {
                var polygon = ToPolygon(region);
                if (polygon.Count < 3)
                {
                    continue; // Degenerate prediction: counted against precision, can match nothing.
                }

                double bestIoU = 0.0;
                int bestCandidate = -1;
                for (int c = 0; c < truthPolygons.Count; c++)
                {
                    if (claimed[c])
                    {
                        continue;
                    }

                    double iou = PolygonIoU(polygon, truthPolygons[c]);
                    if (iou > bestIoU)
                    {
                        bestIoU = iou;
                        bestCandidate = c;
                    }
                }

                if (bestCandidate >= 0 && bestIoU >= iouThreshold)
                {
                    claimed[bestCandidate] = true;
                    matched++;
                }
            }
        }

        double precision = predictedCount > 0 ? matched / (double)predictedCount : 1.0;
        double recall = truthCount > 0 ? matched / (double)truthCount : 1.0;
        double hmean = (precision + recall) > 0.0
            ? 2.0 * precision * recall / (precision + recall)
            : 0.0;

        return (precision, recall, hmean);
    }

    /// <summary>
    /// Converts a detected region to a double-precision polygon, falling back to the corners of its
    /// bounding box when no polygon was supplied.
    /// </summary>
    /// <param name="region">The region to convert.</param>
    /// <returns>The polygon vertices, or an empty list when the region carries neither polygon nor box.</returns>
    internal List<(double X, double Y)> ToPolygon(TextRegion<T> region)
    {
        var polygon = new List<(double X, double Y)>();
        if (region is null)
        {
            return polygon;
        }

        if (region.Polygon is not null && region.Polygon.Count >= 3)
        {
            foreach (var vertex in region.Polygon)
            {
                polygon.Add((_numOps.ToDouble(vertex.X), _numOps.ToDouble(vertex.Y)));
            }

            return polygon;
        }

        if (region.Box is not null)
        {
            var (xMin, yMin, xMax, yMax) = region.Box.ToXYXY();
            polygon.Add((xMin, yMin));
            polygon.Add((xMax, yMin));
            polygon.Add((xMax, yMax));
            polygon.Add((xMin, yMax));
        }

        return polygon;
    }

    private List<List<(double X, double Y)>> ToPolygons(IReadOnlyList<TextRegion<T>>? regions)
    {
        var polygons = new List<List<(double X, double Y)>>();
        if (regions is null)
        {
            return polygons;
        }

        foreach (var region in regions)
        {
            var polygon = ToPolygon(region);
            if (polygon.Count >= 3)
            {
                polygons.Add(polygon);
            }
        }

        return polygons;
    }

    private List<TextRegion<T>> OrderByConfidenceDescending(IReadOnlyList<TextRegion<T>>? regions)
    {
        var kept = new List<TextRegion<T>>();
        if (regions is null)
        {
            return kept;
        }

        foreach (var region in regions)
        {
            if (region is not null)
            {
                kept.Add(region);
            }
        }

        return kept.OrderByDescending(r => _numOps.ToDouble(r.Confidence)).ToList();
    }

    private static double SignedArea(IReadOnlyList<(double X, double Y)> polygon)
    {
        if (polygon is null || polygon.Count < 3)
        {
            return 0.0;
        }

        double sum = 0.0;
        for (int i = 0; i < polygon.Count; i++)
        {
            var current = polygon[i];
            var next = polygon[(i + 1) % polygon.Count];
            sum += (current.X * next.Y) - (next.X * current.Y);
        }

        return sum / 2.0;
    }

    private static List<(double X, double Y)> EnsureCounterClockwise(IReadOnlyList<(double X, double Y)> polygon)
    {
        var ordered = new List<(double X, double Y)>(polygon);
        if (SignedArea(ordered) < 0.0)
        {
            ordered.Reverse();
        }

        return ordered;
    }

    /// <summary>
    /// Sutherland-Hodgman polygon clipping: successively clips the subject polygon against each
    /// directed edge of the (convex, counter-clockwise) clip polygon.
    /// </summary>
    private static List<(double X, double Y)> ClipToConvex(
        List<(double X, double Y)> subject,
        List<(double X, double Y)> clip)
    {
        var output = new List<(double X, double Y)>(subject);

        for (int edge = 0; edge < clip.Count && output.Count > 0; edge++)
        {
            var edgeStart = clip[edge];
            var edgeEnd = clip[(edge + 1) % clip.Count];

            var input = output;
            output = new List<(double X, double Y)>(input.Count + 2);

            for (int i = 0; i < input.Count; i++)
            {
                var current = input[i];
                var previous = input[(i + input.Count - 1) % input.Count];

                bool currentInside = IsLeftOfOrOn(edgeStart, edgeEnd, current);
                bool previousInside = IsLeftOfOrOn(edgeStart, edgeEnd, previous);

                if (currentInside)
                {
                    if (!previousInside)
                    {
                        output.Add(LineIntersection(previous, current, edgeStart, edgeEnd));
                    }

                    output.Add(current);
                }
                else if (previousInside)
                {
                    output.Add(LineIntersection(previous, current, edgeStart, edgeEnd));
                }
            }
        }

        return output;
    }

    /// <summary>
    /// True when <paramref name="point"/> lies to the left of, or on, the directed edge
    /// <paramref name="edgeStart"/> to <paramref name="edgeEnd"/>. For a counter-clockwise polygon
    /// that is the inside half-plane.
    /// </summary>
    private static bool IsLeftOfOrOn(
        (double X, double Y) edgeStart,
        (double X, double Y) edgeEnd,
        (double X, double Y) point)
        => (((edgeEnd.X - edgeStart.X) * (point.Y - edgeStart.Y))
            - ((edgeEnd.Y - edgeStart.Y) * (point.X - edgeStart.X))) >= 0.0;

    private static (double X, double Y) LineIntersection(
        (double X, double Y) firstStart,
        (double X, double Y) firstEnd,
        (double X, double Y) secondStart,
        (double X, double Y) secondEnd)
    {
        double firstCross = (firstStart.X * firstEnd.Y) - (firstStart.Y * firstEnd.X);
        double secondCross = (secondStart.X * secondEnd.Y) - (secondStart.Y * secondEnd.X);

        double firstDx = firstStart.X - firstEnd.X;
        double firstDy = firstStart.Y - firstEnd.Y;
        double secondDx = secondStart.X - secondEnd.X;
        double secondDy = secondStart.Y - secondEnd.Y;

        double denominator = (firstDx * secondDy) - (firstDy * secondDx);
        if (Math.Abs(denominator) < GeometricTolerance)
        {
            // Parallel or coincident edges: the crossing is degenerate, so fall back to the endpoint
            // that the caller was about to emit anyway.
            return firstEnd;
        }

        double x = ((firstCross * secondDx) - (firstDx * secondCross)) / denominator;
        double y = ((firstCross * secondDy) - (firstDy * secondCross)) / denominator;
        return (x, y);
    }
}
