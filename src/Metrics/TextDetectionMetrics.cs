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
/// // One image: a ground-truth text region and a predicted region that overlaps it.
/// var groundTruthRegionsPerImage = new List&lt;IReadOnlyList&lt;AiDotNet.ComputerVision.Detection.TextDetection.TextRegion&lt;double&gt;&gt;&gt;
/// {
///     new[] { new AiDotNet.ComputerVision.Detection.TextDetection.TextRegion&lt;double&gt;(
///         new AiDotNet.Augmentation.Image.BoundingBox&lt;double&gt;(0, 0, 40, 12), 1.0) }
/// };
/// var predictedRegionsPerImage = new List&lt;IReadOnlyList&lt;AiDotNet.ComputerVision.Detection.TextDetection.TextRegion&lt;double&gt;&gt;&gt;
/// {
///     new[] { new AiDotNet.ComputerVision.Detection.TextDetection.TextRegion&lt;double&gt;(
///         new AiDotNet.Augmentation.Image.BoundingBox&lt;double&gt;(1, 0, 40, 12), 0.9) }
/// };
///
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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the <see cref="TextDetectionMetrics{T}"/> class.
    /// </summary>
    public TextDetectionMetrics()
    {
    }

    /// <summary>
    /// Computes the area of a simple polygon using the shoelace formula.
    /// </summary>
    /// <param name="polygon">The polygon vertices, in order. Fewer than three vertices enclose no area.</param>
    /// <returns>The absolute area, so the result does not depend on winding direction.</returns>
    public static T PolygonArea(IReadOnlyList<(T X, T Y)> polygon)
        => NumOps.Abs(SignedArea(polygon));

    /// <summary>
    /// Computes intersection-over-union between two polygons.
    /// </summary>
    /// <param name="first">The first polygon vertices, in order.</param>
    /// <param name="second">The second polygon vertices, in order. This one is used as the clipping
    /// polygon, so the result is exact when it is convex (see the class remarks).</param>
    /// <returns>IoU in [0, 1]. Returns 0 when either polygon is degenerate or they do not overlap.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    public static T PolygonIoU(
        IReadOnlyList<(T X, T Y)> first,
        IReadOnlyList<(T X, T Y)> second)
    {
        if (first is null)
        {
            throw new ArgumentNullException(nameof(first));
        }

        if (second is null)
        {
            throw new ArgumentNullException(nameof(second));
        }

        if (typeof(T) != typeof(double))
            return NumOps.FromDouble(TextDetectionMetrics<double>.PolygonIoU(WidePolygon(first), WidePolygon(second)));

        T areaFirst = PolygonArea(first);
        T areaSecond = PolygonArea(second);
        if (NumOps.LessThanOrEquals(areaFirst, NumOps.Zero) || NumOps.LessThanOrEquals(areaSecond, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        // Sutherland-Hodgman requires both polygons wound the same way and the clip polygon
        // counter-clockwise, so the left-of-edge test means inside.
        var subject = EnsureCounterClockwise(first);
        var clip = EnsureCounterClockwise(second);

        T intersection = PolygonArea(ClipToConvex(subject, clip));
        T union = NumOps.Subtract(NumOps.Add(areaFirst, areaSecond), intersection);

        return NumOps.GreaterThan(union, NumOps.Zero) ? NumOps.Divide(intersection, union) : NumOps.Zero;
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
    public (T Precision, T Recall, T HMean) Evaluate(
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
        T threshold = NumOps.FromDouble(iouThreshold);

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

                if (TryMatch(polygon, truthPolygons, claimed, threshold))
                {
                    matched++;
                }
            }
        }

        return Summarize(matched, predictedCount, truthCount);
    }

    /// <summary>
    /// Claims the unclaimed ground-truth polygon with the highest IoU against
    /// <paramref name="polygon"/> (first one on ties) when that IoU reaches the threshold.
    /// </summary>
    private static bool TryMatch(
        List<(T X, T Y)> polygon, List<List<(T X, T Y)>> truthPolygons, bool[] claimed, T threshold)
    {
        T bestIoU = NumOps.Zero;
        int bestCandidate = -1;
        for (int c = 0; c < truthPolygons.Count; c++)
        {
            if (claimed[c])
            {
                continue;
            }

            T iou = PolygonIoU(polygon, truthPolygons[c]);
            if (IsGreater(iou, bestIoU))
            {
                bestIoU = iou;
                bestCandidate = c;
            }
        }

        if (bestCandidate >= 0 && NumOps.GreaterThanOrEquals(bestIoU, threshold))
        {
            claimed[bestCandidate] = true;
            return true;
        }

        return false;
    }

    /// <summary>
    /// a &gt; b in <typeparamref name="T"/>.
    /// </summary>
    private static bool IsGreater(T a, T b) => NumOps.GreaterThan(a, b);

    private static (double X, double Y)[] WidePolygon(IReadOnlyList<(T X, T Y)> polygon)
    {
        var xs = new Vector<T>(polygon.Select(point => point.X)).AsTensor().Cast<double>().ToArray();
        var ys = new Vector<T>(polygon.Select(point => point.Y)).AsTensor().Cast<double>().ToArray();
        return xs.Zip(ys, (x, y) => (x, y)).ToArray();
    }

    /// <summary>
    /// Precision (1 when nothing was predicted), recall (1 when nothing was to be found) and their
    /// harmonic mean (0 when both are 0).
    /// </summary>
    private static (T Precision, T Recall, T HMean) Summarize(int matched, int predictedCount, int truthCount)
    {
        double precision = predictedCount > 0 ? (double)matched / predictedCount : 1.0;
        double recall = truthCount > 0 ? (double)matched / truthCount : 1.0;
        double sum = precision + recall;
        double hmean = sum > 0 ? 2.0 * precision * recall / sum : 0.0;
        return (NumOps.FromDouble(precision), NumOps.FromDouble(recall), NumOps.FromDouble(hmean));
    }

    /// <summary>
    /// Converts a detected region to a polygon, falling back to the corners of its bounding box when no
    /// polygon was supplied.
    /// </summary>
    /// <param name="region">The region to convert.</param>
    /// <returns>The polygon vertices, or an empty list when the region carries neither polygon nor box.</returns>
    internal List<(T X, T Y)> ToPolygon(TextRegion<T> region)
    {
        if (region is null)
        {
            return new List<(T X, T Y)>();
        }

        if (region.Polygon is not null && region.Polygon.Count >= 3)
        {
            return new List<(T X, T Y)>(region.Polygon);
        }

        var polygon = new List<(T X, T Y)>();
        if (region.Box is not null)
        {
            var (xMin, yMin, xMax, yMax) = region.Box.ToXYXY();
            T left = NumOps.FromDouble(xMin);
            T top = NumOps.FromDouble(yMin);
            T right = NumOps.FromDouble(xMax);
            T bottom = NumOps.FromDouble(yMax);
            polygon.Add((left, top));
            polygon.Add((right, top));
            polygon.Add((right, bottom));
            polygon.Add((left, bottom));
        }

        return polygon;
    }

    private List<List<(T X, T Y)>> ToPolygons(IReadOnlyList<TextRegion<T>>? regions)
    {
        var polygons = new List<List<(T X, T Y)>>();
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

        return kept.OrderByDescending(r => r.Confidence, Comparer<T>.Create(NumOps.Compare)).ToList();
    }

    /// <summary>
    /// Shoelace sum over consecutive vertex pairs: (Σ x_i y_(i+1) - x_(i+1) y_i) / 2.
    /// </summary>
    private static T SignedArea(IReadOnlyList<(T X, T Y)> polygon)
    {
        if (polygon is null || polygon.Count < 3)
        {
            return NumOps.Zero;
        }

        var engine = AiDotNetEngine.Current;

        // Translate so the first vertex is the origin: the area is unchanged, and the products stay at the
        // polygon's own scale instead of its image position (4097 x 4097 does not fit a float exactly).
        var origin = polygon[0];
        var next = polygon.Skip(1).Concat(polygon.Take(1)).ToList();
        var originX = Vector<T>.CreateDefault(polygon.Count, origin.X);
        var originY = Vector<T>.CreateDefault(polygon.Count, origin.Y);
        var x = engine.Subtract(new Vector<T>(polygon.Select(p => p.X)), originX);
        var y = engine.Subtract(new Vector<T>(polygon.Select(p => p.Y)), originY);
        var nextX = engine.Subtract(new Vector<T>(next.Select(p => p.X)), originX);
        var nextY = engine.Subtract(new Vector<T>(next.Select(p => p.Y)), originY);
        var cross = engine.Subtract(engine.Multiply(x, nextY), engine.Multiply(nextX, y));

        return NumOps.Divide(engine.Sum(cross), NumOps.FromDouble(2.0));
    }

    private static List<(T X, T Y)> EnsureCounterClockwise(IReadOnlyList<(T X, T Y)> polygon)
    {
        var ordered = new List<(T X, T Y)>(polygon);
        if (NumOps.LessThan(SignedArea(ordered), NumOps.Zero))
        {
            ordered.Reverse();
        }

        return ordered;
    }

    /// <summary>
    /// Sutherland-Hodgman polygon clipping: successively clips the subject polygon against each
    /// directed edge of the (convex, counter-clockwise) clip polygon.
    /// </summary>
    private static List<(T X, T Y)> ClipToConvex(
        List<(T X, T Y)> subject,
        List<(T X, T Y)> clip)
    {
        var output = new List<(T X, T Y)>(subject);

        for (int edge = 0; edge < clip.Count && output.Count > 0; edge++)
        {
            var edgeStart = clip[edge];
            var edgeEnd = clip[(edge + 1) % clip.Count];

            var input = output;
            output = new List<(T X, T Y)>(input.Count + 2);

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
        (T X, T Y) edgeStart,
        (T X, T Y) edgeEnd,
        (T X, T Y) point)
    {
        T cross = NumOps.Subtract(
            NumOps.Multiply(NumOps.Subtract(edgeEnd.X, edgeStart.X), NumOps.Subtract(point.Y, edgeStart.Y)),
            NumOps.Multiply(NumOps.Subtract(edgeEnd.Y, edgeStart.Y), NumOps.Subtract(point.X, edgeStart.X)));
        return NumOps.GreaterThanOrEquals(cross, NumOps.Zero);
    }

    /// <summary>
    /// Intersection of the line through <paramref name="firstStart"/> and <paramref name="firstEnd"/> with
    /// the line through <paramref name="secondStart"/> and <paramref name="secondEnd"/>, formed from
    /// coordinates relative to <paramref name="firstStart"/>: firstStart + t (firstEnd - firstStart) with
    /// t = cross(secondStart - firstStart, d2) / cross(d1, d2). Absolute-coordinate determinants lose the
    /// fractional part at image-scale coordinates in float.
    /// </summary>
    private static (T X, T Y) LineIntersection(
        (T X, T Y) firstStart,
        (T X, T Y) firstEnd,
        (T X, T Y) secondStart,
        (T X, T Y) secondEnd)
    {
        T firstDx = NumOps.Subtract(firstEnd.X, firstStart.X);
        T firstDy = NumOps.Subtract(firstEnd.Y, firstStart.Y);
        T secondDx = NumOps.Subtract(secondEnd.X, secondStart.X);
        T secondDy = NumOps.Subtract(secondEnd.Y, secondStart.Y);
        T offsetX = NumOps.Subtract(secondStart.X, firstStart.X);
        T offsetY = NumOps.Subtract(secondStart.Y, firstStart.Y);

        T denominator = NumOps.Subtract(NumOps.Multiply(firstDx, secondDy), NumOps.Multiply(firstDy, secondDx));
        if (NumOps.LessThan(NumOps.Abs(denominator), NumOps.FromDouble(GeometricTolerance)))
        {
            // Parallel or coincident edges: the crossing is degenerate, so fall back to the endpoint
            // that the caller was about to emit anyway.
            return firstEnd;
        }

        T t = NumOps.Divide(
            NumOps.Subtract(NumOps.Multiply(offsetX, secondDy), NumOps.Multiply(offsetY, secondDx)), denominator);
        return (NumOps.Add(firstStart.X, NumOps.Multiply(t, firstDx)), NumOps.Add(firstStart.Y, NumOps.Multiply(t, firstDy)));
    }
}
