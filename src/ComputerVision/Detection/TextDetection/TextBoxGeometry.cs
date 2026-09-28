namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>
/// Minimum-area rectangles over text-component pixels, the box step that DBNet and CRAFT both finish with
/// (OpenCV's <c>minAreaRect</c> in the reference implementations).
/// </summary>
internal static class TextBoxGeometry
{
    /// <summary>
    /// The minimum-area rectangle enclosing <paramref name="points"/>, with each side moved outward by
    /// <paramref name="grow"/>(width, height), as four corners. Empty when the points span no area.
    /// </summary>
    internal static List<(double X, double Y)> MinAreaRectangle(
        IReadOnlyList<(int H, int W)> points, Func<double, double, double> grow)
    {
        var hull = ConvexHull(points.Select(p => ((double)p.W, (double)p.H)).Distinct().ToList());
        if (hull.Count < 3)
            return new List<(double X, double Y)>();

        // Rotating calipers: the minimum-area rectangle has one side collinear with a hull edge.
        double bestArea = double.MaxValue;
        (double cx, double cy, double ux, double uy, double halfW, double halfH) best = default;
        for (int i = 0; i < hull.Count; i++)
        {
            var a = hull[i];
            var b = hull[(i + 1) % hull.Count];
            double len = Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
            if (len == 0) continue;
            double ux = (b.X - a.X) / len, uy = (b.Y - a.Y) / len;
            double minU = double.MaxValue, maxU = double.MinValue, minV = double.MaxValue, maxV = double.MinValue;
            foreach (var p in hull)
            {
                double u = p.X * ux + p.Y * uy;
                double v = -p.X * uy + p.Y * ux;
                minU = Math.Min(minU, u); maxU = Math.Max(maxU, u);
                minV = Math.Min(minV, v); maxV = Math.Max(maxV, v);
            }
            double area = (maxU - minU) * (maxV - minV);
            if (area < bestArea)
            {
                bestArea = area;
                double cu = (minU + maxU) / 2, cv = (minV + maxV) / 2;
                best = (cu * ux - cv * uy, cu * uy + cv * ux, ux, uy, (maxU - minU) / 2, (maxV - minV) / 2);
            }
        }

        double offset = grow(2 * best.halfW, 2 * best.halfH);
        double hw = best.halfW + offset, hh = best.halfH + offset;
        var corners = new List<(double X, double Y)>(4);
        foreach (var (su, sv) in new[] { (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0) })
        {
            double du = su * hw, dv = sv * hh;
            corners.Add((best.cx + du * best.ux - dv * best.uy, best.cy + du * best.uy + dv * best.ux));
        }

        return corners;
    }

    // Andrew's monotone chain; the hull counter-clockwise, without the closing point.
    private static List<(double X, double Y)> ConvexHull(List<(double X, double Y)> points)
    {
        if (points.Count < 3)
            return points;
        var sorted = points.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        var hull = new List<(double X, double Y)>(2 * sorted.Count);
        static double Cross((double X, double Y) o, (double X, double Y) a, (double X, double Y) b)
            => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
        foreach (var p in sorted)
        {
            while (hull.Count >= 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        int lower = hull.Count + 1;
        for (int i = sorted.Count - 2; i >= 0; i--)
        {
            var p = sorted[i];
            while (hull.Count >= lower && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        hull.RemoveAt(hull.Count - 1);
        return hull;
    }
}