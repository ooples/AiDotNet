namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>
/// Polygon geometry for building text-detection training targets, shared by DBNet, EAST and CRAFT. Coordinates
/// are map pixels; pixel (x, y) is sampled at its centre (x + 0.5, y + 0.5).
/// </summary>
internal static class TextTargetGeometry
{
    /// <summary>Signed shoelace area; positive for counter-clockwise vertices in y-down image coordinates' mirror.</summary>
    public static double SignedArea(IReadOnlyList<(double X, double Y)> polygon)
    {
        double sum = 0;
        for (int i = 0; i < polygon.Count; i++)
        {
            var a = polygon[i];
            var b = polygon[(i + 1) % polygon.Count];
            sum += (a.X * b.Y) - (b.X * a.Y);
        }
        return sum / 2.0;
    }

    public static double Area(IReadOnlyList<(double X, double Y)> polygon) => Math.Abs(SignedArea(polygon));

    public static double Perimeter(IReadOnlyList<(double X, double Y)> polygon)
    {
        double sum = 0;
        for (int i = 0; i < polygon.Count; i++)
        {
            var a = polygon[i];
            var b = polygon[(i + 1) % polygon.Count];
            sum += Math.Sqrt(((b.X - a.X) * (b.X - a.X)) + ((b.Y - a.Y) * (b.Y - a.Y)));
        }
        return sum;
    }

    /// <summary>The polygon with every coordinate multiplied by (sx, sy).</summary>
    public static (double X, double Y)[] Scale(IReadOnlyList<(double X, double Y)> polygon, double sx, double sy)
        => polygon.Select(p => (p.X * sx, p.Y * sy)).ToArray();

    /// <summary>
    /// Moves every edge along its normal by <paramref name="distance"/>: inward for positive values, outward for
    /// negative. Adjacent offset edges are re-intersected. That is the Vatti clipping offset for convex
    /// polygons, the shapes text annotations are almost always given as. Returns null when the polygon
    /// collapses.
    /// </summary>
    public static (double X, double Y)[]? Offset(IReadOnlyList<(double X, double Y)> polygon, double distance)
    {
        int n = polygon.Count;
        if (n < 3) return null;
        // Orient so that "inward" is to the left of each edge.
        double orientation = SignedArea(polygon) >= 0 ? 1.0 : -1.0;
        var lines = new (double PX, double PY, double DX, double DY)[n];
        for (int i = 0; i < n; i++)
        {
            var a = polygon[i];
            var b = polygon[(i + 1) % n];
            double dx = b.X - a.X, dy = b.Y - a.Y, length = Math.Sqrt((dx * dx) + (dy * dy));
            if (length < 1e-12) return null;
            // Left normal (-dy, dx), which points inward for a positively oriented polygon.
            double nx = -dy / length * orientation, ny = dx / length * orientation;
            lines[i] = (a.X + (nx * distance), a.Y + (ny * distance), dx, dy);
        }
        var result = new (double X, double Y)[n];
        for (int i = 0; i < n; i++)
        {
            var l1 = lines[(i + n - 1) % n];
            var l2 = lines[i];
            double cross = (l1.DX * l2.DY) - (l1.DY * l2.DX);
            if (Math.Abs(cross) < 1e-12)
            {
                result[i] = (l2.PX, l2.PY);
                continue;
            }
            double t = (((l2.PX - l1.PX) * l2.DY) - ((l2.PY - l1.PY) * l2.DX)) / cross;
            result[i] = (l1.PX + (t * l1.DX), l1.PY + (t * l1.DY));
        }
        // An inward offset past the polygon's width turns edges inside out. The orientation sign cannot detect
        // it: a square shrunk past its half-width comes back rotated by 180 degrees, which keeps the sign. So
        // every offset edge must still point the way its original edge does.
        if (distance > 0)
        {
            if (Area(result) < 1e-9) return null;
            for (int i = 0; i < n; i++)
            {
                var (ax, ay) = result[i];
                var (bx, by) = result[(i + 1) % n];
                if (((bx - ax) * lines[i].DX) + ((by - ay) * lines[i].DY) <= 0) return null;
            }
        }
        return result;
    }

    /// <summary>The DBNet / PSENet shrink distance <c>D = A (1 - r^2) / L</c> (Liao et al. 2020, Eq. 9).</summary>
    public static double ShrinkDistance(IReadOnlyList<(double X, double Y)> polygon, double ratio)
    {
        double perimeter = Perimeter(polygon);
        return perimeter < 1e-12 ? 0 : Area(polygon) * (1 - (ratio * ratio)) / perimeter;
    }

    /// <summary>Whether (x, y) lies inside the polygon (even-odd rule).</summary>
    public static bool Contains(IReadOnlyList<(double X, double Y)> polygon, double x, double y)
    {
        bool inside = false;
        for (int i = 0, j = polygon.Count - 1; i < polygon.Count; j = i++)
        {
            var a = polygon[i];
            var b = polygon[j];
            if ((a.Y > y) != (b.Y > y) && x < ((b.X - a.X) * (y - a.Y) / (b.Y - a.Y)) + a.X)
                inside = !inside;
        }
        return inside;
    }

    /// <summary>Pixel bounds (inclusive) of the polygon, clipped to a <paramref name="width"/> x <paramref name="height"/> map.</summary>
    public static (int X0, int Y0, int X1, int Y1) Bounds(IReadOnlyList<(double X, double Y)> polygon, int width, int height)
    {
        int x0 = Math.Max(0, (int)Math.Floor(polygon.Min(p => p.X)));
        int y0 = Math.Max(0, (int)Math.Floor(polygon.Min(p => p.Y)));
        int x1 = Math.Min(width - 1, (int)Math.Ceiling(polygon.Max(p => p.X)));
        int y1 = Math.Min(height - 1, (int)Math.Ceiling(polygon.Max(p => p.Y)));
        return (x0, y0, x1, y1);
    }

    /// <summary>Sets every pixel whose centre lies inside the polygon to <paramref name="value"/>.</summary>
    public static void Fill(double[,] map, IReadOnlyList<(double X, double Y)> polygon, double value)
    {
        int height = map.GetLength(0), width = map.GetLength(1);
        var (x0, y0, x1, y1) = Bounds(polygon, width, height);
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
                if (Contains(polygon, x + 0.5, y + 0.5)) map[y, x] = value;
    }

    /// <summary>Distance from (x, y) to the segment a-b.</summary>
    public static double DistanceToSegment(double x, double y, (double X, double Y) a, (double X, double Y) b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y, lengthSquared = (dx * dx) + (dy * dy);
        double t = lengthSquared < 1e-12 ? 0 : Math.Max(0, Math.Min(1, (((x - a.X) * dx) + ((y - a.Y) * dy)) / lengthSquared));
        double px = a.X + (t * dx) - x, py = a.Y + (t * dy) - y;
        return Math.Sqrt((px * px) + (py * py));
    }

    /// <summary>Distance from (x, y) to the polygon's boundary.</summary>
    public static double DistanceToBoundary(IReadOnlyList<(double X, double Y)> polygon, double x, double y)
    {
        double best = double.PositiveInfinity;
        for (int i = 0; i < polygon.Count; i++)
            best = Math.Min(best, DistanceToSegment(x, y, polygon[i], polygon[(i + 1) % polygon.Count]));
        return best;
    }

    /// <summary>
    /// The minimum-area enclosing rectangle (rotating calipers over the convex hull). Corners are returned in
    /// order, starting from the corner with the smallest x + y.
    /// </summary>
    public static (double X, double Y)[] MinAreaRectangle(IReadOnlyList<(double X, double Y)> polygon)
    {
        var hull = ConvexHull(polygon);
        double bestArea = double.PositiveInfinity;
        (double X, double Y)[] best = hull.ToArray();
        for (int i = 0; i < hull.Count; i++)
        {
            var a = hull[i];
            var b = hull[(i + 1) % hull.Count];
            double dx = b.X - a.X, dy = b.Y - a.Y, length = Math.Sqrt((dx * dx) + (dy * dy));
            if (length < 1e-12) continue;
            double ux = dx / length, uy = dy / length; // edge direction
            double vx = -uy, vy = ux;                  // its normal
            double minU = double.PositiveInfinity, maxU = double.NegativeInfinity, minV = double.PositiveInfinity, maxV = double.NegativeInfinity;
            foreach (var p in hull)
            {
                double u = (p.X * ux) + (p.Y * uy), v = (p.X * vx) + (p.Y * vy);
                minU = Math.Min(minU, u); maxU = Math.Max(maxU, u); minV = Math.Min(minV, v); maxV = Math.Max(maxV, v);
            }
            double area = (maxU - minU) * (maxV - minV);
            if (area < bestArea)
            {
                bestArea = area;
                best = new[]
                {
                    ((minU * ux) + (minV * vx), (minU * uy) + (minV * vy)),
                    ((maxU * ux) + (minV * vx), (maxU * uy) + (minV * vy)),
                    ((maxU * ux) + (maxV * vx), (maxU * uy) + (maxV * vy)),
                    ((minU * ux) + (maxV * vx), (minU * uy) + (maxV * vy)),
                };
            }
        }
        int start = 0;
        for (int i = 1; i < best.Length; i++) if (best[i].X + best[i].Y < best[start].X + best[start].Y) start = i;
        return Enumerable.Range(0, best.Length).Select(i => best[(start + i) % best.Length]).ToArray();
    }

    private static List<(double X, double Y)> ConvexHull(IReadOnlyList<(double X, double Y)> points)
    {
        var sorted = points.Distinct().OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        if (sorted.Count < 3) return sorted;
        double Cross((double X, double Y) o, (double X, double Y) a, (double X, double Y) b)
            => ((a.X - o.X) * (b.Y - o.Y)) - ((a.Y - o.Y) * (b.X - o.X));
        var hull = new List<(double X, double Y)>();
        foreach (var pass in new[] { sorted, Enumerable.Reverse(sorted).ToList() })
        {
            int start = hull.Count;
            foreach (var p in pass)
            {
                while (hull.Count >= start + 2 && Cross(hull[^2], hull[^1], p) <= 0) hull.RemoveAt(hull.Count - 1);
                hull.Add(p);
            }
            hull.RemoveAt(hull.Count - 1);
        }
        return hull;
    }

    /// <summary>
    /// The homography taking the unit square (0,0),(1,0),(1,1),(0,1) to <paramref name="quad"/>'s four corners,
    /// inverted so it maps an image point back to its (u, v) in the quad. Null for a degenerate quad.
    /// </summary>
    public static Func<double, double, (double U, double V)>? InverseQuadMapping(IReadOnlyList<(double X, double Y)> quad)
    {
        if (quad.Count != 4) throw new ArgumentException("A quad has four corners.", nameof(quad));
        // Solve for H with H * (u, v, 1) ~ (x, y, 1) at the four corners (Heckbert 1989).
        var (x0, y0) = quad[0]; var (x1, y1) = quad[1]; var (x2, y2) = quad[2]; var (x3, y3) = quad[3];
        double dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
        double dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
        double a, b, c, d, e, f, g, h;
        if (Math.Abs(dx3) < 1e-12 && Math.Abs(dy3) < 1e-12)
        {
            a = x1 - x0; b = x2 - x1; c = x0; d = y1 - y0; e = y2 - y1; f = y0; g = 0; h = 0;
        }
        else
        {
            double den = (dx1 * dy2) - (dx2 * dy1);
            if (Math.Abs(den) < 1e-12) return null;
            g = ((dx3 * dy2) - (dx2 * dy3)) / den;
            h = ((dx1 * dy3) - (dx3 * dy1)) / den;
            a = x1 - x0 + (g * x1); b = x3 - x0 + (h * x3); c = x0;
            d = y1 - y0 + (g * y1); e = y3 - y0 + (h * y3); f = y0;
        }
        // Invert the 3x3 [[a,b,c],[d,e,f],[g,h,1]].
        double A = e - (f * h), B = (c * h) - b, C = (b * f) - (c * e);
        double D = (f * g) - d, E = a - (c * g), F = (c * d) - (a * f);
        double G = (d * h) - (e * g), Hh = (b * g) - (a * h), I = (a * e) - (b * d);
        if (Math.Abs(I) < 1e-12 && Math.Abs(G) < 1e-12 && Math.Abs(Hh) < 1e-12) return null;
        return (x, y) =>
        {
            double w = (G * x) + (Hh * y) + I;
            return (((A * x) + (B * y) + C) / w, ((D * x) + (E * y) + F) / w);
        };
    }

    /// <summary>
    /// Warps an isotropic 2-D Gaussian into <paramref name="quad"/> and writes the maximum of it and the map's
    /// current value into every pixel the quad covers. This is CRAFT's region and affinity score generation
    /// (Baek et al. 2019, Fig. 3): the peak is 1 at the quad's centre and decays towards its edges.
    /// </summary>
    public static void SplatGaussian(double[,] map, IReadOnlyList<(double X, double Y)> quad, double sigma = 0.25)
    {
        var inverse = InverseQuadMapping(quad);
        if (inverse is null) return;
        int height = map.GetLength(0), width = map.GetLength(1);
        var (x0, y0, x1, y1) = Bounds(quad, width, height);
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                var (u, v) = inverse(x + 0.5, y + 0.5);
                if (u < 0 || u > 1 || v < 0 || v > 1) continue;
                double value = Math.Exp(-(((u - 0.5) * (u - 0.5)) + ((v - 0.5) * (v - 0.5))) / (2 * sigma * sigma));
                if (value > map[y, x]) map[y, x] = value;
            }
    }
}
