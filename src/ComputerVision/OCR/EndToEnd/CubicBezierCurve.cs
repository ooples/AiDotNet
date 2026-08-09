using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ComputerVision.OCR.EndToEnd;

/// <summary>
/// A cubic Bezier curve over four control points, and the least-squares fit that recovers those
/// control points from annotated polygon vertices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the first of ABCNet's two contributions (Liu et al., CVPR 2020, arXiv:2002.10200): a text
/// instance's boundary is carried by TWO cubic Bezier curves, one along the top edge and one along the
/// bottom, so an arbitrarily curved instance is described by 8 control points — 16 numbers — instead of
/// a long polygon. That compactness is what lets the detector regress the shape directly, with only a
/// handful of extra output channels over an ordinary box head.
/// </para>
/// <para>
/// A cubic is used rather than a higher order or a spline, and that is a real constraint rather than an
/// arbitrary choice: the paper's own measurement is that a cubic already covers the overwhelming
/// majority of real curved text, and a higher order would add parameters the detector must regress
/// while making the fit less stable.
/// </para>
/// <para><b>For Beginners:</b> A Bezier curve is a smooth line whose shape is set by a few "control
/// points" that pull it around, like bending a wire with four handles. Describing curved text by two
/// such curves — one for its top edge, one for its bottom — is far more compact than listing dozens of
/// points along its outline.</para>
/// </remarks>
public readonly struct CubicBezierCurve<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The number of control points in a cubic Bezier curve.</summary>
    public const int ControlPointCount = 4;

    private readonly double[] _x;
    private readonly double[] _y;

    /// <summary>
    /// Creates a curve from its four control points, ordered from the curve's start to its end.
    /// </summary>
    /// <param name="controlPoints">Exactly four (x, y) control points.</param>
    public CubicBezierCurve(IReadOnlyList<(double X, double Y)> controlPoints)
    {
        if (controlPoints is null) throw new ArgumentNullException(nameof(controlPoints));
        if (controlPoints.Count != ControlPointCount)
        {
            throw new ArgumentException(
                $"A cubic Bezier curve needs exactly {ControlPointCount} control points; got {controlPoints.Count}.",
                nameof(controlPoints));
        }

        _x = new double[ControlPointCount];
        _y = new double[ControlPointCount];
        for (int i = 0; i < ControlPointCount; i++)
        {
            _x[i] = controlPoints[i].X;
            _y[i] = controlPoints[i].Y;
        }
    }

    /// <summary>Gets the control point at <paramref name="index"/>.</summary>
    public (double X, double Y) ControlPoint(int index)
    {
        if (_x is null) throw new InvalidOperationException("This curve was default-constructed and has no control points.");
        if (index < 0 || index >= ControlPointCount)
            throw new ArgumentOutOfRangeException(nameof(index), index, $"index must be in [0, {ControlPointCount - 1}].");
        return (_x[index], _y[index]);
    }

    /// <summary>
    /// The cubic Bernstein basis at <paramref name="t"/>:
    /// <c>[(1-t)^3, 3t(1-t)^2, 3t^2(1-t), t^3]</c>.
    /// </summary>
    /// <remarks>
    /// Exposed because it is what the least-squares fit builds its design matrix from, and because a
    /// basis that does not sum to 1 at every t is the single most likely way this gets silently wrong —
    /// which is directly checkable against this method.
    /// </remarks>
    public static double[] BernsteinBasis(double t)
    {
        double u = 1.0 - t;
        return new[] { u * u * u, 3.0 * t * u * u, 3.0 * t * t * u, t * t * t };
    }

    /// <summary>
    /// Evaluates the curve at <paramref name="t"/> in <c>[0, 1]</c>.
    /// </summary>
    public (double X, double Y) Evaluate(double t)
    {
        if (_x is null) throw new InvalidOperationException("This curve was default-constructed and has no control points.");
        if (t is < 0.0 or > 1.0)
            throw new ArgumentOutOfRangeException(nameof(t), t, "t must be in [0, 1].");

        var b = BernsteinBasis(t);
        double x = 0.0, y = 0.0;
        for (int i = 0; i < ControlPointCount; i++)
        {
            x += b[i] * _x[i];
            y += b[i] * _y[i];
        }
        return (x, y);
    }

    /// <summary>Evaluates the curve and converts the result to <typeparamref name="T"/>.</summary>
    public (T X, T Y) EvaluateNumeric(double t)
    {
        var (x, y) = Evaluate(t);
        return (NumOps.FromDouble(x), NumOps.FromDouble(y));
    }

    /// <summary>
    /// Fits a cubic Bezier curve to <paramref name="points"/> by least squares.
    /// </summary>
    /// <param name="points">
    /// At least two points, in order along the intended curve — one edge of an annotated text polygon.
    /// </param>
    /// <returns>The fitted curve.</returns>
    /// <remarks>
    /// <para>
    /// Parameterization is by CUMULATIVE CHORD LENGTH, not by uniform index spacing. Uniform spacing is
    /// the tempting simplification and it distorts every unevenly-annotated polygon: real annotations
    /// bunch vertices where the text curves and spread them where it is straight, so treating the
    /// vertices as equally spaced in t pulls the fitted curve toward the densely-annotated stretch.
    /// </para>
    /// <para>
    /// The fit is UNCONSTRAINED — the end control points are solved for rather than pinned to the first
    /// and last input points. Pinning them is a common shortcut that forces the curve through two
    /// possibly-noisy annotation vertices and pushes that error into the interior control points, which
    /// are what the detector has to regress.
    /// </para>
    /// <para>
    /// WHAT THIS DOES NOT PROMISE, because chord length is a heuristic for the unknown true parameter:
    /// feeding back points sampled from a known cubic at uniform <c>t</c> does NOT return that cubic's
    /// control points. For a curved cubic, arc length is not proportional to <c>t</c>, so the chord-length
    /// parameters differ from the ones the samples were generated at and the recovered control points
    /// differ correspondingly (a few percent in practice). What IS guaranteed is the thing that matters
    /// here: the curve passes close to the supplied points, and the result depends on the boundary's
    /// GEOMETRY rather than on how densely it happened to be annotated. Exact recovery does hold where
    /// chord length is proportional to <c>t</c> — a straight edge, which is the common rectangular case.
    /// </para>
    /// <para>
    /// Fewer than four points cannot determine four control points, so the normal equations are
    /// singular. Rather than fail, the system is solved in a least-norm sense by falling back to a
    /// degree-raised interpretation of the available points, which is exact for the 2-point (straight
    /// segment) case that a rectangular text instance produces.
    /// </para>
    /// </remarks>
    public static CubicBezierCurve<T> Fit(IReadOnlyList<(double X, double Y)> points)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        if (points.Count < 2)
            throw new ArgumentException("At least two points are needed to fit a curve.", nameof(points));

        var t = ChordLengthParameters(points);

        // Normal equations: (M^T M) P = M^T Q, with M the m x 4 Bernstein design matrix.
        var ata = new double[ControlPointCount, ControlPointCount];
        var atqx = new double[ControlPointCount];
        var atqy = new double[ControlPointCount];

        for (int i = 0; i < points.Count; i++)
        {
            var b = BernsteinBasis(t[i]);
            for (int r = 0; r < ControlPointCount; r++)
            {
                for (int c = 0; c < ControlPointCount; c++) ata[r, c] += b[r] * b[c];
                atqx[r] += b[r] * points[i].X;
                atqy[r] += b[r] * points[i].Y;
            }
        }

        // With fewer than 4 distinct parameters the normal matrix is rank-deficient. A tiny Tikhonov
        // ridge toward the straight-line control points keeps it solvable while leaving the
        // well-determined case numerically unchanged (the ridge is 1e-9 against diagonal entries of
        // order 1). It is NOT a general regularizer — it exists only to make the degenerate case
        // produce the straight segment those points actually describe.
        var (sx, sy) = (points[0].X, points[0].Y);
        var (ex, ey) = (points[points.Count - 1].X, points[points.Count - 1].Y);
        const double ridge = 1e-9;
        for (int r = 0; r < ControlPointCount; r++)
        {
            double frac = r / (double)(ControlPointCount - 1);
            ata[r, r] += ridge;
            atqx[r] += ridge * (sx + ((ex - sx) * frac));
            atqy[r] += ridge * (sy + ((ey - sy) * frac));
        }

        double[] px = SolveFourByFour(ata, atqx);
        double[] py = SolveFourByFour(ata, atqy);

        var control = new (double X, double Y)[ControlPointCount];
        for (int i = 0; i < ControlPointCount; i++) control[i] = (px[i], py[i]);
        return new CubicBezierCurve<T>(control);
    }

    /// <summary>
    /// Cumulative chord-length parameters in <c>[0, 1]</c>, one per input point.
    /// </summary>
    private static double[] ChordLengthParameters(IReadOnlyList<(double X, double Y)> points)
    {
        var cumulative = new double[points.Count];
        cumulative[0] = 0.0;
        for (int i = 1; i < points.Count; i++)
        {
            double dx = points[i].X - points[i - 1].X;
            double dy = points[i].Y - points[i - 1].Y;
            cumulative[i] = cumulative[i - 1] + Math.Sqrt((dx * dx) + (dy * dy));
        }

        double total = cumulative[points.Count - 1];
        var t = new double[points.Count];
        if (total <= 0.0)
        {
            // Every point coincides, so chord length carries no ordering information. Uniform spacing
            // is the only defensible fallback here — and unlike the general case it distorts nothing,
            // because all the points are at the same place.
            for (int i = 0; i < points.Count; i++) t[i] = i / (double)(points.Count - 1);
            return t;
        }

        for (int i = 0; i < points.Count; i++) t[i] = cumulative[i] / total;
        return t;
    }

    /// <summary>
    /// Solves a 4x4 system by Gaussian elimination with partial pivoting.
    /// </summary>
    /// <remarks>
    /// Written out rather than routed through a general solver because the size is fixed at four and
    /// partial pivoting is what keeps the fit stable when a text instance is nearly straight — the case
    /// where the interior basis columns become close to collinear.
    /// </remarks>
    private static double[] SolveFourByFour(double[,] a, double[] b)
    {
        const int n = ControlPointCount;
        var m = new double[n, n + 1];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++) m[r, c] = a[r, c];
            m[r, n] = b[r];
        }

        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
                if (Math.Abs(m[r, col]) > Math.Abs(m[pivot, col])) pivot = r;

            if (Math.Abs(m[pivot, col]) < 1e-300)
            {
                throw new InvalidOperationException(
                    "The Bezier fit's normal equations are singular even after regularization; the "
                    + "input points do not determine a curve.");
            }

            if (pivot != col)
            {
                for (int c = col; c <= n; c++) (m[col, c], m[pivot, c]) = (m[pivot, c], m[col, c]);
            }

            for (int r = col + 1; r < n; r++)
            {
                // No exact-zero shortcut on `factor`: a magnitude test cannot stand in for one
                // (a 1e-300 factor against a 1e300 pivot row still contributes), and at 4x4 the
                // skip saves five multiply-subtracts. Always eliminate.
                //
                // RE-APPLIED: this fix landed in c52dc3e9d and the #1995 merge reverted it.
                double factor = m[r, col] / m[col, col];
                for (int c = col; c <= n; c++) m[r, c] -= factor * m[col, c];
            }
        }

        var x = new double[n];
        for (int r = n - 1; r >= 0; r--)
        {
            double sum = m[r, n];
            for (int c = r + 1; c < n; c++) sum -= m[r, c] * x[c];
            x[r] = sum / m[r, r];
        }
        return x;
    }
}
