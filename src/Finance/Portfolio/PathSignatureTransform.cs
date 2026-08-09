using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// Truncated path signatures for SIT (Hwang and Zohren, "Signature-Informed Transformer for Asset
/// Allocation", arXiv:2510.03129), including the pairwise signed area that measures lead-lag.
/// </summary>
/// <remarks>
/// <para>
/// The signature of a path is the collection of its iterated integrals,
/// <c>Sig(X) = (1, ∫dX, ∫∫dX⊗dX, ...)</c>, truncated at level M. It is invariant to
/// reparameterization of time, which is what makes it robust to irregular sampling: two paths that
/// trace the same shape at different speeds have the same signature.
/// </para>
/// <para>
/// <b>Level 2 is not a tuning choice.</b> The second-order terms of a joint signature over two asset
/// paths encode their SIGNED AREA (the Lévy area), and that signed area is the lead-lag measure the
/// whole architecture is built around — a strict lead-lag relationship implies a strictly positive
/// second-order cross term. A level-1 signature is just the total increment, which carries no
/// interaction information at all, so truncating lower removes the model's entire inductive bias.
/// The paper fixes the level at 2 rather than searching it.
/// </para>
/// <para>
/// The signed area is ANTI-symmetric: <c>A(j,l) = -A(l,j)</c>. That is precisely why it can express
/// lead-lag when a correlation cannot — correlation is symmetric and so cannot say which asset moved
/// first. A pair moving in exact lockstep traces a straight line and encloses zero area.
/// </para>
/// <para><b>For Beginners:</b> This turns a stretch of price history into a small set of numbers
/// describing its shape. For a PAIR of assets it also measures who tends to move first, which a
/// correlation cannot tell you.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class PathSignatureTransform<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The paper's truncation level. Fixed at 2, not searched.</summary>
    public const int PaperTruncationLevel = 2;

    private readonly int _level;

    /// <summary>Gets the truncation level M.</summary>
    public int Level => _level;

    /// <summary>Creates a signature transform.</summary>
    /// <param name="level">Truncation level M. Paper: 2.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="level"/> is below 1 or above 2. Levels above 2 are not implemented: the
    /// feature count grows exponentially in the level and the architecture only consumes up to the
    /// second-order signed area, so a higher level would cost dimensionality for features nothing
    /// reads.
    /// </exception>
    public PathSignatureTransform(int level = PaperTruncationLevel)
    {
        if (level is < 1 or > 2)
            throw new ArgumentOutOfRangeException(nameof(level), level,
                "Only truncation levels 1 and 2 are supported; the paper uses 2.");
        _level = level;
    }

    /// <summary>
    /// Number of signature terms produced by <see cref="Signature"/> for a path of the given
    /// dimension, EXCLUDING the constant leading 1.
    /// </summary>
    /// <remarks>
    /// Level 1 contributes d increments; level 2 contributes the d*d matrix of second-order terms.
    /// This is the exponential growth in the level that motivates truncating at 2.
    /// </remarks>
    public int FeatureCount(int pathDimension)
    {
        if (pathDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(pathDimension), pathDimension,
                "pathDimension must be positive.");

        return _level == 1 ? pathDimension : pathDimension + (pathDimension * pathDimension);
    }

    /// <summary>
    /// Truncated signature of a discretely sampled path.
    /// </summary>
    /// <param name="path">
    /// The path, shaped <c>[steps, dimension]</c>. Increments are taken between consecutive rows.
    /// </param>
    /// <returns>
    /// The signature terms without the leading constant 1: the <c>d</c> level-1 increments, followed
    /// (when <see cref="Level"/> is 2) by the <c>d*d</c> level-2 terms in row-major order, so entry
    /// <c>d + i*d + j</c> is the iterated integral of coordinate i then coordinate j.
    /// </returns>
    /// <remarks>
    /// Level-2 terms use the standard left-point (Riemann-Stieltjes) discretization
    /// <c>S_ij = sum_k ( X_i(k) - X_i(0) ) * dX_j(k) + 0.5 * dX_i(k) * dX_j(k)</c>.
    /// The half-increment correction is what makes <c>S_ij + S_ji = dX_i * dX_j</c> hold exactly, the
    /// shuffle identity the signature must satisfy; dropping it leaves the antisymmetric part correct
    /// but breaks the symmetric part.
    /// </remarks>
    public Vector<T> Signature(Tensor<T> path)
    {
        if (path == null) throw new ArgumentNullException(nameof(path));
        if (path.Shape.Length != 2)
            throw new ArgumentException(
                $"Path must be [steps, dimension]; got rank {path.Shape.Length}.", nameof(path));

        int steps = path.Shape[0];
        int dim = path.Shape[1];
        if (dim <= 0)
            throw new ArgumentException("Path dimension must be positive.", nameof(path));

        var result = new Vector<T>(FeatureCount(dim));

        // A single sample has no increments, so every term is zero rather than undefined.
        if (steps < 2) return result;

        var total = new double[dim];
        var running = new double[dim];
        var second = _level == 2 ? new double[dim * dim] : Array.Empty<double>();

        // The step's increments are computed ONCE per step and reused. The inner j loop previously
        // recomputed dj for every i, so each step made 2 * dim^2 NumOps.ToDouble calls where 2 * dim
        // suffice. The transform runs per window over an asset panel, so this is a hot path.
        var increment = new double[dim];

        for (int k = 1; k < steps; k++)
        {
            for (int i = 0; i < dim; i++)
                increment[i] = NumOps.ToDouble(path[(k * dim) + i]) - NumOps.ToDouble(path[((k - 1) * dim) + i]);

            for (int i = 0; i < dim; i++)
            {
                double di = increment[i];

                if (_level == 2)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        double dj = increment[j];
                        // running[i] is X_i at the left endpoint, relative to the path's start.
                        second[(i * dim) + j] += (running[i] * dj) + (0.5 * di * dj);
                    }
                }

                total[i] += di;
            }

            for (int i = 0; i < dim; i++)
                running[i] = total[i];
        }

        for (int i = 0; i < dim; i++) result[i] = NumOps.FromDouble(total[i]);
        for (int i = 0; i < second.Length; i++) result[dim + i] = NumOps.FromDouble(second[i]);
        return result;
    }

    /// <summary>
    /// Signed (Lévy) area between two scalar paths: the lead-lag measure.
    /// <c>A = 0.5 * sum_k ( x_k * dy_k - y_k * dx_k )</c>, equal to
    /// <c>0.5 * (S_xy - S_yx)</c> from the level-2 signature of the joint path.
    /// </summary>
    /// <param name="first">First path, one value per time step.</param>
    /// <param name="second">Second path, same length.</param>
    /// <remarks>
    /// <para>
    /// Positive area means the first path tends to lead the second. This is ANTI-symmetric —
    /// <c>Area(x, y) = -Area(y, x)</c> — which is exactly the property a correlation lacks and the
    /// reason a signed area can express lead-lag at all.
    /// </para>
    /// <para>
    /// Two paths moving in exact lockstep trace a straight line in the plane and enclose zero area,
    /// so perfectly synchronized assets correctly register no lead-lag however strongly correlated
    /// they are.
    /// </para>
    /// </remarks>
    public double SignedArea(Vector<T> first, Vector<T> second)
    {
        if (first == null) throw new ArgumentNullException(nameof(first));
        if (second == null) throw new ArgumentNullException(nameof(second));
        if (first.Length != second.Length)
            throw new ArgumentException(
                $"Paths must have equal length; got {first.Length} and {second.Length}.", nameof(second));

        if (first.Length < 2) return 0.0;

        double x0 = NumOps.ToDouble(first[0]);
        double y0 = NumOps.ToDouble(second[0]);
        double area = 0.0;

        for (int k = 1; k < first.Length; k++)
        {
            // Coordinates relative to the start, so the area measures the shape of the path rather
            // than its absolute position. An absolute-coordinate version would report a spurious
            // area for a pair that merely sits far from the origin.
            double x = NumOps.ToDouble(first[k - 1]) - x0;
            double y = NumOps.ToDouble(second[k - 1]) - y0;
            double dx = NumOps.ToDouble(first[k]) - NumOps.ToDouble(first[k - 1]);
            double dy = NumOps.ToDouble(second[k]) - NumOps.ToDouble(second[k - 1]);

            area += 0.5 * ((x * dy) - (y * dx));
        }

        return area;
    }

    /// <summary>
    /// Pairwise signed-area matrix over a panel of asset paths: entry (j, l) is the signed area of
    /// asset j against asset l.
    /// </summary>
    /// <param name="panel">Asset paths, shaped <c>[steps, assets]</c>.</param>
    /// <returns>An <c>[assets, assets]</c> antisymmetric matrix with a zero diagonal.</returns>
    /// <remarks>
    /// The diagonal is exactly zero because an asset cannot lead itself, and the matrix is filled
    /// antisymmetrically from one triangle so that property holds by construction rather than by
    /// accumulated luck.
    /// </remarks>
    public Tensor<T> CrossSignedAreas(Tensor<T> panel)
    {
        if (panel == null) throw new ArgumentNullException(nameof(panel));
        if (panel.Shape.Length != 2)
            throw new ArgumentException(
                $"Panel must be [steps, assets]; got rank {panel.Shape.Length}.", nameof(panel));

        int steps = panel.Shape[0];
        int assets = panel.Shape[1];
        var result = new Tensor<T>(new[] { assets, assets });

        var columns = new Vector<T>[assets];
        for (int a = 0; a < assets; a++)
        {
            var column = new Vector<T>(steps);
            for (int k = 0; k < steps; k++) column[k] = panel[(k * assets) + a];
            columns[a] = column;
        }

        for (int j = 0; j < assets; j++)
        {
            for (int l = j + 1; l < assets; l++)
            {
                double area = SignedArea(columns[j], columns[l]);
                result[(j * assets) + l] = NumOps.FromDouble(area);
                result[(l * assets) + j] = NumOps.FromDouble(-area);
            }
        }

        return result;
    }
}
