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

        // Increments D[k] = X[k + 1] - X[k], k in [0, steps - 1). The level-1 term is their sum; the level-2 term is
        // sum_k R[k, i] D[k, j] + 0.5 D[k, i] D[k, j] = ((R + D / 2)^T D)[i, j], where R[k] = X[k] - X[0] is the
        // running total before step k (the exclusive cumulative sum of D). Detached, like the vector it fills.
        var engine = AiDotNetEngine.Current;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var widePath = path.Cast<double>();
        var increments = engine.TensorSubtract(
            engine.TensorNarrow(widePath, 0, 1, steps - 1), engine.TensorNarrow(widePath, 0, 0, steps - 1));
        var total = engine.ReduceSum(increments, new[] { 0 }, false);                         // [dim]
        Tensor<double> terms = total;
        if (_level == 2)
        {
            // Shift the cumulative sum rather than subtracting the current increment, which can cancel a small prefix.
            var before = steps == 2
                ? new Tensor<double>([1, dim])
                : engine.TensorConcatenate(new[] { new Tensor<double>([1, dim]),
                    engine.TensorNarrow(engine.TensorCumSum(increments, 0), 0, 0, steps - 2) }, 0);
            var left = engine.TensorAdd(before, engine.TensorMultiplyScalar(increments, 0.5));
            var second = engine.TensorMatMul(engine.TensorTranspose(left), increments);       // [dim, dim]
            terms = engine.TensorConcatenate(new[] { total, engine.Reshape(second, new[] { dim * dim }) }, 0);
        }

        terms.Cast<T>().AsSpan().CopyTo(result.AsWritableSpan());
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
    public T SignedArea(Vector<T> first, Vector<T> second)
    {
        if (first == null) throw new ArgumentNullException(nameof(first));
        if (second == null) throw new ArgumentNullException(nameof(second));
        if (first.Length != second.Length)
            throw new ArgumentException(
                $"Paths must have equal length; got {first.Length} and {second.Length}.", nameof(second));

        if (first.Length < 2) return NumOps.Zero;

        // Coordinates relative to the start, so the area measures the shape of the path rather
        // than its absolute position. An absolute-coordinate version would report a spurious
        // area for a pair that merely sits far from the origin.
        // area = 0.5 * sum_k (x[k - 1] * dy[k] - y[k - 1] * dx[k]), k in [1, n).
        var engine = AiDotNetEngine.Current;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int n = first.Length;
        var xs = Tensor<T>.FromVector(first).Cast<double>();
        var ys = Tensor<T>.FromVector(second).Cast<double>();
        var xHead = engine.TensorNarrow(xs, 0, 0, n - 1);
        var yHead = engine.TensorNarrow(ys, 0, 0, n - 1);
        var x = engine.TensorSubtractScalar(xHead, NumOps.ToDouble(first[0]));
        var y = engine.TensorSubtractScalar(yHead, NumOps.ToDouble(second[0]));
        var dx = engine.TensorSubtract(engine.TensorNarrow(xs, 0, 1, n - 1), xHead);
        var dy = engine.TensorSubtract(engine.TensorNarrow(ys, 0, 1, n - 1), yHead);
        double twice = engine.TensorSum(engine.TensorMultiply(x, dy)) - engine.TensorSum(engine.TensorMultiply(y, dx));
        return NumOps.FromDouble(0.5 * twice);
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
        // A single sample has no increments, so every area is zero.
        if (steps < 2) return new Tensor<T>(new[] { assets, assets });

        // Area(j, l) = 0.5 * sum_k (X[k, j] D[k, l] - X[k, l] D[k, j]) with X[k] = P[k] - P[0] (k < steps - 1) and
        // D[k] = P[k + 1] - P[k], i.e. 0.5 * (X^T D - (X^T D)^T). Antisymmetric with a zero diagonal.
        var engine = AiDotNetEngine.Current;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var widePanel = panel.Cast<double>();
        var head = engine.TensorNarrow(widePanel, 0, 0, steps - 1);
        var relative = engine.TensorSubtract(head, engine.TensorNarrow(widePanel, 0, 0, 1));
        var increments = engine.TensorSubtract(engine.TensorNarrow(widePanel, 0, 1, steps - 1), head);
        var cross = engine.TensorMatMul(engine.TensorTranspose(relative), increments);        // [assets, assets]
        var areas = engine.TensorMultiplyScalar(engine.TensorSubtract(cross, engine.TensorTranspose(cross)), 0.5);
        return engine.TensorWhere(engine.TensorEye<double>(assets), new Tensor<double>([assets, assets]), areas).Cast<T>();
    }
}
