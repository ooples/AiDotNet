using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.QuadraticProgramming;

/// <summary>
/// Solves the support-vector-machine dual with Sequential Minimal Optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Solves the quadratic program every support-vector formulation reduces to:
/// <code>
///   minimize    ½·αᵀQα + pᵀα
///   subject to  yᵀα = constant
///               0 ≤ α_i ≤ C_i
/// </code>
/// with <c>Q_ij = y_i·y_j·K(x_i, x_j)</c>. Classification, ν-classification, ε-regression and
/// one-class detection all differ only in what <c>p</c>, <c>y</c>, <c>C</c> and the starting point
/// are — the solver itself is the same, which is why this is written once here rather than
/// separately inside each model.
/// </para>
/// <para>
/// A general-purpose quadratic-programming solver is the wrong tool for this particular QP.
/// <c>Q</c> is dense and <c>n × n</c> in the number of <b>training points</b>, so an active-set or
/// interior-point method factorizes an <c>n × n</c> system per iteration — cubic in dataset size,
/// and it must hold the whole kernel matrix in memory. SMO (Platt, 1998) avoids both: it optimizes
/// exactly <b>two</b> multipliers at a time, the smallest number the equality constraint permits,
/// and each such subproblem has a closed-form answer.
/// </para>
/// <para>
/// This implements the modern formulation used by LIBSVM (Chang and Lin, 2011; Fan, Chen and Lin,
/// 2005) rather than the original error-cache presentation:
/// <list type="bullet">
/// <item>the <b>gradient</b> <c>G_i = Σ_j Q_ij·α_j + p_i</c> is maintained incrementally, which
/// generalizes the classification-only "prediction error" to every formulation above;</item>
/// <item><b>maximal-violating-pair selection</b> chooses the two multipliers whose KKT violation is
/// largest, which is provably convergent and is the step that makes the most progress;</item>
/// <item>the stopping test is the <b>duality gap</b> <c>m(α) − M(α) ≤ tolerance</c>, a real measure
/// of distance from optimality rather than a count of unchanged sweeps.</item>
/// </list>
/// The simplified variant taught in course notes picks the partner multiplier at random and stops
/// after a fixed number of quiet passes; it converges far more slowly and can stall.
/// </para>
/// <para><b>For Beginners:</b> Training a support-vector machine means choosing one number per
/// training example, subject to a rule tying them together — so you cannot adjust one alone without
/// breaking the rule. Two is the smallest number you can move while keeping it satisfied, and
/// moving exactly two has a formula. The method repeatedly picks the pair that is furthest from
/// being correct, fixes that pair exactly, and repeats until nothing is meaningfully wrong.
/// </para>
/// </remarks>
public sealed class SequentialMinimalOptimizationSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly SequentialMinimalOptimizationOptions _options;

    /// <summary>
    /// Creates an SMO solver.
    /// </summary>
    /// <param name="options">
    /// Solver configuration. When omitted, the documented defaults on
    /// <see cref="SequentialMinimalOptimizationOptions"/> are used.
    /// </param>
    /// <param name="random">
    /// Retained for source compatibility and no longer used: maximal-violating-pair selection is
    /// deterministic, so training is reproducible without a random source.
    /// </param>
    public SequentialMinimalOptimizationSolver(
        SequentialMinimalOptimizationOptions? options = null, Random? random = null)
    {
        _options = options ?? new SequentialMinimalOptimizationOptions();
        _ = random;
    }

    /// <summary>
    /// Solves the dual, returning the multipliers and the bias term.
    /// </summary>
    /// <param name="kernel">
    /// The kernel evaluation <c>K(x_i, x_j)</c>, without label signs. Called on demand so the full
    /// kernel matrix is never materialized; supply a caching implementation when evaluations are
    /// expensive.
    /// </param>
    /// <param name="labels">
    /// The value <c>y_i</c> multiplying each multiplier in the equality constraint, which must be
    /// +1 or −1. Classification uses the class labels; regression uses +1 for the first half of the
    /// multipliers and −1 for the second; one-class problems use all +1.
    /// </param>
    /// <param name="linear">
    /// The linear term <c>p</c>. Classification uses all −1; ε-regression uses
    /// <c>[ε − y ; ε + y]</c>; one-class uses all zeros.
    /// </param>
    /// <param name="upperBounds">The per-multiplier upper bound <c>C_i</c>.</param>
    /// <param name="initialAlphas">
    /// Starting multipliers, which must already satisfy the constraints. When omitted, all zeros
    /// are used — feasible whenever the equality constant is zero. One-class formulations, whose
    /// multipliers must sum to a positive constant, must supply a feasible start.
    /// </param>
    /// <param name="kernelIndex">
    /// Maps a multiplier index to the training point it refers to. Regression duplicates every
    /// training point (one multiplier for each side of the ε-tube), so its map is
    /// <c>i =&gt; i % n</c>. When omitted, multipliers map to points one-to-one.
    /// </param>
    /// <returns>The optimized multipliers, the bias, and the number of pairs optimized.</returns>
    /// <exception cref="ArgumentNullException">Thrown when a required argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the argument lengths disagree.</exception>
    public (Vector<T> Alphas, T Bias, int Iterations) Solve(
        Func<int, int, T> kernel,
        Vector<T> labels,
        Vector<T> linear,
        Vector<T> upperBounds,
        Vector<T>? initialAlphas = null,
        Func<int, int>? kernelIndex = null)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (linear is null) throw new ArgumentNullException(nameof(linear));
        if (upperBounds is null) throw new ArgumentNullException(nameof(upperBounds));

        int n = labels.Length;
        if (linear.Length != n || upperBounds.Length != n)
        {
            throw new ArgumentException(
                "labels, linear and upperBounds must all have the same length.", nameof(linear));
        }

        kernelIndex ??= static i => i;

        var alphas = new T[n];
        if (initialAlphas is not null)
        {
            if (initialAlphas.Length != n)
            {
                throw new ArgumentException(
                    "initialAlphas must have the same length as labels.", nameof(initialAlphas));
            }

            for (int i = 0; i < n; i++) alphas[i] = initialAlphas[i];
        }

        // Q_ij = y_i · y_j · K(point(i), point(j)).
        T Q(int i, int j) => NumOps.Multiply(
            NumOps.Multiply(labels[i], labels[j]), kernel(kernelIndex(i), kernelIndex(j)));

        // Gradient of the dual: G_i = Σ_j Q_ij·α_j + p_i. This is the quantity every decision
        // below is made from, and it is what generalizes the classification "error" to the other
        // formulations.
        var gradient = new T[n];
        for (int i = 0; i < n; i++) gradient[i] = linear[i];

        for (int j = 0; j < n; j++)
        {
            if (NumOps.Equals(alphas[j], NumOps.Zero)) continue;
            for (int i = 0; i < n; i++)
            {
                gradient[i] = NumOps.Add(gradient[i], NumOps.Multiply(Q(i, j), alphas[j]));
            }
        }

        var tolerance = NumOps.FromDouble(_options.Tolerance);
        var epsilon = NumOps.FromDouble(_options.StepEpsilon);
        int iterations = 0;

        while (iterations < _options.MaxIterations)
        {
            // Maximal violating pair. A multiplier can decrease its objective by moving "up" the
            // constraint line if it is not already at the relevant bound for its label, and "down"
            // symmetrically; the violation magnitude is −y_i·G_i.
            int i = -1, j = -1;
            T maxUp = NumOps.Zero, minDown = NumOps.Zero;

            for (int k = 0; k < n; k++)
            {
                T violation = NumOps.Negate(NumOps.Multiply(labels[k], gradient[k]));

                if (CanIncreaseAlongConstraint(labels[k], alphas[k], upperBounds[k])
                    && (i < 0 || NumOps.GreaterThan(violation, maxUp)))
                {
                    i = k;
                    maxUp = violation;
                }

                if (CanDecreaseAlongConstraint(labels[k], alphas[k], upperBounds[k])
                    && (j < 0 || NumOps.LessThan(violation, minDown)))
                {
                    j = k;
                    minDown = violation;
                }
            }

            // Duality gap: when the best possible increase no longer exceeds the worst required
            // decrease, no pair can improve the objective and the KKT conditions hold.
            if (i < 0 || j < 0
                || !NumOps.GreaterThan(NumOps.Subtract(maxUp, minDown), tolerance))
            {
                break;
            }

            if (!TakeStep(i, j, Q, alphas, labels, gradient, upperBounds, epsilon, n))
            {
                break;
            }

            iterations++;
        }

        var result = new Vector<T>(n);
        for (int k = 0; k < n; k++) result[k] = alphas[k];

        return (result, ComputeBias(alphas, labels, gradient, upperBounds, n), iterations);
    }

    /// <summary>
    /// Optimizes one pair exactly and updates the gradient incrementally.
    /// </summary>
    /// <returns><c>false</c> when the step is too small to represent progress.</returns>
    private bool TakeStep(
        int i,
        int j,
        Func<int, int, T> q,
        T[] alphas,
        Vector<T> labels,
        T[] gradient,
        Vector<T> upperBounds,
        T epsilon,
        int n)
    {
        T oldAlphaI = alphas[i];
        T oldAlphaJ = alphas[j];
        T yi = labels[i];
        T yj = labels[j];

        // Curvature along the constraint line. The feasible direction is Δα_i = y_i·t,
        // Δα_j = −y_j·t, so the curvature is dᵀQd/t² = Q_ii + Q_jj − 2·y_i·y_j·Q_ij, which reduces
        // to K_ii + K_jj − 2·K_ij in the raw kernel. The y_i·y_j factor on the cross term is
        // essential: without it, a pair with OPPOSITE labels produces exactly zero curvature (the
        // two label signs cancel), the step falls back to the degenerate-curvature branch, and the
        // solver thrashes against its iteration limit instead of taking the one exact step that
        // solves a two-point problem outright.
        //
        // A non-positive value after that means the kernel is genuinely not positive definite;
        // LIBSVM substitutes a small positive number so the step stays well defined.
        T curvature = NumOps.Subtract(
            NumOps.Add(q(i, i), q(j, j)),
            NumOps.Multiply(
                NumOps.FromDouble(2.0),
                NumOps.Multiply(NumOps.Multiply(yi, yj), q(i, j))));

        if (!NumOps.GreaterThan(curvature, NumOps.Zero))
        {
            curvature = NumOps.FromDouble(1e-12);
        }

        // Unconstrained step along the line, then clipped to the box.
        T delta = NumOps.Divide(
            NumOps.Subtract(
                NumOps.Negate(NumOps.Multiply(yi, gradient[i])),
                NumOps.Negate(NumOps.Multiply(yj, gradient[j]))),
            curvature);

        // Moving i by yi·delta and j by −yj·delta keeps yᵀα unchanged.
        T newAlphaI = NumOps.Add(oldAlphaI, NumOps.Multiply(yi, delta));
        T newAlphaJ = NumOps.Subtract(oldAlphaJ, NumOps.Multiply(yj, delta));

        // Clip both to their boxes while preserving the equality, by shrinking the step to whatever
        // both can accommodate.
        newAlphaI = Clamp(newAlphaI, NumOps.Zero, upperBounds[i]);
        T reconstructedDelta = NumOps.Divide(NumOps.Subtract(newAlphaI, oldAlphaI), yi);
        newAlphaJ = NumOps.Subtract(oldAlphaJ, NumOps.Multiply(yj, reconstructedDelta));

        newAlphaJ = Clamp(newAlphaJ, NumOps.Zero, upperBounds[j]);
        reconstructedDelta = NumOps.Divide(NumOps.Subtract(oldAlphaJ, newAlphaJ), yj);
        newAlphaI = NumOps.Add(oldAlphaI, NumOps.Multiply(yi, reconstructedDelta));

        T changeI = NumOps.Subtract(newAlphaI, oldAlphaI);
        T changeJ = NumOps.Subtract(newAlphaJ, oldAlphaJ);

        if (NumOps.LessThan(NumOps.Abs(changeI), epsilon)
            && NumOps.LessThan(NumOps.Abs(changeJ), epsilon))
        {
            return false;
        }

        alphas[i] = newAlphaI;
        alphas[j] = newAlphaJ;

        // Incremental gradient update — the reason each iteration costs O(n) rather than O(n²).
        for (int k = 0; k < n; k++)
        {
            gradient[k] = NumOps.Add(gradient[k], NumOps.Add(
                NumOps.Multiply(q(k, i), changeI),
                NumOps.Multiply(q(k, j), changeJ)));
        }

        return true;
    }

    /// <summary>
    /// Computes the bias of the decision function from the free multipliers.
    /// </summary>
    /// <remarks>
    /// Multipliers strictly between their bounds sit exactly on the margin, so each pins the bias
    /// precisely; averaging over them is the standard, numerically stable choice. With none free,
    /// any value between the two boundary quantities satisfies the conditions, so the midpoint is
    /// taken.
    /// </remarks>
    private static T ComputeBias(
        T[] alphas, Vector<T> labels, T[] gradient, Vector<T> upperBounds, int n)
    {
        T sum = NumOps.Zero;
        int freeCount = 0;
        T upperCandidate = NumOps.Zero;
        T lowerCandidate = NumOps.Zero;
        bool hasUpper = false, hasLower = false;

        for (int k = 0; k < n; k++)
        {
            T violation = NumOps.Negate(NumOps.Multiply(labels[k], gradient[k]));

            if (NumOps.GreaterThan(alphas[k], NumOps.Zero)
                && NumOps.LessThan(alphas[k], upperBounds[k]))
            {
                sum = NumOps.Add(sum, violation);
                freeCount++;
                continue;
            }

            if (CanIncreaseAlongConstraint(labels[k], alphas[k], upperBounds[k]))
            {
                if (!hasUpper || NumOps.GreaterThan(violation, upperCandidate))
                {
                    upperCandidate = violation;
                    hasUpper = true;
                }
            }

            if (CanDecreaseAlongConstraint(labels[k], alphas[k], upperBounds[k]))
            {
                if (!hasLower || NumOps.LessThan(violation, lowerCandidate))
                {
                    lowerCandidate = violation;
                    hasLower = true;
                }
            }
        }

        if (freeCount > 0)
        {
            return NumOps.Divide(sum, NumOps.FromDouble(freeCount));
        }

        if (hasUpper && hasLower)
        {
            return NumOps.Divide(
                NumOps.Add(upperCandidate, lowerCandidate), NumOps.FromDouble(2.0));
        }

        if (hasUpper) return upperCandidate;
        if (hasLower) return lowerCandidate;

        return NumOps.Zero;
    }

    /// <summary>
    /// Whether the multiplier can move in the direction that increases <c>y_i·α_i</c> without
    /// leaving its box.
    /// </summary>
    private static bool CanIncreaseAlongConstraint(T label, T alpha, T upperBound)
    {
        return NumOps.GreaterThan(label, NumOps.Zero)
            ? NumOps.LessThan(alpha, upperBound)
            : NumOps.GreaterThan(alpha, NumOps.Zero);
    }

    /// <summary>
    /// Whether the multiplier can move in the direction that decreases <c>y_i·α_i</c> without
    /// leaving its box.
    /// </summary>
    private static bool CanDecreaseAlongConstraint(T label, T alpha, T upperBound)
    {
        return NumOps.GreaterThan(label, NumOps.Zero)
            ? NumOps.GreaterThan(alpha, NumOps.Zero)
            : NumOps.LessThan(alpha, upperBound);
    }

    private static T Min(T left, T right) => NumOps.LessThan(left, right) ? left : right;

    private static T Max(T left, T right) => NumOps.GreaterThan(left, right) ? left : right;

    private static T Clamp(T value, T low, T high) => Min(Max(value, low), high);
}
