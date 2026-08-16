namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the Sequential Minimal Optimization solver used to train support-vector
/// machines.
/// </summary>
/// <remarks>
/// <para><b>Reference:</b> Platt, “Sequential Minimal Optimization: A Fast Algorithm for Training
/// Support Vector Machines” (1998).</para>
/// <para><b>For Beginners:</b> SMO trains an SVM by repeatedly fixing the two multipliers that are
/// furthest from satisfying the optimality rules.</para>
/// </remarks>
public class SequentialMinimalOptimizationOptions : ModelOptions
{
    public SequentialMinimalOptimizationOptions() { }

    public SequentialMinimalOptimizationOptions(SequentialMinimalOptimizationOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        MaxIterations = other.MaxIterations;
        Tolerance = other.Tolerance;
        StepEpsilon = other.StepEpsilon;
        RestrictPairsToSameLabel = other.RestrictPairsToSameLabel;
    }

    /// <summary>
    /// Gets or sets the maximum number of multiplier pairs optimized.
    /// </summary>
    /// <value>The iteration limit, defaulting to 1000000.</value>
    /// <remarks>
    /// <para>
    /// Each iteration optimizes one pair exactly, so this counts successful steps rather than
    /// passes over the data. The default is deliberately generous: SMO's cost per step is tiny, and
    /// stopping early leaves multipliers that still violate the KKT conditions, which shows up as a
    /// silently worse decision boundary rather than an error.
    /// </para>
    /// <para><b>For Beginners:</b> This is a safety ceiling on how many two-variable corrections
    /// training may perform.</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000000;

    /// <summary>
    /// Gets or sets how far a multiplier may violate the KKT conditions before it is considered for
    /// optimization.
    /// </summary>
    /// <value>The KKT tolerance, defaulting to 1e-3.</value>
    /// <remarks>
    /// <para>
    /// This is the value Platt (1998) uses and LIBSVM defaults to. It is far looser than the
    /// tolerances elsewhere in this library for a good reason: the multipliers only need to be
    /// accurate enough to place the decision boundary correctly, and tightening it multiplies
    /// training time for a boundary that does not visibly move.
    /// </para>
    /// <para><b>For Beginners:</b> How close to perfect the solution has to be before training
    /// stops. Tighter is slower and rarely changes the predictions.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the smallest relative multiplier movement that counts as progress.
    /// </summary>
    /// <value>The step epsilon, defaulting to 1e-12.</value>
    /// <remarks>
    /// <para>
    /// A pair whose optimal step is smaller than this is rejected, which prevents the solver from
    /// looping forever on steps that are pure floating-point noise. It also decides ties when the
    /// objective's curvature along the segment is non-positive.
    /// </para>
    /// <para><b>For Beginners:</b> Changes smaller than this are treated as numerical noise, which
    /// prevents endless microscopic updates.</para>
    /// </remarks>
    public double StepEpsilon { get; set; } = 1e-12;

    /// <summary>
    /// Gets or sets whether the working pair must be drawn from a single class.
    /// </summary>
    /// <value><c>false</c> by default, which is correct for C-parameterized formulations.</value>
    /// <remarks>
    /// <para>
    /// The nu-parameterized formulations (nu-SVC, nu-SVR) carry <b>two</b> equality constraints
    /// rather than one: <c>Σ αᵢyᵢ = 0</c> and <c>Σ αᵢ = ν·n</c>. A step that moves α_i by
    /// <c>y_i·t</c> and α_j by <c>−y_j·t</c> preserves the first for any pair, but preserves the
    /// second only when the two labels agree — with opposite labels the total drifts by <c>2t</c>.
    /// </para>
    /// <para>
    /// Restricting the pair to one class keeps both sums invariant, which is exactly how LIBSVM's
    /// Solver_NU works: it selects the maximal violating pair within each class and takes whichever
    /// is worse. Enable this for nu-parameterized problems and supply a feasible starting point,
    /// since all-zeros does not satisfy <c>Σ αᵢ = ν·n</c>.
    /// </para>
    /// <para><b>For Beginners:</b> The nu variants pin down not just the balance between the two
    /// classes but also the total amount of influence handed out. Swapping influence between two
    /// examples of the same class leaves both quantities untouched; swapping across classes would
    /// change the total. So this setting says "only trade within a class".
    /// </para>
    /// </remarks>
    public bool RestrictPairsToSameLabel { get; set; } = false;
}
