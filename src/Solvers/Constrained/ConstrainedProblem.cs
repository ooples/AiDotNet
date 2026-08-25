using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.Constrained;

/// <summary>
/// A general nonlinear optimization problem: minimize <c>f(x)</c> subject to <c>h(x) = 0</c> and
/// <c>g(x) ≤ 0</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the shape every constrained problem reduces to, and the one Chapter 6 of the optimization
/// literature states the KKT conditions over. Note the sign convention: inequalities are written
/// <c>g(x) ≤ 0</c>, so a limit like <c>x ≤ 5</c> is supplied as <c>g(x) = x − 5</c>.
/// </para>
/// <para>
/// Each constraint block is one function returning both the constraint values and their Jacobian —
/// row <c>i</c> holding the gradient of constraint <c>i</c>. They are supplied together because a
/// solver needs both at the same point on every iteration, and evaluating them in one pass lets a
/// caller share the expensive intermediate work between them.
/// </para>
/// <para><b>For Beginners:</b> Three pieces describe a constrained problem: what you are minimizing,
/// what must come out exactly right (the equalities), and what must not be exceeded (the
/// inequalities). Anything you leave out simply does not constrain the answer — pass <c>null</c>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // minimize x² + y² subject to x + y = 1
/// var problem = new ConstrainedProblem&lt;double&gt;(
///     objective: p =&gt; (p[0] * p[0] + p[1] * p[1],
///                      new Vector&lt;double&gt;(new[] { 2 * p[0], 2 * p[1] })),
///     equalityConstraints: p =&gt;
///     {
///         var values = new Vector&lt;double&gt;(new[] { p[0] + p[1] - 1.0 });
///         var jacobian = new Matrix&lt;double&gt;(1, 2);
///         jacobian[0, 0] = 1.0;
///         jacobian[0, 1] = 1.0;
///         return (values, jacobian);
///     });
/// </code>
/// </example>
public sealed class ConstrainedProblem<T>
{
    /// <summary>
    /// Gets the objective, returning its value and gradient at a point.
    /// </summary>
    public Func<Vector<T>, (T Value, Vector<T> Gradient)> Objective { get; }

    /// <summary>
    /// Gets the equality constraints <c>h(x) = 0</c> and their Jacobian, or <c>null</c> when there
    /// are none.
    /// </summary>
    public Func<Vector<T>, (Vector<T> Values, Matrix<T> Jacobian)>? EqualityConstraints { get; }

    /// <summary>
    /// Gets the inequality constraints <c>g(x) ≤ 0</c> and their Jacobian, or <c>null</c> when there
    /// are none.
    /// </summary>
    public Func<Vector<T>, (Vector<T> Values, Matrix<T> Jacobian)>? InequalityConstraints { get; }

    /// <summary>
    /// Creates a constrained problem.
    /// </summary>
    /// <param name="objective">The function to minimize, returning its value and gradient.</param>
    /// <param name="equalityConstraints">
    /// The equality constraints <c>h(x) = 0</c> and their Jacobian, or <c>null</c>.
    /// </param>
    /// <param name="inequalityConstraints">
    /// The inequality constraints <c>g(x) ≤ 0</c> and their Jacobian, or <c>null</c>.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="objective"/> is null.
    /// </exception>
    public ConstrainedProblem(
        Func<Vector<T>, (T Value, Vector<T> Gradient)> objective,
        Func<Vector<T>, (Vector<T> Values, Matrix<T> Jacobian)>? equalityConstraints = null,
        Func<Vector<T>, (Vector<T> Values, Matrix<T> Jacobian)>? inequalityConstraints = null)
    {
        Objective = objective ?? throw new ArgumentNullException(nameof(objective));
        EqualityConstraints = equalityConstraints;
        InequalityConstraints = inequalityConstraints;
    }
}
