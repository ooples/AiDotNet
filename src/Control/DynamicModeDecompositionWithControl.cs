using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Identifies a linear state-space model from measured trajectories by dynamic mode decomposition
/// with control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements DMDc from J. L. Proctor, S. L. Brunton and J. N. Kutz, "Dynamic Mode Decomposition with
/// Control", <i>SIAM Journal on Applied Dynamical Systems</i> 15(1), 2016, pp. 142-161.
/// </para>
/// <para>
/// <b>The idea, which is simpler than it sounds.</b> Line up every measured transition: a matrix
/// <c>X</c> of states, the matrix <c>X'</c> of the states that followed them, and the matrix
/// <c>Υ</c> of the inputs applied in between. If the system really is <c>x' = Ax + Bu</c> then
/// <c>X' = [A B]·[X; Υ]</c>, and recovering <c>[A B]</c> is a least-squares problem — solved through
/// the pseudoinverse of the stacked data. There is no iteration and no initial guess; the answer is
/// a single matrix factorization.
/// </para>
/// <para>
/// <b>Why the SVD rather than a normal-equations solve.</b> Real trajectory data is
/// close to rank-deficient: states move together, inputs are correlated with the responses they
/// caused, and successive snapshots are nearly the same. Forming the normal equations squares the
/// condition number of all that, and a mildly ill-conditioned data set becomes an unusable one.
/// Truncating the singular value decomposition instead discards precisely the directions the data
/// does not determine, which is both better conditioned and more honest — a direction with no
/// information in it should not be assigned a coefficient.
/// </para>
/// <para>
/// <b>The distinguishing difficulty DMDc solves.</b> Plain dynamic mode decomposition applied to
/// controlled data gives the wrong dynamics: it cannot tell whether the state moved because of the
/// system or because of the input, and so it attributes the actuation to the system's own behaviour.
/// Including the inputs in the regression is what separates the two, and it is why the identified
/// <c>A</c> here describes the uncontrolled system rather than the closed loop that produced the
/// data.
/// </para>
/// <para><b>For Beginners:</b> You poke a system, record what it does, and this works backwards to
/// the equations. It is how you get a model of something you cannot derive from physics — a chemical
/// process, a building's temperature, a machine with unknown friction. What comes out plugs straight
/// into the controllers in this namespace.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var q = 2;
/// var r = 2;
/// var inputs = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // states: one column per snapshot; inputs: the command applied at each of those snapshots.
/// var result = new DynamicModeDecompositionWithControl&lt;double&gt;()
///     .Identify(states, nextStates, inputs);
///
/// var controller = new LinearQuadraticRegulator&lt;double&gt;(
///     result.StateMatrix, result.InputMatrix, q, r);
/// </code>
/// </example>
public sealed class DynamicModeDecompositionWithControl<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _singularValueThreshold;

    /// <summary>
    /// Creates a DMDc identifier.
    /// </summary>
    /// <param name="singularValueThreshold">
    /// Singular values below this fraction of the largest are treated as noise and discarded.
    /// Defaults to 1e-10, which removes only numerically dead directions; raise it to reject
    /// measurement noise.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the threshold is negative or not below one.
    /// </exception>
    public DynamicModeDecompositionWithControl(double singularValueThreshold = 1e-10)
    {
        if (singularValueThreshold < 0.0 || singularValueThreshold >= 1.0)
        {
            throw new ArgumentException(
                "The singular value threshold is a fraction of the largest singular value, so it " +
                "must lie in [0, 1).", nameof(singularValueThreshold));
        }

        _singularValueThreshold = singularValueThreshold;
    }

    /// <summary>
    /// Identifies <c>A</c> and <c>B</c> from measured transitions.
    /// </summary>
    /// <param name="states">
    /// The states, one column per snapshot: <c>n</c>-by-<c>k</c>.
    /// </param>
    /// <param name="nextStates">
    /// The state that followed each snapshot, same shape as <paramref name="states"/>.
    /// </param>
    /// <param name="inputs">
    /// The input applied at each snapshot: <c>m</c>-by-<c>k</c>.
    /// </param>
    /// <param name="rank">
    /// Maximum rank to retain, or <c>null</c> to keep every direction above the threshold. Truncating
    /// is how noise is rejected — see <see cref="SystemIdentificationResult{T}.SingularValues"/> for
    /// how to choose.
    /// </param>
    /// <returns>The identified model, with the diagnostics needed to judge it.</returns>
    /// <exception cref="ArgumentNullException">Thrown when a matrix is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the shapes disagree or there are no snapshots.
    /// </exception>
    public SystemIdentificationResult<T> Identify(
        Matrix<T> states, Matrix<T> nextStates, Matrix<T> inputs, int? rank = null)
    {
        if (states is null) throw new ArgumentNullException(nameof(states));
        if (nextStates is null) throw new ArgumentNullException(nameof(nextStates));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));

        int stateCount = states.Rows;
        int snapshotCount = states.Columns;
        int inputCount = inputs.Rows;

        if (snapshotCount == 0 || stateCount == 0)
        {
            throw new ArgumentException(
                "At least one state and one snapshot are required.", nameof(states));
        }

        if (nextStates.Rows != stateCount || nextStates.Columns != snapshotCount)
        {
            throw new ArgumentException(
                $"nextStates must have the same shape as states ({stateCount}-by-" +
                $"{snapshotCount}); it is {nextStates.Rows}-by-{nextStates.Columns}.",
                nameof(nextStates));
        }

        if (inputs.Columns != snapshotCount)
        {
            throw new ArgumentException(
                $"inputs must have one column per snapshot: expected {snapshotCount} columns, but " +
                $"it has {inputs.Columns}. Each column is the input applied at the corresponding " +
                "state.", nameof(inputs));
        }

        if (rank is not null && rank.Value <= 0)
        {
            throw new ArgumentException("The rank must be positive when supplied.", nameof(rank));
        }

        // Omega = [X; Upsilon], stacking states above inputs so one regression recovers [A B].
        int stackedRows = stateCount + inputCount;
        var stacked = new Matrix<T>(stackedRows, snapshotCount);

        for (int c = 0; c < snapshotCount; c++)
        {
            for (int r = 0; r < stateCount; r++) stacked[r, c] = states[r, c];
            for (int r = 0; r < inputCount; r++) stacked[stateCount + r, c] = inputs[r, c];
        }

        var svd = new SvdDecomposition<T>(stacked);

        int available = Math.Min(svd.S.Length, Math.Min(stackedRows, snapshotCount));
        int retained = DetermineRank(svd.S, available, rank);

        if (retained == 0)
        {
            throw new ArgumentException(
                "The data has no numerically significant content — every singular value of the " +
                "stacked snapshots is negligible. This usually means the recorded states never " +
                "moved.", nameof(states));
        }

        // [A B] = X' * V * S^-1 * U^T, truncated to the retained directions.
        var pseudoInverse = BuildTruncatedPseudoInverse(svd, retained, stackedRows, snapshotCount);
        var operators = ControlMath<T>.Multiply(nextStates, pseudoInverse);

        var stateMatrix = new Matrix<T>(stateCount, stateCount);
        var inputMatrix = new Matrix<T>(stateCount, inputCount);

        for (int r = 0; r < stateCount; r++)
        {
            for (int c = 0; c < stateCount; c++) stateMatrix[r, c] = operators[r, c];
            for (int c = 0; c < inputCount; c++) inputMatrix[r, c] = operators[r, stateCount + c];
        }

        T residual = ComputeResidual(operators, stacked, nextStates);

        var singularValues = new Vector<T>(available);
        for (int i = 0; i < available; i++) singularValues[i] = svd.S[i];

        return new SystemIdentificationResult<T>(
            stateMatrix, inputMatrix, singularValues, retained, residual);
    }

    /// <summary>
    /// Decides how many singular directions to keep.
    /// </summary>
    private int DetermineRank(Vector<T> singularValues, int available, int? requested)
    {
        if (available == 0) return 0;

        double largest = 0.0;
        for (int i = 0; i < available; i++)
        {
            largest = Math.Max(largest, Math.Abs(NumOps.ToDouble(singularValues[i])));
        }

        if (largest <= 0.0) return 0;

        double cutoff = largest * _singularValueThreshold;

        int significant = 0;
        for (int i = 0; i < available; i++)
        {
            if (Math.Abs(NumOps.ToDouble(singularValues[i])) > cutoff) significant++;
        }

        return requested is null ? significant : Math.Min(requested.Value, significant);
    }

    /// <summary>
    /// Builds the truncated pseudoinverse <c>V·S⁻¹·Uᵀ</c> of the stacked data.
    /// </summary>
    private static Matrix<T> BuildTruncatedPseudoInverse(
        SvdDecomposition<T> svd, int retained, int stackedRows, int snapshotCount)
    {
        var pseudoInverse = new Matrix<T>(snapshotCount, stackedRows);

        for (int r = 0; r < snapshotCount; r++)
        {
            for (int c = 0; c < stackedRows; c++)
            {
                T accumulator = NumOps.Zero;
                for (int k = 0; k < retained; k++)
                {
                    // Vt is Vᵀ, so V[r, k] is Vt[k, r]; U is stackedRows-by-something so Uᵀ[k, c]
                    // is U[c, k].
                    T scaled = NumOps.Divide(svd.Vt[k, r], svd.S[k]);
                    accumulator = NumOps.Add(accumulator, NumOps.Multiply(scaled, svd.U[c, k]));
                }

                pseudoInverse[r, c] = accumulator;
            }
        }

        return pseudoInverse;
    }

    /// <summary>
    /// Measures how far the identified operators are from reproducing the recorded transitions.
    /// </summary>
    private static T ComputeResidual(
        Matrix<T> operators, Matrix<T> stacked, Matrix<T> nextStates)
    {
        var reconstructed = ControlMath<T>.Multiply(operators, stacked);
        return NumOps.FromDouble(
            ControlMath<T>.FrobeniusNorm(ControlMath<T>.Subtract(reconstructed, nextStates)));
    }
}
