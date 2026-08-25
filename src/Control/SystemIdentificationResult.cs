using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// A linear state-space model recovered from measured data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The matrices here are exactly what every other class in this module consumes: hand them to
/// <see cref="LinearQuadraticRegulator{T}"/> or <see cref="ModelPredictiveController{T}"/> and you
/// have a controller for a system nobody derived equations for. That path — measure, identify,
/// control — is how most industrial controllers are actually built.
/// </para>
/// </remarks>
public sealed class SystemIdentificationResult<T>
{
    /// <summary>Gets the identified state matrix <c>A</c>.</summary>
    public Matrix<T> StateMatrix { get; }

    /// <summary>Gets the identified input matrix <c>B</c>.</summary>
    public Matrix<T> InputMatrix { get; }

    /// <summary>
    /// Gets the singular values of the stacked data matrix, largest first.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the diagnostic that says whether the identification can be believed. A sharp drop
    /// after <c>r</c> values means the data genuinely lives in an <c>r</c>-dimensional subspace and
    /// truncating there is safe. No drop at all means the data does not determine a model of the
    /// requested size, and the fit is being carried by noise — a fact the residual alone will not
    /// reveal, because a model with enough freedom always fits the data it was built from.
    /// </para>
    /// </remarks>
    public Vector<T> SingularValues { get; }

    /// <summary>
    /// Gets the rank actually used, after truncation.
    /// </summary>
    public int Rank { get; }

    /// <summary>
    /// Gets the Frobenius norm of the one-step prediction residual on the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This measures how well the identified model reproduces the data it was fitted to, which is a
    /// necessary but not sufficient condition for it being right. Read it together with
    /// <see cref="SingularValues"/>.
    /// </para>
    /// </remarks>
    public T Residual { get; }

    /// <summary>
    /// Creates a system identification result.
    /// </summary>
    /// <param name="stateMatrix">The identified state matrix.</param>
    /// <param name="inputMatrix">The identified input matrix.</param>
    /// <param name="singularValues">The singular values of the stacked data matrix.</param>
    /// <param name="rank">The rank used after truncation.</param>
    /// <param name="residual">The one-step prediction residual.</param>
    /// <exception cref="ArgumentNullException">Thrown when a matrix or vector is null.</exception>
    public SystemIdentificationResult(
        Matrix<T> stateMatrix,
        Matrix<T> inputMatrix,
        Vector<T> singularValues,
        int rank,
        T residual)
    {
        StateMatrix = stateMatrix ?? throw new ArgumentNullException(nameof(stateMatrix));
        InputMatrix = inputMatrix ?? throw new ArgumentNullException(nameof(inputMatrix));
        SingularValues = singularValues ?? throw new ArgumentNullException(nameof(singularValues));
        Rank = rank;
        Residual = residual;
    }
}
