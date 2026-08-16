using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Shape checks the Riccati solvers share.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A control problem is described by four matrices whose dimensions must agree in a specific way,
/// and getting one of them transposed is the most common way to state one wrongly. Checking up front
/// turns that into a message naming the offending matrix instead of an index-out-of-range thrown
/// several matrix products later.
/// </para>
/// </remarks>
internal static class RiccatiValidation<T>
{
    /// <summary>
    /// Validates the four matrices of a linear-quadratic problem against each other.
    /// </summary>
    /// <exception cref="ArgumentNullException">Thrown when any matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    public static void Validate(
        Matrix<T> stateMatrix, Matrix<T> inputMatrix, Matrix<T> stateCost, Matrix<T> inputCost)
    {
        if (stateMatrix is null) throw new ArgumentNullException(nameof(stateMatrix));
        if (inputMatrix is null) throw new ArgumentNullException(nameof(inputMatrix));
        if (stateCost is null) throw new ArgumentNullException(nameof(stateCost));
        if (inputCost is null) throw new ArgumentNullException(nameof(inputCost));

        if (stateMatrix.Rows != stateMatrix.Columns)
        {
            throw new ArgumentException(
                $"The state matrix A must be square; it is {stateMatrix.Rows}-by-" +
                $"{stateMatrix.Columns}.", nameof(stateMatrix));
        }

        int stateCount = stateMatrix.Rows;
        if (stateCount == 0)
        {
            throw new ArgumentException(
                "The system must have at least one state.", nameof(stateMatrix));
        }

        if (inputMatrix.Rows != stateCount)
        {
            throw new ArgumentException(
                $"The input matrix B must have one row per state: expected {stateCount} rows to " +
                $"match A, but it has {inputMatrix.Rows}.", nameof(inputMatrix));
        }

        int inputCount = inputMatrix.Columns;
        if (inputCount == 0)
        {
            throw new ArgumentException(
                "The system must have at least one input; with no inputs there is nothing to " +
                "control.", nameof(inputMatrix));
        }

        if (stateCost.Rows != stateCount || stateCost.Columns != stateCount)
        {
            throw new ArgumentException(
                $"The state cost Q must be {stateCount}-by-{stateCount} to match A; it is " +
                $"{stateCost.Rows}-by-{stateCost.Columns}.", nameof(stateCost));
        }

        if (inputCost.Rows != inputCount || inputCost.Columns != inputCount)
        {
            throw new ArgumentException(
                $"The input cost R must be {inputCount}-by-{inputCount} to match the columns of B; " +
                $"it is {inputCost.Rows}-by-{inputCost.Columns}.", nameof(inputCost));
        }
    }
}
