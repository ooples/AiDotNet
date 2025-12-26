using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Specifies the norm to use for sample normalization.
/// </summary>
public enum NormType
{
    /// <summary>
    /// L1 norm (Manhattan distance) - sum of absolute values.
    /// </summary>
    L1,

    /// <summary>
    /// L2 norm (Euclidean distance) - square root of sum of squares.
    /// </summary>
    L2,

    /// <summary>
    /// Max norm (Chebyshev distance) - maximum absolute value.
    /// </summary>
    Max
}

/// <summary>
/// Normalizes samples (rows) individually to unit norm (L1, L2, or Max).
/// </summary>
/// <remarks>
/// <para>
/// Unlike scalers that operate on columns (features), this normalizer operates on rows (samples).
/// Each sample is scaled to have a unit norm (length of 1) in the specified norm type.
/// This is useful when the magnitude of samples varies but their direction matters.
/// </para>
/// <para><b>For Beginners:</b> This normalizer scales each row so its "length" equals 1:
/// - L1 norm: The sum of absolute values equals 1
/// - L2 norm: The Euclidean length (sqrt of sum of squares) equals 1
/// - Max norm: The maximum absolute value equals 1
///
/// Example with L2 norm: [3, 4] has length 5, so it becomes [0.6, 0.8] (length = 1)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class Normalizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly NormType _normType;

    /// <summary>
    /// Gets the norm type used for normalization.
    /// </summary>
    public NormType NormType => _normType;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Inverse transform is not supported because the original row norms are not stored.
    /// </remarks>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="Normalizer{T}"/>.
    /// </summary>
    /// <param name="normType">The type of norm to use (L1, L2, or Max). Defaults to L2.</param>
    /// <param name="columnIndices">The column indices to include in normalization, or null for all columns.</param>
    public Normalizer(NormType normType = NormType.L2, int[]? columnIndices = null)
        : base(columnIndices)
    {
        _normType = normType;
    }

    /// <summary>
    /// Fits the normalizer to the training data.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    /// <remarks>
    /// Normalizer is stateless for transformation - fitting validates the data structure.
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        // Normalizer is stateless - each row is normalized independently
        // No parameters need to be learned from training data
    }

    /// <summary>
    /// Transforms the data by normalizing each row to unit norm.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The normalized data where each row has unit norm.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);

        for (int i = 0; i < numRows; i++)
        {
            // Calculate the norm for this row using only the specified columns
            T norm = CalculateNorm(data, i, columnsToProcess);

            // Handle zero norm case
            if (NumOps.Compare(norm, NumOps.Zero) == 0)
            {
                norm = NumOps.One;
            }

            // Normalize the row
            for (int j = 0; j < numColumns; j++)
            {
                if (columnsToProcess.Contains(j))
                {
                    result[i, j] = NumOps.Divide(data[i, j], norm);
                }
                else
                {
                    result[i, j] = data[i, j];
                }
            }
        }

        return new Matrix<T>(result);
    }

    private T CalculateNorm(Matrix<T> data, int row, int[] columns)
    {
        switch (_normType)
        {
            case NormType.L1:
                return CalculateL1Norm(data, row, columns);
            case NormType.L2:
                return CalculateL2Norm(data, row, columns);
            case NormType.Max:
                return CalculateMaxNorm(data, row, columns);
            default:
                throw new ArgumentException($"Unknown norm type: {_normType}");
        }
    }

    private T CalculateL1Norm(Matrix<T> data, int row, int[] columns)
    {
        T sum = NumOps.Zero;
        foreach (var col in columns)
        {
            sum = NumOps.Add(sum, NumOps.Abs(data[row, col]));
        }
        return sum;
    }

    private T CalculateL2Norm(Matrix<T> data, int row, int[] columns)
    {
        T sumOfSquares = NumOps.Zero;
        foreach (var col in columns)
        {
            T val = data[row, col];
            sumOfSquares = NumOps.Add(sumOfSquares, NumOps.Multiply(val, val));
        }
        return NumOps.Sqrt(sumOfSquares);
    }

    private T CalculateMaxNorm(Matrix<T> data, int row, int[] columns)
    {
        T maxAbs = NumOps.Zero;
        foreach (var col in columns)
        {
            T absVal = NumOps.Abs(data[row, col]);
            if (NumOps.Compare(absVal, maxAbs) > 0)
            {
                maxAbs = absVal;
            }
        }
        return maxAbs;
    }

    /// <summary>
    /// Inverse transformation is not supported for sample normalization.
    /// </summary>
    /// <param name="data">The normalized data.</param>
    /// <returns>Never returns - always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown because sample norms are not stored.</exception>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException(
            "Normalizer does not support inverse transformation. " +
            "The original row norms are not stored during transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (Normalizer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
