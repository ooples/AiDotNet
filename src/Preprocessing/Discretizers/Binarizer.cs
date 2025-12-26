using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Discretizers;

/// <summary>
/// Binarizes features based on a threshold value.
/// </summary>
/// <remarks>
/// <para>
/// Binarization transforms continuous values to binary (0 or 1) based on a threshold.
/// Values greater than the threshold become 1, values less than or equal become 0.
/// </para>
/// <para><b>For Beginners:</b> This transformer converts any values to just 0s and 1s:
/// - If a value is above the threshold → 1
/// - If a value is at or below the threshold → 0
///
/// Example with threshold=5: [3, 6, 2, 8, 5] → [0, 1, 0, 1, 0]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class Binarizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly T _threshold;

    /// <summary>
    /// Gets the threshold value used for binarization.
    /// </summary>
    public T Threshold => _threshold;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Binarization is a lossy transformation - the original values cannot be recovered.
    /// </remarks>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="Binarizer{T}"/> with a default threshold of 0.
    /// </summary>
    /// <param name="columnIndices">The column indices to binarize, or null for all columns.</param>
    public Binarizer(int[]? columnIndices = null)
        : this(0.0, columnIndices)
    {
    }

    /// <summary>
    /// Creates a new instance of <see cref="Binarizer{T}"/> with a custom threshold.
    /// </summary>
    /// <param name="threshold">The threshold value. Values greater than this become 1, others become 0.</param>
    /// <param name="columnIndices">The column indices to binarize, or null for all columns.</param>
    public Binarizer(double threshold, int[]? columnIndices = null)
        : base(columnIndices)
    {
        _threshold = NumOps.FromDouble(threshold);
    }

    /// <summary>
    /// Fits the binarizer to the training data.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    /// <remarks>
    /// Binarization is a stateless transformation - fitting does nothing except validate the data.
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        // Binarizer is stateless - nothing to learn from data
        // The threshold is set at construction time
    }

    /// <summary>
    /// Transforms the data by applying threshold binarization.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The binarized data (0s and 1s).</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    // Values > threshold → 1, otherwise → 0
                    value = NumOps.Compare(value, _threshold) > 0 ? NumOps.One : NumOps.Zero;
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for binarization.
    /// </summary>
    /// <param name="data">The binarized data.</param>
    /// <returns>Never returns - always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown because binarization is lossy.</exception>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException(
            "Binarizer does not support inverse transformation. " +
            "Binarization is a lossy transformation - the original values cannot be recovered.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (Binarizer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
