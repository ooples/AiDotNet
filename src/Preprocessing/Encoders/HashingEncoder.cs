using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using feature hashing (hashing trick).
/// </summary>
/// <remarks>
/// <para>
/// HashingEncoder uses a hash function to map categories to a fixed number of columns.
/// This is useful for high-cardinality categorical features where one-hot encoding
/// would create too many columns.
/// </para>
/// <para>
/// Unlike other encoders, HashingEncoder doesn't need to store the category mappings,
/// making it memory-efficient and able to handle previously unseen categories.
/// </para>
/// <para><b>For Beginners:</b> Instead of creating one column per category:
/// - Hash encoding creates a fixed number of columns (e.g., 8)
/// - Each category is hashed to one of these columns
/// - Multiple categories may share the same column (collision)
///
/// Pros: Fixed memory, handles new categories, fast
/// Cons: Information loss from collisions, not reversible
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class HashingEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly bool _alternateSign;

    private int _nInputFeatures;
    private int _nOutputFeatures;
    private List<int>? _featureIndicesStart;

    /// <summary>
    /// Gets the number of hash components (output features per encoded column).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets whether alternate signs are used for hash collisions.
    /// </summary>
    public bool AlternateSign => _alternateSign;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="HashingEncoder{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of output features per encoded column. Defaults to 8.</param>
    /// <param name="alternateSign">If true, use alternate signs to reduce collision bias. Defaults to true.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public HashingEncoder(
        int nComponents = 8,
        bool alternateSign = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _alternateSign = alternateSign;
    }

    /// <summary>
    /// Computes the output feature structure.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        _featureIndicesStart = new List<int>();
        int currentOutputIndex = 0;

        for (int col = 0; col < _nInputFeatures; col++)
        {
            _featureIndicesStart.Add(currentOutputIndex);

            if (processSet.Contains(col))
            {
                currentOutputIndex += _nComponents;
            }
            else
            {
                currentOutputIndex += 1; // Pass-through
            }
        }

        _nOutputFeatures = currentOutputIndex;
    }

    /// <summary>
    /// Transforms the data using feature hashing.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The hash-encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_featureIndicesStart is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nOutputFeatures];
        var columnsToProcess = GetColumnsToProcess(data.Columns);
        var processSet = new HashSet<int>(columnsToProcess);

        // Initialize to zero
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < _nOutputFeatures; j++)
            {
                result[i, j] = NumOps.Zero;
            }
        }

        for (int i = 0; i < numRows; i++)
        {
            for (int col = 0; col < data.Columns; col++)
            {
                int outputStart = _featureIndicesStart[col];

                if (!processSet.Contains(col))
                {
                    // Pass-through
                    result[i, outputStart] = data[i, col];
                    continue;
                }

                double value = NumOps.ToDouble(data[i, col]);

                // Hash the value
                int hash = ComputeHash(col, value);
                int bucket = Math.Abs(hash) % _nComponents;

                // Determine sign (for alternate sign mode)
                double sign = 1.0;
                if (_alternateSign)
                {
                    sign = (hash >= 0) ? 1.0 : -1.0;
                }

                // Add to the appropriate bucket
                double currentVal = NumOps.ToDouble(result[i, outputStart + bucket]);
                result[i, outputStart + bucket] = NumOps.FromDouble(currentVal + sign);
            }
        }

        return new Matrix<T>(result);
    }

    private int ComputeHash(int columnIndex, double value)
    {
        // Combine column index and value for unique hash
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + columnIndex;
            hash = hash * 31 + value.GetHashCode();
            return hash;
        }
    }

    /// <summary>
    /// Inverse transformation is not supported for hash encoding.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HashingEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_featureIndicesStart is null)
        {
            return Array.Empty<string>();
        }

        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        var names = new List<string>();

        for (int col = 0; col < _nInputFeatures; col++)
        {
            string baseName = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";

            if (!processSet.Contains(col))
            {
                names.Add(baseName);
                continue;
            }

            for (int h = 0; h < _nComponents; h++)
            {
                names.Add($"{baseName}_hash{h}");
            }
        }

        return names.ToArray();
    }
}
