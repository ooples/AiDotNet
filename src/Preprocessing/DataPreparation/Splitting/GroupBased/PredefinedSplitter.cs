using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased;

/// <summary>
/// User-specified split where indices for train/test are provided directly.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sometimes you know exactly which samples should go where.
/// This splitter lets you specify the exact indices for training, testing,
/// and optionally validation.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Reproducing a specific published split
/// - Domain-specific splitting requirements
/// - When automatic splitting isn't appropriate
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PredefinedSplitter<T> : DataSplitterBase<T>
{
    private readonly int[] _trainIndices;
    private readonly int[] _testIndices;
    private readonly int[]? _validationIndices;

    /// <summary>
    /// Creates a new predefined splitter with specific indices.
    /// </summary>
    /// <param name="trainIndices">Indices for training set.</param>
    /// <param name="testIndices">Indices for test set.</param>
    /// <param name="validationIndices">Optional indices for validation set.</param>
    public PredefinedSplitter(int[] trainIndices, int[] testIndices, int[]? validationIndices = null)
        : base(shuffle: false, randomSeed: 42)
    {
        if (trainIndices is null || trainIndices.Length == 0)
        {
            throw new ArgumentNullException(nameof(trainIndices), "Train indices cannot be null or empty.");
        }

        if (testIndices is null || testIndices.Length == 0)
        {
            throw new ArgumentNullException(nameof(testIndices), "Test indices cannot be null or empty.");
        }

        // Check for overlaps
        var trainSet = new HashSet<int>(trainIndices);
        foreach (int idx in testIndices)
        {
            if (trainSet.Contains(idx))
            {
                throw new ArgumentException($"Index {idx} appears in both train and test sets.");
            }
        }

        if (validationIndices != null)
        {
            foreach (int idx in validationIndices)
            {
                if (trainSet.Contains(idx))
                {
                    throw new ArgumentException($"Index {idx} appears in both train and validation sets.");
                }
                if (testIndices.Contains(idx))
                {
                    throw new ArgumentException($"Index {idx} appears in both test and validation sets.");
                }
            }
        }

        _trainIndices = trainIndices;
        _testIndices = testIndices;
        _validationIndices = validationIndices;
    }

    /// <inheritdoc/>
    public override bool SupportsValidation => _validationIndices != null;

    /// <inheritdoc/>
    public override string Description =>
        _validationIndices != null
            ? $"Predefined split (train={_trainIndices.Length}, val={_validationIndices.Length}, test={_testIndices.Length})"
            : $"Predefined split (train={_trainIndices.Length}, test={_testIndices.Length})";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        // Validate indices are in range
        int maxIdx = X.Rows - 1;
        foreach (int idx in _trainIndices.Concat(_testIndices).Concat(_validationIndices ?? Array.Empty<int>()))
        {
            if (idx < 0 || idx > maxIdx)
            {
                throw new ArgumentException($"Index {idx} is out of range [0, {maxIdx}].");
            }
        }

        return BuildResult(X, y, _trainIndices, _testIndices, _validationIndices);
    }
}
