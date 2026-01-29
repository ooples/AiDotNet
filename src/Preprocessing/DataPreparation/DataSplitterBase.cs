using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Base class for all data splitters providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This base class provides shared utilities that all data splitters need:
/// - Shuffling indices randomly
/// - Selecting rows from matrices
/// - Validating inputs
/// - Working with both Matrix and Tensor data
/// </para>
/// <para>
/// When creating a custom splitter, inherit from this class to get these utilities for free.
/// You only need to implement the specific splitting logic for your algorithm.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public abstract class DataSplitterBase<T> : IDataSplitter<T>
{
    /// <summary>
    /// Numeric operations for generic type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The random seed for reproducible splits.
    /// </summary>
    protected readonly int _randomSeed;

    /// <summary>
    /// Random number generator initialized with the seed.
    /// </summary>
    protected readonly Random _random;

    /// <summary>
    /// Whether to shuffle data before splitting.
    /// </summary>
    protected readonly bool _shuffle;

    /// <summary>
    /// Initializes a new instance of the DataSplitterBase class.
    /// </summary>
    /// <param name="shuffle">Whether to shuffle data before splitting. Default is true for random splits.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42 (a common convention).</param>
    protected DataSplitterBase(bool shuffle = true, int randomSeed = 42)
    {
        _shuffle = shuffle;
        _randomSeed = randomSeed;
        _random = RandomHelper.CreateSeededRandom(randomSeed);
    }

    /// <inheritdoc/>
    public abstract DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null);

    /// <inheritdoc/>
    public virtual TensorSplitResult<T> SplitTensor(Tensor<T> X, Tensor<T>? y = null)
    {
        ValidateTensorInputs(X, y);

        // Get indices for splitting
        var matrixResult = SplitIndicesOnly(X.Shape[0], null);

        // Build tensor result from indices
        return BuildTensorResult(X, y, matrixResult.TrainIndices, matrixResult.TestIndices,
            matrixResult.ValidationIndices, matrixResult.FoldIndex, matrixResult.TotalFolds);
    }

    /// <inheritdoc/>
    public virtual IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        // Default implementation: single split
        yield return Split(X, y);
    }

    /// <inheritdoc/>
    public virtual IEnumerable<TensorSplitResult<T>> GetTensorSplits(Tensor<T> X, Tensor<T>? y = null)
    {
        // Default implementation: single split
        yield return SplitTensor(X, y);
    }

    /// <inheritdoc/>
    public virtual int NumSplits => 1;

    /// <inheritdoc/>
    public virtual bool RequiresLabels => false;

    /// <inheritdoc/>
    public virtual bool SupportsValidation => false;

    /// <inheritdoc/>
    public abstract string Description { get; }

    #region Helper Methods

    /// <summary>
    /// Creates an array of indices from 0 to count-1.
    /// </summary>
    /// <param name="count">The number of indices to create.</param>
    /// <returns>An array [0, 1, 2, ..., count-1].</returns>
    protected int[] GetIndices(int count)
    {
        return Enumerable.Range(0, count).ToArray();
    }

    /// <summary>
    /// Shuffles an array of indices in place using Fisher-Yates algorithm.
    /// </summary>
    /// <param name="indices">The indices to shuffle.</param>
    protected void ShuffleIndices(int[] indices)
    {
        // Fisher-Yates shuffle
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }

    /// <summary>
    /// Gets indices, optionally shuffled.
    /// </summary>
    /// <param name="count">Number of indices.</param>
    /// <returns>Array of indices, shuffled if _shuffle is true.</returns>
    protected int[] GetShuffledIndices(int count)
    {
        var indices = GetIndices(count);
        if (_shuffle)
        {
            ShuffleIndices(indices);
        }
        return indices;
    }

    /// <summary>
    /// Validates that the input matrix and optional vector are compatible.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The optional target vector.</param>
    /// <exception cref="ArgumentNullException">If X is null.</exception>
    /// <exception cref="ArgumentException">If X and y have different row counts.</exception>
    protected void ValidateInputs(Matrix<T> X, Vector<T>? y)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X), "Feature matrix cannot be null.");
        }

        if (X.Rows == 0)
        {
            throw new ArgumentException("Feature matrix cannot be empty.", nameof(X));
        }

        if (y != null && y.Length != X.Rows)
        {
            throw new ArgumentException(
                $"X has {X.Rows} rows but y has {y.Length} elements. They must match.",
                nameof(y));
        }

        if (RequiresLabels && y is null)
        {
            throw new ArgumentNullException(nameof(y),
                $"This splitter ({GetType().Name}) requires target labels (y) to be provided.");
        }
    }

    /// <summary>
    /// Validates tensor inputs.
    /// </summary>
    /// <param name="X">The feature tensor.</param>
    /// <param name="y">The optional target tensor.</param>
    protected void ValidateTensorInputs(Tensor<T> X, Tensor<T>? y)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X), "Feature tensor cannot be null.");
        }

        if (X.Shape[0] == 0)
        {
            throw new ArgumentException("Feature tensor cannot be empty.", nameof(X));
        }

        if (y != null && y.Shape[0] != X.Shape[0])
        {
            throw new ArgumentException(
                $"X has {X.Shape[0]} samples but y has {y.Shape[0]} samples. They must match.",
                nameof(y));
        }

        if (RequiresLabels && y is null)
        {
            throw new ArgumentNullException(nameof(y),
                $"This splitter ({GetType().Name}) requires target labels (y) to be provided.");
        }
    }

    /// <summary>
    /// Selects specific rows from a matrix.
    /// </summary>
    /// <param name="X">The source matrix.</param>
    /// <param name="indices">The row indices to select.</param>
    /// <returns>A new matrix containing only the selected rows.</returns>
    protected Matrix<T> SelectRows(Matrix<T> X, int[] indices)
    {
        var result = new Matrix<T>(indices.Length, X.Columns);
        for (int i = 0; i < indices.Length; i++)
        {
            result.SetRow(i, X.GetRow(indices[i]));
        }
        return result;
    }

    /// <summary>
    /// Selects specific elements from a vector.
    /// </summary>
    /// <param name="y">The source vector.</param>
    /// <param name="indices">The indices to select.</param>
    /// <returns>A new vector containing only the selected elements.</returns>
    protected Vector<T> SelectElements(Vector<T> y, int[] indices)
    {
        var result = new Vector<T>(indices.Length);
        for (int i = 0; i < indices.Length; i++)
        {
            result[i] = y[indices[i]];
        }
        return result;
    }

    /// <summary>
    /// Selects specific samples from a tensor.
    /// </summary>
    /// <param name="X">The source tensor.</param>
    /// <param name="indices">The sample indices to select (along first dimension).</param>
    /// <returns>A new tensor containing only the selected samples.</returns>
    protected Tensor<T> SelectSamples(Tensor<T> X, int[] indices)
    {
        // Create new shape with updated first dimension
        int[] newShape = (int[])X.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        // Copy each selected sample
        for (int i = 0; i < indices.Length; i++)
        {
            CopySample(X, result, indices[i], i);
        }

        return result;
    }

    /// <summary>
    /// Copies a single sample from source tensor to destination tensor.
    /// </summary>
    /// <param name="source">Source tensor.</param>
    /// <param name="destination">Destination tensor.</param>
    /// <param name="sourceIndex">Index in source tensor.</param>
    /// <param name="destIndex">Index in destination tensor.</param>
    protected void CopySample(Tensor<T> source, Tensor<T> destination, int sourceIndex, int destIndex)
    {
        CopySampleRecursive(source, destination, sourceIndex, destIndex, 1, new int[source.Rank]);
    }

    private void CopySampleRecursive(
        Tensor<T> source,
        Tensor<T> destination,
        int sourceIndex,
        int destIndex,
        int currentDim,
        int[] indices)
    {
        if (currentDim == source.Rank)
        {
            indices[0] = sourceIndex;
            T value = source[indices];
            indices[0] = destIndex;
            destination[indices] = value;
        }
        else
        {
            for (int i = 0; i < source.Shape[currentDim]; i++)
            {
                indices[currentDim] = i;
                CopySampleRecursive(source, destination, sourceIndex, destIndex, currentDim + 1, indices);
            }
        }
    }

    /// <summary>
    /// Builds a DataSplitResult from computed indices.
    /// </summary>
    protected DataSplitResult<T> BuildResult(
        Matrix<T> X,
        Vector<T>? y,
        int[] trainIndices,
        int[] testIndices,
        int[]? validationIndices = null,
        int? foldIndex = null,
        int? totalFolds = null,
        int? repeatIndex = null,
        int? totalRepeats = null)
    {
        return new DataSplitResult<T>
        {
            XTrain = SelectRows(X, trainIndices),
            XTest = SelectRows(X, testIndices),
            yTrain = y != null ? SelectElements(y, trainIndices) : null,
            yTest = y != null ? SelectElements(y, testIndices) : null,
            XValidation = validationIndices != null ? SelectRows(X, validationIndices) : null,
            yValidation = validationIndices != null && y != null ? SelectElements(y, validationIndices) : null,
            TrainIndices = trainIndices,
            TestIndices = testIndices,
            ValidationIndices = validationIndices,
            FoldIndex = foldIndex,
            TotalFolds = totalFolds,
            RepeatIndex = repeatIndex,
            TotalRepeats = totalRepeats
        };
    }

    /// <summary>
    /// Builds a TensorSplitResult from computed indices.
    /// </summary>
    protected TensorSplitResult<T> BuildTensorResult(
        Tensor<T> X,
        Tensor<T>? y,
        int[] trainIndices,
        int[] testIndices,
        int[]? validationIndices = null,
        int? foldIndex = null,
        int? totalFolds = null,
        int? repeatIndex = null,
        int? totalRepeats = null)
    {
        return new TensorSplitResult<T>
        {
            XTrain = SelectSamples(X, trainIndices),
            XTest = SelectSamples(X, testIndices),
            yTrain = y != null ? SelectSamples(y, trainIndices) : null,
            yTest = y != null ? SelectSamples(y, testIndices) : null,
            XValidation = validationIndices != null ? SelectSamples(X, validationIndices) : null,
            yValidation = validationIndices != null && y != null ? SelectSamples(y, validationIndices) : null,
            TrainIndices = trainIndices,
            TestIndices = testIndices,
            ValidationIndices = validationIndices,
            FoldIndex = foldIndex,
            TotalFolds = totalFolds,
            RepeatIndex = repeatIndex,
            TotalRepeats = totalRepeats
        };
    }

    /// <summary>
    /// Helper method to get split indices without building full results.
    /// Useful for tensor splitting where we want to reuse the same indices.
    /// </summary>
    protected virtual (int[] TrainIndices, int[] TestIndices, int[]? ValidationIndices, int? FoldIndex, int? TotalFolds)
        SplitIndicesOnly(int nSamples, Vector<T>? y)
    {
        // Default implementation for simple train/test split
        // Subclasses should override if they have different logic
        var indices = GetShuffledIndices(nSamples);
        int testSize = Math.Max(1, (int)(nSamples * 0.2)); // Default 20% test
        int trainSize = nSamples - testSize;

        var trainIndices = indices.Take(trainSize).ToArray();
        var testIndices = indices.Skip(trainSize).ToArray();

        return (trainIndices, testIndices, null, null, null);
    }

    /// <summary>
    /// Gets unique class labels from a target vector.
    /// </summary>
    /// <param name="y">The target vector.</param>
    /// <returns>Array of unique labels.</returns>
    protected T[] GetUniqueLabels(Vector<T> y)
    {
        var uniqueSet = new HashSet<double>();
        var labels = new List<T>();

        for (int i = 0; i < y.Length; i++)
        {
            double key = Convert.ToDouble(y[i]);
            if (uniqueSet.Add(key))
            {
                labels.Add(y[i]);
            }
        }

        return labels.ToArray();
    }

    /// <summary>
    /// Groups sample indices by their class label.
    /// </summary>
    /// <param name="y">The target vector.</param>
    /// <returns>Dictionary mapping label to list of indices with that label.</returns>
    protected Dictionary<double, List<int>> GroupByLabel(Vector<T> y)
    {
        var groups = new Dictionary<double, List<int>>();

        for (int i = 0; i < y.Length; i++)
        {
            double key = Convert.ToDouble(y[i]);
            if (!groups.TryGetValue(key, out var list))
            {
                list = new List<int>();
                groups[key] = list;
            }
            list.Add(i);
        }

        return groups;
    }

    /// <summary>
    /// Computes sizes for train/validation/test splits based on ratios.
    /// </summary>
    /// <param name="total">Total number of samples.</param>
    /// <param name="trainRatio">Ratio for training set.</param>
    /// <param name="validationRatio">Ratio for validation set (0 for no validation).</param>
    /// <returns>Tuple of (trainSize, validationSize, testSize).</returns>
    protected (int trainSize, int validationSize, int testSize) ComputeSplitSizes(
        int total, double trainRatio, double validationRatio = 0)
    {
        int trainSize = (int)(total * trainRatio);
        int validationSize = (int)(total * validationRatio);
        int testSize = total - trainSize - validationSize;

        // Ensure at least 1 sample in each set
        if (trainSize == 0) trainSize = 1;
        if (testSize == 0) testSize = 1;

        // Rebalance if needed
        if (trainSize + validationSize + testSize != total)
        {
            testSize = total - trainSize - validationSize;
        }

        return (trainSize, validationSize, testSize);
    }

    #endregion
}
