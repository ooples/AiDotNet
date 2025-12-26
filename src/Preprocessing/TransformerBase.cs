using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Abstract base class for all data transformers providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This class provides the template method pattern for data transformation.
/// Derived classes implement the core fitting and transformation logic while
/// this base class handles validation, state management, and common operations.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all transformers build on.
/// It provides common features like:
/// - Checking if the transformer is ready to use
/// - Managing which columns to transform
/// - Serialization for saving/loading transformers
///
/// When creating a new transformer, you extend this class and implement the abstract methods.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type after transformation.</typeparam>
public abstract class TransformerBase<T, TInput, TOutput> : IDataTransformer<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the numeric operations helper for type T.
    /// </summary>
    protected INumericOperations<T> NumOps { get; }

    /// <summary>
    /// Gets the computational engine for tensor operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets whether this transformer has been fitted to data.
    /// </summary>
    public bool IsFitted { get; protected set; }

    /// <summary>
    /// Gets the column indices this transformer operates on.
    /// </summary>
    public int[]? ColumnIndices { get; }

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public abstract bool SupportsInverseTransform { get; }

    /// <summary>
    /// Creates a new instance of the transformer.
    /// </summary>
    /// <param name="columnIndices">The column indices to operate on, or null for all columns.</param>
    protected TransformerBase(int[]? columnIndices = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        ColumnIndices = columnIndices;
        IsFitted = false;
    }

    /// <summary>
    /// Fits the transformer to the training data.
    /// </summary>
    /// <param name="data">The training data to fit.</param>
    public void Fit(TInput data)
    {
        ValidateInputData(data);
        FitCore(data);
        IsFitted = true;
    }

    /// <summary>
    /// Transforms the input data using fitted parameters.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    /// <exception cref="InvalidOperationException">Thrown if Fit() has not been called.</exception>
    public TOutput Transform(TInput data)
    {
        EnsureFitted();
        ValidateInputData(data);
        return TransformCore(data);
    }

    /// <summary>
    /// Fits the transformer and transforms the data in a single step.
    /// </summary>
    /// <param name="data">The data to fit and transform.</param>
    /// <returns>The transformed data.</returns>
    public TOutput FitTransform(TInput data)
    {
        Fit(data);
        return TransformCore(data);
    }

    /// <summary>
    /// Reverses the transformation.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    /// <exception cref="NotSupportedException">Thrown if inverse transform is not supported.</exception>
    /// <exception cref="InvalidOperationException">Thrown if Fit() has not been called.</exception>
    public TInput InverseTransform(TOutput data)
    {
        if (!SupportsInverseTransform)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support inverse transformation.");
        }

        EnsureFitted();
        return InverseTransformCore(data);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names (optional).</param>
    /// <returns>The output feature names.</returns>
    public virtual string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        // Default implementation returns input names or generic names
        if (inputFeatureNames is not null)
        {
            return inputFeatureNames;
        }

        // If we don't know the number of features, return empty
        return Array.Empty<string>();
    }

    /// <summary>
    /// Core fitting implementation. Override this in derived classes.
    /// </summary>
    /// <param name="data">The training data to fit.</param>
    protected abstract void FitCore(TInput data);

    /// <summary>
    /// Core transformation implementation. Override this in derived classes.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected abstract TOutput TransformCore(TInput data);

    /// <summary>
    /// Core inverse transformation implementation. Override this in derived classes.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    protected virtual TInput InverseTransformCore(TOutput data)
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not support inverse transformation.");
    }

    /// <summary>
    /// Validates input data before fitting or transforming.
    /// </summary>
    /// <param name="data">The data to validate.</param>
    /// <exception cref="ArgumentNullException">Thrown if data is null.</exception>
    protected virtual void ValidateInputData(TInput data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data), "Input data cannot be null.");
        }
    }

    /// <summary>
    /// Ensures the transformer has been fitted before transformation.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if not fitted.</exception>
    protected void EnsureFitted()
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} has not been fitted. Call Fit() or FitTransform() first.");
        }
    }

    /// <summary>
    /// Gets the indices to operate on, defaulting to all if ColumnIndices is null.
    /// </summary>
    /// <param name="totalColumns">The total number of columns in the data.</param>
    /// <returns>The column indices to process.</returns>
    protected int[] GetColumnsToProcess(int totalColumns)
    {
        if (ColumnIndices is null || ColumnIndices.Length == 0)
        {
            // Process all columns
            var indices = new int[totalColumns];
            for (int i = 0; i < totalColumns; i++)
            {
                indices[i] = i;
            }
            return indices;
        }

        // Validate indices are within bounds
        foreach (var idx in ColumnIndices)
        {
            if (idx < 0 || idx >= totalColumns)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(ColumnIndices),
                    $"Column index {idx} is out of range. Total columns: {totalColumns}.");
            }
        }

        return ColumnIndices;
    }
}
