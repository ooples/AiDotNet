using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureGeneration;

/// <summary>
/// Applies a custom function to transform data.
/// </summary>
/// <remarks>
/// <para>
/// FunctionTransformer allows you to apply arbitrary functions to your data as part
/// of a preprocessing pipeline. This is useful for applying domain-specific transformations
/// or wrapping legacy code into the transformer API.
/// </para>
/// <para><b>For Beginners:</b> Sometimes you need a custom transformation that doesn't
/// fit standard transformers. FunctionTransformer lets you plug in your own function:
/// - Apply a mathematical formula to all values
/// - Perform domain-specific feature engineering
/// - Wrap existing transformation code
///
/// Example: Apply a custom normalization formula or domain-specific scaling.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class FunctionTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly Func<Matrix<T>, Matrix<T>>? _func;
    private readonly Func<Matrix<T>, Matrix<T>>? _inverseFunc;
    private readonly Func<double, double>? _elementFunc;
    private readonly Func<double, double>? _inverseElementFunc;
    private readonly bool _validate;

    private int _nInputFeatures;

    /// <summary>
    /// Gets whether this transformer validates input/output shapes.
    /// </summary>
    public bool Validate => _validate;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => _inverseFunc is not null || _inverseElementFunc is not null;

    /// <summary>
    /// Creates a new instance of <see cref="FunctionTransformer{T}"/> with matrix-level functions.
    /// </summary>
    /// <param name="func">The function to apply to the entire matrix. If null, acts as identity.</param>
    /// <param name="inverseFunc">The inverse function (optional).</param>
    /// <param name="validate">If true, validates that output has same number of rows as input.</param>
    /// <param name="columnIndices">The column indices to transform, or null for all columns.</param>
    public FunctionTransformer(
        Func<Matrix<T>, Matrix<T>>? func = null,
        Func<Matrix<T>, Matrix<T>>? inverseFunc = null,
        bool validate = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _func = func;
        _inverseFunc = inverseFunc;
        _validate = validate;
    }

    /// <summary>
    /// Creates a new instance of <see cref="FunctionTransformer{T}"/> with element-wise functions.
    /// </summary>
    /// <param name="elementFunc">The function to apply to each element.</param>
    /// <param name="inverseElementFunc">The inverse function for each element (optional).</param>
    /// <param name="validate">If true, validates shapes.</param>
    /// <param name="columnIndices">The column indices to transform, or null for all columns.</param>
    public static FunctionTransformer<T> FromElementFunction(
        Func<double, double> elementFunc,
        Func<double, double>? inverseElementFunc = null,
        bool validate = true,
        int[]? columnIndices = null)
    {
        return new FunctionTransformer<T>(elementFunc, inverseElementFunc, validate, columnIndices);
    }

    // Private constructor for element-wise functions
    private FunctionTransformer(
        Func<double, double> elementFunc,
        Func<double, double>? inverseElementFunc,
        bool validate,
        int[]? columnIndices)
        : base(columnIndices)
    {
        _elementFunc = elementFunc;
        _inverseElementFunc = inverseElementFunc;
        _validate = validate;
    }

    /// <summary>
    /// Stores the input dimensions for validation.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
    }

    /// <summary>
    /// Applies the transformation function to the data.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        Matrix<T> result;

        if (_elementFunc is not null)
        {
            // Element-wise transformation
            result = ApplyElementWise(data, _elementFunc);
        }
        else if (_func is not null)
        {
            // Matrix-level transformation
            var columnsToProcess = GetColumnsToProcess(data.Columns);

            if (columnsToProcess.Length == data.Columns && ColumnIndices is null)
            {
                // Apply to entire matrix
                result = _func(data);
            }
            else
            {
                // Apply only to specified columns
                result = ApplyToColumns(data, columnsToProcess, _func);
            }
        }
        else
        {
            // Identity function - return copy
            result = CopyMatrix(data);
        }

        if (_validate && result.Rows != data.Rows)
        {
            throw new InvalidOperationException(
                $"Function changed number of rows from {data.Rows} to {result.Rows}. " +
                "Set validate=false if this is intentional.");
        }

        return result;
    }

    /// <summary>
    /// Applies the inverse transformation function to the data.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_inverseElementFunc is not null)
        {
            return ApplyElementWise(data, _inverseElementFunc);
        }

        if (_inverseFunc is not null)
        {
            var columnsToProcess = GetColumnsToProcess(data.Columns);

            if (columnsToProcess.Length == data.Columns && ColumnIndices is null)
            {
                return _inverseFunc(data);
            }

            return ApplyToColumns(data, columnsToProcess, _inverseFunc);
        }

        throw new NotSupportedException("FunctionTransformer was not configured with an inverse function.");
    }

    private Matrix<T> ApplyElementWise(Matrix<T> data, Func<double, double> func)
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
                if (processSet.Contains(j))
                {
                    double value = NumOps.ToDouble(data[i, j]);
                    result[i, j] = NumOps.FromDouble(func(value));
                }
                else
                {
                    result[i, j] = data[i, j];
                }
            }
        }

        return new Matrix<T>(result);
    }

    private Matrix<T> ApplyToColumns(Matrix<T> data, int[] columnsToProcess, Func<Matrix<T>, Matrix<T>> func)
    {
        // Extract columns to process
        int numRows = data.Rows;
        int numColsToProcess = columnsToProcess.Length;
        var subset = new T[numRows, numColsToProcess];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColsToProcess; j++)
            {
                subset[i, j] = data[i, columnsToProcess[j]];
            }
        }

        // Apply function to subset
        var transformedSubset = func(new Matrix<T>(subset));

        // Reconstruct full matrix
        var result = new T[numRows, data.Columns];
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            int subsetCol = 0;
            for (int j = 0; j < data.Columns; j++)
            {
                if (processSet.Contains(j))
                {
                    result[i, j] = transformedSubset[i, subsetCol++];
                }
                else
                {
                    result[i, j] = data[i, j];
                }
            }
        }

        return new Matrix<T>(result);
    }

    private Matrix<T> CopyMatrix(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                result[i, j] = data[i, j];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Creates a FunctionTransformer that applies the natural logarithm.
    /// </summary>
    public static FunctionTransformer<T> Log()
    {
        return FromElementFunction(
            x => Math.Log(x),
            x => Math.Exp(x));
    }

    /// <summary>
    /// Creates a FunctionTransformer that applies log(1 + x).
    /// </summary>
    public static FunctionTransformer<T> Log1p()
    {
        return FromElementFunction(
            x => Math.Log(1 + x),
            x => Math.Exp(x) - 1);
    }

    /// <summary>
    /// Creates a FunctionTransformer that applies the square root.
    /// </summary>
    public static FunctionTransformer<T> Sqrt()
    {
        return FromElementFunction(
            x => Math.Sqrt(x),
            x => x * x);
    }

    /// <summary>
    /// Creates a FunctionTransformer that applies the exponential function.
    /// </summary>
    public static FunctionTransformer<T> Exp()
    {
        return FromElementFunction(
            x => Math.Exp(x),
            x => Math.Log(x));
    }

    /// <summary>
    /// Creates a FunctionTransformer that raises to a power.
    /// </summary>
    /// <param name="power">The power to raise values to.</param>
    public static FunctionTransformer<T> Power(double power)
    {
        return FromElementFunction(
            x => Math.Pow(x, power),
            x => Math.Pow(x, 1.0 / power));
    }

    /// <summary>
    /// Creates a FunctionTransformer that applies the absolute value.
    /// </summary>
    public static FunctionTransformer<T> Abs()
    {
        return FromElementFunction(
            x => Math.Abs(x),
            null); // No unique inverse for absolute value
    }

    /// <summary>
    /// Creates a FunctionTransformer that applies the sign function.
    /// </summary>
    public static FunctionTransformer<T> Sign()
    {
        return FromElementFunction(
            x => Math.Sign(x),
            null); // No inverse for sign function
    }

    /// <summary>
    /// Creates a FunctionTransformer that clips values to a range.
    /// </summary>
    /// <param name="min">Minimum value.</param>
    /// <param name="max">Maximum value.</param>
    public static FunctionTransformer<T> Clip(double min, double max)
    {
        return FromElementFunction(
            x => Math.Max(min, Math.Min(max, x)),
            null); // No inverse for clipping
    }
}
