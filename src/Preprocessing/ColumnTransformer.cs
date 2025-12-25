using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Applies different transformers to different columns of the input.
/// </summary>
/// <remarks>
/// <para>
/// ColumnTransformer allows you to specify which transformer should be applied
/// to which columns. This is useful when different columns require different
/// preprocessing (e.g., scaling numeric columns, encoding categorical columns).
/// </para>
/// <para>
/// Output columns are concatenated in the order transformers are added.
/// Columns not specified in any transformer can be passed through or dropped.
/// </para>
/// <para><b>For Beginners:</b> Different columns often need different treatment:
/// - Numeric columns: scaling, normalization
/// - Categorical columns: one-hot encoding
/// - Text columns: vectorization
///
/// ColumnTransformer lets you apply the right transformation to each column type.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ColumnTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly List<(string Name, IDataTransformer<T, Matrix<T>, Matrix<T>> Transformer, int[] Columns)> _transformers;
    private readonly ColumnTransformerRemainder _remainder;

    // Fitted parameters
    private int _nInputFeatures;
    private int _nOutputFeatures;
    private int[]? _remainderColumns;
    private List<(int OutputStart, int OutputEnd, int TransformerIndex)>? _outputMapping;

    /// <summary>
    /// Gets how columns not specified in any transformer are handled.
    /// </summary>
    public ColumnTransformerRemainder Remainder => _remainder;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="ColumnTransformer{T}"/>.
    /// </summary>
    /// <param name="remainder">How to handle columns not specified in any transformer. Defaults to Drop.</param>
    public ColumnTransformer(ColumnTransformerRemainder remainder = ColumnTransformerRemainder.Drop)
        : base(null)
    {
        _transformers = new List<(string, IDataTransformer<T, Matrix<T>, Matrix<T>>, int[])>();
        _remainder = remainder;
    }

    /// <summary>
    /// Adds a transformer to be applied to specific columns.
    /// </summary>
    /// <param name="name">Name identifier for the transformer.</param>
    /// <param name="transformer">The transformer to apply.</param>
    /// <param name="columns">The column indices to apply this transformer to.</param>
    /// <returns>This instance for method chaining.</returns>
    public ColumnTransformer<T> Add(string name, IDataTransformer<T, Matrix<T>, Matrix<T>> transformer, int[] columns)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Transformer name cannot be null or empty.", nameof(name));
        }

        if (transformer is null)
        {
            throw new ArgumentNullException(nameof(transformer));
        }

        if (columns is null || columns.Length == 0)
        {
            throw new ArgumentException("Columns cannot be null or empty.", nameof(columns));
        }

        _transformers.Add((name, transformer, columns));
        return this;
    }

    /// <summary>
    /// Adds a transformer to be applied to specific columns.
    /// </summary>
    /// <param name="transformer">The transformer to apply.</param>
    /// <param name="columns">The column indices to apply this transformer to.</param>
    /// <returns>This instance for method chaining.</returns>
    public ColumnTransformer<T> Add(IDataTransformer<T, Matrix<T>, Matrix<T>> transformer, int[] columns)
    {
        return Add($"transformer_{_transformers.Count}", transformer, columns);
    }

    /// <summary>
    /// Fits all transformers to their respective columns.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;

        // Validate column indices
        var usedColumns = new HashSet<int>();
        foreach (var (name, transformer, columns) in _transformers)
        {
            foreach (int col in columns)
            {
                if (col < 0 || col >= _nInputFeatures)
                {
                    throw new ArgumentException($"Column index {col} is out of range for transformer '{name}'.");
                }
                usedColumns.Add(col);
            }
        }

        // Determine remainder columns
        _remainderColumns = Enumerable.Range(0, _nInputFeatures)
            .Where(c => !usedColumns.Contains(c))
            .ToArray();

        // Fit each transformer on its subset of columns
        _outputMapping = new List<(int, int, int)>();
        int outputIdx = 0;

        for (int i = 0; i < _transformers.Count; i++)
        {
            var (name, transformer, columns) = _transformers[i];

            // Extract subset of columns
            var subset = ExtractColumns(data, columns);

            // Fit transformer
            transformer.Fit(subset);

            // Transform to get output dimensions
            var transformed = transformer.Transform(subset);

            int outputCols = transformed.Columns;
            _outputMapping.Add((outputIdx, outputIdx + outputCols, i));
            outputIdx += outputCols;
        }

        // Add remainder columns if passthrough
        if (_remainder == ColumnTransformerRemainder.Passthrough && _remainderColumns.Length > 0)
        {
            outputIdx += _remainderColumns.Length;
        }

        _nOutputFeatures = outputIdx;
    }

    private Matrix<T> ExtractColumns(Matrix<T> data, int[] columns)
    {
        int numRows = data.Rows;
        var result = new T[numRows, columns.Length];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < columns.Length; j++)
            {
                result[i, j] = data[i, columns[j]];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms the data by applying each transformer to its columns.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data with all outputs concatenated.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_outputMapping is null || _remainderColumns is null)
        {
            throw new InvalidOperationException("ColumnTransformer has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nOutputFeatures];

        // Initialize with zeros
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < _nOutputFeatures; j++)
            {
                result[i, j] = NumOps.Zero;
            }
        }

        // Apply each transformer
        foreach (var (outputStart, outputEnd, transformerIndex) in _outputMapping)
        {
            var (name, transformer, columns) = _transformers[transformerIndex];

            var subset = ExtractColumns(data, columns);
            var transformed = transformer.Transform(subset);

            // Copy to output
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < transformed.Columns; j++)
                {
                    result[i, outputStart + j] = transformed[i, j];
                }
            }
        }

        // Add remainder columns if passthrough
        if (_remainder == ColumnTransformerRemainder.Passthrough && _remainderColumns.Length > 0)
        {
            int remainderStart = _nOutputFeatures - _remainderColumns.Length;
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < _remainderColumns.Length; j++)
                {
                    result[i, remainderStart + j] = data[i, _remainderColumns[j]];
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for ColumnTransformer.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ColumnTransformer does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_outputMapping is null || _remainderColumns is null)
        {
            return Array.Empty<string>();
        }

        var names = new List<string>();

        foreach (var (outputStart, outputEnd, transformerIndex) in _outputMapping)
        {
            var (name, transformer, columns) = _transformers[transformerIndex];

            // Get input names for this transformer's columns
            string[]? subsetNames = null;
            if (inputFeatureNames is not null)
            {
                subsetNames = columns.Select(c => c < inputFeatureNames.Length ? inputFeatureNames[c] : $"x{c}").ToArray();
            }

            var outNames = transformer.GetFeatureNamesOut(subsetNames);
            names.AddRange(outNames);
        }

        // Add remainder columns if passthrough
        if (_remainder == ColumnTransformerRemainder.Passthrough && _remainderColumns.Length > 0)
        {
            foreach (int col in _remainderColumns)
            {
                string colName = inputFeatureNames is not null && col < inputFeatureNames.Length
                    ? inputFeatureNames[col]
                    : $"x{col}";
                names.Add(colName);
            }
        }

        return names.ToArray();
    }

    /// <summary>
    /// Gets the transformer with the specified name.
    /// </summary>
    /// <param name="name">The transformer name.</param>
    /// <returns>The transformer if found, null otherwise.</returns>
    public IDataTransformer<T, Matrix<T>, Matrix<T>>? GetTransformer(string name)
    {
        foreach (var (n, transformer, _) in _transformers)
        {
            if (n == name) return transformer;
        }
        return null;
    }

    /// <summary>
    /// Gets all transformer names.
    /// </summary>
    /// <returns>Array of transformer names.</returns>
    public string[] GetTransformerNames()
    {
        return _transformers.Select(t => t.Name).ToArray();
    }
}

/// <summary>
/// Specifies how to handle columns not specified in any transformer.
/// </summary>
public enum ColumnTransformerRemainder
{
    /// <summary>
    /// Drop columns not specified in any transformer.
    /// </summary>
    Drop,

    /// <summary>
    /// Pass through columns not specified in any transformer unchanged.
    /// </summary>
    Passthrough
}
