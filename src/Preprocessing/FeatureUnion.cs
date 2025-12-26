using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Concatenates results from multiple transformers horizontally.
/// </summary>
/// <remarks>
/// <para>
/// FeatureUnion applies multiple transformers to the same input data and
/// concatenates their outputs into a single feature matrix. This is useful
/// for combining different feature extraction methods.
/// </para>
/// <para>
/// Each transformer receives the full input matrix and produces its own
/// output. All outputs are then concatenated column-wise.
/// </para>
/// <para><b>For Beginners:</b> Sometimes you want multiple feature sets from the same data:
/// - Polynomial features from numeric columns
/// - Statistics (mean, std) from time windows
/// - Both PCA and manual feature engineering
///
/// FeatureUnion runs all transformers and combines their outputs side by side.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class FeatureUnion<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly List<(string Name, IDataTransformer<T, Matrix<T>, Matrix<T>> Transformer)> _transformers;

    // Fitted parameters
    private int _nInputFeatures;
    private int _nOutputFeatures;
    private List<int>? _outputWidths;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="FeatureUnion{T}"/>.
    /// </summary>
    public FeatureUnion()
        : base(null)
    {
        _transformers = new List<(string, IDataTransformer<T, Matrix<T>, Matrix<T>>)>();
    }

    /// <summary>
    /// Adds a transformer to the union.
    /// </summary>
    /// <param name="name">Name identifier for the transformer.</param>
    /// <param name="transformer">The transformer to add.</param>
    /// <returns>This instance for method chaining.</returns>
    public FeatureUnion<T> Add(string name, IDataTransformer<T, Matrix<T>, Matrix<T>> transformer)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Transformer name cannot be null or empty.", nameof(name));
        }

        if (transformer is null)
        {
            throw new ArgumentNullException(nameof(transformer));
        }

        _transformers.Add((name, transformer));
        return this;
    }

    /// <summary>
    /// Adds a transformer to the union.
    /// </summary>
    /// <param name="transformer">The transformer to add.</param>
    /// <returns>This instance for method chaining.</returns>
    public FeatureUnion<T> Add(IDataTransformer<T, Matrix<T>, Matrix<T>> transformer)
    {
        return Add($"transformer_{_transformers.Count}", transformer);
    }

    /// <summary>
    /// Fits all transformers to the input data.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        if (_transformers.Count == 0)
        {
            throw new InvalidOperationException("FeatureUnion has no transformers. Add at least one transformer.");
        }

        _nInputFeatures = data.Columns;
        _outputWidths = new List<int>();
        _nOutputFeatures = 0;

        // Fit each transformer
        foreach (var (name, transformer) in _transformers)
        {
            transformer.Fit(data);

            // Transform to get output dimensions
            var transformed = transformer.Transform(data);
            _outputWidths.Add(transformed.Columns);
            _nOutputFeatures += transformed.Columns;
        }
    }

    /// <summary>
    /// Transforms the data by applying all transformers and concatenating outputs.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The concatenated transformed features.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_outputWidths is null)
        {
            throw new InvalidOperationException("FeatureUnion has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nOutputFeatures];

        int outputIdx = 0;

        for (int t = 0; t < _transformers.Count; t++)
        {
            var (name, transformer) = _transformers[t];
            var transformed = transformer.Transform(data);

            // Copy to output
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < transformed.Columns; j++)
                {
                    result[i, outputIdx + j] = transformed[i, j];
                }
            }

            outputIdx += _outputWidths[t];
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for FeatureUnion.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FeatureUnion does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_outputWidths is null)
        {
            return Array.Empty<string>();
        }

        var names = new List<string>();

        foreach (var (name, transformer) in _transformers)
        {
            var outNames = transformer.GetFeatureNamesOut(inputFeatureNames);

            // Prefix with transformer name
            foreach (var outName in outNames)
            {
                names.Add($"{name}__{outName}");
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
        foreach (var (n, transformer) in _transformers)
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

    /// <summary>
    /// Gets the number of output features from each transformer.
    /// </summary>
    /// <returns>Dictionary mapping transformer name to output width.</returns>
    public Dictionary<string, int> GetTransformerOutputWidths()
    {
        if (_outputWidths is null)
        {
            return new Dictionary<string, int>();
        }

        var result = new Dictionary<string, int>();
        for (int i = 0; i < _transformers.Count; i++)
        {
            result[_transformers[i].Name] = _outputWidths[i];
        }
        return result;
    }
}
