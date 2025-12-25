using System.Text;
using Newtonsoft.Json;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features as an integer array.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// OrdinalEncoder transforms each categorical feature into a single column of integer values.
/// Each unique category in a column is mapped to a consecutive integer starting from 0.
/// Unlike OneHotEncoder, this does not expand the feature space.
/// </para>
/// <para>
/// <b>For Beginners:</b> OrdinalEncoder is like giving each category a unique ID number.
///
/// Imagine you have features like:
/// | Size   | Color |
/// |--------|-------|
/// | Large  | Red   |
/// | Small  | Blue  |
/// | Medium | Red   |
///
/// After ordinal encoding:
/// | Size | Color |
/// |------|-------|
/// | 0    | 1     | (Large=0, Red=1)
/// | 2    | 0     | (Small=2, Blue=0)
/// | 1    | 1     | (Medium=1, Red=1)
///
/// When to use OrdinalEncoder vs OneHotEncoder:
/// - OrdinalEncoder: When categories have a natural order (e.g., Small &lt; Medium &lt; Large)
///   or when using tree-based models that can handle ordinal encoding
/// - OneHotEncoder: When categories have no natural order and you want to avoid
///   implying any relationship between category numbers
///
/// Warning: Using ordinal encoding when categories have no natural order can mislead
/// some models into thinking Medium (1) is "between" Large (0) and Small (2).
///
/// Example:
/// <code>
/// var encoder = new OrdinalEncoder&lt;double&gt;(columns: new[] { 0, 1 });
/// var X_encoded = encoder.FitTransform(X);
/// </code>
/// </para>
/// </remarks>
public class OrdinalEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Dictionary<int, Dictionary<T, int>>? _categoryMappings;
    private Dictionary<int, T[]>? _categories;

    /// <summary>
    /// Gets how unknown categories are handled during transformation.
    /// </summary>
    public UnknownCategoryHandling HandleUnknown { get; }

    /// <summary>
    /// Gets the value to use for unknown categories when HandleUnknown is Ignore.
    /// Default is -1.
    /// </summary>
    public T UnknownValue { get; }

    /// <summary>
    /// Gets the categories found for each encoded column.
    /// </summary>
    public IReadOnlyDictionary<int, T[]> Categories
    {
        get
        {
            if (!IsFitted || _categories == null)
            {
                throw new InvalidOperationException(
                    "OrdinalEncoder has not been fitted. Call Fit() or FitTransform() first.");
            }
            return _categories;
        }
    }

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="OrdinalEncoder{T}"/> class.
    /// </summary>
    /// <param name="columnIndices">
    /// The column indices to encode. If null or empty, all columns are encoded.
    /// </param>
    /// <param name="handleUnknown">How to handle unknown categories during transformation.</param>
    /// <param name="unknownValue">Value to use for unknown categories when HandleUnknown is Ignore.</param>
    public OrdinalEncoder(
        int[]? columnIndices = null,
        UnknownCategoryHandling handleUnknown = UnknownCategoryHandling.Error,
        T? unknownValue = default)
        : base(columnIndices)
    {
        HandleUnknown = handleUnknown;

        var ops = MathHelper.GetNumericOperations<T>();
        UnknownValue = unknownValue ?? ops.FromDouble(-1);
    }

    /// <summary>
    /// Fits the encoder by learning the unique categories for each specified column.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numCols = data.Columns;

        NumFeaturesIn = numCols;
        NumFeaturesOut = numCols; // Same number of columns, just encoded

        var columns = GetEffectiveColumnIndices(numCols);

        _categoryMappings = new Dictionary<int, Dictionary<T, int>>();
        _categories = new Dictionary<int, T[]>();

        for (int col = 0; col < numCols; col++)
        {
            if (Array.IndexOf(columns, col) >= 0)
            {
                // Get unique categories for this column
                var uniqueValues = new HashSet<T>();

                for (int row = 0; row < numRows; row++)
                {
                    uniqueValues.Add(data[row, col]);
                }

                // Sort categories for consistent ordering
                var sortedCategories = uniqueValues.OrderBy(x => x).ToArray();
                _categories[col] = sortedCategories;

                // Create mapping from category value to index
                var mapping = new Dictionary<T, int>();
                for (int i = 0; i < sortedCategories.Length; i++)
                {
                    mapping[sortedCategories[i]] = i;
                }
                _categoryMappings[col] = mapping;
            }
        }
    }

    /// <summary>
    /// Transforms the input data by applying ordinal encoding.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categoryMappings == null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        ValidateNumFeatures(data.Columns);

        int numRows = data.Rows;
        int numCols = data.Columns;
        var columns = GetEffectiveColumnIndices(numCols);

        var result = new Matrix<T>(numRows, numCols);

        for (int col = 0; col < numCols; col++)
        {
            if (Array.IndexOf(columns, col) >= 0)
            {
                var mapping = _categoryMappings[col];

                for (int row = 0; row < numRows; row++)
                {
                    T value = data[row, col];

                    if (!mapping.TryGetValue(value, out int encodedValue))
                    {
                        // Unknown category
                        if (HandleUnknown == UnknownCategoryHandling.Error)
                        {
                            throw new InvalidOperationException(
                                $"Unknown category '{value}' encountered in column {col}. " +
                                "Set HandleUnknown to Ignore to use UnknownValue for unknown categories.");
                        }
                        // HandleUnknown == Ignore: use UnknownValue
                        result[row, col] = UnknownValue;
                    }
                    else
                    {
                        result[row, col] = NumOps.FromDouble(encodedValue);
                    }
                }
            }
            else
            {
                // Pass through non-encoded columns
                for (int row = 0; row < numRows; row++)
                {
                    result[row, col] = data[row, col];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Inverse-transforms the encoded data back to the original categorical values.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_categories == null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        ValidateNumFeatures(data.Columns);

        int numRows = data.Rows;
        int numCols = data.Columns;
        var columns = GetEffectiveColumnIndices(numCols);

        var result = new Matrix<T>(numRows, numCols);

        for (int col = 0; col < numCols; col++)
        {
            if (Array.IndexOf(columns, col) >= 0)
            {
                var categories = _categories[col];

                for (int row = 0; row < numRows; row++)
                {
                    T value = data[row, col];
                    int encodedValue = (int)NumOps.ToDouble(value);

                    // Check for unknown value marker
                    if (NumOps.Equals(value, UnknownValue))
                    {
                        throw new InvalidOperationException(
                            $"Cannot inverse transform unknown value marker '{UnknownValue}' in column {col}. " +
                            "The original category is not known.");
                    }

                    if (encodedValue < 0 || encodedValue >= categories.Length)
                    {
                        throw new InvalidOperationException(
                            $"Invalid encoded value '{encodedValue}' encountered in column {col} during inverse transform. " +
                            $"Expected values in range [0, {categories.Length - 1}].");
                    }

                    result[row, col] = categories[encodedValue];
                }
            }
            else
            {
                // Pass through non-encoded columns
                for (int row = 0; row < numRows; row++)
                {
                    result[row, col] = data[row, col];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                "OrdinalEncoder has not been fitted. Call Fit() or FitTransform() first.");
        }

        int numCols = NumFeaturesIn ?? 0;
        var names = new string[numCols];
        var columns = GetEffectiveColumnIndices(numCols);

        for (int col = 0; col < numCols; col++)
        {
            string baseName = inputFeatureNames != null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";

            if (Array.IndexOf(columns, col) >= 0)
            {
                names[col] = $"{baseName}_ordinal";
            }
            else
            {
                names[col] = baseName;
            }
        }

        return names;
    }

    /// <summary>
    /// Serializes the encoder-specific state for persistence.
    /// </summary>
    protected override Dictionary<string, object?>? SerializeState()
    {
        if (!IsFitted)
        {
            return null;
        }

        // Convert categories to serializable format
        var categoriesDict = new Dictionary<string, T[]>();
        if (_categories != null)
        {
            foreach (var kvp in _categories)
            {
                categoriesDict[kvp.Key.ToString()] = kvp.Value;
            }
        }

        return new Dictionary<string, object?>
        {
            { "HandleUnknown", HandleUnknown.ToString() },
            { "UnknownValue", UnknownValue },
            { "Categories", categoriesDict }
        };
    }

    /// <summary>
    /// Deserializes the encoder-specific state from persistence.
    /// </summary>
    protected override void DeserializeState(Dictionary<string, object?> state)
    {
        if (state.TryGetValue("Categories", out var categoriesObj) && categoriesObj != null)
        {
            var json = JsonConvert.SerializeObject(categoriesObj);
            var dict = JsonConvert.DeserializeObject<Dictionary<string, T[]>>(json);
            if (dict != null)
            {
                _categories = new Dictionary<int, T[]>();
                _categoryMappings = new Dictionary<int, Dictionary<T, int>>();

                foreach (var kvp in dict)
                {
                    int col = int.Parse(kvp.Key);
                    _categories[col] = kvp.Value;

                    // Rebuild mapping
                    var mapping = new Dictionary<T, int>();
                    for (int i = 0; i < kvp.Value.Length; i++)
                    {
                        mapping[kvp.Value[i]] = i;
                    }
                    _categoryMappings[col] = mapping;
                }
            }
        }
    }
}
