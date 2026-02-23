namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Specifies the data type of a column in a tabular dataset.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Each column in your data falls into one of these categories:
/// - <b>Continuous</b>: Numbers that can take any value (e.g., price = 19.99, temperature = 72.5)
/// - <b>Discrete</b>: Integer counts or ordinal numbers (e.g., number of children = 3, rating = 4)
/// - <b>Categorical</b>: Labels or categories (e.g., color = "red", city = "NYC")
///
/// The generator uses this information to apply the correct preprocessing and generation strategy.
/// </para>
/// </remarks>
public enum ColumnDataType
{
    /// <summary>
    /// A continuous numerical column (real-valued, e.g., price, temperature).
    /// Processed using Variational Gaussian Mixture normalization in CTGAN/TVAE,
    /// or quantile normalization in TabDDPM.
    /// </summary>
    Continuous,

    /// <summary>
    /// A discrete integer column (counts or ordinal values, e.g., age in years, number of items).
    /// Treated similarly to continuous for generation but rounded to integers in output.
    /// </summary>
    Discrete,

    /// <summary>
    /// A categorical column (unordered labels, e.g., "red"/"blue"/"green").
    /// Processed using one-hot encoding in CTGAN/TVAE,
    /// or multinomial diffusion in TabDDPM.
    /// </summary>
    Categorical
}

/// <summary>
/// Describes the metadata for a single column in a tabular dataset, including its name,
/// data type, categories (for categorical columns), and summary statistics.
/// </summary>
/// <remarks>
/// <para>
/// Column metadata is used by synthetic data generators to understand the structure of each
/// column and apply the appropriate preprocessing (e.g., VGM normalization for continuous,
/// one-hot encoding for categorical). Statistics are populated during the <c>Fit</c> step.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a "column profile" that describes everything
/// the generator needs to know about one column in your table:
///
/// - <b>Name</b>: A human-readable label (e.g., "Age", "Income")
/// - <b>DataType</b>: Whether it's continuous, discrete, or categorical
/// - <b>Categories</b>: For categorical columns, the list of possible values
/// - <b>Statistics</b>: Min, max, mean, and standard deviation (computed during fitting)
///
/// Example:
/// <code>
/// // A continuous column
/// var ageCol = new ColumnMetadata("Age", ColumnDataType.Continuous);
///
/// // A categorical column with known categories
/// var colorCol = new ColumnMetadata("Color", ColumnDataType.Categorical,
///     categories: new[] { "Red", "Green", "Blue" });
/// </code>
/// </para>
/// </remarks>
public class ColumnMetadata
{
    /// <summary>
    /// Gets or sets the name of the column.
    /// </summary>
    /// <value>A human-readable column name.</value>
    public string Name { get; set; }

    /// <summary>
    /// Gets or sets the data type of the column.
    /// </summary>
    /// <value>The column's data type (Continuous, Discrete, or Categorical).</value>
    public ColumnDataType DataType { get; set; }

    /// <summary>
    /// Gets or sets the list of category values for categorical columns.
    /// </summary>
    /// <value>
    /// The category labels for this column, or an empty list for non-categorical columns.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For a "Color" column, this would be something like
    /// ["Red", "Green", "Blue"]. For numerical columns, this list is empty.
    /// </para>
    /// </remarks>
    public IReadOnlyList<string> Categories { get; set; }

    /// <summary>
    /// Gets or sets the minimum observed value for numerical columns.
    /// </summary>
    /// <value>The minimum value, populated during fitting. Defaults to 0.</value>
    public double Min { get; set; }

    /// <summary>
    /// Gets or sets the maximum observed value for numerical columns.
    /// </summary>
    /// <value>The maximum value, populated during fitting. Defaults to 0.</value>
    public double Max { get; set; }

    /// <summary>
    /// Gets or sets the mean (average) value for numerical columns.
    /// </summary>
    /// <value>The mean value, populated during fitting. Defaults to 0.</value>
    public double Mean { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation for numerical columns.
    /// </summary>
    /// <value>The standard deviation, populated during fitting. Defaults to 1.</value>
    public double Std { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the index of this column in the original data matrix.
    /// </summary>
    /// <value>The zero-based column index.</value>
    public int ColumnIndex { get; set; }

    /// <summary>
    /// Gets the number of categories for categorical columns.
    /// </summary>
    /// <value>The number of distinct categories, or 0 for non-categorical columns.</value>
    public int NumCategories => Categories.Count;

    /// <summary>
    /// Gets whether this column is categorical.
    /// </summary>
    public bool IsCategorical => DataType == ColumnDataType.Categorical;

    /// <summary>
    /// Gets whether this column is numerical (continuous or discrete).
    /// </summary>
    public bool IsNumerical => DataType == ColumnDataType.Continuous || DataType == ColumnDataType.Discrete;

    /// <summary>
    /// Initializes a new instance of the <see cref="ColumnMetadata"/> class.
    /// </summary>
    /// <param name="name">The column name.</param>
    /// <param name="dataType">The column's data type.</param>
    /// <param name="categories">Optional list of category values for categorical columns.</param>
    /// <param name="columnIndex">The zero-based index of this column in the data matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create one of these for each column in your data.
    /// For categorical columns, provide the list of possible values.
    /// Statistics (min, max, mean, std) are filled in automatically during fitting.
    /// </para>
    /// </remarks>
    public ColumnMetadata(string name, ColumnDataType dataType, IEnumerable<string>? categories = null, int columnIndex = 0)
    {
        Name = name;
        DataType = dataType;
        Categories = categories is not null ? new List<string>(categories).AsReadOnly() : Array.Empty<string>();
        ColumnIndex = columnIndex;
    }

    /// <summary>
    /// Creates a deep copy of this column metadata.
    /// </summary>
    /// <returns>A new <see cref="ColumnMetadata"/> instance with the same values.</returns>
    public ColumnMetadata Clone()
    {
        return new ColumnMetadata(Name, DataType, Categories, ColumnIndex)
        {
            Min = Min,
            Max = Max,
            Mean = Mean,
            Std = Std
        };
    }
}
