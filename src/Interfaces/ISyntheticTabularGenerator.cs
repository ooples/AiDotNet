using AiDotNet.NeuralNetworks.SyntheticData;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for synthetic tabular data generators that learn the distribution
/// of real tabular data and can produce new, realistic synthetic rows.
/// </summary>
/// <remarks>
/// <para>
/// Synthetic tabular data generators fit a model to real tabular data (containing a mix of
/// continuous/numerical and categorical/discrete columns), then generate new rows that
/// preserve the statistical properties and inter-column relationships of the original data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like a "data factory" that learns from real data:
///
/// 1. <b>Fit</b>: You show the generator your real table (like a spreadsheet).
///    It learns the patterns - ranges of numbers, common categories, relationships between columns.
/// 2. <b>Generate</b>: The generator creates brand-new rows that look realistic
///    but aren't copies of any real data.
///
/// Common use cases:
/// - <b>Data augmentation</b>: Make a small dataset larger for better ML training
/// - <b>Privacy</b>: Share synthetic data instead of real data with sensitive information
/// - <b>Testing</b>: Generate realistic test data for development
/// - <b>Imbalanced data</b>: Generate more examples of rare categories
///
/// Example workflow:
/// <code>
/// // 1. Describe your columns
/// var columns = new List&lt;ColumnMetadata&gt;
/// {
///     new("Age", ColumnDataType.Continuous),
///     new("Income", ColumnDataType.Continuous),
///     new("Education", ColumnDataType.Categorical, categories: new[] { "HS", "BS", "MS", "PhD" })
/// };
///
/// // 2. Fit the generator on your real data
/// generator.Fit(realData, columns, epochs: 300);
///
/// // 3. Generate synthetic rows
/// var syntheticData = generator.Generate(numSamples: 1000);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
public interface ISyntheticTabularGenerator<T>
{
    /// <summary>
    /// Fits the generator model to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <remarks>
    /// <para>
    /// This method trains the generative model on the provided data. After fitting,
    /// the model can generate new synthetic rows via <see cref="Generate"/>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data
    /// to understand its patterns and distributions. More epochs generally means better quality
    /// but takes longer to train.
    /// </para>
    /// </remarks>
    void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs);

    /// <summary>
    /// Asynchronously fits the generator model to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="ct">Cancellation token for stopping training early.</param>
    /// <returns>A task representing the asynchronous fit operation.</returns>
    Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default);

    /// <summary>
    /// Generates new synthetic tabular data rows.
    /// </summary>
    /// <param name="numSamples">The number of synthetic rows to generate.</param>
    /// <param name="conditionColumn">
    /// Optional: a vector of length <paramref name="numSamples"/> where each element is the
    /// categorical column index to condition on for that sample. All elements typically have
    /// the same value when conditioning on a single column.
    /// </param>
    /// <param name="conditionValue">
    /// Optional: a vector of length <paramref name="numSamples"/> where each element is the
    /// category value (as a numeric index) to condition on for that sample. Must be provided
    /// together with <paramref name="conditionColumn"/>.
    /// </param>
    /// <returns>A matrix of synthetic data with the same column structure as the training data.</returns>
    /// <remarks>
    /// <para>
    /// The generator must be fitted before calling this method. The returned data will have
    /// columns matching the metadata provided during fitting.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After the generator has learned from real data, this method
    /// creates new fake-but-realistic rows. You can optionally request specific conditions,
    /// like "generate rows where Education = PhD".
    /// </para>
    /// </remarks>
    Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null);

    /// <summary>
    /// Gets the column metadata describing the structure of the data this generator was fitted on.
    /// </summary>
    /// <value>A read-only list of column metadata, or an empty list if the generator is not yet fitted.</value>
    IReadOnlyList<ColumnMetadata> Columns { get; }

    /// <summary>
    /// Gets a value indicating whether the generator has been fitted to data and is ready to generate.
    /// </summary>
    /// <value><c>true</c> if <see cref="Fit"/> has been called successfully; otherwise, <c>false</c>.</value>
    bool IsFitted { get; }
}
