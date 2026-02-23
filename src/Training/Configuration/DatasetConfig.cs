namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for the dataset section of a training recipe.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This defines where your data comes from and how it should be loaded.
/// You can specify a CSV file path, whether it has headers, the batch size for training,
/// and which column contains the labels (the values you want to predict).
/// </para>
/// </remarks>
public class DatasetConfig
{
    /// <summary>
    /// Gets or sets an optional descriptive name for the dataset.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the file path to the dataset (currently supports CSV files).
    /// </summary>
    /// <remarks>
    /// If empty or null, the trainer expects data to be provided programmatically.
    /// </remarks>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of samples per training batch.
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether the CSV file has a header row.
    /// </summary>
    public bool HasHeader { get; set; } = true;

    /// <summary>
    /// Gets or sets the zero-based index of the label column.
    /// Use -1 to indicate the last column.
    /// </summary>
    public int LabelColumn { get; set; } = -1;
}
