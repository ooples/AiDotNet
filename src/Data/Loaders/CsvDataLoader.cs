using System.Globalization;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Loads supervised learning data from CSV files into Matrix/Vector format.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This loader reads CSV (Comma-Separated Values) files and converts them
/// into the format needed for training models. You specify which column contains the labels
/// (the values to predict), and the rest become features (input data).
/// </para>
/// <para>
/// <b>Example:</b>
/// <code>
/// // Load a CSV file where the last column is the label
/// var loader = new CsvDataLoader&lt;double&gt;("data/sales.csv", hasHeader: true, labelColumn: -1);
/// await loader.LoadAsync();
///
/// // Access the data
/// Matrix&lt;double&gt; features = loader.Features;  // Input data
/// Vector&lt;double&gt; labels = loader.Labels;        // Target values
/// </code>
/// </para>
/// </remarks>
internal class CsvDataLoader<T> : InputOutputDataLoaderBase<T, Matrix<T>, Vector<T>>
{
    private readonly string _filePath;
    private readonly bool _hasHeader;
    private readonly int _labelColumn;
    private int _sampleCount;
    private int _featureCount;

    /// <inheritdoc/>
    public override string Name => "CsvDataLoader";

    /// <inheritdoc/>
    public override string Description => $"CSV data loader for {_filePath}";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => 1;

    /// <summary>
    /// Creates a new CSV data loader.
    /// </summary>
    /// <param name="filePath">The path to the CSV file.</param>
    /// <param name="hasHeader">Whether the CSV file has a header row. Default is true.</param>
    /// <param name="labelColumn">The zero-based index of the label column. Use -1 for the last column. Default is -1.</param>
    /// <param name="batchSize">The batch size for iteration. Default is 32.</param>
    /// <exception cref="ArgumentException">Thrown when filePath is null or empty.</exception>
    public CsvDataLoader(string filePath, bool hasHeader = true, int labelColumn = -1, int batchSize = 32)
        : base(batchSize)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        _filePath = filePath;
        _hasHeader = hasHeader;
        _labelColumn = labelColumn;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(_filePath))
        {
            throw new FileNotFoundException($"CSV file not found: {_filePath}", _filePath);
        }

        var lines = await Task.Run(() => File.ReadLines(_filePath).ToList(), cancellationToken).ConfigureAwait(false);
        int startLine = _hasHeader ? 1 : 0;

        if (lines.Count <= startLine)
        {
            throw new InvalidOperationException("CSV file contains no data rows.");
        }

        // Parse all data rows
        var dataRows = new List<double[]>();
        int skippedEmptyRows = 0;
        for (int i = startLine; i < lines.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var line = lines[i].Trim();
            if (string.IsNullOrEmpty(line))
            {
                skippedEmptyRows++;
                continue;
            }

            var parts = ParseCsvLine(line);
            var values = new double[parts.Length];
            for (int j = 0; j < parts.Length; j++)
            {
                if (!double.TryParse(parts[j], NumberStyles.Any, CultureInfo.InvariantCulture, out var val))
                {
                    throw new FormatException(
                        $"Cannot parse value '{parts[j]}' at row {i + 1}, column {j + 1} as a number.");
                }
                values[j] = val;
            }
            dataRows.Add(values);
        }

        if (dataRows.Count == 0)
        {
            throw new InvalidOperationException("CSV file contains no valid data rows.");
        }

        int columnCount = dataRows[0].Length;
        if (columnCount < 2)
        {
            throw new InvalidOperationException("CSV file must have at least 2 columns (features + label).");
        }

        // Resolve label column index
        int labelCol = _labelColumn < 0 ? columnCount + _labelColumn : _labelColumn;
        if (labelCol < 0 || labelCol >= columnCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(_labelColumn),
                $"Label column index {_labelColumn} is out of range for {columnCount} columns.");
        }

        _sampleCount = dataRows.Count;
        _featureCount = columnCount - 1;

        // Build features matrix and labels vector
        var features = new Matrix<T>(_sampleCount, _featureCount);
        var labels = new Vector<T>(_sampleCount);

        for (int row = 0; row < _sampleCount; row++)
        {
            var rowData = dataRows[row];
            if (rowData.Length != columnCount)
            {
                throw new InvalidOperationException(
                    $"Row {row + startLine + 1} has {rowData.Length} columns, expected {columnCount}.");
            }

            labels[row] = NumOps.FromDouble(rowData[labelCol]);

            int featureIdx = 0;
            for (int col = 0; col < columnCount; col++)
            {
                if (col != labelCol)
                {
                    features[row, featureIdx] = NumOps.FromDouble(rowData[col]);
                    featureIdx++;
                }
            }
        }

        LoadedFeatures = features;
        LoadedLabels = labels;
        InitializeIndices(_sampleCount);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
        _featureCount = 0;
    }

    /// <inheritdoc/>
    protected override (Matrix<T> Features, Vector<T> Labels) ExtractBatch(int[] indices)
    {
        EnsureLoaded();

        if (LoadedFeatures is null || LoadedLabels is null)
        {
            throw new InvalidOperationException("Data has not been loaded. Call LoadAsync() first.");
        }

        var batchFeatures = new Matrix<T>(indices.Length, _featureCount);
        var batchLabels = new Vector<T>(indices.Length);

        for (int i = 0; i < indices.Length; i++)
        {
            batchLabels[i] = LoadedLabels[indices[i]];
            batchFeatures.SetRow(i, LoadedFeatures.GetRow(indices[i]));
        }

        return (batchFeatures, batchLabels);
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Matrix<T>, Vector<T>> Train,
        IInputOutputDataLoader<T, Matrix<T>, Vector<T>> Validation,
        IInputOutputDataLoader<T, Matrix<T>, Vector<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffledIndices = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        var trainIndices = shuffledIndices.Take(trainSize).ToArray();
        var valIndices = shuffledIndices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffledIndices.Skip(trainSize + valSize).ToArray();

        var trainLoader = CreateSubsetLoader(trainIndices);
        var valLoader = CreateSubsetLoader(valIndices);
        var testLoader = CreateSubsetLoader(testIndices);

        return (trainLoader, valLoader, testLoader);
    }

    /// <summary>
    /// Parses a single CSV line, handling RFC 4180 quoted fields (e.g., fields containing commas or quotes).
    /// </summary>
    private static string[] ParseCsvLine(string line)
    {
        var fields = new List<string>();
        int i = 0;
        while (i < line.Length)
        {
            if (line[i] == '"')
            {
                // Quoted field
                i++; // skip opening quote
                var field = new System.Text.StringBuilder();
                while (i < line.Length)
                {
                    if (line[i] == '"')
                    {
                        if (i + 1 < line.Length && line[i + 1] == '"')
                        {
                            field.Append('"');
                            i += 2; // skip escaped quote
                        }
                        else
                        {
                            i++; // skip closing quote
                            break;
                        }
                    }
                    else
                    {
                        field.Append(line[i]);
                        i++;
                    }
                }
                fields.Add(field.ToString().Trim());
                // Skip comma after quoted field
                if (i < line.Length && line[i] == ',')
                {
                    i++;
                }
            }
            else
            {
                // Unquoted field
                int start = i;
                while (i < line.Length && line[i] != ',')
                {
                    i++;
                }
                fields.Add(line.Substring(start, i - start).Trim());
                if (i < line.Length)
                {
                    i++; // skip comma
                }
            }
        }
        return fields.ToArray();
    }

    /// <summary>
    /// Creates a new InMemoryDataLoader containing only the data at the specified indices.
    /// </summary>
    private InMemoryDataLoader<T, Matrix<T>, Vector<T>> CreateSubsetLoader(int[] indices)
    {
        if (LoadedFeatures is null || LoadedLabels is null)
        {
            throw new InvalidOperationException("Data has not been loaded. Call LoadAsync() first.");
        }

        var subsetFeatures = new Matrix<T>(indices.Length, _featureCount);
        var subsetLabels = new Vector<T>(indices.Length);

        for (int i = 0; i < indices.Length; i++)
        {
            subsetLabels[i] = LoadedLabels[indices[i]];
            subsetFeatures.SetRow(i, LoadedFeatures.GetRow(indices[i]));
        }

        return new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(subsetFeatures, subsetLabels);
    }
}
