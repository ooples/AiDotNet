using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Parquet;
using Parquet.Schema;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Configuration options for the <see cref="ParquetDataLoader{T}"/>.
/// </summary>
public sealed class ParquetDataLoaderOptions
{
    /// <summary>
    /// Path to the Parquet file.
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Column names to use as features. If null or empty, all numeric columns except the label column are used.
    /// </summary>
    public string[]? FeatureColumns { get; set; }

    /// <summary>
    /// Column name to use as the label. If null, no labels are loaded (unsupervised).
    /// </summary>
    public string? LabelColumn { get; set; }

    /// <summary>
    /// Optional maximum number of rows to load.
    /// </summary>
    public int? MaxRows { get; set; }

    /// <summary>
    /// Optional random seed for reproducible sampling when MaxRows is set.
    /// </summary>
    public int? RandomSeed { get; set; }
}

/// <summary>
/// Reads tabular data from Apache Parquet columnar files using the Parquet.Net library.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Parquet is an efficient columnar storage format widely used in data engineering and ML pipelines.
/// This loader reads Parquet files and converts numeric columns into tensors for training.
/// Supports all standard Parquet compression codecs (Snappy, GZIP, ZSTD, LZ4, Brotli) and encodings.
/// </para>
/// <para><b>For Beginners:</b> Parquet files are commonly produced by Apache Spark, Pandas, and other
/// data processing frameworks. This loader reads them directly into AiDotNet for training.
/// <code>
/// var loader = new ParquetDataLoader&lt;float&gt;(new ParquetDataLoaderOptions
/// {
///     FilePath = "data/training_data.parquet",
///     FeatureColumns = new[] { "feature1", "feature2", "feature3" },
///     LabelColumn = "target"
/// });
/// await loader.LoadAsync();
/// </code>
/// </para>
/// </remarks>
public class ParquetDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly ParquetDataLoaderOptions _options;
    private int _sampleCount;
    private int _featureCount;
    private int _outputDim;

    /// <inheritdoc/>
    public override string Name => "Parquet";

    /// <inheritdoc/>
    public override string Description => $"Parquet dataset from {_options.FilePath}";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => _outputDim;

    /// <summary>
    /// Gets the column names used as features.
    /// </summary>
    public IReadOnlyList<string> ResolvedFeatureColumns { get; private set; } = Array.Empty<string>();

    /// <summary>
    /// Creates a new Parquet data loader.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public ParquetDataLoader(ParquetDataLoaderOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (string.IsNullOrWhiteSpace(options.FilePath))
            throw new ArgumentException("FilePath cannot be empty.", nameof(options));
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(_options.FilePath))
            throw new FileNotFoundException("Parquet file not found.", _options.FilePath);

        using var stream = File.OpenRead(_options.FilePath);
        using var reader = await ParquetReader.CreateAsync(stream);

        var schema = reader.Schema;
        var dataFields = schema.GetDataFields();

        // Resolve feature columns
        var featureFields = ResolveFeatureFields(dataFields);
        ResolvedFeatureColumns = featureFields.Select(f => f.Name).ToArray();
        _featureCount = featureFields.Count;

        if (_featureCount == 0)
            throw new InvalidOperationException("No numeric feature columns found in the Parquet file.");

        // Resolve label column
        DataField? labelField = null;
        if (!string.IsNullOrEmpty(_options.LabelColumn))
        {
            labelField = dataFields.FirstOrDefault(f =>
                string.Equals(f.Name, _options.LabelColumn, StringComparison.OrdinalIgnoreCase));

            if (labelField is null)
            {
                throw new InvalidOperationException(
                    $"Label column '{_options.LabelColumn}' not found in Parquet schema. " +
                    $"Available columns: {string.Join(", ", dataFields.Select(f => f.Name))}");
            }
        }

        _outputDim = labelField is not null ? 1 : 0;

        // Read all row groups and collect column data
        var featureColumnData = new List<double[]>(_featureCount);
        for (int c = 0; c < _featureCount; c++)
        {
            featureColumnData.Add(Array.Empty<double>());
        }

        double[] labelColumnData = Array.Empty<double>();
        int totalRows = 0;

        for (int g = 0; g < reader.RowGroupCount; g++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var groupReader = reader.OpenRowGroupReader(g);

            // Read feature columns
            var groupFeatureArrays = new List<double[]>(_featureCount);
            int groupRowCount = 0;

            for (int c = 0; c < _featureCount; c++)
            {
                var column = await groupReader.ReadColumnAsync(featureFields[c], cancellationToken);
                var values = ConvertColumnToDoubles(column);
                groupFeatureArrays.Add(values);
                groupRowCount = values.Length;
            }

            // Read label column
            double[]? groupLabelValues = null;
            if (labelField is not null)
            {
                var labelColumn = await groupReader.ReadColumnAsync(labelField, cancellationToken);
                groupLabelValues = ConvertColumnToDoubles(labelColumn);
            }

            // Append to accumulated data
            for (int c = 0; c < _featureCount; c++)
            {
                var existing = featureColumnData[c];
                var newData = new double[existing.Length + groupFeatureArrays[c].Length];
                Array.Copy(existing, 0, newData, 0, existing.Length);
                Array.Copy(groupFeatureArrays[c], 0, newData, existing.Length, groupFeatureArrays[c].Length);
                featureColumnData[c] = newData;
            }

            if (groupLabelValues is not null)
            {
                var existing = labelColumnData;
                var newData = new double[existing.Length + groupLabelValues.Length];
                Array.Copy(existing, 0, newData, 0, existing.Length);
                Array.Copy(groupLabelValues, 0, newData, existing.Length, groupLabelValues.Length);
                labelColumnData = newData;
            }

            totalRows += groupRowCount;
        }

        // Apply max rows limit
        int[] selectedIndices;
        if (_options.MaxRows.HasValue && _options.MaxRows.Value < totalRows)
        {
            var random = _options.RandomSeed.HasValue
                ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
                : RandomHelper.CreateSecureRandom();

            selectedIndices = Enumerable.Range(0, totalRows)
                .OrderBy(_ => random.Next())
                .Take(_options.MaxRows.Value)
                .OrderBy(i => i)
                .ToArray();
        }
        else
        {
            selectedIndices = Enumerable.Range(0, totalRows).ToArray();
        }

        _sampleCount = selectedIndices.Length;

        if (_sampleCount == 0)
            throw new InvalidOperationException("Parquet file contains no rows.");

        // Build feature tensor [N, featureCount]
        var featuresData = new T[_sampleCount * _featureCount];

        for (int i = 0; i < _sampleCount; i++)
        {
            int srcRow = selectedIndices[i];
            for (int c = 0; c < _featureCount; c++)
            {
                double value = featureColumnData[c][srcRow];
                featuresData[i * _featureCount + c] = NumOps.FromDouble(value);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { _sampleCount, _featureCount });

        // Build label tensor [N, 1] or empty
        if (labelField is not null)
        {
            var labelsData = new T[_sampleCount];
            for (int i = 0; i < _sampleCount; i++)
            {
                int srcRow = selectedIndices[i];
                labelsData[i] = NumOps.FromDouble(labelColumnData[srcRow]);
            }

            LoadedLabels = new Tensor<T>(labelsData, new[] { _sampleCount, 1 });
        }
        else
        {
            // Unsupervised: dummy labels
            LoadedLabels = new Tensor<T>(new T[_sampleCount], new[] { _sampleCount, 1 });
        }

        InitializeIndices(_sampleCount);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

        return (ExtractTensorBatch(features, indices), ExtractTensorBatch(labels, indices));
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
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
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(LoadedLabels ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private List<DataField> ResolveFeatureFields(DataField[] allFields)
    {
        if (_options.FeatureColumns is not null && _options.FeatureColumns.Length > 0)
        {
            // Use specified columns
            var result = new List<DataField>(_options.FeatureColumns.Length);
            foreach (string colName in _options.FeatureColumns)
            {
                var field = allFields.FirstOrDefault(f =>
                    string.Equals(f.Name, colName, StringComparison.OrdinalIgnoreCase));

                if (field is null)
                {
                    throw new InvalidOperationException(
                        $"Feature column '{colName}' not found in Parquet schema. " +
                        $"Available columns: {string.Join(", ", allFields.Select(f => f.Name))}");
                }

                result.Add(field);
            }

            return result;
        }

        // Auto-detect: all numeric columns except the label column
        return allFields
            .Where(f => IsNumericField(f) &&
                        !string.Equals(f.Name, _options.LabelColumn, StringComparison.OrdinalIgnoreCase))
            .ToList();
    }

    private static bool IsNumericField(DataField field)
    {
        var clrType = field.ClrType;
        return clrType == typeof(int) ||
               clrType == typeof(long) ||
               clrType == typeof(float) ||
               clrType == typeof(double) ||
               clrType == typeof(decimal) ||
               clrType == typeof(short) ||
               clrType == typeof(byte) ||
               clrType == typeof(sbyte) ||
               clrType == typeof(ushort) ||
               clrType == typeof(uint) ||
               clrType == typeof(ulong) ||
               clrType == typeof(int?) ||
               clrType == typeof(long?) ||
               clrType == typeof(float?) ||
               clrType == typeof(double?) ||
               clrType == typeof(decimal?) ||
               clrType == typeof(short?) ||
               clrType == typeof(byte?) ||
               clrType == typeof(bool);
    }

    private static double[] ConvertColumnToDoubles(Parquet.Data.DataColumn column)
    {
        var data = column.Data;
        int length = data.Length;
        var result = new double[length];

        for (int i = 0; i < length; i++)
        {
            var value = data.GetValue(i);
            result[i] = ConvertToDouble(value);
        }

        return result;
    }

    private static double ConvertToDouble(object? value)
    {
        if (value is null) return 0.0;

        return value switch
        {
            double d => d,
            float f => f,
            int i => i,
            long l => l,
            short s => s,
            byte b => b,
            sbyte sb => sb,
            ushort us => us,
            uint ui => ui,
            ulong ul => ul,
            decimal dec => (double)dec,
            bool boolVal => boolVal ? 1.0 : 0.0,
            _ => 0.0
        };
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
