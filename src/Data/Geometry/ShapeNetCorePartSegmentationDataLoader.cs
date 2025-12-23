using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO.Compression;
using System.Net.Http;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Loads the ShapeNetCore part segmentation dataset.
/// </summary>
public sealed class ShapeNetCorePartSegmentationDataLoader<T> : PointCloudDatasetLoaderBase<T>
{
    private static readonly string DownloadUrl =
        "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip";

    private readonly ShapeNetCorePartSegmentationDataLoaderOptions _options;
    private readonly string _dataPath;
    private readonly bool _autoDownload;
    private Dictionary<string, string> _categoryMappings = new(StringComparer.OrdinalIgnoreCase);
    private List<string> _categoryNames = new();

    /// <inheritdoc />
    public override string Name => "ShapeNetCorePartSegmentation";

    /// <inheritdoc />
    public override string Description => "ShapeNetCore part segmentation dataset loader.";

    /// <summary>
    /// Gets the synset-to-category mappings.
    /// </summary>
    public IReadOnlyDictionary<string, string> CategoryMappings => _categoryMappings;

    /// <summary>
    /// Gets the category names defined in the dataset metadata.
    /// </summary>
    public IReadOnlyList<string> CategoryNames => _categoryNames;

    /// <summary>
    /// Gets the number of part classes.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <summary>
    /// Initializes a new instance of the <see cref="ShapeNetCorePartSegmentationDataLoader{T}"/> class.
    /// </summary>
    public ShapeNetCorePartSegmentationDataLoader(ShapeNetCorePartSegmentationDataLoaderOptions? options = null)
    {
        _options = options ?? new ShapeNetCorePartSegmentationDataLoaderOptions();
        _dataPath = _options.DataPath ?? GetDefaultDataPath();
        _autoDownload = _options.AutoDownload;
    }

    /// <inheritdoc />
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        await EnsureDataExistsAsync(_dataPath, cancellationToken);
        string root = FindDatasetRoot(_dataPath);

        await LoadCategoryMappingsAsync(root, cancellationToken);
        var samples = await LoadSamplesAsync(root, cancellationToken);

        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < samples.Count)
        {
            samples = samples.Take(_options.MaxSamples.Value).ToList();
        }

        if (samples.Count == 0)
        {
            throw new InvalidOperationException("ShapeNetCore dataset contains no samples for the requested split.");
        }

        int featureDim = _options.IncludeNormals ? 6 : 3;
        var featuresData = new T[samples.Count * _options.PointsPerSample * featureDim];
        var labelsData = new T[samples.Count * _options.PointsPerSample];

        var random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var sample = samples[sampleIndex];
            var rows = await PointCloudTextParser.ReadRowsAsync(sample.PointsPath, cancellationToken);
            var labelValues = await ReadLabelValuesAsync(sample.LabelsPath, cancellationToken);

            int[] indices = BuildSampleIndices(rows.Count, _options.PointsPerSample, _options.SamplingStrategy, _options.PaddingStrategy, random);

            FillSampleFeatures(rows, indices, featureDim, sampleIndex, featuresData);
            FillSampleLabels(labelValues, indices, sampleIndex, labelsData);
        }

        var features = new Tensor<T>(featuresData, new[] { samples.Count, _options.PointsPerSample, featureDim });
        var labels = new Tensor<T>(labelsData, new[] { samples.Count, _options.PointsPerSample });
        SetLoadedData(features, labels);
    }

    /// <inheritdoc />
    protected override void UnloadDataCore()
    {
        LoadedFeatures = null;
        LoadedLabels = null;
        _categoryMappings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        _categoryNames = new List<string>();
    }

    private void FillSampleFeatures(List<double[]> rows, int[] indices, int featureDim, int sampleIndex, T[] featuresData)
    {
        int sampleOffset = sampleIndex * _options.PointsPerSample * featureDim;
        for (int i = 0; i < indices.Length; i++)
        {
            int rowIndex = indices[i];
            int destOffset = sampleOffset + i * featureDim;

            if (rowIndex < 0 || rowIndex >= rows.Count)
            {
                continue;
            }

            double[] row = rows[rowIndex];
            double x = row.Length > 0 ? row[0] : 0.0;
            double y = row.Length > 1 ? row[1] : 0.0;
            double z = row.Length > 2 ? row[2] : 0.0;

            featuresData[destOffset] = NumOps.FromDouble(x);
            featuresData[destOffset + 1] = NumOps.FromDouble(y);
            featuresData[destOffset + 2] = NumOps.FromDouble(z);

            if (_options.IncludeNormals)
            {
                double nx = row.Length > 3 ? row[3] : 0.0;
                double ny = row.Length > 4 ? row[4] : 0.0;
                double nz = row.Length > 5 ? row[5] : 0.0;

                featuresData[destOffset + 3] = NumOps.FromDouble(nx);
                featuresData[destOffset + 4] = NumOps.FromDouble(ny);
                featuresData[destOffset + 5] = NumOps.FromDouble(nz);
            }
        }
    }

    private void FillSampleLabels(List<int> labels, int[] indices, int sampleIndex, T[] labelsData)
    {
        int sampleOffset = sampleIndex * _options.PointsPerSample;
        for (int i = 0; i < indices.Length; i++)
        {
            int rowIndex = indices[i];
            if (rowIndex < 0 || rowIndex >= labels.Count)
            {
                continue;
            }

            int labelValue = labels[rowIndex];
            if (labelValue < 0)
            {
                labelValue = 0;
            }
            if (labelValue >= _options.NumClasses)
            {
                labelValue = _options.NumClasses - 1;
            }

            labelsData[sampleOffset + i] = NumOps.FromDouble(labelValue);
        }
    }

    private async Task<List<ShapeNetSample>> LoadSamplesAsync(string root, CancellationToken cancellationToken)
    {
        var samples = new List<ShapeNetSample>();
        var splitFiles = GetSplitFiles(root);

        if (splitFiles.Count == 0)
        {
            string pointsDir = Path.Combine(root, "points");
            if (!Directory.Exists(pointsDir))
            {
                return samples;
            }

            var pointFiles = Directory.GetFiles(pointsDir, "*.pts", SearchOption.AllDirectories)
                .Concat(Directory.GetFiles(pointsDir, "*.txt", SearchOption.AllDirectories))
                .ToList();

            foreach (string pointPath in pointFiles)
            {
                if (!TryResolveLabelPath(root, pointPath, out string? labelPath))
                {
                    continue;
                }

                samples.Add(new ShapeNetSample(pointPath, labelPath));
            }

            return samples;
        }

        foreach (string splitFile in splitFiles)
        {
            string json = await FilePolyfill.ReadAllTextAsync(splitFile, cancellationToken);
            var array = JArray.Parse(json);
            foreach (var token in array)
            {
                string entry = token.ToString().Trim();
                if (entry.Length == 0)
                {
                    continue;
                }

                if (!TryResolveEntryPaths(root, entry, out ShapeNetSample sample))
                {
                    continue;
                }

                samples.Add(sample);
            }
        }

        return samples;
    }

    private List<string> GetSplitFiles(string root)
    {
        var files = new List<string>();
        string splitDir = Path.Combine(root, "train_test_split");
        string trainFile = Path.Combine(splitDir, "shuffled_train_file_list.json");
        string valFile = Path.Combine(splitDir, "shuffled_val_file_list.json");
        string testFile = Path.Combine(splitDir, "shuffled_test_file_list.json");

        switch (_options.Split)
        {
            case DatasetSplit.Train:
                if (File.Exists(trainFile))
                {
                    files.Add(trainFile);
                }
                break;
            case DatasetSplit.Validation:
                if (File.Exists(valFile))
                {
                    files.Add(valFile);
                }
                else if (File.Exists(trainFile))
                {
                    files.Add(trainFile);
                }
                break;
            case DatasetSplit.Test:
                if (File.Exists(testFile))
                {
                    files.Add(testFile);
                }
                break;
            case DatasetSplit.All:
                if (File.Exists(trainFile))
                {
                    files.Add(trainFile);
                }
                if (File.Exists(valFile))
                {
                    files.Add(valFile);
                }
                if (File.Exists(testFile))
                {
                    files.Add(testFile);
                }
                break;
            default:
                break;
        }

        return files;
    }

    private bool TryResolveEntryPaths(string root, string entry, out ShapeNetSample sample)
    {
        sample = new ShapeNetSample(string.Empty, string.Empty);

        string normalized = entry.Replace('\\', '/');
        if (normalized.StartsWith("points/", StringComparison.OrdinalIgnoreCase))
        {
            normalized = normalized.Substring("points/".Length);
        }

        string relativePath = normalized;
        if (!Path.HasExtension(relativePath))
        {
            relativePath += ".pts";
        }

        string pointsPath = Path.Combine(root, "points", relativePath);
        if (!File.Exists(pointsPath))
        {
            string txtPath = Path.ChangeExtension(pointsPath, ".txt");
            if (File.Exists(txtPath))
            {
                pointsPath = txtPath;
            }
        }

        if (!File.Exists(pointsPath))
        {
            return false;
        }

        if (!TryResolveLabelPath(root, pointsPath, out string? labelPath))
        {
            return false;
        }

        sample = new ShapeNetSample(pointsPath, labelPath);
        return true;
    }

    private bool TryResolveLabelPath(string root, string pointsPath, [NotNullWhen(true)] out string? labelPath)
    {
        labelPath = null;
        string pointsDir = Path.Combine(root, "points");
        if (!pointsPath.StartsWith(pointsDir, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        string relative = pointsPath.Substring(pointsDir.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string labelRelative = Path.ChangeExtension(relative, ".seg");
        string candidate = Path.Combine(root, "points_label", labelRelative);
        if (File.Exists(candidate))
        {
            labelPath = candidate;
            return true;
        }

        return false;
    }

    private async Task LoadCategoryMappingsAsync(string root, CancellationToken cancellationToken)
    {
        string mappingFile = Path.Combine(root, "synsetoffset2category.txt");
        if (!File.Exists(mappingFile))
        {
            _categoryMappings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            _categoryNames = new List<string>();
            return;
        }

        string[] lines = await FilePolyfill.ReadAllLinesAsync(mappingFile, cancellationToken);
        var mappings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var names = new List<string>();

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2)
            {
                continue;
            }

            string category = parts[0];
            string synset = parts[1];

            mappings[synset] = category;
            if (!names.Contains(category, StringComparer.OrdinalIgnoreCase))
            {
                names.Add(category);
            }
        }

        _categoryMappings = mappings;
        _categoryNames = names;
    }

    private async Task<List<int>> ReadLabelValuesAsync(string path, CancellationToken cancellationToken)
    {
        string[] lines = await FilePolyfill.ReadAllLinesAsync(path, cancellationToken);
        var labels = new List<int>(lines.Length);

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            if (!int.TryParse(line.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
            {
                value = 0;
            }

            labels.Add(value);
        }

        return labels;
    }

    private async Task EnsureDataExistsAsync(string dataPath, CancellationToken cancellationToken)
    {
        string rootCandidate = FindDatasetRoot(dataPath);
        if (Directory.Exists(Path.Combine(rootCandidate, "points")) ||
            File.Exists(Path.Combine(rootCandidate, "synsetoffset2category.txt")))
        {
            return;
        }

        if (!_autoDownload)
        {
            throw new FileNotFoundException(
                $"ShapeNetCore dataset not found at {dataPath}. " +
                "Provide the dataset locally or enable AutoDownload.");
        }

        await DownloadDatasetAsync(dataPath, cancellationToken);
    }

    private async Task DownloadDatasetAsync(string dataPath, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(dataPath);

        string tempFile = Path.Combine(Path.GetTempPath(), $"shapenetcore_{Guid.NewGuid()}.zip");
        try
        {
            using (var httpClient = new HttpClient())
            {
                httpClient.Timeout = TimeSpan.FromMinutes(60);
                using var response = await httpClient.GetAsync(DownloadUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                response.EnsureSuccessStatusCode();

                using var fileStream = new FileStream(tempFile, FileMode.Create, FileAccess.Write, FileShare.None);
                await response.Content.CopyToAsync(fileStream);
            }

            ZipFile.ExtractToDirectory(tempFile, dataPath);
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                try
                {
                    File.Delete(tempFile);
                }
                catch
                {
                }
            }
        }
    }

    private string FindDatasetRoot(string dataPath)
    {
        string direct = Path.Combine(dataPath, "shapenetcore_partanno_segmentation_benchmark_v0");
        if (Directory.Exists(direct))
        {
            return direct;
        }

        if (Directory.Exists(dataPath) && File.Exists(Path.Combine(dataPath, "synsetoffset2category.txt")))
        {
            return dataPath;
        }

        if (Directory.Exists(dataPath))
        {
            string[] matches = Directory.GetDirectories(dataPath, "shapenetcore_partanno_segmentation_benchmark_v0", SearchOption.AllDirectories);
            if (matches.Length > 0)
            {
                return matches[0];
            }
        }

        return dataPath;
    }

    private string GetDefaultDataPath()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".aidotnet",
            "datasets",
            "shapenetcore");
    }

    private sealed class ShapeNetSample
    {
        public ShapeNetSample(string pointsPath, string labelsPath)
        {
            PointsPath = pointsPath;
            LabelsPath = labelsPath;
        }

        public string PointsPath { get; }
        public string LabelsPath { get; }
    }
}
