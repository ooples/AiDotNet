using System.IO.Compression;
using System.Net.Http;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Loads the ModelNet40 point cloud classification dataset.
/// </summary>
public sealed class ModelNet40ClassificationDataLoader<T> : PointCloudDatasetLoaderBase<T>
{
    private static readonly string DownloadUrl = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip";
    private readonly ModelNet40ClassificationDataLoaderOptions _options;
    private readonly string _dataPath;
    private readonly bool _autoDownload;
    private List<string> _classNames = new List<string>();

    /// <inheritdoc />
    public override string Name => "ModelNet40Classification";

    /// <inheritdoc />
    public override string Description => "ModelNet40 point cloud classification dataset loader.";

    /// <summary>
    /// Gets the class names for ModelNet40.
    /// </summary>
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>
    /// Gets the number of classes in the dataset.
    /// </summary>
    public int NumClasses => _classNames.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelNet40ClassificationDataLoader{T}"/> class.
    /// </summary>
    public ModelNet40ClassificationDataLoader(ModelNet40ClassificationDataLoaderOptions? options = null)
    {
        _options = options ?? new ModelNet40ClassificationDataLoaderOptions();
        _dataPath = _options.DataPath ?? GetDefaultDataPath();
        _autoDownload = _options.AutoDownload;
    }

    /// <inheritdoc />
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        await EnsureDataExistsAsync(_dataPath, cancellationToken);
        string root = FindDatasetRoot(_dataPath);

        _classNames = await LoadClassNamesAsync(root, cancellationToken);
        if (_classNames.Count == 0)
        {
            throw new InvalidOperationException("ModelNet40 class list could not be resolved.");
        }

        var classIndexMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < _classNames.Count; i++)
        {
            classIndexMap[_classNames[i]] = i;
        }

        var samples = await LoadSamplesAsync(root, classIndexMap, cancellationToken);
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < samples.Count)
        {
            samples = samples.Take(_options.MaxSamples.Value).ToList();
        }

        if (samples.Count == 0)
        {
            throw new InvalidOperationException("ModelNet40 dataset contains no samples for the requested split.");
        }

        int featureDim = _options.IncludeNormals ? 6 : 3;
        var featuresData = new T[samples.Count * _options.PointsPerSample * featureDim];
        var labelsData = new T[samples.Count * _classNames.Count];

        var random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var sample = samples[sampleIndex];
            var rows = await PointCloudTextParser.ReadRowsAsync(sample.FilePath, cancellationToken);
            int[] indices = BuildSampleIndices(rows.Count, _options.PointsPerSample, _options.SamplingStrategy, _options.PaddingStrategy, random);

            FillSampleFeatures(rows, indices, featureDim, sampleIndex, featuresData);

            int labelOffset = sampleIndex * _classNames.Count;
            labelsData[labelOffset + sample.ClassIndex] = NumOps.One;
        }

        var features = new Tensor<T>(featuresData, new[] { samples.Count, _options.PointsPerSample, featureDim });
        var labels = new Tensor<T>(labelsData, new[] { samples.Count, _classNames.Count });
        SetLoadedData(features, labels);
    }

    /// <inheritdoc />
    protected override void UnloadDataCore()
    {
        LoadedFeatures = null;
        LoadedLabels = null;
        _classNames = new List<string>();
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

    private async Task<List<ModelNetSample>> LoadSamplesAsync(
        string root,
        Dictionary<string, int> classIndexMap,
        CancellationToken cancellationToken)
    {
        var samples = new List<ModelNetSample>();
        var splitFiles = GetSplitFiles(root);

        if (splitFiles.Count == 0)
        {
            var allFiles = Directory.GetFiles(root, "*.txt", SearchOption.AllDirectories)
                .Where(path => !Path.GetFileName(path).StartsWith("modelnet40_", StringComparison.OrdinalIgnoreCase))
                .ToList();

            foreach (string filePath in allFiles)
            {
                string? className = Path.GetFileName(Path.GetDirectoryName(filePath) ?? string.Empty);
                if (string.IsNullOrWhiteSpace(className))
                {
                    continue;
                }

                if (!classIndexMap.TryGetValue(className, out int classIndex))
                {
                    continue;
                }

                samples.Add(new ModelNetSample(filePath, classIndex));
            }

            return samples;
        }

        var classNamesByLength = classIndexMap.Keys
            .OrderByDescending(name => name.Length)
            .ToList();

        foreach (string splitFile in splitFiles)
        {
            string[] lines = await FilePolyfill.ReadAllLinesAsync(splitFile, cancellationToken);
            foreach (string line in lines)
            {
                string entry = line.Trim();
                if (entry.Length == 0)
                {
                    continue;
                }

                if (!TryResolveSample(root, entry, classNamesByLength, classIndexMap, out ModelNetSample sample))
                {
                    continue;
                }

                samples.Add(sample);
            }
        }

        return samples;
    }

    private bool TryResolveSample(
        string root,
        string entry,
        IReadOnlyList<string> classNamesByLength,
        Dictionary<string, int> classIndexMap,
        out ModelNetSample sample)
    {
        sample = new ModelNetSample(string.Empty, 0);

        string entryPath = entry.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        string? className = null;

        if (entryPath.Contains(Path.DirectorySeparatorChar))
        {
            string[] parts = entryPath.Split(new[] { Path.DirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length > 0)
            {
                className = parts[0];
            }
        }
        else
        {
            foreach (string candidate in classNamesByLength)
            {
                if (entry.StartsWith(candidate + "_", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(entry, candidate, StringComparison.OrdinalIgnoreCase))
                {
                    className = candidate;
                    break;
                }
            }
        }

        if (string.IsNullOrWhiteSpace(className))
        {
            return false;
        }
        string safeClassName = className ?? string.Empty;

        if (!classIndexMap.TryGetValue(safeClassName, out int classIndex))
        {
            return false;
        }

        string fileName = Path.HasExtension(entryPath) ? entryPath : entryPath + ".txt";
        string relativePath = entryPath.Contains(Path.DirectorySeparatorChar)
            ? fileName
            : Path.Combine(safeClassName, fileName);

        string filePath = Path.Combine(root, relativePath);
        if (!File.Exists(filePath))
        {
            string[] matches = Directory.GetFiles(root, Path.GetFileName(filePath), SearchOption.AllDirectories);
            if (matches.Length == 0)
            {
                return false;
            }
            filePath = matches[0];
        }

        sample = new ModelNetSample(filePath, classIndex);
        return true;
    }

    private List<string> GetSplitFiles(string root)
    {
        var files = new List<string>();
        string trainFile = Path.Combine(root, "modelnet40_train.txt");
        string testFile = Path.Combine(root, "modelnet40_test.txt");
        string valFile = Path.Combine(root, "modelnet40_val.txt");

        switch (_options.Split)
        {
            case DatasetSplit.Train:
                if (File.Exists(trainFile))
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
            case DatasetSplit.All:
                if (File.Exists(trainFile))
                {
                    files.Add(trainFile);
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

    private async Task<List<string>> LoadClassNamesAsync(string root, CancellationToken cancellationToken)
    {
        string namesFile = Path.Combine(root, "modelnet40_shape_names.txt");
        if (File.Exists(namesFile))
        {
            string[] lines = await FilePolyfill.ReadAllLinesAsync(namesFile, cancellationToken);
            var names = lines
                .Select(line => line.Trim())
                .Where(line => line.Length > 0)
                .ToList();
            if (names.Count > 0)
            {
                return names;
            }
        }

        var directories = Directory.GetDirectories(root)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name ?? string.Empty)
            .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
            .ToList();

        return directories;
    }

    private async Task EnsureDataExistsAsync(string dataPath, CancellationToken cancellationToken)
    {
        string rootCandidate = FindDatasetRoot(dataPath);
        if (File.Exists(Path.Combine(rootCandidate, "modelnet40_train.txt")) ||
            File.Exists(Path.Combine(rootCandidate, "modelnet40_shape_names.txt")))
        {
            return;
        }

        if (!_autoDownload)
        {
            throw new FileNotFoundException(
                $"ModelNet40 dataset not found at {dataPath}. " +
                "Provide the dataset locally or enable AutoDownload.");
        }

        await DownloadDatasetAsync(dataPath, cancellationToken);
    }

    private async Task DownloadDatasetAsync(string dataPath, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(dataPath);

        string tempFile = Path.Combine(Path.GetTempPath(), $"modelnet40_{Guid.NewGuid()}.zip");
        try
        {
            using (var httpClient = new HttpClient())
            {
                httpClient.Timeout = TimeSpan.FromMinutes(30);
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
        string direct = Path.Combine(dataPath, "modelnet40_normal_resampled");
        if (Directory.Exists(direct))
        {
            return direct;
        }

        if (Directory.Exists(dataPath))
        {
            string trainFile = Path.Combine(dataPath, "modelnet40_train.txt");
            if (File.Exists(trainFile))
            {
                return dataPath;
            }
        }

        if (Directory.Exists(dataPath))
        {
            string[] matches = Directory.GetDirectories(dataPath, "modelnet40_normal_resampled", SearchOption.AllDirectories);
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
            "modelnet40");
    }

    private sealed class ModelNetSample
    {
        public ModelNetSample(string filePath, int classIndex)
        {
            FilePath = filePath;
            ClassIndex = classIndex;
        }

        public string FilePath { get; }
        public int ClassIndex { get; }
    }
}
