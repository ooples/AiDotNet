using System.Globalization;
using AiDotNet.Geometry.IO;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Loads the ScanNet semantic segmentation dataset.
/// </summary>
public sealed class ScanNetSemanticSegmentationDataLoader<T> : PointCloudDatasetLoaderBase<T>
{
    private static readonly int[] Train20Ids =
    {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 14, 16, 24, 28, 33, 34, 36, 39
    };

    private static readonly string[] Train20Names =
    {
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "other furniture"
    };

    private static readonly Dictionary<int, int> Train20IdToIndex = BuildTrain20Map();

    private readonly ScanNetSemanticSegmentationDataLoaderOptions _options;
    private readonly string _dataPath;
    private readonly bool _autoDownload;
    private List<string> _classNames = new();

    /// <inheritdoc />
    public override string Name => "ScanNetSemanticSegmentation";

    /// <inheritdoc />
    public override string Description => "ScanNet semantic segmentation dataset loader.";

    /// <summary>
    /// Gets the class names for the selected label mode.
    /// </summary>
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>
    /// Gets the number of classes for the selected label mode.
    /// </summary>
    public int NumClasses => GetNumClasses();

    /// <summary>
    /// Initializes a new instance of the <see cref="ScanNetSemanticSegmentationDataLoader{T}"/> class.
    /// </summary>
    public ScanNetSemanticSegmentationDataLoader(ScanNetSemanticSegmentationDataLoaderOptions? options = null)
    {
        _options = options ?? new ScanNetSemanticSegmentationDataLoaderOptions();
        _dataPath = _options.DataPath ?? GetDefaultDataPath();
        _autoDownload = _options.AutoDownload;
    }

    /// <inheritdoc />
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        await EnsureDataExistsAsync(_dataPath, cancellationToken);
        string root = FindDatasetRoot(_dataPath);

        var sceneIds = await LoadSceneIdsAsync(root, cancellationToken);
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < sceneIds.Count)
        {
            sceneIds = sceneIds.Take(_options.MaxSamples.Value).ToList();
        }

        if (sceneIds.Count == 0)
        {
            throw new InvalidOperationException("ScanNet dataset contains no scenes for the requested split.");
        }

        int featureDim = 3 + (_options.IncludeColors ? 3 : 0) + (_options.IncludeNormals ? 3 : 0);
        int numClasses = GetNumClasses();

        var featuresData = new T[sceneIds.Count * _options.PointsPerSample * featureDim];
        var labelsData = new T[sceneIds.Count * _options.PointsPerSample];

        var random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        ScanNetLabelMapping? labelMapping = null;
        if (RequiresLabelMapping(root))
        {
            string mappingPath = FindLabelMappingFile(root);
            labelMapping = await LoadScanNetLabelMappingAsync(mappingPath, cancellationToken);
            _classNames = BuildClassNames(labelMapping);
        }
        else
        {
            _classNames = BuildClassNames(null);
        }

        for (int sampleIndex = 0; sampleIndex < sceneIds.Count; sampleIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string sceneId = sceneIds[sampleIndex];
            bool useRaw = ShouldUseRawFormat(root, sceneId);

            if (useRaw)
            {
                if (!TryGetRawSceneFiles(root, sceneId, out RawSceneFiles rawFiles))
                {
                    throw new FileNotFoundException($"Raw ScanNet files not found for scene {sceneId}.");
                }

                if (labelMapping == null)
                {
                    string mappingPath = FindLabelMappingFile(root);
                    labelMapping = await LoadScanNetLabelMappingAsync(mappingPath, cancellationToken);
                }

                var rawData = await LoadRawSceneAsync(rawFiles, labelMapping, cancellationToken);
                FillSceneSample(rawData, rawData.Labels, featureDim, numClasses, sampleIndex, random, featuresData, labelsData);
            }
            else
            {
                if (!TryGetPreprocessedSceneFiles(root, sceneId, out PreprocessedSceneFiles files))
                {
                    throw new FileNotFoundException($"Preprocessed ScanNet files not found for scene {sceneId}.");
                }

                var preprocessed = await LoadPreprocessedSceneAsync(files, cancellationToken);
                FillSceneSample(preprocessed, preprocessed.Labels, featureDim, numClasses, sampleIndex, random, featuresData, labelsData);
            }
        }

        var features = new Tensor<T>(featuresData, new[] { sceneIds.Count, _options.PointsPerSample, featureDim });
        var labels = new Tensor<T>(labelsData, new[] { sceneIds.Count, _options.PointsPerSample });
        SetLoadedData(features, labels);
    }

    /// <inheritdoc />
    protected override void UnloadDataCore()
    {
        LoadedFeatures = null;
        LoadedLabels = null;
        _classNames = new List<string>();
    }

    private void FillSceneSample(
        ScenePointData points,
        IReadOnlyList<int> labels,
        int featureDim,
        int numClasses,
        int sampleIndex,
        Random random,
        T[] featuresData,
        T[] labelsData)
    {
        int pointCount = points.Count;
        int[] indices = BuildSampleIndices(pointCount, _options.PointsPerSample, _options.SamplingStrategy, _options.PaddingStrategy, random);

        int sampleOffset = sampleIndex * _options.PointsPerSample * featureDim;
        for (int i = 0; i < indices.Length; i++)
        {
            int rowIndex = indices[i];
            int destOffset = sampleOffset + i * featureDim;

            if (rowIndex < 0 || rowIndex >= pointCount)
            {
                continue;
            }

            var point = points.GetPoint(rowIndex);
            featuresData[destOffset] = NumOps.FromDouble(point.X);
            featuresData[destOffset + 1] = NumOps.FromDouble(point.Y);
            featuresData[destOffset + 2] = NumOps.FromDouble(point.Z);

            int offset = destOffset + 3;
            if (_options.IncludeColors)
            {
                featuresData[offset] = NumOps.FromDouble(NormalizeColor(point.R));
                featuresData[offset + 1] = NumOps.FromDouble(NormalizeColor(point.G));
                featuresData[offset + 2] = NumOps.FromDouble(NormalizeColor(point.B));
                offset += 3;
            }

            if (_options.IncludeNormals)
            {
                featuresData[offset] = NumOps.FromDouble(point.Nx);
                featuresData[offset + 1] = NumOps.FromDouble(point.Ny);
                featuresData[offset + 2] = NumOps.FromDouble(point.Nz);
            }
        }

        int labelOffset = sampleIndex * _options.PointsPerSample;
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
            if (labelValue >= numClasses)
            {
                labelValue = numClasses - 1;
            }

            labelsData[labelOffset + i] = NumOps.FromDouble(labelValue);
        }
    }

    private async Task<ScenePointData> LoadPreprocessedSceneAsync(PreprocessedSceneFiles files, CancellationToken cancellationToken)
    {
        string[] lines = await FilePolyfill.ReadAllLinesAsync(files.PointsPath, cancellationToken);
        var rows = new List<double[]>(lines.Length);
        var labels = new List<int>(lines.Length);

        bool useLabelColumn = false;
        int expectedColumns = 3 + (_options.IncludeColors ? 3 : 0) + (_options.IncludeNormals ? 3 : 0);

        if (files.LabelsPath == null && _options.AutoDetectLabelColumn)
        {
            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                string[] tokens = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length > expectedColumns &&
                    int.TryParse(tokens[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
                {
                    useLabelColumn = true;
                    break;
                }
            }
        }

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] tokens = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length < 3)
            {
                continue;
            }

            int tokenCount = tokens.Length;
            int labelValue = 0;

            if (useLabelColumn && tokenCount > expectedColumns)
            {
                if (int.TryParse(tokens[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsedLabel))
                {
                    labelValue = parsedLabel;
                }

                tokenCount -= 1;
            }

            var values = new double[tokenCount];
            for (int i = 0; i < tokenCount; i++)
            {
                if (!double.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
                {
                    value = 0.0;
                }
                values[i] = value;
            }

            rows.Add(values);

            if (useLabelColumn)
            {
                labels.Add(labelValue);
            }
        }

        if (files.LabelsPath != null)
        {
            labels = await ReadLabelValuesAsync(files.LabelsPath, cancellationToken);
        }
        else if (!useLabelColumn)
        {
            labels = Enumerable.Repeat(0, rows.Count).ToList();
        }

        return new ScenePointData(rows, labels);
    }
    private async Task<ScenePointData> LoadRawSceneAsync(
        RawSceneFiles files,
        ScanNetLabelMapping labelMapping,
        CancellationToken cancellationToken)
    {
        PointCloudData<double> pointCloud = PlyMeshIO.LoadPointCloud<double>(files.MeshPath);
        int pointCount = pointCloud.NumPoints;

        int[] segIndices = await ReadSegIndicesAsync(files.SegmentsPath, cancellationToken);
        if (segIndices.Length != pointCount)
        {
            int minCount = Math.Min(segIndices.Length, pointCount);
            Array.Resize(ref segIndices, minCount);
            pointCount = minCount;
        }

        var segmentToLabel = await ReadAggregationLabelsAsync(files.AggregationPath, cancellationToken);
        var labels = new List<int>(pointCount);

        for (int i = 0; i < pointCount; i++)
        {
            int segId = segIndices[i];
            int nyu40Id = 0;
            if (segmentToLabel.TryGetValue(segId, out string? labelName) &&
                labelMapping.LabelToNyu40.TryGetValue(labelName, out int mapped))
            {
                nyu40Id = mapped;
            }

            labels.Add(MapScanNetLabel(nyu40Id));
        }

        var rows = BuildPointRowsFromPointCloud(pointCloud, pointCount);
        return new ScenePointData(rows, labels);
    }

    private List<double[]> BuildPointRowsFromPointCloud(PointCloudData<double> pointCloud, int pointCount)
    {
        var rows = new List<double[]>(pointCount);
        int featureDim = pointCloud.NumFeatures;
        var data = pointCloud.Points.Data;

        bool hasColors = featureDim >= 6;
        bool hasNormals = featureDim >= (hasColors ? 9 : 6);
        int colorOffset = hasColors ? 3 : -1;
        int normalOffset = hasNormals ? (hasColors ? 6 : 3) : -1;

        for (int i = 0; i < pointCount; i++)
        {
            int baseOffset = i * featureDim;
            double x = data[baseOffset];
            double y = data[baseOffset + 1];
            double z = data[baseOffset + 2];

            double r = hasColors ? data[baseOffset + colorOffset] : 0.0;
            double g = hasColors ? data[baseOffset + colorOffset + 1] : 0.0;
            double b = hasColors ? data[baseOffset + colorOffset + 2] : 0.0;

            double nx = hasNormals ? data[baseOffset + normalOffset] : 0.0;
            double ny = hasNormals ? data[baseOffset + normalOffset + 1] : 0.0;
            double nz = hasNormals ? data[baseOffset + normalOffset + 2] : 0.0;

            var row = new double[9];
            row[0] = x;
            row[1] = y;
            row[2] = z;
            row[3] = r;
            row[4] = g;
            row[5] = b;
            row[6] = nx;
            row[7] = ny;
            row[8] = nz;
            rows.Add(row);
        }

        return rows;
    }

    private async Task<int[]> ReadSegIndicesAsync(string segmentsPath, CancellationToken cancellationToken)
    {
        string json = await FilePolyfill.ReadAllTextAsync(segmentsPath, cancellationToken);
        var root = JObject.Parse(json);
        var array = root["segIndices"] as JArray;
        if (array == null)
        {
            return Array.Empty<int>();
        }

        var segIndices = new int[array.Count];
        for (int i = 0; i < array.Count; i++)
        {
            segIndices[i] = array[i]?.Value<int>() ?? 0;
        }

        return segIndices;
    }

    private async Task<Dictionary<int, string>> ReadAggregationLabelsAsync(string aggregationPath, CancellationToken cancellationToken)
    {
        string json = await FilePolyfill.ReadAllTextAsync(aggregationPath, cancellationToken);
        var root = JObject.Parse(json);
        var segGroups = root["segGroups"] as JArray;
        var segmentToLabel = new Dictionary<int, string>();

        if (segGroups == null)
        {
            return segmentToLabel;
        }

        foreach (var groupToken in segGroups)
        {
            string? labelName = groupToken?["label"]?.Value<string>();
            if (string.IsNullOrWhiteSpace(labelName))
            {
                continue;
            }

            var segments = groupToken?["segments"] as JArray;
            if (segments == null)
            {
                continue;
            }

            foreach (var segToken in segments)
            {
                int segId = segToken?.Value<int>() ?? -1;
                if (segId >= 0)
                {
                    segmentToLabel[segId] = labelName;
                }
            }
        }

        return segmentToLabel;
    }

    private double NormalizeColor(double value)
    {
        if (!_options.NormalizeColors)
        {
            return value;
        }

        return value > 1.0 ? value / 255.0 : value;
    }

    private int MapScanNetLabel(int nyu40Id)
    {
        if (_options.LabelMode == ScanNetLabelMode.Train20)
        {
            if (!Train20IdToIndex.TryGetValue(nyu40Id, out int mapped))
            {
                return _options.IncludeUnknownClass ? 0 : 0;
            }

            return _options.IncludeUnknownClass ? mapped + 1 : mapped;
        }

        if (nyu40Id <= 0)
        {
            return _options.IncludeUnknownClass ? 0 : 0;
        }

        return _options.IncludeUnknownClass ? nyu40Id : nyu40Id - 1;
    }

    private int GetNumClasses()
    {
        int baseCount = _options.LabelMode == ScanNetLabelMode.Train20 ? 20 : 40;
        return _options.IncludeUnknownClass ? baseCount + 1 : baseCount;
    }

    private bool RequiresLabelMapping(string root)
    {
        return _options.InputFormat == ScanNetInputFormat.RawScanNet ||
               (_options.InputFormat == ScanNetInputFormat.Auto && HasRawSceneData(root));
    }

    private bool ShouldUseRawFormat(string root, string sceneId)
    {
        if (_options.InputFormat == ScanNetInputFormat.RawScanNet)
        {
            return true;
        }

        if (_options.InputFormat == ScanNetInputFormat.PreprocessedText)
        {
            return false;
        }

        return TryGetRawSceneFiles(root, sceneId, out _);
    }

    private bool HasRawSceneData(string root)
    {
        string scansDir = Path.Combine(root, "scans");
        if (!Directory.Exists(scansDir))
        {
            return false;
        }

        return Directory.GetFiles(scansDir, "*_vh_clean_2.ply", SearchOption.AllDirectories).Length > 0;
    }

    private bool TryGetRawSceneFiles(string root, string sceneId, out RawSceneFiles files)
    {
        files = new RawSceneFiles(string.Empty, string.Empty, string.Empty);
        string sceneDir = Path.Combine(root, "scans", sceneId);
        if (!Directory.Exists(sceneDir))
        {
            return false;
        }

        string meshPath = Path.Combine(sceneDir, sceneId + "_vh_clean_2.ply");
        if (!File.Exists(meshPath))
        {
            meshPath = Path.Combine(sceneDir, sceneId + "_vh_clean.ply");
        }

        if (!File.Exists(meshPath))
        {
            return false;
        }

        string[] segCandidates = Directory.GetFiles(sceneDir, sceneId + "_vh_clean_2.*.segs.json", SearchOption.TopDirectoryOnly);
        if (segCandidates.Length == 0)
        {
            segCandidates = Directory.GetFiles(sceneDir, sceneId + "_vh_clean.*.segs.json", SearchOption.TopDirectoryOnly);
        }

        if (segCandidates.Length == 0)
        {
            return false;
        }

        string aggregationPath = Path.Combine(sceneDir, sceneId + ".aggregation.json");
        if (!File.Exists(aggregationPath))
        {
            return false;
        }

        string mappingPath = FindLabelMappingFile(root);
        if (string.IsNullOrWhiteSpace(mappingPath))
        {
            return false;
        }

        files = new RawSceneFiles(meshPath, segCandidates[0], aggregationPath);
        return true;
    }

    private bool TryGetPreprocessedSceneFiles(string root, string sceneId, out PreprocessedSceneFiles files)
    {
        files = new PreprocessedSceneFiles(string.Empty, null);
        string? pointsPath = FindPreprocessedPointsPath(root, sceneId);
        if (string.IsNullOrWhiteSpace(pointsPath))
        {
            return false;
        }

        string? labelsPath = FindPreprocessedLabelsPath(pointsPath);
        files = new PreprocessedSceneFiles(pointsPath, labelsPath);
        return true;
    }

    private string? FindPreprocessedPointsPath(string root, string sceneId)
    {
        var candidates = new List<string>
        {
            Path.Combine(root, "scans", sceneId, sceneId + ".txt"),
            Path.Combine(root, "scans", sceneId, sceneId + ".pts"),
            Path.Combine(root, sceneId + ".txt"),
            Path.Combine(root, sceneId + ".pts"),
            Path.Combine(root, "points", sceneId + ".txt"),
            Path.Combine(root, "points", sceneId + ".pts")
        };

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private string? FindPreprocessedLabelsPath(string pointsPath)
    {
        string directory = Path.GetDirectoryName(pointsPath) ?? string.Empty;
        string fileName = Path.GetFileNameWithoutExtension(pointsPath);

        var candidates = new List<string>
        {
            Path.Combine(directory, fileName + ".labels"),
            Path.Combine(directory, fileName + ".labels.txt"),
            Path.Combine(directory, fileName + ".label"),
            Path.Combine(directory, fileName + ".seg"),
            Path.Combine(directory, fileName + ".seg.txt")
        };

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private async Task<List<string>> LoadSceneIdsAsync(string root, CancellationToken cancellationToken)
    {
        var sceneIds = new List<string>();
        string trainFile = Path.Combine(root, "scannetv2_train.txt");
        string valFile = Path.Combine(root, "scannetv2_val.txt");
        string testFile = Path.Combine(root, "scannetv2_test.txt");

        switch (_options.Split)
        {
            case DatasetSplit.Train:
                if (File.Exists(trainFile))
                {
                    sceneIds.AddRange(await ReadSceneListAsync(trainFile, cancellationToken));
                }
                break;
            case DatasetSplit.Validation:
                if (File.Exists(valFile))
                {
                    sceneIds.AddRange(await ReadSceneListAsync(valFile, cancellationToken));
                }
                else if (File.Exists(trainFile))
                {
                    sceneIds.AddRange(await ReadSceneListAsync(trainFile, cancellationToken));
                }
                break;
            case DatasetSplit.Test:
                if (File.Exists(testFile))
                {
                    sceneIds.AddRange(await ReadSceneListAsync(testFile, cancellationToken));
                }
                break;
            case DatasetSplit.All:
                if (File.Exists(trainFile))
                {
                    sceneIds.AddRange(await ReadSceneListAsync(trainFile, cancellationToken));
                }
                if (File.Exists(valFile))
                {
                    sceneIds.AddRange(await ReadSceneListAsync(valFile, cancellationToken));
                }
                if (File.Exists(testFile))
                {
                    sceneIds.AddRange(await ReadSceneListAsync(testFile, cancellationToken));
                }
                break;
            default:
                break;
        }

        if (sceneIds.Count == 0)
        {
            string scansDir = Path.Combine(root, "scans");
            if (Directory.Exists(scansDir))
            {
                var names = Directory.GetDirectories(scansDir)
                    .Select(Path.GetFileName)
                    .Where(name => !string.IsNullOrWhiteSpace(name))
                    .Cast<string>();
                sceneIds.AddRange(names);
            }
        }

        return sceneIds.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    private async Task<List<string>> ReadSceneListAsync(string filePath, CancellationToken cancellationToken)
    {
        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);
        return lines.Select(line => line.Trim()).Where(line => line.Length > 0).ToList();
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
        if (Directory.Exists(Path.Combine(rootCandidate, "scans")) ||
            File.Exists(Path.Combine(rootCandidate, "scannetv2_train.txt")))
        {
            return;
        }

        if (_autoDownload)
        {
            throw new NotSupportedException("ScanNet requires manual download and license agreement.");
        }

        await Task.CompletedTask;
        throw new FileNotFoundException(
            $"ScanNet dataset not found at {dataPath}. " +
            "Provide the dataset locally.");
    }

    private string FindDatasetRoot(string dataPath)
    {
        if (Directory.Exists(dataPath) &&
            (Directory.Exists(Path.Combine(dataPath, "scans")) || File.Exists(Path.Combine(dataPath, "scannetv2_train.txt"))))
        {
            return dataPath;
        }

        if (Directory.Exists(dataPath))
        {
            string[] scansDirs = Directory.GetDirectories(dataPath, "scans", SearchOption.AllDirectories);
            if (scansDirs.Length > 0)
            {
                return Path.GetDirectoryName(scansDirs[0]) ?? dataPath;
            }
        }

        return dataPath;
    }

    private string FindLabelMappingFile(string root)
    {
        string direct = Path.Combine(root, "scannetv2-labels.combined.tsv");
        if (File.Exists(direct))
        {
            return direct;
        }

        if (Directory.Exists(root))
        {
            string[] matches = Directory.GetFiles(root, "scannetv2-labels.combined.tsv", SearchOption.AllDirectories);
            if (matches.Length > 0)
            {
                return matches[0];
            }
        }

        return string.Empty;
    }

    private async Task<ScanNetLabelMapping> LoadScanNetLabelMappingAsync(string mappingPath, CancellationToken cancellationToken)
    {
        string[] lines = await FilePolyfill.ReadAllLinesAsync(mappingPath, cancellationToken);
        var labelToNyu = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var nyuIdToName = new Dictionary<int, string>();

        if (lines.Length == 0)
        {
            return new ScanNetLabelMapping(labelToNyu, nyuIdToName);
        }

        string[] header = lines[0].Split('\t');
        int rawIndex = Array.IndexOf(header, "raw_category");
        int categoryIndex = Array.IndexOf(header, "category");
        int nyuIndex = Array.IndexOf(header, "nyu40id");
        int nyuNameIndex = Array.IndexOf(header, "nyu40class");

        if (nyuIndex < 0)
        {
            return new ScanNetLabelMapping(labelToNyu, nyuIdToName);
        }

        for (int i = 1; i < lines.Length; i++)
        {
            string line = lines[i];
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split('\t');
            if (parts.Length <= Math.Max(nyuIndex, Math.Max(rawIndex, categoryIndex)))
            {
                continue;
            }

            if (!int.TryParse(parts[nyuIndex], NumberStyles.Integer, CultureInfo.InvariantCulture, out int nyuId))
            {
                continue;
            }

            if (rawIndex >= 0 && rawIndex < parts.Length)
            {
                string raw = parts[rawIndex];
                if (!string.IsNullOrWhiteSpace(raw))
                {
                    labelToNyu[raw] = nyuId;
                }
            }

            if (categoryIndex >= 0 && categoryIndex < parts.Length)
            {
                string category = parts[categoryIndex];
                if (!string.IsNullOrWhiteSpace(category))
                {
                    labelToNyu[category] = nyuId;
                }
            }

            if (nyuNameIndex >= 0 && nyuNameIndex < parts.Length)
            {
                string nyuName = parts[nyuNameIndex];
                if (!string.IsNullOrWhiteSpace(nyuName) && !nyuIdToName.ContainsKey(nyuId))
                {
                    nyuIdToName[nyuId] = nyuName;
                }
            }
        }

        return new ScanNetLabelMapping(labelToNyu, nyuIdToName);
    }

    private List<string> BuildClassNames(ScanNetLabelMapping? mapping)
    {
        var names = new List<string>();
        if (_options.LabelMode == ScanNetLabelMode.Train20)
        {
            names.AddRange(Train20Names);
        }
        else if (mapping != null && mapping.Nyu40IdToName.Count > 0)
        {
            var ordered = mapping.Nyu40IdToName
                .Where(pair => pair.Key > 0)
                .OrderBy(pair => pair.Key)
                .Select(pair => pair.Value)
                .ToList();
            names.AddRange(ordered);
        }
        else
        {
            for (int i = 1; i <= 40; i++)
            {
                names.Add($"class_{i}");
            }
        }

        if (_options.IncludeUnknownClass)
        {
            names.Insert(0, "unknown");
        }

        return names;
    }

    private string GetDefaultDataPath()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".aidotnet",
            "datasets",
            "scannet");
    }

    private static Dictionary<int, int> BuildTrain20Map()
    {
        var map = new Dictionary<int, int>();
        for (int i = 0; i < Train20Ids.Length; i++)
        {
            map[Train20Ids[i]] = i;
        }

        return map;
    }

    private sealed class ScanNetLabelMapping
    {
        public ScanNetLabelMapping(Dictionary<string, int> labelToNyu40, Dictionary<int, string> nyu40IdToName)
        {
            LabelToNyu40 = labelToNyu40;
            Nyu40IdToName = nyu40IdToName;
        }

        public Dictionary<string, int> LabelToNyu40 { get; }
        public Dictionary<int, string> Nyu40IdToName { get; }
    }

    private sealed class RawSceneFiles
    {
        public RawSceneFiles(string meshPath, string segmentsPath, string aggregationPath)
        {
            MeshPath = meshPath;
            SegmentsPath = segmentsPath;
            AggregationPath = aggregationPath;
        }

        public string MeshPath { get; }
        public string SegmentsPath { get; }
        public string AggregationPath { get; }
    }

    private sealed class PreprocessedSceneFiles
    {
        public PreprocessedSceneFiles(string pointsPath, string? labelsPath)
        {
            PointsPath = pointsPath;
            LabelsPath = labelsPath;
        }

        public string PointsPath { get; }
        public string? LabelsPath { get; }
    }

    private sealed class ScenePointData
    {
        public ScenePointData(List<double[]> pointRows, List<int> labels)
        {
            PointRows = pointRows;
            Labels = labels;
        }

        public List<double[]> PointRows { get; }
        public List<int> Labels { get; }

        public int Count => PointRows.Count;

        public PointSample GetPoint(int index)
        {
            double[] row = PointRows[index];
            double x = row.Length > 0 ? row[0] : 0.0;
            double y = row.Length > 1 ? row[1] : 0.0;
            double z = row.Length > 2 ? row[2] : 0.0;
            double r = row.Length > 3 ? row[3] : 0.0;
            double g = row.Length > 4 ? row[4] : 0.0;
            double b = row.Length > 5 ? row[5] : 0.0;
            double nx = row.Length > 6 ? row[6] : 0.0;
            double ny = row.Length > 7 ? row[7] : 0.0;
            double nz = row.Length > 8 ? row[8] : 0.0;

            return new PointSample(x, y, z, r, g, b, nx, ny, nz);
        }
    }

    private readonly struct PointSample
    {
        public PointSample(double x, double y, double z, double r, double g, double b, double nx, double ny, double nz)
        {
            X = x;
            Y = y;
            Z = z;
            R = r;
            G = g;
            B = b;
            Nx = nx;
            Ny = ny;
            Nz = nz;
        }

        public double X { get; }
        public double Y { get; }
        public double Z { get; }
        public double R { get; }
        public double G { get; }
        public double B { get; }
        public double Nx { get; }
        public double Ny { get; }
        public double Nz { get; }
    }
}

