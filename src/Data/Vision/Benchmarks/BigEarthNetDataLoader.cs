using System.Text.Json;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the BigEarthNet multi-label remote sensing dataset (590K Sentinel-2 patches, 19 or 43 classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// BigEarthNet expects:
/// <code>
/// {DataPath}/
///   BigEarthNet-S2-v1.0/
///     S2A_MSIL2A_..._patch_0/
///       S2A_MSIL2A_..._B02.tif   (Blue)
///       S2A_MSIL2A_..._B03.tif   (Green)
///       S2A_MSIL2A_..._B04.tif   (Red)
///       ...
///       S2A_MSIL2A_..._labels_metadata.json
/// </code>
/// Each patch directory contains GeoTIFF band files and a JSON metadata file with CORINE Land Cover labels.
/// Labels are multi-hot encoded as Tensor[N, NumClasses].
/// For RGB mode (NumBands=3), only B04(R), B03(G), B02(B) are loaded.
/// </para>
/// </remarks>
public class BigEarthNetDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int PatchSize = 120;
    private const int NumClasses19 = 19;
    private const int NumClasses43 = 43;

    private readonly BigEarthNetDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _numClasses;

    /// <inheritdoc/>
    public override string Name => "BigEarthNet";
    /// <inheritdoc/>
    public override string Description => "BigEarthNet multi-label remote sensing dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => PatchSize * PatchSize * _options.NumBands;
    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Creates a new BigEarthNet data loader.</summary>
    public BigEarthNetDataLoader(BigEarthNetDataLoaderOptions? options = null)
    {
        _options = options ?? new BigEarthNetDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("bigearthnet");
        _numClasses = _options.Use19ClassScheme ? NumClasses19 : NumClasses43;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string dataDir = FindDataDirectory(_dataPath);

        if (dataDir.Length == 0)
        {
            throw new DirectoryNotFoundException(
                $"BigEarthNet data not found at {_dataPath}. " +
                "Download from https://bigearth.net/.");
        }

        // Each patch is a subdirectory
        var patchDirs = Directory.GetDirectories(dataDir)
            .OrderBy(d => Path.GetFileName(d), StringComparer.Ordinal)
            .ToArray();

        int totalSamples = patchDirs.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int numBands = _options.NumBands;
        int pixelsPerImage = PatchSize * PatchSize * numBands;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * _numClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string patchDir = patchDirs[i];
            string patchName = Path.GetFileName(patchDir);

            // Load band files (RGB: B04, B03, B02)
            string[] bandSuffixes = numBands == 3
                ? new[] { "B04", "B03", "B02" }
                : new[] { "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B11", "B12" };

            int featureOffset = i * pixelsPerImage;
            int bandOffset = 0;

            foreach (var bandSuffix in bandSuffixes.Take(numBands))
            {
                string bandFile = Path.Combine(patchDir, $"{patchName}_{bandSuffix}.tif");
                if (File.Exists(bandFile))
                {
                    byte[] fileBytes = await FilePolyfill.ReadAllBytesAsync(bandFile, cancellationToken);
                    int bandPixels = PatchSize * PatchSize;
                    int copyLen = Math.Min(fileBytes.Length, bandPixels);
                    for (int p = 0; p < copyLen; p++)
                    {
                        double value = fileBytes[p];
                        if (_options.Normalize) value /= 255.0;
                        featuresData[featureOffset + bandOffset + p] = NumOps.FromDouble(value);
                    }
                }

                bandOffset += PatchSize * PatchSize;
            }

            // Load labels from metadata JSON
            string metaFile = Path.Combine(patchDir, $"{patchName}_labels_metadata.json");
            if (File.Exists(metaFile))
            {
                string jsonText = await FilePolyfill.ReadAllTextAsync(metaFile, cancellationToken);
                using var doc = JsonDocument.Parse(jsonText);
                var root = doc.RootElement;

                string labelsKey = _options.Use19ClassScheme ? "new_labels" : "labels";
                if (!root.TryGetProperty(labelsKey, out var labelsElem))
                {
                    labelsKey = "labels";
                    root.TryGetProperty(labelsKey, out labelsElem);
                }

                if (labelsElem.ValueKind == JsonValueKind.Array)
                {
                    int labelOffset = i * _numClasses;
                    foreach (var labelElem in labelsElem.EnumerateArray())
                    {
                        // Labels are stored as class index integers or class name strings
                        if (labelElem.ValueKind == JsonValueKind.Number)
                        {
                            int classIdx = labelElem.GetInt32();
                            if (classIdx >= 0 && classIdx < _numClasses)
                            {
                                labelsData[labelOffset + classIdx] = NumOps.One;
                            }
                        }
                    }
                }
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, PatchSize, PatchSize, numBands });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, _numClasses });
        InitializeIndices(totalSamples);
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
        double trainRatio = 0.7, double validationRatio = 0.15, int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);
        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private static string FindDataDirectory(string rootPath)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        string[] candidates = { "BigEarthNet-S2-v1.0", "BigEarthNet-v1.0", "bigearthnet" };
        foreach (var candidate in candidates)
        {
            string path = Path.Combine(rootPath, candidate);
            if (Directory.Exists(path)) return path;
        }

        // Check if root itself contains patch directories
        var subDirs = Directory.GetDirectories(rootPath);
        if (subDirs.Length > 0 && subDirs.Any(d => Path.GetFileName(d).StartsWith("S2")))
            return rootPath;

        return string.Empty;
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
