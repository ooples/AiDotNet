using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the EuroSAT land use/land cover classification dataset (27K patches, 64x64 RGB, 10 classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// EuroSAT expects a folder-per-class structure:
/// <code>
/// {DataPath}/
///   EuroSAT/       (or 2750/)
///     AnnualCrop/
///       AnnualCrop_1.jpg
///       ...
///     Forest/
///     HerbaceousVegetation/
///     Highway/
///     Industrial/
///     Pasture/
///     PermanentCrop/
///     Residential/
///     River/
///     SeaLake/
/// </code>
/// Each class folder contains ~2,000-3,000 images. Labels are one-hot encoded as Tensor[N, 10].
/// </para>
/// </remarks>
public class EuroSatDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip";

    private static readonly string[] ClassLabels =
    {
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    };

    private const int NumClasses = 10;
    private const int ImageSize = 64;

    private readonly EuroSatDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "EuroSAT";
    /// <inheritdoc/>
    public override string Description => "EuroSAT land use/land cover classification dataset (10 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => ImageSize * ImageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;
    /// <summary>Gets the class names.</summary>
    public IReadOnlyList<string> ClassNames => ClassLabels;

    /// <summary>Creates a new EuroSAT data loader.</summary>
    public EuroSatDataLoader(EuroSatDataLoaderOptions? options = null)
    {
        _options = options ?? new EuroSatDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("eurosat");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string dataDir = FindDataDirectory(_dataPath);

        if (dataDir.Length == 0 && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractZipAsync(DownloadUrl, _dataPath, cancellationToken);
            dataDir = FindDataDirectory(_dataPath);
        }

        if (dataDir.Length == 0)
        {
            throw new DirectoryNotFoundException(
                $"EuroSAT data not found at {_dataPath}. Enable AutoDownload or provide data locally.");
        }

        var imagePaths = new List<(string Path, int Label)>();

        for (int classIdx = 0; classIdx < ClassLabels.Length; classIdx++)
        {
            string classDir = Path.Combine(dataDir, ClassLabels[classIdx]);
            if (!Directory.Exists(classDir)) continue;

            var files = Directory.EnumerateFiles(classDir, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".tif", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase));

            foreach (var file in files)
            {
                imagePaths.Add((file, classIdx));
            }
        }

        int totalSamples = imagePaths.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int pixelsPerImage = ImageSize * ImageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (imgPath, label) = imagePaths[i];
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, ImageSize, ImageSize, 3, _options.Normalize);

            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);

            if (label >= 0 && label < NumClasses)
                labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, ImageSize, ImageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
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

        string[] candidates = { "EuroSAT", "EuroSAT_RGB", "2750" };
        foreach (var candidate in candidates)
        {
            string path = Path.Combine(rootPath, candidate);
            if (Directory.Exists(path) && Directory.Exists(Path.Combine(path, "AnnualCrop")))
                return path;
        }

        if (Directory.Exists(Path.Combine(rootPath, "AnnualCrop")))
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
