using System.Text.Json;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Functional Map of the World (fMoW) satellite imagery dataset (1M+ images, 62 categories).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// fMoW expects:
/// <code>
/// {DataPath}/
///   train/
///     airport/
///       airport_0/
///         airport_0_0_rgb.jpg
///         airport_0_0_rgb.json
///         ...
///     barn/
///     ...
///   val/
///   test/
/// </code>
/// Each category has sequenced image folders with RGB images and JSON metadata.
/// Labels are one-hot encoded as Tensor[N, 62].
/// </para>
/// </remarks>
public class FMoWDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string[] ClassLabels =
    {
        "airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture",
        "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership",
        "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution",
        "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
        "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital",
        "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility",
        "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility",
        "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison",
        "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway",
        "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm",
        "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool",
        "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
        "wind_farm", "zoo"
    };

    private static readonly Dictionary<string, int> ClassToIndex;

    static FMoWDataLoader()
    {
        ClassToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < ClassLabels.Length; i++)
            ClassToIndex[ClassLabels[i]] = i;
    }

    private const int NumClasses = 62;

    private readonly FMoWDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => "fMoW";
    /// <inheritdoc/>
    public override string Description => "Functional Map of the World satellite imagery dataset (62 categories)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;
    /// <summary>Gets the class names.</summary>
    public IReadOnlyList<string> ClassNames => ClassLabels;

    /// <summary>Creates a new fMoW data loader.</summary>
    public FMoWDataLoader(FMoWDataLoaderOptions? options = null)
    {
        _options = options ?? new FMoWDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("fmow");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "val",
            _ => "train"
        };
        string splitDir = Path.Combine(_dataPath, splitName);

        if (!Directory.Exists(splitDir))
        {
            throw new DirectoryNotFoundException(
                $"fMoW data not found at {splitDir}. " +
                "Download from https://github.com/fMoW/dataset.");
        }

        var imagePaths = new List<(string Path, int Label)>();

        // Iterate class directories
        var classDirs = Directory.GetDirectories(splitDir);
        foreach (var classDir in classDirs)
        {
            string className = Path.GetFileName(classDir);
            if (!ClassToIndex.TryGetValue(className, out int classIdx)) continue;

            // Each class has sequence directories
            var seqDirs = Directory.GetDirectories(classDir);
            foreach (var seqDir in seqDirs)
            {
                var files = Directory.EnumerateFiles(seqDir, "*_rgb.jpg", SearchOption.TopDirectoryOnly)
                    .Concat(Directory.EnumerateFiles(seqDir, "*_rgb.jpeg", SearchOption.TopDirectoryOnly));

                foreach (var file in files)
                {
                    imagePaths.Add((file, classIdx));
                }
            }

            // Also check for images directly in the class folder
            var directFiles = Directory.EnumerateFiles(classDir, "*_rgb.jpg", SearchOption.TopDirectoryOnly)
                .Concat(Directory.EnumerateFiles(classDir, "*.jpg", SearchOption.TopDirectoryOnly));
            foreach (var file in directFiles)
            {
                if (!imagePaths.Any(x => x.Path == file))
                    imagePaths.Add((file, classIdx));
            }
        }

        int totalSamples = imagePaths.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (imgPath, label) = imagePaths[i];
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);

            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);

            if (label >= 0 && label < NumClasses)
                labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
        InitializeIndices(totalSamples);

        await Task.CompletedTask;
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

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.ToArray().Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
