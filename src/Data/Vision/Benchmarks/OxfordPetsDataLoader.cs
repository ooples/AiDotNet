using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Oxford-IIIT Pet dataset (Parkhi et al. 2012) — 37 dog/cat breeds.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/
///   images/{Breed}_{N}.jpg
///   annotations/trainval.txt   (id breed_id species_id breed_id_v2)
///   annotations/test.txt
/// </code>
/// Auto-download fetches the canonical robots.ox.ac.uk tarballs (images +
/// annotations as separate archives).
/// </para>
/// </remarks>
public class OxfordPetsDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 37;
    private static readonly string ImagesUrl = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz";
    private static readonly string AnnotationsUrl = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz";

    private readonly OxfordPetsDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    public override string Name => "Oxford-IIIT Pets";
    public override string Description => "Oxford-IIIT Pet dataset (37 breeds)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _imageSize * _imageSize * 3;
    public override int OutputDimension => NumClasses;

    public OxfordPetsDataLoader(OxfordPetsDataLoaderOptions? options = null)
    {
        _options = options ?? new OxfordPetsDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("oxford-pets");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string imagesDir = Path.Combine(_dataPath, "images");
        string annDir = Path.Combine(_dataPath, "annotations");
        if (_options.AutoDownload)
        {
            if (!Directory.Exists(imagesDir))
                await DatasetDownloader.DownloadAndExtractTarGzAsync(ImagesUrl, _dataPath, cancellationToken);
            if (!Directory.Exists(annDir))
                await DatasetDownloader.DownloadAndExtractTarGzAsync(AnnotationsUrl, _dataPath, cancellationToken);
        }
        if (!Directory.Exists(imagesDir) || !Directory.Exists(annDir))
            throw new DirectoryNotFoundException($"Oxford-IIIT Pets data not found at {_dataPath}.");

        string splitFile = _options.Split == Geometry.DatasetSplit.Test
            ? Path.Combine(annDir, "test.txt")
            : Path.Combine(annDir, "trainval.txt");
        if (!File.Exists(splitFile))
            throw new FileNotFoundException($"Pets split file not found: {splitFile}");

        // Each line: "<image_id> <class_id> <species> <breed_id>"
        // class_id in [1, 37]; we convert to 0-indexed.
        var paths = new List<(string Path, int Label)>();
        foreach (string line in await FilePolyfill.ReadAllLinesAsync(splitFile, cancellationToken))
        {
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
            var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;
            if (!int.TryParse(parts[1], out int classId1)) continue;
            int label = classId1 - 1;
            if (label < 0 || label >= NumClasses) continue;
            string imgPath = Path.Combine(imagesDir, parts[0] + ".jpg");
            if (!File.Exists(imgPath)) continue;
            paths.Add((imgPath, label));
        }

        int totalSamples = paths.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (imgPath, label) = paths[i];
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
        InitializeIndices(totalSamples);
        await Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore() { LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0; }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (ExtractTensorBatchLocal(features, indices), ExtractTensorBatchLocal(labels, indices));
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
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();
        var features = LoadedFeatures ?? throw new InvalidOperationException("Not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Take(trainSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Skip(trainSize + valSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private static Tensor<T> ExtractTensorBatchLocal(Tensor<T> source, int[] indices)
    {
        var result = AiDotNet.Helpers.TensorCopyHelper.CreateEmptyBatchLike(source, indices.Length);
        for (int i = 0; i < indices.Length; i++)
            AiDotNet.Helpers.TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
