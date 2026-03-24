using System.Text.Json;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the iNaturalist species classification dataset (~2.7M images, 10,000 species in 2021 version).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// iNaturalist uses a JSON annotation file with COCO-style format:
/// <code>
/// {DataPath}/
///   train_val2021/
///     images/
///       00000/
///         00000.jpg
///         ...
///   train.json (or train_mini.json)
///   val.json
/// </code>
/// The dataset has a long-tailed class distribution, making it valuable for imbalanced learning research.
/// Labels are one-hot encoded across the number of species categories.
/// </para>
/// </remarks>
public class INaturalistDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly INaturalistDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _numClasses;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => $"iNaturalist-{_options.Version}";
    /// <inheritdoc/>
    public override string Description => $"iNaturalist {_options.Version} species classification dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Creates a new iNaturalist data loader.</summary>
    public INaturalistDataLoader(INaturalistDataLoaderOptions? options = null)
    {
        _options = options ?? new INaturalistDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath($"inaturalist-{_options.Version}");
        _imageSize = _options.ImageSize;
        _numClasses = 10000; // Default for 2021 version
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Determine annotation file
        string annotationFile = _options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation
            ? Path.Combine(_dataPath, "val.json")
            : Path.Combine(_dataPath, "train.json");

        // Fall back to mini version if full train.json not found
        if (!File.Exists(annotationFile) && _options.Split == Geometry.DatasetSplit.Train)
        {
            annotationFile = Path.Combine(_dataPath, "train_mini.json");
        }

        if (!File.Exists(annotationFile))
        {
            throw new FileNotFoundException(
                $"iNaturalist annotation file not found at {annotationFile}. " +
                "Download the dataset from https://github.com/visipedia/inat_comp and extract to the data path.");
        }

        // Parse JSON annotations (COCO-style)
        using var stream = File.OpenRead(annotationFile);
        var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        var root = doc.RootElement;

        // Build image ID -> file_name map
        var imageMap = new Dictionary<long, string>();
        if (root.TryGetProperty("images", out var imagesElem))
        {
            foreach (var img in imagesElem.EnumerateArray())
            {
                long id = img.GetProperty("id").GetInt64();
                string fileName = img.GetProperty("file_name").GetString() ?? string.Empty;
                imageMap[id] = fileName;
            }
        }

        // Determine number of categories
        if (root.TryGetProperty("categories", out var catsElem))
        {
            _numClasses = catsElem.GetArrayLength();
        }

        // Build category ID -> index map
        var categoryMap = new Dictionary<long, int>();
        if (root.TryGetProperty("categories", out var categoriesElem))
        {
            int idx = 0;
            foreach (var cat in categoriesElem.EnumerateArray())
            {
                long catId = cat.GetProperty("id").GetInt64();
                categoryMap[catId] = idx++;
            }
        }

        // Collect annotations
        var samples = new List<(string ImagePath, int Label)>();
        if (root.TryGetProperty("annotations", out var annotationsElem))
        {
            foreach (var ann in annotationsElem.EnumerateArray())
            {
                long imageId = ann.GetProperty("image_id").GetInt64();
                long categoryId = ann.GetProperty("category_id").GetInt64();

                if (imageMap.TryGetValue(imageId, out var fileName) &&
                    categoryMap.TryGetValue(categoryId, out var classIdx))
                {
                    string fullPath = Path.Combine(_dataPath, fileName);
                    samples.Add((fullPath, classIdx));
                }
            }
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * _numClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (imgPath, label) = samples[i];
            if (File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
                int featureOffset = i * pixelsPerImage;
                int copyLen = Math.Min(pixels.Length, pixelsPerImage);
                Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            }

            if (label >= 0 && label < _numClasses)
                labelsData[i * _numClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
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

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape._dims.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
