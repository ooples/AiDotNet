using System.Text.Json;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the COCO 2017 object detection dataset (118K train / 5K val, 80 categories).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// COCO Detection expects:
/// <code>
/// {DataPath}/
///   train2017/
///     000000000009.jpg
///     ...
///   val2017/
///     ...
///   annotations/
///     instances_train2017.json
///     instances_val2017.json
/// </code>
/// Labels are stored as Tensor[N, MaxDetections, 5] where each detection is (class_id, x, y, w, h)
/// with bounding box coordinates normalized to [0, 1]. Unused detection slots are zero-padded.
/// </para>
/// </remarks>
public class CocoDetectionDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string TrainImagesUrl = "https://images.cocodataset.org/zips/train2017.zip";
    private static readonly string ValImagesUrl = "https://images.cocodataset.org/zips/val2017.zip";
    private static readonly string AnnotationsUrl = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip";

    private const int NumClasses = 80;

    private readonly CocoDetectionDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;
    private int _maxDetections;

    /// <inheritdoc/>
    public override string Name => "COCO-Detection";
    /// <inheritdoc/>
    public override string Description => "COCO 2017 object detection dataset (80 categories)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => _maxDetections * 5;

    /// <summary>Creates a new COCO Detection data loader.</summary>
    public CocoDetectionDataLoader(CocoDetectionDataLoaderOptions? options = null)
    {
        _options = options ?? new CocoDetectionDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("coco-detection");
        _imageSize = _options.ImageSize;
        _maxDetections = _options.MaxDetections;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        bool isVal = _options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation;
        string splitName = isVal ? "val2017" : "train2017";
        string imageDir = Path.Combine(_dataPath, splitName);
        string annotationFile = Path.Combine(_dataPath, "annotations", $"instances_{splitName}.json");

        if (!Directory.Exists(imageDir) && _options.AutoDownload)
        {
            // Download annotations first
            if (!File.Exists(annotationFile))
            {
                await DatasetDownloader.DownloadAndExtractZipAsync(AnnotationsUrl, _dataPath, cancellationToken);
            }

            // Download images
            string imagesUrl = isVal ? ValImagesUrl : TrainImagesUrl;
            await DatasetDownloader.DownloadAndExtractZipAsync(imagesUrl, _dataPath, cancellationToken);
        }

        if (!File.Exists(annotationFile))
        {
            throw new FileNotFoundException(
                $"COCO annotation file not found at {annotationFile}. Enable AutoDownload or provide data locally.");
        }

        // Parse COCO JSON annotations
        using var stream = File.OpenRead(annotationFile);
        var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        var root = doc.RootElement;

        // Build image ID -> (file_name, width, height) map
        var imageMap = new Dictionary<long, (string FileName, int Width, int Height)>();
        if (root.TryGetProperty("images", out var imagesElem))
        {
            foreach (var img in imagesElem.EnumerateArray())
            {
                long id = img.GetProperty("id").GetInt64();
                string fileName = img.GetProperty("file_name").GetString() ?? string.Empty;
                int width = img.GetProperty("width").GetInt32();
                int height = img.GetProperty("height").GetInt32();
                imageMap[id] = (fileName, width, height);
            }
        }

        // Build COCO category ID -> contiguous index map (COCO IDs are not contiguous)
        var categoryMap = new Dictionary<long, int>();
        if (root.TryGetProperty("categories", out var catsElem))
        {
            int idx = 0;
            foreach (var cat in catsElem.EnumerateArray())
            {
                long catId = cat.GetProperty("id").GetInt64();
                categoryMap[catId] = idx++;
            }
        }

        // Group annotations by image
        var annotationsByImage = new Dictionary<long, List<(int ClassIdx, double X, double Y, double W, double H)>>();
        if (root.TryGetProperty("annotations", out var annotationsElem))
        {
            foreach (var ann in annotationsElem.EnumerateArray())
            {
                long imageId = ann.GetProperty("image_id").GetInt64();
                long categoryId = ann.GetProperty("category_id").GetInt64();

                if (!categoryMap.TryGetValue(categoryId, out var classIdx)) continue;
                if (!imageMap.TryGetValue(imageId, out var imgInfo)) continue;

                // bbox format is [x, y, width, height] in absolute pixels
                var bbox = ann.GetProperty("bbox");
                double bx = bbox[0].GetDouble() / imgInfo.Width;
                double by = bbox[1].GetDouble() / imgInfo.Height;
                double bw = bbox[2].GetDouble() / imgInfo.Width;
                double bh = bbox[3].GetDouble() / imgInfo.Height;

                if (!annotationsByImage.TryGetValue(imageId, out var list))
                {
                    list = new List<(int, double, double, double, double)>();
                    annotationsByImage[imageId] = list;
                }

                list.Add((classIdx, bx, by, bw, bh));
            }
        }

        // Only include images that have annotations
        var imageIds = annotationsByImage.Keys.ToList();
        int totalSamples = imageIds.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * _maxDetections * 5];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            long imageId = imageIds[i];
            var imgInfo = imageMap[imageId];
            string imgPath = Path.Combine(imageDir, imgInfo.FileName);

            // Load image
            if (File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
                int featureOffset = i * pixelsPerImage;
                int copyLen = Math.Min(pixels.Length, pixelsPerImage);
                Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            }

            // Load detections (zero-padded to MaxDetections)
            var detections = annotationsByImage[imageId];
            int labelOffset = i * _maxDetections * 5;
            int detCount = Math.Min(detections.Count, _maxDetections);
            for (int d = 0; d < detCount; d++)
            {
                var (classIdx, bx, by, bw, bh) = detections[d];
                int dOffset = labelOffset + d * 5;
                labelsData[dOffset + 0] = NumOps.FromDouble(classIdx);
                labelsData[dOffset + 1] = NumOps.FromDouble(bx);
                labelsData[dOffset + 2] = NumOps.FromDouble(by);
                labelsData[dOffset + 3] = NumOps.FromDouble(bw);
                labelsData[dOffset + 4] = NumOps.FromDouble(bh);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, _maxDetections, 5 });
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
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
