using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Open Images V7 object detection dataset (~9M images, 600 categories).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Open Images expects:
/// <code>
/// {DataPath}/
///   train/
///     000002b66c9c498e.jpg
///     ...
///   validation/
///     ...
///   test/
///     ...
///   sub-annotations/           (or annotations/)
///     oidv7-train-annotations-bbox.csv
///     validation-annotations-bbox.csv
///     test-annotations-bbox.csv
///   metadata/
///     oidv7-class-descriptions-boxable.csv
/// </code>
/// The CSV annotation format contains: ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax.
/// Bounding box coordinates are already normalized to [0, 1].
/// Labels are stored as Tensor[N, MaxDetections, 5] (class_index, xmin, ymin, width, height).
/// </para>
/// </remarks>
public class OpenImagesDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 600;

    private readonly OpenImagesDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;
    private int _maxDetections;

    /// <inheritdoc/>
    public override string Name => "Open-Images-V7";
    /// <inheritdoc/>
    public override string Description => "Open Images V7 object detection dataset (600 categories)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => _maxDetections * 5;

    /// <summary>Creates a new Open Images data loader.</summary>
    public OpenImagesDataLoader(OpenImagesDataLoaderOptions? options = null)
    {
        _options = options ?? new OpenImagesDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("open-images-v7");
        _imageSize = _options.ImageSize;
        _maxDetections = _options.MaxDetections;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        bool isVal = _options.Split == Geometry.DatasetSplit.Validation;
        bool isTest = _options.Split == Geometry.DatasetSplit.Test;
        string splitName = isTest ? "test" : isVal ? "validation" : "train";

        string imageDir = Path.Combine(_dataPath, splitName);
        if (!Directory.Exists(imageDir))
        {
            throw new DirectoryNotFoundException(
                $"Open Images data not found at {imageDir}. " +
                "Download the dataset from https://storage.googleapis.com/openimages/web/index.html.");
        }

        // Load class descriptions to build label name -> index mapping
        var classMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        string classFile = FindFile(_dataPath, "*class-descriptions*.csv");
        if (classFile.Length > 0)
        {
            var classLines = await FilePolyfill.ReadAllLinesAsync(classFile, cancellationToken);
            int idx = 0;
            foreach (var line in classLines)
            {
                var parts = line.Split(new[] { ',' }, 2);
                if (parts.Length >= 1)
                {
                    classMap[parts[0].Trim()] = idx++;
                }
            }
        }

        // Load bbox annotations CSV
        string annotationFile = FindFile(_dataPath, $"*{splitName}*annotations*bbox*.csv");
        if (annotationFile.Length == 0)
        {
            throw new FileNotFoundException(
                $"Open Images annotation file for {splitName} not found in {_dataPath}.");
        }

        // Parse CSV annotations grouped by image
        var annotationsByImage = new Dictionary<string, List<(int ClassIdx, double XMin, double YMin, double W, double H)>>();
        var lines = await FilePolyfill.ReadAllLinesAsync(annotationFile, cancellationToken);

        // Skip header
        for (int lineIdx = 1; lineIdx < lines.Length; lineIdx++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var parts = lines[lineIdx].Split(',');
            if (parts.Length < 8) continue;

            string imageId = parts[0].Trim();
            string labelName = parts[2].Trim();

            if (!classMap.TryGetValue(labelName, out int classIdx)) continue;

            if (!double.TryParse(parts[4], out double xmin) ||
                !double.TryParse(parts[5], out double xmax) ||
                !double.TryParse(parts[6], out double ymin) ||
                !double.TryParse(parts[7], out double ymax))
                continue;

            double w = xmax - xmin;
            double h = ymax - ymin;

            if (!annotationsByImage.TryGetValue(imageId, out var list))
            {
                list = new List<(int, double, double, double, double)>();
                annotationsByImage[imageId] = list;
            }

            list.Add((classIdx, xmin, ymin, w, h));
        }

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

            string imageId = imageIds[i];
            string imgPath = Path.Combine(imageDir, $"{imageId}.jpg");

            if (File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
                int featureOffset = i * pixelsPerImage;
                int copyLen = Math.Min(pixels.Length, pixelsPerImage);
                Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            }

            var detections = annotationsByImage[imageId];
            int labelOffset = i * _maxDetections * 5;
            int detCount = Math.Min(detections.Count, _maxDetections);
            for (int d = 0; d < detCount; d++)
            {
                var (classIdx, xmin, ymin, w, h) = detections[d];
                int dOffset = labelOffset + d * 5;
                labelsData[dOffset + 0] = NumOps.FromDouble(classIdx);
                labelsData[dOffset + 1] = NumOps.FromDouble(xmin);
                labelsData[dOffset + 2] = NumOps.FromDouble(ymin);
                labelsData[dOffset + 3] = NumOps.FromDouble(w);
                labelsData[dOffset + 4] = NumOps.FromDouble(h);
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

    private static string FindFile(string rootPath, string pattern)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        // Search in root and common subdirectories
        string[] searchDirs = { rootPath, Path.Combine(rootPath, "annotations"),
            Path.Combine(rootPath, "sub-annotations"), Path.Combine(rootPath, "metadata") };

        foreach (var dir in searchDirs)
        {
            if (!Directory.Exists(dir)) continue;
            var files = Directory.GetFiles(dir, pattern, SearchOption.TopDirectoryOnly);
            if (files.Length > 0) return files[0];
        }

        return string.Empty;
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
