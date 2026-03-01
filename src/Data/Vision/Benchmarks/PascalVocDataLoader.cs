using System.Xml.Linq;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Pascal VOC object detection dataset (20 categories, XML annotations).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Pascal VOC expects a standard directory structure:
/// <code>
/// {DataPath}/VOCdevkit/VOC{Year}/
///   JPEGImages/
///     000001.jpg
///     ...
///   Annotations/
///     000001.xml
///     ...
///   ImageSets/Main/
///     train.txt
///     val.txt
///     trainval.txt
/// </code>
/// Each XML annotation file contains bounding boxes with class names.
/// Labels are stored as Tensor[N, MaxDetections, 5] (class_id, x, y, w, h) normalized to [0, 1].
/// </para>
/// </remarks>
public class PascalVocDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string[] ClassLabels =
    {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };

    private static readonly Dictionary<string, int> ClassToIndex;

    static PascalVocDataLoader()
    {
        ClassToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < ClassLabels.Length; i++)
            ClassToIndex[ClassLabels[i]] = i;
    }

    private const int NumClasses = 20;

    private readonly PascalVocDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;
    private int _maxDetections;

    /// <inheritdoc/>
    public override string Name => $"Pascal-VOC-{_options.Year}";
    /// <inheritdoc/>
    public override string Description => $"Pascal VOC {_options.Year} object detection dataset (20 categories)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => _maxDetections * 5;
    /// <summary>Gets the class names.</summary>
    public IReadOnlyList<string> ClassNames => ClassLabels;

    /// <summary>Creates a new Pascal VOC data loader.</summary>
    public PascalVocDataLoader(PascalVocDataLoaderOptions? options = null)
    {
        _options = options ?? new PascalVocDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath($"pascal-voc-{_options.Year}");
        _imageSize = _options.ImageSize;
        _maxDetections = _options.MaxDetections;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Find VOCdevkit directory
        string vocDir = FindVocDirectory(_dataPath, _options.Year);

        if (vocDir.Length == 0 && _options.AutoDownload)
        {
            string url = _options.Year == "2007"
                ? "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
                : "https://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar";
            await DatasetDownloader.DownloadAndExtractTarGzAsync(url, _dataPath, cancellationToken);
            vocDir = FindVocDirectory(_dataPath, _options.Year);
        }

        if (vocDir.Length == 0)
        {
            throw new DirectoryNotFoundException(
                $"Pascal VOC data not found at {_dataPath}. Enable AutoDownload or provide data locally.");
        }

        string imageDir = Path.Combine(vocDir, "JPEGImages");
        string annotationDir = Path.Combine(vocDir, "Annotations");
        string imageSetsDir = Path.Combine(vocDir, "ImageSets", "Main");

        // Load image IDs from split file
        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "val",
            _ => "trainval"
        };
        string splitFile = Path.Combine(imageSetsDir, $"{splitName}.txt");

        List<string> imageIds;
        if (File.Exists(splitFile))
        {
            imageIds = (await FilePolyfill.ReadAllLinesAsync(splitFile, cancellationToken))
                .Select(l => l.Trim())
                .Where(l => l.Length > 0)
                .ToList();
        }
        else
        {
            // Fall back to all annotation files
            imageIds = Directory.GetFiles(annotationDir, "*.xml")
                .Select(f => Path.GetFileNameWithoutExtension(f))
                .ToList();
        }

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
            string annPath = Path.Combine(annotationDir, $"{imageId}.xml");

            // Load image
            if (File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
                int featureOffset = i * pixelsPerImage;
                int copyLen = Math.Min(pixels.Length, pixelsPerImage);
                Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            }

            // Parse XML annotation
            if (File.Exists(annPath))
            {
                var doc = XDocument.Load(annPath);
                var sizeElem = doc.Root?.Element("size");
                double imgWidth = double.Parse(sizeElem?.Element("width")?.Value ?? "1");
                double imgHeight = double.Parse(sizeElem?.Element("height")?.Value ?? "1");

                var objects = doc.Root?.Elements("object").ToList() ?? new List<XElement>();
                int labelOffset = i * _maxDetections * 5;
                int detCount = Math.Min(objects.Count, _maxDetections);

                for (int d = 0; d < detCount; d++)
                {
                    var obj = objects[d];
                    string className = obj.Element("name")?.Value ?? string.Empty;
                    var bndbox = obj.Element("bndbox");

                    if (bndbox != null && ClassToIndex.TryGetValue(className, out int classIdx))
                    {
                        double xmin = double.Parse(bndbox.Element("xmin")?.Value ?? "0") / imgWidth;
                        double ymin = double.Parse(bndbox.Element("ymin")?.Value ?? "0") / imgHeight;
                        double xmax = double.Parse(bndbox.Element("xmax")?.Value ?? "0") / imgWidth;
                        double ymax = double.Parse(bndbox.Element("ymax")?.Value ?? "0") / imgHeight;

                        int dOffset = labelOffset + d * 5;
                        labelsData[dOffset + 0] = NumOps.FromDouble(classIdx);
                        labelsData[dOffset + 1] = NumOps.FromDouble(xmin);
                        labelsData[dOffset + 2] = NumOps.FromDouble(ymin);
                        labelsData[dOffset + 3] = NumOps.FromDouble(xmax - xmin); // width
                        labelsData[dOffset + 4] = NumOps.FromDouble(ymax - ymin); // height
                    }
                }
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

    private static string FindVocDirectory(string rootPath, string year)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        string[] candidates =
        {
            Path.Combine(rootPath, $"VOCdevkit", $"VOC{year}"),
            Path.Combine(rootPath, $"VOC{year}"),
            rootPath
        };

        foreach (string candidate in candidates)
        {
            if (Directory.Exists(candidate) &&
                Directory.Exists(Path.Combine(candidate, "JPEGImages")) &&
                Directory.Exists(Path.Combine(candidate, "Annotations")))
            {
                return candidate;
            }
        }

        var dirs = Directory.GetDirectories(rootPath, $"VOC{year}", SearchOption.AllDirectories);
        if (dirs.Length > 0) return dirs[0];

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
