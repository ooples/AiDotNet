using AiDotNet.Data.Loaders;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

namespace AiDotNet.Data.Vision;

/// <summary>
/// Loads images from a directory structure where each subdirectory is a class label.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Mirrors PyTorch's <c>torchvision.datasets.ImageFolder</c>. Expects a directory structure:
/// <code>
/// root/
///   class1/
///     image1.png
///     image2.jpg
///   class2/
///     image3.png
///     image4.jpg
/// </code>
/// </para>
/// <para><b>For Beginners:</b> Organize your images into folders named after the classes,
/// and this loader handles the rest:
/// <code>
/// var loader = new ImageFolderDataset&lt;float&gt;(new ImageFolderDatasetOptions
/// {
///     RootDirectory = "data/images",
///     ImageWidth = 28,
///     ImageHeight = 28,
///     Channels = 1
/// });
/// await loader.LoadAsync();
/// </code>
/// </para>
/// </remarks>
public class ImageFolderDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly ImageFolderDatasetOptions _options;
    private int _sampleCount;
    private int _featureCount;
    private int _numClasses;
    private List<string> _classNames = new List<string>();

    /// <inheritdoc/>
    public override string Name => "ImageFolder";

    /// <inheritdoc/>
    public override string Description => $"Image folder dataset from {_options.RootDirectory}";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>
    /// Gets the class names discovered from directory names.
    /// </summary>
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>
    /// Gets the number of classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Creates a new ImageFolderDataset with the specified options.
    /// </summary>
    public ImageFolderDataset(ImageFolderDatasetOptions options)
    {
        Guard.NotNull(options);
        _options = options;

        if (string.IsNullOrWhiteSpace(options.RootDirectory))
        {
            throw new ArgumentException("Root directory cannot be empty.", nameof(options));
        }
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string rootDir = _options.RootDirectory;
        if (!Directory.Exists(rootDir))
        {
            throw new DirectoryNotFoundException($"Image folder root directory not found: {rootDir}");
        }

        // Discover class directories
        var classDirs = Directory.GetDirectories(rootDir)
            .OrderBy(d => Path.GetFileName(d), StringComparer.OrdinalIgnoreCase)
            .ToList();

        _classNames = classDirs
            .Select(d => Path.GetFileName(d) ?? string.Empty)
            .Where(n => n.Length > 0)
            .ToList();
        _numClasses = _classNames.Count;

        if (_numClasses == 0)
        {
            throw new InvalidOperationException($"No class subdirectories found in {rootDir}.");
        }

        // Collect all image files with their class index
        var samples = new List<(string FilePath, int ClassIndex)>();
        var extensions = new HashSet<string>(
            _options.Extensions.Select(e => e.StartsWith(".", StringComparison.Ordinal) ? e : "." + e),
            StringComparer.OrdinalIgnoreCase);

        for (int classIdx = 0; classIdx < classDirs.Count; classIdx++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var searchOption = _options.Recursive
                ? SearchOption.AllDirectories
                : SearchOption.TopDirectoryOnly;

            var files = Directory.GetFiles(classDirs[classIdx], "*.*", searchOption)
                .Where(f => extensions.Contains(Path.GetExtension(f)))
                .ToList();

            foreach (var file in files)
            {
                samples.Add((file, classIdx));
            }
        }

        // Apply max samples limit
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < samples.Count)
        {
            var random = _options.RandomSeed.HasValue
                ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
                : RandomHelper.CreateSecureRandom();

            samples = samples.OrderBy(_ => random.Next()).Take(_options.MaxSamples.Value).ToList();
        }

        _sampleCount = samples.Count;
        int width = _options.ImageWidth;
        int height = _options.ImageHeight;
        int channels = _options.Channels;
        _featureCount = width * height * channels;

        if (_sampleCount == 0)
        {
            throw new InvalidOperationException("No image files found matching the specified extensions.");
        }

        // Build tensors - features: [N, H, W, C], labels: [N, numClasses] one-hot
        var featuresData = new T[_sampleCount * height * width * channels];
        var labelsData = new T[_sampleCount * _numClasses];

        for (int i = 0; i < _sampleCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (filePath, classIndex) = samples[i];

            // Read raw image bytes and convert to pixel values
            var pixelData = LoadImagePixels(filePath, width, height, channels);
            int featureOffset = i * height * width * channels;
            Array.Copy(pixelData, 0, featuresData, featureOffset, pixelData.Length);

            // One-hot encode the label
            int labelOffset = i * _numClasses;
            labelsData[labelOffset + classIndex] = NumOps.One;
        }

        var features = new Tensor<T>(featuresData, new[] { _sampleCount, height, width, channels });
        var labels = new Tensor<T>(labelsData, new[] { _sampleCount, _numClasses });

        LoadedFeatures = features;
        LoadedLabels = labels;
        InitializeIndices(_sampleCount);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
        _classNames = new List<string>();
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

        var batchFeatures = ExtractTensorBatch(features, indices);
        var batchLabels = ExtractTensorBatch(labels, indices);

        return (batchFeatures, batchLabels);
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        var trainFeatures = ExtractTensorBatch(LoadedFeatures!, shuffled.Take(trainSize).ToArray());
        var trainLabels = ExtractTensorBatch(LoadedLabels!, shuffled.Take(trainSize).ToArray());
        var valFeatures = ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize).Take(valSize).ToArray());
        var valLabels = ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize).Take(valSize).ToArray());
        var testFeatures = ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize + valSize).ToArray());
        var testLabels = ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize + valSize).ToArray());

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(trainFeatures, trainLabels),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(valFeatures, valLabels),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(testFeatures, testLabels)
        );
    }

    private T[] LoadImagePixels(string filePath, int targetWidth, int targetHeight, int targetChannels)
    {
        // Load image using ImageHelper which properly decodes BMP, PPM, PGM formats
        // ImageHelper returns [1, C, H, W] in CHW format
        Tensor<T> imageTensor = ImageHelper<T>.LoadImage(filePath, _options.NormalizePixels);

        int srcChannels = imageTensor.Shape[1];
        int srcHeight = imageTensor.Shape[2];
        int srcWidth = imageTensor.Shape[3];
        var srcSpan = imageTensor.AsSpan();

        int totalPixels = targetHeight * targetWidth * targetChannels;
        var pixels = new T[totalPixels];

        // Bilinear resize from source to target dimensions, converting CHW -> HWC
        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                // Map target coordinates to source coordinates
                double srcY = (double)y * (srcHeight - 1) / Math.Max(1, targetHeight - 1);
                double srcX = (double)x * (srcWidth - 1) / Math.Max(1, targetWidth - 1);

                int y0 = (int)Math.Floor(srcY);
                int y1 = Math.Min(y0 + 1, srcHeight - 1);
                int x0 = (int)Math.Floor(srcX);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);

                double dy = srcY - y0;
                double dx = srcX - x0;

                for (int c = 0; c < targetChannels; c++)
                {
                    double value;
                    if (c < srcChannels)
                    {
                        // Bilinear interpolation from CHW source
                        int chOffset = c * srcHeight * srcWidth;
                        double v00 = NumOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x0]);
                        double v01 = NumOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x1]);
                        double v10 = NumOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x0]);
                        double v11 = NumOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x1]);

                        value = v00 * (1 - dx) * (1 - dy) +
                                v01 * dx * (1 - dy) +
                                v10 * (1 - dx) * dy +
                                v11 * dx * dy;
                    }
                    else if (srcChannels == 1)
                    {
                        // Grayscale to RGB: replicate the single channel
                        int chOffset = 0;
                        double v00 = NumOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x0]);
                        double v01 = NumOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x1]);
                        double v10 = NumOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x0]);
                        double v11 = NumOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x1]);

                        value = v00 * (1 - dx) * (1 - dy) +
                                v01 * dx * (1 - dy) +
                                v10 * (1 - dx) * dy +
                                v11 * dx * dy;
                    }
                    else
                    {
                        value = 0;
                    }

                    // Store in HWC format: [y, x, c]
                    pixels[(y * targetWidth + x) * targetChannels + c] = NumOps.FromDouble(value);
                }
            }
        }

        return pixels;
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
