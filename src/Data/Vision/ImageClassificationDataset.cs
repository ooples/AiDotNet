using AiDotNet.Data.Loaders;
using AiDotNet.Data.Transforms;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision;

/// <summary>
/// An in-memory image classification dataset with an optional composable transform pipeline.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Holds image tensors and class labels in memory with optional transforms applied on access.
/// Use this when you already have image data as tensors and want to attach a transform pipeline.
/// For loading from disk, use <see cref="ImageFolderDataset{T}"/> instead.
/// </para>
/// <para><b>For Beginners:</b> If you have images already loaded as arrays/tensors and want
/// to apply normalization or other transforms:
/// <code>
/// var images = new T[][] { image1Pixels, image2Pixels };
/// var labels = new int[] { 0, 1 };
/// var transform = new NormalizeTransform&lt;float&gt;(mean, std);
/// var dataset = new ImageClassificationDataset&lt;float&gt;(images, labels, 28, 28, 1, transform);
/// await dataset.LoadAsync();
/// </code>
/// </para>
/// </remarks>
public class ImageClassificationDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly T[][] _rawImages;
    private readonly int[] _labels;
    private readonly int _imageHeight;
    private readonly int _imageWidth;
    private readonly int _channels;
    private readonly ITransform<T[], T[]>? _transform;
    private int _numClasses;

    /// <inheritdoc/>
    public override string Name => "ImageClassification";

    /// <inheritdoc/>
    public override string Description => $"In-memory image classification dataset ({_rawImages.Length} samples)";

    /// <inheritdoc/>
    public override int TotalCount => _rawImages.Length;

    /// <inheritdoc/>
    public override int FeatureCount => _imageHeight * _imageWidth * _channels;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>
    /// Gets the number of classes in the dataset.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets the image height.
    /// </summary>
    public int ImageHeight => _imageHeight;

    /// <summary>
    /// Gets the image width.
    /// </summary>
    public int ImageWidth => _imageWidth;

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Creates an in-memory image classification dataset.
    /// </summary>
    /// <param name="images">Flat pixel arrays for each image (length must be height * width * channels).</param>
    /// <param name="labels">Class label for each image (0-based).</param>
    /// <param name="imageHeight">Image height in pixels.</param>
    /// <param name="imageWidth">Image width in pixels.</param>
    /// <param name="channels">Number of channels (1 = grayscale, 3 = RGB).</param>
    /// <param name="transform">Optional transform pipeline applied to each image's pixel array before batching.</param>
    public ImageClassificationDataset(
        T[][] images,
        int[] labels,
        int imageHeight,
        int imageWidth,
        int channels = 3,
        ITransform<T[], T[]>? transform = null)
    {
        if (images is null)
        {
            throw new ArgumentNullException(nameof(images));
        }

        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (images.Length != labels.Length)
        {
            throw new ArgumentException(
                $"Images count ({images.Length}) must match labels count ({labels.Length}).",
                nameof(labels));
        }

        if (images.Length == 0)
        {
            throw new ArgumentException("Dataset cannot be empty.", nameof(images));
        }

        if (imageHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageHeight), "Image height must be positive.");
        }

        if (imageWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageWidth), "Image width must be positive.");
        }

        if (channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");
        }

        int expectedLen = imageHeight * imageWidth * channels;
        for (int i = 0; i < images.Length; i++)
        {
            if (images[i] is null)
            {
                throw new ArgumentNullException(nameof(images), $"Image at index {i} is null.");
            }

            if (images[i].Length != expectedLen)
            {
                throw new ArgumentException(
                    $"Image at index {i} has length {images[i].Length}, expected {expectedLen} " +
                    $"({imageHeight} x {imageWidth} x {channels}).",
                    nameof(images));
            }
        }

        _rawImages = images;
        _labels = (int[])labels.Clone();
        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _channels = channels;
        _transform = transform;
        _numClasses = labels.Length > 0 ? labels.Max() + 1 : 0;
    }

    /// <summary>
    /// Creates an in-memory image classification dataset from tensors.
    /// </summary>
    /// <param name="imageTensors">Individual image tensors, each of shape [H, W, C] or [C, H, W].</param>
    /// <param name="labels">Class label for each image (0-based).</param>
    /// <param name="transform">Optional transform pipeline applied to pixel data.</param>
    public ImageClassificationDataset(
        Tensor<T>[] imageTensors,
        int[] labels,
        ITransform<T[], T[]>? transform = null)
    {
        if (imageTensors is null)
        {
            throw new ArgumentNullException(nameof(imageTensors));
        }

        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (imageTensors.Length != labels.Length)
        {
            throw new ArgumentException(
                $"Image tensor count ({imageTensors.Length}) must match labels count ({labels.Length}).",
                nameof(labels));
        }

        if (imageTensors.Length == 0)
        {
            throw new ArgumentException("Dataset cannot be empty.", nameof(imageTensors));
        }

        // Infer dimensions from first tensor (assume all same shape)
        var firstShape = imageTensors[0].Shape;
        if (firstShape.Length < 2)
        {
            throw new ArgumentException("Image tensors must have at least 2 dimensions.", nameof(imageTensors));
        }

        if (firstShape.Length == 3)
        {
            _imageHeight = firstShape[0];
            _imageWidth = firstShape[1];
            _channels = firstShape[2];
        }
        else if (firstShape.Length == 2)
        {
            _imageHeight = firstShape[0];
            _imageWidth = firstShape[1];
            _channels = 1;
        }
        else
        {
            throw new ArgumentException(
                $"Image tensors must be 2D [H,W] or 3D [H,W,C], got {firstShape.Length}D.",
                nameof(imageTensors));
        }

        int expectedLen = _imageHeight * _imageWidth * _channels;

        // Convert tensors to flat arrays
        _rawImages = new T[imageTensors.Length][];
        for (int i = 0; i < imageTensors.Length; i++)
        {
            var span = imageTensors[i].Data.Span;
            if (span.Length != expectedLen)
            {
                throw new ArgumentException(
                    $"Image tensor at index {i} has {span.Length} elements, expected {expectedLen}.",
                    nameof(imageTensors));
            }

            _rawImages[i] = new T[expectedLen];
            span.CopyTo(_rawImages[i].AsSpan());
        }

        _labels = (int[])labels.Clone();
        _transform = transform;
        _numClasses = labels.Length > 0 ? labels.Max() + 1 : 0;
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        int n = _rawImages.Length;
        int pixelsPerImage = _imageHeight * _imageWidth * _channels;
        var featuresData = new T[n * pixelsPerImage];
        var labelsData = new T[n * _numClasses];

        for (int i = 0; i < n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Apply transform if set
            T[] pixels = _transform is not null
                ? _transform.Apply(_rawImages[i])
                : _rawImages[i];

            int featureOffset = i * pixelsPerImage;
            Array.Copy(pixels, 0, featuresData, featureOffset, pixelsPerImage);

            // One-hot encode label
            int labelOffset = i * _numClasses;
            if (_labels[i] >= 0 && _labels[i] < _numClasses)
            {
                labelsData[labelOffset + _labels[i]] = NumOps.One;
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { n, _imageHeight, _imageWidth, _channels });
        LoadedLabels = new Tensor<T>(labelsData, new[] { n, _numClasses });
        InitializeIndices(n);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
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
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        int n = TotalCount;
        var (trainSize, valSize, _) = ComputeSplitSizes(n, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize + valSize).ToArray()))
        );
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
