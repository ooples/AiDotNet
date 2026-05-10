using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the STL-10 image classification dataset (Coates et al. 2011) at native 96×96 resolution.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects the canonical Stanford binary release:
/// <code>
/// {DataPath}/stl10_binary/
///   train_X.bin       (5000 × 96×96×3 bytes, channel-major)
///   train_y.bin       (5000 bytes, labels 1..10)
///   test_X.bin        (8000 ×  96×96×3 bytes)
///   test_y.bin        (8000 bytes, labels 1..10)
///   unlabeled_X.bin   (100,000 × 96×96×3 bytes)
///   class_names.txt
/// </code>
/// Auto-downloads the canonical tarball. Pixel data is stored channel-major
/// (R·R·…G·G·…B·B·…) and column-major within a channel — this loader
/// transposes to standard row-major HWC during decoding.
/// </para>
/// </remarks>
public class Stl10DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int ImageSize = 96;
    private const int NumClasses = 10;
    private const int PixelsPerImage = ImageSize * ImageSize * 3;

    private static readonly string DownloadUrl =
        "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz";

    private readonly Stl10DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "STL-10";
    public override string Description => "STL-10 96×96 image classification + unlabeled SSL split";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => PixelsPerImage;
    public override int OutputDimension => NumClasses;

    public Stl10DataLoader(Stl10DataLoaderOptions? options = null)
    {
        _options = options ?? new Stl10DataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("stl10");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"STL-10 not found at {_dataPath}.");

        string xFile, yFile;
        if (_options.UseUnlabeled)
        {
            xFile = Path.Combine(root, "unlabeled_X.bin");
            yFile = string.Empty;
        }
        else if (_options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation)
        {
            xFile = Path.Combine(root, "test_X.bin");
            yFile = Path.Combine(root, "test_y.bin");
        }
        else
        {
            xFile = Path.Combine(root, "train_X.bin");
            yFile = Path.Combine(root, "train_y.bin");
        }

        if (!File.Exists(xFile))
            throw new FileNotFoundException($"STL-10 image binary not found: {xFile}");

        byte[] xBytes = File.ReadAllBytes(xFile);
        int totalImagesInFile = xBytes.Length / PixelsPerImage;
        int totalSamples = totalImagesInFile;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        // For labeled splits the y_file MUST exist; we must not silently fall back
        // to all-zero labels which would corrupt training metrics.
        byte[]? labelBytes = null;
        if (!_options.UseUnlabeled)
        {
            if (!File.Exists(yFile))
                throw new FileNotFoundException(
                    $"STL-10 label binary not found: {yFile}. " +
                    "Re-extract stl10_binary.tar.gz or check archive integrity.");
            labelBytes = File.ReadAllBytes(yFile);
            if (labelBytes.Length < totalImagesInFile)
                throw new InvalidDataException(
                    $"STL-10 label binary {yFile} is truncated: got {labelBytes.Length} bytes, expected {totalImagesInFile}.");
        }

        var featuresData = new T[totalSamples * PixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        // STL-10 storage: per-image, the bytes are organized as 3 channels of
        // 96×96 each, stored column-major within the channel. We transpose to
        // (h, w, c) row-major HWC for the output tensor.
        const int Channel = ImageSize * ImageSize;
        bool normalize = _options.Normalize;
        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int srcBase = i * PixelsPerImage;
            int dstBase = i * PixelsPerImage;
            for (int c = 0; c < 3; c++)
            {
                int chSrc = srcBase + c * Channel;
                for (int col = 0; col < ImageSize; col++)
                    for (int row = 0; row < ImageSize; row++)
                    {
                        int srcIdx = chSrc + col * ImageSize + row;
                        int dstIdx = dstBase + (row * ImageSize + col) * 3 + c;
                        byte b = xBytes[srcIdx];
                        featuresData[dstIdx] = normalize
                            ? NumOps.FromDouble(b / 255.0)
                            : NumOps.FromDouble(b);
                    }
            }
            if (labelBytes is not null)
            {
                int label = labelBytes[i] - 1; // STL-10 labels are 1..10
                if (label < 0 || label >= NumClasses)
                    throw new InvalidDataException(
                        $"STL-10 {yFile} contains out-of-range label {labelBytes[i]} at index {i}. Expected 1..{NumClasses}.");
                labelsData[i * NumClasses + label] = NumOps.One;
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, ImageSize, ImageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
        InitializeIndices(totalSamples);
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
        var trainIndices = shuffled.Take(trainSize).ToArray();
        var valIndices = shuffled.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffled.Skip(trainSize + valSize).ToArray();
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, trainIndices), ExtractTensorBatchLocal(labels, trainIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, valIndices), ExtractTensorBatchLocal(labels, valIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, testIndices), ExtractTensorBatchLocal(labels, testIndices))
        );
    }

    private string ResolveDataDir()
    {
        string sub = Path.Combine(_dataPath, "stl10_binary");
        return Directory.Exists(sub) ? sub : _dataPath;
    }

    private static Tensor<T> ExtractTensorBatchLocal(Tensor<T> source, int[] indices)
    {
        var result = AiDotNet.Helpers.TensorCopyHelper.CreateEmptyBatchLike(source, indices.Length);
        for (int i = 0; i < indices.Length; i++)
            AiDotNet.Helpers.TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
