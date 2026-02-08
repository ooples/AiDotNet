using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the CIFAR-100 image classification dataset (50k train / 10k test, 32x32 RGB, 100 fine / 20 coarse classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// CIFAR-100 has 100 fine-grained classes grouped into 20 superclasses. Each sample is stored as
/// 2 bytes (coarse label, fine label) + 3072 bytes of pixel data.
/// </para>
/// </remarks>
public class Cifar100DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";

    private readonly Cifar100DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _numClasses;

    /// <inheritdoc/>
    public override string Name => "CIFAR-100";
    /// <inheritdoc/>
    public override string Description => "CIFAR-100 image classification dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => 32 * 32 * 3;
    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Creates a new CIFAR-100 data loader.</summary>
    public Cifar100DataLoader(Cifar100DataLoaderOptions? options = null)
    {
        _options = options ?? new Cifar100DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("cifar-100");
        _numClasses = _options.UseFineLabels ? 100 : 20;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string dataDir = FindDataDirectory(_dataPath);

        if (dataDir.Length == 0 && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            dataDir = FindDataDirectory(_dataPath);
        }

        if (dataDir.Length == 0)
            throw new FileNotFoundException($"CIFAR-100 data not found at {_dataPath}.");

        string dataFile = _options.Split == DatasetSplit.Test
            ? Path.Combine(dataDir, "test.bin")
            : Path.Combine(dataDir, "train.bin");

        if (!File.Exists(dataFile))
            throw new FileNotFoundException($"CIFAR-100 data file not found: {dataFile}");

        byte[] data = File.ReadAllBytes(dataFile);

        // Each sample: 1 byte coarse label + 1 byte fine label + 3072 pixels
        const int bytesPerSample = 2 + 3072;
        if (data.Length % bytesPerSample != 0)
            throw new InvalidDataException(
                $"CIFAR-100 file size ({data.Length} bytes) is not a multiple of sample size ({bytesPerSample} bytes). File may be corrupt.");
        int totalSamples = data.Length / bytesPerSample;

        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;

        var featuresData = new T[totalSamples * 32 * 32 * 3];
        var labelsData = new T[totalSamples * _numClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int sampleOffset = i * bytesPerSample;
            int label = _options.UseFineLabels ? data[sampleOffset + 1] : data[sampleOffset];

            int featureOffset = i * 32 * 32 * 3;
            for (int h = 0; h < 32; h++)
            {
                for (int w = 0; w < 32; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        int srcIdx = sampleOffset + 2 + c * 1024 + h * 32 + w;
                        int dstIdx = featureOffset + (h * 32 + w) * 3 + c;
                        double value = data[srcIdx];
                        if (_options.Normalize) value /= 255.0;
                        featuresData[dstIdx] = NumOps.FromDouble(value);
                    }
                }
            }

            if (label >= 0 && label < _numClasses)
                labelsData[i * _numClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, 32, 32, 3 });
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
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded. Call LoadAsync() first.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded. Call LoadAsync() first.");

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
        string[] candidates = new[] { rootPath, Path.Combine(rootPath, "cifar-100-binary") };
        foreach (string candidate in candidates)
        {
            if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, "train.bin")))
                return candidate;
        }
        var dirs = Directory.GetDirectories(rootPath, "cifar-100-binary", SearchOption.AllDirectories);
        return dirs.Length > 0 ? dirs[0] : string.Empty;
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
