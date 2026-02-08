using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the CIFAR-10 image classification dataset (50k train / 10k test, 32x32 RGB, 10 classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// CIFAR-10 contains 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat,
/// deer, dog, frog, horse, ship, truck. The dataset is stored in a binary format where each sample
/// is 1 byte label + 3072 bytes of pixel data (32x32x3 in CHW order: 1024 red, 1024 green, 1024 blue).
/// </para>
/// </remarks>
public class Cifar10DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

    private static readonly string[] ClassLabels = new[]
    {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    private readonly Cifar10DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "CIFAR-10";
    /// <inheritdoc/>
    public override string Description => "CIFAR-10 image classification dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => 32 * 32 * 3;
    /// <inheritdoc/>
    public override int OutputDimension => 10;
    /// <summary>Gets the class names.</summary>
    public IReadOnlyList<string> ClassNames => ClassLabels;

    /// <summary>Creates a new CIFAR-10 data loader.</summary>
    public Cifar10DataLoader(Cifar10DataLoaderOptions? options = null)
    {
        _options = options ?? new Cifar10DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("cifar-10");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Try to find the binary data files
        string dataDir = FindDataDirectory(_dataPath);

        if (dataDir.Length == 0 && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            dataDir = FindDataDirectory(_dataPath);
        }

        if (dataDir.Length == 0)
        {
            throw new FileNotFoundException($"CIFAR-10 data not found at {_dataPath}. Enable AutoDownload or provide data locally.");
        }

        // Load binary batch files
        var allData = new List<byte[]>();

        if (_options.Split == DatasetSplit.Test)
        {
            LoadBatchFile(dataDir, "test_batch.bin", allData);
        }
        else
        {
            for (int i = 1; i <= 5; i++)
            {
                LoadBatchFile(dataDir, $"data_batch_{i}.bin", allData);
            }
        }

        int totalSamples = allData.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;

        // Each CIFAR-10 sample: 1 byte label + 3072 bytes pixels (CHW: 1024R + 1024G + 1024B)
        const int pixelsPerImage = 32 * 32 * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * 10];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            byte[] sample = allData[i];
            int label = sample[0];

            // Convert CHW to HWC format for consistency
            int featureOffset = i * pixelsPerImage;
            for (int h = 0; h < 32; h++)
            {
                for (int w = 0; w < 32; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        int srcIdx = 1 + c * 1024 + h * 32 + w; // CHW in source
                        int dstIdx = featureOffset + (h * 32 + w) * 3 + c; // HWC in dest
                        double value = sample[srcIdx];
                        if (_options.Normalize) value /= 255.0;
                        featuresData[dstIdx] = NumOps.FromDouble(value);
                    }
                }
            }

            if (label >= 0 && label < 10)
                labelsData[i * 10 + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, 32, 32, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 10 });
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

    private static void LoadBatchFile(string dataDir, string fileName, List<byte[]> samples)
    {
        string filePath = Path.Combine(dataDir, fileName);
        if (!File.Exists(filePath)) return;

        byte[] data = File.ReadAllBytes(filePath);
        const int bytesPerSample = 1 + 3072; // 1 label + 3072 pixels

        int offset = 0;
        while (offset + bytesPerSample <= data.Length)
        {
            byte[] sample = new byte[bytesPerSample];
            Array.Copy(data, offset, sample, 0, bytesPerSample);
            samples.Add(sample);
            offset += bytesPerSample;
        }
    }

    private static string FindDataDirectory(string rootPath)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        // Check common locations
        string[] candidates = new[]
        {
            rootPath,
            Path.Combine(rootPath, "cifar-10-batches-bin"),
            Path.Combine(rootPath, "cifar-10-binary")
        };

        foreach (string candidate in candidates)
        {
            if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, "data_batch_1.bin")))
            {
                return candidate;
            }
        }

        // Search recursively
        var dirs = Directory.GetDirectories(rootPath, "cifar-10-batches-bin", SearchOption.AllDirectories);
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
