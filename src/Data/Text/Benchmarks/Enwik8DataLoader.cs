using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the enwik8 character-level Wikipedia language modeling benchmark
/// (first 100M bytes of an English Wikipedia XML dump).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects a single file <c>{DataPath}/enwik8</c>. Auto-download fetches the
/// canonical zip from mattmahoney.net. Operates byte-by-byte (no tokenization).
/// Standard split: first 90M chars → train, next 5M → val, last 5M → test.
/// Features are byte-id sequences Tensor[N, SequenceLength]; labels are
/// next-byte targets shifted by 1.
/// </para>
/// </remarks>
public class Enwik8DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly Enwik8DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    private const long TrainEnd = 90_000_000L;
    private const long ValEnd = 95_000_000L;
    private const long TotalSize = 100_000_000L;

    /// <inheritdoc/>
    public override string Name => "enwik8";
    /// <inheritdoc/>
    public override string Description => "enwik8 character-level Wikipedia LM benchmark (Hutter Prize)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.SequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => _options.SequenceLength;

    /// <summary>Creates a new enwik8 data loader.</summary>
    public Enwik8DataLoader(Enwik8DataLoaderOptions? options = null)
    {
        _options = options ?? new Enwik8DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("enwik8");
    }

    private static readonly string DownloadUrl = "https://mattmahoney.net/dc/enwik8.zip";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string filePath = Path.Combine(_dataPath, "enwik8");
        if (!File.Exists(filePath) && _options.AutoDownload)
        {
            try
            {
                await DatasetDownloader.DownloadAndExtractZipAsync(
                    DownloadUrl, _dataPath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException(
                    $"Failed to download enwik8 from {DownloadUrl}.", ex);
            }
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException(
                $"enwik8 not found at {filePath}. Enable AutoDownload or place the file manually.");
        }

        // Read the relevant byte range for the requested split.
        // PTB split bounds: 0..90M train, 90M..95M val, 95M..100M test.
        long startByte, endByte;
        switch (_options.Split)
        {
            case Geometry.DatasetSplit.Validation: startByte = TrainEnd; endByte = ValEnd; break;
            case Geometry.DatasetSplit.Test: startByte = ValEnd; endByte = TotalSize; break;
            default: startByte = 0; endByte = TrainEnd; break;
        }

        long fileSize = new FileInfo(filePath).Length;
        endByte = Math.Min(endByte, fileSize);
        if (startByte >= endByte)
            throw new InvalidDataException($"enwik8 file is shorter than expected ({fileSize} bytes).");

        int seqLen = _options.SequenceLength;
        long byteCount = endByte - startByte;
        int numSequences = (int)Math.Min(int.MaxValue, (byteCount - 1) / seqLen);
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < numSequences)
            numSequences = _options.MaxSamples.Value;

        _sampleCount = numSequences;
        var featuresData = new T[numSequences * seqLen];
        var labelsData = new T[numSequences * seqLen];

        // Stream the byte range. We read seqLen+1 bytes per sample (input + 1 for shifted target).
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        fs.Seek(startByte, SeekOrigin.Begin);

        // Buffer enough bytes to cover all sequences plus the trailing target byte.
        long bytesNeeded = (long)numSequences * seqLen + 1;
        var buffer = new byte[bytesNeeded];
        int totalRead = 0;
        while (totalRead < buffer.Length)
        {
            int read = await fs.ReadAsync(buffer, totalRead, buffer.Length - totalRead, cancellationToken);
            if (read == 0) break;
            totalRead += read;
        }

        int vocabCap = _options.VocabularySize;
        for (int i = 0; i < numSequences; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int featureOffset = i * seqLen;
            int byteOffset = i * seqLen;
            for (int j = 0; j < seqLen; j++)
            {
                int srcIdx = byteOffset + j;
                int b = srcIdx < totalRead ? buffer[srcIdx] : 0;
                int t = (srcIdx + 1) < totalRead ? buffer[srcIdx + 1] : 0;
                if (b >= vocabCap) b = vocabCap - 1;
                if (t >= vocabCap) t = vocabCap - 1;
                featuresData[featureOffset + j] = NumOps.FromDouble(b);
                labelsData[featureOffset + j] = NumOps.FromDouble(t);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { numSequences, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { numSequences, seqLen });
        InitializeIndices(numSequences);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (TextLoaderHelper.ExtractTensorBatch(features, indices), TextLoaderHelper.ExtractTensorBatch(labels, indices));
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
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, trainIndices), TextLoaderHelper.ExtractTensorBatch(labels, trainIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, valIndices), TextLoaderHelper.ExtractTensorBatch(labels, valIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, testIndices), TextLoaderHelper.ExtractTensorBatch(labels, testIndices))
        );
    }
}
