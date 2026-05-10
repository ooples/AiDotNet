using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using MatFileHandler;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Oxford Flowers-102 dataset (Nilsback &amp; Zisserman 2008) — 102 fine-grained flower species.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/
///   jpg/image_00001.jpg ... image_08189.jpg
///   imagelabels.mat   (1xN labels in [1, 102])
///   setid.mat         (struct with trnid/valid/tstid 1-indexed image lists)
/// </code>
/// Auto-downloads the canonical Oxford VGG release (3 separate files).
/// </para>
/// </remarks>
public class Flowers102DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 102;
    private static readonly string ImagesUrl = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz";
    private static readonly string LabelsUrl = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat";
    private static readonly string SetidUrl = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat";

    private readonly Flowers102DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    public override string Name => "Flowers-102";
    public override string Description => "Oxford Flowers-102 fine-grained classification";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _imageSize * _imageSize * 3;
    public override int OutputDimension => NumClasses;

    public Flowers102DataLoader(Flowers102DataLoaderOptions? options = null)
    {
        _options = options ?? new Flowers102DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("flowers-102");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string jpgDir = Path.Combine(_dataPath, "jpg");
        string labelsPath = Path.Combine(_dataPath, "imagelabels.mat");
        string setidPath = Path.Combine(_dataPath, "setid.mat");

        if (_options.AutoDownload)
        {
            Directory.CreateDirectory(_dataPath);
            if (!Directory.Exists(jpgDir))
                await DatasetDownloader.DownloadAndExtractTarGzAsync(ImagesUrl, _dataPath, cancellationToken);
            if (!File.Exists(labelsPath))
                await DatasetDownloader.DownloadFileAsync(LabelsUrl, labelsPath, cancellationToken);
            if (!File.Exists(setidPath))
                await DatasetDownloader.DownloadFileAsync(SetidUrl, setidPath, cancellationToken);
        }

        if (!Directory.Exists(jpgDir) || !File.Exists(labelsPath) || !File.Exists(setidPath))
            throw new FileNotFoundException($"Flowers-102 missing files at {_dataPath} (need jpg/, imagelabels.mat, setid.mat).");

        // imagelabels.mat: variable 'labels' shape (1, 8189) int16 or int32, values 1..102
        int[] labels;
        using (var fs = new FileStream(labelsPath, FileMode.Open, FileAccess.Read))
        {
            var mat = new MatFileReader(fs).Read();
            labels = ReadIntVector(mat, "labels");
        }

        // setid.mat: variables trnid, valid, tstid (1-indexed image ids, 1..8189)
        int[] trnid, valid, tstid;
        using (var fs = new FileStream(setidPath, FileMode.Open, FileAccess.Read))
        {
            var mat = new MatFileReader(fs).Read();
            trnid = ReadIntVector(mat, "trnid");
            valid = ReadIntVector(mat, "valid");
            tstid = ReadIntVector(mat, "tstid");
        }
        int[] imgIds = _options.Split switch
        {
            Geometry.DatasetSplit.Test => tstid,
            Geometry.DatasetSplit.Validation => valid,
            _ => trnid
        };

        int totalSamples = imgIds.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int imgId = imgIds[i]; // 1-indexed
            string imgPath = Path.Combine(jpgDir, $"image_{imgId:D5}.jpg");
            if (!File.Exists(imgPath)) continue;
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            int label1 = imgId - 1 < labels.Length ? labels[imgId - 1] : 0; // 1..102 → 0..101
            int label = label1 - 1;
            if (label >= 0 && label < NumClasses) labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
        InitializeIndices(totalSamples);
        await Task.CompletedTask;
    }

    /// <summary>Reads an int-typed MAT variable of any width (uint8/int16/int32/double) as int[].</summary>
    private static int[] ReadIntVector(IMatFile mat, string varName)
    {
        foreach (var v in mat.Variables)
        {
            if (v.Name != varName) continue;
            if (v.Value is IArrayOf<double> dArr) return dArr.Data.Select(d => (int)d).ToArray();
            if (v.Value is IArrayOf<int> iArr) return iArr.Data.ToArray();
            if (v.Value is IArrayOf<short> sArr) return sArr.Data.Select(s => (int)s).ToArray();
            if (v.Value is IArrayOf<ushort> usArr) return usArr.Data.Select(s => (int)s).ToArray();
            if (v.Value is IArrayOf<byte> bArr) return bArr.Data.Select(b => (int)b).ToArray();
            if (v.Value is IArrayOf<uint> uiArr) return uiArr.Data.Select(u => (int)u).ToArray();
            throw new InvalidDataException($"Variable {varName} has unsupported numeric type {v.Value.GetType().Name}.");
        }
        throw new InvalidDataException($"Variable {varName} not found in MAT file.");
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
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Take(trainSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Skip(trainSize + valSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private static Tensor<T> ExtractTensorBatchLocal(Tensor<T> source, int[] indices)
    {
        var result = AiDotNet.Helpers.TensorCopyHelper.CreateEmptyBatchLike(source, indices.Length);
        for (int i = 0; i < indices.Length; i++)
            AiDotNet.Helpers.TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
