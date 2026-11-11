using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using MatFileHandler;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the SVHN (Street View House Numbers) Format-2 dataset (Netzer et al. 2011) — 32×32 RGB digits.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/
///   train_32x32.mat
///   test_32x32.mat
///   extra_32x32.mat   (optional, ~531k extras)
/// </code>
/// Each .mat file contains <c>X</c> (uint8 array shaped [32, 32, 3, N], column-
/// major) and <c>y</c> (int8 array shaped [N, 1]) with labels 1..10 (10 = digit 0).
/// Auto-downloads from Stanford's canonical mirror.
/// </para>
/// </remarks>
public class SvhnDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int ImageSize = 32;
    private const int NumClasses = 10;
    private const int PixelsPerImage = ImageSize * ImageSize * 3;

    private static readonly string TrainUrl = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat";
    private static readonly string TestUrl = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat";
    private static readonly string ExtraUrl = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat";

    private readonly SvhnDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc />
    public override string Name => "SVHN";
    /// <inheritdoc />
    public override string Description => "Street View House Numbers Format-2 (32x32 RGB digit classification)";
    /// <inheritdoc />
    public override int TotalCount => _sampleCount;
    /// <inheritdoc />
    public override int FeatureCount => PixelsPerImage;
    /// <inheritdoc />
    public override int OutputDimension => NumClasses;

    public SvhnDataLoader(SvhnDataLoaderOptions? options = null)
    {
        _options = options ?? new SvhnDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("svhn");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        bool isTest = _options.Split == Geometry.DatasetSplit.Test
                   || _options.Split == Geometry.DatasetSplit.Validation;
        string mainFile = isTest ? "test_32x32.mat" : "train_32x32.mat";
        string mainUrl = isTest ? TestUrl : TrainUrl;
        string mainPath = Path.Combine(_dataPath, mainFile);

        if (!File.Exists(mainPath) && _options.AutoDownload)
        {
            Directory.CreateDirectory(_dataPath);
            await DatasetDownloader.DownloadFileAsync(mainUrl, mainPath, cancellationToken);
        }
        if (!File.Exists(mainPath))
            throw new FileNotFoundException($"SVHN .mat not found at {mainPath}.");

        var slabs = new List<(byte[,,,] X, int[] y)>();
        slabs.Add(LoadSvhnMat(mainPath));

        if (!isTest && _options.IncludeExtra)
        {
            string extraPath = Path.Combine(_dataPath, "extra_32x32.mat");
            if (!File.Exists(extraPath) && _options.AutoDownload)
                await DatasetDownloader.DownloadFileAsync(ExtraUrl, extraPath, cancellationToken);
            if (File.Exists(extraPath))
                slabs.Add(LoadSvhnMat(extraPath));
        }

        int totalSamples = slabs.Sum(s => s.y.Length);
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var featuresData = new int[totalSamples * PixelsPerImage];
        var labelsData = new int[totalSamples * NumClasses];

        var engine = AiDotNetEngine.Current;
        int dst = 0;
        foreach (var (X, y) in slabs)
        {
            int n = y.Length;
            int take = Math.Min(n, totalSamples - dst);
            if (take > 0)
            {
                if (X.GetLength(0) != ImageSize || X.GetLength(1) != ImageSize || X.GetLength(2) != 3 || X.GetLength(3) != n)
                    throw new InvalidDataException(
                        $"SVHN image block has shape [{X.GetLength(0)}, {X.GetLength(1)}, {X.GetLength(2)}, {X.GetLength(3)}], " +
                        $"expected [{ImageSize}, {ImageSize}, 3, {n}].");

                // X is (h, w, c, n) row-major, i.e. [pixel, sample]; one engine transpose gives [sample, pixel]
                // in (h, w, c) HWC order, and the first `take` samples are copied as one block.
                var raw = new byte[X.Length];
                Buffer.BlockCopy(X, 0, raw, 0, X.Length);
                var values = new int[X.Length];
                Array.Copy(raw, values, X.Length);
                var sampleMajor = engine.TensorPermute(new Tensor<int>(values, new[] { PixelsPerImage, n }), new[] { 1, 0 })
                    .Contiguous();
                sampleMajor.AsSpan().Slice(0, take * PixelsPerImage)
                    .CopyTo(featuresData.AsSpan(dst * PixelsPerImage, take * PixelsPerImage));
            }

            for (int i = 0; i < n && dst < totalSamples; i++, dst++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int label1 = y[i];           // 1..10 in source
                int label = label1 == 10 ? 0 : label1; // map 10 -> 0 (digit 0)
                if (label >= 0 && label < NumClasses)
                    labelsData[dst * NumClasses + label] = 1;
            }
            if (dst >= totalSamples) break;
        }

        var features = new Tensor<int>(featuresData, new[] { totalSamples, ImageSize, ImageSize, 3 }).Cast<T>();
        if (_options.Normalize)
        {
            // Raw bytes -> [0, 1] in one engine op (rows never filled stay 0)
            features = engine.TensorDivideScalar(features, NumOps.FromDouble(255.0));
        }

        LoadedFeatures = features;
        LoadedLabels = new Tensor<int>(labelsData, new[] { totalSamples, NumClasses }).Cast<T>();
        InitializeIndices(totalSamples);
        await Task.CompletedTask;
    }

    /// <summary>
    /// Reads an SVHN-format .mat file (variables <c>X</c>: uint8 4-D and
    /// <c>y</c>: int8 2-D). Returns a managed (h, w, c, n) byte array and an
    /// int label vector.
    /// </summary>
    private static (byte[,,,] X, int[] y) LoadSvhnMat(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        var reader = new MatFileReader(fs);
        var mat = reader.Read();

        // SVHN Format-2: X is uint8 (channel-major HWC), y is uint8 (labels 1..10, where 10 = digit 0).
        // Accept int8 too — older converters and some scipy.io paths emit int8 for the label tensor.
        IArrayOf<byte>? xArr = null;
        byte[]? yData = null;
        int[]? yDims = null;
        foreach (var v in mat.Variables)
        {
            if (v.Name == "X" && v.Value is IArrayOf<byte> xb) xArr = xb;
            else if (v.Name == "y")
            {
                if (v.Value is IArrayOf<byte> yu)
                {
                    yData = yu.Data.ToArray();
                    yDims = yu.Dimensions;
                }
                else if (v.Value is IArrayOf<sbyte> ys)
                {
                    yData = ys.Data.Select(x => (byte)x).ToArray();
                    yDims = ys.Dimensions;
                }
                else if (v.Value is IArrayOf<short> yshort)
                {
                    yData = yshort.Data.Select(x => (byte)x).ToArray();
                    yDims = yshort.Dimensions;
                }
                else if (v.Value is IArrayOf<int> yint)
                {
                    yData = yint.Data.Select(x => (byte)x).ToArray();
                    yDims = yint.Dimensions;
                }
                else if (v.Value is IArrayOf<double> ydbl)
                {
                    yData = ydbl.Data.Select(x => (byte)Math.Round(x)).ToArray();
                    yDims = ydbl.Dimensions;
                }
            }
        }
        if (xArr is null || yData is null)
            throw new InvalidDataException(
                $"{path} missing required SVHN variables X (uint8) and y (uint8/int8/int/double).");

        // Dimensions are (h=32, w=32, c=3, n=N) for X; (n, 1) for y.
        var dims = xArr.Dimensions;
        if (dims.Length != 4 || dims[0] != ImageSize || dims[1] != ImageSize || dims[2] != 3)
            throw new InvalidDataException($"Unexpected SVHN X dims: [{string.Join(",", dims)}].");
        int n = dims[3];

        var X = new byte[ImageSize, ImageSize, 3, n];
        var data = xArr.Data; // column-major flat
        // MatFileHandler exposes Data as a flat array in column-major (Fortran) order:
        // index linearization (i0, i1, i2, i3) = i0 + d0*(i1 + d1*(i2 + d2*i3))
        for (int i3 = 0; i3 < n; i3++)
            for (int i2 = 0; i2 < 3; i2++)
                for (int i1 = 0; i1 < ImageSize; i1++)
                    for (int i0 = 0; i0 < ImageSize; i0++)
                    {
                        int flat = i0 + ImageSize * (i1 + ImageSize * (i2 + 3 * i3));
                        X[i0, i1, i2, i3] = data[flat];
                    }

        var y = new int[n];
        for (int i = 0; i < n; i++) y[i] = yData[i];
        return (X, y);
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
