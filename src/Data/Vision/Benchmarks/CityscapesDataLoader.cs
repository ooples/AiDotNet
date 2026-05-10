using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Cityscapes semantic-segmentation dataset (Cordts et al. 2016).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects the canonical extracted layout (after manually downloading
/// both archives from cityscapes-dataset.com — sign-up required, no
/// auto-download):
/// <code>
/// {DataPath}/
///   leftImg8bit/{train,val,test}/{city}/{frame}_leftImg8bit.png
///   gtFine/{train,val,test}/{city}/{frame}_gtFine_labelIds.png
/// </code>
/// </para>
/// <para>
/// <b>Output shape:</b> features Tensor[N, H, W, 3] (resized RGB image,
/// normalized to [0, 1]); labels Tensor[N, H, W] (integer class IDs per
/// pixel — the standard semantic-seg label format used by PyTorch /
/// TF / SegFormer / DeepLab loss heads). With <see cref="CityscapesDataLoaderOptions.MapToTrainIds"/>
/// (default) source IDs 0..33 are remapped to the 19 evaluation classes;
/// "ignore" pixels get sentinel value 255.
/// </para>
/// </remarks>
public class CityscapesDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    /// <summary>Number of evaluation classes when MapToTrainIds is true.</summary>
    public const int NumEvalClasses = 19;
    /// <summary>Sentinel ID for "ignore" pixels (background, out-of-eval classes).</summary>
    public const int IgnoreLabel = 255;

    /// <summary>
    /// Cityscapes ID-to-trainID lookup (Cordts et al. 2016, table 4 in the
    /// dataset paper, also implemented in CityscapesScripts/labels.py).
    /// 30 source IDs → 19 evaluation IDs; everything else maps to 255 (ignore).
    /// </summary>
    private static readonly byte[] IdToTrainId =
    {
        // 0..6: unlabeled..ground (all ignored)
        IgnoreLabel, IgnoreLabel, IgnoreLabel, IgnoreLabel, IgnoreLabel, IgnoreLabel, IgnoreLabel,
        // 7: road  → 0
        0,
        // 8: sidewalk → 1
        1,
        // 9: parking, 10: rail track (ignored)
        IgnoreLabel, IgnoreLabel,
        // 11: building → 2,  12: wall → 3,  13: fence → 4
        2, 3, 4,
        // 14..16: guard rail / bridge / tunnel (ignored)
        IgnoreLabel, IgnoreLabel, IgnoreLabel,
        // 17: pole → 5
        5,
        // 18: polegroup (ignored)
        IgnoreLabel,
        // 19: traffic light → 6, 20: traffic sign → 7
        6, 7,
        // 21: vegetation → 8, 22: terrain → 9, 23: sky → 10
        8, 9, 10,
        // 24: person → 11, 25: rider → 12,  26: car → 13,  27: truck → 14,  28: bus → 15
        11, 12, 13, 14, 15,
        // 29: caravan, 30: trailer (ignored)
        IgnoreLabel, IgnoreLabel,
        // 31: train → 16, 32: motorcycle → 17, 33: bicycle → 18
        16, 17, 18,
    };

    private readonly CityscapesDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _h;
    private int _w;

    public override string Name => "Cityscapes";
    public override string Description => "Cityscapes urban-driving semantic segmentation (fine annotations)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _h * _w * 3;
    public override int OutputDimension => _h * _w; // per-pixel int labels (not one-hot)

    public CityscapesDataLoader(CityscapesDataLoaderOptions? options = null)
    {
        _options = options ?? new CityscapesDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("cityscapes");
        _h = _options.ImageHeight;
        _w = _options.ImageWidth;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        if (_options.AutoDownload)
            throw new InvalidOperationException(
                "Cityscapes requires account sign-up at cityscapes-dataset.com — auto-download is not supported. " +
                $"Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip manually and extract under {_dataPath}.");

        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "val",
            _ => "train"
        };
        string imgRoot = Path.Combine(_dataPath, "leftImg8bit", splitName);
        string lblRoot = Path.Combine(_dataPath, "gtFine", splitName);
        if (!Directory.Exists(imgRoot))
            throw new DirectoryNotFoundException($"Cityscapes image dir not found: {imgRoot}");

        // Walk city subdirectories under leftImg8bit/{split}/.
        var pairs = new List<(string Img, string Lbl)>();
        foreach (string cityDir in Directory.EnumerateDirectories(imgRoot).OrderBy(d => d, StringComparer.Ordinal))
        {
            string city = Path.GetFileName(cityDir);
            string lblCity = Path.Combine(lblRoot, city);
            foreach (string imgFile in Directory.EnumerateFiles(cityDir, "*_leftImg8bit.png").OrderBy(f => f, StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                string frame = Path.GetFileName(imgFile).Replace("_leftImg8bit.png", "");
                string lblFile = Path.Combine(lblCity, frame + "_gtFine_labelIds.png");
                if (Directory.Exists(lblCity) && File.Exists(lblFile))
                    pairs.Add((imgFile, lblFile));
                else if (splitName == "test")
                    // test split has no public labels — emit zero label maps
                    pairs.Add((imgFile, string.Empty));
            }
        }

        int totalSamples = pairs.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        int pixelsPerImage = _h * _w * 3;
        int pixelsPerLabel = _h * _w;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * pixelsPerLabel];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (imgPath, lblPath) = pairs[i];

            // Image: resized HWC RGB
            var imgPixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _h, _w, 3, _options.Normalize);
            int imgOff = i * pixelsPerImage;
            int copyLen = Math.Min(imgPixels.Length, pixelsPerImage);
            Array.Copy(imgPixels, 0, featuresData, imgOff, copyLen);

            if (string.IsNullOrEmpty(lblPath))
            {
                // Test split: label map is all zeros.
                continue;
            }

            // Label map: load PNG, take R channel only (label IDs are encoded in red channel
            // for *_labelIds.png — single-channel grayscale or first channel of RGB).
            var lblPixels = VisionLoaderHelper.LoadAndResizeImage<T>(lblPath, _h, _w, 3, normalize: false);
            int lblOff = i * pixelsPerLabel;
            for (int p = 0; p < pixelsPerLabel; p++)
            {
                // Take the R channel — first of every 3 entries.
                double idVal = NumOps.ToDouble(lblPixels[p * 3]);
                int srcId = (int)Math.Round(idVal);
                int outId = _options.MapToTrainIds
                    ? (srcId >= 0 && srcId < IdToTrainId.Length ? IdToTrainId[srcId] : IgnoreLabel)
                    : srcId;
                labelsData[lblOff + p] = NumOps.FromDouble(outId);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _h, _w, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, _h, _w });
        InitializeIndices(totalSamples);
        await Task.CompletedTask;
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
