using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Tiny ImageNet 200-class image-classification dataset
/// (500 train + 50 val + 50 test per class at 64×64).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects the canonical Stanford CS231n directory layout:
/// <code>
/// {DataPath}/tiny-imagenet-200/
///   train/{wnid}/images/*.JPEG
///   val/images/*.JPEG  +  val/val_annotations.txt
///   test/images/*.JPEG  (test labels not released)
///   wnids.txt           (class wnids)
/// </code>
/// Auto-download fetches the canonical zip from cs231n.stanford.edu.
/// Validation labels are mapped via <c>val_annotations.txt</c>; test
/// labels are zeroed since they're hidden upstream.
/// </para>
/// </remarks>
public class TinyImageNetDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 200;
    private static readonly string DownloadUrl =
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip";

    private readonly TinyImageNetDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => "Tiny ImageNet";
    /// <inheritdoc/>
    public override string Description => "Tiny ImageNet 200-class image classification (64x64)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;

    /// <summary>Creates a new Tiny ImageNet data loader.</summary>
    public TinyImageNetDataLoader(TinyImageNetDataLoaderOptions? options = null)
    {
        _options = options ?? new TinyImageNetDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("tiny-imagenet");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractZipAsync(DownloadUrl, _dataPath, cancellationToken);
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"Tiny ImageNet not found at {_dataPath}.");

        // Build wnid -> classIdx map from wnids.txt
        string wnidsPath = Path.Combine(root, "wnids.txt");
        if (!File.Exists(wnidsPath))
            throw new FileNotFoundException($"wnids.txt not found at {wnidsPath}.");
        var wnids = (await FilePolyfill.ReadAllLinesAsync(wnidsPath, cancellationToken))
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .Select(l => l.Trim())
            .ToArray();
        var wnidIdx = new Dictionary<string, int>(wnids.Length);
        for (int i = 0; i < wnids.Length; i++) wnidIdx[wnids[i]] = i;

        var paths = new List<(string Path, int Label)>();

        switch (_options.Split)
        {
            case Geometry.DatasetSplit.Validation:
            case Geometry.DatasetSplit.Test:
            {
                // val/val_annotations.txt: filename<TAB>wnid<TAB>x1<TAB>y1<TAB>x2<TAB>y2
                string valDir = Path.Combine(root, _options.Split == Geometry.DatasetSplit.Test ? "test" : "val");
                string imagesDir = Path.Combine(valDir, "images");
                string annPath = Path.Combine(valDir, "val_annotations.txt");
                if (!Directory.Exists(imagesDir))
                    throw new DirectoryNotFoundException($"Split images dir not found: {imagesDir}");
                if (File.Exists(annPath))
                {
                    foreach (string line in await FilePolyfill.ReadAllLinesAsync(annPath, cancellationToken))
                    {
                        if (string.IsNullOrWhiteSpace(line)) continue;
                        var parts = line.Split('\t');
                        if (parts.Length < 2) continue;
                        if (!wnidIdx.TryGetValue(parts[1], out int label)) continue;
                        paths.Add((Path.Combine(imagesDir, parts[0]), label));
                    }
                }
                else
                {
                    // Test split has no labels — use 0 for every image.
                    foreach (var f in Directory.EnumerateFiles(imagesDir, "*.JPEG"))
                        paths.Add((f, 0));
                }
                break;
            }
            default:
            {
                string trainRoot = Path.Combine(root, "train");
                if (!Directory.Exists(trainRoot))
                    throw new DirectoryNotFoundException($"train dir not found: {trainRoot}");
                foreach (string wnidDir in Directory.EnumerateDirectories(trainRoot))
                {
                    string wnid = Path.GetFileName(wnidDir);
                    if (!wnidIdx.TryGetValue(wnid, out int label)) continue;
                    string imagesDir = Path.Combine(wnidDir, "images");
                    string searchDir = Directory.Exists(imagesDir) ? imagesDir : wnidDir;
                    foreach (var f in Directory.EnumerateFiles(searchDir, "*.JPEG"))
                        paths.Add((f, label));
                }
                break;
            }
        }

        int totalSamples = paths.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (imgPath, label) = paths[i];
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            if (label >= 0 && label < NumClasses)
                labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
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
        string sub = Path.Combine(_dataPath, "tiny-imagenet-200");
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
