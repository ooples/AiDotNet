using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the ADE20K semantic segmentation dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// ADE20K expects:
/// <code>
/// {DataPath}/
///   images/
///     training/ or validation/   (.jpg files)
///   annotations/
///     training/ or validation/   (.png segmentation masks)
/// </code>
/// Features are flattened image pixels Tensor[N, H * W * 3].
/// Labels are flattened segmentation mask Tensor[N, H * W] with class indices.
/// </para>
/// </remarks>
public class Ade20kDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly Ade20kDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "ADE20K";
    public override string Description => "ADE20K semantic segmentation (150 classes)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.ImageWidth * _options.ImageHeight * 3;
    public override int OutputDimension => _options.ImageWidth * _options.ImageHeight;

    public Ade20kDataLoader(Ade20kDataLoaderOptions? options = null)
    {
        _options = options ?? new Ade20kDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("ade20k");
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitName = _options.Split == DatasetSplit.Validation ? "validation" : "training";

        // Try standard ADE20K layout
        string imgDir = Path.Combine(_dataPath, "images", splitName);
        string annDir = Path.Combine(_dataPath, "annotations", splitName);

        if (!Directory.Exists(imgDir))
        {
            imgDir = Path.Combine(_dataPath, splitName);
            annDir = Path.Combine(_dataPath, splitName + "_annotations");
        }

        if (!Directory.Exists(imgDir))
        {
            imgDir = _dataPath;
            annDir = _dataPath;
        }

        if (!Directory.Exists(imgDir))
            throw new DirectoryNotFoundException($"ADE20K data not found at {imgDir}.");

        var imageFiles = Directory.GetFiles(imgDir, "*.jpg")
            .Concat(Directory.GetFiles(imgDir, "*.png")).ToArray();
        Array.Sort(imageFiles, StringComparer.OrdinalIgnoreCase);

        int totalSamples = imageFiles.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int w = _options.ImageWidth;
        int h = _options.ImageHeight;
        int featureSize = w * h * 3;
        int labelSize = w * h;
        var featuresData = new T[totalSamples * featureSize];
        var labelsData = new T[totalSamples * labelSize];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imageFiles[i], h, w, 3, _options.Normalize);
            int featOff = i * featureSize;
            int copyLen = Math.Min(pixels.Length, featureSize);
            Array.Copy(pixels, 0, featuresData, featOff, copyLen);

            // Try to find corresponding annotation mask
            string baseName = Path.GetFileNameWithoutExtension(imageFiles[i]);
            string annFile = Path.Combine(annDir, baseName + ".png");
            int lblOff = i * labelSize;
            if (File.Exists(annFile))
            {
                // Load annotation mask as single-channel image (class indices per pixel)
                var annPixels = VisionLoaderHelper.LoadAndResizeImage<T>(annFile, h, w, 1, false);
                int annToRead = Math.Min(annPixels.Length, labelSize);
                var numOps = MathHelper.GetNumericOperations<T>();
                for (int p = 0; p < annToRead; p++)
                {
                    double classVal = numOps.ToDouble(annPixels[p]);
                    labelsData[lblOff + p] = numOps.FromDouble(((int)classVal) % _options.NumClasses);
                }
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, labelSize });
        InitializeIndices(totalSamples);
    }

    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        var nfs = (int[])features.Shape._dims.Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels.Shape._dims.Clone(); nls[0] = indices.Length;
        var bf = new Tensor<T>(nfs);
        var bl = new Tensor<T>(nls);
        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(features, bf, indices[i], i);
            TensorCopyHelper.CopySample(labels, bl, indices[i], i);
        }
        return (bf, bl);
    }

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
        return (
            CreateSplit(shuffled.Take(trainSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize).Take(valSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize + valSize).ToArray())
        );
    }

    private InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSplit(int[] indices)
    {
        var (bf, bl) = ExtractBatch(indices);
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(bf, bl);
    }
}
