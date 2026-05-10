using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using MatFileHandler;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Stanford Cars dataset (Krause et al. 2013) — 196 fine-grained car classes.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects (manually-extracted):
/// <code>
/// {DataPath}/
///   cars_train/*.jpg
///   cars_test/*.jpg
///   cars_train_annos.mat                  (struct with class, fname, bbox)
///   cars_test_annos_withlabels.mat        (struct with class, fname, bbox)
///   devkit/cars_meta.mat                  (class names; optional)
/// </code>
/// MAT annotations parsed via MatFileHandler; <c>class</c> field is 1-indexed (1..196).
/// </para>
/// </remarks>
public class StanfordCarsDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    public const int NumClasses = 196;

    private readonly StanfordCarsDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    public override string Name => "Stanford Cars";
    public override string Description => "Stanford Cars fine-grained classification (196 classes)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _imageSize * _imageSize * 3;
    public override int OutputDimension => NumClasses;

    public StanfordCarsDataLoader(StanfordCarsDataLoaderOptions? options = null)
    {
        _options = options ?? new StanfordCarsDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("stanford-cars");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        if (_options.AutoDownload)
            throw new InvalidOperationException(
                "StanfordCars auto-download disabled (canonical Stanford URLs are unstable). " +
                $"Manually extract cars_train/, cars_test/, cars_train_annos.mat and " +
                $"cars_test_annos_withlabels.mat into {_dataPath}.");

        bool isTest = _options.Split == Geometry.DatasetSplit.Test
                   || _options.Split == Geometry.DatasetSplit.Validation;
        string imgDir = Path.Combine(_dataPath, isTest ? "cars_test" : "cars_train");
        string annoMat = Path.Combine(_dataPath,
            isTest ? "cars_test_annos_withlabels.mat" : "cars_train_annos.mat");
        if (!Directory.Exists(imgDir))
            throw new DirectoryNotFoundException($"Stanford Cars image dir not found: {imgDir}");
        if (!File.Exists(annoMat))
            throw new FileNotFoundException($"Stanford Cars annotation .mat not found: {annoMat}");

        var paths = new List<(string Path, int Label)>();
        using (var fs = new FileStream(annoMat, FileMode.Open, FileAccess.Read))
        {
            var mat = new MatFileReader(fs).Read();
            // The 'annotations' variable is a struct array with fields:
            // bbox_x1, bbox_y1, bbox_x2, bbox_y2, class, fname.
            foreach (var v in mat.Variables)
            {
                if (v.Name != "annotations") continue;
                if (v.Value is not IStructureArray structArr) continue;
                int n = structArr.Count;
                for (int i = 0; i < n; i++)
                {
                    var cls = structArr["class", i] as IArrayOf<double>;
                    var fname = structArr["fname", i] as ICharArray;
                    if (cls is null || fname is null) continue;
                    int classId1 = cls.Data.Length > 0 ? (int)cls.Data[0] : 0;
                    int label = classId1 - 1; // 1..196 → 0..195
                    if (label < 0 || label >= NumClasses) continue;
                    string filename = fname.String;
                    string imgPath = Path.Combine(imgDir, filename);
                    if (File.Exists(imgPath)) paths.Add((imgPath, label));
                }
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
            labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
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
