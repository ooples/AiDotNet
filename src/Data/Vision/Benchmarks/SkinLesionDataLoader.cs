using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the ISIC Skin Lesion classification dataset (~25K images, 8 diagnostic categories).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Skin Lesion expects:
/// <code>
/// {DataPath}/
///   ISIC_2019_Training_Input/
///     ISIC_0024306.jpg
///     ...
///   ISIC_2019_Training_GroundTruth.csv
/// </code>
/// The ground truth CSV has columns: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC.
/// Each column is 0.0 or 1.0, with exactly one positive per row (one-hot).
/// Labels are stored as Tensor[N, 8].
/// </para>
/// </remarks>
public class SkinLesionDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string[] ClassLabels =
    {
        "MEL",  // Melanoma
        "NV",   // Melanocytic nevus
        "BCC",  // Basal cell carcinoma
        "AK",   // Actinic keratosis
        "BKL",  // Benign keratosis
        "DF",   // Dermatofibroma
        "VASC", // Vascular lesion
        "SCC"   // Squamous cell carcinoma
    };

    private const int NumClasses = 8;

    private readonly SkinLesionDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => "ISIC-SkinLesion";
    /// <inheritdoc/>
    public override string Description => "ISIC Skin Lesion classification dataset (8 categories)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;
    /// <summary>Gets the class label names.</summary>
    public IReadOnlyList<string> ClassNames => ClassLabels;

    /// <summary>Creates a new Skin Lesion data loader.</summary>
    public SkinLesionDataLoader(SkinLesionDataLoaderOptions? options = null)
    {
        _options = options ?? new SkinLesionDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("isic-skin-lesion");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Find ground truth CSV
        string csvFile = FindGroundTruthCsv(_dataPath);

        if (csvFile.Length == 0)
        {
            throw new FileNotFoundException(
                $"ISIC Skin Lesion ground truth CSV not found at {_dataPath}. " +
                "Download from https://challenge.isic-archive.com/.");
        }

        // Find images directory
        string imageDir = FindImageDirectory(_dataPath);
        if (imageDir.Length == 0) imageDir = _dataPath;

        var csvLines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string ImageFile, double[] Labels)>();

        // Parse CSV (skip header)
        for (int lineIdx = 1; lineIdx < csvLines.Length; lineIdx++)
        {
            var parts = csvLines[lineIdx].Split(',');
            if (parts.Length < 1 + NumClasses) continue;

            string imageId = parts[0].Trim();
            var labels = new double[NumClasses];

            for (int j = 0; j < NumClasses; j++)
            {
                if (j + 1 < parts.Length && double.TryParse(parts[j + 1].Trim(), out double val))
                {
                    labels[j] = val;
                }
            }

            samples.Add((imageId, labels));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (imageId, labels) = samples[i];
            string imgPath = Path.Combine(imageDir, $"{imageId}.jpg");

            if (File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
                int featureOffset = i * pixelsPerImage;
                int copyLen = Math.Min(pixels.Length, pixelsPerImage);
                Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            }

            int labelOffset = i * NumClasses;
            for (int j = 0; j < NumClasses; j++)
            {
                labelsData[labelOffset + j] = NumOps.FromDouble(labels[j]);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
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
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

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

    private static string FindGroundTruthCsv(string rootPath)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        var files = Directory.GetFiles(rootPath, "*GroundTruth*.csv", SearchOption.AllDirectories);
        if (files.Length > 0) return files[0];

        files = Directory.GetFiles(rootPath, "*ground_truth*.csv", SearchOption.AllDirectories);
        if (files.Length > 0) return files[0];

        files = Directory.GetFiles(rootPath, "*.csv", SearchOption.TopDirectoryOnly);
        if (files.Length > 0) return files[0];

        return string.Empty;
    }

    private static string FindImageDirectory(string rootPath)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        string[] candidates = { "ISIC_2019_Training_Input", "Training_Input", "images", "train" };
        foreach (var candidate in candidates)
        {
            string path = Path.Combine(rootPath, candidate);
            if (Directory.Exists(path)) return path;
        }

        var dirs = Directory.GetDirectories(rootPath, "*Training_Input*", SearchOption.AllDirectories);
        if (dirs.Length > 0) return dirs[0];

        return rootPath;
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape._dims.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
