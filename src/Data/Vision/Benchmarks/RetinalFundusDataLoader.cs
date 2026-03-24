using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads retinal fundus photography datasets for diabetic retinopathy detection (5-class grading).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Retinal Fundus expects:
/// <code>
/// {DataPath}/
///   train/
///     10_left.jpeg
///     10_right.jpeg
///     ...
///   test/
///     ...
///   trainLabels.csv  (or train.csv)
/// </code>
/// The CSV has columns: image (filename without extension), level (0-4 severity grade).
/// Labels are one-hot encoded as Tensor[N, 5] for the 5 DR severity levels.
/// </para>
/// </remarks>
public class RetinalFundusDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string[] SeverityLabels =
    {
        "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
    };

    private const int NumClasses = 5;

    private readonly RetinalFundusDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => "RetinalFundus-DR";
    /// <inheritdoc/>
    public override string Description => "Retinal fundus diabetic retinopathy grading dataset (5 levels)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;
    /// <summary>Gets the severity level names.</summary>
    public IReadOnlyList<string> SeverityNames => SeverityLabels;

    /// <summary>Creates a new Retinal Fundus data loader.</summary>
    public RetinalFundusDataLoader(RetinalFundusDataLoaderOptions? options = null)
    {
        _options = options ?? new RetinalFundusDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("retinal-fundus-dr");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        bool isTest = _options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation;
        string imageDir = Path.Combine(_dataPath, isTest ? "test" : "train");

        // Load labels CSV (try split-specific first, then fallback to common files)
        string csvFile;
        if (isTest)
        {
            csvFile = Path.Combine(_dataPath, "testLabels.csv");
            if (!File.Exists(csvFile))
                csvFile = Path.Combine(_dataPath, "test.csv");
            if (!File.Exists(csvFile))
                csvFile = Path.Combine(_dataPath, "retinopathy_solution.csv");
        }
        else
        {
            csvFile = Path.Combine(_dataPath, "trainLabels.csv");
            if (!File.Exists(csvFile))
                csvFile = Path.Combine(_dataPath, "train.csv");
        }

        if (!File.Exists(csvFile))
        {
            throw new FileNotFoundException(
                $"Retinal fundus labels CSV not found at {_dataPath}. " +
                "Download from https://www.kaggle.com/c/diabetic-retinopathy-detection/.");
        }

        var csvLines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string ImageName, int Label)>();

        // Skip header
        for (int lineIdx = 1; lineIdx < csvLines.Length; lineIdx++)
        {
            var parts = csvLines[lineIdx].Split(',');
            if (parts.Length < 2) continue;

            string imageName = parts[0].Trim();
            if (int.TryParse(parts[1].Trim(), out int level) && level >= 0 && level < NumClasses)
            {
                samples.Add((imageName, level));
            }
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

            var (imageName, label) = samples[i];
            // Try multiple extensions
            string imgPath = FindImageFile(imageDir, imageName);

            if (imgPath.Length > 0 && File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
                int featureOffset = i * pixelsPerImage;
                int copyLen = Math.Min(pixels.Length, pixelsPerImage);
                Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            }
            else
            {
                System.Diagnostics.Debug.WriteLine(
                    $"[RetinalFundusDataLoader] Warning: Image not found for '{imageName}'. Features will be zero-filled.");
            }

            labelsData[i * NumClasses + label] = NumOps.One;
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

    private static string FindImageFile(string imageDir, string baseName)
    {
        string[] extensions = { ".jpeg", ".jpg", ".png", ".tiff", ".bmp" };
        foreach (var ext in extensions)
        {
            string path = Path.Combine(imageDir, baseName + ext);
            if (File.Exists(path)) return path;
        }

        return string.Empty;
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
