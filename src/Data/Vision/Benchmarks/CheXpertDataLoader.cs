using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the CheXpert chest radiograph dataset (224K images, 14 observations with uncertainty).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// CheXpert expects:
/// <code>
/// {DataPath}/
///   CheXpert-v1.0-small/    (or CheXpert-v1.0/)
///     train/
///       patient00001/
///         study1/
///           view1_frontal.jpg
///     valid/
///     train.csv
///     valid.csv
/// </code>
/// The CSV has columns: Path, Sex, Age, Frontal/Lateral, AP/PA, then 14 observation columns.
/// Observation values: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), blank (not mentioned).
/// Labels are stored as Tensor[N, 14] with uncertainty handled per the UncertaintyPolicy option.
/// </para>
/// </remarks>
public class CheXpertDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string[] ObservationLabels =
    {
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices"
    };

    private const int NumClasses = 14;

    private readonly CheXpertDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => "CheXpert";
    /// <inheritdoc/>
    public override string Description => "CheXpert chest radiograph dataset (14 observations)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;
    /// <summary>Gets the observation label names.</summary>
    public IReadOnlyList<string> ObservationNames => ObservationLabels;

    /// <summary>Creates a new CheXpert data loader.</summary>
    public CheXpertDataLoader(CheXpertDataLoaderOptions? options = null)
    {
        _options = options ?? new CheXpertDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("chexpert");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Find CheXpert directory
        string baseDir = FindCheXpertDirectory(_dataPath);
        if (baseDir.Length == 0) baseDir = _dataPath;

        bool isVal = _options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation;
        string csvFile = Path.Combine(baseDir, isVal ? "valid.csv" : "train.csv");

        if (!File.Exists(csvFile))
        {
            throw new FileNotFoundException(
                $"CheXpert CSV file not found at {csvFile}. " +
                "Download from https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2.");
        }

        var csvLines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string ImagePath, double[] Labels)>();

        // Skip header and parse
        for (int lineIdx = 1; lineIdx < csvLines.Length; lineIdx++)
        {
            var parts = csvLines[lineIdx].Split(',');
            if (parts.Length < 5 + NumClasses) continue;

            string imagePath = parts[0].Trim();
            var labels = new double[NumClasses];

            for (int j = 0; j < NumClasses; j++)
            {
                int csvIdx = 5 + j; // Skip Path, Sex, Age, Frontal/Lateral, AP/PA
                if (csvIdx >= parts.Length || string.IsNullOrWhiteSpace(parts[csvIdx]))
                {
                    labels[j] = 0; // Not mentioned
                    continue;
                }

                if (double.TryParse(parts[csvIdx].Trim(), out double val))
                {
                    if (val < 0) // Uncertain
                    {
                        labels[j] = _options.UncertaintyHandling switch
                        {
                            UncertaintyPolicy.Ones => 1.0,
                            _ => 0.0
                        };
                    }
                    else
                    {
                        labels[j] = val > 0 ? 1.0 : 0.0;
                    }
                }
            }

            // Resolve relative path: CheXpert CSV paths are relative to the CSV file's directory (baseDir),
            // not _dataPath. E.g., "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
            string fullPath = Path.IsPathRooted(imagePath) ? imagePath : Path.Combine(baseDir, imagePath);
            // If path includes the dataset directory prefix, try resolving from _dataPath as fallback
            if (!File.Exists(fullPath))
                fullPath = Path.IsPathRooted(imagePath) ? imagePath : Path.Combine(_dataPath, imagePath);
            samples.Add((fullPath, labels));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        long pixelsPerImageLong = (long)_imageSize * _imageSize;
        if (pixelsPerImageLong > int.MaxValue)
            throw new InvalidOperationException($"Image dimensions too large: {_imageSize}x{_imageSize} exceeds max array size.");
        int pixelsPerImage = (int)pixelsPerImageLong;
        long totalFeaturesLong = (long)totalSamples * pixelsPerImage;
        if (totalFeaturesLong > int.MaxValue)
            throw new InvalidOperationException($"Total allocation too large: {totalSamples} samples x {pixelsPerImage} pixels exceeds max array size. Use MaxSamples to limit.");
        var featuresData = new T[totalFeaturesLong];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (imgPath, labels) = samples[i];

            if (File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 1, _options.Normalize);
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

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 1 });
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

    private static string FindCheXpertDirectory(string rootPath)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        string[] candidates = { "CheXpert-v1.0-small", "CheXpert-v1.0", "chexpert" };
        foreach (var candidate in candidates)
        {
            string path = Path.Combine(rootPath, candidate);
            if (Directory.Exists(path)) return path;
        }

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
