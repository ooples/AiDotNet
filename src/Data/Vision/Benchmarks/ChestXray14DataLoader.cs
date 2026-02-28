using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the NIH Chest X-ray 14 multi-label classification dataset (112K images, 14 disease labels).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Chest X-ray 14 expects:
/// <code>
/// {DataPath}/
///   images/
///     00000001_000.png
///     ...
///   Data_Entry_2017_v2020.csv
///   train_val_list.txt
///   test_list.txt
/// </code>
/// The CSV contains columns: Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, etc.
/// Finding Labels is a pipe-separated list of disease names (e.g., "Atelectasis|Effusion").
/// Labels are multi-hot encoded as Tensor[N, 14] where each dimension corresponds to a disease.
/// </para>
/// </remarks>
public class ChestXray14DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string[] DiseaseLabels =
    {
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    };

    private static readonly Dictionary<string, int> DiseaseToIndex;

    static ChestXray14DataLoader()
    {
        DiseaseToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < DiseaseLabels.Length; i++)
            DiseaseToIndex[DiseaseLabels[i]] = i;
    }

    private const int NumClasses = 14;

    private readonly ChestXray14DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => "ChestX-ray14";
    /// <inheritdoc/>
    public override string Description => "NIH Chest X-ray 14 multi-label classification dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;
    /// <summary>Gets the disease label names.</summary>
    public IReadOnlyList<string> DiseaseNames => DiseaseLabels;

    /// <summary>Creates a new ChestX-ray14 data loader.</summary>
    public ChestXray14DataLoader(ChestXray14DataLoaderOptions? options = null)
    {
        _options = options ?? new ChestXray14DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("chest-xray-14");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Load split file to determine which images belong to this split
        string splitFile = _options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation
            ? Path.Combine(_dataPath, "test_list.txt")
            : Path.Combine(_dataPath, "train_val_list.txt");

        HashSet<string>? splitImages = null;
        if (File.Exists(splitFile))
        {
            var splitLines = await FilePolyfill.ReadAllLinesAsync(splitFile, cancellationToken);
            splitImages = new HashSet<string>(splitLines.Select(l => l.Trim()).Where(l => l.Length > 0),
                StringComparer.OrdinalIgnoreCase);
        }

        // Load the main CSV data entry file
        string csvFile = Path.Combine(_dataPath, "Data_Entry_2017_v2020.csv");
        if (!File.Exists(csvFile))
        {
            csvFile = Path.Combine(_dataPath, "Data_Entry_2017.csv");
        }

        if (!File.Exists(csvFile))
        {
            throw new FileNotFoundException(
                $"Chest X-ray 14 data entry CSV not found at {_dataPath}. " +
                "Download from https://nihcc.app.box.com/v/ChestXray-NIHCC.");
        }

        var csvLines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string ImageFile, int[] Labels)>();

        // Skip header
        for (int lineIdx = 1; lineIdx < csvLines.Length; lineIdx++)
        {
            var parts = csvLines[lineIdx].Split(',');
            if (parts.Length < 2) continue;

            string imageFile = parts[0].Trim();

            // Apply split filter
            if (splitImages != null && !splitImages.Contains(imageFile)) continue;

            // Parse finding labels (pipe-separated)
            string findings = parts[1].Trim();
            var labelIndices = new List<int>();
            if (!findings.Equals("No Finding", StringComparison.OrdinalIgnoreCase))
            {
                foreach (var finding in findings.Split('|'))
                {
                    if (DiseaseToIndex.TryGetValue(finding.Trim(), out int idx))
                    {
                        labelIndices.Add(idx);
                    }
                }
            }

            samples.Add((imageFile, labelIndices.ToArray()));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int pixelsPerImage = _imageSize * _imageSize; // Grayscale
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        string imageDir = Path.Combine(_dataPath, "images");

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (imageFile, labelIndices) = samples[i];
            string imgPath = Path.Combine(imageDir, imageFile);

            if (File.Exists(imgPath))
            {
                var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 1, _options.Normalize);
                int featureOffset = i * pixelsPerImage;
                int copyLen = Math.Min(pixels.Length, pixelsPerImage);
                Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            }

            // Multi-hot encode labels
            int labelOffset = i * NumClasses;
            foreach (int idx in labelIndices)
            {
                labelsData[labelOffset + idx] = NumOps.One;
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
