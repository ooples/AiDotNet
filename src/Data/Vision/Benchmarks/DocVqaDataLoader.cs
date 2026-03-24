using System.Text.Json;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the DocVQA document visual question answering dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// DocVQA expects:
/// <code>
/// {DataPath}/
///   train/ or val/ or test/
///     documents/    (image files: .png, .jpg)
///     annotations.json or qa.json (question-answer pairs)
/// </code>
/// Features are flattened image pixels Tensor[N, H * W * 3].
/// Labels are answer text encoded as character indices Tensor[N, MaxAnswerLength],
/// where each element is the Unicode code point of the character.
/// </para>
/// </remarks>
public class DocVqaDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly DocVqaDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "DocVQA";
    public override string Description => "DocVQA document visual question answering";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.ImageWidth * _options.ImageHeight * 3;
    public override int OutputDimension => _options.MaxAnswerLength;

    public DocVqaDataLoader(DocVqaDataLoaderOptions? options = null)
    {
        _options = options ?? new DocVqaDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("docvqa");
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitName = _options.Split == DatasetSplit.Test ? "test"
            : _options.Split == DatasetSplit.Validation ? "val"
            : "train";

        string splitDir = Path.Combine(_dataPath, splitName);
        if (!Directory.Exists(splitDir))
            splitDir = _dataPath;

        // Find document images
        string docsDir = Path.Combine(splitDir, "documents");
        if (!Directory.Exists(docsDir))
            docsDir = splitDir;

        if (!Directory.Exists(docsDir))
            throw new DirectoryNotFoundException($"DocVQA data not found at {docsDir}.");

        // Parse annotations JSON to get image->answer mapping
        var imageAnswers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        string[] annotationFiles = { "annotations.json", "qa.json", "train_v1.0.json", "val_v1.0.json" };
        foreach (string annFile in annotationFiles)
        {
            string annPath = Path.Combine(splitDir, annFile);
            if (!File.Exists(annPath))
                annPath = Path.Combine(_dataPath, annFile);
            if (!File.Exists(annPath)) continue;

            using var stream = File.OpenRead(annPath);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
            var root = doc.RootElement;

            // DocVQA format: {"data": [{"image": "...", "answers": ["..."], "question": "..."}]}
            JsonElement dataArray;
            if (root.TryGetProperty("data", out dataArray))
            {
                foreach (var item in dataArray.EnumerateArray())
                {
                    string imageName = item.TryGetProperty("image", out var imgElem)
                        ? imgElem.GetString() ?? string.Empty
                        : string.Empty;

                    string answer = string.Empty;
                    if (item.TryGetProperty("answers", out var answersElem) && answersElem.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var ans in answersElem.EnumerateArray())
                        {
                            answer = ans.GetString() ?? string.Empty;
                            if (answer.Length > 0) break; // Take first answer
                        }
                    }

                    if (imageName.Length > 0 && !imageAnswers.ContainsKey(imageName))
                        imageAnswers[imageName] = answer;
                }
            }
            break; // Use first found annotation file
        }

        var imageFiles = Directory.GetFiles(docsDir, "*.png")
            .Concat(Directory.GetFiles(docsDir, "*.jpg"))
            .Concat(Directory.GetFiles(docsDir, "*.jpeg")).ToArray();
        Array.Sort(imageFiles, StringComparer.OrdinalIgnoreCase);

        int totalSamples = imageFiles.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int w = _options.ImageWidth;
        int h = _options.ImageHeight;
        int featureSize = w * h * 3;
        int maxAnswerLen = _options.MaxAnswerLength;
        var featuresData = new T[totalSamples * featureSize];
        var labelsData = new T[totalSamples * maxAnswerLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imageFiles[i], h, w, 3, true);
            int featOff = i * featureSize;
            int copyLen = Math.Min(pixels.Length, featureSize);
            Array.Copy(pixels, 0, featuresData, featOff, copyLen);

            // Encode answer text as character indices
            string fileName = Path.GetFileName(imageFiles[i]);
            string answer = imageAnswers.TryGetValue(fileName, out string? ans) ? ans : string.Empty;
            int labelOffset = i * maxAnswerLen;
            int charCount = Math.Min(answer.Length, maxAnswerLen);
            for (int c = 0; c < charCount; c++)
                labelsData[labelOffset + c] = NumOps.FromDouble(answer[c]);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, maxAnswerLen });
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
