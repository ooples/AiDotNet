using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the Google Speech Commands v2 dataset (65K clips, 35 words, 16kHz).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Speech Commands v2 expects the following directory structure after extraction:
/// <code>
/// {DataPath}/
///   yes/
///     {speaker_id}_nohash_{index}.wav
///   no/
///     ...
///   up/
///     ...
///   validation_list.txt
///   testing_list.txt
/// </code>
/// Each subdirectory is a word class. WAV files are 16-bit PCM mono at 16kHz, ~1 second each.
/// Features are raw waveform Tensor[N, TargetLength]. Labels are one-hot Tensor[N, NumClasses].
/// </para>
/// <para>
/// The "core" 12-class subset (default) uses: yes, no, up, down, left, right, on, off,
/// stop, go + silence + unknown. The full 35-class mode uses all word directories.
/// </para>
/// </remarks>
public class SpeechCommandsDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl =
        "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz";

    /// <summary>Core 10 spoken words (the standard benchmark subset).</summary>
    public static readonly string[] CoreWords =
        ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"];

    /// <summary>All 35 word classes in the full dataset.</summary>
    public static readonly string[] AllWords =
        ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
         "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
         "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
         "tree", "two", "up", "visual", "wow", "yes", "zero"];

    private readonly SpeechCommandsDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private readonly int _numClasses;
    private readonly string[] _wordList;

    /// <inheritdoc/>
    public override string Name => "SpeechCommands";

    /// <inheritdoc/>
    public override string Description => _options.UseCoreSubset
        ? "Google Speech Commands v2 (core 10 words)"
        : "Google Speech Commands v2 (all 35 words)";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _options.TargetLength;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Number of word classes.</summary>
    public int NumClasses => _numClasses;

    /// <summary>Gets the word list being used.</summary>
    public IReadOnlyList<string> WordList => _wordList;

    /// <summary>Creates a new Speech Commands data loader.</summary>
    public SpeechCommandsDataLoader(SpeechCommandsDataLoaderOptions? options = null)
    {
        _options = options ?? new SpeechCommandsDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("speech_commands");
        _wordList = _options.UseCoreSubset ? CoreWords : AllWords;
        _numClasses = _wordList.Length;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Build file list from directory structure
        var testingListPath = Path.Combine(_dataPath, "testing_list.txt");
        var validationListPath = Path.Combine(_dataPath, "validation_list.txt");

        // Load official split lists if available
        var testFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var valFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (File.Exists(testingListPath))
        {
            foreach (var line in await FilePolyfill.ReadAllLinesAsync(testingListPath, cancellationToken))
            {
                string trimmed = line.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                    testFiles.Add(trimmed.Replace('/', Path.DirectorySeparatorChar));
            }
        }

        if (File.Exists(validationListPath))
        {
            foreach (var line in await FilePolyfill.ReadAllLinesAsync(validationListPath, cancellationToken))
            {
                string trimmed = line.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                    valFiles.Add(trimmed.Replace('/', Path.DirectorySeparatorChar));
            }
        }

        // Collect samples
        var samples = new List<(string AudioPath, int ClassIndex)>();
        var classCounts = new int[_numClasses];

        for (int c = 0; c < _numClasses; c++)
        {
            string wordDir = Path.Combine(_dataPath, _wordList[c]);
            if (!Directory.Exists(wordDir)) continue;

            foreach (string wavFile in Directory.GetFiles(wordDir, "*.wav"))
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Determine which split this file belongs to
                string relPath = Path.Combine(_wordList[c], Path.GetFileName(wavFile));
                bool isTest = testFiles.Contains(relPath);
                bool isVal = valFiles.Contains(relPath);
                bool isTrain = !isTest && !isVal;

                bool include = _options.Split switch
                {
                    DatasetSplit.Test => isTest,
                    DatasetSplit.Validation => isVal,
                    _ => isTrain
                };

                if (!include) continue;

                // Respect per-class limit
                if (_options.MaxSamplesPerClass.HasValue &&
                    classCounts[c] >= _options.MaxSamplesPerClass.Value)
                    continue;

                samples.Add((wavFile, c));
                classCounts[c]++;
            }
        }

        int totalSamples = samples.Count;
        _sampleCount = totalSamples;

        if (totalSamples == 0)
            throw new InvalidOperationException(
                $"No Speech Commands data found at {_dataPath}. " +
                $"Download from {DownloadUrl} and extract to {_dataPath}.");

        // Load audio data
        int targetLen = _options.TargetLength;
        var featuresData = new T[totalSamples * targetLen];
        var labelsData = new T[totalSamples * _numClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (audioPath, classIndex) = samples[i];

            if (File.Exists(audioPath))
            {
                byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(audioPath, cancellationToken);
                AudioLoaderHelper.LoadAudioSamples(audioBytes, featuresData,
                    i * targetLen, targetLen, NumOps);
            }

            // One-hot encode class
            if (classIndex >= 0 && classIndex < _numClasses)
                labelsData[i * _numClasses + classIndex] = NumOps.FromDouble(1.0);
        }

        LoadedFeatures = new Tensor<T>(featuresData, [totalSamples, targetLen]);
        LoadedLabels = new Tensor<T>(labelsData, [totalSamples, _numClasses]);
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
        return (AudioLoaderHelper.ExtractTensorBatch(features, indices),
                AudioLoaderHelper.ExtractTensorBatch(labels, indices));
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);
        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()),
                AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()),
                AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }
}
