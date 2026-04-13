using AiDotNet.Data;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the Google Speech Commands v2 dataset (~65K clips, 35 words, 16kHz).
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
///   _background_noise_/      (used to synthesize the _silence_ class)
///     ...
///   validation_list.txt
///   testing_list.txt
/// </code>
/// Each subdirectory is a word class. WAV files are 16-bit PCM mono at 16kHz, ~1 second each.
/// </para>
/// <para>
/// <b>Streaming behaviour:</b> the loader does NOT preload audio waveforms into memory.
/// File paths and class indices are scanned eagerly during <c>LoadAsync</c>, but the
/// actual WAV decoding (and resampling, if configured) happens per-batch in
/// <see cref="ExtractBatch"/>. This is intentional: the full 35-class dataset would
/// allocate ~4 GB of float32 features in memory, so eager loading was guaranteed to OOM.
/// Use <see cref="GetBatches"/> / <see cref="GetBatchesAsync"/> for training; direct
/// <see cref="Features"/>/<see cref="Labels"/> access on this loader is intentionally
/// not supported.
/// </para>
/// <para>
/// <b>Class scheme:</b> the "core" 12-class subset (default) follows Warden 2018:
/// the 10 keyword classes (yes, no, up, down, left, right, on, off, stop, go) plus
/// <c>_silence_</c> (sampled from <c>_background_noise_/</c>) and <c>_unknown_</c>
/// (collapses every non-core word directory). The full 35-class mode uses all word
/// directories with no synthetic classes.
/// </para>
/// <para>
/// <b>Auto-download:</b> when <see cref="SpeechCommandsDataLoaderOptions.AutoDownload"/>
/// is true (default), the loader fetches and extracts the tarball from
/// <see cref="DownloadUrl"/> if the data directory is empty, mirroring
/// <c>LibriSpeechDataLoader</c>.
/// </para>
/// </remarks>
public class SpeechCommandsDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    /// <summary>HTTPS URL of the Speech Commands v2 archive.</summary>
    public static readonly string DownloadUrl =
        "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz";

    /// <summary>Core 10 spoken words (the standard benchmark subset).</summary>
    public static readonly string[] CoreWords =
        ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"];

    /// <summary>All 35 word classes in the full dataset.</summary>
    public static readonly string[] AllWords =
        ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
         "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
         "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
         "tree", "two", "up", "visual", "wow", "yes", "zero"];

    /// <summary>Synthetic class label for non-keyword speech in the 12-class subset.</summary>
    public const string UnknownLabel = "_unknown_";

    /// <summary>Synthetic class label for background-noise / silence in the 12-class subset.</summary>
    public const string SilenceLabel = "_silence_";

    /// <summary>Subdirectory containing the long background-noise WAVs distributed with the dataset.</summary>
    public const string BackgroundNoiseDir = "_background_noise_";

    private readonly SpeechCommandsDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private readonly int _numClasses;
    private readonly string[] _wordList;
    private readonly int _silenceClassIndex; // -1 when not in 12-class core mode
    private readonly int _unknownClassIndex; // -1 when not in 12-class core mode

    // Per-sample metadata captured during LoadDataCoreAsync. The actual audio is
    // decoded per-batch inside ExtractBatch — we never materialize all waveforms in
    // memory, which is what makes the full 35-class path safe on commodity hardware.
    private readonly List<SampleEntry> _samples = new();
    private List<string>? _backgroundNoiseFiles; // null until LoadDataCoreAsync runs

    /// <inheritdoc/>
    public override string Name => "SpeechCommands";

    /// <inheritdoc/>
    public override string Description => _options.UseCoreSubset
        ? "Google Speech Commands v2 (core 12 classes: 10 keywords + silence + unknown)"
        : "Google Speech Commands v2 (all 35 words)";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _options.TargetLength;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Number of word classes (12 in core mode, 35 in full mode).</summary>
    public int NumClasses => _numClasses;

    /// <summary>Gets the word list being used (10/35 word names plus the two synthetic labels in core mode).</summary>
    public IReadOnlyList<string> WordList => _wordList;

    /// <summary>Creates a new Speech Commands data loader.</summary>
    public SpeechCommandsDataLoader(SpeechCommandsDataLoaderOptions? options = null)
    {
        _options = options ?? new SpeechCommandsDataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("speech_commands");

        if (_options.UseCoreSubset)
        {
            // Per Warden 2018: 10 keywords + silence + unknown = 12 classes.
            _wordList = new string[CoreWords.Length + 2];
            Array.Copy(CoreWords, _wordList, CoreWords.Length);
            _wordList[CoreWords.Length] = SilenceLabel;
            _wordList[CoreWords.Length + 1] = UnknownLabel;
            _silenceClassIndex = CoreWords.Length;
            _unknownClassIndex = CoreWords.Length + 1;
        }
        else
        {
            _wordList = AllWords;
            _silenceClassIndex = -1;
            _unknownClassIndex = -1;
        }
        _numClasses = _wordList.Length;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // 1) Make sure the dataset is on disk. Mirrors LibriSpeechDataLoader.
        await EnsureDatasetPresentAsync(cancellationToken);

        var testingListPath = Path.Combine(_dataPath, "testing_list.txt");
        var validationListPath = Path.Combine(_dataPath, "validation_list.txt");

        // 2) Load official split lists if available.
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

        // 3) Build the per-class entry list (paths only, NO audio decoding here).
        _samples.Clear();
        var classCounts = new int[_numClasses];

        // 3a) Keyword classes (10 in core mode, 35 in full mode).
        int keywordEnd = _options.UseCoreSubset ? CoreWords.Length : _wordList.Length;
        for (int c = 0; c < keywordEnd; c++)
        {
            string word = _wordList[c];
            string wordDir = Path.Combine(_dataPath, word);
            if (!Directory.Exists(wordDir)) continue;

            foreach (string wavFile in Directory.GetFiles(wordDir, "*.wav"))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!IsInRequestedSplit(wavFile, word, testFiles, valFiles))
                    continue;

                if (_options.MaxSamplesPerClass.HasValue &&
                    classCounts[c] >= _options.MaxSamplesPerClass.Value)
                    continue;

                _samples.Add(new SampleEntry(wavFile, c, SampleSource.WavFile));
                classCounts[c]++;
            }
        }

        // 3b) Synthetic _unknown_ class: collapse every non-core word directory.
        if (_options.UseCoreSubset && _unknownClassIndex >= 0)
        {
            var coreSet = new HashSet<string>(CoreWords, StringComparer.OrdinalIgnoreCase);
            foreach (string nonCoreWord in AllWords)
            {
                if (coreSet.Contains(nonCoreWord)) continue;
                string wordDir = Path.Combine(_dataPath, nonCoreWord);
                if (!Directory.Exists(wordDir)) continue;

                foreach (string wavFile in Directory.GetFiles(wordDir, "*.wav"))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!IsInRequestedSplit(wavFile, nonCoreWord, testFiles, valFiles))
                        continue;

                    if (_options.MaxSamplesPerClass.HasValue &&
                        classCounts[_unknownClassIndex] >= _options.MaxSamplesPerClass.Value)
                        break;

                    _samples.Add(new SampleEntry(wavFile, _unknownClassIndex, SampleSource.WavFile));
                    classCounts[_unknownClassIndex]++;
                }
            }
        }

        // 3c) Synthetic _silence_ class: random crops of the background-noise WAVs,
        // or pure silence when the directory is missing (degenerate but never crashes).
        if (_options.UseCoreSubset && _silenceClassIndex >= 0 && _options.SilenceSampleCount > 0)
        {
            string bgDir = Path.Combine(_dataPath, BackgroundNoiseDir);
            _backgroundNoiseFiles = Directory.Exists(bgDir)
                ? Directory.GetFiles(bgDir, "*.wav").ToList()
                : new List<string>();

            int silenceCount = _options.MaxSamplesPerClass.HasValue
                ? Math.Min(_options.SilenceSampleCount, _options.MaxSamplesPerClass.Value)
                : _options.SilenceSampleCount;
            for (int i = 0; i < silenceCount; i++)
            {
                _samples.Add(new SampleEntry(string.Empty, _silenceClassIndex, SampleSource.Silence, syntheticIndex: i));
                classCounts[_silenceClassIndex]++;
            }
        }

        _sampleCount = _samples.Count;
        if (_sampleCount == 0)
        {
            throw new InvalidOperationException(
                $"No Speech Commands data found at {_dataPath}. " +
                $"Set AutoDownload=true (the default) or download manually from {DownloadUrl}.");
        }

        // 4) Pre-compute the labels tensor (cheap, small — N * NumClasses).
        var labelsData = new T[_sampleCount * _numClasses];
        for (int i = 0; i < _sampleCount; i++)
            labelsData[i * _numClasses + _samples[i].ClassIndex] = NumOps.FromDouble(1.0);
        LoadedLabels = new Tensor<T>(labelsData, [_sampleCount, _numClasses]);

        // 5) Features tensor stays null — see the class remarks. Audio is decoded
        // lazily per-batch in ExtractBatch.
        LoadedFeatures = default;

        InitializeIndices(_sampleCount);
    }

    /// <summary>
    /// Decides whether a given WAV path falls into the requested split based on the
    /// official testing_list.txt / validation_list.txt files (Warden 2018 spec).
    /// </summary>
    private bool IsInRequestedSplit(string wavFile, string word, HashSet<string> testFiles, HashSet<string> valFiles)
    {
        string relPath = Path.Combine(word, Path.GetFileName(wavFile));
        bool isTest = testFiles.Contains(relPath);
        bool isVal = valFiles.Contains(relPath);
        bool isTrain = !isTest && !isVal;

        return _options.Split switch
        {
            DatasetSplit.Test => isTest,
            DatasetSplit.Validation => isVal,
            _ => isTrain
        };
    }

    /// <summary>
    /// Triggers <see cref="DatasetDownloader.DownloadAndExtractTarGzAsync"/> when the data
    /// directory is empty AND <see cref="SpeechCommandsDataLoaderOptions.AutoDownload"/>
    /// is enabled. Throws a clear error otherwise so users aren't stuck wondering why
    /// LoadAsync returns "no data found".
    /// </summary>
    private async Task EnsureDatasetPresentAsync(CancellationToken cancellationToken)
    {
        bool hasAnyData = Directory.Exists(_dataPath) && Directory.EnumerateFileSystemEntries(_dataPath).Any();
        if (hasAnyData) return;

        if (!_options.AutoDownload)
        {
            throw new InvalidOperationException(
                $"No Speech Commands data found at {_dataPath} and AutoDownload is disabled. " +
                $"Either set AutoDownload=true or manually download from {DownloadUrl} and extract to {_dataPath}.");
        }

        await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _samples.Clear();
        _backgroundNoiseFiles = null;
        _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        // Streaming path: decode WAV (and resample if configured) for just this batch.
        // Total memory is O(batchSize * targetLength), independent of dataset size.
        int targetLen = _options.TargetLength;
        var featuresData = new T[indices.Length * targetLen];

        for (int i = 0; i < indices.Length; i++)
        {
            int sampleIdx = indices[i];
            if (sampleIdx < 0 || sampleIdx >= _samples.Count) continue;

            var entry = _samples[sampleIdx];
            DecodeSampleInto(entry, featuresData, i * targetLen, targetLen);
        }
        var features = new Tensor<T>(featuresData, [indices.Length, targetLen]);

        var allLabels = LoadedLabels ?? throw new InvalidOperationException(
            "Labels not loaded — call LoadAsync() before ExtractBatch().");
        var labels = AudioLoaderHelper.ExtractTensorBatch(allLabels, indices);
        return (features, labels);
    }

    /// <summary>
    /// Decodes a single sample (real WAV or synthesized silence) into the destination
    /// span at the given offset, applying resampling when the configured target rate
    /// differs from the dataset's native 16kHz rate.
    /// </summary>
    private void DecodeSampleInto(SampleEntry entry, T[] dst, int dstOffset, int targetLen)
    {
        int nativeRate = SpeechCommandsDataLoaderOptions.NativeSampleRate;
        int targetRate = _options.SampleRate;

        if (entry.Source == SampleSource.Silence)
        {
            // _silence_ class. If background noise files are available, sample a
            // random crop of one of them; otherwise leave the slot zero-filled
            // (acceptable degenerate behavior — pure silence is still valid input).
            if (_backgroundNoiseFiles is { Count: > 0 })
            {
                // Deterministic per-sample picker so reload + reshuffle yields stable
                // training data (the dataset class index alone disambiguates samples,
                // and we want the test fixture to be reproducible).
                int fileIdx = entry.SyntheticIndex % _backgroundNoiseFiles.Count;
                string path = _backgroundNoiseFiles[fileIdx];
                if (File.Exists(path))
                {
                    DecodeFileWithResample(path, dst, dstOffset, targetLen, nativeRate, targetRate);
                }
            }
            // Else: dst[dstOffset..dstOffset+targetLen] stays at default(T) which is 0.
            return;
        }

        if (!File.Exists(entry.AudioPath)) return;
        DecodeFileWithResample(entry.AudioPath, dst, dstOffset, targetLen, nativeRate, targetRate);
    }

    /// <summary>
    /// Decodes a WAV file and writes <paramref name="targetLen"/> samples into
    /// <paramref name="dst"/> starting at <paramref name="dstOffset"/>, resampling
    /// from <paramref name="nativeRate"/> to <paramref name="targetRate"/> via
    /// linear interpolation when the rates differ.
    /// </summary>
    private void DecodeFileWithResample(
        string path, T[] dst, int dstOffset, int targetLen, int nativeRate, int targetRate)
    {
        byte[] audioBytes = File.ReadAllBytes(path);

        if (nativeRate == targetRate)
        {
            // Fast path: no resampling, decode straight into the destination buffer.
            AudioLoaderHelper.LoadAudioSamples(audioBytes, dst, dstOffset, targetLen, NumOps);
            return;
        }

        // Resample: load into a native-rate temp buffer sized for `targetLen` of output.
        // We need ceil(targetLen * nativeRate / targetRate) native samples to cover
        // the full target window, plus +1 slack for the linear-interp ceiling lookup.
        int neededNative = checked((int)Math.Ceiling((double)targetLen * nativeRate / targetRate)) + 1;
        var nativeBuf = new T[neededNative];
        AudioLoaderHelper.LoadAudioSamples(audioBytes, nativeBuf, 0, neededNative, NumOps);

        double rateRatio = (double)nativeRate / targetRate;
        int lastIdx = neededNative - 1;
        for (int t = 0; t < targetLen; t++)
        {
            double srcPos = t * rateRatio;
            int floorIdx = (int)Math.Floor(srcPos);
            int ceilIdx = floorIdx + 1;
            if (floorIdx > lastIdx) floorIdx = lastIdx;
            if (ceilIdx > lastIdx) ceilIdx = lastIdx;

            double frac = srcPos - Math.Floor(srcPos);
            double v0 = NumOps.ToDouble(nativeBuf[floorIdx]);
            double v1 = NumOps.ToDouble(nativeBuf[ceilIdx]);
            dst[dstOffset + t] = NumOps.FromDouble(v0 * (1.0 - frac) + v1 * frac);
        }
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

        var trainIdx = shuffled.Take(trainSize).ToArray();
        var valIdx = shuffled.Skip(trainSize).Take(valSize).ToArray();
        var testIdx = shuffled.Skip(trainSize + valSize).ToArray();

        // Each split eagerly materializes its own (smaller) batch — this is fine for
        // typical 70/15/15 splits because the per-split memory is bounded by
        // splitSize * targetLength, which is much smaller than the full-dataset case.
        var trainBatch = ExtractBatch(trainIdx);
        var valBatch = ExtractBatch(valIdx);
        var testBatch = ExtractBatch(testIdx);

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(trainBatch.Features, trainBatch.Labels),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(valBatch.Features, valBatch.Labels),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(testBatch.Features, testBatch.Labels)
        );
    }

    /// <summary>What kind of source a particular sample entry references.</summary>
    private enum SampleSource
    {
        WavFile,
        Silence,
    }

    /// <summary>
    /// Compact per-sample record stored in <see cref="_samples"/>. Holds only
    /// metadata — no audio bytes — so the full 35-class manifest fits comfortably
    /// in memory and ExtractBatch decodes on demand.
    /// </summary>
    private readonly struct SampleEntry
    {
        public string AudioPath { get; }
        public int ClassIndex { get; }
        public SampleSource Source { get; }
        public int SyntheticIndex { get; }

        public SampleEntry(string audioPath, int classIndex, SampleSource source, int syntheticIndex = 0)
        {
            AudioPath = audioPath;
            ClassIndex = classIndex;
            Source = source;
            SyntheticIndex = syntheticIndex;
        }
    }
}
