using System.Collections.Concurrent;
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
/// <see cref="Features"/> access on this loader is intentionally not supported (features
/// are decoded lazily per batch). <see cref="Labels"/> is a small <c>[N, numClasses]</c>
/// one-hot tensor and is materialized during <c>LoadAsync</c>, so it remains available.
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

    // Internal backing arrays — NEVER expose these directly. The public API exposes
    // IReadOnlyList<string> views so callers can't mutate (or cast-then-mutate) the
    // canonical class ordering, which would corrupt directory → class-index mapping
    // for every loader instance.
    private static readonly string[] CoreWordsArray =
        ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"];
    private static readonly string[] AllWordsArray =
        ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
         "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
         "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
         "tree", "two", "up", "visual", "wow", "yes", "zero"];

    /// <summary>Core 10 spoken words (the standard benchmark subset). Read-only view.</summary>
    public static IReadOnlyList<string> CoreWords { get; } = Array.AsReadOnly(CoreWordsArray);

    /// <summary>All 35 word classes in the full dataset. Read-only view.</summary>
    public static IReadOnlyList<string> AllWords { get; } = Array.AsReadOnly(AllWordsArray);

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
    private readonly IReadOnlyList<string> _wordListReadOnly; // cached AsReadOnly wrapper
    private readonly int _silenceClassIndex; // -1 when not in 12-class core mode
    private readonly int _unknownClassIndex; // -1 when not in 12-class core mode

    // Per-sample metadata captured during LoadDataCoreAsync. The actual audio is
    // decoded per-batch inside ExtractBatch — we never materialize all waveforms in
    // memory, which is what makes the full 35-class path safe on commodity hardware.
    private readonly List<SampleEntry> _samples = new();
    private List<string>? _backgroundNoiseFiles; // null until LoadDataCoreAsync runs

    // Lazy cache of fully-decoded background-noise waveforms, keyed by file path.
    // Each entry holds the full native-rate decoded audio as a T[] so synthetic
    // silence samples can be sliced at distinct offsets without reloading from disk.
    // ConcurrentDictionary because ExtractBatch is called from parallel batch
    // prefetchers and multi-threaded input pipelines — a plain Dictionary would
    // race between concurrent silence-decode calls.
    private readonly ConcurrentDictionary<string, T[]> _backgroundNoiseBuffers = new();

    // How many native samples we pre-decode from each background-noise WAV. Speech
    // Commands background files are ~60 s at 16 kHz; 30 s is more than enough to
    // give us many distinct 1-second crops per file while keeping the cache small
    // (~1.8 MB per file for float32).
    private const int BackgroundNoisePrefetchSamples = 30 * SpeechCommandsDataLoaderOptions.NativeSampleRate;

    // Step between consecutive silence crop offsets, in native-rate samples.
    // 0.5 s gives plenty of variety (60 unique crops per 30-second buffer) while
    // still leaving the crops partially overlapping — the benchmark expects
    // neighboring noise windows to look similar, not i.i.d.
    private const int SilenceCropHopNativeSamples = SpeechCommandsDataLoaderOptions.NativeSampleRate / 2;

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
    /// <remarks>
    /// Wrapped via <see cref="Array.AsReadOnly{T}(T[])"/> so callers can't downcast back to
    /// <c>string[]</c> and mutate the canonical class ordering — that would corrupt the
    /// directory→class-index mapping for every loader instance. The wrapper is cached in
    /// the constructor so repeated <see cref="WordList"/> access doesn't re-allocate.
    /// </remarks>
    public IReadOnlyList<string> WordList => _wordListReadOnly;

    /// <summary>Creates a new Speech Commands data loader.</summary>
    public SpeechCommandsDataLoader(SpeechCommandsDataLoaderOptions? options = null)
    {
        _options = options ?? new SpeechCommandsDataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("speech_commands");

        if (_options.UseCoreSubset)
        {
            // Per Warden 2018: 10 keywords + silence + unknown = 12 classes.
            // Use the internal array directly to avoid paying for ReadOnlyCollection
            // enumerator overhead on a hot path that runs once per loader.
            int coreLen = CoreWordsArray.Length;
            _wordList = new string[coreLen + 2];
            Array.Copy(CoreWordsArray, _wordList, coreLen);
            _wordList[coreLen] = SilenceLabel;
            _wordList[coreLen + 1] = UnknownLabel;
            _silenceClassIndex = coreLen;
            _unknownClassIndex = coreLen + 1;
        }
        else
        {
            // Clone so an external caller can't mutate our internal AllWordsArray via
            // any side-channel that might cast _wordList back to string[].
            _wordList = (string[])AllWordsArray.Clone();
            _silenceClassIndex = -1;
            _unknownClassIndex = -1;
        }
        _numClasses = _wordList.Length;
        _wordListReadOnly = Array.AsReadOnly(_wordList);
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
        int keywordEnd = _options.UseCoreSubset ? CoreWordsArray.Length : _wordList.Length;
        for (int c = 0; c < keywordEnd; c++)
        {
            string word = _wordList[c];
            string wordDir = Path.Combine(_dataPath, word);
            if (!Directory.Exists(wordDir)) continue;

            // EnumerateFiles so the `break` below can short-circuit the directory
            // scan once MaxSamplesPerClass is hit; GetFiles would materialize the
            // entire file list up-front (tens of thousands of paths for some classes)
            // regardless of how many we actually keep.
            foreach (string wavFile in Directory.EnumerateFiles(wordDir, "*.wav"))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!IsInRequestedSplit(wavFile, word, testFiles, valFiles))
                    continue;

                if (_options.MaxSamplesPerClass.HasValue &&
                    classCounts[c] >= _options.MaxSamplesPerClass.Value)
                {
                    // Cap reached for this class — stop scanning the rest of the
                    // directory.
                    break;
                }

                _samples.Add(new SampleEntry(wavFile, c, SampleSource.WavFile));
                classCounts[c]++;
            }
        }

        // 3b) Synthetic _unknown_ class: collapse every non-core word directory.
        if (_options.UseCoreSubset && _unknownClassIndex >= 0)
        {
            var coreSet = new HashSet<string>(CoreWordsArray, StringComparer.OrdinalIgnoreCase);
            foreach (string nonCoreWord in AllWordsArray)
            {
                if (coreSet.Contains(nonCoreWord)) continue;
                // Cap already hit from a prior non-core word — stop enumerating
                // remaining directories entirely rather than opening each one just to
                // immediately `break` out of its inner loop.
                if (_options.MaxSamplesPerClass.HasValue &&
                    classCounts[_unknownClassIndex] >= _options.MaxSamplesPerClass.Value)
                    break;

                string wordDir = Path.Combine(_dataPath, nonCoreWord);
                if (!Directory.Exists(wordDir)) continue;

                // EnumerateFiles so the cap-reached break can short-circuit, same
                // rationale as the keyword loop above.
                foreach (string wavFile in Directory.EnumerateFiles(wordDir, "*.wav"))
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

        // 3c) Synthetic _silence_ class: split-aware crops of the background-noise
        // WAVs per Warden 2018 §6 (Speech Commands v2):
        //  - Train draws overlapping 1-second crops from all background files except
        //    the one reserved for validation.
        //  - Validation uses a single designated file (running_tap.wav by convention)
        //    so the val set stays speaker/noise-source-independent.
        //  - Test draws from the remaining background files (same pool as train but
        //    with a proportionally smaller sample count).
        // The generated silence count is scaled by the split proportion so silence
        // stays at ~8.3% of samples across splits (the 12-class balanced setting
        // from the TFDS reference implementation). Previously the same
        // SilenceSampleCount was emitted regardless of Split, producing test/val
        // distributions that skewed heavily toward silence.
        if (_options.UseCoreSubset && _silenceClassIndex >= 0 && _options.SilenceSampleCount > 0)
        {
            string bgDir = Path.Combine(_dataPath, BackgroundNoiseDir);
            var allBgFiles = Directory.Exists(bgDir)
                ? Directory.GetFiles(bgDir, "*.wav").ToList()
                : new List<string>();

            // Fail fast if the whole _background_noise_ pool is missing — silently
            // emitting zero-filled silence would turn an incomplete dataset into a
            // different benchmark without the caller noticing. The per-split pool
            // can still fall back to the full set when one designated file is
            // missing (handled below); this check only guards the all-empty case.
            if (allBgFiles.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Speech Commands core mode requires at least one WAV under '{bgDir}' " +
                    "to synthesize '_silence_' samples. Re-download the dataset or disable " +
                    "the silence class by setting SilenceSampleCount to 0.");
            }

            const string ValidationBackgroundFile = "running_tap.wav";
            bool IsValidationBg(string path) => string.Equals(
                Path.GetFileName(path), ValidationBackgroundFile, StringComparison.OrdinalIgnoreCase);

            _backgroundNoiseFiles = _options.Split switch
            {
                DatasetSplit.Validation => allBgFiles.Where(IsValidationBg).ToList(),
                _                       => allBgFiles.Where(f => !IsValidationBg(f)).ToList(),
            };
            // If the designated per-split pool is empty (e.g., running_tap.wav is
            // missing from the archive), fall back to the full background file list
            // so we emit silence samples instead of collapsing the class to size 0.
            if (_backgroundNoiseFiles.Count == 0) _backgroundNoiseFiles = allBgFiles;

            // Split sizes from Warden 2018 §4 (TFDS reference, 12-class benchmark):
            // train ≈ 85,511, val ≈ 10,102, test ≈ 4,890. Scale silence per split to
            // preserve the target class ratio across splits.
            double splitScale = _options.Split switch
            {
                DatasetSplit.Validation => 10102.0 / 85511.0,
                DatasetSplit.Test       => 4890.0 / 85511.0,
                _                       => 1.0,
            };
            int silenceCount = (int)Math.Round(_options.SilenceSampleCount * splitScale);
            if (_options.MaxSamplesPerClass.HasValue)
                silenceCount = Math.Min(silenceCount, _options.MaxSamplesPerClass.Value);

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
        // Decoded background-noise clips can be ~30s of float32 audio each. Clearing
        // the cache here ensures Unload() actually releases the largest buffers and
        // prevents stale audio from being reused if the dataset changes on disk.
        _backgroundNoiseBuffers.Clear();
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
            if (sampleIdx < 0 || sampleIdx >= _samples.Count)
            {
                // Fail loud on an out-of-range index — silently skipping would leave
                // this batch row zero-filled while the corresponding label row is
                // whatever random class happens to sit at index i in the label
                // tensor. That's a data/logic bug we never want to mask.
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Index {sampleIdx} at position {i} is outside the valid range [0, {_samples.Count - 1}].");
            }

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
            // _silence_ class. Slice a distinct crop of one of the background-noise
            // WAVs using a deterministic per-sample offset (derived from SyntheticIndex)
            // so that consecutive silence samples are NOT duplicate prefixes of the
            // same waveform. Normal loading always populates _backgroundNoiseFiles
            // (LoadDataCoreAsync fails fast when _background_noise_/ is empty) — the
            // zero-fill branch below is only a defensive fallback for states where
            // no silence entries could have been emitted in the first place.
            if (_backgroundNoiseFiles is { Count: > 0 })
            {
                int fileIdx = entry.SyntheticIndex % _backgroundNoiseFiles.Count;
                string path = _backgroundNoiseFiles[fileIdx];
                if (!File.Exists(path))
                {
                    // Expected synthesized-silence source file is gone — surface this
                    // explicitly rather than silently emitting a zero-filled row that
                    // would pose as a legitimate silence label.
                    throw new FileNotFoundException(
                        $"Expected background-noise WAV was not found while decoding a synthetic _silence_ sample: '{path}'.",
                        path);
                }

                var noiseBuffer = GetOrLoadBackgroundNoiseBuffer(path);

                // Derive a deterministic native-rate crop offset so each SyntheticIndex
                // produces a DIFFERENT 1-second window. Using integer division by the
                // file count means the first `files.Count` silence samples each pick
                // offset 0 (one per file), the next cohort picks offset
                // SilenceCropHopNativeSamples, and so on.
                int nativeWindow = checked((int)Math.Ceiling((double)targetLen * nativeRate / targetRate)) + 1;
                int cropOffsetNative = 0;
                if (noiseBuffer.Length > nativeWindow)
                {
                    int maxOffset = noiseBuffer.Length - nativeWindow;
                    // Modulo (maxOffset + 1) — maxOffset is itself a *valid* starting
                    // offset, so the cohort range must be inclusive of it. Using
                    // `% maxOffset` excluded the last valid window and biased the
                    // sampler away from the tail of each background file.
                    int offsetRange = maxOffset + 1;
                    int cohort = entry.SyntheticIndex / _backgroundNoiseFiles.Count;
                    cropOffsetNative = (cohort * SilenceCropHopNativeSamples) % offsetRange;
                }

                ResampleIntoDestination(noiseBuffer, cropOffsetNative, dst, dstOffset, targetLen, nativeRate, targetRate);
            }
            // Else: dst[dstOffset..dstOffset+targetLen] stays at default(T) which is 0.
            return;
        }

        if (!File.Exists(entry.AudioPath))
        {
            // Loader scanned this file during LoadAsync but it disappeared before
            // ExtractBatch reached it. Silent zero-fill would mislabel the row as
            // "real class + silence waveform" — throw instead so the data issue is
            // visible to the caller.
            throw new FileNotFoundException(
                $"Expected Speech Commands WAV file was not found during batch decoding: '{entry.AudioPath}'. " +
                "The file may have been removed or renamed after LoadAsync scanned the dataset.",
                entry.AudioPath);
        }
        byte[] audioBytes = File.ReadAllBytes(entry.AudioPath);
        DecodeBytesWithResample(audioBytes, dst, dstOffset, targetLen, nativeRate, targetRate);
    }

    /// <summary>
    /// Returns a cached fully-decoded waveform for the given background-noise file,
    /// loading and caching it on first access. All synthetic _silence_ samples for a
    /// particular background file share this buffer so per-sample crop offsets can be
    /// sliced without repeated file I/O or WAV decoding.
    /// </summary>
    private T[] GetOrLoadBackgroundNoiseBuffer(string path)
    {
        // GetOrAdd is atomic across concurrent callers; the value factory may run
        // more than once for the same key under heavy contention, but only one
        // result is installed. That's acceptable here — duplicate decode work is
        // bounded (a few MB) and cheaper than holding a lock across file I/O.
        return _backgroundNoiseBuffers.GetOrAdd(path, p =>
        {
            byte[] audioBytes = File.ReadAllBytes(p);
            var buffer = new T[BackgroundNoisePrefetchSamples];
            AudioLoaderHelper.LoadAudioSamples(audioBytes, buffer, 0, BackgroundNoisePrefetchSamples, NumOps);
            return buffer;
        });
    }

    /// <summary>
    /// Decodes a WAV byte payload, then writes <paramref name="targetLen"/> samples
    /// into <paramref name="dst"/> starting at <paramref name="dstOffset"/>, resampling
    /// from <paramref name="nativeRate"/> to <paramref name="targetRate"/> via linear
    /// interpolation when the rates differ.
    /// </summary>
    private void DecodeBytesWithResample(
        byte[] audioBytes, T[] dst, int dstOffset, int targetLen, int nativeRate, int targetRate)
    {
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
        ResampleIntoDestination(nativeBuf, nativeOffset: 0, dst, dstOffset, targetLen, nativeRate, targetRate);
    }

    /// <summary>
    /// Linear-interpolation resample of <paramref name="nativeBuf"/> starting at
    /// <paramref name="nativeOffset"/> into <paramref name="dst"/>[<paramref name="dstOffset"/>..].
    /// When <paramref name="nativeRate"/> equals <paramref name="targetRate"/> this
    /// collapses to a plain copy loop (still honors nativeOffset for silence crops).
    /// </summary>
    private void ResampleIntoDestination(
        T[] nativeBuf, int nativeOffset, T[] dst, int dstOffset, int targetLen, int nativeRate, int targetRate)
    {
        int lastIdx = nativeBuf.Length - 1;
        if (nativeRate == targetRate)
        {
            // No resampling — still respect the offset (used for silence crops).
            for (int t = 0; t < targetLen; t++)
            {
                int idx = nativeOffset + t;
                if (idx > lastIdx) idx = lastIdx;
                dst[dstOffset + t] = nativeBuf[idx];
            }
            return;
        }

        double rateRatio = (double)nativeRate / targetRate;
        for (int t = 0; t < targetLen; t++)
        {
            double srcPos = nativeOffset + t * rateRatio;
            int floorIdx = (int)Math.Floor(srcPos);
            // frac = srcPos - floor(srcPos); reuse floorIdx to avoid a second
            // Math.Floor call per sample in this hot resampling loop.
            double frac = srcPos - floorIdx;
            int ceilIdx = floorIdx + 1;
            if (floorIdx > lastIdx) floorIdx = lastIdx;
            if (ceilIdx > lastIdx) ceilIdx = lastIdx;

            double v0 = NumOps.ToDouble(nativeBuf[floorIdx]);
            double v1 = NumOps.ToDouble(nativeBuf[ceilIdx]);
            dst[dstOffset + t] = NumOps.FromDouble(v0 * (1.0 - frac) + v1 * frac);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Speech Commands is a streaming loader — see class remarks. Splitting the full
    /// dataset via this base-class API would require eagerly decoding every clip into
    /// a contiguous <c>Tensor&lt;T&gt;</c>, which is ~4 GB of float32 features for the
    /// 35-class mode (or ~500 MB for a 70 % train split). That would defeat the whole
    /// reason the loader went streaming in the first place, and would quietly OOM on
    /// commodity hardware. Use the dataset's official Train/Validation/Test splits
    /// via the <c>DatasetSplit</c> option + one loader per split, or iterate
    /// <see cref="GetBatches"/>/<see cref="GetBatchesAsync"/> directly and partition
    /// indices yourself.
    /// </remarks>
    /// <exception cref="NotSupportedException">Always thrown — see remarks.</exception>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        throw new NotSupportedException(
            "SpeechCommandsDataLoader is a streaming loader — Split() is not supported because " +
            "it would eagerly decode and materialize every clip into memory (~4 GB for the full " +
            "35-class dataset), defeating the point of streaming. " +
            "Instead: construct separate loaders with SpeechCommandsDataLoaderOptions.Split set to " +
            "Train / Validation / Test (which honours the official Warden 2018 split lists), or " +
            "iterate GetBatches()/GetBatchesAsync() and partition indices yourself.");
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
