using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the MAESTRO piano performance dataset (~200 hours, aligned MIDI and audio).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// MAESTRO expects:
/// <code>
/// {DataPath}/maestro-{Version}/
///   maestro-{Version}.csv     (metadata CSV)
///   {year}/
///     MIDI-Unprocessed_*.wav
///     MIDI-Unprocessed_*.midi
/// </code>
/// CSV columns: canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration.
/// Features are raw waveform Tensor[N, MaxSamples]. Labels are MIDI note activations Tensor[N, 128] (one per MIDI note).
/// </para>
/// </remarks>
public class MaestroDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int MidiNotes = 128;

    private readonly MaestroDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => $"MAESTRO-{_options.Version}";
    /// <inheritdoc/>
    public override string Description => $"MAESTRO piano dataset ({_options.Version})";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => MidiNotes;

    /// <summary>Creates a new MAESTRO data loader.</summary>
    public MaestroDataLoader(MaestroDataLoaderOptions? options = null)
    {
        _options = options ?? new MaestroDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("maestro");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string versionDir = Path.Combine(_dataPath, $"maestro-{_options.Version}");
        if (!Directory.Exists(versionDir))
            versionDir = _dataPath;

        string csvFile = Path.Combine(versionDir, $"maestro-{_options.Version}.csv");
        if (!File.Exists(csvFile))
        {
            // Try without version in filename
            var csvCandidates = Directory.GetFiles(versionDir, "maestro*.csv");
            if (csvCandidates.Length > 0)
                csvFile = csvCandidates[0];
            else
            {
                throw new FileNotFoundException(
                    $"MAESTRO metadata not found at {csvFile}. " +
                    "Download from https://magenta.tensorflow.org/datasets/maestro.");
            }
        }

        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "validation",
            _ => "train"
        };

        var lines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string AudioPath, string MidiPath)>();

        // CSV: canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(new[] { ',' }, 7);
            if (parts.Length < 6) continue;

            string split = parts[2].Trim();
            if (!split.Equals(splitName, StringComparison.OrdinalIgnoreCase)) continue;

            string midiFile = parts[4].Trim();
            string audioFile = parts[5].Trim();

            string audioPath = Path.Combine(versionDir, audioFile);
            string midiPath = Path.Combine(versionDir, midiFile);

            if (File.Exists(audioPath))
                samples.Add((audioPath, midiPath));
        }

        if (samples.Count == 0)
        {
            // Fallback: collect WAV files directly
            var wavFiles = Directory.GetFiles(versionDir, "*.wav", SearchOption.AllDirectories);
            foreach (var wav in wavFiles)
                samples.Add((wav, string.Empty));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * _maxAudioSamples];
        var labelsData = new T[totalSamples * MidiNotes];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (audioPath, midiPath) = samples[i];

            // Load audio
            if (File.Exists(audioPath))
            {
                byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(audioPath, cancellationToken);
                AudioLoaderHelper.LoadWavSamples(audioBytes, featuresData, i * _maxAudioSamples, _maxAudioSamples, NumOps);
            }

            // Load MIDI note activations (simplified: extract active notes from MIDI file)
            if (midiPath.Length > 0 && File.Exists(midiPath))
            {
                byte[] midiBytes = await FilePolyfill.ReadAllBytesAsync(midiPath, cancellationToken);
                ExtractMidiNotes(midiBytes, labelsData, i * MidiNotes);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _maxAudioSamples });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, MidiNotes });
        InitializeIndices(totalSamples);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (AudioLoaderHelper.ExtractTensorBatch(features, indices), AudioLoaderHelper.ExtractTensorBatch(labels, indices));
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
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private void ExtractMidiNotes(byte[] midiBytes, T[] target, int offset)
    {
        // Simplified MIDI parsing: scan for Note On events (status 0x9n) and mark active notes
        // Real MIDI format: "MThd" header, then track chunks "MTrk"
        if (midiBytes.Length < 14) return;

        for (int b = 0; b < midiBytes.Length - 2; b++)
        {
            byte status = midiBytes[b];
            // Note On event: 0x90-0x9F
            if ((status & 0xF0) == 0x90)
            {
                int note = midiBytes[b + 1] & 0x7F;
                int velocity = midiBytes[b + 2] & 0x7F;
                if (velocity > 0 && note < MidiNotes)
                    target[offset + note] = NumOps.FromDouble(1.0);
            }
        }
    }
}
