using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio;

/// <summary>
/// Loads audio files from directories for audio classification and processing tasks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Supports WAV and raw PCM files. Audio is loaded as raw waveform samples
/// and can optionally be converted to mono and normalized.
/// </para>
/// <para><b>For Beginners:</b> Organize your audio files into class folders (like ImageFolder),
/// and this loader reads the raw waveforms into tensors for training.
/// </para>
/// </remarks>
public class AudioFileDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly AudioFileDatasetOptions _options;
    private int _sampleCount;
    private int _samplesPerFile;
    private int _numClasses;
    private List<string> _classNames = new List<string>();

    /// <inheritdoc/>
    public override string Name => "AudioFile";

    /// <inheritdoc/>
    public override string Description => $"Audio file dataset from {_options.RootDirectory}";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _samplesPerFile;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>
    /// Gets the class names discovered from directory names.
    /// </summary>
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>
    /// Creates a new AudioFileDataset with the specified options.
    /// </summary>
    public AudioFileDataset(AudioFileDatasetOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (string.IsNullOrWhiteSpace(options.RootDirectory))
        {
            throw new ArgumentException("Root directory cannot be empty.", nameof(options));
        }

        _samplesPerFile = (int)(options.SampleRate * options.DurationSeconds);
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string rootDir = _options.RootDirectory;
        if (!Directory.Exists(rootDir))
        {
            throw new DirectoryNotFoundException($"Audio folder root directory not found: {rootDir}");
        }

        var extensions = new HashSet<string>(
            _options.Extensions.Select(e => e.StartsWith(".", StringComparison.Ordinal) ? e : "." + e),
            StringComparer.OrdinalIgnoreCase);

        List<(string FilePath, int ClassIndex)> samples;

        if (_options.UseDirectoryLabels)
        {
            var classDirs = Directory.GetDirectories(rootDir)
                .OrderBy(d => Path.GetFileName(d), StringComparer.OrdinalIgnoreCase)
                .ToList();

            _classNames = classDirs
                .Select(d => Path.GetFileName(d) ?? string.Empty)
                .Where(n => n.Length > 0)
                .ToList();
            _numClasses = _classNames.Count;

            samples = new List<(string, int)>();
            for (int classIdx = 0; classIdx < classDirs.Count; classIdx++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var files = Directory.GetFiles(classDirs[classIdx], "*.*", SearchOption.TopDirectoryOnly)
                    .Where(f => extensions.Contains(Path.GetExtension(f)))
                    .ToList();

                foreach (var file in files)
                {
                    samples.Add((file, classIdx));
                }
            }
        }
        else
        {
            _classNames = new List<string> { "default" };
            _numClasses = 1;

            samples = Directory.GetFiles(rootDir, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => extensions.Contains(Path.GetExtension(f)))
                .Select(f => (f, 0))
                .ToList();
        }

        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < samples.Count)
        {
            var random = _options.RandomSeed.HasValue
                ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
                : RandomHelper.CreateSecureRandom();
            samples = samples.OrderBy(_ => random.Next()).Take(_options.MaxSamples.Value).ToList();
        }

        _sampleCount = samples.Count;
        if (_sampleCount == 0)
        {
            throw new InvalidOperationException("No audio files found matching the specified extensions.");
        }

        // Build tensors: features [N, SamplesPerFile], labels [N, NumClasses]
        var featuresData = new T[_sampleCount * _samplesPerFile];
        var labelsData = new T[_sampleCount * _numClasses];

        for (int i = 0; i < _sampleCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (filePath, classIndex) = samples[i];
            var audioSamples = LoadAudioSamples(filePath);

            int featureOffset = i * _samplesPerFile;
            int copyLen = Math.Min(audioSamples.Length, _samplesPerFile);
            Array.Copy(audioSamples, 0, featuresData, featureOffset, copyLen);

            int labelOffset = i * _numClasses;
            labelsData[labelOffset + classIndex] = NumOps.One;
        }

        var features = new Tensor<T>(featuresData, new[] { _sampleCount, _samplesPerFile });
        var labels = new Tensor<T>(labelsData, new[] { _sampleCount, _numClasses });

        LoadedFeatures = features;
        LoadedLabels = labels;
        InitializeIndices(_sampleCount);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
        _classNames = new List<string>();
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

        var batchFeatures = ExtractTensorBatch(features, indices);
        var batchLabels = ExtractTensorBatch(labels, indices);

        return (batchFeatures, batchLabels);
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
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        return (
            CreateSubset(shuffled.Take(trainSize).ToArray()),
            CreateSubset(shuffled.Skip(trainSize).Take(valSize).ToArray()),
            CreateSubset(shuffled.Skip(trainSize + valSize).ToArray())
        );
    }

    private InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSubset(int[] indices)
    {
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
            ExtractTensorBatch(LoadedFeatures!, indices),
            ExtractTensorBatch(LoadedLabels!, indices));
    }

    private T[] LoadAudioSamples(string filePath)
    {
        byte[] rawBytes = File.ReadAllBytes(filePath);
        var samples = new T[_samplesPerFile];

        string ext = Path.GetExtension(filePath);
        if (ext.Equals(".wav", StringComparison.OrdinalIgnoreCase))
        {
            LoadWavSamples(rawBytes, samples);
        }
        else
        {
            // Raw PCM: assume 16-bit signed little-endian
            LoadPcmSamples(rawBytes, 0, rawBytes.Length, 16, 1, samples);
        }

        return samples;
    }

    private void LoadWavSamples(byte[] rawBytes, T[] samples)
    {
        // Parse RIFF/WAVE header properly
        if (rawBytes.Length < 12)
            throw new InvalidDataException("File too small to be a valid WAV file.");

        // Check RIFF header
        string riffTag = System.Text.Encoding.ASCII.GetString(rawBytes, 0, 4);
        string waveTag = System.Text.Encoding.ASCII.GetString(rawBytes, 8, 4);
        if (riffTag != "RIFF" || waveTag != "WAVE")
            throw new InvalidDataException("Not a valid WAV file (missing RIFF/WAVE header).");

        // Walk through chunks to find 'fmt ' and 'data'
        int bitsPerSample = 16;
        int numChannels = 1;
        int dataOffset = -1;
        int dataSize = 0;

        int pos = 12;
        while (pos + 8 <= rawBytes.Length)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(rawBytes, pos, 4);
            int chunkSize = BitConverter.ToInt32(rawBytes, pos + 4);

            if (chunkId == "fmt ")
            {
                if (chunkSize >= 16 && pos + 8 + 16 <= rawBytes.Length)
                {
                    int audioFormat = BitConverter.ToInt16(rawBytes, pos + 8);
                    numChannels = BitConverter.ToInt16(rawBytes, pos + 10);
                    // int sampleRate = BitConverter.ToInt32(rawBytes, pos + 12);
                    // int byteRate = BitConverter.ToInt32(rawBytes, pos + 16);
                    // int blockAlign = BitConverter.ToInt16(rawBytes, pos + 20);
                    bitsPerSample = BitConverter.ToInt16(rawBytes, pos + 22);

                    if (audioFormat != 1) // PCM
                        throw new InvalidDataException(
                            $"Unsupported WAV format: {audioFormat}. Only PCM (format 1) is supported.");
                }
            }
            else if (chunkId == "data")
            {
                dataOffset = pos + 8;
                dataSize = chunkSize;
                break;
            }

            // Move to next chunk (chunks are word-aligned)
            pos += 8 + chunkSize;
            if (chunkSize % 2 != 0) pos++;
        }

        if (dataOffset < 0)
            throw new InvalidDataException("WAV file missing 'data' chunk.");

        LoadPcmSamples(rawBytes, dataOffset, dataSize, bitsPerSample, numChannels, samples);
    }

    private void LoadPcmSamples(byte[] rawBytes, int dataOffset, int dataSize, int bitsPerSample,
        int numChannels, T[] samples)
    {
        int bytesPerSample = bitsPerSample / 8;
        int bytesPerFrame = bytesPerSample * numChannels;

        if (bytesPerFrame == 0) return;

        int totalFrames = dataSize / bytesPerFrame;
        int framesToRead = Math.Min(totalFrames, _samplesPerFile);

        for (int i = 0; i < framesToRead; i++)
        {
            int frameOffset = dataOffset + i * bytesPerFrame;
            if (frameOffset + bytesPerSample > rawBytes.Length) break;

            double value;
            if (bitsPerSample == 8)
            {
                // 8-bit PCM is unsigned (0-255, centered at 128)
                value = rawBytes[frameOffset] - 128.0;
                if (_options.Normalize) value /= 128.0;
            }
            else if (bitsPerSample == 16)
            {
                short pcm16 = (short)(rawBytes[frameOffset] | (rawBytes[frameOffset + 1] << 8));
                value = pcm16;
                if (_options.Normalize) value /= 32768.0;
            }
            else if (bitsPerSample == 24)
            {
                int pcm24 = rawBytes[frameOffset] | (rawBytes[frameOffset + 1] << 8) | (rawBytes[frameOffset + 2] << 16);
                if ((pcm24 & 0x800000) != 0) pcm24 |= unchecked((int)0xFF000000); // Sign extend
                value = pcm24;
                if (_options.Normalize) value /= 8388608.0;
            }
            else if (bitsPerSample == 32)
            {
                int pcm32 = BitConverter.ToInt32(rawBytes, frameOffset);
                value = pcm32;
                if (_options.Normalize) value /= 2147483648.0;
            }
            else
            {
                throw new InvalidDataException($"Unsupported bits per sample: {bitsPerSample}. Supported: 8, 16, 24, 32.");
            }

            // If multi-channel and Mono requested, average all channels
            if (numChannels > 1 && _options.Mono)
            {
                double sum = value;
                for (int ch = 1; ch < numChannels; ch++)
                {
                    int chOffset = frameOffset + ch * bytesPerSample;
                    if (chOffset + bytesPerSample > rawBytes.Length) break;

                    if (bitsPerSample == 16)
                    {
                        short s = (short)(rawBytes[chOffset] | (rawBytes[chOffset + 1] << 8));
                        double chVal = s;
                        if (_options.Normalize) chVal /= 32768.0;
                        sum += chVal;
                    }
                    else if (bitsPerSample == 8)
                    {
                        double chVal = rawBytes[chOffset] - 128.0;
                        if (_options.Normalize) chVal /= 128.0;
                        sum += chVal;
                    }
                }
                value = sum / numChannels;
            }

            samples[i] = NumOps.FromDouble(value);
        }
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
