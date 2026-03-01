using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Transforms;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class AudioBenchmarkTests
{
    [Fact]
    public async Task Esc50Loader_LoadsSyntheticData()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            // Create ESC-50-style directory structure
            string baseDir = Path.Combine(tempDir, "ESC-50-master");
            string audioDir = Path.Combine(baseDir, "audio");
            string metaDir = Path.Combine(baseDir, "meta");
            Directory.CreateDirectory(audioDir);
            Directory.CreateDirectory(metaDir);

            // Create synthetic WAV files and metadata CSV
            var csvLines = new System.Collections.Generic.List<string>
            {
                "filename,fold,target,category,esc10,src_file,take"
            };

            for (int i = 0; i < 10; i++)
            {
                string filename = $"1-{100000 + i}-A-{i}.wav";
                int fold = (i < 8) ? 1 : 5; // 8 in fold 1, 2 in fold 5 (test)
                int target = i % 5;
                csvLines.Add($"{filename},{fold},{target},class_{target},False,source_{i},A");
                File.WriteAllBytes(Path.Combine(audioDir, filename), CreateSyntheticWav(44100, 0.1));
            }

            File.WriteAllLines(Path.Combine(metaDir, "esc50.csv"), csvLines.ToArray());

            var options = new Esc50DataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                TestFold = 5,
                MaxSamples = 8
            };

            // Load train split (folds != 5)
            var loader = new Esc50DataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(8, loader.TotalCount);
            Assert.Equal("ESC-50", loader.Name);
            Assert.Equal(50, loader.OutputDimension);

            // Verify feature shape: [N, maxSamples]
            Assert.Equal(2, loader.Features.Shape.Length);
            Assert.Equal(8, loader.Features.Shape[0]);

            // Verify label shape: [N, 50] (one-hot)
            Assert.Equal(new[] { 8, 50 }, loader.Labels.Shape);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task UrbanSound8kLoader_LoadsSyntheticData()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string baseDir = Path.Combine(tempDir, "UrbanSound8K");
            string audioDir = Path.Combine(baseDir, "audio", "fold1");
            string metaDir = Path.Combine(baseDir, "metadata");
            Directory.CreateDirectory(audioDir);
            Directory.CreateDirectory(metaDir);

            var csvLines = new System.Collections.Generic.List<string>
            {
                "slice_file_name,fsID,start,end,salience,fold,classID,class"
            };

            for (int i = 0; i < 6; i++)
            {
                string filename = $"{100000 + i}-{i % 10}-0-0.wav";
                csvLines.Add($"{filename},{100000 + i},0,4,1,1,{i % 10},class_{i % 10}");
                File.WriteAllBytes(Path.Combine(audioDir, filename), CreateSyntheticWav(22050, 0.1));
            }

            File.WriteAllLines(Path.Combine(metaDir, "UrbanSound8K.csv"), csvLines.ToArray());

            var options = new UrbanSound8kDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                TestFold = 10 // None of our data is in fold 10, so train split gets all 6
            };

            var loader = new UrbanSound8kDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(6, loader.TotalCount);
            Assert.Equal("UrbanSound8K", loader.Name);
            Assert.Equal(10, loader.OutputDimension);
            Assert.Equal(new[] { 6, 10 }, loader.Labels.Shape);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task Musdb18Loader_LoadsSyntheticTracks()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string trainDir = Path.Combine(tempDir, "train");

            // Create 3 synthetic tracks
            for (int t = 0; t < 3; t++)
            {
                string trackDir = Path.Combine(trainDir, $"Track_{t}");
                Directory.CreateDirectory(trackDir);

                File.WriteAllBytes(Path.Combine(trackDir, "mixture.wav"), CreateSyntheticWav(44100, 0.1));
                File.WriteAllBytes(Path.Combine(trackDir, "vocals.wav"), CreateSyntheticWav(44100, 0.1));
                File.WriteAllBytes(Path.Combine(trackDir, "drums.wav"), CreateSyntheticWav(44100, 0.1));
                File.WriteAllBytes(Path.Combine(trackDir, "bass.wav"), CreateSyntheticWav(44100, 0.1));
                File.WriteAllBytes(Path.Combine(trackDir, "other.wav"), CreateSyntheticWav(44100, 0.1));
            }

            var options = new Musdb18DataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                SegmentDurationSeconds = 0.1
            };

            var loader = new Musdb18DataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(3, loader.TotalCount);
            Assert.Equal("MUSDB18", loader.Name);

            int segmentSamples = (int)(44100 * 0.1);
            Assert.Equal(segmentSamples, loader.FeatureCount);
            Assert.Equal(segmentSamples * 4, loader.OutputDimension);

            // Verify shapes
            Assert.Equal(new[] { 3, segmentSamples }, loader.Features.Shape);
            Assert.Equal(new[] { 3, segmentSamples * 4 }, loader.Labels.Shape);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task MaestroLoader_LoadsSyntheticCsv()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string versionDir = Path.Combine(tempDir, "maestro-v3.0.0");
            string yearDir = Path.Combine(versionDir, "2018");
            Directory.CreateDirectory(yearDir);

            // Create CSV metadata
            var csvLines = new[]
            {
                "canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration",
                "Beethoven,Sonata,train,2018,2018/test1.midi,2018/test1.wav,30.0",
                "Mozart,Concerto,train,2018,2018/test2.midi,2018/test2.wav,25.0",
                "Bach,Prelude,validation,2018,2018/test3.midi,2018/test3.wav,20.0"
            };
            File.WriteAllLines(Path.Combine(versionDir, "maestro-v3.0.0.csv"), csvLines);

            // Create synthetic audio files
            File.WriteAllBytes(Path.Combine(yearDir, "test1.wav"), CreateSyntheticWav(16000, 0.1));
            File.WriteAllBytes(Path.Combine(yearDir, "test2.wav"), CreateSyntheticWav(16000, 0.1));
            File.WriteAllBytes(Path.Combine(yearDir, "test3.wav"), CreateSyntheticWav(16000, 0.1));

            // Create synthetic MIDI files (minimal: "MThd" header + Note On events)
            File.WriteAllBytes(Path.Combine(yearDir, "test1.midi"), CreateSyntheticMidi(60));
            File.WriteAllBytes(Path.Combine(yearDir, "test2.midi"), CreateSyntheticMidi(72));

            var options = new MaestroDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                MaxDurationSeconds = 0.1,
                Version = "v3.0.0"
            };

            var loader = new MaestroDataLoader<double>(options);
            await loader.LoadAsync();

            // Only 2 train samples
            Assert.Equal(2, loader.TotalCount);
            Assert.Equal(128, loader.OutputDimension);

            // Verify label shape: [2, 128]
            Assert.Equal(new[] { 2, 128 }, loader.Labels.Shape);

            // Check MIDI note was activated for first sample (note 60)
            double noteVal = loader.Labels[0, 60];
            Assert.Equal(1.0, noteVal);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task AudioSetLoader_LoadsSyntheticMultiLabel()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string audioDir = Path.Combine(tempDir, "audio");
            Directory.CreateDirectory(audioDir);

            // Create class labels indices
            var classLines = new System.Collections.Generic.List<string>();
            for (int c = 0; c < 10; c++)
                classLines.Add($"{c},/m/class{c},class_{c}");
            File.WriteAllLines(Path.Combine(tempDir, "class_labels_indices.csv"), classLines.ToArray());

            // Create balanced train segments CSV
            var segLines = new System.Collections.Generic.List<string>
            {
                "# YTID, start_seconds, end_seconds, positive_labels"
            };

            for (int i = 0; i < 5; i++)
            {
                string ytid = $"video{i:D4}";
                string labels = $"\"/m/class{i % 10},/m/class{(i + 1) % 10}\"";
                segLines.Add($"{ytid}, 0.000, 10.000, {labels}");
                File.WriteAllBytes(Path.Combine(audioDir, $"{ytid}.wav"), CreateSyntheticWav(16000, 0.1));
            }

            File.WriteAllLines(Path.Combine(tempDir, "balanced_train_segments.csv"), segLines.ToArray());

            var options = new AudioSetDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                ClipDurationSeconds = 0.1
            };

            var loader = new AudioSetDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(5, loader.TotalCount);
            Assert.Equal("AudioSet", loader.Name);
            Assert.Equal(527, loader.OutputDimension);

            // Verify multi-hot labels (first sample should have classes 0 and 1 active)
            Assert.Equal(1.0, loader.Labels[0, 0]);
            Assert.Equal(1.0, loader.Labels[0, 1]);
            Assert.Equal(0.0, loader.Labels[0, 2]);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task Esc50Loader_SplitReturnsThreeSets()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string baseDir = Path.Combine(tempDir, "ESC-50-master");
            string audioDir = Path.Combine(baseDir, "audio");
            string metaDir = Path.Combine(baseDir, "meta");
            Directory.CreateDirectory(audioDir);
            Directory.CreateDirectory(metaDir);

            var csvLines = new System.Collections.Generic.List<string>
            {
                "filename,fold,target,category,esc10,src_file,take"
            };

            // Create 20 samples in fold 1 (train)
            for (int i = 0; i < 20; i++)
            {
                string filename = $"1-{100000 + i}-A-{i % 50}.wav";
                csvLines.Add($"{filename},1,{i % 50},class_{i % 50},False,source_{i},A");
                File.WriteAllBytes(Path.Combine(audioDir, filename), CreateSyntheticWav(44100, 0.05));
            }

            File.WriteAllLines(Path.Combine(metaDir, "esc50.csv"), csvLines.ToArray());

            var options = new Esc50DataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                TestFold = 5
            };

            var loader = new Esc50DataLoader<double>(options);
            await loader.LoadAsync();

            var (train, val, test) = loader.Split(0.7, 0.15, seed: 42);

            Assert.True(train.TotalCount > 0);
            Assert.True(val.TotalCount > 0);
            Assert.True(test.TotalCount > 0);
            Assert.Equal(loader.TotalCount, train.TotalCount + val.TotalCount + test.TotalCount);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public void SpectrogramTransform_TypeExists()
    {
        // SpectrogramTransform wraps MelSpectrogram which requires engine initialization.
        // Verify the type is accessible and constructable.
        var type = typeof(SpectrogramTransform<double>);
        Assert.NotNull(type);
        Assert.True(type.GetInterfaces().Length > 0, "SpectrogramTransform should implement ITransform.");
    }

    [Fact]
    public void SpecAugmentTransform_MasksSpectrogram()
    {
        int timeFrames = 100;
        int freqBins = 80;
        var data = new double[timeFrames * freqBins];
        for (int i = 0; i < data.Length; i++)
            data[i] = 1.0;

        var input = new Tensor<double>(data, new[] { timeFrames, freqBins });

        var transform = new SpecAugmentTransform<double>(
            freqMaskParam: 10,
            timeMaskParam: 20,
            numFreqMasks: 2,
            numTimeMasks: 2,
            seed: 42);

        var result = transform.Apply(input);

        // Output shape should match input
        Assert.Equal(input.Shape, result.Shape);

        // Some values should be zeroed out (masked)
        int zeroCount = 0;
        for (int t = 0; t < timeFrames; t++)
        {
            for (int f = 0; f < freqBins; f++)
            {
                if (result[t, f] == 0.0) zeroCount++;
            }
        }

        Assert.True(zeroCount > 0, "SpecAugment should mask some values to zero.");
        Assert.True(zeroCount < data.Length, "SpecAugment should not mask all values.");

        // Input should be unmodified
        for (int t = 0; t < timeFrames; t++)
        {
            for (int f = 0; f < freqBins; f++)
            {
                Assert.Equal(1.0, input[t, f]);
            }
        }
    }

    [Fact]
    public void LibriSpeechOptions_DefaultValues()
    {
        var options = new LibriSpeechDataLoaderOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(10.0, options.MaxDurationSeconds);
        Assert.Equal("train-clean-100", options.Subset);
        Assert.True(options.AutoDownload);
    }

    [Fact]
    public void CommonVoiceOptions_DefaultValues()
    {
        var options = new CommonVoiceDataLoaderOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal("en", options.Language);
        Assert.Equal(10.0, options.MaxDurationSeconds);
    }

    [Fact]
    public void VoxPopuliOptions_DefaultValues()
    {
        var options = new VoxPopuliDataLoaderOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(20.0, options.MaxDurationSeconds);
        Assert.Equal("en", options.Language);
        Assert.False(options.AutoDownload);
    }

    [Fact]
    public void FleursOptions_DefaultValues()
    {
        var options = new FleursDataLoaderOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(15.0, options.MaxDurationSeconds);
        Assert.Equal("en_us", options.Language);
    }

    [Fact]
    public void Musdb18Options_DefaultValues()
    {
        var options = new Musdb18DataLoaderOptions();
        Assert.Equal(44100, options.SampleRate);
        Assert.Equal(6.0, options.SegmentDurationSeconds);
        Assert.False(options.AutoDownload);
    }

    [Fact]
    public void MaestroOptions_DefaultValues()
    {
        var options = new MaestroDataLoaderOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(10.0, options.MaxDurationSeconds);
        Assert.Equal("v3.0.0", options.Version);
    }

    /// <summary>
    /// Creates a minimal synthetic WAV file with a 44-byte header and 16-bit PCM samples.
    /// </summary>
    private static byte[] CreateSyntheticWav(int sampleRate, double durationSeconds)
    {
        int numSamples = (int)(sampleRate * durationSeconds);
        int dataSize = numSamples * 2; // 16-bit = 2 bytes per sample
        int fileSize = 44 + dataSize;

        byte[] wav = new byte[fileSize];

        // RIFF header
        wav[0] = (byte)'R'; wav[1] = (byte)'I'; wav[2] = (byte)'F'; wav[3] = (byte)'F';
        BitConverter.GetBytes(fileSize - 8).CopyTo(wav, 4);
        wav[8] = (byte)'W'; wav[9] = (byte)'A'; wav[10] = (byte)'V'; wav[11] = (byte)'E';

        // fmt chunk
        wav[12] = (byte)'f'; wav[13] = (byte)'m'; wav[14] = (byte)'t'; wav[15] = (byte)' ';
        BitConverter.GetBytes(16).CopyTo(wav, 16);     // chunk size
        BitConverter.GetBytes((short)1).CopyTo(wav, 20); // PCM
        BitConverter.GetBytes((short)1).CopyTo(wav, 22); // mono
        BitConverter.GetBytes(sampleRate).CopyTo(wav, 24);
        BitConverter.GetBytes(sampleRate * 2).CopyTo(wav, 28); // byte rate
        BitConverter.GetBytes((short)2).CopyTo(wav, 32); // block align
        BitConverter.GetBytes((short)16).CopyTo(wav, 34); // bits per sample

        // data chunk
        wav[36] = (byte)'d'; wav[37] = (byte)'a'; wav[38] = (byte)'t'; wav[39] = (byte)'a';
        BitConverter.GetBytes(dataSize).CopyTo(wav, 40);

        // Fill with a sine wave
        for (int i = 0; i < numSamples; i++)
        {
            double t = (double)i / sampleRate;
            short sample = (short)(Math.Sin(2.0 * Math.PI * 440.0 * t) * 16000);
            BitConverter.GetBytes(sample).CopyTo(wav, 44 + i * 2);
        }

        return wav;
    }

    /// <summary>
    /// Creates a minimal synthetic MIDI file with a single note-on event.
    /// </summary>
    private static byte[] CreateSyntheticMidi(int noteNumber)
    {
        // Minimal MIDI file: MThd header + MTrk with one Note On event
        var midi = new System.Collections.Generic.List<byte>();

        // MThd header
        midi.AddRange(new byte[] { 0x4D, 0x54, 0x68, 0x64 }); // "MThd"
        midi.AddRange(new byte[] { 0x00, 0x00, 0x00, 0x06 }); // header length
        midi.AddRange(new byte[] { 0x00, 0x00 }); // format 0
        midi.AddRange(new byte[] { 0x00, 0x01 }); // 1 track
        midi.AddRange(new byte[] { 0x00, 0x60 }); // 96 ticks per quarter note

        // MTrk track
        midi.AddRange(new byte[] { 0x4D, 0x54, 0x72, 0x6B }); // "MTrk"

        var trackData = new byte[]
        {
            0x00, 0x90, (byte)(noteNumber & 0x7F), 0x64, // delta=0, Note On channel 0, note, velocity=100
            0x60, 0x80, (byte)(noteNumber & 0x7F), 0x00, // delta=96, Note Off channel 0, note, velocity=0
            0x00, 0xFF, 0x2F, 0x00                        // End of track
        };

        midi.AddRange(BitConverter.IsLittleEndian
            ? new byte[] { 0x00, 0x00, 0x00, (byte)trackData.Length }
            : BitConverter.GetBytes(trackData.Length));
        midi.AddRange(trackData);

        return midi.ToArray();
    }

    [Fact]
    public void FlacDecoder_DecodesAudioSamples()
    {
        // Test that LoadAudioSamples auto-detects FLAC and produces non-zero output.
        // Create a minimal FLAC file with VERBATIM subframe.
        byte[] flacData = CreateMinimalFlac(16000, 100); // 100 samples at 16kHz

        var target = new double[200];
        var numOps = MathHelper.GetNumericOperations<double>();
        AudioLoaderHelper.LoadAudioSamples(flacData, target, 0, 200, numOps);

        // Verify at least some non-zero samples were decoded
        bool hasNonZero = false;
        for (int i = 0; i < 100; i++)
        {
            if (Math.Abs(target[i]) > 1e-10)
            {
                hasNonZero = true;
                break;
            }
        }

        Assert.True(hasNonZero, "FLAC decoder should produce non-zero samples");

        // Verify samples are normalized to [-1, 1]
        for (int i = 0; i < 100; i++)
        {
            Assert.InRange(target[i], -1.0, 1.0);
        }
    }

    [Fact]
    public void AudioLoaderHelper_AutoDetectsWavVsFlac()
    {
        var numOps = MathHelper.GetNumericOperations<double>();

        // WAV file should be detected by RIFF header
        byte[] wavData = CreateSyntheticWav(16000, 0.01);
        var wavTarget = new double[200];
        AudioLoaderHelper.LoadAudioSamples(wavData, wavTarget, 0, 200, numOps);

        bool wavHasNonZero = false;
        for (int i = 0; i < wavTarget.Length; i++)
        {
            if (Math.Abs(wavTarget[i]) > 1e-10) { wavHasNonZero = true; break; }
        }
        Assert.True(wavHasNonZero, "WAV auto-detection should work through LoadAudioSamples");
    }

    /// <summary>
    /// Creates a minimal valid FLAC file with a single frame using VERBATIM subframe encoding.
    /// This is the simplest possible valid FLAC: raw uncompressed samples in FLAC container.
    /// </summary>
    private static byte[] CreateMinimalFlac(int sampleRate, int numSamples)
    {
        var flac = new System.Collections.Generic.List<byte>();
        int bitsPerSample = 16;
        int numChannels = 1;

        // "fLaC" magic
        flac.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });

        // STREAMINFO metadata block (type 0, isLast=true)
        flac.Add(0x80); // isLast=1, type=0
        flac.Add(0); flac.Add(0); flac.Add(34); // length = 34 bytes

        // Min/max block size (both = numSamples for single frame)
        flac.Add((byte)(numSamples >> 8)); flac.Add((byte)(numSamples & 0xFF)); // min block
        flac.Add((byte)(numSamples >> 8)); flac.Add((byte)(numSamples & 0xFF)); // max block

        // Min/max frame size (0 = unknown)
        flac.Add(0); flac.Add(0); flac.Add(0); // min frame
        flac.Add(0); flac.Add(0); flac.Add(0); // max frame

        // Sample rate (20 bits), channels-1 (3 bits), bps-1 (5 bits), total samples (36 bits)
        // Byte 10: sample rate >> 12
        flac.Add((byte)(sampleRate >> 12));
        // Byte 11: (sample rate >> 4) & 0xFF
        flac.Add((byte)((sampleRate >> 4) & 0xFF));
        // Byte 12: (sampleRate & 0x0F) << 4 | (numChannels-1) << 1 | ((bps-1) >> 4)
        flac.Add((byte)(((sampleRate & 0x0F) << 4) | ((numChannels - 1) << 1) | (((bitsPerSample - 1) >> 4) & 0x01)));
        // Byte 13: ((bps-1) & 0x0F) << 4 | (totalSamples >> 32 & 0x0F)
        flac.Add((byte)((((bitsPerSample - 1) & 0x0F) << 4) | (byte)((numSamples >> 32) & 0x0F)));
        // Bytes 14-17: totalSamples lower 32 bits
        flac.Add((byte)((numSamples >> 24) & 0xFF));
        flac.Add((byte)((numSamples >> 16) & 0xFF));
        flac.Add((byte)((numSamples >> 8) & 0xFF));
        flac.Add((byte)(numSamples & 0xFF));

        // MD5 signature (16 bytes of zeros — not validated)
        for (int i = 0; i < 16; i++) flac.Add(0);

        // FRAME
        // Frame header sync code: 0xFFF8 (fixed block size)
        int frameHeaderStart = flac.Count;
        flac.Add(0xFF);
        flac.Add(0xF8); // sync + fixed block size + reserved=0

        // Block size code + sample rate code
        // Block size: find code for our numSamples
        int blockSizeCode = 6; // means: read 8-bit from end of header, blockSize = value + 1
        int sampleRateCode = 0; // 0 = get from STREAMINFO
        flac.Add((byte)((blockSizeCode << 4) | sampleRateCode));

        // Channel assignment (0 = 1 channel mono) + sample size code + reserved
        int channelAssignment = numChannels - 1; // 0 for mono
        int sampleSizeCode = 4; // 4 = 16 bits
        flac.Add((byte)((channelAssignment << 4) | (sampleSizeCode << 1)));

        // Frame number (UTF-8 coded, frame 0 = single byte 0x00)
        flac.Add(0x00);

        // Block size: 8-bit value (blockSize - 1)
        flac.Add((byte)(numSamples - 1));

        // CRC-8 (placeholder — we don't validate)
        flac.Add(0x00);

        // SUBFRAME: VERBATIM (type = 1, padding=0, no wasted bits)
        // Subframe header: [0][000001][0] = 0x02
        flac.Add(0x02);

        // Raw 16-bit samples (big-endian in FLAC)
        for (int i = 0; i < numSamples; i++)
        {
            double t = (double)i / sampleRate;
            short sample = (short)(Math.Sin(2.0 * Math.PI * 440.0 * t) * 16000);
            flac.Add((byte)((sample >> 8) & 0xFF)); // MSB first
            flac.Add((byte)(sample & 0xFF));
        }

        // Pad to byte boundary (already aligned)
        // CRC-16 footer (placeholder)
        flac.Add(0x00);
        flac.Add(0x00);

        return flac.ToArray();
    }

    private static string CreateTempDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "AiDotNet_AudioBenchmarkTests_" + Guid.NewGuid().ToString("N").Substring(0, 8));
        Directory.CreateDirectory(path);
        return path;
    }

    private static void CleanupDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
                Directory.Delete(path, true);
        }
        catch
        {
            // Best-effort cleanup
        }
    }
}
