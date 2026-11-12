using System;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class SparkOptionConstraintTests
{
    [Theory]
    [InlineData(1600, 10, 2, 320)]
    [InlineData(3200, 10, 4, 640)]
    [InlineData(1600, 20, 1, 160)]
    public void PairedWaveformGeometryControlsActualDecodedSampleCount(int sampleRate, int frameRate, int lastRate, int expected)
    {
        var options = SparkTTSPaperTests.SmallOptions();
        options.SampleRate = sampleRate;
        options.CodecFrameRate = frameRate;
        options.WaveDecoderRates[3] = lastRate;
        options.WaveDecoderKernels[3] = lastRate * 2;
        using var model = new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320), options);
        var global = model.EncodeGlobalTokens(SparkNativeFixture.Recording<double>());
        var semantic = new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new[] { 1.0, 2.0 }));
        var waveform = model.DecodeFromTokens(semantic, global);
        Assert.Equal(expected, waveform.Length);
        Assert.All(waveform, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }

    [Theory]
    [InlineData(3, 8)]
    [InlineData(4, 16)]
    public void PairedFsqLevelsAndVocabularyEncodeAndDecodeRealSpeakerTokens(int dimensions, int vocabulary)
    {
        var options = SparkTTSPaperTests.SmallOptions();
        options.FsqLevels = Enumerable.Repeat(2, dimensions).ToArray();
        options.GlobalCodebookSize = vocabulary;
        using var model = new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320), options);
        var global = model.EncodeGlobalTokens(SparkNativeFixture.Recording<double>());
        Assert.Equal(options.GlobalTokenCount, global.Length);
        Assert.All(global, code => Assert.InRange(code, 0, vocabulary - 1));
        var semantic = model.EncodeToTokens(SparkNativeFixture.Recording<double>());
        Assert.Equal(1, semantic.Rank);
        Assert.Equal(1, model.CodecTokenCodebooks);
        var waveform = model.DecodeFromTokens(semantic, global);
        Assert.NotEmpty(waveform);
        Assert.All(waveform, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }

    [Theory]
    [InlineData("NumCodebooks")]
    [InlineData("SampleRate")]
    [InlineData("CodecFrameRate")]
    [InlineData("FsqLevels")]
    [InlineData("GlobalCodebookSize")]
    public void UnpairedCodecChangesAreRejected(string option)
    {
        var options = SparkTTSPaperTests.SmallOptions();
        switch (option)
        {
            case "NumCodebooks": options.NumCodebooks = 2; break;
            case "SampleRate": options.SampleRate *= 2; break;
            case "CodecFrameRate": options.CodecFrameRate *= 2; break;
            case "FsqLevels": options.FsqLevels = new[] { 2, 2, 2 }; break;
            case "GlobalCodebookSize": options.GlobalCodebookSize = 8; break;
        }
        Assert.Throws<ArgumentException>(() => new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320), options));
    }

    [Fact]
    public void ExplicitInitializationSeedIsRepeatableAndIndependentOfSamplingSeed()
    {
        SparkTTS<double> Create(int seed, int samplingSeed)
        {
            var options = SparkTTSPaperTests.SmallOptions();
            options.Seed = seed;
            options.GenerationSeed = samplingSeed;
            return new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320)
            { RandomSeed = 2294 }, options);
        }
        using var first = Create(11, 2025);
        using var repeat = Create(11, 2026);
        using var changed = Create(12, 2025);
        Assert.Equal(first.GetParameters().ToArray(), repeat.GetParameters().ToArray());
        Assert.NotEqual(first.GetParameters().ToArray(), changed.GetParameters().ToArray());
    }

    [Theory]
    [InlineData(1, 0.75, 0)]
    [InlineData(0, 0.5, 0.5)]
    [InlineData(0, double.NaN, 1)]
    [InlineData(0, 0.5, 1.1)]
    public void InvalidDiscriminatorBandsAreRejectedBeforeTraining(double first, double middle, double last)
    {
        var options = SparkTTSPaperTests.SmallOptions();
        options.DiscriminatorBands = new[] { first, middle, last };
        Assert.Throws<ArgumentException>(() => new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320), options));
    }

    [Fact]
    public void ValidAlternativeBandsRunTheActualBiCodecTrainingObjective()
    {
        var options = SparkTTSPaperTests.SmallOptions(SparkTTSTrainingComponent.BiCodec);
        options.DiscriminatorBands = new[] { 0.0, 0.25, 0.5, 1.0 };
        using var model = new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320), options);
        double loss = model.Train(new TtsTrainingSample<double>
        {
            Text = "hi",
            Tokens = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 3.0, 4.0 })),
            Audio = SparkNativeFixture.Recording<double>(),
        });
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
    }
}
