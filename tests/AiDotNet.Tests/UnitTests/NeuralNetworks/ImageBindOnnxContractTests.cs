using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Onnx.Protobuf;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using Xunit;
using EncoderKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.EncoderKind;
using OutputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.OutputKind;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class ImageBindOnnxContractTests
{
    public enum NativeSetting { HiddenDim, NumEncoderLayers, NumHeads, PatchSize, VocabSize, Channels, AudioSampleRate, AudioMaxDuration, ImuTimesteps, NumVideoFrames }
    public enum GraphConflict { ImageWidth, TextWidth, AudioWidth, ImageSize, TextContext, AudioBins, TokenType, RequiredInput, ImageBatch, TextBatch, AudioBatch }
    public enum ImageGeometry { WrongRank, BatchTwo, WrongChannels, WrongSize }

    public ImageBindOnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(NativeSetting.HiddenDim)]
    [InlineData(NativeSetting.NumEncoderLayers)]
    [InlineData(NativeSetting.NumHeads)]
    [InlineData(NativeSetting.PatchSize)]
    [InlineData(NativeSetting.VocabSize)]
    [InlineData(NativeSetting.Channels)]
    [InlineData(NativeSetting.AudioSampleRate)]
    [InlineData(NativeSetting.AudioMaxDuration)]
    [InlineData(NativeSetting.ImuTimesteps)]
    [InlineData(NativeSetting.NumVideoFrames)]
    public void NativeOnlyOverridesAreRejectedRatherThanIgnored(NativeSetting setting)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        switch (setting)
        {
            case NativeSetting.HiddenDim: options.HiddenDim = 16; break;
            case NativeSetting.NumEncoderLayers: options.NumEncoderLayers = 2; break;
            case NativeSetting.NumHeads: options.NumHeads = 2; break;
            case NativeSetting.PatchSize: options.PatchSize = 8; break;
            case NativeSetting.VocabSize: options.VocabSize = 512; break;
            case NativeSetting.Channels: options.Channels = 4; break;
            case NativeSetting.AudioSampleRate: options.AudioSampleRate = 8000; break;
            case NativeSetting.AudioMaxDuration: options.AudioMaxDuration = 3; break;
            case NativeSetting.ImuTimesteps: options.ImuTimesteps = 100; break;
            case NativeSetting.NumVideoFrames: options.NumVideoFrames = 4; break;
            default: throw new ArgumentOutOfRangeException(nameof(setting));
        }
        var error = Assert.Throws<ArgumentException>(() => { using var model = Create(fixture, options); });
        Assert.Contains(setting.ToString(), error.Message);
    }

    [Theory]
    [InlineData(GraphConflict.ImageWidth)]
    [InlineData(GraphConflict.TextWidth)]
    [InlineData(GraphConflict.AudioWidth)]
    [InlineData(GraphConflict.ImageSize)]
    [InlineData(GraphConflict.TextContext)]
    [InlineData(GraphConflict.AudioBins)]
    [InlineData(GraphConflict.TokenType)]
    [InlineData(GraphConflict.RequiredInput)]
    [InlineData(GraphConflict.ImageBatch)]
    [InlineData(GraphConflict.TextBatch)]
    [InlineData(GraphConflict.AudioBatch)]
    public void FixedGraphConflictsAreRejectedDuringConstruction(GraphConflict conflict)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new ImageBindNeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Image, embedding: conflict == GraphConflict.ImageWidth ? 6 : 4,
                    image: conflict == GraphConflict.ImageSize ? 32 : 16,
                    outputKind: conflict == GraphConflict.ImageBatch ? OutputKind.BatchedTokenFeatures : OutputKind.FixedEmbedding),
                fixture.WriteEncoder(EncoderKind.Text, embedding: conflict == GraphConflict.TextWidth ? 6 : 4,
                    context: conflict == GraphConflict.TextContext ? 9 : 8,
                    tokenType: conflict == GraphConflict.TokenType ? TensorProto.Types.DataType.Float : TensorProto.Types.DataType.Int64,
                    extraRequiredInput: conflict == GraphConflict.RequiredInput,
                    outputKind: conflict == GraphConflict.TextBatch ? OutputKind.BatchedEmbedding : OutputKind.FixedEmbedding),
                fixture.WriteEncoder(EncoderKind.Audio, embedding: conflict == GraphConflict.AudioWidth ? 6 : 4,
                    audioBins: conflict == GraphConflict.AudioBins ? 64 : 128,
                    outputKind: conflict == GraphConflict.AudioBatch ? OutputKind.BatchedTokenFeatures : OutputKind.FixedEmbedding),
                ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options());
        });
        Assert.Contains("ONNX", error.Message);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ActualThreeGraphEmbeddingsPreserveExistingHostPreprocessing(bool dynamicInputs, bool tokenFeatures)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs, tokenFeatures);
        using var image = new Tensor<float>(new[] { 3, 16, 16 });
        image.Fill(2f);
        AssertNormalized(model.GetImageEmbedding(image), image.Length * 2);
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        AssertNormalized(model.GetTextEmbedding("a"), tokenizer.Encode("a").TokenIds.Take(8).Sum());

        // This verifies the wrapper's existing energy-feature input, NOT a real mel transform
        // or pretrained audio quality. Both frames see a constant amplitude-two waveform.
        using var waveform = new Tensor<float>(new[] { 320 });
        waveform.Fill(2f);
        double featureSum = 2 * Enumerable.Range(0, 128).Sum(bin =>
            (double)(float)(Math.Log(4 + 1e-10) * (1 - bin / 128.0 * 0.5)));
        AssertNormalized(model.GetAudioEmbedding(waveform), featureSum);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.Equal(3, configuration.Graphs.Count);
        Assert.True(configuration.Graphs.ContainsKey(OnnxModelRole.AudioEncoder));
        var metadata = model.GetModelMetadata();
        Assert.Same(configuration, metadata.AdditionalInfo[nameof(model.OnnxConfiguration)]);
        Assert.False(metadata.AdditionalInfo.ContainsKey("HiddenDimension"));
        Assert.False(metadata.AdditionalInfo.ContainsKey("VocabularySize"));
    }

    [Fact]
    public void OnnxImageGeometryDoesNotRequireANativePatch()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.ImageSize = 7;
        using var model = Create(fixture, options);
        using var image = new Tensor<float>(new[] { 3, 7, 7 });
        image.Fill(2f);
        AssertNormalized(model.GetImageEmbedding(image), image.Length * 2);
    }

    [Fact]
    public void SymbolicTextOutputWidthCannotBeTruncated()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = new ImageBindNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true, outputKind: OutputKind.TokenSequence),
            fixture.WriteEncoder(EncoderKind.Audio),
            ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options());
        var error = Assert.Throws<InvalidOperationException>(() => model.GetTextEmbedding("a"));
        Assert.Contains("embedding width 8", error.Message);
    }

    [Theory]
    [InlineData(ImageGeometry.WrongRank)]
    [InlineData(ImageGeometry.BatchTwo)]
    [InlineData(ImageGeometry.WrongChannels)]
    [InlineData(ImageGeometry.WrongSize)]
    public void SymbolicImageGraphStillEnforcesTheHostGeometry(ImageGeometry geometry)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: true);
        int[] shape = geometry switch
        {
            ImageGeometry.WrongRank => new[] { 16, 16 },
            ImageGeometry.BatchTwo => new[] { 2, 3, 16, 16 },
            ImageGeometry.WrongChannels => new[] { 4, 16, 16 },
            ImageGeometry.WrongSize => new[] { 3, 15, 16 },
            _ => throw new ArgumentOutOfRangeException(nameof(geometry))
        };
        using var image = new Tensor<float>(shape);
        var error = Assert.Throws<ArgumentException>(() => model.GetImageEmbedding(image));
        Assert.Equal("image", error.ParamName);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AudioTimeAxisIsCheckedAgainstTheActualWaveform(bool dynamicInput)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: dynamicInput);
        // The fixed fixture accepts two feature frames; 480 samples produce three.
        using var waveform = new Tensor<float>(new[] { 480 });
        waveform.Fill(2f);
        if (dynamicInput)
        {
            double featureSum = 3 * Enumerable.Range(0, 128).Sum(bin =>
                (double)(float)(Math.Log(4 + 1e-10) * (1 - bin / 128.0 * 0.5)));
            AssertNormalized(model.GetAudioEmbedding(waveform), featureSum);
        }
        else
        {
            var error = Assert.Throws<ArgumentException>(() => model.GetAudioEmbedding(waveform));
            Assert.Contains("axis 3", error.Message);
        }
    }

    private static void AssertNormalized(Vector<float> actual, double sum)
    {
        Assert.Equal(4, actual.Length);
        double norm = Math.Sqrt(Enumerable.Range(1, 4).Sum(offset => (sum + offset) * (sum + offset)));
        for (int index = 0; index < 4; index++) Assert.Equal((sum + index + 1) / norm, actual[index], 4);
    }

    private static ImageBindNeuralNetwork<float> Create(OnnxVisionLanguageFixture fixture, ImageBindOptions options,
        bool dynamicInputs = false, bool tokenFeatures = false)
    {
        var output = tokenFeatures ? OutputKind.FirstTokenEmbedding : OutputKind.FixedEmbedding;
        return new ImageBindNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: options.ImageSize, dynamicInputs: dynamicInputs, outputKind: output),
            fixture.WriteEncoder(EncoderKind.Text, context: options.MaxSequenceLength, dynamicInputs: dynamicInputs, outputKind: output),
            fixture.WriteEncoder(EncoderKind.Audio, dynamicInputs: dynamicInputs, outputKind: output),
            ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), options);
    }

    private static ImageBindOptions Options() => new() { ImageSize = 16, EmbeddingDimension = 4, MaxSequenceLength = 8 };
    private static NeuralNetworkArchitecture<float> Architecture() => new(InputType.ThreeDimensional,
        NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);
}
