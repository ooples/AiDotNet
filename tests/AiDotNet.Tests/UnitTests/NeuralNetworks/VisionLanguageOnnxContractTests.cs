using System;
using System.Collections.Generic;
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

public sealed class VisionLanguageOnnxContractTests
{
    public VisionLanguageOnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    public enum BoundaryMismatch { Embedding, TextContext, FrameCount, ImageSize, TokenElementType }
    public enum NativeOnlyOption
    {
        PatchSize, VocabSize, VisionDim, TextHiddenDim, NumFrameEncoderLayers, NumTemporalLayers,
        NumTextLayers, NumHeads, TemporalAggregation, HiddenDim, NumEncoderLayers, VisionLayers
    }

    [Fact]
    public void Clip_ValidGraphsExecuteBothDataDependentEncoders()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new ClipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image), fixture.WriteEncoder(EncoderKind.Text), tokenizer,
            new ClipOptions { EmbeddingDimension = 4, MaxSequenceLength = 8, ImageSize = 16 });
        double[] pixels = Enumerable.Repeat(2.0, 3 * 16 * 16).ToArray();
        AssertNormalizedGraphResult(model.EncodeImage(pixels), pixels.Sum());
        // CLIP's current text API returns the unnormalized graph embedding.
        double tokenSum = tokenizer.Encode("a").TokenIds.Take(8).Sum(value => (double)value);
        var text = model.EncodeText("a");
        for (int index = 0; index < 4; index++) Assert.Equal(tokenSum + index + 1, text[index], 5);
    }

    [Fact]
    public void VideoClip_ValidGraphsExecuteBothDataDependentEncoders()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new VideoCLIPNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Video), fixture.WriteEncoder(EncoderKind.Text), tokenizer, Options());
        var frame = new Tensor<float>(new[] { 3, 16, 16 });
        for (int index = 0; index < frame.Length; index++) frame[index] = 2;
        AssertNormalizedGraphResult(model.GetVideoEmbedding(new[] { frame, frame }), 2.0 * frame.Length * 2);
        double tokenSum = tokenizer.Encode("a").TokenIds.Take(8).Sum(value => (double)value);
        AssertNormalizedGraphResult(model.GetTextEmbedding("a"), tokenSum);
    }

    [Theory]
    [InlineData(BoundaryMismatch.Embedding)]
    [InlineData(BoundaryMismatch.TextContext)]
    [InlineData(BoundaryMismatch.FrameCount)]
    [InlineData(BoundaryMismatch.ImageSize)]
    [InlineData(BoundaryMismatch.TokenElementType)]
    public void VideoClip_RejectsFixedGraphConflictsBeforeReturningAModel(BoundaryMismatch mismatch)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        string video = fixture.WriteEncoder(EncoderKind.Video,
            frames: mismatch == BoundaryMismatch.FrameCount ? 3 : 2,
            image: mismatch == BoundaryMismatch.ImageSize ? 32 : 16);
        string text = fixture.WriteEncoder(EncoderKind.Text,
            embedding: mismatch == BoundaryMismatch.Embedding ? 6 : 4,
            context: mismatch == BoundaryMismatch.TextContext ? 9 : 8,
            tokenType: mismatch == BoundaryMismatch.TokenElementType
                ? TensorProto.Types.DataType.Float : TensorProto.Types.DataType.Int64);
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var exception = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new VideoCLIPNeuralNetwork<float>(Architecture(), video, text, tokenizer, Options());
        });
        Assert.Contains("ONNX", exception.Message);
    }

    [Fact]
    public void Clip_RejectsAnEncoderEmbeddingWidthConflictAtConstruction()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var exception = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new ClipNeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Image), fixture.WriteEncoder(EncoderKind.Text, embedding: 6), tokenizer,
                new ClipOptions { EmbeddingDimension = 4, MaxSequenceLength = 8, ImageSize = 16 });
        });
        Assert.Contains(nameof(ClipOptions.EmbeddingDimension), exception.Message);
    }

    [Theory]
    [InlineData(NativeOnlyOption.PatchSize)]
    [InlineData(NativeOnlyOption.VocabSize)]
    [InlineData(NativeOnlyOption.VisionDim)]
    [InlineData(NativeOnlyOption.TextHiddenDim)]
    [InlineData(NativeOnlyOption.NumFrameEncoderLayers)]
    [InlineData(NativeOnlyOption.NumTemporalLayers)]
    [InlineData(NativeOnlyOption.NumTextLayers)]
    [InlineData(NativeOnlyOption.NumHeads)]
    [InlineData(NativeOnlyOption.TemporalAggregation)]
    [InlineData(NativeOnlyOption.HiddenDim)]
    [InlineData(NativeOnlyOption.NumEncoderLayers)]
    [InlineData(NativeOnlyOption.VisionLayers)]
    public void VideoClip_RejectsANativeOnlyTowerOverrideInsteadOfIgnoringIt(NativeOnlyOption property)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        switch (property)
        {
            case NativeOnlyOption.PatchSize: options.PatchSize = 8; break;
            case NativeOnlyOption.VocabSize: options.VocabSize = 128; break;
            case NativeOnlyOption.VisionDim: options.VisionDim = 64; break;
            case NativeOnlyOption.TextHiddenDim: options.TextHiddenDim = 64; break;
            case NativeOnlyOption.NumFrameEncoderLayers: options.NumFrameEncoderLayers = 2; break;
            case NativeOnlyOption.NumTemporalLayers: options.NumTemporalLayers = 2; break;
            case NativeOnlyOption.NumTextLayers: options.NumTextLayers = 2; break;
            case NativeOnlyOption.NumHeads: options.NumHeads = 2; break;
            case NativeOnlyOption.TemporalAggregation: options.TemporalAggregation = TemporalAggregationType.MeanPooling; break;
            case NativeOnlyOption.HiddenDim: options.HiddenDim = 64; break;
            case NativeOnlyOption.NumEncoderLayers: options.NumEncoderLayers = 2; break;
            case NativeOnlyOption.VisionLayers: options.VisionLayers = 2; break;
            default: throw new ArgumentOutOfRangeException(nameof(property));
        }
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var exception = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new VideoCLIPNeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Video), fixture.WriteEncoder(EncoderKind.Text), tokenizer, options);
        });
        Assert.Equal("options", exception.ParamName);
        Assert.Contains(property.ToString(), exception.Message);
    }

    [Fact]
    public void VideoClip_DoesNotInventNativeTowerGeometryInOnnxMetadata()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new VideoCLIPNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Video), fixture.WriteEncoder(EncoderKind.Text), tokenizer, Options());
        var metadata = model.GetModelMetadata();
        Assert.DoesNotContain("VisionHiddenDim", metadata.AdditionalInfo.Keys);
        Assert.DoesNotContain("NumTemporalLayers", metadata.AdditionalInfo.Keys);
    }

    [Fact]
    public void VideoClip_OnnxImageInputsDoNotInheritNativePatchConstraints()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.ImageSize = 4;
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new VideoCLIPNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Video, image: 4), fixture.WriteEncoder(EncoderKind.Text), tokenizer, options);
        var frame = new Tensor<float>(new[] { 3, 4, 4 });
        for (int index = 0; index < frame.Length; index++) frame[index] = 2;
        AssertNormalizedGraphResult(model.GetVideoEmbedding(new[] { frame, frame }), 2.0 * frame.Length * 2);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    public void Clip_DynamicOutputWidthCannotSilentlyTruncateOrZeroPad(int context)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new ClipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true, outputKind: OutputKind.TokenSequence), tokenizer,
            new ClipOptions { EmbeddingDimension = 4, MaxSequenceLength = context, ImageSize = 16 });
        var exception = Assert.Throws<InvalidOperationException>(() => model.EncodeText("a"));
        Assert.Contains(nameof(ClipOptions.EmbeddingDimension), exception.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BothWrappers_RejectAdditionalRequiredInputsBeforeInference(bool video)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var exception = Assert.Throws<ArgumentException>(() =>
        {
            string pixels = fixture.WriteEncoder(video ? EncoderKind.Video : EncoderKind.Image);
            string text = fixture.WriteEncoder(EncoderKind.Text, extraRequiredInput: true);
            using IDisposable model = video
                ? new VideoCLIPNeuralNetwork<float>(Architecture(), pixels, text, tokenizer, Options())
                : new ClipNeuralNetwork<float>(Architecture(), pixels, text, tokenizer, CreateClipOptions());
        });
        Assert.Contains("unsupported_required_input", exception.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BothWrappers_RejectBatchedOutputsInsteadOfTakingOnlyTheFirstExample(bool video)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var exception = Assert.Throws<ArgumentException>(() =>
        {
            string pixels = fixture.WriteEncoder(video ? EncoderKind.Video : EncoderKind.Image,
                outputKind: OutputKind.BatchedEmbedding);
            string text = fixture.WriteEncoder(EncoderKind.Text);
            using IDisposable model = video
                ? new VideoCLIPNeuralNetwork<float>(Architecture(), pixels, text, tokenizer, Options())
                : new ClipNeuralNetwork<float>(Architecture(), pixels, text, tokenizer, CreateClipOptions());
        });
        Assert.Contains("batch", exception.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Clip_DynamicInputAxesAndSupportedOutputLayoutsExecute(bool firstToken)
    {
        var outputKind = firstToken ? OutputKind.FirstTokenEmbedding : OutputKind.FixedEmbedding;
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new ClipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, dynamicInputs: true, outputKind: outputKind),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true, outputKind: outputKind), tokenizer, CreateClipOptions());
        var pixels = Enumerable.Repeat(2.0, 3 * 16 * 16).ToArray();
        AssertNormalizedGraphResult(model.EncodeImage(pixels), pixels.Sum());
        double sum = tokenizer.Encode("a").TokenIds.Take(8).Sum(value => (double)value);
        var text = model.EncodeText("a");
        for (int index = 0; index < 4; index++) Assert.Equal(sum + index + 1, text[index], 5);
    }

    [Fact]
    public void VideoClip_OnnxChannelOverrideControlsTheActualInputTensor()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var options = Options();
        options.Channels = 1;
        using var model = new VideoCLIPNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Video, channels: 1), fixture.WriteEncoder(EncoderKind.Text), tokenizer, options);
        var frame = new Tensor<float>(new[] { 1, 16, 16 });
        for (int index = 0; index < frame.Length; index++) frame[index] = 2;
        AssertNormalizedGraphResult(model.GetVideoEmbedding(new[] { frame, frame }), 2.0 * frame.Length * 2);
        Assert.Throws<ArgumentException>(() => model.GetVideoEmbedding(new[] { new Tensor<float>(new[] { 3, 16, 16 }) }));
    }

    [Fact]
    public void GraphConfigurationIsAnImmutableSnapshotSeparateFromMutableRequestedOptions()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var options = CreateClipOptions();
        using var model = new ClipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, dynamicInputs: true),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true), tokenizer, options);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        var image = configuration.Graphs[OnnxModelRole.ImageEncoder].Inputs["pixel_values"];
        Assert.Equal(new int?[] { 1, 3, null, null }, image.Dimensions);
        Assert.Equal("axis_2", image.SymbolicDimensions[2]);
        Assert.Equal(tokenizer.VocabularySize, configuration.TokenizerVocabularySize);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<OnnxModelRole, OnnxGraphSignature>)configuration.Graphs).Clear());
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, OnnxValueSignature>)configuration.Graphs[OnnxModelRole.ImageEncoder].Inputs).Clear());
        Assert.Throws<NotSupportedException>(() => ((IList<int?>)image.Dimensions)[0] = 9);
        Assert.Throws<NotSupportedException>(() => ((IList<string>)image.SymbolicDimensions)[2] = "changed");
        options.ImageSize = 99;
        options.EmbeddingDimension = 99;
        options.MaxSequenceLength = 99;
        Assert.Equal(16, configuration.ImageSize);
        Assert.Equal(4, configuration.EmbeddingDimension);
        Assert.Equal(8, configuration.MaxSequenceLength);
        AssertNormalizedGraphResult(model.EncodeImage(Enumerable.Repeat(2.0, 3 * 16 * 16).ToArray()), 2.0 * 3 * 16 * 16);
        Assert.Same(configuration, model.GetModelMetadata().AdditionalInfo[nameof(model.OnnxConfiguration)]);
    }

    [Fact]
    public void Clip_UnusedNonTensorOutputsRemainSupportedAndHonestlyDescribed()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new ClipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, auxiliarySequenceOutput: true),
            fixture.WriteEncoder(EncoderKind.Text, auxiliarySequenceOutput: true), tokenizer, CreateClipOptions());
        AssertNormalizedGraphResult(model.EncodeImage(Enumerable.Repeat(2.0, 3 * 16 * 16).ToArray()), 2.0 * 3 * 16 * 16);
        Assert.Equal(4, model.EncodeText("a").Length);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        var auxiliary = configuration.Graphs[OnnxModelRole.ImageEncoder].Outputs["auxiliary_sequence"];
        Assert.False(auxiliary.IsTensor);
        Assert.Null(auxiliary.ElementType);
        Assert.Empty(auxiliary.Dimensions);
        Assert.Empty(auxiliary.SymbolicDimensions);
    }

    private static ClipOptions CreateClipOptions() => new()
    {
        EmbeddingDimension = 4, MaxSequenceLength = 8, ImageSize = 16
    };

    private static VideoCLIPOptions Options() => new()
    {
        EmbeddingDimension = 4, MaxSequenceLength = 8, ImageSize = 16, NumFrames = 2
    };

    private static NeuralNetworkArchitecture<float> Architecture() => new(
        InputType.ThreeDimensional, NeuralNetworkTaskType.ImageClassification,
        inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);

    private static void AssertNormalizedGraphResult(Vector<float> actual, double sum)
    {
        Assert.Equal(4, actual.Length);
        double norm = Math.Sqrt(Enumerable.Range(1, 4).Sum(offset => (sum + offset) * (sum + offset)));
        for (int index = 0; index < 4; index++)
            Assert.InRange(Math.Abs(actual[index] - (sum + index + 1) / norm), 0, 1e-5);
    }
}
