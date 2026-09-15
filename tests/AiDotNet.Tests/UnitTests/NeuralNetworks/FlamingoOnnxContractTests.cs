using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using Xunit;
using EncoderKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.EncoderKind;
using OutputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.OutputKind;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The two-file ONNX constructor cannot run Flamingo (no Perceiver Resampler, no gated cross-attention), so it must
/// refuse construction instead of returning a model whose every inference call fails later.
/// </summary>
public sealed class FlamingoOnnxContractTests
{
    public FlamingoOnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TwoFileOnnxConstructionIsRejectedBeforeAnySessionOpens(bool dynamicInput)
    {
        // The fixture deletes its directory on dispose; an ONNX session left open on either file would make that
        // delete fail on Windows, so a clean dispose also proves nothing was loaded.
        using var fixture = new OnnxVisionLanguageFixture();
        string vision = fixture.WriteEncoder(EncoderKind.Image, dynamicInputs: dynamicInput, outputKind: OutputKind.FirstTokenEmbedding);
        string language = fixture.WriteEmbeddedLanguageModel();

        var error = Assert.Throws<NotSupportedException>(() => new FlamingoNeuralNetwork<float>(Architecture(),
            vision, language, ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options()));

        Assert.Contains("Perceiver Resampler", error.Message);
        Assert.Contains("cross-attention", error.Message);
        Assert.Contains("native", error.Message);
    }

    private static FlamingoOptions Options() => new() { ImageSize = 16, EmbeddingDimension = 4, VisionDim = 4, MaxSequenceLength = 8 };
    private static NeuralNetworkArchitecture<float> Architecture() => new(InputType.ThreeDimensional,
        NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);
}
