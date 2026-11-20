using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Latest;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class OuteTTSIntegerTokenTests
{
    private static OuteTTS<Half> Model() => new(
        new NeuralNetworkArchitecture<Half>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1),
        new OuteTTSOptions
        {
            Backbone = OuteTTSBackbone.Llama1B,
            HiddenDim = 4, Layers = 1, Heads = 1, KeyValueHeads = 1, HeadDim = 4, Intermediate = 8,
            RopeScalingFactor = 0, MaxLength = 1, Temperature = 0,
            CodecEncoderDim = 2, CodecEncoderRates = new[] { 2 }, CodecLatentDim = 2,
            CodecDecoderDim = 4, CodecDecoderRates = new[] { 2 }, CodecCodebookDim = 2,
            AlignerMels = 80, AlignerWidth = 4, AlignerHeads = 1, AlignerEncoderLayers = 1, AlignerDecoderLayers = 1,
        });

    [Fact]
    public void IntegerTokenIds_KeepTheReleaseIdThroughDecodeAndEmbedding()
    {
        using var model = Model();
        model.SetTrainingMode(false);
        using var noGrad = new NoGradScope<Half>();

        var ids = model.TextToTokenIds("Hello");

        // OuteAI/Llama-OuteTTS-1.0-1B's published tokenizer encodes Hello as 9906; Half rounds it to 9904.
        Assert.Equal(new[] { 9906 }, ids.ToArray());
        Assert.Equal("Hello", model.TextFromTokenIds(ids));
        var embeddings = model.LanguageModel.Embeddings!;
        var parameters = new Vector<Half>(embeddings.GetParameters().Length);
        for (int d = 0; d < 4; d++)
        {
            parameters[9906 * 4 + d] = (Half)(d + 1);
            parameters[9904 * 4 + d] = (Half)(-d - 1);
        }
        embeddings.SetParameters(parameters);

        var actual = model.LanguageModel.Embed(new[,] { { ids[0] } });

        Assert.Equal(new[] { (Half)1, (Half)2, (Half)3, (Half)4 }, actual.ToArray());
        // The prompt already exceeds MaxLength, so the public native path embeds it but emits no codec tokens.
        Assert.Empty(model.PredictFromTokenIds(ids).ToArray());
    }

#if NET8_0_OR_GREATER
    [Fact]
    public void LegacyHalfTokens_RejectLossyIdsAndDirectCallersToIntegerInput()
    {
        using var model = Model();

        var error = Assert.Throws<NotSupportedException>(() => model.TextToTokens("Hello"));

        Assert.Contains("TextToTokenIds", error.Message);
        Assert.Contains("PredictFromTokenIds", error.Message);
        Assert.Equal(new[] { 9906 }, model.TextToTokenIds("Hello").ToArray());
    }
#endif

    [Fact]
    public void TrainingTranscript_PrefersExactTextOverRoundedNumericTokens()
    {
        using var model = Model();
        var rounded = new Tensor<Half>(new[] { (Half)9904 }, new[] { 1 });
        var sample = new TtsTrainingSample<Half> { Tokens = rounded, Text = "Hello" };

        string transcript = model.TrainingTranscript(sample);

        Assert.Equal("Hello", transcript);
        Assert.Equal(new[] { 9906 }, model.TextToTokenIds(transcript).ToArray());
        Assert.Equal("!", model.TrainingTranscript(new TtsTrainingSample<Half>
        {
            Tokens = new Tensor<Half>(new[] { (Half)0 }, new[] { 1 }),
        }));
    }

    [Fact]
    public void IntegerPrediction_RejectsEmptyAndOutOfVocabularyIdsBeforeInference()
    {
        using var model = Model();

        Assert.Throws<ArgumentException>(() => model.PredictFromTokenIds(new Tensor<int>(new[] { 0 })));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.PredictFromTokenIds(new Tensor<int>(new[] { -1 }, new[] { 1 })));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.PredictFromTokenIds(new Tensor<int>(new[] { 134400 }, new[] { 1 })));
    }
}
