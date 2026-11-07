using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SpeechRecognition;

/// <summary>
/// Native transcripts are rendered through each options class's Vocabulary (#2393). These models cast token ids to
/// characters instead, so with the vocabulary { blank, "x", "y", "z" } token 1 came out as U+0001, not "x", and the
/// Vocabulary option did nothing. ContextNet and RWKVTransducer were fixed the same way earlier.
/// </summary>
public class ConformerFamilyVocabularyTests
{
    private static readonly string[] Vocabulary = { "<blank>", "x", "y", "z" };

    private static NeuralNetworkArchitecture<double> Architecture() => new(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 16, outputSize: 4);

    private static ISpeechRecognizer<double> Create(string model) => model switch
    {
        nameof(CIFEncoder<double>) => new CIFEncoder<double>(Architecture(), new CIFEncoderOptions
        { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        nameof(ConformerCTC<double>) => new ConformerCTC<double>(Architecture(), new ConformerCTCOptions
        { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        nameof(ConformerTransducer<double>) => new ConformerTransducer<double>(Architecture(), new ConformerTransducerOptions
        { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, PredictionDim = 16, JointDim = 16, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        nameof(ConvTransformer<double>) => new ConvTransformer<double>(Architecture(), new ConvTransformerOptions
        { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardMultiplier = 2, NumDecoderLayers = 1, DecoderEmbeddingDim = 16, MaxTextLength = 4, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        nameof(EBranchformer<double>) => new EBranchformer<double>(Architecture(), new EBranchformerOptions
        { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, CgmlpDim = 32, AttentionDecoderLayers = 1, AttentionDecoderFeedForward = 32, MaxTextLength = 4, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        nameof(EfficientConformer<double>) => new EfficientConformer<double>(Architecture(), new EfficientConformerOptions
        { NumEncoderLayers = 3, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        nameof(Squeezeformer<double>) => new Squeezeformer<double>(Architecture(), new SqueezeformerOptions
        { EncoderDim = 16, NumEncoderLayers = 2, NumAttentionHeads = 2, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    private static Tensor<double> Audio()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        var audio = new Tensor<double>([16000]);
        for (int i = 0; i < audio.Length; i++) audio[i] = rng.NextDouble() - 0.5;
        return audio;
    }

    [Theory]
    [InlineData("CIFEncoder")]
    [InlineData("ConformerCTC")]
    [InlineData("ConformerTransducer")]
    [InlineData("ConvTransformer")]
    [InlineData("EBranchformer")]
    [InlineData("EfficientConformer")]
    [InlineData("Squeezeformer")]
    public void Transcript_IsSpelledWithTheConfiguredVocabulary(string model)
    {
        var recognizer = Create(model);
        string text = recognizer.Transcribe(Audio()).Text;

        // Untrained weights still emit non-blank tokens; an empty transcript would make the check below vacuous.
        Assert.False(string.IsNullOrWhiteSpace(text), $"{model} produced no tokens, so the vocabulary check proves nothing.");
        Assert.All(text, c => Assert.Contains(c, "xyz "));
    }
}
