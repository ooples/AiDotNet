using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
        { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
        nameof(EBranchformer<double>) => new EBranchformer<double>(Architecture(), new EBranchformerOptions
        { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, CgmlpDim = 32, MergeDim = 16, NumMels = 16, VocabSize = 4, Vocabulary = Vocabulary, Seed = 3 }),
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
        // White-noise frames are alike, so an untrained CTC head can pick the blank at every one and the check below
        // would hold vacuously. Pushing the blank's bias down makes the greedy decoder emit a real token per frame.
        // The head builds its weights on the first forward, so transcribe once before reading them.
        _ = recognizer.Transcribe(Audio());
        if (recognizer is NeuralNetworkBase<double> network
            && network.Layers.LastOrDefault() is DenseLayer<double> head
            && head.GetBiases() is { } biases && biases.Length == Vocabulary.Length)
            biases[0] -= 10.0;
        string text = recognizer.Transcribe(Audio()).Text;

        Assert.False(string.IsNullOrWhiteSpace(text), $"{model} produced no tokens, so the vocabulary check proves nothing.");
        Assert.All(text, c => Assert.Contains(c, "xyz "));
    }
}
