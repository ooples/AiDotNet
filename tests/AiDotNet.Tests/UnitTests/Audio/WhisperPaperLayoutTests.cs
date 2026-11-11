using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// Whisper's layers as the reference implementation builds them (openai/whisper, model.py): a convolutional stem,
/// pre-norm encoder and decoder blocks whose attention biases the query and value projections but not the key, a
/// causal decoder self-attention, learned decoder positions and an output head tied to the token embedding. The
/// whisper-tiny instance has 37,184,640 trainable parameters, transformers' count without its frozen encoder positions.
/// </summary>
public class WhisperPaperLayoutTests
{
    private const int Width = 32, Heads = 4, Ffn = 64, Mels = 8, Vocab = 100, Tokens = 8;

    private static List<ILayer<double>> Layers() => LayerHelper<double>.CreateDefaultWhisperLayers(
        modelDim: Width, numEncoderLayers: 1, numDecoderLayers: 1, numHeads: Heads, ffDim: Ffn, numMels: Mels,
        maxFrames: 20, maxTokens: Tokens, vocabSize: Vocab, dropoutRate: 0.0).ToList();

    [Fact]
    public void DecoderBlocks_ArePreNormAndCausal_WithQueryAndValueBiases()
    {
        var decoder = Assert.Single(Layers().OfType<TransformerDecoderBlock<double>>());
        var selfAttention = Assert.IsType<MultiHeadAttentionLayer<double>>(decoder.SelfAttentionLayer);

        Assert.True(selfAttention.UseCausalMask);
        Assert.True(selfAttention.HasProjectionBiases);
        Assert.True(decoder.CrossAttentionLayer.HasProjectionBiases);
        Assert.IsType<GELUActivation<double>>(((DenseLayer<double>)decoder.FfnUpLayer).ScalarActivation);
    }

    [Fact]
    public void EncoderAttention_HasQueryAndValueBiases()
    {
        var encoder = Assert.Single(Layers().OfType<TransformerEncoderBlock<double>>());

        Assert.True(Assert.IsType<MultiHeadAttentionLayer<double>>(encoder.AttentionLayer).HasProjectionBiases);
    }

    [Fact]
    public void ProjectionBiases_AddTwoVectors_AndChangeTheOutput()
    {
        var plain = new MultiHeadAttentionLayer<double>(Heads, Width / Heads);
        var biased = new MultiHeadAttentionLayer<double>(Heads, Width / Heads).ConfigureProjectionBiases(query: true, value: true);
        var input = Sequence(new Random(3), 5);
        _ = plain.Forward(input);
        _ = biased.Forward(input);

        // Q, K, V and output weights, the output bias, then the query and value biases.
        Assert.Equal(4L * Width * Width + Width, plain.ParameterCount);
        Assert.Equal(4L * Width * Width + 3 * Width, biased.ParameterCount);

        // Copy the shared weights and set the two biases nonzero: only the biases separate the two layers.
        var parameters = biased.GetParameters();
        var shared = plain.GetParameters();
        for (int i = 0; i < shared.Length; i++) parameters[i] = shared[i];
        for (int i = shared.Length; i < parameters.Length; i++) parameters[i] = 0.5;
        biased.SetParameters(parameters);

        var plainOutput = plain.Forward(input);
        var biasedOutput = biased.Forward(input);
        double largest = 0;
        for (int i = 0; i < plainOutput.Length; i++) largest = Math.Max(largest, Math.Abs(plainOutput[i] - biasedOutput[i]));
        Assert.True(largest > 1e-6, $"The projection biases changed the output by at most {largest:G3}.");
    }

    [Fact]
    public void ProjectionBiases_CannotBeAddedAfterTheWeightsExist()
    {
        var attention = new MultiHeadAttentionLayer<double>(Heads, Width / Heads);
        _ = attention.Forward(Sequence(new Random(5), 3));

        Assert.Throws<InvalidOperationException>(() => attention.ConfigureProjectionBiases(query: true, value: true));
    }

    [Fact]
    public void TiedHead_ComputesHiddenTimesTheTransposedTable()
    {
        var embedding = new EmbeddingLayer<double>(Vocab, Width);
        _ = embedding.Forward(new Tensor<double>([1]));
        var head = new TiedEmbeddingHeadLayer<double>(embedding);
        var hidden = Sequence(new Random(9), 3);

        var logits = head.Forward(hidden);

        var table = embedding.GetParameters();
        for (int row = 0; row < 3; row++)
        {
            for (int token = 0; token < Vocab; token++)
            {
                double expected = 0;
                for (int k = 0; k < Width; k++) expected += hidden[0, row, k] * table[token * Width + k];
                Assert.Equal(expected, logits[0, row, token], 10);
            }
        }
    }

    private static Tensor<double> Sequence(Random random, int length)
    {
        var tensor = new Tensor<double>([1, length, Width]);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = random.NextDouble() - 0.5;
        return tensor;
    }
}
