using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// The default acoustic stack shared by FastSpeech, FastSpeech 2, AdaSpeech and the other FFT-block TTS models.
/// Self-attention is order-blind, so without positional encodings the encoder sees a bag of phonemes; FastSpeech adds
/// them to the phoneme embeddings and again to the decoder input (Ren et al. 2019, Fig. 1).
/// </summary>
public class AcousticModelPositionalEncodingTests
{
    [Fact]
    public void PositionalEncodings_FollowTheEmbedding_AndPrecedeTheDecoder()
    {
        var layers = LayerHelper<double>.CreateDefaultAcousticModelLayers(
            encoderDim: 16, decoderDim: 8, hiddenDim: 12, numEncoderLayers: 2, numDecoderLayers: 2, numHeads: 2,
            dropoutRate: 0.0, vocabSize: 32, maxSequenceLength: 64).ToList();

        Assert.IsType<EmbeddingLayer<double>>(layers[0]);
        Assert.IsType<PositionalEncodingLayer<double>>(layers[1]);

        // Encoder blocks, the 16 -> 12 projection, then the decoder's own encoding ahead of its first block.
        int decoderEncoding = layers.FindLastIndex(layer => layer is PositionalEncodingLayer<double>);
        Assert.True(decoderEncoding > 1, "the decoder input has no positional encoding");
        Assert.IsType<DenseLayer<double>>(layers[decoderEncoding - 1]);
        Assert.IsType<TransformerEncoderBlock<double>>(layers[decoderEncoding + 1]);
        Assert.Equal(2, layers.Count(layer => layer is PositionalEncodingLayer<double>));
    }
}
