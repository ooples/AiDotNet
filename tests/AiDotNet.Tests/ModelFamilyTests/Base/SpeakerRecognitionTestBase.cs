using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for speaker recognition/verification models. Inherits audio NN invariants
/// and adds speaker-specific: embedding similarity for same input and bounded embeddings.
/// </summary>
public abstract class SpeakerRecognitionTestBase : AudioNNModelTestBase
{
    [Fact]
    public void SameInput_SameEmbedding()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var emb1 = network.Predict(input);
        var emb2 = network.Predict(input);

        Assert.Equal(emb1.Length, emb2.Length);
        for (int i = 0; i < emb1.Length; i++)
            Assert.Equal(emb1[i], emb2[i]);
    }

    [Fact]
    public void SpeakerEmbedding_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var embedding = network.Predict(input);

        double normSq = 0;
        for (int i = 0; i < embedding.Length; i++)
        {
            Assert.False(double.IsNaN(embedding[i]),
                $"Speaker embedding[{i}] is NaN.");
            normSq += embedding[i] * embedding[i];
        }
        Assert.True(Math.Sqrt(normSq) < 1e4,
            $"Speaker embedding norm = {Math.Sqrt(normSq):E4} is unbounded.");
    }
}
