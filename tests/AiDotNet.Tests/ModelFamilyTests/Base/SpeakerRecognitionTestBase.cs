using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for speaker recognition/verification models. Inherits audio NN invariants
/// and adds speaker-specific: embedding similarity for same input and bounded embeddings.
/// </summary>
public abstract class SpeakerRecognitionTestBase : AudioNNModelTestBase
{
    [Fact(Timeout = 60000)]
    public async Task SameInput_SameEmbedding()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var emb1 = network.Predict(input);
        var emb2 = network.Predict(input);

        Assert.Equal(emb1.Length, emb2.Length);
        for (int i = 0; i < emb1.Length; i++)
            Assert.Equal(emb1[i], emb2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task SpeakerEmbedding_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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
