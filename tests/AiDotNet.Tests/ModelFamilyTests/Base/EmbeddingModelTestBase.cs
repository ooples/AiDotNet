using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for embedding models (models that map inputs to dense vector representations).
/// Inherits all neural network invariant tests and adds embedding-specific invariants:
/// similarity preservation, bounded outputs, and output dimensionality.
/// </summary>
public abstract class EmbeddingModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // EMBEDDING INVARIANT: Similar Inputs → Similar Embeddings
    // Epsilon-close inputs should produce high cosine similarity embeddings.
    // This tests the local Lipschitz continuity of the embedding function.
    // =====================================================

    [Fact]
    public void SimilarInputs_ProduceSimilarEmbeddings()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var input1 = CreateRandomTensor(InputShape, rng);
        // Create a near-identical input (small perturbation)
        var input2 = new Tensor<double>(InputShape);
        for (int i = 0; i < input1.Length; i++)
            input2[i] = input1[i] + 1e-6;

        var emb1 = network.Predict(input1);
        var emb2 = network.Predict(input2);

        // Compute cosine similarity
        double dot = 0, norm1 = 0, norm2 = 0;
        int minLen = Math.Min(emb1.Length, emb2.Length);
        for (int i = 0; i < minLen; i++)
        {
            dot += emb1[i] * emb2[i];
            norm1 += emb1[i] * emb1[i];
            norm2 += emb2[i] * emb2[i];
        }

        if (norm1 > 1e-15 && norm2 > 1e-15)
        {
            double cosineSim = dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2));
            Assert.True(cosineSim > 0.9,
                $"Cosine similarity = {cosineSim:F4} for epsilon-close inputs. " +
                "Embedding function is not locally continuous.");
        }
    }

    // =====================================================
    // EMBEDDING INVARIANT: Finite and Bounded Outputs
    // Embedding values should be finite and bounded — embeddings with
    // extreme values are numerically unstable and unusable downstream.
    // =====================================================

    [Fact]
    public void Embeddings_ShouldBeFiniteAndBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var embedding = network.Predict(input);

        for (int i = 0; i < embedding.Length; i++)
        {
            Assert.False(double.IsNaN(embedding[i]),
                $"Embedding[{i}] is NaN — numerical instability in embedding computation.");
            Assert.False(double.IsInfinity(embedding[i]),
                $"Embedding[{i}] is Infinity — overflow in embedding computation.");
            Assert.True(Math.Abs(embedding[i]) < 1e4,
                $"Embedding[{i}] = {embedding[i]:E4} exceeds bound of 1e4 — embedding is not well-bounded.");
        }
    }

    // =====================================================
    // EMBEDDING INVARIANT: Output Dimensionality
    // The output length should match the expected embedding dimension
    // (product of OutputShape dimensions).
    // =====================================================

    [Fact]
    public void OutputDimensionality_MatchesEmbeddingDim()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var embedding = network.Predict(input);

        int expectedLength = 1;
        foreach (var dim in OutputShape)
            expectedLength *= dim;

        Assert.Equal(expectedLength, embedding.Length);
    }
}
