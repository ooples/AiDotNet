using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Externally-verifiable invariants for
/// <see cref="SparseEmbeddingOptimizerHelpers.TryApplyAdamSparse{T}"/>.
/// The full sparse-scatter semantics (per-row m/v/θ update with duplicate-index
/// scatter-add and AdamW decoupled weight decay) are exercised end-to-end by
/// the embedding-layer ModelFamily tests in CI — a unit test here would need
/// to bind <c>_gradIndex</c> through the tape's non-public assignment hook,
/// which only happens during a real <c>ComputeGradients</c> walk.
/// </summary>
public class SparseEmbeddingOptimizerHelpersTests
{
    private const int VocabSize = 32;
    private const int EmbeddingDim = 4;

    [Fact]
    public void TryApplyAdamSparse_NoSparseGrads_ReturnsFalse()
    {
        // No tape, no SetIndexedSparseGrads — DifferentiableOps.GetSparseEmbeddingGradsFor
        // returns null for this param, and the helper must report "didn't apply"
        // so the caller falls back to the dense Adam path. This is the
        // backward-compatibility guarantee: optimizers that opt into the sparse
        // hint must remain correct on params that never receive sparse grads
        // (i.e. every non-embedding parameter in the network).
        var param = new Tensor<float>(new[] { VocabSize, EmbeddingDim });
        var m = new Tensor<float>(new[] { VocabSize, EmbeddingDim });
        var v = new Tensor<float>(new[] { VocabSize, EmbeddingDim });

        bool applied = SparseEmbeddingOptimizerHelpers.TryApplyAdamSparse(
            param, m, v,
            lr: 0.001, b1: 0.9, b2: 0.999,
            bc1: 0.1, bc2: 0.001, eps: 1e-8);

        Assert.False(applied);
    }

    [Fact]
    public void TryApplyAdamSparse_NonRank2Param_ReturnsFalse()
    {
        // Embedding tables are rank-2 [vocab, dim]. A rank-3 input is either
        // wrapped under extra batch dims (which the in-place scatter can't
        // walk) or a non-embedding parameter that got misrouted — either way
        // the safe answer is "fall through to dense".
        var param = new Tensor<float>(new[] { 4, VocabSize, EmbeddingDim });
        var m = new Tensor<float>(new[] { 4, VocabSize, EmbeddingDim });
        var v = new Tensor<float>(new[] { 4, VocabSize, EmbeddingDim });

        bool applied = SparseEmbeddingOptimizerHelpers.TryApplyAdamSparse(
            param, m, v,
            lr: 0.001, b1: 0.9, b2: 0.999,
            bc1: 0.1, bc2: 0.001, eps: 1e-8);

        Assert.False(applied);
    }
}
