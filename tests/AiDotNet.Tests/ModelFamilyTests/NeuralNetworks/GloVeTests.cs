using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Paper-faithful invariant tests for the GloVe embedding model.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Pennington, Socher, Manning (2014), "GloVe: Global Vectors for
/// Word Representation", Stanford, EMNLP 2014.
/// </para>
/// <para>
/// These shapes follow the paper's inference contract (Section 4.3 footnote 5):
/// given a sequence of token indices, GloVe emits <c>W[i] + W̃[i]</c> per token,
/// where W is the word embedding matrix and W̃ is the context embedding matrix.
/// The default <c>GloVe&lt;double&gt;()</c> ctor uses the paper-reported
/// embedding dimension <c>d = 100</c> (paper Table 2 reports d ∈ {50, 100, 200, 300};
/// 100 is one of the four standard sizes the paper benchmarks).
/// </para>
/// <para>
/// <c>InputShape = [4]</c> is a 4-token sequence; <c>OutputShape = [4, 100]</c>
/// is the per-token paper-faithful sum <c>W + W̃</c> at <c>d = 100</c>.
/// </para>
/// </remarks>
public class GloVeTests : NeuralNetworkModelTestBase
{
    // Paper-faithful inference: per-token output is W[i] + W̃[i] at d = 100,
    // so a 4-token input sequence produces a [4, 100] embedding tensor.
    protected override int[] InputShape => [4];
    protected override int[] OutputShape => [4, 100];

    // Bound test indices to the lower part of GloVe's default vocab=10000 so
    // every drawn index is valid AND the index space is small enough that
    // the OutputSensitivity invariants reliably surface distinct rows of W
    // and W̃ across different test inputs.
    private const int TestVocabUpperBound = 100;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GloVe<double>();

    /// <summary>
    /// GloVe operates on integer token indices per the paper objective
    /// (Eq. 8: <c>w_i^T w̃_j + b_i + b̃_j = log X_ij</c> indexes by i, j).
    /// The base test base's <c>CreateRandomTensor</c> emits doubles in [0,1)
    /// which would all collapse to index 0, defeating the invariants. We
    /// override only the rank-1 input case (<see cref="InputShape"/>) so
    /// inputs are valid token indices; targets remain continuous-valued.
    /// </summary>
    protected override Tensor<double> CreateRandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<double>(shape);
        if (IsInputShape(shape))
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = rng.Next(0, TestVocabUpperBound);
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = rng.NextDouble();
        }
        return tensor;
    }

    /// <summary>
    /// Constant tensors in input-shape position must still be legal token
    /// indices — the base test passes 0.1 and 0.9 to verify input sensitivity,
    /// which would both round to 0 under <c>(int)</c> conversion. Map the
    /// scalar to deterministic but distinct indices instead.
    /// </summary>
    protected override Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<double>(shape);
        if (IsInputShape(shape))
        {
            // Deterministic non-zero index from the scalar so 0.1 and 0.9 map
            // to different rows of W / W̃ and the invariants are exercised.
            int idx = (int)(value * TestVocabUpperBound) % TestVocabUpperBound;
            if (idx < 0) idx += TestVocabUpperBound;
            for (int i = 0; i < tensor.Length; i++) tensor[i] = idx;
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++) tensor[i] = value;
        }
        return tensor;
    }

    private bool IsInputShape(int[] shape)
    {
        if (shape.Length != InputShape.Length) return false;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] != InputShape[i]) return false;
        }
        return true;
    }
}
