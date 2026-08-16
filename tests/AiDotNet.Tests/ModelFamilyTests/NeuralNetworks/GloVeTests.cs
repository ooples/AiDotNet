using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
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
public class GloVeTests : NeuralNetworkModelTestBase<float>
{
    // Paper-faithful inference: per-token output is W[i] + W̃[i] at d = 100,
    // so a 4-token input sequence produces a [4, 100] embedding tensor.
    protected override int[] InputShape => [4];
    protected override int[] OutputShape => [4, 100];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new GloVe<float>();
}
