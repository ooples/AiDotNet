using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class TransformerEmbeddingNetworkTests : EmbeddingModelTestBase<float>
{
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [768];

    // Every other invariant (10 iterations) fits the 120s budget comfortably; only
    // MoreData_ShouldNotDegrade times out, because at the base 50+200 = 250 training iterations a
    // BERT-scale transformer embedding forward+backward (single-threaded determinism BLAS) exceeds
    // 120s even serialized in the T-Z shard. Override the sanctioned MoreData iteration knob down to
    // the embedding-family value (matches MatryoshkaEmbeddingTests) so the "more training does not
    // degrade" invariant still runs in the default gate at full model fidelity — the model, its
    // dimensions, and the loss are untouched; only this one test's iteration budget is reduced to fit
    // the timeout. Long (8) > short (4) keeps the monotonicity comparison meaningful. #1706/#1305.
    protected override int MoreDataShortIterations => 4;
    protected override int MoreDataLongIterations => 8;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new TransformerEmbeddingNetwork<float>();
}
