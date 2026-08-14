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

    // The MoreData cap above fixed the only probe that was tripping the PER-TEST gate, so this
    // class then failed a different way: it owns the NeuralNetworks T shard death in run
    // 31797679144, which produced NO TRX at all because the JOB hit its 45-minute ceiling. No
    // per-test gate ever fired, which is why it read as infrastructure noise rather than a slow
    // model. The streamed xUnit lines name the cost precisely -- all PASSING, just ruinously slow:
    //   LossStrictlyDecreasesOnMemorizationTask            2 m 08 s  (100 iterations)
    //   DifferentInputs_AfterTraining_ShouldProduceDiff..   1 m 38 s  (10 iterations)
    //   Training_ShouldChangeParameters                     1 m 33 s  (10 iterations)
    // Rung 1 (<float>) is already spent -- this fixture is EmbeddingModelTestBase<float>.
    //
    // Bound the two remaining uncapped budgets to the same rung-2 values the generator emits for
    // heavy models. 15 memorization iterations still clears the documented Adam warm-up hump, and
    // 5 training iterations still runs a real loss -> backward -> update cycle; tolerances and
    // decrease thresholds are untouched, as are the model, its dimensions and its loss.
    protected override int MemorizationTaskIterations => 15;
    protected override int TrainingIterations => 5;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new TransformerEmbeddingNetwork<float>();
}
