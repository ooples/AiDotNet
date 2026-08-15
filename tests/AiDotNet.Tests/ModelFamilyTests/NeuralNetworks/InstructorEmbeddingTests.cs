using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #1706/#1305: InstructorEmbedding wires a full 768-dim BERT-scale transformer. Its
// MoreData_ShouldNotDegrade runs 200 iterations of that training and is inherently >120s under the
// suite's single-threaded determinism BLAS even uncontended (confirmed: times out in a fully
// serialized run) — not a regression and not shrinkable. Tag HeavyTimeout so it runs full-fidelity
// nightly (deferred, not skipped). Matches the SimCSE/SPLADE/SGPT embedding-model precedent.
//
// The nightly lane still runs at full xUnit width, and OptimizerStep_ParamL2_DoesNotExplode was timing
// out THERE through starvation rather than slowness: measured alone on the CI-matched Release build it
// takes 5 s against its 120 s gate (24x headroom). That invariant is a SINGLE Train call plus two
// whole-parameter L2 sweeps, so the float rung (already applied — this fixture is <float>) and the
// iteration-cap rung cannot reach it; there is no iteration count to trim. Serialize instead, exactly
// as FoundationScaleSerialCollection documents, and leave the 768-dim BERT-scale fixture alone.
[Trait("Category", "HeavyTimeout")]
[Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class InstructorEmbeddingTests : EmbeddingModelTestBase<float>
{
    // InstructorEmbedding's default ctor wires a 768-dim transformer
    // (inputSize=768, outputSize=768). The test base defaults to [1, 4]
    // input and [1, 1] target, which caused the loss computation to try
    // subtracting a [1, 768] prediction from a [1, 1] target and throw
    // "Tensor shapes must match. Got [1, 768] and [1, 1]." in
    // MeanSquaredErrorLoss.ComputeTapeLoss. Align the test shapes with
    // the model's actual input/output dimensions.
    protected override int[] InputShape => [1, 768];
    protected override int[] OutputShape => [1, 768];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new InstructorEmbedding<float>();
}
