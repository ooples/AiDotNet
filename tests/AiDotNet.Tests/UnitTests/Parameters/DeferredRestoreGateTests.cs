using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Parameters;

/// <summary>
/// Phase 0 deletion gates for the deferred-restore capability.
/// </summary>
/// <remarks>
/// <para>
/// These gate the overrides that DCCRN, DeepFilterNet, SAM and ViMUNet each hand-wrote: an
/// <c>UpdateParameters</c> whose first statement is <c>ResolveLazyLayerShapes()</c>. Four models
/// independently patching the same base defect is the signal it belongs in the base, and the
/// governing rule for this refactor is that none of those overrides may be deleted until the base
/// demonstrably does the job — proven by a test that fails without it.
/// </para>
/// <para>
/// Driven at the LAYER level on purpose. A model-level test would run through whichever override is
/// still present and prove nothing about the base; <c>DenseLayer(outputSize)</c> declares its input
/// as <c>[-1]</c>, so it is genuinely deferred and exercises the base path directly.
/// </para>
/// <para>
/// The specific failure being gated: restore used to reject or skip a layer whose count was still 0.
/// DCCRN's own comment records the consequence — "a bare Layers walk would see 0 parameters, skip
/// every layer, and leave the clone on its own random initialization". Nothing threw; the clone was
/// simply wrong.
/// </para>
/// </remarks>
public class DeferredRestoreGateTests
{
    /// <summary>
    /// A deferred layer must ACCEPT a restore rather than reject it for having a zero count.
    /// </summary>
    /// <remarks>
    /// Before the readiness guard, this threw "Expected 0 parameters, but got N" — refusing the one
    /// piece of information that could have told the layer its shape. A zero count means "not sized
    /// yet", not "has none".
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task DeferredLayer_AcceptsRestore_InsteadOfRejectingItForAZeroCount()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(4);

        var payload = new Vector<double>(12);
        for (int i = 0; i < payload.Length; i++) payload[i] = i + 1;

        // The assertion is that this does not throw. An exception here is the B1 defect.
        layer.SetParameters(payload);
    }

    /// <summary>
    /// The parked payload must survive to materialization, not be truncated to a placeholder.
    /// </summary>
    /// <remarks>
    /// This is the half that the readiness guard alone would have broken. Letting a deferred layer
    /// past the guard drops it into the slicing path, which slices by the CURRENT slot lengths —
    /// placeholders before resolution. LayerBase records the real incident: it "cut a 144-value
    /// restore down to the 32-element placeholder". Parking the whole vector is the only correct
    /// action, and this asserts the payload is still intact afterwards rather than silently shortened.
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task DeferredLayer_ParksTheWholePayload_WithoutTruncatingToAPlaceholder()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(4);

        var payload = new Vector<double>(12);
        for (int i = 0; i < payload.Length; i++) payload[i] = i + 1;

        layer.SetParameters(payload);

        var readBack = layer.GetParameters();
        Assert.Equal(payload.Length, readBack.Length);
        for (int i = 0; i < payload.Length; i++)
        {
            Assert.Equal(payload[i], readBack[i]);
        }
    }

    /// <summary>
    /// A layer whose weights are sized entirely by constructor arguments is NOT deferred, so a
    /// genuine length mismatch must still be reported rather than silently parked.
    /// </summary>
    /// <remarks>
    /// The counterpart to the two above, and the reason the guard tests readiness rather than simply
    /// being deleted. Relaxing it unconditionally would turn every real restore-size bug into silent
    /// weight loss — strictly worse than the over-strict behaviour it replaced. EmbeddingLayer sizes
    /// its table from vocabulary and width alone, so it always knows its own count.
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task ConstructionSizedLayer_StillRejectsAWrongLengthRestore()
    {
        await Task.Yield();
        var layer = new EmbeddingLayer<double>(8, 3);   // 8 * 3 = 24 parameters, known up front
        layer.MaterializeParameters();

        var wrong = new Vector<double>(5);

        Assert.Throws<System.ArgumentException>(() => layer.SetParameters(wrong));
    }
}
