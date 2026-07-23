using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Root-cause coverage for #1824: the inference arena now Resets between layers so a single
/// transformer forward is memory-BOUNDED (peak ≈ O(one encoder block), independent of depth)
/// instead of accumulating every block's intermediates live for the whole pass (the ~817 MB
/// LAMBADA-eval forward that the arena — on OR off — never bounded, because it only recycled
/// ACROSS calls, not WITHIN one).
///
/// <para>Runs in the serialized <c>InferenceArena</c> collection: these flip the process-global
/// <see cref="InferenceArenaSettings"/> and read a thread-local arena peak counter.</para>
/// </summary>
[Collection("InferenceArena")]
public class InferenceArenaLayerRecycleTests
{
    private readonly ITestOutputHelper _output;

    public InferenceArenaLayerRecycleTests(ITestOutputHelper output) => _output = output;

    // Encoder-only Transformer<double> — double intentionally forces the eager RunLayerWalk path
    // (no fused float kernels) that Rents the per-block intermediates the arena recycles, and gives
    // bit-exact comparisons. numDecoderLayers:0 so encoderOutput capture stays null and the walk
    // exercises the running-stream + mask recycling.
    private static Transformer<double> BuildEncoderTransformer(int depth, int seqLen, int embDim, int heads)
    {
        var arch = new TransformerArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            numEncoderLayers: depth,
            numDecoderLayers: 0,
            numHeads: heads,
            modelDimension: embDim,
            feedForwardDimension: embDim * 4,
            inputSize: embDim,
            outputSize: embDim,
            maxSequenceLength: seqLen);
        var net = new Transformer<double>(arch);
        net.SetTrainingMode(false);
        return net;
    }

    private static Tensor<double> Batch(int batch, int seqLen, int embDim, int seed)
    {
        var rng = new Random(seed);
        var data = new double[batch * seqLen * embDim];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 2 - 1;
        return new Tensor<double>(data, new[] { batch, seqLen, embDim });
    }

    private static T WithFlags<T>(bool arenaEnabled, bool recycle, Func<T> body)
    {
        bool pe = InferenceArenaSettings.Enabled;
        bool pr = InferenceArenaSettings.RecycleLayerScratch;
        InferenceArenaSettings.Enabled = arenaEnabled;
        InferenceArenaSettings.RecycleLayerScratch = recycle;
        try { return body(); }
        finally
        {
            InferenceArenaSettings.Enabled = pe;
            InferenceArenaSettings.RecycleLayerScratch = pr;
        }
    }

    // ── correctness ──────────────────────────────────────────────────────────────

    [Fact]
    public void Transformer_Predict_IsBitIdentical_AcrossArenaAndRecycleModes()
    {
        var model = BuildEncoderTransformer(depth: 4, seqLen: 32, embDim: 32, heads: 4);
        var x = Batch(batch: 8, seqLen: 32, embDim: 32, seed: 7);

        // Warm up so lazy MHA/FFN weights materialize (mode-independent) before any comparison.
        WithFlags(true, true, () => { model.Predict(x); return 0; });

        double[] baseline = WithFlags(false, false, () => model.Predict(x).ToArray());
        double[] arenaNoRecycle = WithFlags(true, false, () => model.Predict(x).ToArray());
        double[] arenaRecycle = WithFlags(true, true, () => model.Predict(x).ToArray());

        Assert.Equal(baseline.Length, arenaNoRecycle.Length);
        Assert.Equal(baseline.Length, arenaRecycle.Length);
        for (int i = 0; i < baseline.Length; i++)
        {
            // Bit-exact (double, no tolerance): the per-layer detach + Reset must not perturb a
            // single value — it only recycles dead scratch and copies the boundary out verbatim.
            Assert.Equal(baseline[i], arenaNoRecycle[i]);
            Assert.Equal(baseline[i], arenaRecycle[i]);
        }

        // Argmax over the flattened output is identical too — the reduction a real eval consumes.
        Assert.Equal(ArgMax(baseline), ArgMax(arenaRecycle));
    }

    [Fact]
    public void Transformer_Predict_RepeatedRecycle_StaysBitStable()
    {
        var model = BuildEncoderTransformer(depth: 4, seqLen: 32, embDim: 32, heads: 4);
        var x = Batch(batch: 8, seqLen: 32, embDim: 32, seed: 21);

        double[] baseline = WithFlags(false, false, () => model.Predict(x).ToArray());

        // 5 recycled forwards in a row: a stale-scratch or missing-detach bug (the diffusion-style
        // cross-forward recycle hazard) would diverge on repeat 2+.
        WithFlags(true, true, () =>
        {
            for (int r = 0; r < 5; r++)
            {
                double[] got = model.Predict(x).ToArray();
                for (int i = 0; i < baseline.Length; i++) Assert.Equal(baseline[i], got[i]);
            }
            return 0;
        });
    }

    // ── memory bound ─────────────────────────────────────────────────────────────

    [Fact]
    public void Transformer_Forward_PeakIsBounded_RecycleVsNoRecycle()
    {
        const int depth = 6, seqLen = 64, embDim = 64, heads = 4, batch = 16;
        var model = BuildEncoderTransformer(depth, seqLen, embDim, heads);
        var x = Batch(batch, seqLen, embDim, seed: 3);

        // Warm up (materialize lazy weights) so the measured forwards allocate scratch only.
        WithFlags(true, true, () => { model.Predict(x); model.Predict(x); return 0; });

        long noRecyclePeak = MeasurePeak(model, x, recycle: false);
        long recyclePeak = MeasurePeak(model, x, recycle: true);

        double ratio = recyclePeak == 0 ? 0 : (double)noRecyclePeak / recyclePeak;
        _output.WriteLine(
            $"depth={depth} batch={batch} seq={seqLen} dim={embDim}: " +
            $"arena peak backing bytes — no-recycle={noRecyclePeak:N0}  recycle={recyclePeak:N0}  " +
            $"({ratio:F2}x lower)");

        // The pre-fix behavior (arena on, no per-layer Reset) holds every block's scratch live:
        // peak ≈ O(depth · block). With recycling the arena reuses one block's scratch for all
        // layers: peak ≈ O(one block). At depth 6 that is a large, deterministic gap — require the
        // recycled peak to be well under HALF the non-recycled peak (in practice ~4-6x lower).
        Assert.True(recyclePeak < noRecyclePeak / 2,
            $"recycled forward peak ({recyclePeak} B) should be far below the non-recycled peak " +
            $"({noRecyclePeak} B) — the arena is not bounding the per-layer churn.");
    }

    [Fact]
    public void Transformer_Forward_RecycledPeak_IsDepthIndependent()
    {
        const int seqLen = 48, embDim = 48, heads = 4, batch = 12;

        long ShallowRecycle() => MeasureFresh(depth: 3, seqLen, embDim, heads, batch, recycle: true);
        long DeepRecycle() => MeasureFresh(depth: 9, seqLen, embDim, heads, batch, recycle: true);
        long ShallowNo() => MeasureFresh(depth: 3, seqLen, embDim, heads, batch, recycle: false);
        long DeepNo() => MeasureFresh(depth: 9, seqLen, embDim, heads, batch, recycle: false);

        long recShallow = ShallowRecycle();
        long recDeep = DeepRecycle();
        long noShallow = ShallowNo();
        long noDeep = DeepNo();

        _output.WriteLine(
            $"peak backing bytes (batch={batch} seq={seqLen} dim={embDim}):\n" +
            $"  recycle:    depth3={recShallow:N0}  depth9={recDeep:N0}  (growth {(double)recDeep / recShallow:F2}x)\n" +
            $"  no-recycle: depth3={noShallow:N0}  depth9={noDeep:N0}  (growth {(double)noDeep / noShallow:F2}x)");

        // Tripling depth triples the non-recycled peak (each block adds its own live scratch)…
        Assert.True(noDeep > noShallow * 2,
            $"non-recycled peak must grow with depth: depth9={noDeep} should be >2x depth3={noShallow}.");
        // …while the recycled peak stays essentially flat (bounded to ~one block + boundary carry).
        Assert.True(recDeep < recShallow * 3 / 2,
            $"recycled peak must be depth-independent: depth9={recDeep} should be <1.5x depth3={recShallow}.");
    }

    private static long MeasureFresh(int depth, int seqLen, int embDim, int heads, int batch, bool recycle)
    {
        var model = BuildEncoderTransformer(depth, seqLen, embDim, heads);
        var x = Batch(batch, seqLen, embDim, seed: 5);
        WithFlags(true, true, () => { model.Predict(x); model.Predict(x); return 0; });
        return MeasurePeak(model, x, recycle);
    }

    private static long MeasurePeak(Transformer<double> model, Tensor<double> x, bool recycle)
    {
        return WithFlags(true, recycle, () =>
        {
            model.Predict(x); // the measured forward creates + disposes one per-call arena
            return TensorArena.LastDisposedPeakBackingBytes;
        });
    }

    private static int ArgMax(double[] a)
    {
        int idx = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[idx]) idx = i;
        return idx;
    }
}
