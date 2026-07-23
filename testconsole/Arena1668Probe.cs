using System;
using AiDotNet.Diffusion;
using AiDotNet.NeuralNetworks;

namespace AiDotNetTestConsole;

/// <summary>
/// Issue #1668 verification probe. The diffusion denoise loop must produce
/// bit-identical, finite, deterministic output with the inference TensorArena
/// (and its per-step Reset) ENABLED vs disabled. Before the layer no_grad gating
/// (InferenceMode scope + ShouldCacheForBackward), arena-on corrupted the output:
/// recycled scratch aliased layer activation caches (_lastInput, conv
/// _preAllocatedOutput, ...) that were still referenced across the Reset.
///
/// Run: dotnet run --project testconsole -- arena1668-probe
/// </summary>
internal static class Arena1668Probe
{
    public static void Run()
    {
        var shape = new[] { 1, 4, 8, 8 };
        const int seed = 42, steps = 10;

        var off = Generate(shape, steps, seed, arena: false);
        var on = Generate(shape, steps, seed, arena: true);

        int nanOff = CountNonFinite(off), nanOn = CountNonFinite(on);
        float maxDiff = 0f;
        int firstDiff = -1;
        for (int i = 0; i < off.Length; i++)
        {
            float d = Math.Abs(off[i] - on[i]);
            if (d > maxDiff) maxDiff = d;
            if (d != 0f && firstDiff < 0) firstDiff = i;
        }

        Console.WriteLine($"[#1668] DDPM Generate arena OFF vs ON  shape=[{string.Join(",", shape)}] steps={steps} seed={seed}");
        Console.WriteLine($"  non-finite:  off={nanOff}  on={nanOn}");
        Console.WriteLine($"  maxAbsDiff={maxDiff:G9}  firstDiffIdx={firstDiff}  len={off.Length}");
        bool bitIdentical = maxDiff == 0f && nanOff == 0 && nanOn == 0;
        Console.WriteLine($"  BIT-IDENTICAL (arena on==off): {(bitIdentical ? "PASS" : "FAIL")}");

        // Replay determinism with the denoise arena ON: two generations from the
        // same seed on the same instance must be bit-identical (catches arena state
        // bleeding between generations / a carried buffer aliasing recycled scratch).
        var replayShape = new[] { 1, 4, 4, 4 };
        float rmax = float.NaN;
        try
        {
            InferenceArenaSettings.Enabled = true;
            InferenceArenaSettings.DiffusionDenoiseEnabled = true;
            var m = new DDPMModel<float>(7);
            var ta = m.Generate(replayShape, 5, 7);
            var tb = m.Generate(replayShape, 5, 7);
            rmax = 0f;
            for (int i = 0; i < ta.Length; i++) rmax = Math.Max(rmax, Math.Abs(ta[i] - tb[i]));
        }
        finally
        {
            InferenceArenaSettings.DiffusionDenoiseEnabled = false;
            InferenceArenaSettings.Enabled = false;
        }
        bool replayOk = rmax == 0f;
        Console.WriteLine($"  REPLAY-DETERMINISTIC (arena on): {(replayOk ? "PASS" : "FAIL")}  maxAbsDiff={rmax:G9}");

        // Fail the process on any #1668 regression so this probe is usable as a gate
        // (otherwise a regression would still exit 0 and look green).
        if (!bitIdentical || !replayOk)
        {
            Console.Error.WriteLine("[#1668] PROBE FAILED: denoise arena is not bit-identical/deterministic.");
            Environment.Exit(1);
        }
    }

    private static float[] Generate(int[] shape, int steps, int seed, bool arena)
    {
        InferenceArenaSettings.Enabled = arena;
        InferenceArenaSettings.DiffusionDenoiseEnabled = arena;
        try
        {
            var t = new DDPMModel<float>(seed).Generate(shape, steps, seed);
            var arr = new float[t.Length];
            for (int i = 0; i < t.Length; i++) arr[i] = t[i];
            return arr;
        }
        finally
        {
            InferenceArenaSettings.DiffusionDenoiseEnabled = false;
            InferenceArenaSettings.Enabled = false;
        }
    }

    private static int CountNonFinite(float[] a)
    {
        int n = 0;
        foreach (var x in a) if (float.IsNaN(x) || float.IsInfinity(x)) n++;
        return n;
    }
}
