using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.DiffusionTraceProbe;

/// <summary>
/// Replays the EXACT workload of the timeout-prone diffusion model-family tests
/// (<c>Training_ShouldReducePredictionError</c> / <c>ForwardPass_ShouldBeFinite_AfterTraining</c>)
/// for a single named model, so a sampling profiler (dotnet-trace) attached to
/// this process produces a clean, single-model hot-path report — without the
/// xunit/testhost noise and without the 16 GB-runner cross-model contention.
///
/// Construction and the [1,4,64,64] latent input mirror DiffusionModelTestBase&lt;float&gt;.
///
/// Usage:
///   DiffusionTraceProbe &lt;ModelName&gt; [--train-iters N] [--predict 0|1]
///   e.g.  DiffusionTraceProbe KandinskyModel --train-iters 5 --predict 1
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: DiffusionTraceProbe <ModelName> [--train-iters N] [--predict 0|1]");
            return 2;
        }

        string modelName = args[0];
        int trainIters = ArgInt(args, "--train-iters", 5);
        bool doPredict = ArgInt(args, "--predict", 1) != 0;
        bool correctness = args.Contains("--correctness");
        bool allocIsolate = args.Contains("--alloc-isolate");

        // Pin the CPU engine + disable GPU so the profile reflects the CPU path the
        // CI shard runs (AIDOTNET_DISABLE_GPU=1 is also set by the test harness).
        AiDotNetEngine.Current = new CpuEngine();

        var model = Construct(modelName);
        if (model is null) return 1;

        try
        {
            var diff = (IDiffusionModel<float>)model;
            var full = (IFullModel<float, Tensor<float>, Tensor<float>>)model;

            var rng = new Random(42);
            var input = RandomTensor(new[] { 1, 4, 64, 64 }, rng);

            Console.WriteLine($"=== DiffusionTraceProbe: {modelName} ===");
            Console.WriteLine($".NET {Environment.Version}  ParameterCount={SafeParamCount(full)}");

            // In-process correctness A/B: same model instance + same input, gate forced OFF then ON.
            // The DiT weights are randomly initialized per process (seed: null), so only a same-process
            // comparison can prove the *Into path is bit-identical to the allocating path.
            if (correctness)
            {
                int t = Math.Max(1, diff.Scheduler.TrainTimesteps / 2);
                var probeInput = RandomTensor(new[] { 1, 4, 64, 64 }, new Random(1234));
                // Force lazy init + warm so the first measured forward isn't paying init cost,
                // then measure a fresh forward each side for the per-forward allocation delta.
                // Allocation is deterministic (immune to this box's 2x timing swings), so it is
                // the primary proof the *Into / resident-scratch path removed the churn (#1672).
                AiDotNet.Helpers.ForwardScratchGate.Override = false;
                var warmOff = diff.PredictNoise(probeInput, t);
                long offA0 = GC.GetTotalAllocatedBytes(true);
                var offOut = diff.PredictNoise(probeInput, t);
                long offAlloc = GC.GetTotalAllocatedBytes(true) - offA0;

                AiDotNet.Helpers.ForwardScratchGate.Override = true;
                var warmOn = diff.PredictNoise(probeInput, t);
                long onA0 = GC.GetTotalAllocatedBytes(true);
                var onOut = diff.PredictNoise(probeInput, t);
                long onAlloc = GC.GetTotalAllocatedBytes(true) - onA0;
                AiDotNet.Helpers.ForwardScratchGate.Override = null;

                double mb = 1024.0 * 1024.0;
                double reductionPct = offAlloc > 0 ? (offAlloc - onAlloc) * 100.0 / offAlloc : 0.0;
                Console.WriteLine(
                    $"[ALLOC/forward] OFF={offAlloc / mb:F1} MB  ON={onAlloc / mb:F1} MB  " +
                    $"reduction={reductionPct:F1}%  saved={(offAlloc - onAlloc) / mb:F1} MB");

                double maxAbs = 0.0, maxRel = 0.0, refMax = 0.0;
                int n = Math.Min(offOut.Length, onOut.Length);
                for (int i = 0; i < n; i++)
                {
                    double a = Convert.ToDouble(offOut[i]);
                    double b = Convert.ToDouble(onOut[i]);
                    double diffAbs = Math.Abs(a - b);
                    if (diffAbs > maxAbs) maxAbs = diffAbs;
                    double denom = Math.Abs(a);
                    if (denom > 1e-8 && diffAbs / denom > maxRel) maxRel = diffAbs / denom;
                    if (Math.Abs(a) > refMax) refMax = Math.Abs(a);
                }
                Console.WriteLine(
                    $"[CORRECTNESS] len={n} maxAbs={maxAbs:R} maxRel={maxRel:R} refMax={refMax:R} " +
                    $"identical={(maxAbs == 0.0)}");
                Console.WriteLine("=== done ===");
                return 0;
            }

            // Steady-state per-forward GC-allocation isolation. Unlike --correctness (which
            // uses ForwardScratchGate.Override + GC.GetTotalAllocatedBytes(true), where the
            // forced GC reclaim makes the ON number swing wildly / negative on this box), this
            // mode honors the ENV gate settings (no Override) and measures the steady-state GC
            // allocation per forward WITHOUT a forced GC, averaged over many forwards so the
            // transient pool churn settles. Run it twice across two processes — once with
            // AIDOTNET_FWD_SCRATCH_FUSEDLINEAR=1 and once =0 (both with SDPA+ADALN on) — to
            // isolate the FusedLinear sub-gate's allocation contribution.
            if (allocIsolate)
            {
                int tt = Math.Max(1, diff.Scheduler.TrainTimesteps / 2);
                var ai = RandomTensor(new[] { 1, 4, 64, 64 }, new Random(1234));
                // Warm thoroughly so lazy init + JIT + scratch sizing are paid before measuring.
                for (int w = 0; w < 5; w++) { var _ = diff.PredictNoise(ai, tt); }
                const int reps = 20;
                long a0 = GC.GetTotalAllocatedBytes(false);
                for (int r = 0; r < reps; r++) { var _ = diff.PredictNoise(ai, tt); }
                long aN = GC.GetTotalAllocatedBytes(false);
                double mbI = 1024.0 * 1024.0;
                Console.WriteLine(
                    $"[ALLOC-ISOLATE] FWD_SCRATCH={AiDotNet.Helpers.ForwardScratchGate.Enabled} " +
                    $"FUSEDLINEAR={AiDotNet.Helpers.ForwardScratchGate.FusedLinear} " +
                    $"SDPA={AiDotNet.Helpers.ForwardScratchGate.Sdpa} ADALN={AiDotNet.Helpers.ForwardScratchGate.AdaLn} " +
                    $"perForward={(aN - a0) / mbI / reps:F1} MB  (reps={reps})");
                Console.WriteLine("=== done ===");
                return 0;
            }

            // DEFINITIVE interleaved in-process A/B: toggle ForwardScratchGate.Override OFF/ON
            // per-forward within ONE process so both conditions see identical box/thermal/GC
            // drift (cross-process launches differ ±36% even on the warm MIN harness — too noisy
            // for small deltas). Alternating per-forward cancels slow drift; MIN-of-N each side
            // is then a true apples-to-apples comparison of the FULL *Into gate vs the original
            // allocating path. NOTE: Override forces ALL sub-gates together, so this measures the
            // cumulative *Into win (rounds 4-6 + fused-QKV), not one sub-gate in isolation.
            if (args.Contains("--time-ab"))
            {
                int tt = Math.Max(1, diff.Scheduler.TrainTimesteps / 2);
                var ti = RandomTensor(new[] { 1, 4, 64, 64 }, new Random(1234));
                // Warm both paths.
                AiDotNet.Helpers.ForwardScratchGate.Override = false;
                for (int w = 0; w < 4; w++) { var _ = diff.PredictNoise(ti, tt); }
                AiDotNet.Helpers.ForwardScratchGate.Override = true;
                for (int w = 0; w < 4; w++) { var _ = diff.PredictNoise(ti, tt); }
                int reps = ArgInt(args, "--reps", 15);
                var off = new double[reps];
                var on = new double[reps];
                for (int r = 0; r < reps; r++)
                {
                    AiDotNet.Helpers.ForwardScratchGate.Override = false;
                    var s1 = Stopwatch.StartNew(); var _o = diff.PredictNoise(ti, tt); s1.Stop();
                    off[r] = s1.Elapsed.TotalMilliseconds;
                    AiDotNet.Helpers.ForwardScratchGate.Override = true;
                    var s2 = Stopwatch.StartNew(); var _n = diff.PredictNoise(ti, tt); s2.Stop();
                    on[r] = s2.Elapsed.TotalMilliseconds;
                }
                AiDotNet.Helpers.ForwardScratchGate.Override = null;
                Array.Sort(off); Array.Sort(on);
                double offMin = off[0], onMin = on[0];
                double offMed = off[reps / 2], onMed = on[reps / 2];
                Console.WriteLine(
                    $"[TIME-AB] OFF min={offMin:F1} median={offMed:F1} ms | ON min={onMin:F1} median={onMed:F1} ms | " +
                    $"MIN {offMin / onMin:F2}x ({100 * (offMin - onMin) / offMin:F1}%)  " +
                    $"MED {offMed / onMed:F2}x ({100 * (offMed - onMed) / offMed:F1}%)  (reps={reps}, interleaved)");
                Console.WriteLine("=== done ===");
                return 0;
            }

            // Reliable in-process per-forward TIMING (the cross-process full-Predict A/B is
            // unusable on a load-noisy box — swings 85-160s). Warm thoroughly, then time each
            // of N forwards on the SAME process with the cores already hot; report MIN (the
            // user's MIN-of-many) + median + p90 so noise is visible. Honors the ENV gate (no
            // Override) so OFF vs ON is a clean same-binary toggle. PredictNoise = ONE forward
            // (the unit the profiler analyzed); full Predict ~= 10x this (DDIM loop).
            if (args.Contains("--time-forward"))
            {
                int tt = Math.Max(1, diff.Scheduler.TrainTimesteps / 2);
                var ti = RandomTensor(new[] { 1, 4, 64, 64 }, new Random(1234));
                for (int w = 0; w < 8; w++) { var _ = diff.PredictNoise(ti, tt); }
                int reps = ArgInt(args, "--reps", 30);
                var ms = new double[reps];
                for (int r = 0; r < reps; r++)
                {
                    var sw2 = Stopwatch.StartNew();
                    var _ = diff.PredictNoise(ti, tt);
                    sw2.Stop();
                    ms[r] = sw2.Elapsed.TotalMilliseconds;
                }
                Array.Sort(ms);
                double median = ms[reps / 2];
                double p90 = ms[(int)(reps * 0.9)];
                Console.WriteLine(
                    $"[TIME-FWD] FWD_SCRATCH={AiDotNet.Helpers.ForwardScratchGate.Enabled} " +
                    $"FUSEDLINEAR={AiDotNet.Helpers.ForwardScratchGate.FusedLinear} " +
                    $"SDPA={AiDotNet.Helpers.ForwardScratchGate.Sdpa} ADALN={AiDotNet.Helpers.ForwardScratchGate.AdaLn} " +
                    $"min={ms[0]:F1} ms  median={median:F1} ms  p90={p90:F1} ms  (reps={reps})");
                Console.WriteLine("=== done ===");
                return 0;
            }

            // Phase 1: warm-up noise-prediction probe (the test's errBefore measurement).
            int probeT = Math.Max(1, diff.Scheduler.TrainTimesteps / 2);
            var noisy = RandomTensor(new[] { 1, 4, 64, 64 }, rng);
            var sw = Stopwatch.StartNew();
            var noisePred = diff.PredictNoise(noisy, probeT);
            sw.Stop();
            Console.WriteLine($"[PredictNoise warm-up]      {sw.Elapsed.TotalSeconds,8:F2} s");
            {
                double psum = 0.0, psumSq = 0.0;
                for (int i = 0; i < noisePred.Length; i++)
                {
                    double vv = Convert.ToDouble(noisePred[i]);
                    psum += vv; psumSq += vv * vv;
                }
                float pf = noisePred.Length > 0 ? Convert.ToSingle(noisePred[0]) : 0f;
                float pl = noisePred.Length > 0 ? Convert.ToSingle(noisePred[noisePred.Length - 1]) : 0f;
                Console.WriteLine(
                    $"[NOISE_CHECKSUM] len={noisePred.Length} sum={psum:R} sumSq={psumSq:R} first={pf:R} last={pl:R}");
            }

            // Phase 2: training loop (forward + backward) — the timeout test runs 5-10 iters.
            sw.Restart();
            for (int i = 0; i < trainIters; i++)
            {
                full.Train(input, input);
                Console.WriteLine($"  train iter {i + 1}/{trainIters}   {sw.Elapsed.TotalSeconds,8:F2} s cumulative");
            }
            sw.Stop();
            Console.WriteLine($"[Train x{trainIters}]               {sw.Elapsed.TotalSeconds,8:F2} s total");

            // Phase 3: full sampling Predict (DDIM loop x SD-UNet forward) — the heaviest call.
            if (doPredict)
            {
                sw.Restart();
                var output = full.Predict(input);
                sw.Stop();
                Console.WriteLine($"[Predict (sampling loop)]   {sw.Elapsed.TotalSeconds,8:F2} s  (out len {output.Length})");

                // Output fingerprint for the flag-OFF vs flag-ON bit-identicality A/B (#1672).
                double sum = 0.0, sumSq = 0.0;
                float first = output.Length > 0 ? Convert.ToSingle(output[0]) : 0f;
                float last = output.Length > 0 ? Convert.ToSingle(output[output.Length - 1]) : 0f;
                for (int i = 0; i < output.Length; i++)
                {
                    double vv = Convert.ToDouble(output[i]);
                    sum += vv;
                    sumSq += vv * vv;
                }
                Console.WriteLine(
                    $"[CHECKSUM] sum={sum:R} sumSq={sumSq:R} first={first:R} last={last:R}");
            }

            Console.WriteLine("=== done ===");
            return 0;
        }
        finally
        {
            if (model is IDisposable d) d.Dispose();
        }
    }

    private static object? Construct(string modelName)
    {
        var asm = typeof(AiDotNet.NeuralNetworks.NeuralNetworkBase<>).Assembly;
        var open = asm.GetTypes().FirstOrDefault(t =>
            t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
            && t.GetGenericArguments().Length == 1
            && string.Equals(t.Name, modelName + "`1", StringComparison.Ordinal));

        if (open is null)
        {
            // Allow passing the raw CLR name "Foo`1" too.
            open = asm.GetTypes().FirstOrDefault(t =>
                t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                && string.Equals(t.Name, modelName, StringComparison.Ordinal));
        }
        if (open is null)
        {
            Console.Error.WriteLine($"ERROR: no generic model type named '{modelName}' in {asm.GetName().Name}.");
            return null;
        }

        var closed = open.MakeGenericType(typeof(float));

        // Validate that the closed type implements IDiffusionModel<float> before attempting to cast it.
        var diffusionModelInterface = typeof(IDiffusionModel<float>);
        if (!diffusionModelInterface.IsAssignableFrom(closed))
        {
            Console.Error.WriteLine($"ERROR: {closed.Name} does not implement IDiffusionModel<float>.");
            return null;
        }

        // Pick the ctor with the most parameters that are ALL optional (matches the
        // (arch?, options?, scheduler?, predictor?, vae?, conditioner?, seed?) shape);
        // fill each with its compile-time default, but pin seed=42 like the tests.
        var ctor = closed.GetConstructors()
            .Where(c => c.GetParameters().All(p => p.HasDefaultValue))
            .OrderByDescending(c => c.GetParameters().Length)
            .FirstOrDefault();

        if (ctor is null)
        {
            Console.Error.WriteLine($"ERROR: {closed.Name} has no all-optional ctor to probe.");
            return null;
        }

        var pars = ctor.GetParameters();
        var ctorArgs = new object?[pars.Length];
        for (int i = 0; i < pars.Length; i++)
            ctorArgs[i] = string.Equals(pars[i].Name, "seed", StringComparison.OrdinalIgnoreCase)
                ? 42
                : pars[i].DefaultValue;

        try
        {
            var swCtor = Stopwatch.StartNew();
            var model = ctor.Invoke(ctorArgs);
            swCtor.Stop();
            Console.WriteLine($"[construct]                 {swCtor.Elapsed.TotalSeconds,8:F2} s");
            return model;
        }
        catch (TargetInvocationException tie)
        {
            var inner = tie.InnerException ?? tie;
            Console.Error.WriteLine($"ERROR constructing {closed.Name}: {inner.GetType().Name}: {inner.Message}");
            return null;
        }
    }

    private static long SafeParamCount(IFullModel<float, Tensor<float>, Tensor<float>> model)
    {
        try { return ((IDiffusionModel<float>)model).ParameterCount; }
        catch { return -1; }
    }

    private static Tensor<float> RandomTensor(int[] shape, Random rng)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)rng.NextDouble();
        return t;
    }

    private static int ArgInt(string[] args, string flag, int fallback)
    {
        int idx = Array.IndexOf(args, flag);
        if (idx >= 0 && idx + 1 < args.Length && int.TryParse(args[idx + 1], out int v)) return v;
        return fallback;
    }
}
