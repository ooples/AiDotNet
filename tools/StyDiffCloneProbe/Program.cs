using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.StyDiffCloneProbe;

/// <summary>
/// TEMPORARY DIAGNOSTIC SCAFFOLDING — deleted before the fix PR is opened.
///
/// Reproduces StyDiffModelTests.Clone_ShouldProduceIdenticalOutput on a Linux CI
/// runner and localises WHERE the clone-vs-original divergence first appears.
///
/// The test itself only compares the FINAL denoised tensor, which cannot tell a
/// clone-fidelity bug apart from a warm-vs-cold cache effect. This probe adds the
/// discriminators the test lacks:
///
///   A) P1 vs P2  — the SAME model predicted twice. Isolates warm-vs-cold
///                  process-global cache state (packed weights keyed by array
///                  identity) from anything clone-specific. If P1 != P2 the
///                  divergence is NOT about cloning at all.
///   B) P1 vs C1  — the actual test assertion.
///   C) P2 vs C1  — original-when-warm vs clone. If P1!=C1 but P2==C1, the clone
///                  is simply inheriting the original's warmed cache state.
///   D) PredictNoise stage compare — one UNet forward, same input sample, original
///                  vs clone, at every scheduler timestep. Removes the 10-step DDIM
///                  compounding so the FIRST differing operation is visible.
///   E) Conv2D vs Conv2DInto — the two arms of the explicit
///                  !IsOSPlatform(Windows) branch at ConvolutionalLayer.cs:1503,
///                  measured directly on this runner at the UNet's own conv shapes.
/// </summary>
internal static class Program
{
    private const int LatentH = 16;
    private const int LatentW = 16;
    private static readonly int[] InputShape = [1, 4, LatentH, LatentW];

    private static int Main(string[] args)
    {
        int iterations = ArgInt(args, "--iterations", 10);
        string sweep = ArgStr(args, "--mdop-sweep", "1,2,4,8");

        ConfigureLikeTestAssembly();
        PrintBanner(iterations);
        ProbeColdStart();
        ProbeConvolutionPlatformArms();
        ProbeInstantStyle(Math.Max(3, iterations));
        ProbeMdopFlip(Math.Max(3, iterations));

        var rng = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomTensor(InputShape, rng);

        LocaliseOnce(input);
        ProbePackCacheArms(input, Math.Max(3, iterations));

        Console.WriteLine("=== MDOP x CONTENTION SWEEP ===");
        Console.WriteLine("Run #1 (clean process, MDOP=ProcessorCount, no contention) gave 0/50 on this");
        Console.WriteLine("same runner, so the untested variables are the managed BlasManaged GEMM's");
        Console.WriteLine("partition count and CPU contention. FoundationScaleCpuFixture.Dispose restores");
        Console.WriteLine("MDOP to whatever a PREVIOUS collection left, so its value when StyDiff runs in");
        Console.WriteLine("a real shard is load-dependent -- this sweep covers that range directly.");
        Console.WriteLine();
        Console.WriteLine($"{"MDOP",5} {"burners",8} {"iters",6} {"P1!=C1",7} {"P1!=P2",7} {"P2!=C1",7} "
            + $"{"worst|P1-C1|",14} {"worst|P1-P2|",14}");

        bool anyDivergence = false;
        foreach (int mdop in ParseInts(sweep))
        {
            foreach (int burners in new[] { 0, Environment.ProcessorCount, Environment.ProcessorCount * 2 })
            {
                var cell = RunCell(mdop, burners, iterations, input);
                anyDivergence |= cell.CloneDiverged > 0 || cell.SelfDiverged > 0;
                Console.WriteLine($"{mdop,5} {burners,8} {iterations,6} {cell.CloneDiverged,7} "
                    + $"{cell.SelfDiverged,7} {cell.WarmDiverged,7} "
                    + $"{cell.WorstClone.ToString("R", CultureInfo.InvariantCulture),14} "
                    + $"{cell.WorstSelf.ToString("R", CultureInfo.InvariantCulture),14}");
            }
        }

        Console.WriteLine();
        Console.WriteLine(anyDivergence
            ? "RESULT: divergence REPRODUCED -- per-iteration dumps above locate it."
            : "RESULT: no divergence in any cell.");
        Console.WriteLine("=== done ===");
        return 0;
    }

    /// <summary>
    /// Mirrors the two module initializers plus the diffusion fixture, so the probe
    /// runs under exactly the process-global state the failing test runs under:
    /// TestAssemblyDeterminismInit.cs:39,51-53, ModuleInitializer.cs:59-61,87-95,
    /// DiffusionModelTestBase.cs:107 and FoundationScaleSerialCollection.cs:36-37.
    /// </summary>
    private static void ConfigureLikeTestAssembly()
    {
        Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");
        Environment.SetEnvironmentVariable("MKL_NUM_THREADS", "1");
        Environment.SetEnvironmentVariable("OPENBLAS_NUM_THREADS", "1");
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");

        AiDotNetEngine.SetDeterministicMode(true);

        // ResetToCpu() alone did NOT stick on a GPU-equipped box (the banner still
        // reported DirectGpuTensorEngine), which would have compared the GPU path
        // rather than the CPU path the failing test runs. Pin the engine outright,
        // as tools/DiffusionTraceProbe/Program.cs:36 does.
        AiDotNetEngine.Current = new CpuEngine();

        // NOTE: the diffusion base and the FoundationScaleSerial fixture both RAISE
        // this back to ProcessorCount, overriding ModuleInitializer's cap of 1. The
        // managed BlasManaged GEMM partitions over this value, so on a 4-core Linux
        // runner the reduction is split 4 ways where this Windows box splits it far wider.
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    private static void PrintBanner(int iterations)
    {
        Console.WriteLine("=== StyDiff clone probe ===");
        Console.WriteLine($"iterations              : {iterations}");
        Console.WriteLine($"OSDescription           : {RuntimeInformation.OSDescription}");
        Console.WriteLine($"OSArchitecture          : {RuntimeInformation.OSArchitecture}");
        Console.WriteLine($"ProcessArchitecture     : {RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"FrameworkDescription    : {RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"IsOSPlatform(Windows)   : {RuntimeInformation.IsOSPlatform(OSPlatform.Windows)}"
            + "   <- ConvolutionalLayer.cs:1503 preferConv2DInto = !this");
        Console.WriteLine($"ProcessorCount          : {Environment.ProcessorCount}");
        Console.WriteLine($"CpuParallelSettings.MDOP: {CpuParallelSettings.MaxDegreeOfParallelism}");
        Console.WriteLine($"Vector<float>.Count     : {System.Numerics.Vector<float>.Count}");
        Console.WriteLine($"Vector.IsHardwareAccel  : {System.Numerics.Vector.IsHardwareAccelerated}");
        Console.WriteLine($"Engine                  : {AiDotNetEngine.Current.GetType().FullName}");
        Console.WriteLine($"Sse2/Avx/Avx2/Fma/Avx512: "
            + $"{System.Runtime.Intrinsics.X86.Sse2.IsSupported}/"
            + $"{System.Runtime.Intrinsics.X86.Avx.IsSupported}/"
            + $"{System.Runtime.Intrinsics.X86.Avx2.IsSupported}/"
            + $"{System.Runtime.Intrinsics.X86.Fma.IsSupported}/"
            + $"{System.Runtime.Intrinsics.X86.Avx512F.IsSupported}");
        Console.WriteLine();
    }

    /// <summary>
    /// Measures the two arms of the platform branch directly. Windows runs
    /// Engine.Conv2D (Winograd); non-Windows runs Engine.Conv2DInto. If these
    /// disagree here, any cache/ordering effect that flips a layer between them
    /// shows up as exactly this magnitude of output error.
    /// </summary>
    private static void ProbeConvolutionPlatformArms()
    {
        Console.WriteLine("=== Conv2D vs Conv2DInto (ConvolutionalLayer.cs:1503 arms) ===");
        int[][] shapes =
        [
            [1, 32, 16, 16], [1, 64, 8, 8], [1, 128, 4, 4],
        ];
        var rng = RandomHelper.CreateSeededRandom(7);

        foreach (var s in shapes)
        {
            int inC = s[1];
            var inputT = CreateRandomTensor(s, rng);
            var kernel = CreateRandomTensor([inC, inC, 3, 3], rng);

            try
            {
                var engine = AiDotNetEngine.Current;
                var allocating = engine.Conv2D(inputT, kernel, 1, 1, 1);
                // 3x3 stride 1 pad 1 with outC == inC preserves the input shape, so `s`
                // is the destination shape without going through TensorShape.
                var into = new Tensor<float>(s);
                engine.Conv2DInto(into, inputT, kernel, 1, 1, 1);

                var cmp = Compare(allocating, into);
                Console.WriteLine($"  [{string.Join(",", s)}] k3s1 : differing={cmp.DiffCount}/{allocating.Length}"
                    + $"  maxAbs={cmp.MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
                    + $"  maxRel={cmp.MaxRel.ToString("R", CultureInfo.InvariantCulture)}");

                // SENTINEL COVERAGE TEST — does Conv2DInto write EVERY element it owns?
                //
                // On Linux preferConv2DInto is true, so TrackedConv2DInto
                // (ConvolutionalLayer.cs:1487-1488) rents an UNINITIALIZED buffer from
                // TensorAllocator and relies on Conv2DInto to fill it. Any position left
                // unwritten keeps whatever the pool last had there — and that residue
                // differs between the original's forward and the clone's, because they
                // occur at different points in the allocation sequence. Windows takes the
                // allocating Winograd Conv2D instead, which is why it would never show this.
                //
                // Comparing Conv2D vs Conv2DInto cannot detect it (a fresh buffer's residue
                // is zeros, so the values coincide). Flooding the destination with an absurd
                // sentinel first does: any survivor is a byte the kernel never wrote.
                const float sentinel = 987654.3f;
                var probe = new Tensor<float>(s);
                for (int i = 0; i < probe.Length; i++) probe[i] = sentinel;
                engine.Conv2DInto(probe, inputT, kernel, 1, 1, 1);

                int unwritten = 0, firstUnwritten = -1;
                for (int i = 0; i < probe.Length; i++)
                {
                    if (probe[i] == sentinel)
                    {
                        unwritten++;
                        if (firstUnwritten < 0) firstUnwritten = i;
                    }
                }

                Console.WriteLine(unwritten == 0
                    ? $"      sentinel: Conv2DInto wrote all {probe.Length} elements (no residue window)"
                    : $"      sentinel: *** {unwritten}/{probe.Length} elements LEFT UNWRITTEN, "
                      + $"firstIndex={firstUnwritten} -> reads pool residue ***");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [{string.Join(",", s)}] probe failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        Console.WriteLine();
    }

    /// <summary>
    /// One UNet forward per scheduler timestep, original vs clone, on an identical
    /// input sample. Strips out the 10-step DDIM feedback so a difference here is a
    /// single forward's difference, not compounded drift.
    /// </summary>
    private static void LocaliseByStage(
        IDiffusionModel<float> model, IDiffusionModel<float> clone, Tensor<float> input)
    {
        Console.WriteLine("  --- per-timestep PredictNoise (single forward, identical input) ---");
        int train = model.Scheduler.TrainTimesteps;
        int stride = Math.Max(1, train / 10);

        for (int i = 0; i < 10; i++)
        {
            int t = Math.Max(0, train - 1 - (i * stride));
            try
            {
                var a = model.PredictNoise(input, t);
                var b = clone.PredictNoise(input, t);
                var cmp = Compare(a, b);
                Console.WriteLine($"    t={t,4}: differing={cmp.DiffCount,5}/{a.Length}"
                    + $"  maxAbs={cmp.MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
                    + $"  atIndex={cmp.Index}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    t={t,4}: failed {ex.GetType().Name}: {ex.Message}");
            }
        }

        Console.WriteLine("  --- repeated PredictNoise on the SAME instance (warm-vs-cold) ---");
        try
        {
            int t = train - 1;
            var f1 = model.PredictNoise(input, t);
            var f2 = model.PredictNoise(input, t);
            var f3 = model.PredictNoise(input, t);
            Console.WriteLine($"    call1-vs-call2 maxAbs={Compare(f1, f2).MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
                + $"   call2-vs-call3 maxAbs={Compare(f2, f3).MaxDiff.ToString("R", CultureInfo.InvariantCulture)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"    repeat probe failed: {ex.GetType().Name}: {ex.Message}");
        }
    }

    /// <summary>
    /// InstantStyle is the DETERMINISTIC case, and it is the one worth chasing.
    ///
    /// CI history shows the identical assertion in 6 separate runs across 4 days
    /// (2026-09-06, 09-07 x4, 09-08), always the ModelFamily - Diffusion D-I shard:
    ///
    ///   Clone() output[2] = -4.626447E+000 differs from -4.626425E+000
    ///   by 2.241135E-005, which exceeds its own tolerance 1.601435E-005
    ///
    /// Same element, same two values, same diff to seven digits, every time — that is
    /// not a race. (StyDiff by contrast hits a different element each run: 311, 347, 58.)
    /// A deterministic divergence can be localised directly, so this reproduces the
    /// fixture exactly as InstantStyleModelTests.cs:30-43 declares it.
    /// </summary>
    /// <summary>
    /// MDOP FLIPPED BETWEEN THE ORIGINAL'S PREDICT AND THE CLONE'S.
    ///
    /// AiDotNet.Native.OpenBLAS/OneDNN/CLBlast 0.130.3 ship runtimes/win-x64 ONLY —
    /// there is no linux-x64 native binary in any of them. So on Linux every GEMM
    /// takes the MANAGED BlasManaged path, which partitions over
    /// CpuParallelSettings.MaxDegreeOfParallelism. (It also means the suite's
    /// determinism guard is inoperative there: OPENBLAS_NUM_THREADS=1 and
    /// BlasProvider.SetDeterministicMode -> openblas_set_num_threads(1) pin a library
    /// that was never loaded on Linux.)
    ///
    /// MDOP is PROCESS-GLOBAL and test fixtures mutate it:
    ///   DiffusionModelTestBase.cs:107      static ctor  -> ProcessorCount
    ///   ModuleInitializer.cs:87-95         -> 1
    ///   FoundationScaleCpuFixture.cs:32-44 ctor -> ProcessorCount, Dispose RESTORES
    ///                                       whatever a previous collection left
    ///
    /// InstantStyleModelTests carries no [Collection("FoundationScaleSerial")], so it
    /// runs in the default PARALLEL collection — where another collection finishing can
    /// flip MDOP in between this test's two Predict calls. Same weights, different
    /// partition count, different reduction order, small divergence. This forces that
    /// interleaving deterministically instead of waiting for the race.
    /// </summary>
    /// <summary>
    /// COLD START — the clone comparison as the FIRST computation in the process.
    ///
    /// Every previous probe run executed the conv arms and warm-up loops before the
    /// measured comparison, which tiers the hot kernels up to Tier1 BEFORE anything is
    /// compared. That systematically hides the one mechanism still untested:
    ///
    ///   .NET promotes a hot method from Tier0 to Tier1 on a background thread after
    ///   ~30 calls, and Tier1 may vectorize / contract FMAs differently. In the real
    ///   test the original's Predict (10 DDIM steps x many layers) drives those kernels
    ///   past the threshold, so the CLONE's Predict can execute Tier1 code while the
    ///   original ran Tier0. Same weights, different codegen, small divergence.
    ///
    /// That is clone-specific, deterministic for a fixed test ordering (InstantStyle),
    /// intermittent when call counts shift (StyDiff), and invisible to
    /// Predict_ShouldBeDeterministic when both its calls land on one side of the tier-up.
    ///
    /// Pair this with DOTNET_TieredCompilation=0 as the A/B: if divergence appears here
    /// and vanishes with tiering off, the JIT tier transition is the mechanism.
    /// </summary>
    private static void ProbeColdStart()
    {
        Console.WriteLine("=== COLD START — clone comparison as the FIRST computation ===");
        string tiered = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") ?? "<default: on>";
        string tieredPgo = Environment.GetEnvironmentVariable("DOTNET_TieredPGO") ?? "<default>";
        string osr = Environment.GetEnvironmentVariable("DOTNET_TC_QuickJitForLoops") ?? "<default>";
        Console.WriteLine($"  DOTNET_TieredCompilation={tiered}  DOTNET_TieredPGO={tieredPgo}  TC_QuickJitForLoops={osr}");

        foreach (var (name, factory, shape) in new (string, Func<IDiffusionModel<float>>, int[])[]
                 {
                     ("InstantStyle", CreateInstantStyleModel, [1, 4, 8, 8]),
                     ("StyDiff", CreateModel, InputShape),
                 })
        {
            var rng = RandomHelper.CreateSeededRandom(42);
            var input = CreateRandomTensor(shape, rng);

            using var arena = TensorArena.Create();
            using var model = factory();

            var p1 = model.Predict(input);                     // Tier0-ish, cold process
            using var clone = (IDiffusionModel<float>)model.Clone();
            var c1 = clone.Predict(input);                     // may now be Tier1
            var p2 = model.Predict(input);

            var cloneCmp = Compare(p1, c1);
            var selfCmp = Compare(p1, p2);
            Console.WriteLine($"  {name,-13} COLD: P1-vs-C1 maxAbs="
                + $"{cloneCmp.MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
                + $" differing={cloneCmp.DiffCount}/{p1.Length}"
                + $" | P1-vs-P2 maxAbs={selfCmp.MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
                + $" | out[2]={(p1.Length > 2 ? p1[2].ToString("R", CultureInfo.InvariantCulture) : "n/a")}");

            if (cloneCmp.MaxDiff > 0) Report($"  {name} COLD divergence", cloneCmp, p1);
        }

        Console.WriteLine();
    }

    private static void ProbeMdopFlip(int iterations)
    {
        Console.WriteLine("=== MDOP FLIPPED BETWEEN ORIGINAL AND CLONE ===");
        int[][] pairs = [[4, 1], [1, 4], [4, 8], [8, 4], [4, 2], [2, 4]];

        var models = new (string Name, Func<IDiffusionModel<float>> Factory, int[] Shape)[]
        {
            ("StyDiff", CreateModel, InputShape),
            ("InstantStyle", CreateInstantStyleModel, [1, 4, 8, 8]),
        };

        foreach (var (name, factory, shape) in models)
        {
            var rng = RandomHelper.CreateSeededRandom(42);
            var input = CreateRandomTensor(shape, rng);

            foreach (var pair in pairs)
            {
                int a = pair[0], b = pair[1];
                int diverged = 0;
                double worst = 0.0;

                for (int i = 0; i < iterations; i++)
                {
                    using var arena = TensorArena.Create();
                    using var model = factory();

                    CpuParallelSettings.MaxDegreeOfParallelism = a;
                    var p1 = model.Predict(input);

                    CpuParallelSettings.MaxDegreeOfParallelism = b;
                    using var clone = (IDiffusionModel<float>)model.Clone();
                    var c1 = clone.Predict(input);

                    var cmp = Compare(p1, c1);
                    if (cmp.MaxDiff > worst) worst = cmp.MaxDiff;
                    if (cmp.MaxDiff > 0)
                    {
                        diverged++;
                        if (diverged == 1) Report($"  {name} mdop {a}->{b} first divergence", cmp, p1);
                    }
                }

                Console.WriteLine($"  {name,-13} mdop {a,2} -> {b,-2} : diverged={diverged}/{iterations}"
                    + $"  worstAbs={worst.ToString("R", CultureInfo.InvariantCulture)}");
            }
        }

        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        Console.WriteLine();
    }

    private static void ProbeInstantStyle(int iterations)
    {
        Console.WriteLine("=== INSTANTSTYLE — deterministic CI signature ===");
        Console.WriteLine("  expected from CI: output[2] original=-4.626425E+000 clone=-4.626447E+000");

        int[] shape = [1, 4, 8, 8];
        var rng = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomTensor(shape, rng);

        for (int i = 0; i < iterations; i++)
        {
            using var arena = TensorArena.Create();
            using var model = CreateInstantStyleModel();

            var p1 = model.Predict(input);
            using var clone = (IDiffusionModel<float>)model.Clone();
            CompareParameters($"iter {i} InstantStyle weights", model, clone);
            var c1 = clone.Predict(input);
            var p2 = model.Predict(input);

            var cloneCmp = Compare(p1, c1);
            var selfCmp = Compare(p1, p2);

            string o2 = p1.Length > 2 ? p1[2].ToString("R", CultureInfo.InvariantCulture) : "n/a";
            string c2 = c1.Length > 2 ? c1[2].ToString("R", CultureInfo.InvariantCulture) : "n/a";
            Console.WriteLine($"  iter {i}: P1-vs-C1 maxAbs="
                + $"{cloneCmp.MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
                + $" differing={cloneCmp.DiffCount}/{p1.Length} atIndex={cloneCmp.Index}"
                + $" | P1-vs-P2 maxAbs={selfCmp.MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
                + $" | output[2] orig={o2} clone={c2}");

            if (cloneCmp.MaxDiff > 0)
            {
                Report("  InstantStyle P1-vs-C1", cloneCmp, p1);
                LocaliseByStage(model, clone, input);
                break;
            }
        }

        Console.WriteLine();
    }

    /// <summary>
    /// THE DISCRIMINATOR THE SUITE NEVER HAD.
    ///
    /// DiffusionModelTestBase only ever compares clone OUTPUT (Clone_ShouldProduce-
    /// IdenticalOutput) — unlike NeuralNetworkModelTestBase, which also asserts
    /// AssertCloneOwnsIndependentParameterStorage. So for all 241 diffusion fixtures
    /// "the clone carries different weights" has never actually been ruled out, and an
    /// output delta cannot distinguish:
    ///
    ///   parameters IDENTICAL -> same weights, so the delta is a COMPUTE-PATH difference
    ///   parameters DIFFER    -> a real clone weight-fidelity bug (and deterministic,
    ///                           which is exactly InstantStyle's signature)
    /// </summary>
    private static void CompareParameters(string label, IDiffusionModel<float> a, IDiffusionModel<float> b)
    {
        // Some model families deliberately refuse a flat parameter vector
        // ("use WriteParameters/ReadParameters"). That is an opt-out, not a finding —
        // and it must not crash a CI probe run.
        AiDotNet.Tensors.LinearAlgebra.Vector<float> pa, pb;
        try
        {
            pa = a.GetParameters();
            pb = b.GetParameters();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  {label}: parameter comparison unavailable ({ex.GetType().Name}: {ex.Message})");
            return;
        }

        if (pa.Length != pb.Length)
        {
            Console.WriteLine($"  {label}: PARAMETER COUNT MISMATCH {pa.Length} vs {pb.Length}");
            return;
        }

        int diffCount = 0, firstIdx = -1;
        double maxAbs = 0.0;
        for (int i = 0; i < pa.Length; i++)
        {
            double d = Math.Abs((double)pa[i] - pb[i]);
            if (d > 0)
            {
                diffCount++;
                if (firstIdx < 0) firstIdx = i;
                if (d > maxAbs) maxAbs = d;
            }
        }

        Console.WriteLine(diffCount == 0
            ? $"  {label}: parameters IDENTICAL ({pa.Length}) -> any output delta is COMPUTE-PATH"
            : $"  {label}: parameters DIFFER {diffCount}/{pa.Length} firstIndex={firstIdx} "
              + $"maxAbs={maxAbs.ToString("R", CultureInfo.InvariantCulture)} "
              + $"orig={pa[firstIdx].ToString("R", CultureInfo.InvariantCulture)} "
              + $"clone={pb[firstIdx].ToString("R", CultureInfo.InvariantCulture)} "
              + "-> CLONE WEIGHT-FIDELITY BUG");
    }

    private static IDiffusionModel<float> CreateInstantStyleModel()
        => new AiDotNet.Diffusion.StyleTransfer.InstantStyleModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                architecture: null, inputChannels: 4, outputChannels: 4,
                baseChannels: 32, channelMultipliers: [1, 2],
                numResBlocks: 1, attentionResolutions: [2], contextDim: 64, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4,
                baseChannels: 16, channelMultipliers: [1, 2],
                numResBlocksPerLevel: 1, seed: 42),
            seed: 42);

    private static IDiffusionModel<float> CreateModel()
        => new AiDotNet.Diffusion.StyleTransfer.StyDiffModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: [1, 2, 4], numResBlocks: 1,
                attentionResolutions: [1, 2], contextDim: 768, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: [1, 2], numResBlocksPerLevel: 1, seed: 42),
            seed: 42);

    private static Tensor<float> CreateRandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = (float)rng.NextDouble();
        return tensor;
    }

    private readonly struct Cmp
    {
        public double MaxDiff { get; init; }
        public double MaxRel { get; init; }
        public double RefMax { get; init; }
        public int Index { get; init; }
        public int DiffCount { get; init; }
        public float A { get; init; }
        public float B { get; init; }
    }

    private static Cmp Compare(Tensor<float> a, Tensor<float> b)
    {
        double maxDiff = 0, maxRel = 0, refMax = 0;
        int index = -1, diffCount = 0;
        float av = 0, bv = 0;
        int n = Math.Min(a.Length, b.Length);

        for (int i = 0; i < n; i++)
        {
            float x = a[i], y = b[i];
            double d = Math.Abs((double)x - y);
            if (d > 0) diffCount++;
            if (Math.Abs((double)x) > refMax) refMax = Math.Abs((double)x);
            if (d > maxDiff)
            {
                maxDiff = d;
                index = i;
                av = x;
                bv = y;
                double denom = Math.Abs((double)x);
                if (denom > 1e-12) maxRel = d / denom;
            }
        }

        return new Cmp
        {
            MaxDiff = maxDiff, MaxRel = maxRel, RefMax = refMax,
            Index = index, DiffCount = diffCount, A = av, B = bv,
        };
    }

    private static void Report(string label, Cmp cmp, Tensor<float> reference)
    {
        if (cmp.MaxDiff == 0)
        {
            Console.WriteLine($"  {label}: identical (0 differing of {reference.Length})");
            return;
        }

        Console.WriteLine($"  {label}: differing={cmp.DiffCount}/{reference.Length}"
            + $" firstWorstIndex={cmp.Index}"
            + $" a={cmp.A.ToString("R", CultureInfo.InvariantCulture)}"
            + $" b={cmp.B.ToString("R", CultureInfo.InvariantCulture)}"
            + $" bitsA=0x{BitConverter.SingleToInt32Bits(cmp.A):X8}"
            + $" bitsB=0x{BitConverter.SingleToInt32Bits(cmp.B):X8}"
            + $" maxAbs={cmp.MaxDiff.ToString("R", CultureInfo.InvariantCulture)}"
            + $" maxRel={cmp.MaxRel.ToString("R", CultureInfo.InvariantCulture)}"
            + $" refMax={cmp.RefMax.ToString("R", CultureInfo.InvariantCulture)}");
    }

    private readonly struct Cell
    {
        public int CloneDiverged { get; init; }
        public int SelfDiverged { get; init; }
        public int WarmDiverged { get; init; }
        public double WorstClone { get; init; }
        public double WorstSelf { get; init; }
    }

    /// <summary>
    /// One sweep cell: a fixed managed-GEMM partition count and a fixed number of
    /// CPU burner threads, over N clone comparisons. P1-vs-P2 is reported alongside
    /// P1-vs-C1 deliberately — if the same model predicted twice diverges, the defect
    /// is contention-dependent nondeterminism, not clone fidelity.
    /// </summary>
    private static Cell RunCell(int mdop, int burners, int iterations, Tensor<float> input)
    {
        CpuParallelSettings.MaxDegreeOfParallelism = mdop;

        using var cts = new System.Threading.CancellationTokenSource();
        var threads = StartBurners(burners, cts.Token);

        int cloneDiverged = 0, selfDiverged = 0, warmDiverged = 0;
        double worstClone = 0.0, worstSelf = 0.0;

        try
        {
            for (int iter = 0; iter < iterations; iter++)
            {
                using var arena = TensorArena.Create();
                using var model = CreateModel();

                var p1 = model.Predict(input);
                using var clone = (IDiffusionModel<float>)model.Clone();
                var c1 = clone.Predict(input);
                var p2 = model.Predict(input);

                var cloneCmp = Compare(p1, c1);
                var selfCmp = Compare(p1, p2);
                var warmCmp = Compare(p2, c1);

                if (cloneCmp.MaxDiff > worstClone) worstClone = cloneCmp.MaxDiff;
                if (selfCmp.MaxDiff > worstSelf) worstSelf = selfCmp.MaxDiff;
                if (cloneCmp.MaxDiff > 0) cloneDiverged++;
                if (selfCmp.MaxDiff > 0) selfDiverged++;
                if (warmCmp.MaxDiff > 0) warmDiverged++;

                if (cloneCmp.MaxDiff > 0 || selfCmp.MaxDiff > 0)
                {
                    Console.WriteLine($"  --- DIVERGENCE mdop={mdop} burners={burners} iter={iter} ---");
                    Report("P1-vs-C1 (the test assertion)", cloneCmp, p1);
                    Report("P1-vs-P2 (same model twice)  ", selfCmp, p1);
                    Report("P2-vs-C1 (warm orig vs clone)", warmCmp, p2);
                    LocaliseByStage(model, clone, input);
                }
            }
        }
        finally
        {
            cts.Cancel();
            foreach (var t in threads)
                t.Join();
        }

        return new Cell
        {
            CloneDiverged = cloneDiverged, SelfDiverged = selfDiverged,
            WarmDiverged = warmDiverged, WorstClone = worstClone, WorstSelf = worstSelf,
        };
    }

    private static System.Threading.Thread[] StartBurners(
        int count, System.Threading.CancellationToken token)
    {
        var threads = new System.Threading.Thread[Math.Max(0, count)];
        for (int i = 0; i < threads.Length; i++)
        {
            var t = new System.Threading.Thread(() =>
            {
                double acc = 0.0;
                while (!token.IsCancellationRequested)
                {
                    for (int k = 0; k < 200_000; k++) acc += k * 0.5;
                }

                if (double.IsNaN(acc)) Console.Write(string.Empty);
            })
            {
                IsBackground = true,
            };
            t.Start();
            threads[i] = t;
        }

        return threads;
    }

    /// <summary>
    /// DETERMINISTIC test of the mechanism the test's own comment blames ("cold
    /// packed-weight rounding path", DiffusionModelTestBase.cs:191-197).
    ///
    /// Rather than waiting for the rare CI race, this FORCES the cold/warm asymmetry
    /// using the public InferenceWeightCache API and asks whether it can produce a
    /// divergence at all:
    ///
    ///   A baseline          original.Predict then clone.Predict (what the test does)
    ///   B invalidate-between original runs WARM, clone then runs COLD
    ///   C invalidate-before-both  both run COLD
    ///   D double-invalidate  cold original vs warm clone (the reverse asymmetry)
    ///
    /// If A is clean but B or D diverges, the packed-weight path is the mechanism and
    /// the flake is whatever makes the cache state differ in a loaded shard. If every
    /// arm is clean, the packed-weight explanation in that comment is wrong.
    /// </summary>
    private static void ProbePackCacheArms(Tensor<float> input, int iterations)
    {
        Console.WriteLine("=== PACKED-WEIGHT CACHE ARMS (deterministic cold/warm forcing) ===");

        // VALIDITY CHECK FIRST. If the diffusion forward never engages the pack
        // cache, InvalidateAll() is a no-op and every arm below is INERT — a clean
        // sweep would then mean "the probe did nothing", not "the packed-weight
        // explanation is refuted". A live pack cache has to re-pack after an
        // invalidate, so a cold predict must be measurably slower than a warm one.
        {
            using var arena = TensorArena.Create();
            using var model = CreateModel();
            model.Predict(input);                       // warm everything up
            var sw = System.Diagnostics.Stopwatch.StartNew();
            model.Predict(input);
            double warmMs = sw.Elapsed.TotalMilliseconds;

            InferenceWeightCache.InvalidateAll();
            sw.Restart();
            model.Predict(input);
            double coldMs = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            model.Predict(input);
            double rewarmMs = sw.Elapsed.TotalMilliseconds;

            double ratio = warmMs > 0 ? coldMs / warmMs : 0.0;
            Console.WriteLine($"  validity: warm={warmMs:F1}ms  afterInvalidate={coldMs:F1}ms  "
                + $"rewarm={rewarmMs:F1}ms  cold/warm={ratio:F2}x");
            Console.WriteLine(ratio > 1.15
                ? "  -> pack cache appears LIVE on this path (invalidate costs time); arms are meaningful."
                : "  -> NO measurable re-pack cost; the arms below may be INERT for this model.");
        }

        Console.WriteLine($"{"arm",-22} {"iters",6} {"diverged",9} {"worstAbs",16}");

        foreach (var arm in new[]
                 {
                     "A-baseline", "B-cold-clone", "C-cold-both", "D-cold-original",
                     "E-polluted", "F-polluted-cold-clone",
                 })
        {
            int diverged = 0;
            double worst = 0.0;

            for (int i = 0; i < iterations; i++)
            {
                // The real shard runs ~7 sibling model classes in ONE process before
                // StyDiff. Every clean-room run so far lacked that, and it is the last
                // untested variable: recycled pool buffers carry residue from those
                // models, so a partially-written rent reads different garbage for the
                // original than for the clone. Same class as Tensors #755 ("zero the
                // unwritten tail ... rented UNINITIALIZED from AutoTensorCache, a tail
                // left dirty by an earlier op").
                if (arm[0] is 'E' or 'F') PolluteProcess(4);

                using var arena = TensorArena.Create();
                using var model = CreateModel();

                if (arm is "C-cold-both" or "D-cold-original") InferenceWeightCache.InvalidateAll();
                var p1 = model.Predict(input);

                if (arm is "B-cold-clone" or "C-cold-both" or "F-polluted-cold-clone")
                    InferenceWeightCache.InvalidateAll();
                using var clone = (IDiffusionModel<float>)model.Clone();
                var c1 = clone.Predict(input);

                var cmp = Compare(p1, c1);
                if (cmp.MaxDiff > worst) worst = cmp.MaxDiff;
                if (cmp.MaxDiff > 0)
                {
                    diverged++;
                    if (diverged == 1) Report($"  {arm} first divergence", cmp, p1);
                }
            }

            Console.WriteLine($"{arm,-22} {iterations,6} {diverged,9} "
                + $"{worst.ToString("R", CultureInfo.InvariantCulture),16}");
        }

        Console.WriteLine();
    }

    /// <summary>
    /// Dirties the process the way a real shard does: build several sibling models at
    /// DIFFERENT widths, run a forward through each, and drop them. Their scratch
    /// buffers go back to the shared pool carrying residue, so the next model's rents
    /// are no longer freshly-zeroed GC memory. This is the single environmental
    /// difference between the failing shard and every clean-room run that found nothing.
    /// </summary>
    private static void PolluteProcess(int models)
    {
        int[] widths = [16, 24, 32, 48];
        for (int i = 0; i < models; i++)
        {
            int w = widths[i % widths.Length];
            using var arena = TensorArena.Create();
            using var m = new AiDotNet.Diffusion.StyleTransfer.StyDiffModel<float>(
                predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                    inputChannels: 4, outputChannels: 4, baseChannels: w,
                    channelMultipliers: [1, 2, 4], numResBlocks: 1,
                    attentionResolutions: [1, 2], contextDim: 768, seed: 7 + i),
                vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                    inputChannels: 3, latentChannels: 4, baseChannels: 16,
                    channelMultipliers: [1, 2], numResBlocksPerLevel: 1, seed: 7 + i),
                seed: 7 + i);

            var rng = RandomHelper.CreateSeededRandom(100 + i);
            m.Predict(CreateRandomTensor(InputShape, rng));
        }
    }

    private static void LocaliseOnce(Tensor<float> input)
    {
        using var arena = TensorArena.Create();
        using var model = CreateModel();
        model.Predict(input);
        using var clone = (IDiffusionModel<float>)model.Clone();
        clone.Predict(input);
        LocaliseByStage(model, clone, input);
    }

    private static int[] ParseInts(string csv)
    {
        var parts = csv.Split(',', StringSplitOptions.RemoveEmptyEntries
            | StringSplitOptions.TrimEntries);
        var result = new System.Collections.Generic.List<int>(parts.Length);
        foreach (var p in parts)
        {
            if (int.TryParse(p, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0)
                result.Add(v);
        }

        return result.Count > 0 ? result.ToArray() : [Environment.ProcessorCount];
    }

    private static string ArgStr(string[] args, string name, string fallback)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.Ordinal))
                return args[i + 1];
        }

        return fallback;
    }

    private static int ArgInt(string[] args, string name, int fallback)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.Ordinal)
                && int.TryParse(args[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
            {
                return v;
            }
        }

        return fallback;
    }
}
