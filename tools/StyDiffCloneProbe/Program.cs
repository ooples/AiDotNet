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
        ProbeConvolutionPlatformArms();

        var rng = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomTensor(InputShape, rng);

        LocaliseOnce(input);

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
