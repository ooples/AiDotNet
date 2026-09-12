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
        int iterations = ArgInt(args, "--iterations", 50);

        ConfigureLikeTestAssembly();
        PrintBanner(iterations);
        ProbeConvolutionPlatformArms();

        var rng = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomTensor(InputShape, rng);

        int cloneDiverged = 0, selfDiverged = 0, warmMatched = 0;
        double worstCloneDiff = 0.0, worstSelfDiff = 0.0;

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

            if (cloneCmp.MaxDiff > worstCloneDiff) worstCloneDiff = cloneCmp.MaxDiff;
            if (selfCmp.MaxDiff > worstSelfDiff) worstSelfDiff = selfCmp.MaxDiff;

            if (cloneCmp.MaxDiff > 0) cloneDiverged++;
            if (selfCmp.MaxDiff > 0) selfDiverged++;
            if (warmCmp.MaxDiff == 0) warmMatched++;

            bool interesting = cloneCmp.MaxDiff > 0 || selfCmp.MaxDiff > 0;
            if (interesting)
            {
                Console.WriteLine($"--- iteration {iter}: DIVERGENCE ---");
                Report("P1-vs-C1 (the test assertion)", cloneCmp, p1);
                Report("P1-vs-P2 (same model twice)  ", selfCmp, p1);
                Report("P2-vs-C1 (warm orig vs clone)", warmCmp, p2);

                if (iter == 0 || cloneDiverged == 1)
                    LocaliseByStage(model, clone, input);
            }
            else if (iter == 0)
            {
                Console.WriteLine($"iteration 0: all three comparisons bit-identical "
                    + $"(maxAbs |out|={cloneCmp.RefMax.ToString("R", CultureInfo.InvariantCulture)})");
                LocaliseByStage(model, clone, input);
            }
        }

        Console.WriteLine();
        Console.WriteLine("=== SUMMARY ===");
        Console.WriteLine($"iterations                 : {iterations}");
        Console.WriteLine($"P1!=C1 (clone diverged)    : {cloneDiverged}");
        Console.WriteLine($"P1!=P2 (same model twice)  : {selfDiverged}");
        Console.WriteLine($"P2==C1 (warm orig == clone): {warmMatched}");
        Console.WriteLine($"worst |P1-C1|              : {worstCloneDiff.ToString("R", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"worst |P1-P2|              : {worstSelfDiff.ToString("R", CultureInfo.InvariantCulture)}");
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
        if (AiDotNetEngine.Current is not CpuEngine)
            AiDotNetEngine.ResetToCpu();

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
