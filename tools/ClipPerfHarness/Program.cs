using System;
using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Encoders;
using AiDotNet.VisionLanguage.Robotics;
using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.Tools.ClipPerfHarness;

/// <summary>
/// Stand-alone harness for profiling a single CLIP / Hawk training step
/// under PerfView / dotnet-trace. Mirrors what the auto-generated invariant
/// tests do (CreateNetwork → CreateRandomTensor(InputShape) → Train(input,
/// target) → Predict(input)) but in a process-level entrypoint we can
/// attach a sampling profiler to.
///
/// Usage:
///   dotnet run -c Release --project tools/ClipPerfHarness -- biomed
///   dotnet run -c Release --project tools/ClipPerfHarness -- dfn
///   dotnet run -c Release --project tools/ClipPerfHarness -- hawk
///
/// To capture a CPU profile that opens in PerfView:
///   dotnet-trace collect --providers Microsoft-DotNETCore-SampleProfiler \
///     --output traces/&lt;mode&gt;-train.nettrace --duration 00:00:30 -- \
///     dotnet tools/ClipPerfHarness/bin/Release/net10.0/ClipPerfHarness.dll &lt;mode&gt;
/// </summary>
internal enum HarnessMode
{
    BiomedClip,
    DfnClip,
    Hawk,
    Vit,
    Helix,
    Gpt4Point,
}

internal static class Program
{
    private static int Main(string[] args)
    {
        // Match the test assembly's ModuleInitializer (TestAssemblyDeterminismInit):
        // ModelFamily invariant tests run CPU-only (AIDOTNET_DISABLE_GPU=1 +
        // AiDotNetEngine.ResetToCpu()). Without this the harness silently runs on
        // the DirectGpu/OpenCL engine, whose per-op host↔device copy + double→float
        // conversion is a completely different (and misleading) hot path.
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
        AiDotNet.Tensors.Engines.AiDotNetEngine.ResetToCpu();

        HarnessMode mode = ParseMode(args);

        var swCtor = Stopwatch.StartNew();
        NeuralNetworkBase<double> network;
        var rng = new Random(42);
        Tensor<double> input;

        switch (mode)
        {
            case HarnessMode.Hawk:
            {
                // Hawk language model: 1D input [128] (language domain in TestScaffoldGenerator).
                var arch = new NeuralNetworkArchitecture<double>(
                    inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputSize: 128, outputSize: 4);
                Console.WriteLine($"[mode={mode}] Constructing Hawk language model...");
                network = new HawkLanguageModel<double>(arch);
                input = new Tensor<double>(new[] { 128 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                break;
            }
            case HarnessMode.DfnClip:
            case HarnessMode.BiomedClip:
            {
                var arch = new NeuralNetworkArchitecture<double>(
                    inputType: InputType.ThreeDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputHeight: 128, inputWidth: 128, inputDepth: 3, outputSize: 4);
                Console.WriteLine($"[mode={mode}] Constructing CLIP model...");
                network = mode == HarnessMode.DfnClip
                    ? new DFNCLIP<double>(arch)
                    : new BiomedCLIP<double>(arch);
                input = new Tensor<double>(new[] { 3, 128, 128 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                break;
            }
            case HarnessMode.Vit:
            {
                // VisionTransformer (Dosovitskiy et al. 2021): paper-default
                // ViT-Base config — 224×224×3 input, 16-px patches, hidden=768,
                // 12 layers, 12 heads, mlp=3072. Same shape the
                // VisionTransformerTests fixture uses.
                Console.WriteLine($"[mode={mode}] Constructing VisionTransformer (ViT-Base, 224x224)...");
                network = new VisionTransformer<double>();
                input = new Tensor<double>(new[] { 1, 3, 224, 224 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                break;
            }
            case HarnessMode.Helix:
            {
                // Helix (Figure AI 2025): dual-system VLA. Native layer chain
                // consumes post-patch-embedding token features [1, 4, VisionDim=1024]
                // and runs vision encoder + System-2 VLM decoder (DecoderDim=4096,
                // 32 layers) + System-1 visuomotor transformer → action head.
                // Paper-scale ~7B params: this is the >120s double train step we
                // are profiling.
                var arch = new NeuralNetworkArchitecture<double>(
                    inputType: InputType.ThreeDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 4);
                Console.WriteLine($"[mode={mode}] Constructing Helix (paper-scale 7B VLA)...");
                network = new Helix<double>(arch);
                input = new Tensor<double>(new[] { 1, 4, 1024 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                break;
            }
            case HarnessMode.Gpt4Point:
            {
                // GPT4Point (Qi et al. 2024): point-language VLM. Native layer
                // chain consumes token features [1, 4, VisionDim=512] and runs the
                // point-cloud VLM encoder + Q-Former + LLM decoder (DecoderDim=4096,
                // 32 layers). Paper-scale ~7B params.
                var arch = new NeuralNetworkArchitecture<double>(
                    inputType: InputType.ThreeDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 4);
                Console.WriteLine($"[mode={mode}] Constructing GPT4Point (paper-scale 7B VLA)...");
                network = new GPT4Point<double>(arch);
                input = new Tensor<double>(new[] { 1, 4, 512 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                break;
            }
            default:
                throw new InvalidOperationException($"Unknown harness mode {mode}");
        }
        swCtor.Stop();
        Console.WriteLine($"  ctor: {swCtor.ElapsedMilliseconds} ms");

        // Warm-up forward (matches the EffectiveOutputShape warm-up the
        // test base does). Pays first-touch lazy-init costs we don't want
        // to attribute to the training step.
        Console.WriteLine("[warmup] Predict...");
        var swWarm = Stopwatch.StartNew();
        var warmOut = network.Predict(input);
        swWarm.Stop();
        Console.WriteLine($"  predict: {swWarm.ElapsedMilliseconds} ms, output rank={warmOut.Rank}, length={warmOut.Length}");

        // Build target with the inferred output shape.
        var outShape = warmOut.Shape;
        var outDims = new int[outShape.Length];
        for (int i = 0; i < outShape.Length; i++) outDims[i] = outShape[i];
        var target = new Tensor<double>(outDims);
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();

        // Paper-scale 7B VLA models (Helix, GPT4Point) take >100 s per double
        // train step, so the 5-step + 2×10-rep benchmark below would run for
        // hours. For these we run a SINGLE train step (enough for a sampling
        // profiler to attribute the hot path) and skip the replay benchmarks.
        bool heavy = mode is HarnessMode.Helix or HarnessMode.Gpt4Point;

        // Forward-only profiling path for the paper-scale 7B VLA models: a full
        // double train step exhausts memory (Adam moments ≈ 2× the 51 GB weight
        // set), so to attribute the hot path we profile repeated Predicts — the
        // forward already runs at ~4 GFLOP/s (10-40× under a many-core box), so
        // its hot methods ARE the weak point to fix. Set HELIX_FWD_ONLY=1.
        if (heavy && Environment.GetEnvironmentVariable("HELIX_FWD_ONLY") == "1")
        {
            Console.WriteLine("[fwd-only] Predict x4 (profiling forward hot path)");
            for (int i = 0; i < 4; i++)
            {
                var swf = Stopwatch.StartNew();
                _ = network.Predict(input);
                swf.Stop();
                Console.WriteLine($"  predict {i}: {swf.ElapsedMilliseconds} ms");
            }
            return 0;
        }

        int trainReps = heavy ? 1 : 5;

        // Train trainReps steps in a row — first step pays compile cost,
        // subsequent steps replay the compiled plan. This matches the test's
        // invariant pattern: warm probe + N training iterations + verify predict.
        Console.WriteLine($"[train] Train(input, target) — {trainReps} step(s)");
        var trainStepMs = new long[trainReps];
        for (int s = 0; s < trainReps; s++)
        {
            var sw = Stopwatch.StartNew();
            network.Train(input, target);
            sw.Stop();
            trainStepMs[s] = sw.ElapsedMilliseconds;
            Console.WriteLine($"  train step {s}: {trainStepMs[s]} ms");
        }
        long swTrainTotal = 0;
        for (int s = 0; s < trainReps; s++) swTrainTotal += trainStepMs[s];

        if (heavy)
        {
            Console.WriteLine();
            Console.WriteLine($"SUMMARY [{mode}]:");
            Console.WriteLine($"  ctor:    {swCtor.ElapsedMilliseconds} ms");
            Console.WriteLine($"  warm:    {swWarm.ElapsedMilliseconds} ms");
            Console.WriteLine($"  train:   {swTrainTotal} ms ({trainReps} step)");
            return 0;
        }

        // Sub-phase break-down for steady-state cost: forward (Predict)
        // versus full Train (forward + backward + optimizer step). Helps
        // pin down whether the remaining time is dominated by backward
        // (gradient computation through the LM head) or by the optimizer
        // path. Adam fast-path is now SIMD-fused (4.4× steady-state
        // speedup measured on Hawk under net10), so anything still
        // hot is upstream of the optimizer.
        Console.WriteLine("[bench] sub-phase break-down (10 reps each)");
        var swP = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++) _ = network.Predict(input);
        swP.Stop();
        Console.WriteLine($"  predict x10: {swP.ElapsedMilliseconds} ms total ({swP.ElapsedMilliseconds / 10.0:F1} ms/iter)");
        var swT = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++) network.Train(input, target);
        swT.Stop();
        Console.WriteLine($"  train   x10: {swT.ElapsedMilliseconds} ms total ({swT.ElapsedMilliseconds / 10.0:F1} ms/iter)");
        long backwardEstimateMs = (swT.ElapsedMilliseconds - swP.ElapsedMilliseconds) / 10;
        Console.WriteLine($"  ==> backward+optimizer estimate: {backwardEstimateMs} ms/iter");
        var swTrain = Stopwatch.StartNew();
        swTrain.Stop();
        // Recreate the field used by SUMMARY (sum of all 5 step times).
        // This keeps the summary line meaningful: total training wall-clock.
        var trainTotalField = swTrainTotal;

        Console.WriteLine("[post] Predict...");
        var swPost = Stopwatch.StartNew();
        var postOut = network.Predict(input);
        swPost.Stop();
        Console.WriteLine($"  predict: {swPost.ElapsedMilliseconds} ms");

        Console.WriteLine();
        Console.WriteLine($"SUMMARY [{mode}]:");
        Console.WriteLine($"  ctor:    {swCtor.ElapsedMilliseconds} ms");
        Console.WriteLine($"  warm:    {swWarm.ElapsedMilliseconds} ms");
        Console.WriteLine($"  train:   {trainTotalField} ms (5 steps, per-step {string.Join("/", trainStepMs)})");
        Console.WriteLine($"  post:    {swPost.ElapsedMilliseconds} ms");
        Console.WriteLine($"  TOTAL:   {(swCtor.ElapsedMilliseconds + swWarm.ElapsedMilliseconds + trainTotalField + swPost.ElapsedMilliseconds)} ms");

        return 0;
    }

    /// <summary>
    /// Parses the harness mode from a command-line token. Mode names are
    /// matched case-insensitively against a fixed set of aliases per mode
    /// — strings only flow through here at process boundary; everything
    /// downstream uses the <see cref="HarnessMode"/> enum for type safety.
    /// </summary>
    private static HarnessMode ParseMode(string[] args)
    {
        if (args.Length == 0) return HarnessMode.BiomedClip;
        string token = args[0].Trim().ToLowerInvariant();
        return token switch
        {
            "hawk" => HarnessMode.Hawk,
            "dfn" or "dfnclip" or "dfn-clip" => HarnessMode.DfnClip,
            "biomed" or "biomedclip" or "biomed-clip" => HarnessMode.BiomedClip,
            "vit" or "visiontransformer" or "vision-transformer" => HarnessMode.Vit,
            "helix" => HarnessMode.Helix,
            "gpt4point" or "gpt4pt" or "gpt-4-point" => HarnessMode.Gpt4Point,
            _ => throw new ArgumentException(
                $"Unknown mode '{token}'. Valid modes: biomed, dfn, hawk, vit, helix, gpt4point.", nameof(args)),
        };
    }
}
