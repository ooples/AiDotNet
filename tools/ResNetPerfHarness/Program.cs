using System;
using System.Diagnostics;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.ResNetPerfHarness;

/// <summary>
/// Lightweight perf-timing harness for paper-scale CNN training paths
/// (ResNet50 @ 224×224 / VGG11 @ 32×32). Used to validate the
/// BlasEnvDefault ModuleInitializer's effect on training-step wall time
/// and to provide a profileable, deterministic target for dotnet-trace.
/// Runs <c>--warmup</c> iterations first (lazy weight allocation, first-
/// forward JIT) then reports per-iteration timings + average for
/// <c>--iters</c> measured iterations.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        int warmup = 1;
        int iters = 3;
        string model = "resnet50";
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--warmup": warmup = int.Parse(args[++i]); break;
                case "--iters": iters = int.Parse(args[++i]); break;
                case "--model": model = args[++i].ToLowerInvariant(); break;
            }
        }

        Console.WriteLine($"[harness] model={model} warmup={warmup} iters={iters}");
        var (net, input, target) = Build(model);
        Console.WriteLine($"[harness] ParameterCount={net.ParameterCount}, Layers={net.Layers.Count}");

        for (int w = 0; w < warmup; w++)
        {
            var sw = Stopwatch.StartNew();
            net.Train(input, target);
            sw.Stop();
            Console.WriteLine($"[harness] warmup#{w + 1}: {sw.ElapsedMilliseconds} ms");
        }

        long total = 0;
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            net.Train(input, target);
            sw.Stop();
            total += sw.ElapsedMilliseconds;
            Console.WriteLine($"[harness] iter#{i + 1}: {sw.ElapsedMilliseconds} ms");
        }
        double avg = (double)total / iters;
        Console.WriteLine($"[harness] avg over {iters} iters: {avg:F1} ms");
        return 0;
    }

    private static (NeuralNetworkBase<double> net, Tensor<double> input, Tensor<double> target) Build(string model)
    {
        // Use RandomHelper for crypto-grade RNG (CreateSeededRandom is the
        // reproducible variant — the harness wants deterministic input/target
        // tensors so per-iter timings are comparable across runs).
        var rng = RandomHelper.CreateSeededRandom(42);
        switch (model)
        {
            case "resnet50":
            {
                var net = new ResNetNetwork<double>();
                var input = new Tensor<double>(new[] { 1, 3, 224, 224 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                var target = new Tensor<double>(new[] { 1000 });
                for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();
                return (net, input, target);
            }
            case "vgg11":
            {
                var arch = new NeuralNetworkArchitecture<double>(
                    inputType: AiDotNet.Enums.InputType.ThreeDimensional,
                    taskType: AiDotNet.Enums.NeuralNetworkTaskType.MultiClassClassification,
                    inputHeight: 32, inputWidth: 32, inputDepth: 3,
                    outputSize: 10);
                var config = AiDotNet.Configuration.VGGConfiguration.CreateForCIFAR(
                    AiDotNet.Enums.VGGVariant.VGG11, numClasses: 10);
                var net = new VGGNetwork<double>(arch, config);
                var input = new Tensor<double>(new[] { 1, 3, 32, 32 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                var target = new Tensor<double>(new[] { 10 });
                for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();
                return (net, input, target);
            }
            default:
                throw new ArgumentException($"Unknown model: {model}");
        }
    }
}
