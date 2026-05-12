using System;
using System.Diagnostics;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.ResNetPerfHarness;

internal static class Program
{
    private static int Main(string[] args)
    {
        int warmup = 1;
        int iters = 3;
        string model = "resnet50";
        for (int i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            string TakeValue(string flag)
            {
                if (i + 1 >= args.Length)
                {
                    Console.Error.WriteLine($"[harness] {flag} requires a value");
                    System.Environment.Exit(2);
                }
                return args[++i];
            }
            switch (arg)
            {
                case "--warmup":
                    if (!int.TryParse(TakeValue(arg), out warmup) || warmup < 0)
                    {
                        Console.Error.WriteLine("[harness] --warmup must be a non-negative integer");
                        return 2;
                    }
                    break;
                case "--iters":
                    if (!int.TryParse(TakeValue(arg), out iters) || iters < 1)
                    {
                        Console.Error.WriteLine("[harness] --iters must be a positive integer (>= 1)");
                        return 2;
                    }
                    break;
                case "--model":
                    model = TakeValue(arg).ToLowerInvariant();
                    break;
                case "-h":
                case "--help":
                    Console.WriteLine("Usage: ResNetPerfHarness [--model <name>] [--warmup <N>] [--iters <N>]");
                    Console.WriteLine("  --model    one of: resnet50, vgg11, hope, sgpt, siglip2-ctor, sd15-ctor, t5xxl-ctor");
                    Console.WriteLine("  --warmup   non-negative integer (default: 1)");
                    Console.WriteLine("  --iters    positive integer (default: 3)");
                    return 0;
                default:
                    Console.Error.WriteLine($"[harness] unknown flag: {arg} (use --help)");
                    return 2;
            }
        }

        Console.WriteLine($"[harness] model={model} warmup={warmup} iters={iters}");
        // Ctor-only probes (siglip2-ctor / sd15-ctor / t5xxl-ctor) measure
        // constructor wall-clock and exit. Handle them BEFORE Build() so the
        // builder stays a pure factory (model, input, target) instead of
        // mixing Environment.Exit into the middle of the model selector.
        if (TryRunCtorProbe(model)) return 0;
        using var arena = TensorArena.Create();
        var (net, input, target) = Build(model);
        // Dispose the network when we're done: NeuralNetworkBase<T> owns
        // potentially large managed buffers and pooled native handles
        // (engine workspaces, BLAS scratch, parameter tensor lifetimes).
        // Even though this is a short-lived tool, an explicit using
        // keeps the timing window honest by releasing those resources
        // before the process exits and the runtime tears them down.
        using var netHandle = net;
        Console.WriteLine($"[harness] ParameterCount={net.ParameterCount}, Layers={net.Layers.Count}");

        double L2() { double s = 0; foreach (var c in net.GetParameterChunks()) for (int i = 0; i < c.Length; i++) s += c[i] * c[i]; return Math.Sqrt(s); }
        double LossNow()
        {
            var o = net.Predict(input);
            double s = 0; int len = Math.Min(o.Length, target.Length);
            for (int i = 0; i < len; i++) { double d = o[i] - target[i]; s += d * d; }
            return s / len;
        }
        bool HasNaN()
        {
            foreach (var c in net.GetParameterChunks())
                for (int i = 0; i < c.Length; i++) if (double.IsNaN(c[i]) || double.IsInfinity(c[i])) return true;
            return false;
        }

        Console.WriteLine($"[harness] init: L2={L2():F4} loss={LossNow():F6}");
        for (int w = 0; w < warmup; w++)
        {
            var sw = Stopwatch.StartNew();
            net.Train(input, target);
            sw.Stop();
            Console.WriteLine($"[harness] warmup#{w + 1}: {sw.ElapsedMilliseconds} ms  L2={L2():F4}  loss={LossNow():F6}  hasNaN={HasNaN()}");
        }

        long total = 0;
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            net.Train(input, target);
            sw.Stop();
            total += sw.ElapsedMilliseconds;
            Console.WriteLine($"[harness] iter#{i + 1}: {sw.ElapsedMilliseconds,5} ms  L2={L2():F4}  loss={LossNow():F6}  hasNaN={HasNaN()}");
        }
        double avg = (double)total / iters;
        Console.WriteLine($"[harness] avg over {iters} iters: {avg:F1} ms");
        return 0;
    }

    private static (NeuralNetworkBase<double> net, Tensor<double> input, Tensor<double> target) Build(string model)
    {
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
                    inputType: InputType.ThreeDimensional,
                    taskType: NeuralNetworkTaskType.MultiClassClassification,
                    inputHeight: 32, inputWidth: 32, inputDepth: 3,
                    outputSize: 10);
                var config = VGGConfiguration.CreateForCIFAR(VGGVariant.VGG11, numClasses: 10);
                var net = new VGGNetwork<double>(arch, config);
                var input = new Tensor<double>(new[] { 1, 3, 32, 32 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                var target = new Tensor<double>(new[] { 10 });
                for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();
                return (net, input, target);
            }
            case "hope":
            {
                var net = new HopeNetwork<double>();
                var input = new Tensor<double>(new[] { 256 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                var target = new Tensor<double>(new[] { 256 });
                for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();
                return (net, input, target);
            }
            case "sgpt":
            {
                var net = new SGPT<double>();
                var input = new Tensor<double>(new[] { 1, 4 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                var target = new Tensor<double>(new[] { 1, 1 });
                for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();
                return (net, input, target);
            }
            default:
                throw new ArgumentException($"Unknown model: {model}");
        }
    }

    /// <summary>
    /// Handles the "*-ctor" probe modes that only need to measure a
    /// constructor's wall-clock and exit. Returns true if a probe ran
    /// (the caller should then return its own exit code), false if the
    /// model name isn't a ctor probe (the caller falls through to Build
    /// + training loop). Keeps Build() as a pure (model, input, target)
    /// factory by lifting the side-effects + early-exit out.
    /// </summary>
    private static bool TryRunCtorProbe(string model)
    {
        switch (model)
        {
            case "siglip2-ctor":
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                _ = new AiDotNet.Diffusion.Conditioning.SigLIP2TextConditioner<double>();
                sw.Stop();
                Console.WriteLine($"[harness] SigLIP2TextConditioner ctor: {sw.ElapsedMilliseconds} ms");
                return true;
            }
            case "sd15-ctor":
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                _ = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
                sw.Stop();
                Console.WriteLine($"[harness] StableDiffusion15Model ctor: {sw.ElapsedMilliseconds} ms");
                return true;
            }
            case "t5xxl-ctor":
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                _ = new AiDotNet.Diffusion.Conditioning.T5TextConditioner<double>("T5-XXL");
                sw.Stop();
                Console.WriteLine($"[harness] T5TextConditioner(T5-XXL) ctor: {sw.ElapsedMilliseconds} ms");
                return true;
            }
            default:
                return false;
        }
    }
}
