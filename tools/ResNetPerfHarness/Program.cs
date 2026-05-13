using System;
using System.Collections.Generic;
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
    // Models BuildFloat() actually wires up. Kept in sync with the switch
    // in BuildFloat() — any new case there must add its key here, or the
    // pre-dispatch validation in Main() will reject it before ever calling
    // RunFloat(), which is exactly what we want for unsupported names.
    private static readonly HashSet<string> SupportedFloatModels =
        new(StringComparer.Ordinal) { "resnet50", "vgg11", "vgg16bn" };

    private const string UsageText =
        "Usage: ResNetPerfHarness [--warmup N] [--iters N] [--model NAME] [--dtype double|float]\n" +
        "  --warmup N   Number of warm-up training iterations (default: 1).\n" +
        "  --iters  N   Number of measured training iterations (default: 3).\n" +
        "  --model NAME One of: resnet50, vgg11, vgg16bn, hope, sgpt, siglip2-ctor, sd15-ctor, t5xxl-ctor (default: resnet50).\n" +
        "  --dtype TYPE One of: double, float (default: double). Models that support float run via the fused-compiled\n" +
        "               training path which is typically 5-10× faster than the eager autograd tape on CNNs.\n" +
        "  --help, -h   Show this message and exit.";

    private static int Main(string[] args)
    {
        int warmup = 1;
        int iters = 3;
        string model = "resnet50";
        string dtype = "double";
        for (int i = 0; i < args.Length; i++)
        {
            string flag = args[i];
            switch (flag)
            {
                case "--help":
                case "-h":
                case "/?":
                    Console.WriteLine(UsageText);
                    return 0;
                case "--warmup":
                    if (!TryTakeIntArg(args, ref i, flag, out warmup, minInclusive: 0)) return 2;
                    break;
                case "--iters":
                    if (!TryTakeIntArg(args, ref i, flag, out iters, minInclusive: 1)) return 2;
                    break;
                case "--model":
                    if (!TryTakeStringArg(args, ref i, flag, out var modelArg)) return 2;
                    model = modelArg.ToLowerInvariant();
                    break;
                case "--dtype":
                    if (!TryTakeStringArg(args, ref i, flag, out var dtypeArg)) return 2;
                    dtype = dtypeArg.ToLowerInvariant();
                    if (dtype != "double" && dtype != "float")
                    {
                        Console.Error.WriteLine($"[harness] error: --dtype expects 'double' or 'float', got '{dtype}'.");
                        Console.Error.WriteLine(UsageText);
                        return 2;
                    }
                    break;
                default:
                    Console.Error.WriteLine($"[harness] error: unknown argument '{flag}'.");
                    Console.Error.WriteLine(UsageText);
                    return 2;
            }
        }

        if (dtype == "float")
        {
            if (!SupportedFloatModels.Contains(model))
            {
                Console.Error.WriteLine(
                    $"[harness] error: --dtype float does not support model '{model}'. " +
                    $"Supported: {string.Join(", ", SupportedFloatModels)}.");
                Console.Error.WriteLine(UsageText);
                return 2;
            }
            return RunFloat(model, warmup, iters);
        }

        Console.WriteLine($"[harness] model={model} warmup={warmup} iters={iters} dtype={dtype}");
        Console.WriteLine(AiDotNet.Diagnostics.AccelerationDiagnostics.GetReport());
        Console.WriteLine(
            $"[harness] TensorCodecOptions.EnableCompilation = " +
            $"{AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation}");

        // Surface CompiledTapeTrainingStep.TryStepWithFusedOptimizer Trace
        // warnings to the console so fused-path compile failures aren't
        // silently swallowed during a harness run.
        Trace.Listeners.Add(new ConsoleTraceListener());

        // ctor-probe modes measure construction cost only and exit early
        // BEFORE the harness's TensorArena / Build / training-loop setup
        // runs. Keeping them in Main means the timing doesn't include
        // unrelated arena setup and no resources are allocated that
        // would be skipped by the explicit Exit() path inside Build.
        if (TryRunCtorProbe(model))
        {
            return 0;
        }

        using var arena = TensorArena.Create();
        var (net, input, target) = Build(model);
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

        // Diagnostic: did the fused-compiled training path engage? Non-zero
        // count = forward+backward+optimizer-step ran inside the compiled plan.
        // Zero count after iters Train() calls = every step fell back to the
        // eager autograd-tape path, which is typically 5-10x slower for
        // paper-scale CNNs and is the single biggest perf knob on these
        // tests (see VGGNetwork bottleneck analysis on PR #1299).
        long fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<double>.GetFusedStepCount();
        Console.WriteLine(
            fusedSteps > 0
                ? $"[harness] fused-compiled training path engaged on {fusedSteps} step(s) of {warmup + iters} total"
                : $"[harness] fused-compiled training path did NOT engage — every step ran on the eager tape (see TryStepWithFusedOptimizer gates)");

        return 0;
    }

    private static int RunFloat(string model, int warmup, int iters)
    {
        Console.WriteLine($"[harness] model={model} warmup={warmup} iters={iters} dtype=float");
        Console.WriteLine(
            $"[harness] TensorCodecOptions.EnableCompilation = " +
            $"{AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation}");
        Console.WriteLine(AiDotNet.Diagnostics.AccelerationDiagnostics.GetReport());
        Trace.Listeners.Add(new ConsoleTraceListener());
        using var arena = TensorArena.Create();
        var (net, input, target) = BuildFloat(model);
        Console.WriteLine($"[harness] ParameterCount={net.ParameterCount}, Layers={net.Layers.Count}");

        double L2() { double s = 0; foreach (var c in net.GetParameterChunks()) for (int i = 0; i < c.Length; i++) s += (double)c[i] * c[i]; return Math.Sqrt(s); }
        double LossNow()
        {
            var o = net.Predict(input);
            double s = 0; int len = Math.Min(o.Length, target.Length);
            for (int i = 0; i < len; i++) { double d = (double)o[i] - target[i]; s += d * d; }
            return s / len;
        }
        bool HasNaN()
        {
            foreach (var c in net.GetParameterChunks())
                for (int i = 0; i < c.Length; i++) if (float.IsNaN(c[i]) || float.IsInfinity(c[i])) return true;
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

        long fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
        Console.WriteLine(
            fusedSteps > 0
                ? $"[harness] fused-compiled training path engaged on {fusedSteps} step(s) of {warmup + iters} total"
                : $"[harness] fused-compiled training path did NOT engage — every step ran on the eager tape (see TryStepWithFusedOptimizer gates)");

        return 0;
    }

    /// <summary>
    /// Runs the *-ctor probe modes and returns <c>true</c> when one fired.
    /// Probe modes measure construction wall-time only — they intentionally
    /// run BEFORE the harness's <see cref="TensorArena"/> + Build + train-loop
    /// setup so the measured cost reflects the constructor alone, and so no
    /// arena resources are leaked by an early process exit.
    /// </summary>
    private static bool TryRunCtorProbe(string model)
    {
        switch (model)
        {
            case "siglip2-ctor":
            {
                var sw = Stopwatch.StartNew();
                _ = new AiDotNet.Diffusion.Conditioning.SigLIP2TextConditioner<double>();
                sw.Stop();
                Console.WriteLine($"[harness] SigLIP2TextConditioner ctor: {sw.ElapsedMilliseconds} ms");
                return true;
            }
            case "sd15-ctor":
            {
                var sw = Stopwatch.StartNew();
                _ = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
                sw.Stop();
                Console.WriteLine($"[harness] StableDiffusion15Model ctor: {sw.ElapsedMilliseconds} ms");
                return true;
            }
            case "t5xxl-ctor":
            {
                var sw = Stopwatch.StartNew();
                _ = new AiDotNet.Diffusion.Conditioning.T5TextConditioner<double>("T5-XXL");
                sw.Stop();
                Console.WriteLine($"[harness] T5TextConditioner(T5-XXL) ctor: {sw.ElapsedMilliseconds} ms");
                return true;
            }
            default:
                return false;
        }
    }

    private static (NeuralNetworkBase<float> net, Tensor<float> input, Tensor<float> target) BuildFloat(string model)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        switch (model)
        {
            case "vgg16bn":
            {
                var net = new VGGNetwork<float>();
                var input = new Tensor<float>(new[] { 1, 3, 224, 224 });
                for (int i = 0; i < input.Length; i++) input[i] = (float)rng.NextDouble();
                var target = new Tensor<float>(new[] { 1000 });
                for (int i = 0; i < target.Length; i++) target[i] = (float)rng.NextDouble();
                return (net, input, target);
            }
            case "vgg11":
            {
                var arch = new NeuralNetworkArchitecture<float>(
                    inputType: InputType.ThreeDimensional,
                    taskType: NeuralNetworkTaskType.MultiClassClassification,
                    inputHeight: 32, inputWidth: 32, inputDepth: 3,
                    outputSize: 10);
                var config = VGGConfiguration.CreateForCIFAR(VGGVariant.VGG11, numClasses: 10);
                var net = new VGGNetwork<float>(arch, config);
                var input = new Tensor<float>(new[] { 1, 3, 32, 32 });
                for (int i = 0; i < input.Length; i++) input[i] = (float)rng.NextDouble();
                var target = new Tensor<float>(new[] { 10 });
                for (int i = 0; i < target.Length; i++) target[i] = (float)rng.NextDouble();
                return (net, input, target);
            }
            case "resnet50":
            {
                var net = new ResNetNetwork<float>();
                var input = new Tensor<float>(new[] { 1, 3, 224, 224 });
                for (int i = 0; i < input.Length; i++) input[i] = (float)rng.NextDouble();
                var target = new Tensor<float>(new[] { 1000 });
                for (int i = 0; i < target.Length; i++) target[i] = (float)rng.NextDouble();
                return (net, input, target);
            }
            default:
                throw new ArgumentException($"BuildFloat does not yet wire up model '{model}'. Add a case mirroring the Build switch.");
        }
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
            case "vgg16bn":
            {
                // Paper-scale VGG16-BN per Simonyan & Zisserman 2014: 224×224×3
                // ImageNet input → 1000 classes. Same shape the parameterless
                // VGGNetwork() ctor produces and the same shape the model-family
                // invariant tests train against. Measures the production hot path,
                // not a CIFAR-shrunk surrogate.
                //
                // Note: T=double currently bails out of the fused-compiled
                // training path because AiDotNet.Tensors 0.75.5
                // CompiledTrainingPlan<T>.TryBuildSpecializedForward casts
                // Tensor<double> to Tensor<float> unconditionally
                // (InvalidCastException at runtime). Use the `vgg16bn-float`
                // case instead to measure the fused-path performance.
                var net = new VGGNetwork<double>();
                var input = new Tensor<double>(new[] { 1, 3, 224, 224 });
                for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
                var target = new Tensor<double>(new[] { 1000 });
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
            // Note: the "*-ctor" probe modes are handled in Main via
            // TryRunCtorProbe before this switch runs. Falling through to the
            // default below would mean a probe model was requested AFTER the
            // ctor probe already exited — unreachable in practice, but
            // listed here so the switch is exhaustive for documented values.
            default:
                throw new ArgumentException($"Unknown model: {model}");
        }
    }

    /// <summary>
    /// Consumes the value following <paramref name="flag"/> in
    /// <paramref name="args"/>, parses it as an integer with an optional
    /// minimum bound, and emits a clear diagnostic + non-zero exit code
    /// when the value is missing or malformed. Advances
    /// <paramref name="i"/> past the consumed value on success.
    /// </summary>
    private static bool TryTakeIntArg(string[] args, ref int i, string flag, out int value, int minInclusive)
    {
        value = 0;
        if (i + 1 >= args.Length)
        {
            Console.Error.WriteLine($"[harness] error: '{flag}' requires a value.");
            Console.Error.WriteLine(UsageText);
            return false;
        }
        i++;
        if (!int.TryParse(args[i], System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out value))
        {
            Console.Error.WriteLine($"[harness] error: '{flag}' expects an integer, got '{args[i]}'.");
            Console.Error.WriteLine(UsageText);
            return false;
        }
        if (value < minInclusive)
        {
            Console.Error.WriteLine($"[harness] error: '{flag}' must be >= {minInclusive}, got {value}.");
            Console.Error.WriteLine(UsageText);
            return false;
        }
        return true;
    }

    private static bool TryTakeStringArg(string[] args, ref int i, string flag, out string value)
    {
        value = string.Empty;
        if (i + 1 >= args.Length)
        {
            Console.Error.WriteLine($"[harness] error: '{flag}' requires a value.");
            Console.Error.WriteLine(UsageText);
            return false;
        }
        i++;
        value = args[i];
        return true;
    }
}
