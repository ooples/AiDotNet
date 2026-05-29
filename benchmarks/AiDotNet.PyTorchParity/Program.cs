// AiDotNet ⇄ PyTorch parity benchmark harness.
//
// A self-contained, in-repo twin of the AIsEval aidotnet-benchmarks project
// (https://github.com/ooples/AIsEval). It builds the SAME four reference
// models — MLP / CNN / LSTM / Transformer — with paper-matched layer shapes,
// runs the SAME training + multi-batch inference measurement loop, and emits
// the SAME JSON schema as the PyTorch side (pytorch/benchmark.py), so the two
// reports drop straight into the same comparison.
//
// Why this lives in the AiDotNet repo (vs only in AIsEval): it references the
// working-tree AiDotNet *source* (ProjectReference, not a NuGet package), so it
// measures the CURRENT branch. That makes it the validation harness for changes
// like PR #1469 (default-Adam fused-training gate) and the
// FeedForwardNeuralNetwork.Predict -> IEngine.MlpForward fused-inference wiring,
// which a released-package benchmark cannot see until a publish.
//
// Usage:
//   dotnet run -c Release --project benchmarks/AiDotNet.PyTorchParity -- \
//       --models mlp,cnn,lstm,transformer,mlp-fused --output results/aidotnet.json
// Then run the PyTorch side (see pytorch/README in this folder) and compare.

using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using System.Diagnostics;
using System.Text.Json;

// Fair comparison vs PyTorch-CPU: force the native CPU engine.
//
// AiDotNet.Tensors ships a [ModuleInitializer] (GpuAutoDetectModuleInit) that
// auto-detects a GPU/OpenCL device at assembly load and switches the global
// engine to DirectGpu/OpenCL. On an integrated-GPU rig that path is SLOWER than
// the native OneDNN/OpenBLAS CPU path for these small-to-medium workloads, and
// is not the path the Tensors micro-benchmarks beat PyTorch-CPU on. ResetToCpu()
// pins the CPU engine so this harness compares CPU-vs-CPU. (AIDOTNET_DISABLE_GPU=1
// is the documented before-startup opt-out; this in-code reset also covers the
// published-DLL path where launchSettings env vars don't apply.)
AiDotNet.Tensors.Engines.AiDotNetEngine.ResetToCpu();
Console.WriteLine($"[bench] engine pinned to CPU: {AiDotNet.Tensors.Engines.AiDotNetEngine.Current.GetType().Name}");

// Opt-in (AIDOTNET_FUSED_DIAG=1): surface whether the compiled fused-optimizer
// training path actually runs (Hit) or silently falls back to the eager tape
// (and why). This is the exact instrument that found the "compiled training does
// nothing" bug fixed by PR #1469: EnableCompilation defaults to true and the
// fused step is ATTEMPTED every step, but if TryMapToFusedOptimizerConfig rejects
// the model's optimizer, every step silently falls back to the eager tape. The
// fallback is invisible at the default Silent diagnostic level — only PerStep
// surfaces the FusedOptimizerPathEvent. With the fix, Hit=True should appear.
var fusedDiag = Environment.GetEnvironmentVariable("AIDOTNET_FUSED_DIAG") == "1";
if (fusedDiag)
{
    AiDotNet.Configuration.TrainingDiagnosticsConfig.Level =
        AiDotNet.Configuration.TrainingDiagnosticLevel.PerStep;
    AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
    var fusedSeen = new HashSet<string>();
    AiDotNet.Configuration.TrainingDiagnosticsConfig.Sink = evt =>
    {
        if (evt is AiDotNet.Configuration.FusedOptimizerPathEvent f)
        {
            var key = $"{f.Hit}:{f.Reason}";
            if (fusedSeen.Add(key))
                Console.WriteLine($"[bench] FUSED-PATH event: Hit={f.Hit} Reason={f.Reason ?? "(none)"}");
        }
    };
}

var benchOptions = BenchmarkOptions.Parse(args);
var report = new BenchmarkRunner(benchOptions).Run();
var outputPath = Path.GetFullPath(benchOptions.OutputPath);
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
File.WriteAllText(outputPath, JsonSerializer.Serialize(report, JsonOptions.Default));
Console.WriteLine($"Benchmark report written to {outputPath}");

if (fusedDiag)
{
    // Answer "did compiled fused training actually run, and if not, why?" from a
    // single run. A step count of 0 with a captured fallback exception is the
    // signature of the "compiled does nothing" failure mode: the FIRST fused
    // step fell back (silently, reason often null when the Tensors plan refuses
    // the model), which sticky-disables the path for the rest of the session.
    var fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
    var lastFallback = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetLastFallbackException();
    Console.WriteLine($"[bench] fused training steps that engaged: {fusedSteps}");
    Console.WriteLine(lastFallback is null
        ? "[bench] last fused-fallback exception: (none captured)"
        : $"[bench] last fused-fallback exception: {lastFallback.GetType().FullName}: {lastFallback.Message}");
}

internal sealed record BenchmarkOptions(
    string[] Models,
    int Epochs,
    int TrainBatches,
    int BatchSize,
    int InferenceIterations,
    int WarmupIterations,
    int Seed,
    string OutputPath)
{
    public static BenchmarkOptions Parse(string[] args)
    {
        // Convert --key value command-line pairs into a simple lookup table so
        // individual options can fall back to sensible defaults when omitted.
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length - 1; i += 2)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal)) map[args[i][2..]] = args[i + 1];
        }

        static int Int(Dictionary<string, string> map, string key, int fallback) =>
            map.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;

        var models = map.GetValueOrDefault("models", "mlp,cnn,lstm,transformer")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        return new BenchmarkOptions(
            models,
            Int(map, "epochs", 3),
            Int(map, "train-batches", 20),
            Int(map, "batch-size", 64),
            Int(map, "inference-iterations", 100),
            Int(map, "warmup-iterations", 10),
            Int(map, "seed", 1234),
            map.GetValueOrDefault("output", "results/aidotnet.json"));
    }
}

internal sealed class BenchmarkRunner(BenchmarkOptions options)
{
    private static readonly int[] InferenceBatchSizes = [1, 8, 32, 128];

    public BenchmarkReport Run()
    {
        // Create each requested model, measure its training and inference
        // phases, and collect those measurements into one framework-level report.
        var factory = new AiDotNetTensorBackend(options.Seed);
        var results = new List<ModelReport>();
        foreach (var modelName in options.Models)
        {
            var modelStart = Stopwatch.StartNew();
            Console.WriteLine($"[bench] {modelName}: building network…");
            var model = factory.Create(modelName);
            Console.WriteLine($"[bench] {modelName}: training ({options.Epochs}e × {options.TrainBatches}b × {options.BatchSize}bs, {model.ParameterCount} params)…");
            var training = BenchmarkTraining(model);
            Console.WriteLine($"[bench] {modelName}: training done in {training.TotalSeconds:F2}s; running inference…");
            var inference = BenchmarkInference(model);
            modelStart.Stop();
            Console.WriteLine($"[bench] {modelName}: complete in {modelStart.Elapsed.TotalSeconds:F2}s");
            results.Add(new ModelReport(modelName, "AiDotNetNeuralNetwork", model.ParameterCount, training, inference));
        }

        return new BenchmarkReport(
            "AiDotNet",
            Environment.Version.ToString(),
            AiDotNetProbe.Describe(),
            results);
    }

    private TrainingReport BenchmarkTraining(IBenchmarkModel model)
    {
        // Track per-epoch duration plus the synthetic data-loading and gradient
        // phases, while a background monitor samples process resources.
        var epochSeconds = new List<double>();
        var gradientSeconds = new List<double>();
        var dataSeconds = new List<double>();
        var total = Stopwatch.StartNew();
        using var monitor = ResourceMonitor.Start();

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            var epochTimer = Stopwatch.StartNew();
            for (var batch = 0; batch < options.TrainBatches; batch++)
            {
                var dataTimer = Stopwatch.StartNew();
                model.LoadSyntheticBatch(options.BatchSize);
                dataTimer.Stop();
                dataSeconds.Add(dataTimer.Elapsed.TotalSeconds);

                model.Forward();
                var gradientTimer = Stopwatch.StartNew();
                model.Backward();
                gradientTimer.Stop();
                gradientSeconds.Add(gradientTimer.Elapsed.TotalSeconds);
                model.Step();
            }
            epochTimer.Stop();
            epochSeconds.Add(epochTimer.Elapsed.TotalSeconds);
        }

        total.Stop();
        return new TrainingReport(
            epochSeconds.Select(Round6).ToArray(),
            Round6(total.Elapsed.TotalSeconds),
            Round6(gradientSeconds.Count == 0 ? 0 : gradientSeconds.Average()),
            Round6(dataSeconds.Count == 0 ? 0 : dataSeconds.Average()),
            monitor.Summary());
    }

    private List<InferenceReport> BenchmarkInference(IBenchmarkModel model)
    {
        // Measure inference at multiple batch sizes so the report captures both
        // latency and throughput behavior under different request shapes.
        var reports = new List<InferenceReport>();
        var process = Process.GetCurrentProcess();
        foreach (var batchSize in InferenceBatchSizes)
        {
            model.LoadSyntheticBatch(batchSize);

            // Warmup iterations prime JIT compilation and tensor internals before
            // steady-state measurements are recorded.
            var warmup = new List<double>();
            for (var i = 0; i < options.WarmupIterations; i++)
            {
                var timer = Stopwatch.StartNew();
                model.Forward();
                timer.Stop();
                warmup.Add(timer.Elapsed.TotalSeconds);
            }

            // RSS via Process.WorkingSet64 mirrors the PyTorch side's
            // psutil rss (whole-process resident set incl. native allocs),
            // so both report the same kind of memory number.
            process.Refresh();
            var peakBefore = process.WorkingSet64 / 1024d / 1024d;
            var steady = new List<double>();
            var peak = peakBefore;

            for (var i = 0; i < options.InferenceIterations; i++)
            {
                var timer = Stopwatch.StartNew();
                model.Forward();
                timer.Stop();
                steady.Add(timer.Elapsed.TotalSeconds);
                process.Refresh();
                peak = Math.Max(peak, process.WorkingSet64 / 1024d / 1024d);
            }
            var totalSteady = steady.Sum();
            // p95 latency (symmetric with the PyTorch side): robust to the
            // rig-contention noise that swings the mean. The Tensors perf gate is
            // p95(ours) < median(PyTorch).
            var steadySorted = steady.OrderBy(x => x).ToList();
            int p95Idx = Math.Min(steadySorted.Count - 1, (int)Math.Round(0.95 * (steadySorted.Count - 1)));
            reports.Add(new InferenceReport(
                batchSize,
                Round6(warmup.Average()),
                Math.Round(steady.Average() * 1000d, 3),
                Math.Round(steadySorted[p95Idx] * 1000d, 3),
                Math.Round(options.InferenceIterations * batchSize / totalSteady, 3),
                Math.Round(peak, 3)));
        }
        return reports;
    }

    private static double Round6(double value) => Math.Round(value, 6);
}

internal interface IBenchmarkModel
{
    long ParameterCount { get; }
    void LoadSyntheticBatch(int batchSize);
    void Forward();
    void Backward();
    void Step();
}

internal sealed class AiDotNetTensorBackend(int seed)
{
    // Each model constructs the real AiDotNet neural-network class with
    // paper-matched layer shapes so the comparison is real-Conv2D vs
    // real-Conv2D, real-LSTM vs real-LSTM, etc. — not "PyTorch real model vs
    // AiDotNet MLP". The shapes mirror pytorch/benchmark.py exactly.
    public IBenchmarkModel Create(string model) => model.ToLowerInvariant() switch
    {
        // PyTorch: nn.Sequential(Linear(784,512), ReLU, Linear(512,128), ReLU, Linear(128,10)).
        "mlp" => new AiDotNetMlpModel(seed),
        // PyTorch: Conv2d(1,16,3,pad=1)+ReLU+MaxPool(2)+Conv2d(16,32,3,pad=1)+ReLU+AdaptiveAvgPool((4,4))+Linear(512,10).
        "cnn" => new AiDotNetCnnModel(seed),
        // PyTorch: nn.LSTM(input=32, hidden=64) + Linear(64, 10). Input [B, 32, 32].
        "lstm" => new AiDotNetLstmModel(seed),
        // PyTorch: Linear(32,64) + 2× TransformerEncoderLayer(d_model=64,nhead=4,dim_ff=128) + mean(seq) + Linear(64,10).
        "transformer" => new AiDotNetTransformerModel(seed),
        // Fused-primitive INFERENCE path: same 784->512->128->10 ReLU MLP, but
        // routed explicitly through the IEngine.MlpForward fused kernel. With
        // PR #1469's FeedForwardNeuralNetwork.Predict wiring, the plain "mlp"
        // model should now ALSO hit this kernel through Predict() — compare the
        // two rows to confirm the high-level path no longer leaves perf on the table.
        "mlp-fused" => new AiDotNetMlpFusedModel(seed),
        _ => throw new ArgumentException($"Unknown model '{model}'.")
    };
}

/// <summary>
/// Base for the benchmark models. Centralises the IBenchmarkModel contract so
/// each subclass only declares its architecture + input shape. Training runs
/// through AiDotNet's real <c>NeuralNetworkBase.Train</c> (forward + GradientTape
/// backward + optimizer step under the covers); inference runs through real
/// <c>NeuralNetworkBase.Predict</c>.
/// </summary>
internal abstract class AiDotNetBenchmarkModel : IBenchmarkModel
{
    protected readonly Random Random;
    protected readonly NeuralNetworkBase<float> Network;
    protected Tensor<float> Input = Tensor<float>.Empty();
    protected Tensor<float> Label = Tensor<float>.Empty();

    protected AiDotNetBenchmarkModel(int seed)
    {
        Random = new Random(seed);
        Network = BuildNetwork();
        ParameterCount = Network.GetParameters().Length;
    }

    public long ParameterCount { get; }

    protected abstract NeuralNetworkBase<float> BuildNetwork();
    protected abstract int[] InputShapePerSample { get; }   // shape WITHOUT batch dim
    protected abstract int OutputClasses { get; }

    public void LoadSyntheticBatch(int batchSize)
    {
        var perSample = InputShapePerSample;
        var sampleSize = perSample.Aggregate(1, (a, b) => a * b);
        var fullShape = new[] { batchSize }.Concat(perSample).ToArray();
        var inputs = new float[batchSize * sampleSize];
        for (var i = 0; i < inputs.Length; i++) inputs[i] = (float)Random.NextDouble();
        Input = new Tensor<float>(inputs, fullShape);

        var labels = new float[batchSize * OutputClasses];
        for (var b = 0; b < batchSize; b++)
        {
            var cls = Random.Next(OutputClasses);
            labels[b * OutputClasses + cls] = 1f;
        }
        Label = new Tensor<float>(labels, [batchSize, OutputClasses]);
    }

    public void Forward()
    {
        // Real inference: walks the layer stack, runs activations + ops.
        var _ = Network.Predict(Input);
    }

    public void Backward()
    {
        // PyTorch runs `loss = criterion(model(x), y); loss.backward()` then
        // optimizer.step(). AiDotNet's Train(input, expected) is the equivalent:
        // forward + GradientTape backward + optimizer step in one call (and, with
        // PR #1469's gate, the compiled fused step when the optimizer maps). For
        // the per-batch wall-time measurement Backward does the full train step
        // and Step is a no-op — gradientSeconds therefore captures backward+step.
        Network.Train(Input, Label);
    }

    public void Step()
    {
        // Real optimizer step happened inside Backward()'s Train call.
    }
}

internal sealed class AiDotNetMlpModel : AiDotNetBenchmarkModel
{
    public AiDotNetMlpModel(int seed) : base(seed) { }
    protected override int[] InputShapePerSample => new[] { 784 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // Linear(784,512)+ReLU+Linear(512,128)+ReLU+Linear(128,10).
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(512, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(128, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 784,
            outputSize: 10,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch);
    }
}

/// <summary>
/// Fused-primitive MLP: identical 784→512→128→10 ReLU shape as
/// <see cref="AiDotNetMlpModel"/>, but inference routes explicitly through
/// <c>IEngine.MlpForward</c> — the fused, thread-capped multi-layer kernel.
/// Forward-only (MlpForward throws under a GradientTape), so training is a no-op;
/// training perf for this shape is measured by the plain "mlp" model.
/// </summary>
internal sealed class AiDotNetMlpFusedModel : IBenchmarkModel
{
    private static readonly int[] LayerSizes = [784, 512, 128, 10];
    private readonly Tensor<float>[] _weights;
    private readonly Tensor<float>?[] _biases;
    private Tensor<float> _input = Tensor<float>.Empty();

    public AiDotNetMlpFusedModel(int seed)
    {
        var rng = new Random(seed);
        _weights = new Tensor<float>[LayerSizes.Length - 1];
        _biases = new Tensor<float>?[LayerSizes.Length - 1];
        for (var i = 0; i < _weights.Length; i++)
        {
            int inF = LayerSizes[i], outF = LayerSizes[i + 1];
            var scale = (float)Math.Sqrt(2.0 / inF);
            var w = new float[inF * outF];
            for (var k = 0; k < w.Length; k++) w[k] = (float)(rng.NextDouble() - 0.5) * 2f * scale;
            _weights[i] = new Tensor<float>(w, [inF, outF]);
            var b = new float[outF];
            for (var k = 0; k < b.Length; k++) b[k] = (float)(rng.NextDouble() - 0.5) * 0.01f;
            _biases[i] = new Tensor<float>(b, [outF]);
        }
        ParameterCount = _weights.Sum(w => (long)w.Length) + _biases.Sum(b => (long)(b?.Length ?? 0));
    }

    public long ParameterCount { get; }

    public void LoadSyntheticBatch(int batchSize)
    {
        var data = new float[batchSize * 784];
        var rng = new Random(1234);
        for (var i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
        _input = new Tensor<float>(data, [batchSize, 784]);
    }

    public void Forward()
    {
        // activation(x @ Wᵢ + bᵢ) for every layer in one fused call.
        var _ = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.MlpForward(
            _input, _weights, _biases,
            AiDotNet.Tensors.Engines.FusedActivationType.ReLU,
            AiDotNet.Tensors.Engines.FusedActivationType.None);
    }

    public void Backward() { }
    public void Step() { }
}

internal sealed class AiDotNetCnnModel : AiDotNetBenchmarkModel
{
    public AiDotNetCnnModel(int seed) : base(seed) { }
    protected override int[] InputShapePerSample => new[] { 1, 28, 28 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // Conv2d(1,16,3,pad=1)+ReLU+MaxPool(2)+Conv2d(16,32,3,pad=1)+ReLU+MaxPool(2)+Flatten+Linear(10).
        var layers = new List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 16, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new ConvolutionalLayer<float>(outputDepth: 32, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 28, inputWidth: 28, inputDepth: 1,
            outputSize: 10,
            layers: layers);
        return new ConvolutionalNeuralNetwork<float>(arch);
    }
}

internal sealed class AiDotNetLstmModel : AiDotNetBenchmarkModel
{
    public AiDotNetLstmModel(int seed) : base(seed) { }
    protected override int[] InputShapePerSample => new[] { 32, 32 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // LSTM(input=32, hidden=64) + last-timestep slice + Linear(64, 10).
        var layers = new List<ILayer<float>>
        {
            new LSTMLayer<float>(hiddenSize: 64),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 32,
            outputSize: 10,
            layers: layers);
        return new LSTMNeuralNetwork<float>(arch, outputActivation: (IActivationFunction<float>?)null);
    }
}

internal sealed class AiDotNetTransformerModel : AiDotNetBenchmarkModel
{
    public AiDotNetTransformerModel(int seed) : base(seed) { }
    protected override int[] InputShapePerSample => new[] { 32, 32 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // Linear(32,64) + 2× TransformerEncoderLayer(d_model=64,nhead=4,dim_ff=128)
        // + mean(seq) + Linear(64,10), via AiDotNet's production Transformer<float>.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 128,
            inputSize: 32,
            outputSize: 10,
            dropoutRate: 0.0,
            maxSequenceLength: 32,
            vocabularySize: 0,
            usePositionalEncoding: true,
            sequencePooling: SequencePoolingMode.MeanPool);
        return new Transformer<float>(arch);
    }
}

internal sealed class ResourceMonitor : IDisposable
{
    private readonly Process _process = Process.GetCurrentProcess();
    private readonly CancellationTokenSource _cts = new();
    private readonly List<double> _rssMb = [];
    private readonly Task _task;

    private ResourceMonitor()
    {
        _task = Task.Run(async () =>
        {
            while (!_cts.IsCancellationRequested)
            {
                _process.Refresh();
                _rssMb.Add(_process.WorkingSet64 / 1024d / 1024d);
                await Task.Delay(100, _cts.Token).ContinueWith(_ => { });
            }
        });
    }

    public static ResourceMonitor Start() => new();

    public ResourceReport Summary() => new(Math.Round(_rssMb.Count == 0 ? 0 : _rssMb.Max(), 3), NvidiaSmi.TryRead());

    public void Dispose()
    {
        _cts.Cancel();
        _task.Wait(TimeSpan.FromSeconds(2));
        _cts.Dispose();
    }
}

internal static class NvidiaSmi
{
    public static string? TryRead()
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                ArgumentList = { "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits" },
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            });
            if (process is null || !process.WaitForExit(1000) || process.ExitCode != 0) return null;
            return process.StandardOutput.ReadLine();
        }
        catch
        {
            return null;
        }
    }
}

internal static class AiDotNetProbe
{
    public static object Describe()
    {
        // Report the loaded AiDotNet assembly metadata so the report identifies
        // which build ran (working-tree source vs a package).
        var assembly = typeof(NeuralNetworkBase<float>).Assembly;
        var location = string.Empty;
        try { location = assembly.Location; } catch { /* single-file: no location */ }
        return new
        {
            loaded = true,
            name = assembly.GetName().Name,
            version = assembly.GetName().Version?.ToString(),
            location,
        };
    }
}

internal sealed record BenchmarkReport(string Framework, string DotNetRuntime, object AiDotNet, List<ModelReport> Results);
internal sealed record ModelReport(string Model, string Backend, long Parameters, TrainingReport Training, List<InferenceReport> Inference);
internal sealed record TrainingReport(double[] EpochSeconds, double TotalSeconds, double GradientSecondsAvg, double DataLoadingSecondsAvg, ResourceReport Resources);
internal sealed record ResourceReport(double ManagedRssMbPeak, string? NvidiaSmiSample);
internal sealed record InferenceReport(int BatchSize, double WarmupSecondsAvg, double SteadyStateLatencyMsAvg, double SteadyStateLatencyMsP95, double ThroughputSamplesPerSecond, double MemoryMbPeak);

internal static class JsonOptions
{
    public static readonly JsonSerializerOptions Default = new() { WriteIndented = true };
}
