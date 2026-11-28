using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.CrossFamilyCompiledSuite;

public sealed class BenchmarkCliOptions
{
    public string Device { get; set; } = "cpu";
    public string Mode { get; set; } = "auto";
    public string Precision { get; set; } = "fp32";
    public string Wave { get; set; } = "all";
    public string? Filter { get; set; }
    public bool Smoke { get; set; }
    public int Seed { get; set; } = 42;
    public int WarmupSteps { get; set; } = 5;
    public int MeasureSteps { get; set; } = 20;
    public string? OutputJsonPath { get; set; }
}

public sealed class TimingSummary
{
    [JsonPropertyName("samples_ms")]
    public double[] SamplesMs { get; set; } = Array.Empty<double>();

    [JsonPropertyName("median_ms")]
    public double MedianMs { get; set; }

    [JsonPropertyName("iqr_ms")]
    public double IqrMs { get; set; }

    [JsonPropertyName("p95_ms")]
    public double P95Ms { get; set; }

    [JsonPropertyName("mean_ms")]
    public double MeanMs { get; set; }

    [JsonPropertyName("sd_ms")]
    public double SdMs { get; set; }

    [JsonPropertyName("min_ms")]
    public double MinMs { get; set; }

    [JsonPropertyName("max_ms")]
    public double MaxMs { get; set; }
}

public sealed class ModelSpecTag
{
    [JsonPropertyName("architecture")]
    public string Architecture { get; set; } = string.Empty;

    [JsonPropertyName("init")]
    public string Init { get; set; } = "SeededRandom";

    [JsonPropertyName("layers")]
    public int Layers { get; set; }

    [JsonPropertyName("hidden_dim")]
    public int HiddenDim { get; set; }

    [JsonPropertyName("vocab_or_classes")]
    public int VocabOrClasses { get; set; }

    [JsonPropertyName("context_or_spatial")]
    public int ContextOrSpatial { get; set; }

    [JsonPropertyName("parameter_count")]
    public long ParameterCount { get; set; }
}

public sealed class WorkloadBenchmarkRecord
{
    [JsonPropertyName("workload_id")]
    public string WorkloadId { get; set; } = string.Empty;

    [JsonPropertyName("wave")]
    public int Wave { get; set; }

    [JsonPropertyName("framework")]
    public string Framework { get; set; } = "AiDotNet";

    [JsonPropertyName("execution_mode")]
    public string ExecutionMode { get; set; } = string.Empty;

    [JsonPropertyName("precision_tier")]
    public string PrecisionTier { get; set; } = string.Empty;

    [JsonPropertyName("device")]
    public string Device { get; set; } = string.Empty;

    [JsonPropertyName("seed")]
    public int Seed { get; set; }

    [JsonPropertyName("batch_size")]
    public int BatchSize { get; set; }

    [JsonPropertyName("train_eval_window_overlap")]
    public int TrainEvalWindowOverlap { get; set; }

    [JsonPropertyName("model_spec")]
    public ModelSpecTag ModelSpec { get; set; } = new();

    [JsonPropertyName("b1_available")]
    public bool B1Available { get; set; } = true;

    [JsonPropertyName("b1_inference_latency")]
    public TimingSummary B1InferenceLatency { get; set; } = new();

    [JsonPropertyName("batched_inference_latency")]
    public TimingSummary BatchedInferenceLatency { get; set; } = new();

    [JsonPropertyName("batched_train_step_latency")]
    public TimingSummary BatchedTrainStepLatency { get; set; } = new();

    [JsonPropertyName("batched_infer_items_per_sec")]
    public double BatchedInferItemsPerSec { get; set; }

    [JsonPropertyName("batched_train_items_per_sec")]
    public double BatchedTrainItemsPerSec { get; set; }

    [JsonPropertyName("initial_eval_loss")]
    public double InitialEvalLoss { get; set; }

    [JsonPropertyName("final_eval_loss")]
    public double FinalEvalLoss { get; set; }

    [JsonPropertyName("managed_alloc_bytes_per_step")]
    public double ManagedAllocBytesPerStep { get; set; }

    [JsonPropertyName("notes")]
    public string Notes { get; set; } = string.Empty;
}

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        var cli = ParseArgs(args);
        if (cli.Smoke)
        {
            cli.WarmupSteps = Math.Min(cli.WarmupSteps, 2);
            cli.MeasureSteps = Math.Min(cli.MeasureSteps, 3);
        }

        string wave = cli.Wave.ToLowerInvariant();
        if (wave is not ("all" or "1" or "2" or "wave1" or "wave2"))
        {
            throw new ArgumentException($"Unsupported --wave '{cli.Wave}'. Expected all, 1, 2, wave1, or wave2.");
        }
        cli.Wave = wave;

        ConfigureEngineDevice(cli.Device);
        var compilationOptions = BuildCompilationOptions(cli);
        compilationOptions.Validate();

        Console.WriteLine(
            $"[AiDotNet.CrossFamilyCompiledSuite] device={cli.Device} engine={AiDotNetEngine.Current.GetType().Name} " +
            $"mode={cli.Mode} precision={cli.Precision} wave={cli.Wave} smoke={cli.Smoke} seed={cli.Seed}");

        var records = new List<WorkloadBenchmarkRecord>();
        bool runWave1 = cli.Wave is "all" or "1" or "wave1";
        bool runWave2 = cli.Wave is "all" or "2" or "wave2";

        if (runWave1)
        {
            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_causal_transformer_lm_backprop", () =>
                RunCausalTransformerLmAsync(cli, compilationOptions, useLocalCredit: false));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_causal_transformer_lm_local_credit", () =>
                RunCausalTransformerLmAsync(cli, compilationOptions, useLocalCredit: true));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_gated_linear_ssm_lm", () =>
                RunGatedLinearSsmLmAsync(cli, compilationOptions));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_householder_orthogonal_projection", () =>
                RunHouseholderProjectionAsync(cli, compilationOptions));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_optimizer_base_sweep", () =>
                RunOptimizerSweepAsync(cli, compilationOptions, paired: true));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_optimizer_unpaired_dotnet_only_sweep", () =>
                RunOptimizerSweepAsync(cli, compilationOptions, paired: false));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_vision_resnet", () =>
                RunVisionResNetAsync(cli, compilationOptions));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_vision_vit", () =>
                RunVisionViTAsync(cli, compilationOptions));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave1_tabular_mlp", () =>
                RunTabularMlpAsync(cli, compilationOptions));
        }

        if (runWave2)
        {
            await RunIfMatchedAsync(records, cli, compilationOptions, "wave2_diffusion_dit_unet", () =>
                RunDiffusionDenoiserStepAsync(cli, compilationOptions));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave2_timeseries_patchtst_tcn", () =>
                RunTimeSeriesTcnTransformerAsync(cli, compilationOptions));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave2_audio_conformer", () =>
                RunAudioConformerAsync(cli, compilationOptions));

            await RunIfMatchedAsync(records, cli, compilationOptions, "wave2_gnn_gcn_gat", () =>
                RunGraphNeuralNetworkAsync(cli, compilationOptions));
        }

        foreach (var rec in records)
        {
            Console.WriteLine(
                string.Format(
                    CultureInfo.InvariantCulture,
                    "  {0,-42} | params={1,9} | B=1 med={2,8:F3} ms | BatchInfer med={3,8:F3} ms ({4,8:F1} it/s) | TrainStep med={5,8:F3} ms | loss {6:F4}->{7:F4}",
                    rec.WorkloadId,
                    rec.ModelSpec.ParameterCount,
                    rec.B1InferenceLatency.MedianMs,
                    rec.BatchedInferenceLatency.MedianMs,
                    rec.BatchedInferItemsPerSec,
                    rec.BatchedTrainStepLatency.MedianMs,
                    rec.InitialEvalLoss,
                    rec.FinalEvalLoss));
        }

        if (!string.IsNullOrWhiteSpace(cli.OutputJsonPath))
        {
            string fullPath = Path.GetFullPath(cli.OutputJsonPath);
            string? dir = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);
            var jsonOpts = new JsonSerializerOptions { WriteIndented = true };
            await File.WriteAllTextAsync(fullPath, JsonSerializer.Serialize(records, jsonOpts));
            Console.WriteLine($"[AiDotNet.CrossFamilyCompiledSuite] Wrote {records.Count} records -> {fullPath}");
        }

        return 0;
    }

    private static async Task RunIfMatchedAsync(
        List<WorkloadBenchmarkRecord> sink,
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions,
        string workloadId,
        Func<Task<WorkloadBenchmarkRecord>> runner)
    {
        if (!string.IsNullOrWhiteSpace(cli.Filter)
            && workloadId.IndexOf(cli.Filter, StringComparison.OrdinalIgnoreCase) < 0)
        {
            return;
        }

        _ = compilationOptions;
        var rec = await runner();
        sink.Add(rec);
    }

    private static void ConfigureEngineDevice(string device)
    {
        if (string.Equals(device, "cpu", StringComparison.OrdinalIgnoreCase))
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
            AiDotNetEngine.Current = new CpuEngine();
            return;
        }

        if (string.Equals(device, "cuda", StringComparison.OrdinalIgnoreCase)
            || string.Equals(device, "gpu", StringComparison.OrdinalIgnoreCase))
        {
            string? vis = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
            if (string.Equals(vis, "0", StringComparison.Ordinal))
            {
                throw new InvalidOperationException("Safety gate: GPU 0 is off-limits. Set CUDA_VISIBLE_DEVICES=1.");
            }

            var gpuEngine = new DirectGpuTensorEngine();
            if (!gpuEngine.IsGpuAvailable)
            {
                throw new InvalidOperationException("Requested --device cuda, but DirectGpuTensorEngine.IsGpuAvailable is false.");
            }
            string boundName = gpuEngine.GetBackend()?.DeviceName ?? "UnknownGpu";
            if (boundName.IndexOf("3080", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                throw new InvalidOperationException($"Safety gate: Bound to '{boundName}' (GPU 0 RTX 3080 is off-limits).");
            }
            Console.WriteLine($"[AiDotNet.CrossFamilyCompiledSuite] Bound GPU Backend: {boundName}");
            AiDotNetEngine.Current = gpuEngine;
            return;
        }

        throw new ArgumentException($"Unsupported --device '{device}'. Expected 'cpu' or 'cuda'.", nameof(device));
    }

    private static void SynchronizeActiveDevice()
    {
        if (AiDotNetEngine.Current.Unwrap() is DirectGpuTensorEngine gpuEngine)
        {
            gpuEngine.GetBackend()?.Synchronize();
        }
    }

    private static CompilationOptions BuildCompilationOptions(BenchmarkCliOptions cli)
    {
        CompilationOptions opts = cli.Mode.ToLowerInvariant() switch
        {
            "auto" or "autocompiled" => CompilationOptions.MaxAutotune,
            "direct-ptx" or "puredirectptx" or "ptx" => CompilationOptions.PureDirectPtx,
            "eager" => CompilationOptions.Eager,
            "uncached-eager" => new CompilationOptions(CompilationOptions.Eager) { EnableDerivedWeightCaching = false },
            _ => throw new ArgumentException($"Unsupported --mode '{cli.Mode}'. Expected auto, direct-ptx, eager, or uncached-eager."),
        };

        opts.PrecisionTier = cli.Precision.ToLowerInvariant() switch
        {
            "fp32" => ComputePrecisionTier.FP32,
            "tf32" => ComputePrecisionTier.TF32,
            "bf16" => ComputePrecisionTier.BF16,
            "fp16" => ComputePrecisionTier.FP16,
            _ => throw new ArgumentException($"Unsupported --precision '{cli.Precision}'. Expected fp32, tf32, bf16, or fp16."),
        };

        return opts;
    }

    private static TimingSummary SummarizeLatencies(double[] rawSamplesMs)
    {
        if (rawSamplesMs.Length == 0)
            return new TimingSummary();

        var sorted = (double[])rawSamplesMs.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        double median = (n & 1) == 1 ? sorted[n / 2] : 0.5 * (sorted[(n / 2) - 1] + sorted[n / 2]);
        double q25 = PercentileSorted(sorted, 0.25);
        double q75 = PercentileSorted(sorted, 0.75);
        double p95 = PercentileSorted(sorted, 0.95);
        double mean = rawSamplesMs.Average();
        double varSum = 0.0;
        for (int i = 0; i < n; i++)
        {
            double d = rawSamplesMs[i] - mean;
            varSum += d * d;
        }
        double sd = n > 1 ? Math.Sqrt(varSum / (n - 1)) : 0.0;

        return new TimingSummary
        {
            SamplesMs = rawSamplesMs,
            MedianMs = median,
            IqrMs = q75 - q25,
            P95Ms = p95,
            MeanMs = mean,
            SdMs = sd,
            MinMs = sorted[0],
            MaxMs = sorted[n - 1],
        };
    }

    private static double PercentileSorted(double[] sorted, double p)
    {
        if (sorted.Length == 1) return sorted[0];
        double pos = p * (sorted.Length - 1);
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        if (lo == hi) return sorted[lo];
        double frac = pos - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    private static TimingSummary MeasureRawAction(int warmup, int iterations, Action body, out double allocBytesPerStep)
    {
        for (int w = 0; w < warmup; w++)
        {
            body();
            SynchronizeActiveDevice();
        }

        double[] samples = new double[iterations];
        double tickToMs = 1000.0 / Stopwatch.Frequency;
        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        SynchronizeActiveDevice();

        for (int i = 0; i < iterations; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            body();
            SynchronizeActiveDevice();
            long t1 = Stopwatch.GetTimestamp();
            samples[i] = (t1 - t0) * tickToMs;
        }

        long allocAfter = GC.GetAllocatedBytesForCurrentThread();
        allocBytesPerStep = iterations > 0 ? (double)(allocAfter - allocBefore) / iterations : 0.0;
        return SummarizeLatencies(samples);
    }

    private static Tensor<float> CreateDisjointFeatures(int[] shape, int seed, int partitionOffset)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(unchecked(seed * 7919 + partitionOffset * 104729));
        int total = 1;
        for (int d = 0; d < shape.Length; d++) total *= shape[d];
        float[] data = new float[total];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
        }
        return new Tensor<float>(data, shape);
    }

    private static Tensor<float> CreateOneHotTargets(int[] leadingDims, int numClasses, int seed, int partitionOffset)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(unchecked(seed * 6151 + partitionOffset * 99991));
        int[] fullShape = new int[leadingDims.Length + 1];
        int rows = 1;
        for (int i = 0; i < leadingDims.Length; i++)
        {
            fullShape[i] = leadingDims[i];
            rows *= leadingDims[i];
        }
        fullShape[leadingDims.Length] = numClasses;

        float[] data = new float[rows * numClasses];
        for (int r = 0; r < rows; r++)
        {
            int cls = rng.Next(0, numClasses);
            data[r * numClasses + cls] = 1.0f;
        }
        return new Tensor<float>(data, fullShape);
    }

    private static int VerifyZeroRowOverlap(Tensor<float> trainX, Tensor<float> evalX)
    {
        int rowWidth = trainX.Length / Math.Max(1, trainX.Shape[0]);
        if (evalX.Length / Math.Max(1, evalX.Shape[0]) != rowWidth) return 0;

        var trainHashes = new HashSet<long>();
        for (int r = 0; r < trainX.Shape[0]; r++)
        {
            trainHashes.Add(HashRow(trainX, r * rowWidth, rowWidth));
        }

        int overlap = 0;
        for (int r = 0; r < evalX.Shape[0]; r++)
        {
            if (trainHashes.Contains(HashRow(evalX, r * rowWidth, rowWidth)))
                overlap++;
        }

        if (overlap != 0)
        {
            throw new InvalidOperationException($"Adversarial Gate 7 violation: detected {overlap} overlapping train/eval windows.");
        }
        return overlap;
    }

    private static long HashRow(Tensor<float> tensor, int start, int length)
    {
        unchecked
        {
            long h = 1469598103934665603L;
            int take = Math.Min(length, 64);
            for (int i = 0; i < take; i++)
            {
                h ^= BitConverter.SingleToInt32Bits(tensor[start + i]);
                h *= 1099511628211L;
            }
            return h;
        }
    }

    /// <summary>
    /// Builds the network with the architecture seeded from the CLI seed (equal seeds give equal initial
    /// weights) and with the benchmark's AdamW (lr 1e-3, weight decay 0.01, norm clip 1.0) installed as the
    /// network's own training optimizer, so <c>network.Train</c> in the timed step runs exactly this AdamW
    /// update - the same optimizer and clip the PyTorch arm times.
    /// </summary>
    private static (NeuralNetwork<float> Network, AdamWOptimizer<float, Tensor<float>, Tensor<float>> Optimizer) CreateSeededNetworkWithAdamW(
        NeuralNetworkArchitecture<float> arch,
        LossFunctionBase<float> lossFn,
        int batchSize,
        int seed)
    {
        arch.RandomSeed = seed;
        var opt = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-3,
                WeightDecay = 0.01,
                BatchSize = batchSize,
                MaxIterations = 1,
                EnableGradientClipping = true,
                GradientClippingMethod = GradientClippingMethod.ByNorm,
                MaxGradientNorm = 1.0,
            });
        var net = new NeuralNetwork<float>(arch, optimizer: opt, lossFunction: lossFn);
        opt.SetModel(net);
        return (net, opt);
    }

    private static double EvaluateMeanLoss(NeuralNetwork<float> net, LossFunctionBase<float> lossFn, Tensor<float> evalX, Tensor<float> evalY)
    {
        using var noGrad = new NoGradScope<float>();
        var pred = net.Predict(evalX);
        var lossTensor = lossFn.ComputeTapeLoss(pred, evalY);
        double sum = 0.0;
        for (int i = 0; i < lossTensor.Length; i++) sum += lossTensor[i];
        return lossTensor.Length > 0 ? sum / lossTensor.Length : 0.0;
    }

    private static async Task<WorkloadBenchmarkRecord> ExecuteStandardNeuralWorkloadAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions,
        string workloadId,
        int wave,
        ModelSpecTag spec,
        NeuralNetwork<float> network,
        AdamWOptimizer<float, Tensor<float>, Tensor<float>> opt,
        LossFunctionBase<float> lossFn,
        Tensor<float> b1Input,
        Tensor<float> batchTrainX,
        Tensor<float> batchTrainY,
        Tensor<float> batchEvalX,
        Tensor<float> batchEvalY,
        ICreditRule<float>? creditRule = null,
        string notes = "",
        bool measureB1 = true)
    {
        int overlap = VerifyZeroRowOverlap(batchTrainX, batchEvalX);
        int batchSize = batchTrainX.Shape[0];

        // Materialize lazy parameters before counting and benchmarking.
        _ = network.Predict(b1Input);
        spec.ParameterCount = network.ParameterCount;

        if (creditRule is not null)
        {
            network.SetCreditRule(creditRule, cli.Seed);
        }

        double initialEvalLoss = EvaluateMeanLoss(network, lossFn, batchEvalX, batchEvalY);

        // Build inference facade handle through AiModelBuilder.ConfigureCompilation(...)
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(network)
            .ConfigureOptimizer(opt)
            .ConfigureCompilation(compilationOptions)
            .ConfigureLossFunction(lossFn);
        if (creditRule is not null)
        {
            builder = builder.ConfigureCreditRule(creditRule, cli.Seed);
        }
        if (!compilationOptions.AllowExternalVendorLibraries || compilationOptions.Mode == ExecutionMode.AutoCompiled)
        {
            builder = builder.AllowNondeterminism();
        }

        var inferFacade = builder.BuildForInference();

        // 1. Measure B = 1 inference latency via AiModelResult.Predict (zero target tensors passed).
        //    Skipped (reported unavailable) when the workload has no distinct single-item input.
        var b1Summary = measureB1
            ? MeasureRawAction(
                cli.WarmupSteps,
                cli.MeasureSteps,
                () => { _ = inferFacade.Predict(b1Input); },
                out _)
            : new TimingSummary();

        // 2. Measure batched inference throughput via AiModelResult.Predict.
        var batchInferSummary = MeasureRawAction(
            cli.WarmupSteps,
            cli.MeasureSteps,
            () => { _ = inferFacade.Predict(batchEvalX); },
            out _);

        // 3. Measure batched training step (Forward + Loss + Backward/LocalCredit + GradClip + AdamW.Step).
        //    network.Train resolves the network's own training optimizer, which CreateSeededNetworkWithAdamW
        //    installed as `opt`.
        network.SetTrainingMode(true);
        var trainStepSummary = MeasureRawAction(
            cli.WarmupSteps,
            cli.MeasureSteps,
            () => { network.Train(batchTrainX, batchTrainY); },
            out double allocBytesPerStep);
        network.SetTrainingMode(false);

        double finalEvalLoss = EvaluateMeanLoss(network, lossFn, batchEvalX, batchEvalY);
        if (double.IsNaN(initialEvalLoss) || double.IsInfinity(initialEvalLoss)
            || double.IsNaN(finalEvalLoss) || double.IsInfinity(finalEvalLoss))
        {
            throw new InvalidOperationException(
                $"Adversarial Gate 10 violation in '{workloadId}': non-finite eval loss ({initialEvalLoss} -> {finalEvalLoss}).");
        }
        await Task.CompletedTask;

        return new WorkloadBenchmarkRecord
        {
            WorkloadId = workloadId,
            Wave = wave,
            ExecutionMode = cli.Mode,
            PrecisionTier = cli.Precision.ToUpperInvariant(),
            Device = cli.Device,
            Seed = cli.Seed,
            BatchSize = batchSize,
            TrainEvalWindowOverlap = overlap,
            ModelSpec = spec,
            B1Available = measureB1,
            B1InferenceLatency = b1Summary,
            BatchedInferenceLatency = batchInferSummary,
            BatchedTrainStepLatency = trainStepSummary,
            BatchedInferItemsPerSec = batchInferSummary.MedianMs > 0 ? (batchSize * 1000.0 / batchInferSummary.MedianMs) : 0.0,
            BatchedTrainItemsPerSec = trainStepSummary.MedianMs > 0 ? (batchSize * 1000.0 / trainStepSummary.MedianMs) : 0.0,
            InitialEvalLoss = initialEvalLoss,
            FinalEvalLoss = finalEvalLoss,
            ManagedAllocBytesPerStep = allocBytesPerStep,
            Notes = notes,
        };
    }

    private static Task<WorkloadBenchmarkRecord> RunCausalTransformerLmAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions,
        bool useLocalCredit)
    {
        int batch = cli.Smoke ? 2 : 8;
        int seq = cli.Smoke ? 16 : 64;
        int dModel = cli.Smoke ? 64 : 256;
        int heads = cli.Smoke ? 4 : 8;
        int nBlocks = cli.Smoke ? 2 : 4;
        int vocab = cli.Smoke ? 128 : 1024;

        var layers = new List<ILayer<float>>();
        for (int l = 0; l < nBlocks; l++)
        {
            layers.Add(new RMSNormalizationLayer<float>(dModel));
            var mha = new MultiHeadAttentionLayer<float>(heads, dModel / heads);
            mha.UseCausalMask = true;
            layers.Add(mha);
            layers.Add(new RMSNormalizationLayer<float>(dModel));
            layers.Add(new DenseLayer<float>(dModel, activationFunction: new SiLUActivation<float>()));
            layers.Add(new DenseLayer<float>(dModel, activationFunction: new IdentityActivation<float>()));
        }
        layers.Add(new RMSNormalizationLayer<float>(dModel));
        layers.Add(new DenseLayer<float>(vocab, vectorActivation: new SoftmaxActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Medium,
            inputSize: dModel,
            outputSize: vocab,
            layers: layers);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, seq, dModel }, cli.Seed, 10);
        var trainX = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 11);
        var trainY = CreateOneHotTargets(new[] { batch, seq }, vocab, cli.Seed, 12);
        var evalX = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 13);
        var evalY = CreateOneHotTargets(new[] { batch, seq }, vocab, cli.Seed, 14);

        ICreditRule<float>? rule = useLocalCredit
            ? new TiedReadoutLocalErrorCreditRule<float>(readoutLayerCount: 2)
            : null;

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            useLocalCredit ? "wave1_causal_transformer_lm_local_credit" : "wave1_causal_transformer_lm_backprop",
            wave: 1,
            new ModelSpecTag
            {
                Architecture = useLocalCredit ? "CausalTransformerLM-TiedReadoutLocalCredit" : "CausalTransformerLM-Backprop",
                Layers = nBlocks,
                HiddenDim = dModel,
                VocabOrClasses = vocab,
                ContextOrSpatial = seq,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY,
            creditRule: rule);
    }

    private static Task<WorkloadBenchmarkRecord> RunGatedLinearSsmLmAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 2 : 8;
        int seq = cli.Smoke ? 16 : 64;
        int dModel = cli.Smoke ? 64 : 256;
        int heads = cli.Smoke ? 4 : 8;
        int nBlocks = cli.Smoke ? 2 : 4;
        int vocab = cli.Smoke ? 128 : 1024;

        var layers = new List<ILayer<float>>();
        for (int l = 0; l < nBlocks; l++)
        {
            layers.Add(new GatedLinearAttentionLayer<float>(seq, dModel, heads));
        }
        layers.Add(new DenseLayer<float>(dModel * 2, activationFunction: new SiLUActivation<float>()));
        layers.Add(new DenseLayer<float>(vocab, vectorActivation: new SoftmaxActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Medium,
            inputDepth: 1,
            inputHeight: seq,
            inputWidth: dModel,
            outputSize: vocab,
            layers: layers);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, seq, dModel }, cli.Seed, 20);
        var trainX = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 21);
        var trainY = CreateOneHotTargets(new[] { batch, seq }, vocab, cli.Seed, 22);
        var evalX = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 23);
        var evalY = CreateOneHotTargets(new[] { batch, seq }, vocab, cli.Seed, 24);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave1_gated_linear_ssm_lm",
            wave: 1,
            new ModelSpecTag
            {
                Architecture = "GatedLinearAttentionSSM-LM",
                Layers = nBlocks,
                HiddenDim = dModel,
                VocabOrClasses = vocab,
                ContextOrSpatial = seq,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY);
    }

    private static Task<WorkloadBenchmarkRecord> RunHouseholderProjectionAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 4 : 16;
        int seq = cli.Smoke ? 16 : 64;
        int dModel = cli.Smoke ? 64 : 256;
        int kRefl = 4;
        int nBlocks = cli.Smoke ? 4 : 8;

        var layers = new List<ILayer<float>>();
        for (int l = 0; l < nBlocks; l++)
        {
            layers.Add(new HouseholderSubspaceLayer<float>(dModel, kRefl, transpose: (l & 1) == 1));
            layers.Add(new DenseLayer<float>(dModel, activationFunction: new GELUActivation<float>()));
        }
        layers.Add(new DenseLayer<float>(dModel, activationFunction: new IdentityActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Medium,
            inputSize: dModel,
            outputSize: dModel,
            layers: layers);
        var lossFn = new MeanSquaredErrorLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, seq, dModel }, cli.Seed, 30);
        var trainX = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 31);
        var trainY = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 32);
        var evalX = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 33);
        var evalY = CreateDisjointFeatures(new[] { batch, seq, dModel }, cli.Seed, 34);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave1_householder_orthogonal_projection",
            wave: 1,
            new ModelSpecTag
            {
                Architecture = $"HouseholderOrthogonalStack-k{kRefl}",
                Layers = nBlocks,
                HiddenDim = dModel,
                VocabOrClasses = dModel,
                ContextOrSpatial = seq,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY,
            notes: $"DerivedWeightCaching={compilationOptions.EnableDerivedWeightCaching}");
    }

    /// <summary>
    /// Optimizer step-latency sweep. <paramref name="paired"/> = true times only the optimizers the PyTorch
    /// runner also times (same names, same timed work: forward + loss + backward + optimizer step, no gradient
    /// clipping), so the paired median compares like with like. <paramref name="paired"/> = false times the
    /// AiDotNet-only optimizers under a separate, unpaired workload id that has no PyTorch counterpart.
    /// </summary>
    private static Task<WorkloadBenchmarkRecord> RunOptimizerSweepAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions,
        bool paired)
    {
        _ = compilationOptions;
        string workloadId = paired ? "wave1_optimizer_base_sweep" : "wave1_optimizer_unpaired_dotnet_only_sweep";
        int batch = cli.Smoke ? 8 : 32;
        int inDim = cli.Smoke ? 32 : 256;
        int hidDim = cli.Smoke ? 64 : 512;
        int classes = cli.Smoke ? 8 : 32;

        var trainX = CreateDisjointFeatures(new[] { batch, inDim }, cli.Seed, 41);
        var trainY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 42);
        var evalX = CreateDisjointFeatures(new[] { batch, inDim }, cli.Seed, 43);
        var evalY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 44);
        int overlap = VerifyZeroRowOverlap(trainX, evalX);
        var b1 = CreateDisjointFeatures(new[] { 1, inDim }, cli.Seed, 45);

        NeuralNetwork<float> BuildMlp()
        {
            var arch = new NeuralNetworkArchitecture<float>(
                InputType.OneDimensional,
                NeuralNetworkTaskType.MultiClassClassification,
                NetworkComplexity.Simple,
                inputSize: inDim,
                outputSize: classes,
                layers: new List<ILayer<float>>
                {
                    new DenseLayer<float>(hidDim, activationFunction: new GELUActivation<float>()),
                    new DenseLayer<float>(hidDim, activationFunction: new GELUActivation<float>()),
                    new DenseLayer<float>(classes, vectorActivation: new SoftmaxActivation<float>()),
                });
            // Same seed -> same initial weights for every optimizer case and every process.
            arch.RandomSeed = cli.Seed;
            var m = new NeuralNetwork<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
            _ = m.Predict(b1);
            return m;
        }

        // Paired set: exactly the optimizer names the PyTorch runner's sweep times.
        var pairedFactories = new (string Name, Func<NeuralNetwork<float>, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>> Build)[]
        {
            ("AdamW", n => new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("Adam", n => new AdamOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("SGD", n => new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("Momentum", n => new MomentumOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("NAG", n => new NesterovAcceleratedGradientOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("RMSProp", n => new RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("Adagrad", n => new AdagradOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("AdaDelta", n => new AdaDeltaOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("AdaMax", n => new AdaMaxOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("AMSGrad", n => new AMSGradOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("Nadam", n => new NadamOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("RAdam", n => new RAdamOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("Rprop", n => new RpropOptimizer<float, Tensor<float>, Tensor<float>>(n)),
        };

        // AiDotNet-only optimizers: reported in their own unpaired record, never in the paired median.
        var unpairedFactories = new (string Name, Func<NeuralNetwork<float>, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>> Build)[]
        {
            ("Adam8Bit", n => new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("Lion", n => new LionOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("LAMB", n => new LAMBOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("LARS", n => new LARSOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("Adafactor", n => new AdafactorOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("ScheduleFreeAdamW", n => new ScheduleFreeAdamWOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("DAdaptAdam", n => new DAdaptAdamOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("ScaledAdam", n => new ScaledAdamOptimizer<float, Tensor<float>, Tensor<float>>(n)),
            ("FTRL", n => new FTRLOptimizer<float, Tensor<float>, Tensor<float>>(n)),
        };

        var optimizerFactories = paired ? pairedFactories : unpairedFactories;
        var perOptimizerMedianMs = new List<double>(optimizerFactories.Length);
        var perOptimizerInitialLoss = new List<double>(optimizerFactories.Length);
        var perOptimizerFinalLoss = new List<double>(optimizerFactories.Length);
        var perOptimizerNotes = new List<string>(optimizerFactories.Length);
        long paramCount = 0;
        var lossFn = new CategoricalCrossEntropyLoss<float>();

        foreach (var (optName, factory) in optimizerFactories)
        {
            var net = BuildMlp();
            paramCount = net.ParameterCount;
            var opt = factory(net);

            var trainable = new List<Tensor<float>>();
            foreach (var layer in net.Layers)
            {
                if (layer is ITrainableLayer<float> tl)
                {
                    foreach (var p in tl.GetTrainableParameters())
                    {
                        if (p.Length > 0) trainable.Add(p);
                    }
                }
            }

            // Loss of THIS optimizer's model, before and after its own warmup + measured steps.
            double optInitialLoss = EvaluateMeanLoss(net, lossFn, evalX, evalY);

            var stepSummary = MeasureRawAction(
                warmup: 1,
                iterations: cli.Smoke ? 2 : 5,
                () =>
                {
                    using var tape = new GradientTape<float>();
                    var pred = net.ForwardForTraining(trainX);
                    var loss = lossFn.ComputeTapeLoss(pred, trainY);
                    var grads = tape.ComputeGradients(loss, trainable);
                    opt.Step(new TapeStepContext<float>(trainable, grads, loss[0]));
                },
                out _);

            var checkParams = net.GetParameters();
            for (int i = 0; i < checkParams.Length; i++)
            {
                if (float.IsNaN(checkParams[i]) || float.IsInfinity(checkParams[i]))
                    throw new InvalidOperationException($"Optimizer '{optName}' produced non-finite weights.");
            }

            double optFinalLoss = EvaluateMeanLoss(net, lossFn, evalX, evalY);
            if (double.IsNaN(optInitialLoss) || double.IsInfinity(optInitialLoss)
                || double.IsNaN(optFinalLoss) || double.IsInfinity(optFinalLoss))
            {
                throw new InvalidOperationException(
                    $"Adversarial Gate 10 violation in '{workloadId}' optimizer '{optName}': non-finite eval loss ({optInitialLoss} -> {optFinalLoss}).");
            }

            perOptimizerMedianMs.Add(stepSummary.MedianMs);
            perOptimizerInitialLoss.Add(optInitialLoss);
            perOptimizerFinalLoss.Add(optFinalLoss);
            perOptimizerNotes.Add(string.Format(
                CultureInfo.InvariantCulture,
                "{0}:step_median_ms={1:F4},loss={2:F4}->{3:F4}",
                optName,
                stepSummary.MedianMs,
                optInitialLoss,
                optFinalLoss));
        }

        // Inference latency is optimizer-independent: time a freshly built (seeded) copy of the same MLP.
        var probeNet = BuildMlp();
        var b1Summary = MeasureRawAction(1, cli.Smoke ? 2 : 5, () => { _ = probeNet.Predict(b1); }, out _);
        var batchInferSummary = MeasureRawAction(1, cli.Smoke ? 2 : 5, () => { _ = probeNet.Predict(evalX); }, out _);
        var sweepSummary = SummarizeLatencies(perOptimizerMedianMs.ToArray());

        return Task.FromResult(new WorkloadBenchmarkRecord
        {
            WorkloadId = workloadId,
            Wave = 1,
            ExecutionMode = cli.Mode,
            PrecisionTier = cli.Precision.ToUpperInvariant(),
            Device = cli.Device,
            Seed = cli.Seed,
            BatchSize = batch,
            TrainEvalWindowOverlap = overlap,
            ModelSpec = new ModelSpecTag
            {
                Architecture = $"GradientBasedOptimizerBase-{(paired ? "PairedSweep" : "UnpairedSweep")}-{optimizerFactories.Length}Opts",
                Layers = 3,
                HiddenDim = hidDim,
                VocabOrClasses = classes,
                ContextOrSpatial = inDim,
                ParameterCount = paramCount,
            },
            B1InferenceLatency = b1Summary,
            BatchedInferenceLatency = batchInferSummary,
            BatchedTrainStepLatency = sweepSummary,
            BatchedInferItemsPerSec = batchInferSummary.MedianMs > 0 ? (batch * 1000.0 / batchInferSummary.MedianMs) : 0.0,
            BatchedTrainItemsPerSec = sweepSummary.MedianMs > 0 ? (batch * 1000.0 / sweepSummary.MedianMs) : 0.0,
            // Median across optimizer cases of each case's own pre-/post-run eval loss.
            InitialEvalLoss = SummarizeLatencies(perOptimizerInitialLoss.ToArray()).MedianMs,
            FinalEvalLoss = SummarizeLatencies(perOptimizerFinalLoss.ToArray()).MedianMs,
            Notes = $"{(paired ? "Paired" : "Unpaired (AiDotNet-only)")} sweep of {optimizerFactories.Length} optimizers; timed op = forward+loss+backward+step (no clip). "
                + string.Join("; ", perOptimizerNotes),
        });
    }

    private static Task<WorkloadBenchmarkRecord> RunVisionResNetAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 2 : 16;
        int channels = 3;
        int spatial = cli.Smoke ? 16 : 32;
        int width = cli.Smoke ? 16 : 64;
        int classes = cli.Smoke ? 10 : 100;

        var layers = new List<ILayer<float>>
        {
            new BasicBlock<float>(outChannels: width, stride: 2),
            new BasicBlock<float>(outChannels: width * 2, stride: 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(classes, vectorActivation: new SoftmaxActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Medium,
            inputDepth: channels,
            inputHeight: spatial,
            inputWidth: spatial,
            outputSize: classes,
            layers: layers);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, channels, spatial, spatial }, cli.Seed, 50);
        var trainX = CreateDisjointFeatures(new[] { batch, channels, spatial, spatial }, cli.Seed, 51);
        var trainY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 52);
        var evalX = CreateDisjointFeatures(new[] { batch, channels, spatial, spatial }, cli.Seed, 53);
        var evalY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 54);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave1_vision_resnet",
            wave: 1,
            new ModelSpecTag
            {
                Architecture = "VisionConvBatchNormStageNet",
                Layers = layers.Count,
                HiddenDim = width * 2,
                VocabOrClasses = classes,
                ContextOrSpatial = spatial,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY);
    }

    private static Task<WorkloadBenchmarkRecord> RunVisionViTAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 2 : 16;
        int numPatches = cli.Smoke ? 16 : 64;
        int dModel = cli.Smoke ? 64 : 192;
        int heads = cli.Smoke ? 4 : 6;
        int nBlocks = cli.Smoke ? 2 : 4;
        int classes = cli.Smoke ? 10 : 100;

        var layers = new List<ILayer<float>>();
        for (int l = 0; l < nBlocks; l++)
        {
            // Pre-LN encoder block with both residuals (x + Attn(LN(x)); y + FFN(LN(y))), matching the
            // PyTorch EncoderBlock this workload is paired with.
            layers.Add(new TransformerEncoderBlock<float>(
                dModel, heads, dModel * 4, dropoutRate: 0.0, ffnActivation: new GELUActivation<float>()));
        }
        layers.Add(new FlattenLayer<float>());
        layers.Add(new DenseLayer<float>(classes, vectorActivation: new SoftmaxActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Medium,
            inputSize: dModel,
            outputSize: classes,
            layers: layers);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, numPatches, dModel }, cli.Seed, 60);
        var trainX = CreateDisjointFeatures(new[] { batch, numPatches, dModel }, cli.Seed, 61);
        var trainY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 62);
        var evalX = CreateDisjointFeatures(new[] { batch, numPatches, dModel }, cli.Seed, 63);
        var evalY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 64);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave1_vision_vit",
            wave: 1,
            new ModelSpecTag
            {
                Architecture = "VisionTransformerEncoder",
                Layers = nBlocks,
                HiddenDim = dModel,
                VocabOrClasses = classes,
                ContextOrSpatial = numPatches,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY);
    }

    private static Task<WorkloadBenchmarkRecord> RunTabularMlpAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 8 : 64;
        int inFeatures = cli.Smoke ? 64 : 256;
        int hidden = cli.Smoke ? 128 : 512;
        int nHidden = 4;
        int classes = cli.Smoke ? 8 : 32;

        var layers = new List<ILayer<float>>();
        for (int i = 0; i < nHidden; i++)
        {
            layers.Add(new DenseLayer<float>(hidden, activationFunction: new GELUActivation<float>()));
        }
        layers.Add(new DenseLayer<float>(classes, vectorActivation: new SoftmaxActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: inFeatures,
            outputSize: classes,
            layers: layers);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, inFeatures }, cli.Seed, 70);
        var trainX = CreateDisjointFeatures(new[] { batch, inFeatures }, cli.Seed, 71);
        var trainY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 72);
        var evalX = CreateDisjointFeatures(new[] { batch, inFeatures }, cli.Seed, 73);
        var evalY = CreateOneHotTargets(new[] { batch }, classes, cli.Seed, 74);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave1_tabular_mlp",
            wave: 1,
            new ModelSpecTag
            {
                Architecture = "TabularMLP-GELU",
                Layers = nHidden + 1,
                HiddenDim = hidden,
                VocabOrClasses = classes,
                ContextOrSpatial = inFeatures,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY);
    }

    private static Task<WorkloadBenchmarkRecord> RunDiffusionDenoiserStepAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 2 : 8;
        int latentTokens = cli.Smoke ? 16 : 64;
        int dModel = cli.Smoke ? 64 : 256;
        int heads = cli.Smoke ? 4 : 8;
        int nBlocks = cli.Smoke ? 2 : 4;

        var layers = new List<ILayer<float>>();
        for (int l = 0; l < nBlocks; l++)
        {
            // Pre-LN encoder block with both residuals (x + Attn(LN(x)); y + FFN(LN(y))), matching the
            // PyTorch EncoderBlock this workload is paired with.
            layers.Add(new TransformerEncoderBlock<float>(
                dModel, heads, dModel * 4, dropoutRate: 0.0, ffnActivation: new SiLUActivation<float>()));
        }
        layers.Add(new LayerNormalizationLayer<float>(dModel));
        layers.Add(new DenseLayer<float>(dModel, activationFunction: new IdentityActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Medium,
            inputSize: dModel,
            outputSize: dModel,
            layers: layers);
        var lossFn = new MeanSquaredErrorLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, latentTokens, dModel }, cli.Seed, 80);
        var trainX = CreateDisjointFeatures(new[] { batch, latentTokens, dModel }, cli.Seed, 81);
        var trainNoiseTarget = CreateDisjointFeatures(new[] { batch, latentTokens, dModel }, cli.Seed, 82);
        var evalX = CreateDisjointFeatures(new[] { batch, latentTokens, dModel }, cli.Seed, 83);
        var evalNoiseTarget = CreateDisjointFeatures(new[] { batch, latentTokens, dModel }, cli.Seed, 84);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave2_diffusion_dit_unet",
            wave: 2,
            new ModelSpecTag
            {
                Architecture = "DiffusionDiTDenoiserBlocks",
                Layers = nBlocks,
                HiddenDim = dModel,
                VocabOrClasses = dModel,
                ContextOrSpatial = latentTokens,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainNoiseTarget,
            evalX,
            evalNoiseTarget);
    }

    private static Task<WorkloadBenchmarkRecord> RunTimeSeriesTcnTransformerAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 2 : 16;
        int channels = cli.Smoke ? 8 : 32;
        int lookback = cli.Smoke ? 24 : 96;
        int horizon = cli.Smoke ? 12 : 24;

        var layers = new List<ILayer<float>>
        {
            new WeightNormConv1DLayer<float>(inChannels: channels, outChannels: channels * 2, kernelSize: 3, dilation: 1, padding: 1),
            new WeightNormConv1DLayer<float>(inChannels: channels * 2, outChannels: channels * 2, kernelSize: 3, dilation: 2, padding: 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(channels * 4, activationFunction: new GELUActivation<float>()),
            new DenseLayer<float>(horizon, activationFunction: new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Medium,
            inputSize: channels,
            outputSize: horizon,
            layers: layers);
        var lossFn = new MeanSquaredErrorLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, channels, lookback }, cli.Seed, 90);
        var trainX = CreateDisjointFeatures(new[] { batch, channels, lookback }, cli.Seed, 91);
        var trainY = CreateDisjointFeatures(new[] { batch, horizon }, cli.Seed, 92);
        var evalX = CreateDisjointFeatures(new[] { batch, channels, lookback }, cli.Seed, 93);
        var evalY = CreateDisjointFeatures(new[] { batch, horizon }, cli.Seed, 94);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave2_timeseries_patchtst_tcn",
            wave: 2,
            new ModelSpecTag
            {
                Architecture = "TimeSeriesDilatedWeightNormTCN",
                Layers = layers.Count,
                HiddenDim = channels * 2,
                VocabOrClasses = horizon,
                ContextOrSpatial = lookback,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY);
    }

    private static Task<WorkloadBenchmarkRecord> RunAudioConformerAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int batch = cli.Smoke ? 2 : 8;
        int frames = cli.Smoke ? 20 : 80;
        int dModel = cli.Smoke ? 64 : 192;
        int heads = cli.Smoke ? 4 : 6;
        int nBlocks = cli.Smoke ? 2 : 4;
        int vocab = cli.Smoke ? 32 : 128;

        var layers = new List<ILayer<float>>();
        for (int l = 0; l < nBlocks; l++)
        {
            layers.Add(new LayerNormalizationLayer<float>(dModel));
            layers.Add(new DenseLayer<float>(dModel * 2, activationFunction: new SiLUActivation<float>()));
            layers.Add(new DenseLayer<float>(dModel, activationFunction: new IdentityActivation<float>()));
            layers.Add(new LayerNormalizationLayer<float>(dModel));
            layers.Add(new MultiHeadAttentionLayer<float>(heads, dModel / heads));
            layers.Add(new LayerNormalizationLayer<float>(dModel));
            layers.Add(new DenseLayer<float>(dModel * 2, activationFunction: new SiLUActivation<float>()));
            layers.Add(new DenseLayer<float>(dModel, activationFunction: new IdentityActivation<float>()));
        }
        layers.Add(new LayerNormalizationLayer<float>(dModel));
        layers.Add(new DenseLayer<float>(vocab, vectorActivation: new SoftmaxActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Medium,
            inputSize: dModel,
            outputSize: vocab,
            layers: layers);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, batch, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { 1, frames, dModel }, cli.Seed, 100);
        var trainX = CreateDisjointFeatures(new[] { batch, frames, dModel }, cli.Seed, 101);
        var trainY = CreateOneHotTargets(new[] { batch, frames }, vocab, cli.Seed, 102);
        var evalX = CreateDisjointFeatures(new[] { batch, frames, dModel }, cli.Seed, 103);
        var evalY = CreateOneHotTargets(new[] { batch, frames }, vocab, cli.Seed, 104);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave2_audio_conformer",
            wave: 2,
            new ModelSpecTag
            {
                Architecture = "AudioConformerMacaronEncoder",
                Layers = nBlocks,
                HiddenDim = dModel,
                VocabOrClasses = vocab,
                ContextOrSpatial = frames,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY);
    }

    private static Task<WorkloadBenchmarkRecord> RunGraphNeuralNetworkAsync(
        BenchmarkCliOptions cli,
        CompilationOptions compilationOptions)
    {
        int totalNodes = (cli.Smoke ? 2 : 8) * (cli.Smoke ? 16 : 64);
        int nodeFeatures = cli.Smoke ? 32 : 128;
        int hidden = cli.Smoke ? 64 : 128;
        int classes = cli.Smoke ? 4 : 16;

        // Normalized symmetric ring-plus-chord graph adjacency [totalNodes, totalNodes].
        var adj = new Tensor<float>(new[] { totalNodes, totalNodes });
        for (int i = 0; i < totalNodes; i++)
        {
            int prev = (i - 1 + totalNodes) % totalNodes;
            int next = (i + 1) % totalNodes;
            int chord = (i + 4) % totalNodes;
            adj[i, i] = 0.25f;
            adj[i, prev] = 0.25f;
            adj[i, next] = 0.25f;
            adj[i, chord] = 0.25f;
        }

        var gcn1 = new GraphConvolutionalLayer<float>(nodeFeatures, hidden, new ReLUActivation<float>());
        gcn1.SetAdjacencyMatrix(adj);
        var gcn2 = new GraphConvolutionalLayer<float>(hidden, hidden, new ReLUActivation<float>());
        gcn2.SetAdjacencyMatrix(adj);

        var layers = new List<ILayer<float>>
        {
            gcn1,
            gcn2,
            new DenseLayer<float>(classes, vectorActivation: new SoftmaxActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Medium,
            inputSize: nodeFeatures,
            outputSize: classes,
            layers: layers);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var (net, opt) = CreateSeededNetworkWithAdamW(arch, lossFn, totalNodes, cli.Seed);

        var b1 = CreateDisjointFeatures(new[] { totalNodes, nodeFeatures }, cli.Seed, 110);
        var trainX = CreateDisjointFeatures(new[] { totalNodes, nodeFeatures }, cli.Seed, 111);
        var trainY = CreateOneHotTargets(new[] { totalNodes }, classes, cli.Seed, 112);
        var evalX = CreateDisjointFeatures(new[] { totalNodes, nodeFeatures }, cli.Seed, 113);
        var evalY = CreateOneHotTargets(new[] { totalNodes }, classes, cli.Seed, 114);

        return ExecuteStandardNeuralWorkloadAsync(
            cli,
            compilationOptions,
            "wave2_gnn_gcn_gat",
            wave: 2,
            new ModelSpecTag
            {
                Architecture = "GraphConvolutionalNetwork-2Layer",
                Layers = 3,
                HiddenDim = hidden,
                VocabOrClasses = classes,
                ContextOrSpatial = totalNodes,
            },
            net,
            opt,
            lossFn,
            b1,
            trainX,
            trainY,
            evalX,
            evalY,
            notes: $"FullGraphNodes={totalNodes}; B=1 unavailable (transductive full-graph GCN: every inference covers all {totalNodes} nodes)",
            measureB1: false);
    }

    private static BenchmarkCliOptions ParseArgs(string[] args)
    {
        var opts = new BenchmarkCliOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            switch (arg)
            {
                case "--smoke":
                case "--cpu-dry-run":
                    opts.Smoke = true;
                    break;
                case "--device":
                    if (i + 1 < args.Length) opts.Device = args[++i];
                    break;
                case "--mode":
                    if (i + 1 < args.Length) opts.Mode = args[++i];
                    break;
                case "--precision":
                    if (i + 1 < args.Length) opts.Precision = args[++i];
                    break;
                case "--wave":
                    if (i + 1 < args.Length) opts.Wave = args[++i];
                    break;
                case "--filter":
                    if (i + 1 < args.Length) opts.Filter = args[++i];
                    break;
                case "--seed":
                    if (i + 1 < args.Length && int.TryParse(args[++i], NumberStyles.Integer, CultureInfo.InvariantCulture, out int seed))
                        opts.Seed = seed;
                    break;
                case "--warmup":
                    if (i + 1 < args.Length && int.TryParse(args[++i], NumberStyles.Integer, CultureInfo.InvariantCulture, out int w))
                        opts.WarmupSteps = Math.Max(0, w);
                    break;
                case "--steps":
                    if (i + 1 < args.Length && int.TryParse(args[++i], NumberStyles.Integer, CultureInfo.InvariantCulture, out int s))
                        opts.MeasureSteps = Math.Max(1, s);
                    break;
                case "--output":
                    if (i + 1 < args.Length) opts.OutputJsonPath = args[++i];
                    break;
            }
        }
        return opts;
    }
}
