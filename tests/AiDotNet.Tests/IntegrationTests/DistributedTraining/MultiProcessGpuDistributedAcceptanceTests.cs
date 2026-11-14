using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.CheckpointManagement;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Xunit;
using Xunit.Abstractions;
#if NET8_0_OR_GREATER
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
#endif

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Multi-process, multi-GPU acceptance tests and real wall-clock throughput benchmarks for
/// distributed training across separate OS processes on heterogeneous NVIDIA GPUs
/// (<c>GPU 0 = RTX 3080</c> 12 GB, <c>GPU 1 = RTX 3090</c> 24 GB).
/// </summary>
[Collection("SequentialMultiProcessGpuDistributed")]
public sealed class MultiProcessGpuDistributedAcceptanceTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _tempRoot = Path.Combine(
        Path.GetTempPath(),
        "aidotnet-mp-gpu-" + Guid.NewGuid().ToString("N"));

    public MultiProcessGpuDistributedAcceptanceTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempRoot))
            {
                Directory.Delete(_tempRoot, recursive: true);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

#if NET8_0_OR_GREATER

    #region Worker Specification & Result DTOs

    public sealed class WorkerSpec
    {
        public string Mode { get; set; } = "StreamingTrain"; // "StreamingTrain" or "Benchmark4LayerMlp"
        public string BackendKind { get; set; } = "SharedMemory"; // "SharedMemory", "GlooTcp", "SingleProcess"
        public int Rank { get; set; }
        public int WorldSize { get; set; } = 1;
        public string SessionName { get; set; } = "aidotnet_mp_test";
        public int MasterPort { get; set; } = 29500;
        public string Strategy { get; set; } = "DDP";
        public bool UseGpu { get; set; } = true;
        public int PhysicalGpuIndex { get; set; }
        public int SampleCount { get; set; } = 32;
        public int LocalBatchSize { get; set; } = 8;
        public long MaxSteps { get; set; } = 8;
        public int DataSeed { get; set; } = 2026;
        public int InitWeightSeed { get; set; } = 77;
        public bool UseScrambledInit { get; set; }
        public int SampleDelayMs { get; set; }
        public string? CheckpointDir { get; set; }
        public int SaveFrequency { get; set; }
        public bool ResumeFromCheckpoint { get; set; }
        public int GradientBucketSizeBytes { get; set; } = 25 * 1024 * 1024;
        public bool OverlapCommunication { get; set; } = true;
        public int GlobalBatchSizeForBenchmark { get; set; } = 64;
        public int WarmupSteps { get; set; } = 5;
        public int TimedSteps { get; set; } = 30;
        public int[] BenchmarkSeeds { get; set; } = new[] { 101, 202, 303 };
        public bool DisableFusedSession { get; set; } = false;
        public string ResultPath { get; set; } = string.Empty;
    }

    public sealed class BenchmarkSeedRecord
    {
        public int Seed { get; set; }
        public int Rank { get; set; }
        public int LocalBatchSize { get; set; }
        public int GlobalBatchSize { get; set; }
        public int ParamCount { get; set; }
        public double[] StepMs { get; set; } = Array.Empty<double>();
        public double[] CommMs { get; set; } = Array.Empty<double>();
        public double[] ExposedCommMs { get; set; } = Array.Empty<double>();
        public double[] CommShare { get; set; } = Array.Empty<double>();
        public double[] ThroughputSamplesPerSec { get; set; } = Array.Empty<double>();
        public double FirstLoss { get; set; }
        public double LastLoss { get; set; }
    }

    public sealed class WorkerResult
    {
        public bool Succeeded { get; set; }
        public int ProcessId { get; set; }
        public int Rank { get; set; }
        public string EngineName { get; set; } = string.Empty;
        public bool SupportsGpu { get; set; }
        public string CudaVisibleDevices { get; set; } = string.Empty;
        public float[] FinalWeights { get; set; } = Array.Empty<float>();
        public double InitialMse { get; set; }
        public double FinalMse { get; set; }
        public double FinalLearningRate { get; set; }
        public string? ExceptionType { get; set; }
        public string? ExceptionChainTypes { get; set; }
        public string? ExceptionMessage { get; set; }
        public List<BenchmarkSeedRecord> BenchmarkRecords { get; set; } = new();
    }

    #endregion

    #region Child Worker Entrypoint (Executed Inside Spawned OS Process)

    /// <summary>
    /// Entrypoint invoked inside each isolated child <c>dotnet</c> process spawned by
    /// <see cref="StartWorkerProcess"/>. Never called directly during normal in-process xUnit discovery.
    /// </summary>
    public static async Task<int> RunChildWorkerFromSpecAsync(string specPath)
    {
        string specJson = File.ReadAllText(specPath);
        var spec = JsonSerializer.Deserialize<WorkerSpec>(specJson)
            ?? throw new InvalidOperationException("Failed to deserialize WorkerSpec.");

        var result = new WorkerResult
        {
            ProcessId = Process.GetCurrentProcess().Id,
            Rank = spec.Rank,
            CudaVisibleDevices = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? string.Empty,
        };

        try
        {
            if (spec.UseGpu)
            {
                // TestAssemblyDeterminismInit.[ModuleInitializer] sets AIDOTNET_DISABLE_GPU="1" when AiDotNetTests.dll loads;
                // clear it inside this isolated child GPU worker process so AutoDetectAndConfigureGpu binds CUDA_VISIBLE_DEVICES.
                Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", null);
                if (spec.Mode == "Benchmark4LayerMlp")
                {
                    AiDotNetEngine.SetDeterministicMode(false);
                    AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism =
                        Math.Max(1, Environment.ProcessorCount / Math.Max(1, spec.WorldSize));
                }
            }

            if (spec.Mode == "Benchmark4LayerMlp")
            {
                RunFourLayerMlpBenchmarkWorker(spec, result);
            }
            else
            {
                await RunStreamingTrainWorkerAsync(spec, result).ConfigureAwait(false);
            }

            result.Succeeded = true;
            WriteWorkerResult(spec.ResultPath, result);
            return 0;
        }
        catch (Exception ex)
        {
            result.Succeeded = false;
            result.ExceptionType = ex.GetType().FullName;
            result.ExceptionChainTypes = CollectExceptionTypes(ex);
            result.ExceptionMessage = ex.ToString();
            WriteWorkerResult(spec.ResultPath, result);
            return 2;
        }
    }

    private static string CollectExceptionTypes(Exception ex)
    {
        var names = new List<string>();
        var stack = new Stack<Exception>();
        stack.Push(ex);
        while (stack.Count > 0)
        {
            var cur = stack.Pop();
            names.Add(cur.GetType().FullName ?? cur.GetType().Name);
            if (cur.InnerException is not null)
            {
                stack.Push(cur.InnerException);
            }

            if (cur is AggregateException agg)
            {
                foreach (var inner in agg.InnerExceptions)
                {
                    stack.Push(inner);
                }
            }
        }

        return string.Join(" -> ", names);
    }

    private static void WriteWorkerResult(string path, WorkerResult result)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }

        string tmp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = false }));
        if (File.Exists(path))
        {
            File.Delete(path);
        }

        File.Move(tmp, path);
    }

    private static ICommunicationBackend<float>? CreateWorkerBackend(WorkerSpec spec)
    {
        if (spec.BackendKind == "SingleProcess" || spec.WorldSize <= 1)
        {
            return null;
        }

        if (spec.BackendKind == "GlooTcp")
        {
            Environment.SetEnvironmentVariable("AIDOTNET_MASTER_ADDR", "127.0.0.1");
            Environment.SetEnvironmentVariable("AIDOTNET_MASTER_PORT", spec.MasterPort.ToString(CultureInfo.InvariantCulture));
            return new GlooCommunicationBackend<float>(spec.Rank, spec.WorldSize);
        }

        return new SharedMemoryCommunicationBackend<float>(
            spec.Rank,
            spec.WorldSize,
            spec.SessionName,
            slotCapacityBytes: 32L * 1024L * 1024L,
            timeoutMilliseconds: 30_000);
    }

    private static GpuAccelerationConfig BuildGpuConfig(bool useGpu)
    {
        if (!useGpu)
        {
            return new GpuAccelerationConfig
            {
                DeviceType = GpuDeviceType.CPU,
                UsageLevel = GpuUsageLevel.AlwaysCpu,
            };
        }

        return new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CUDA,
            UsageLevel = GpuUsageLevel.AlwaysGpu,
            EnableGpuPersistence = true,
        };
    }

    private static async Task RunStreamingTrainWorkerAsync(WorkerSpec spec, WorkerResult result)
    {
        const int inDim = 16;
        const int hiddenDim = 32;
        const int outDim = 8;

        var (inputs, targets) = CreateSmallDataset(spec.SampleCount, inDim, outDim, seed: 505);
        var scheduler = new WarmupStableDecayScheduler(0.015, warmupSteps: 2, decayStartStep: 5, decaySteps: 6);
        var optOpts = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.015,
            MaxIterations = 100,
            UseAdaptiveLearningRate = false,
            UseEarlyStopping = false,
            Tolerance = 0.0,
            EnableGradientClipping = true,
            MaxGradientNorm = 0.5,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>(),
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optOpts);

        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(inDim),
            new DenseLayer<float>(outputSize: hiddenDim, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(outputSize: hiddenDim, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(outputSize: outDim, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inDim,
            outputSize: outDim,
            layers: layers);
        var model = new FeedForwardNeuralNetwork<float>(arch, optimizer: optimizer, lossFunction: new MeanSquaredErrorLoss<float>());
        _ = model.Predict(new Tensor<float>(new[] { 1, inDim }));

        int paramLen = model.GetParameters().Length;
        if (spec.UseScrambledInit)
        {
            var scrambled = new float[paramLen];
            for (int i = 0; i < paramLen; i++)
            {
                scrambled[i] = 0.37f * (spec.Rank + 1);
            }

            model.SetParameters(new Vector<float>(scrambled));
        }
        else
        {
            var rng = RandomHelper.CreateSeededRandom(spec.InitWeightSeed);
            var w = new float[paramLen];
            for (int i = 0; i < paramLen; i++)
            {
                w[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
            }

            model.SetParameters(new Vector<float>(w));
        }

        result.InitialMse = EvaluateDatasetMse(model, inputs, targets);

        var loader = new StreamingDataLoader<float, Tensor<float>, Tensor<float>>(
            sampleCount: spec.SampleCount,
            sampleReader: async (idx, ct) =>
            {
                if (spec.SampleDelayMs > 0)
                {
                    await Task.Delay(spec.SampleDelayMs, ct).ConfigureAwait(false);
                }

                return (inputs[idx], targets[idx]);
            },
            batchSize: spec.LocalBatchSize,
            prefetchCount: 1,
            numWorkers: 1);

        var backend = CreateWorkerBackend(spec);
        try
        {
            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(loader)
                .ConfigureGpuAcceleration(BuildGpuConfig(spec.UseGpu))
                .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                {
                    Seed = spec.DataSeed,
                    MaxSteps = spec.MaxSteps,
                    ResumeFromLatestCheckpoint = spec.ResumeFromCheckpoint,
                });

            if (!string.IsNullOrWhiteSpace(spec.CheckpointDir))
            {
                var ckpt = new CheckpointManager<float, Tensor<float>, Tensor<float>>(spec.CheckpointDir);
                ckpt.ConfigureAutoCheckpointing(
                    saveFrequency: spec.SaveFrequency,
                    keepLast: 5,
                    saveOnImprovement: false);
                builder.ConfigureCheckpointManager(ckpt);
            }

            if (backend is not null)
            {
                var strategy = (DistributedStrategy)Enum.Parse(typeof(DistributedStrategy), spec.Strategy, ignoreCase: true);
                var shardingCfg = new ShardingConfiguration<float>(backend)
                {
                    OverlapCommunication = spec.OverlapCommunication,
                    GradientBucketSizeBytes = spec.GradientBucketSizeBytes,
                };
                builder.ConfigureDistributedTraining(backend, strategy, shardingCfg);
            }

            await builder.BuildAsync().ConfigureAwait(false);

            result.EngineName = AiDotNetEngine.Current.Name;
            result.SupportsGpu = AiDotNetEngine.Current.SupportsGpu;
            result.FinalWeights = model.GetParameters().ToArray();
            result.FinalMse = EvaluateDatasetMse(model, inputs, targets);
            result.FinalLearningRate = optimizer.GetCurrentLearningRate();
        }
        finally
        {
            if (backend is not null)
            {
                try
                {
                    if (backend.IsInitialized)
                    {
                        backend.Shutdown();
                    }

                    if (backend is IDisposable disp)
                    {
                        disp.Dispose();
                    }
                }
                catch (Exception)
                {
                }
            }
        }
    }

    private static void RunFourLayerMlpBenchmarkWorker(WorkerSpec spec, WorkerResult result)
    {
        // Exact same 4-layer architecture as bench_pytorch_ddp_gloo.py:
        // 1024 -> 2048 (ReLU) -> 2048 (ReLU) -> 2048 (ReLU) -> 1024 (Identity) = 12,590,080 FP32 params (50.36 MB)
        const int inputDim = 1024;
        const int hiddenDim = 2048;
        const int outputDim = 1024;

        if (spec.UseGpu)
        {
            bool gpuReady = AiDotNetEngine.AutoDetectAndConfigureGpu();
            if (!gpuReady || !AiDotNetEngine.Current.SupportsGpu)
            {
                throw new InvalidOperationException("GPU engine failed to initialize in benchmark worker.");
            }
        }

        result.EngineName = AiDotNetEngine.Current.Name;
        result.SupportsGpu = AiDotNetEngine.Current.SupportsGpu;

        var backend = CreateWorkerBackend(spec)
            ?? throw new InvalidOperationException("Distributed benchmark worker requires a communication backend.");
        backend.Initialize();

        try
        {
            int totalIters = spec.WarmupSteps + spec.TimedSteps;
            int localBs = spec.LocalBatchSize;
            int globalBs = spec.GlobalBatchSizeForBenchmark;

            foreach (int seed in spec.BenchmarkSeeds)
            {
                var optOpts = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 1e-3,
                    MaxIterations = 1000,
                    UseAdaptiveLearningRate = false,
                    UseEarlyStopping = false,
                    Tolerance = 0.0,
                    EnableGradientClipping = false,
                    FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>(),
                };
                var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optOpts);

                var layers = new List<ILayer<float>>
                {
                    new InputLayer<float>(inputDim),
                    new DenseLayer<float>(outputSize: hiddenDim, activationFunction: new ReLUActivation<float>()),
                    new DenseLayer<float>(outputSize: hiddenDim, activationFunction: new ReLUActivation<float>()),
                    new DenseLayer<float>(outputSize: hiddenDim, activationFunction: new ReLUActivation<float>()),
                    new DenseLayer<float>(outputSize: outputDim, activationFunction: new IdentityActivation<float>()),
                };
                var arch = new NeuralNetworkArchitecture<float>(
                    inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputSize: inputDim,
                    outputSize: outputDim,
                    layers: layers);
                var model = new FeedForwardNeuralNetwork<float>(arch, optimizer: optimizer, lossFunction: new MeanSquaredErrorLoss<float>());
                _ = model.Predict(new Tensor<float>(new[] { 1, inputDim }));

                var initRng = RandomHelper.CreateSeededRandom(seed);
                int paramCount = model.GetParameters().Length;
                var initArr = new float[paramCount];
                float scale = (float)Math.Sqrt(2.0 / inputDim);
                for (int i = 0; i < paramCount; i++)
                {
                    initArr[i] = (float)(initRng.NextDouble() * 2.0 - 1.0) * scale;
                }

                model.SetParameters(new Vector<float>(initArr));
                model.SetBaseTrainOptimizer(optimizer);
                model.FusedSession.IsDisabled = spec.DisableFusedSession; // Set after Predict/SetParameters so InvalidateParameterCountCache cannot reset it

                var shardingCfg = new ShardingConfiguration<float>(backend)
                {
                    OverlapCommunication = spec.OverlapCommunication,
                    GradientBucketSizeBytes = spec.GradientBucketSizeBytes,
                };
                using var coordinator = new DistributedTapeCoordinator<float>(backend, DistributedStrategy.DDP, shardingCfg);
                model.DistributedCoordinator = coordinator;

                var trainableParams = model.Layers.OfType<ITrainableLayer<float>>().SelectMany(l => l.GetTrainableParameters()).ToList();
                coordinator.EnsureInitialParametersSynchronized(
                    trainableParams,
                    () => (AiDotNetEngine.Current as DirectGpuTensorEngine)?.InvalidateAllWeightCaches());

                // Pre-generate deterministic input/target batches outside timed region (zero target leakage)
                var dataRng = RandomHelper.CreateSeededRandom(seed * 1000 + spec.Rank * 97 + 17);
                var xBatches = new Tensor<float>[totalIters];
                var yBatches = new Tensor<float>[totalIters];
                for (int s = 0; s < totalIters; s++)
                {
                    var xArr = new float[localBs * inputDim];
                    var yArr = new float[localBs * outputDim];
                    for (int i = 0; i < xArr.Length; i++)
                    {
                        xArr[i] = (float)(dataRng.NextDouble() * 2.0 - 1.0);
                    }

                    for (int i = 0; i < yArr.Length; i++)
                    {
                        yArr[i] = (float)(dataRng.NextDouble() * 2.0 - 1.0);
                    }

                    xBatches[s] = new Tensor<float>(new[] { localBs, inputDim }, new Vector<float>(xArr));
                    yBatches[s] = new Tensor<float>(new[] { localBs, outputDim }, new Vector<float>(yArr));
                }

                var stepMs = new double[spec.TimedSteps];
                var commMs = new double[spec.TimedSteps];
                var exposedMs = new double[spec.TimedSteps];
                var commShare = new double[spec.TimedSteps];
                var tput = new double[spec.TimedSteps];
                double firstLoss = 0.0;
                double lastLoss = 0.0;

                var asyncBackend = backend as IAsyncCommunicationBackend<float>;
                backend.Barrier();

                for (int stepIdx = 0; stepIdx < totalIters; stepIdx++)
                {
                    backend.Barrier();
                    double commBefore = asyncBackend?.TotalCollectiveMilliseconds ?? 0.0;

                    long t0 = Stopwatch.GetTimestamp();
                    model.Train(xBatches[stepIdx], yBatches[stepIdx]);
                    (AiDotNetEngine.Current as DirectGpuTensorEngine)?.GetBackend()?.Synchronize();

                    long t1 = Stopwatch.GetTimestamp();

                    double rawStepMs = (t1 - t0) * 1000.0 / Stopwatch.Frequency;
                    double rawCommMs = coordinator.LastStepCommunicationMilliseconds > 0.0
                        ? coordinator.LastStepCommunicationMilliseconds
                        : Math.Max(0.0, (asyncBackend?.TotalCollectiveMilliseconds ?? 0.0) - commBefore);
                    double rawExposedMs = coordinator.LastStepExposedCommunicationMilliseconds;
                    double share = rawStepMs > 0.0 ? rawCommMs / rawStepMs : double.NaN;
                    double samplesPerSec = rawStepMs > 0.0 ? globalBs / (rawStepMs / 1000.0) : double.NaN;

                    if (stepIdx >= spec.WarmupSteps)
                    {
                        int recIdx = stepIdx - spec.WarmupSteps;
                        stepMs[recIdx] = rawStepMs;
                        commMs[recIdx] = rawCommMs;
                        exposedMs[recIdx] = rawExposedMs;
                        commShare[recIdx] = share;
                        tput[recIdx] = samplesPerSec;
                        double lossVal = Convert.ToDouble(model.GetLastLoss());
                        if (recIdx == 0)
                        {
                            firstLoss = lossVal;
                        }

                        lastLoss = lossVal;
                    }
                }

                model.DistributedCoordinator = null;

                result.BenchmarkRecords.Add(new BenchmarkSeedRecord
                {
                    Seed = seed,
                    Rank = spec.Rank,
                    LocalBatchSize = localBs,
                    GlobalBatchSize = globalBs,
                    ParamCount = paramCount,
                    StepMs = stepMs,
                    CommMs = commMs,
                    ExposedCommMs = exposedMs,
                    CommShare = commShare,
                    ThroughputSamplesPerSec = tput,
                    FirstLoss = firstLoss,
                    LastLoss = lastLoss,
                });
            }
        }
        finally
        {
            if (backend.IsInitialized)
            {
                backend.Shutdown();
            }

            if (backend is IDisposable disp)
            {
                disp.Dispose();
            }
        }
    }

    private static (Tensor<float>[] Inputs, Tensor<float>[] Targets) CreateSmallDataset(
        int count,
        int inDim,
        int outDim,
        int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var inputs = new Tensor<float>[count];
        var targets = new Tensor<float>[count];
        for (int i = 0; i < count; i++)
        {
            var x = new float[inDim];
            for (int d = 0; d < inDim; d++)
            {
                x[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var y = new float[outDim];
            for (int o = 0; o < outDim; o++)
            {
                float sum = 0.05f * (o + 1);
                for (int d = 0; d < inDim; d++)
                {
                    sum += (((d + o) & 1) == 0 ? 0.5f : -0.35f) * x[d];
                }

                y[o] = sum;
            }

            inputs[i] = new Tensor<float>(new[] { inDim }, new Vector<float>(x));
            targets[i] = new Tensor<float>(new[] { outDim }, new Vector<float>(y));
        }

        return (inputs, targets);
    }

    private static double EvaluateDatasetMse(FeedForwardNeuralNetwork<float> model, Tensor<float>[] inputs, Tensor<float>[] targets)
    {
        int n = inputs.Length;
        int inW = inputs[0].Length;
        int outW = targets[0].Length;
        var flatX = new float[n * inW];
        var flatY = new float[n * outW];
        for (int i = 0; i < n; i++)
        {
            inputs[i].Data.Span.CopyTo(flatX.AsSpan(i * inW, inW));
            targets[i].Data.Span.CopyTo(flatY.AsSpan(i * outW, outW));
        }

        var pred = model.Predict(new Tensor<float>(new[] { n, inW }, new Vector<float>(flatX)));
        var pSpan = pred.AsSpan();
        double sum = 0.0;
        for (int i = 0; i < pSpan.Length; i++)
        {
            double d = (double)pSpan[i] - flatY[i];
            sum += d * d;
        }

        return sum / pSpan.Length;
    }

    #endregion

    #region Dynamic Child Host Compilation & Process Orchestration

    private static readonly Lazy<string> WorkerHostDllPath = new Lazy<string>(BuildWorkerHostExecutable, isThreadSafe: true);
    private static readonly Lazy<IReadOnlyList<string>> DetectedNvidiaGpus = new Lazy<IReadOnlyList<string>>(QueryNvidiaSmiGpus, isThreadSafe: true);

    private static string BuildWorkerHostExecutable()
    {
        string hostDir = Path.Combine(Path.GetTempPath(), "aidotnet-dist-worker-host-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(hostDir);

        const string hostSource = @"
using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Threading.Tasks;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        string baseDir = args[0];
        string specPath = args[1];
        string testDll = Path.Combine(baseDir, ""AiDotNetTests.dll"");
        var resolver = new AssemblyDependencyResolver(testDll);

        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            string? resolved = resolver.ResolveAssemblyToPath(name);
            if (resolved != null && File.Exists(resolved))
            {
                return ctx.LoadFromAssemblyPath(resolved);
            }
            string local = Path.Combine(baseDir, name.Name + "".dll"");
            return File.Exists(local) ? ctx.LoadFromAssemblyPath(local) : null;
        };

        AssemblyLoadContext.Default.ResolvingUnmanagedDll += (asm, unmanagedName) =>
        {
            string? resolved = resolver.ResolveUnmanagedDllToPath(unmanagedName);
            if (resolved != null && File.Exists(resolved))
            {
                return NativeLibrary.Load(resolved);
            }
            return IntPtr.Zero;
        };

        var testAsm = AssemblyLoadContext.Default.LoadFromAssemblyPath(testDll);
        var testType = testAsm.GetType(
            ""AiDotNet.Tests.IntegrationTests.DistributedTraining.MultiProcessGpuDistributedAcceptanceTests"",
            throwOnError: true)!;
        var method = testType.GetMethod(""RunChildWorkerFromSpecAsync"", BindingFlags.Public | BindingFlags.Static)!;
        var task = (Task<int>)method.Invoke(null, new object[] { specPath })!;
        return await task.ConfigureAwait(false);
    }
}";

        string tpa = (string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")
            ?? throw new InvalidOperationException("TRUSTED_PLATFORM_ASSEMBLIES is unavailable.");
        var references = new List<MetadataReference>();
        foreach (string asmPath in tpa.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            string fileName = Path.GetFileName(asmPath);
            if (fileName.StartsWith("System.", StringComparison.OrdinalIgnoreCase) ||
                fileName.Equals("mscorlib.dll", StringComparison.OrdinalIgnoreCase) ||
                fileName.Equals("netstandard.dll", StringComparison.OrdinalIgnoreCase))
            {
                references.Add(MetadataReference.CreateFromFile(asmPath));
            }
        }

        var tree = CSharpSyntaxTree.ParseText(hostSource);
        var compilation = CSharpCompilation.Create(
            assemblyName: "DistRankWorkerHost",
            syntaxTrees: new[] { tree },
            references: references,
            options: new CSharpCompilationOptions(OutputKind.ConsoleApplication));

        string dllPath = Path.Combine(hostDir, "DistRankWorkerHost.dll");
        var emitResult = compilation.Emit(dllPath);
        if (!emitResult.Success)
        {
            string errors = string.Join(Environment.NewLine, emitResult.Diagnostics.Select(d => d.ToString()));
            throw new InvalidOperationException($"Failed to compile DistRankWorkerHost:{Environment.NewLine}{errors}");
        }

        string sourceRuntimeConfig = Path.Combine(AppContext.BaseDirectory, "AiDotNetTests.runtimeconfig.json");
        string targetRuntimeConfig = Path.Combine(hostDir, "DistRankWorkerHost.runtimeconfig.json");
        File.Copy(sourceRuntimeConfig, targetRuntimeConfig, overwrite: true);

        return dllPath;
    }

    private static IReadOnlyList<string> QueryNvidiaSmiGpus()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add("--query-gpu=name");
            psi.ArgumentList.Add("--format=csv,noheader");
            using var proc = Process.Start(psi);
            if (proc is null)
            {
                return Array.Empty<string>();
            }

            string output = proc.StandardOutput.ReadToEnd();
            proc.WaitForExit(5000);
            if (proc.ExitCode != 0)
            {
                return Array.Empty<string>();
            }

            return output
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => s.Length > 0)
                .ToArray();
        }
        catch (Exception)
        {
            return Array.Empty<string>();
        }
    }

    private sealed class RunningWorker : IDisposable
    {
        public RunningWorker(Process process, string specPath, string resultPath, Task<string> stdoutTask, Task<string> stderrTask)
        {
            Process = process;
            SpecPath = specPath;
            ResultPath = resultPath;
            StdoutTask = stdoutTask;
            StderrTask = stderrTask;
        }

        public Process Process { get; }
        public string SpecPath { get; }
        public string ResultPath { get; }
        public Task<string> StdoutTask { get; }
        public Task<string> StderrTask { get; }

        public void Dispose()
        {
            try
            {
                if (!Process.HasExited)
                {
                    Process.Kill(entireProcessTree: true);
                }
            }
            catch (Exception)
            {
            }

            Process.Dispose();
        }
    }

    private RunningWorker StartWorkerProcess(WorkerSpec spec)
    {
        string id = Guid.NewGuid().ToString("N");
        string specPath = Path.Combine(_tempRoot, $"worker_spec_r{spec.Rank}_{id}.json");
        string resultPath = Path.Combine(_tempRoot, $"worker_result_r{spec.Rank}_{id}.json");
        spec.ResultPath = resultPath;
        File.WriteAllText(specPath, JsonSerializer.Serialize(spec));

        string hostDll = WorkerHostDllPath.Value;
        var psi = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = AppContext.BaseDirectory,
        };
        psi.ArgumentList.Add(hostDll);
        psi.ArgumentList.Add(AppContext.BaseDirectory);
        psi.ArgumentList.Add(specPath);

        psi.Environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
        if (spec.UseGpu)
        {
            psi.Environment["CUDA_VISIBLE_DEVICES"] = spec.PhysicalGpuIndex.ToString(CultureInfo.InvariantCulture);
            psi.Environment.Remove("AIDOTNET_DISABLE_GPU");
        }
        else
        {
            psi.Environment["AIDOTNET_DISABLE_GPU"] = "1";
        }

        var proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start child dotnet worker process.");
        var stdoutTask = proc.StandardOutput.ReadToEndAsync();
        var stderrTask = proc.StandardError.ReadToEndAsync();
        return new RunningWorker(proc, specPath, resultPath, stdoutTask, stderrTask);
    }

    private static async Task<(int ExitCode, WorkerResult? Result, string Stdout, string Stderr)> WaitForWorkerAsync(
        RunningWorker worker,
        TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        try
        {
            await worker.Process.WaitForExitAsync(cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            try
            {
                worker.Process.Kill(entireProcessTree: true);
            }
            catch (Exception)
            {
            }

            throw new TimeoutException($"Child worker PID {worker.Process.Id} timed out after {timeout.TotalSeconds:F1}s.");
        }

        string stdout = await worker.StdoutTask.ConfigureAwait(false);
        string stderr = await worker.StderrTask.ConfigureAwait(false);
        WorkerResult? res = null;
        if (File.Exists(worker.ResultPath))
        {
            res = JsonSerializer.Deserialize<WorkerResult>(File.ReadAllText(worker.ResultPath));
        }

        return (worker.Process.ExitCode, res, stdout, stderr);
    }

    private static async Task<((int ExitCode, WorkerResult? Result, string Stdout, string Stderr) Rank0, (int ExitCode, WorkerResult? Result, string Stdout, string Stderr) Rank1)>
        WaitForWorkerPairAsync(RunningWorker worker0, RunningWorker worker1, TimeSpan timeout)
    {
        var t0 = WaitForWorkerAsync(worker0, timeout);
        var t1 = WaitForWorkerAsync(worker1, timeout);

        var firstDone = await Task.WhenAny(t0, t1).ConfigureAwait(false);
        var firstRes = await firstDone.ConfigureAwait(false);
        if (firstRes.ExitCode != 0 || firstRes.Result is not { Succeeded: true })
        {
            var otherTask = ReferenceEquals(firstDone, t0) ? t1 : t0;
            var otherWorker = ReferenceEquals(firstDone, t0) ? worker1 : worker0;
            var grace = await Task.WhenAny(otherTask, Task.Delay(TimeSpan.FromSeconds(3))).ConfigureAwait(false);
            if (!ReferenceEquals(grace, otherTask))
            {
                try
                {
                    otherWorker.Process.Kill(entireProcessTree: true);
                }
                catch (Exception)
                {
                }
            }
        }

        var both = await Task.WhenAll(t0, t1).ConfigureAwait(false);
        return (both[0], both[1]);
    }

    private static int s_portProbeCounter = Environment.TickCount & 0x3FFF;

    private static int FindFreePortPair()
    {
        // Stay strictly below Windows' dynamic ephemeral port pool (49152..65535) and probe IPAddress.Any
        // to match GlooCommunicationBackend's listener bind address.
        for (int attempt = 0; attempt < 60; attempt++)
        {
            int step = Interlocked.Increment(ref s_portProbeCounter);
            int basePort = 22000 + ((step * 37) % 10000) * 2;
            TcpListener? l0 = null;
            TcpListener? l1 = null;
            try
            {
                l0 = new TcpListener(IPAddress.Any, basePort);
                l0.Start();
                l1 = new TcpListener(IPAddress.Any, basePort + 1);
                l1.Start();
                l1.Stop();
                l0.Stop();
                return basePort;
            }
            catch (SocketException)
            {
                try { l1?.Stop(); } catch { }
                try { l0?.Stop(); } catch { }
            }
        }

        throw new IOException("Could not allocate two consecutive free TCP ports in 22000..42000.");
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs((double)a[i] - b[i]);
            if (d > max)
            {
                max = d;
            }
        }

        return max;
    }

    private static (double Median, double Iqr, double P95, double Mean, double Sd) Summarize(IReadOnlyList<double> values)
    {
        var sorted = values.OrderBy(x => x).ToArray();
        int n = sorted.Length;
        if (n == 0)
        {
            return (double.NaN, double.NaN, double.NaN, double.NaN, double.NaN);
        }

        double Percentile(double p)
        {
            double idx = (n - 1) * p;
            int lo = (int)Math.Floor(idx);
            int hi = (int)Math.Ceiling(idx);
            return lo == hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
        }

        double p25 = Percentile(0.25);
        double med = Percentile(0.50);
        double p75 = Percentile(0.75);
        double p95 = Percentile(0.95);
        double mean = sorted.Average();
        double sd = 0.0;
        if (n > 1)
        {
            double sumSq = 0.0;
            for (int i = 0; i < n; i++)
            {
                double d = sorted[i] - mean;
                sumSq += d * d;
            }

            sd = Math.Sqrt(sumSq / (n - 1));
        }

        return (med, p75 - p25, p95, mean, sd);
    }

    #endregion

    #region (a) 2-Process Trajectory Parity (Deterministic CPU + 2-GPU RTX 3080 & RTX 3090)

    [SkippableFact(Timeout = 240000)]
    public async Task TwoProcess_CpuDeterministicAndTwoGpuStreaming_ReproducesSingleProcessRun()
    {
        var gpus = DetectedNvidiaGpus.Value;
        Skip.If(gpus.Count < 2, $"Requires >=2 NVIDIA GPUs; found {gpus.Count}.");
        _output.WriteLine($"Detected GPUs (PCI_BUS_ID): GPU 0 = {gpus[0]}, GPU 1 = {gpus[1]}");

        // Part 1: Deterministic CPU/FMA-matched 2-process vs 1-process parity
        using (var singleCpu = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SingleProcess",
            Rank = 0,
            WorldSize = 1,
            UseGpu = false,
            SampleCount = 32,
            LocalBatchSize = 16,
            MaxSteps = 8,
            DataSeed = 2026,
            InitWeightSeed = 88,
        }))
        {
            var (code1, res1, _, err1) = await WaitForWorkerAsync(singleCpu, TimeSpan.FromSeconds(60));
            Assert.True(code1 == 0 && res1 is { Succeeded: true }, $"1-process CPU failed: {res1?.ExceptionMessage ?? err1}");

            string cpuShm = "mp_cpu_" + Guid.NewGuid().ToString("N");
            using var r0Cpu = StartWorkerProcess(new WorkerSpec
            {
                Mode = "StreamingTrain",
                BackendKind = "SharedMemory",
                Rank = 0,
                WorldSize = 2,
                SessionName = cpuShm,
                UseGpu = false,
                SampleCount = 32,
                LocalBatchSize = 8,
                MaxSteps = 8,
                DataSeed = 2026,
                InitWeightSeed = 88,
            });
            using var r1Cpu = StartWorkerProcess(new WorkerSpec
            {
                Mode = "StreamingTrain",
                BackendKind = "SharedMemory",
                Rank = 1,
                WorldSize = 2,
                SessionName = cpuShm,
                UseGpu = false,
                SampleCount = 32,
                LocalBatchSize = 8,
                MaxSteps = 8,
                DataSeed = 2026,
                InitWeightSeed = 88,
            });

            var (cpu0, cpu1) = await WaitForWorkerPairAsync(r0Cpu, r1Cpu, TimeSpan.FromSeconds(60));

            Assert.True(cpu0.ExitCode == 0 && cpu0.Result is { Succeeded: true },
                $"Rank 0 CPU failed: {cpu0.Result?.ExceptionMessage ?? cpu0.Stderr}");
            Assert.True(cpu1.ExitCode == 0 && cpu1.Result is { Succeeded: true },
                $"Rank 1 CPU failed: {cpu1.Result?.ExceptionMessage ?? cpu1.Stderr}");

            // Canonical ordered reduction across shared memory yields bit-identical post-step weights across ranks!
            Assert.Equal(0.0, MaxAbsDiff(cpu0.Result!.FinalWeights, cpu1.Result!.FinalWeights), 9);
            Assert.True(MaxAbsDiff(res1!.FinalWeights, cpu0.Result!.FinalWeights) < 1e-5,
                $"2-process CPU diverged from 1-process CPU by {MaxAbsDiff(res1.FinalWeights, cpu0.Result.FinalWeights)}.");
        }

        // Part 2: 2-Process 2-GPU (GPU 0 = RTX 3080, GPU 1 = RTX 3090) across 3 seeds vs 1-Process GPU
        int[] seeds = { 101, 202, 303 };
        var singleGpuMses = new List<double>();
        var twoGpuMses = new List<double>();

        foreach (int seed in seeds)
        {
            WorkerResult singleGpuRes;
            using (var singleGpu = StartWorkerProcess(new WorkerSpec
            {
                Mode = "StreamingTrain",
                BackendKind = "SingleProcess",
                Rank = 0,
                WorldSize = 1,
                UseGpu = true,
                PhysicalGpuIndex = 0,
                SampleCount = 32,
                LocalBatchSize = 16,
                MaxSteps = 8,
                DataSeed = seed,
                InitWeightSeed = seed + 50,
            }))
            {
                var (c, r, _, e) = await WaitForWorkerAsync(singleGpu, TimeSpan.FromSeconds(90));
                Assert.True(c == 0 && r is { Succeeded: true }, $"1-process GPU seed {seed} failed: {r?.ExceptionMessage ?? e}");
                Assert.True(r!.SupportsGpu, $"Expected GPU engine in 1-process worker, got {r.EngineName}.");
                singleGpuRes = r;
            }

            string gpuShm = $"mp_gpu_{seed}_" + Guid.NewGuid().ToString("N");
            using var r0Gpu = StartWorkerProcess(new WorkerSpec
            {
                Mode = "StreamingTrain",
                BackendKind = "SharedMemory",
                Rank = 0,
                WorldSize = 2,
                SessionName = gpuShm,
                UseGpu = true,
                PhysicalGpuIndex = 0, // RTX 3080
                SampleCount = 32,
                LocalBatchSize = 8,
                MaxSteps = 8,
                DataSeed = seed,
                InitWeightSeed = seed + 50,
            });
            using var r1Gpu = StartWorkerProcess(new WorkerSpec
            {
                Mode = "StreamingTrain",
                BackendKind = "SharedMemory",
                Rank = 1,
                WorldSize = 2,
                SessionName = gpuShm,
                UseGpu = true,
                PhysicalGpuIndex = 1, // RTX 3090
                SampleCount = 32,
                LocalBatchSize = 8,
                MaxSteps = 8,
                DataSeed = seed,
                InitWeightSeed = seed + 50,
            });

            var (gpu0, gpu1) = await WaitForWorkerPairAsync(r0Gpu, r1Gpu, TimeSpan.FromSeconds(90));

            Assert.True(gpu0.ExitCode == 0 && gpu0.Result is { Succeeded: true },
                $"Rank 0 GPU seed {seed} failed: {gpu0.Result?.ExceptionMessage ?? gpu0.Stderr}");
            Assert.True(gpu1.ExitCode == 0 && gpu1.Result is { Succeeded: true },
                $"Rank 1 GPU seed {seed} failed: {gpu1.Result?.ExceptionMessage ?? gpu1.Stderr}");

            var res0 = gpu0.Result!;
            var res1 = gpu1.Result!;
            Assert.True(res0.SupportsGpu && res1.SupportsGpu);
            Assert.Equal("0", res0.CudaVisibleDevices);
            Assert.Equal("1", res1.CudaVisibleDevices);
            Assert.NotEqual(res0.ProcessId, res1.ProcessId);

            Assert.True(res0.FinalMse < res0.InitialMse, $"2-GPU seed {seed} did not reduce MSE ({res0.InitialMse} -> {res0.FinalMse}).");
            Assert.True(MaxAbsDiff(res0.FinalWeights, res1.FinalWeights) < 1e-6,
                $"Rank 0 and Rank 1 GPU weights diverged on seed {seed}: {MaxAbsDiff(res0.FinalWeights, res1.FinalWeights)}.");
            double diffVsSingle = MaxAbsDiff(singleGpuRes.FinalWeights, res0.FinalWeights);
            _output.WriteLine($"Seed {seed}: 1-GPU FinalMSE={singleGpuRes.FinalMse:F6}, 2-GPU FinalMSE={res0.FinalMse:F6}, MaxWeightDiff={diffVsSingle:E3}");
            Assert.True(diffVsSingle < 1e-3,
                $"2-GPU (RTX 3080 + RTX 3090) diverged from 1-GPU on seed {seed} by {diffVsSingle}.");

            singleGpuMses.Add(singleGpuRes.FinalMse);
            twoGpuMses.Add(res0.FinalMse);
        }

        double seedSpread = singleGpuMses.Max() - singleGpuMses.Min();
        double meanMseDiff = Math.Abs(singleGpuMses.Average() - twoGpuMses.Average());
        Assert.True(meanMseDiff <= Math.Max(1e-5, seedSpread),
            $"Mean MSE diff ({meanMseDiff:E3}) exceeded 3-seed spread ({seedSpread:E3}).");
    }

    #endregion

    #region (b) Multi-Process Cross-World-Size Checkpoint Resume (2->1 and 1->2 on Real GPUs)

    [SkippableFact(Timeout = 240000)]
    public async Task TwoProcess_CrossWorldSizeCheckpointResume_OnRealGpus_MatchesUninterruptedRun()
    {
        var gpus = DetectedNvidiaGpus.Value;
        Skip.If(gpus.Count < 2, $"Requires >=2 NVIDIA GPUs; found {gpus.Count}.");

        // Uninterrupted 8-step 1-process GPU baseline (B=16)
        WorkerResult uninterrupted;
        using (var straight = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SingleProcess",
            Rank = 0,
            WorldSize = 1,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 16,
            MaxSteps = 8,
            DataSeed = 9090,
            InitWeightSeed = 333,
        }))
        {
            var (c, r, _, e) = await WaitForWorkerAsync(straight, TimeSpan.FromSeconds(90));
            Assert.True(c == 0 && r is { Succeeded: true }, $"Uninterrupted 1-GPU failed: {r?.ExceptionMessage ?? e}");
            uninterrupted = r!;
        }

        // Case A: 2-process GPU (ZeRO2, B0=8, B1=8) for 4 steps -> resume on 1-process GPU (B=16) to step 8
        string ckpt2To1 = Path.Combine(_tempRoot, "mp_ckpt_2to1");
        string shm2To1 = "mp_2to1_" + Guid.NewGuid().ToString("N");
        using (var p0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "ZeRO2",
            Rank = 0,
            WorldSize = 2,
            SessionName = shm2To1,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 4,
            DataSeed = 9090,
            InitWeightSeed = 333,
            CheckpointDir = ckpt2To1,
            SaveFrequency = 4,
        }))
        using (var p1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "ZeRO2",
            Rank = 1,
            WorldSize = 2,
            SessionName = shm2To1,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 4,
            DataSeed = 9090,
            InitWeightSeed = 333,
            CheckpointDir = ckpt2To1,
            SaveFrequency = 4,
        }))
        {
            var (ph0, ph1) = await WaitForWorkerPairAsync(p0, p1, TimeSpan.FromSeconds(90));
            Assert.True(ph0.ExitCode == 0 && ph0.Result is { Succeeded: true },
                $"2->1 Phase 1 Rank 0 failed: {ph0.Result?.ExceptionMessage ?? ph0.Stderr}");
            Assert.True(ph1.ExitCode == 0 && ph1.Result is { Succeeded: true },
                $"2->1 Phase 1 Rank 1 failed: {ph1.Result?.ExceptionMessage ?? ph1.Stderr}");
        }

        using (var resume1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SingleProcess",
            Rank = 0,
            WorldSize = 1,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 16,
            MaxSteps = 8,
            DataSeed = 9090,
            UseScrambledInit = true,
            CheckpointDir = ckpt2To1,
            SaveFrequency = 100,
            ResumeFromCheckpoint = true,
        }))
        {
            var (c, r, _, e) = await WaitForWorkerAsync(resume1, TimeSpan.FromSeconds(90));
            Assert.True(c == 0 && r is { Succeeded: true }, $"2->1 Phase 2 failed: {r?.ExceptionMessage ?? e}");
            Assert.True(MaxAbsDiff(uninterrupted.FinalWeights, r!.FinalWeights) < 1e-3,
                $"Multi-process 2->1 GPU resume diverged by {MaxAbsDiff(uninterrupted.FinalWeights, r.FinalWeights)}.");
            Assert.Equal(uninterrupted.FinalLearningRate, r.FinalLearningRate, 9);
        }

        // Case B: 1-process GPU (B=16) for 4 steps -> resume on 2-process GPU (FSDP, uneven B0=4, B1=12) to step 8
        string ckpt1To2 = Path.Combine(_tempRoot, "mp_ckpt_1to2");
        using (var singlePhase1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SingleProcess",
            Rank = 0,
            WorldSize = 1,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 16,
            MaxSteps = 4,
            DataSeed = 9090,
            InitWeightSeed = 333,
            CheckpointDir = ckpt1To2,
            SaveFrequency = 4,
        }))
        {
            var (c, r, _, e) = await WaitForWorkerAsync(singlePhase1, TimeSpan.FromSeconds(90));
            Assert.True(c == 0 && r is { Succeeded: true }, $"1->2 Phase 1 failed: {r?.ExceptionMessage ?? e}");
        }

        string shm1To2 = "mp_1to2_" + Guid.NewGuid().ToString("N");
        using (var r0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "FSDP",
            Rank = 0,
            WorldSize = 2,
            SessionName = shm1To2,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 4, // RTX 3080 takes 4
            MaxSteps = 8,
            DataSeed = 9090,
            UseScrambledInit = true,
            CheckpointDir = ckpt1To2,
            SaveFrequency = 100,
            ResumeFromCheckpoint = true,
        }))
        using (var r1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "FSDP",
            Rank = 1,
            WorldSize = 2,
            SessionName = shm1To2,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 32,
            LocalBatchSize = 12, // RTX 3090 takes 12
            MaxSteps = 8,
            DataSeed = 9090,
            UseScrambledInit = true,
            CheckpointDir = ckpt1To2,
            SaveFrequency = 100,
            ResumeFromCheckpoint = true,
        }))
        {
            var (rb0, rb1) = await WaitForWorkerPairAsync(r0, r1, TimeSpan.FromSeconds(90));
            Assert.True(rb0.ExitCode == 0 && rb0.Result is { Succeeded: true },
                $"1->2 Phase 2 Rank 0 failed: {rb0.Result?.ExceptionMessage ?? rb0.Stderr}");
            Assert.True(rb1.ExitCode == 0 && rb1.Result is { Succeeded: true },
                $"1->2 Phase 2 Rank 1 failed: {rb1.Result?.ExceptionMessage ?? rb1.Stderr}");

            Assert.True(MaxAbsDiff(rb0.Result!.FinalWeights, rb1.Result!.FinalWeights) < 1e-6);
            Assert.True(MaxAbsDiff(uninterrupted.FinalWeights, rb0.Result.FinalWeights) < 1e-3,
                $"Multi-process 1->2 GPU FSDP resume diverged by {MaxAbsDiff(uninterrupted.FinalWeights, rb0.Result.FinalWeights)}.");
        }
    }

    [SkippableFact(Timeout = 240000)]
    public async Task TwoProcess_SequentialMultiCheckpointGpuSaveStepSaveResume_MatchesUninterruptedRun()
    {
        var gpus = DetectedNvidiaGpus.Value;
        Skip.If(gpus.Count < 2, $"Requires >=2 NVIDIA GPUs; found {gpus.Count}.");

        // Uninterrupted 8-step 2-process GPU DDP baseline (B0=8, B1=8 -> GlobalBatch=16)
        string refShm = "mp_multickpt_ref_" + Guid.NewGuid().ToString("N");
        WorkerResult uninterrupted;
        using (var ref0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 0,
            WorldSize = 2,
            SessionName = refShm,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 8,
            DataSeed = 6060,
            InitWeightSeed = 222,
        }))
        using (var ref1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 1,
            WorldSize = 2,
            SessionName = refShm,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 8,
            DataSeed = 6060,
            InitWeightSeed = 222,
        }))
        {
            var (u0, u1) = await WaitForWorkerPairAsync(ref0, ref1, TimeSpan.FromSeconds(90));
            Assert.True(u0.ExitCode == 0 && u0.Result is { Succeeded: true }, $"Uninterrupted Rank 0 failed: {u0.Result?.ExceptionMessage ?? u0.Stderr}");
            Assert.True(u1.ExitCode == 0 && u1.Result is { Succeeded: true }, $"Uninterrupted Rank 1 failed: {u1.Result?.ExceptionMessage ?? u1.Stderr}");
            uninterrupted = u0.Result!;
        }

        // Phase 1: Train 4 steps with SaveFrequency = 2 so Checkpoint #1 (step 2) downloads GPU moments via HostSync,
        // steps 3..4 update GPU moments in-place (re-arming HostSync), and Checkpoint #2 (step 4) must serialize fresh step-4 moments!
        string ckptMulti = Path.Combine(_tempRoot, "mp_ckpt_sequential_multi");
        string shmP1 = "mp_multickpt_p1_" + Guid.NewGuid().ToString("N");
        WorkerResult step4Res;
        using (var p0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 0,
            WorldSize = 2,
            SessionName = shmP1,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 4,
            DataSeed = 6060,
            InitWeightSeed = 222,
            CheckpointDir = ckptMulti,
            SaveFrequency = 2,
        }))
        using (var p1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 1,
            WorldSize = 2,
            SessionName = shmP1,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 4,
            DataSeed = 6060,
            InitWeightSeed = 222,
            CheckpointDir = ckptMulti,
            SaveFrequency = 2,
        }))
        {
            var (ph0, ph1) = await WaitForWorkerPairAsync(p0, p1, TimeSpan.FromSeconds(90));
            Assert.True(ph0.ExitCode == 0 && ph0.Result is { Succeeded: true }, $"Multi-ckpt Phase 1 Rank 0 failed: {ph0.Result?.ExceptionMessage ?? ph0.Stderr}");
            Assert.True(ph1.ExitCode == 0 && ph1.Result is { Succeeded: true }, $"Multi-ckpt Phase 1 Rank 1 failed: {ph1.Result?.ExceptionMessage ?? ph1.Stderr}");
            step4Res = ph0.Result!;
        }

        var savedPhase1Files = Directory.GetFiles(ckptMulti, "*.json")
            .Where(f => f.IndexOf(".tmp", StringComparison.OrdinalIgnoreCase) < 0)
            .ToArray();
        Assert.True(savedPhase1Files.Length >= 2, $"Expected >=2 sequential checkpoints in Phase 1 (step 2 and step 4), found {savedPhase1Files.Length}.");
        Assert.True(MaxAbsDiff(step4Res.FinalWeights, uninterrupted.FinalWeights) > 1e-3,
            "Step 4 weights must differ from step 8 weights so sequential checkpoint resume verification is non-vacuous.");

        // Phase 2: Resume from Checkpoint #2 (step 4) on 2 GPUs with SaveFrequency = 2 to MaxSteps = 8
        // (exercises Deserialize GPU moment promotion -> step 6 Checkpoint #3 -> step 8 Checkpoint #4!)
        string shmP2 = "mp_multickpt_p2_" + Guid.NewGuid().ToString("N");
        using (var r0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 0,
            WorldSize = 2,
            SessionName = shmP2,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 8,
            DataSeed = 6060,
            UseScrambledInit = true,
            CheckpointDir = ckptMulti,
            SaveFrequency = 2,
            ResumeFromCheckpoint = true,
        }))
        using (var r1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 1,
            WorldSize = 2,
            SessionName = shmP2,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 32,
            LocalBatchSize = 8,
            MaxSteps = 8,
            DataSeed = 6060,
            UseScrambledInit = true,
            CheckpointDir = ckptMulti,
            SaveFrequency = 2,
            ResumeFromCheckpoint = true,
        }))
        {
            var (res0, res1) = await WaitForWorkerPairAsync(r0, r1, TimeSpan.FromSeconds(90));
            Assert.True(res0.ExitCode == 0 && res0.Result is { Succeeded: true }, $"Multi-ckpt Phase 2 Rank 0 failed: {res0.Result?.ExceptionMessage ?? res0.Stderr}");
            Assert.True(res1.ExitCode == 0 && res1.Result is { Succeeded: true }, $"Multi-ckpt Phase 2 Rank 1 failed: {res1.Result?.ExceptionMessage ?? res1.Stderr}");

            Assert.True(MaxAbsDiff(res0.Result!.FinalWeights, res1.Result!.FinalWeights) < 1e-6);
            double diff = MaxAbsDiff(uninterrupted.FinalWeights, res0.Result.FinalWeights);
            _output.WriteLine($"Sequential multi-checkpoint GPU save(2)->step->save(4)->resume->save(6)->save(8) MaxWeightDiff vs uninterrupted = {diff:E3}");
            Assert.True(diff < 1e-4,
                $"Sequential multi-checkpoint GPU save->step->save->resume diverged from uninterrupted 2-GPU run by {diff}.");
            Assert.Equal(uninterrupted.FinalLearningRate, res0.Result.FinalLearningRate, 9);
        }
    }

    #endregion

    #region (c) Mid-Run Rank Kill & Fault-Tolerant Checkpoint Resume

    [SkippableFact(Timeout = 240000)]
    public async Task MidRunRankKill_Rank0DetectsDeadPeerPidQuicklyAndResumesCleanlyToMaxSteps()
    {
        var gpus = DetectedNvidiaGpus.Value;
        Skip.If(gpus.Count < 2, $"Requires >=2 NVIDIA GPUs; found {gpus.Count}.");

        // Reference uninterrupted 10-step 2-GPU run (GlobalBatch = 8)
        string refShm = "mp_kill_ref_" + Guid.NewGuid().ToString("N");
        WorkerResult expectedRef;
        using (var ref0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 0,
            WorldSize = 2,
            SessionName = refShm,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 40,
            LocalBatchSize = 4,
            MaxSteps = 10,
            DataSeed = 7171,
            InitWeightSeed = 444,
        }))
        using (var ref1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 1,
            WorldSize = 2,
            SessionName = refShm,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 40,
            LocalBatchSize = 4,
            MaxSteps = 10,
            DataSeed = 7171,
            InitWeightSeed = 444,
        }))
        {
            var (refRes0, refRes1) = await WaitForWorkerPairAsync(ref0, ref1, TimeSpan.FromSeconds(90));
            Assert.True(refRes0.ExitCode == 0 && refRes0.Result is { Succeeded: true },
                $"Reference Rank 0 failed: {refRes0.Result?.ExceptionMessage ?? refRes0.Stderr}");
            Assert.True(refRes1.ExitCode == 0 && refRes1.Result is { Succeeded: true },
                $"Reference Rank 1 failed: {refRes1.Result?.ExceptionMessage ?? refRes1.Stderr}");
            expectedRef = refRes0.Result!;
        }

        // Phase 1: Spawn Rank 0 + Rank 1 with SampleDelayMs = 45 (~180ms/step) and SaveFrequency = 2.
        // Wait until >=1 atomic checkpoint JSON file is published, then kill Rank 1 mid-run!
        string ckptDir = Path.Combine(_tempRoot, "mp_kill_ckpt");
        string killShm = "mp_kill_p1_" + Guid.NewGuid().ToString("N");

        using (var rank0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 0,
            WorldSize = 2,
            SessionName = killShm,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 40,
            LocalBatchSize = 4,
            MaxSteps = 10,
            DataSeed = 7171,
            InitWeightSeed = 444,
            SampleDelayMs = 45,
            CheckpointDir = ckptDir,
            SaveFrequency = 2,
        }))
        using (var rank1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 1,
            WorldSize = 2,
            SessionName = killShm,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 40,
            LocalBatchSize = 4,
            MaxSteps = 10,
            DataSeed = 7171,
            InitWeightSeed = 444,
            SampleDelayMs = 45,
            CheckpointDir = ckptDir,
            SaveFrequency = 2,
        }))
        {
            // Poll checkpoint directory until Rank 0 atomically publishes >=1 checkpoint JSON
            long waitStart = Stopwatch.GetTimestamp();
            while (true)
            {
                if (Directory.Exists(ckptDir))
                {
                    var committed = Directory.GetFiles(ckptDir, "*.json")
                        .Where(f => f.IndexOf(".tmp", StringComparison.OrdinalIgnoreCase) < 0)
                        .ToArray();
                    if (committed.Length >= 1)
                    {
                        break;
                    }
                }

                if (rank0.Process.HasExited || rank1.Process.HasExited)
                {
                    var (_, r0Early, _, err0Early) = await WaitForWorkerAsync(rank0, TimeSpan.FromSeconds(5));
                    throw new InvalidOperationException(
                        $"Worker exited before saving first checkpoint: {r0Early?.ExceptionMessage ?? err0Early}");
                }

                double waitedSec = (Stopwatch.GetTimestamp() - waitStart) / (double)Stopwatch.Frequency;
                if (waitedSec > 30.0)
                {
                    throw new TimeoutException("Timed out waiting for first auto-checkpoint before killing Rank 1.");
                }

                await Task.Delay(25);
            }

            // Kill Rank 1 abruptly mid-run!
            long killTicks = Stopwatch.GetTimestamp();
            rank1.Process.Kill(entireProcessTree: true);

            // Rank 0 MUST detect dead peer PID and fail fast with IOException within seconds (not 120s timeout!)
            var (r0ExitCode, r0Res, _, r0Stderr) = await WaitForWorkerAsync(rank0, TimeSpan.FromSeconds(15));
            double detectSeconds = (Stopwatch.GetTimestamp() - killTicks) / (double)Stopwatch.Frequency;
            _output.WriteLine($"Rank 0 detected killed Rank 1 (PID {rank1.Process.Id}) in {detectSeconds * 1000.0:F1} ms.");

            Assert.NotEqual(0, r0ExitCode);
            Assert.True(detectSeconds < 10.0, $"Rank 0 took too long ({detectSeconds:F2}s) to detect dead peer.");
            Assert.NotNull(r0Res);
            Assert.False(r0Res!.Succeeded);
            Assert.Contains("System.IO.IOException", r0Res.ExceptionChainTypes ?? string.Empty, StringComparison.Ordinal);
        }

        // Phase 2: Respawn Rank 0 + Rank 1 with ResumeFromCheckpoint = true from scrambled weights and complete to MaxSteps=10!
        string resumeShm = "mp_kill_p2_" + Guid.NewGuid().ToString("N");
        using (var res0 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 0,
            WorldSize = 2,
            SessionName = resumeShm,
            UseGpu = true,
            PhysicalGpuIndex = 0,
            SampleCount = 40,
            LocalBatchSize = 4,
            MaxSteps = 10,
            DataSeed = 7171,
            UseScrambledInit = true,
            SampleDelayMs = 0,
            CheckpointDir = ckptDir,
            SaveFrequency = 100,
            ResumeFromCheckpoint = true,
        }))
        using (var res1 = StartWorkerProcess(new WorkerSpec
        {
            Mode = "StreamingTrain",
            BackendKind = "SharedMemory",
            Strategy = "DDP",
            Rank = 1,
            WorldSize = 2,
            SessionName = resumeShm,
            UseGpu = true,
            PhysicalGpuIndex = 1,
            SampleCount = 40,
            LocalBatchSize = 4,
            MaxSteps = 10,
            DataSeed = 7171,
            UseScrambledInit = true,
            SampleDelayMs = 0,
            CheckpointDir = ckptDir,
            SaveFrequency = 100,
            ResumeFromCheckpoint = true,
        }))
        {
            var (rec0, rec1) = await WaitForWorkerPairAsync(res0, res1, TimeSpan.FromSeconds(90));

            Assert.True(rec0.ExitCode == 0 && rec0.Result is { Succeeded: true },
                $"Respawned Rank 0 failed: {rec0.Result?.ExceptionMessage ?? rec0.Stderr}");
            Assert.True(rec1.ExitCode == 0 && rec1.Result is { Succeeded: true },
                $"Respawned Rank 1 failed: {rec1.Result?.ExceptionMessage ?? rec1.Stderr}");

            Assert.True(MaxAbsDiff(rec0.Result!.FinalWeights, rec1.Result!.FinalWeights) < 1e-6);
            double diffVsClean = MaxAbsDiff(expectedRef.FinalWeights, rec0.Result.FinalWeights);
            _output.WriteLine($"Kill & Resume recovered to MaxSteps=10 cleanly; MaxWeightDiff vs uninterrupted = {diffVsClean:E3}");
            Assert.True(diffVsClean < 1e-4,
                $"Recovered run after Rank 1 kill diverged from uninterrupted 2-GPU run by {diffVsClean}.");
        }
    }

    #endregion

    #region (d) 2-Process 2-GPU Benchmark (SharedMemory + Gloo TCP) vs PyTorch DDP (gloo)

    [SkippableFact(Timeout = 600000)]
    public async Task Benchmark_TwoProcessTwoGpu_SharedMemoryAndGlooTcp_OutputsRawStopwatchMetrics()
    {
        var gpus = DetectedNvidiaGpus.Value;
        Skip.If(gpus.Count < 2, $"Requires >=2 NVIDIA GPUs; found {gpus.Count}.");

        bool fullSweep = string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_DIST_BENCH_FULL"), "1", StringComparison.Ordinal)
            || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AIDOTNET_DIST_BENCH_OUT"));
        bool disableFused = string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_DIST_BENCH_DISABLE_FUSED"), "1", StringComparison.Ordinal);
        bool controlOnly = string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_DIST_BENCH_CONTROL_ONLY"), "1", StringComparison.Ordinal);
        int warmupSteps = fullSweep ? 5 : 2;
        int timedSteps = fullSweep ? 30 : 3;
        int[] benchSeeds = fullSweep ? new[] { 101, 202, 303 } : new[] { 101 };

        var allCases = new (string Label, string BackendKind, int B0, int B1, int BucketBytes)[]
        {
            ("SharedMemory_Homogeneous_32_32_Cap25MB", "SharedMemory", 32, 32, 25 * 1024 * 1024),
            ("SharedMemory_Heterogeneous_16_48_Cap25MB", "SharedMemory", 16, 48, 25 * 1024 * 1024),
            ("SharedMemory_Homogeneous_32_32_Cap8MB", "SharedMemory", 32, 32, 8 * 1024 * 1024),
            ("SharedMemory_Heterogeneous_16_48_Cap8MB", "SharedMemory", 16, 48, 8 * 1024 * 1024),
            ("GlooTcp_Homogeneous_32_32_Cap25MB", "GlooTcp", 32, 32, 25 * 1024 * 1024),
            ("GlooTcp_Heterogeneous_16_48_Cap25MB", "GlooTcp", 16, 48, 25 * 1024 * 1024),
            ("GlooTcp_Homogeneous_32_32_Cap8MB", "GlooTcp", 32, 32, 8 * 1024 * 1024),
            ("GlooTcp_Heterogeneous_16_48_Cap8MB", "GlooTcp", 16, 48, 8 * 1024 * 1024),
        };
        var cases = controlOnly
            ? allCases.Where(c => c.B0 == 32 && c.B1 == 32 && c.BucketBytes == 25 * 1024 * 1024).ToArray()
            : allCases;

        var armSummaries = new List<Dictionary<string, object>>();

        foreach (var arm in cases)
        {
            int globalBs = arm.B0 + arm.B1;
            (int ExitCode, WorkerResult? Result, string Stdout, string Stderr) f0 = default;
            (int ExitCode, WorkerResult? Result, string Stdout, string Stderr) f1 = default;

            int maxAttempts = arm.BackendKind == "GlooTcp" ? 3 : 1;
            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                string sessionName = "bench_" + Guid.NewGuid().ToString("N");
                int basePort = arm.BackendKind == "GlooTcp" ? FindFreePortPair() : 29500;

                using var w0 = StartWorkerProcess(new WorkerSpec
                {
                    Mode = "Benchmark4LayerMlp",
                    BackendKind = arm.BackendKind,
                    Rank = 0,
                    WorldSize = 2,
                    SessionName = sessionName,
                    MasterPort = basePort,
                    UseGpu = true,
                    PhysicalGpuIndex = 0, // RTX 3080
                    LocalBatchSize = arm.B0,
                    GlobalBatchSizeForBenchmark = globalBs,
                    GradientBucketSizeBytes = arm.BucketBytes,
                    OverlapCommunication = true,
                    WarmupSteps = warmupSteps,
                    TimedSteps = timedSteps,
                    BenchmarkSeeds = benchSeeds,
                    DisableFusedSession = disableFused,
                });
                using var w1 = StartWorkerProcess(new WorkerSpec
                {
                    Mode = "Benchmark4LayerMlp",
                    BackendKind = arm.BackendKind,
                    Rank = 1,
                    WorldSize = 2,
                    SessionName = sessionName,
                    MasterPort = basePort,
                    UseGpu = true,
                    PhysicalGpuIndex = 1, // RTX 3090
                    LocalBatchSize = arm.B1,
                    GlobalBatchSizeForBenchmark = globalBs,
                    GradientBucketSizeBytes = arm.BucketBytes,
                    OverlapCommunication = true,
                    WarmupSteps = warmupSteps,
                    TimedSteps = timedSteps,
                    BenchmarkSeeds = benchSeeds,
                    DisableFusedSession = disableFused,
                });

                (f0, f1) = await WaitForWorkerPairAsync(w0, w1, TimeSpan.FromSeconds(180));
                if (f0.ExitCode == 0 && f0.Result is { Succeeded: true } &&
                    f1.ExitCode == 0 && f1.Result is { Succeeded: true })
                {
                    break;
                }

                string combinedErr = $"{f0.Result?.ExceptionChainTypes} {f0.Result?.ExceptionMessage} {f1.Result?.ExceptionChainTypes} {f1.Result?.ExceptionMessage}";
                if (attempt < maxAttempts && combinedErr.IndexOf("SocketException", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    _output.WriteLine($"[Retry {attempt}/{maxAttempts}] {arm.Label} transient TCP port collision on {basePort}; retrying with fresh port pair.");
                    continue;
                }

                break;
            }

            Assert.True(f0.ExitCode == 0 && f0.Result is { Succeeded: true },
                $"{arm.Label} Rank 0 failed: {f0.Result?.ExceptionMessage ?? f0.Stderr}");
            Assert.True(f1.ExitCode == 0 && f1.Result is { Succeeded: true },
                $"{arm.Label} Rank 1 failed: {f1.Result?.ExceptionMessage ?? f1.Stderr}");

            var r0 = f0.Result!;
            var r1 = f1.Result!;
            Assert.True(r0.SupportsGpu && r1.SupportsGpu);
            Assert.Equal(benchSeeds.Length, r0.BenchmarkRecords.Count);
            Assert.Equal(benchSeeds.Length, r1.BenchmarkRecords.Count);
            Assert.Equal(12_590_080, r0.BenchmarkRecords[0].ParamCount);

            var globalStepMs = new List<double>();
            var globalTput = new List<double>();
            var rank0CommMs = new List<double>();
            var rank0ExposedMs = new List<double>();
            var rank0CommShare = new List<double>();

            for (int s = 0; s < r0.BenchmarkRecords.Count; s++)
            {
                var s0 = r0.BenchmarkRecords[s];
                var s1 = r1.BenchmarkRecords[s];
                for (int i = 0; i < s0.StepMs.Length; i++)
                {
                    double maxStep = Math.Max(s0.StepMs[i], s1.StepMs[i]);
                    globalStepMs.Add(maxStep);
                    globalTput.Add(globalBs / (maxStep / 1000.0));
                    rank0CommMs.Add(s0.CommMs[i]);
                    rank0ExposedMs.Add(s0.ExposedCommMs[i]);
                    rank0CommShare.Add(s0.CommShare[i]);
                }
            }

            var stepStats = Summarize(globalStepMs);
            var tputStats = Summarize(globalTput);
            var commStats = Summarize(rank0CommMs);
            var expStats = Summarize(rank0ExposedMs);
            var shareStats = Summarize(rank0CommShare);

            _output.WriteLine(
                $"[AiDotNet 2-GPU {arm.Label}] " +
                $"step_ms median={stepStats.Median:F3} (IQR={stepStats.Iqr:F3}, p95={stepStats.P95:F3}, mean±SD={stepStats.Mean:F3}±{stepStats.Sd:F3}), " +
                $"comm_ms median={commStats.Median:F3} (exposed={expStats.Median:F3}), " +
                $"comm_share median={shareStats.Median * 100.0:F1}%, " +
                $"throughput median={tputStats.Median:F1} samples/s");

            Assert.True(stepStats.Median > 0.0 && !double.IsNaN(stepStats.Median));
            Assert.True(tputStats.Median > 0.0 && !double.IsNaN(tputStats.Median));

            armSummaries.Add(new Dictionary<string, object>
            {
                ["label"] = arm.Label,
                ["backend"] = arm.BackendKind,
                ["execution_mode"] = disableFused ? "eager_tape_control" : "fused_session_default",
                ["disable_fused_session"] = disableFused,
                ["batch_per_rank"] = new[] { arm.B0, arm.B1 },
                ["global_batch_size"] = globalBs,
                ["bucket_size_bytes"] = arm.BucketBytes,
                ["gpu_0"] = gpus[0],
                ["gpu_1"] = gpus[1],
                ["step_ms_median"] = stepStats.Median,
                ["step_ms_iqr"] = stepStats.Iqr,
                ["step_ms_p95"] = stepStats.P95,
                ["step_ms_mean"] = stepStats.Mean,
                ["step_ms_sd"] = stepStats.Sd,
                ["comm_ms_median"] = commStats.Median,
                ["comm_ms_iqr"] = commStats.Iqr,
                ["exposed_comm_ms_median"] = expStats.Median,
                ["comm_share_median"] = shareStats.Median,
                ["throughput_samples_per_sec_median"] = tputStats.Median,
                ["throughput_samples_per_sec_iqr"] = tputStats.Iqr,
                ["raw_rank0"] = r0.BenchmarkRecords,
                ["raw_rank1"] = r1.BenchmarkRecords,
            });
        }

        var ledger = new Dictionary<string, object>
        {
            ["benchmark"] = "AiDotNet 2-Process 2-GPU Distributed Benchmark (RTX 3080 + RTX 3090)",
            ["timestamp_utc"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture),
            ["model_spec"] = new Dictionary<string, object>
            {
                ["Architecture"] = "4-Layer MLP (1024->2048->2048->2048->1024, ReLU, FP32)",
                ["CheckpointInit"] = "RandomSeededInit (ScaledUniform, Untrained Throughput Benchmark)",
                ["L"] = 4,
                ["D"] = 2048,
                ["InputDim"] = 1024,
                ["OutputDim"] = 1024,
                ["V"] = 0,
                ["ContextLength"] = 1,
                ["Params"] = 12_590_080,
                ["ParamBytesFp32"] = 50_360_320,
                ["Optimizer"] = "AdamOptimizer(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)",
                ["Loss"] = "MeanSquaredErrorLoss(sample-weighted across ranks)",
            },
            ["protocol"] = new Dictionary<string, object>
            {
                ["mode"] = fullSweep ? "full_3seed_90step" : "ci_smoke_1seed_3step",
                ["execution_mode"] = disableFused ? "eager_tape_control" : "fused_session_default",
                ["disable_fused_session"] = disableFused,
                ["control_only"] = controlOnly,
                ["warmup_steps"] = warmupSteps,
                ["timed_steps_per_seed"] = timedSteps,
                ["seeds"] = benchSeeds,
                ["total_timed_steps_per_arm"] = timedSteps * benchSeeds.Length,
                ["timing_method"] = "DirectGpuTensorEngine.GetBackend().Synchronize() + raw Stopwatch.GetTimestamp()",
            },
            ["runs"] = armSummaries,
        };

        string outJson = JsonSerializer.Serialize(ledger, new JsonSerializerOptions { WriteIndented = true });
        string localLedgerPath = Path.Combine(_tempRoot, "aidotnet_2gpu_bench_results.json");
        File.WriteAllText(localLedgerPath, outJson);

        string? optionalOutPath = Environment.GetEnvironmentVariable("AIDOTNET_DIST_BENCH_OUT");
        if (!string.IsNullOrWhiteSpace(optionalOutPath))
        {
            string? dir = Path.GetDirectoryName(optionalOutPath);
            if (!string.IsNullOrWhiteSpace(dir))
            {
                Directory.CreateDirectory(dir);
            }

            File.WriteAllText(optionalOutPath, outJson);
            _output.WriteLine($"Saved AiDotNet 2-GPU benchmark JSON ledger to: {optionalOutPath}");
        }
    }

    #endregion

#else

    [Fact]
    public void MultiProcessGpuSuite_CompilesCleanlyOnLegacyFramework()
    {
        // Multi-process child-host execution runs on net10.0; net471 compiles cleanly without executing.
        Assert.NotNull(_output);
    }

#endif
}
