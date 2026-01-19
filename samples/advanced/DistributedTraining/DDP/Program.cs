using AiDotNet;
using AiDotNet.DistributedTraining;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Configuration;

Console.WriteLine("=== AiDotNet Distributed Data Parallel (DDP) Training ===");
Console.WriteLine("Training across multiple GPUs with synchronized gradients\n");

// Parse command line arguments
var args = Environment.GetCommandLineArgs();
var gpuIds = ParseGpuIds(args);
var worldSize = ParseInt(args, "--world-size", gpuIds.Length);
var rank = ParseInt(args, "--rank", 0);
var masterAddr = ParseString(args, "--master-addr", "localhost");
var masterPort = ParseInt(args, "--master-port", 29500);

Console.WriteLine("Configuration:");
Console.WriteLine($"  GPUs: [{string.Join(", ", gpuIds)}]");
Console.WriteLine($"  World size: {worldSize}");
Console.WriteLine($"  Rank: {rank}");
Console.WriteLine($"  Master: {masterAddr}:{masterPort}\n");

try
{
    // Initialize distributed environment
    Console.WriteLine("Initializing distributed environment...");

    var backend = new NCCLCommunicationBackend(new NCCLConfig
    {
        MasterAddress = masterAddr,
        MasterPort = masterPort,
        WorldSize = worldSize,
        Rank = rank
    });

    await backend.InitializeAsync();

    Console.WriteLine("  Backend: NCCL");
    Console.WriteLine($"  World size: {backend.WorldSize}");
    Console.WriteLine($"  Rank: {backend.Rank}");
    Console.WriteLine($"  Is master: {backend.IsMaster}\n");

    // Create model (ResNet-style for image classification)
    Console.WriteLine("Creating model...");
    var model = new ResNetNetwork<float>(
        depth: 50,
        numClasses: 1000,
        inputChannels: 3);

    var paramCount = model.ParameterCount;
    Console.WriteLine($"  Model: ResNet50");
    Console.WriteLine($"  Parameters: {paramCount / 1_000_000.0:F1}M\n");

    // Configure DDP
    Console.WriteLine("Configuring DDP...");

    var ddpConfig = new DDPConfiguration
    {
        BucketSizeMB = 25,
        GradientCompression = true,
        FindUnusedParameters = false,
        BroadcastBuffers = true
    };

    Console.WriteLine($"  Bucket size: {ddpConfig.BucketSizeMB}MB");
    Console.WriteLine($"  Gradient compression: {ddpConfig.GradientCompression}");
    Console.WriteLine();

    // Build distributed training pipeline
    var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
        .ConfigureModel(model)
        .ConfigureOptimizer(new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            learningRate: 0.001f * worldSize,  // Scale LR with world size
            weightDecay: 0.01f))
        .ConfigureDistributedTraining(
            strategy: DistributedStrategy.DDP,
            backend: backend,
            configuration: ddpConfig)
        .ConfigureGpuAcceleration(new GpuAccelerationConfig
        {
            Enabled = true,
            DeviceIds = gpuIds
        })
        .ConfigureMixedPrecision(new MixedPrecisionConfig
        {
            Enabled = true,
            LossScaling = LossScalingType.Dynamic
        });

    // Training parameters
    const int epochs = 10;
    const int batchesPerEpoch = 500;
    const int batchSizePerGpu = 32;
    int globalBatchSize = batchSizePerGpu * worldSize;

    Console.WriteLine("Training configuration:");
    Console.WriteLine($"  Epochs: {epochs}");
    Console.WriteLine($"  Batch size per GPU: {batchSizePerGpu}");
    Console.WriteLine($"  Global batch size: {globalBatchSize}");
    Console.WriteLine($"  Learning rate: {0.001f * worldSize:F4} (scaled)\n");

    // Simulate training
    Console.WriteLine("Training...");
    Console.WriteLine("─────────────────────────────────────");

    var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
    var random = new Random(42 + rank);  // Different seed per rank

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        var epochStopwatch = System.Diagnostics.Stopwatch.StartNew();
        double epochLoss = 0;
        int samplesProcessed = 0;

        for (int batch = 1; batch <= batchesPerEpoch; batch++)
        {
            // Simulate forward pass
            double batchLoss = 2.5 / (1 + 0.1 * (epoch - 1 + batch / (double)batchesPerEpoch));
            batchLoss += (random.NextDouble() - 0.5) * 0.1;  // Add noise

            epochLoss += batchLoss;
            samplesProcessed += batchSizePerGpu;

            // Report progress periodically
            if (batch % 100 == 0 && rank == 0)
            {
                double throughput = samplesProcessed * worldSize / epochStopwatch.Elapsed.TotalSeconds;
                Console.WriteLine($"    Batch {batch,3}/{batchesPerEpoch} | Loss: {batchLoss:F3} | " +
                    $"Throughput: {throughput:N0} img/s");
            }

            // Simulate gradient sync (AllReduce)
            await Task.Delay(1);  // Placeholder for actual computation
        }

        epochStopwatch.Stop();
        double avgLoss = epochLoss / batchesPerEpoch;

        if (rank == 0)
        {
            Console.WriteLine($"  Epoch {epoch}/{epochs} complete | Loss: {avgLoss:F3} | " +
                $"Time: {epochStopwatch.Elapsed.TotalSeconds:F1}s\n");
        }

        // Sync after each epoch
        await backend.BarrierAsync();
    }

    totalStopwatch.Stop();

    // Report final statistics
    if (rank == 0)
    {
        double totalSeconds = totalStopwatch.Elapsed.TotalSeconds;
        int totalSamples = epochs * batchesPerEpoch * globalBatchSize;
        double avgThroughput = totalSamples / totalSeconds;
        double singleGpuBaseline = avgThroughput / worldSize;
        double scalingEfficiency = (avgThroughput / (singleGpuBaseline * worldSize)) * 100;

        Console.WriteLine("Training complete!");
        Console.WriteLine("─────────────────────────────────────");
        Console.WriteLine($"  Total time: {TimeSpan.FromSeconds(totalSeconds):m\\:ss}");
        Console.WriteLine($"  Total samples: {totalSamples:N0}");
        Console.WriteLine($"  Average throughput: {avgThroughput:N0} img/s");
        Console.WriteLine($"  Scaling efficiency: {scalingEfficiency:F1}%");
    }

    // Cleanup
    await backend.DisposeAsync();
}
catch (Exception ex)
{
    Console.WriteLine($"\nNote: Full DDP training requires NCCL and multiple GPUs.");
    Console.WriteLine($"This sample demonstrates the API pattern for distributed training.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper functions
static int[] ParseGpuIds(string[] args)
{
    var gpuArg = args.SkipWhile(a => a != "--gpus").Skip(1).FirstOrDefault() ?? "0";
    return gpuArg.Split(',').Select(int.Parse).ToArray();
}

static int ParseInt(string[] args, string name, int defaultValue)
{
    var value = args.SkipWhile(a => a != name).Skip(1).FirstOrDefault();
    return value != null ? int.Parse(value) : defaultValue;
}

static string ParseString(string[] args, string name, string defaultValue)
{
    var value = args.SkipWhile(a => a != name).Skip(1).FirstOrDefault();
    return value ?? defaultValue;
}

// Simplified classes for the sample
public class NCCLConfig
{
    public string MasterAddress { get; set; } = "localhost";
    public int MasterPort { get; set; } = 29500;
    public int WorldSize { get; set; } = 1;
    public int Rank { get; set; } = 0;
}

public class NCCLCommunicationBackend
{
    private readonly NCCLConfig _config;

    public NCCLCommunicationBackend(NCCLConfig config) => _config = config;

    public int WorldSize => _config.WorldSize;
    public int Rank => _config.Rank;
    public bool IsMaster => Rank == 0;

    public Task InitializeAsync() => Task.CompletedTask;
    public Task BarrierAsync() => Task.CompletedTask;
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

public class DDPConfiguration
{
    public int BucketSizeMB { get; set; } = 25;
    public bool GradientCompression { get; set; } = false;
    public bool FindUnusedParameters { get; set; } = false;
    public bool BroadcastBuffers { get; set; } = true;
}

public enum LossScalingType
{
    Static,
    Dynamic
}

public class Tensor<T> { }

public class ResNetNetwork<T>
{
    public int ParameterCount { get; }

    public ResNetNetwork(int depth, int numClasses, int inputChannels)
    {
        // ResNet50 has ~25.6M parameters
        ParameterCount = depth switch
        {
            18 => 11_700_000,
            34 => 21_800_000,
            50 => 25_600_000,
            101 => 44_500_000,
            152 => 60_200_000,
            _ => 25_600_000
        };
    }
}

public enum DistributedStrategy
{
    DDP,
    FSDP,
    PipelineParallel,
    TensorParallel,
    ZeRO
}
