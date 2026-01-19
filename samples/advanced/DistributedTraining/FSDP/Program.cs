using AiDotNet;
using AiDotNet.DistributedTraining;
using AiDotNet.DistributedTraining.FSDP;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Transformers;
using AiDotNet.Optimizers;
using AiDotNet.LossFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Configuration;

Console.WriteLine("=============================================");
Console.WriteLine("  FSDP - Fully Sharded Data Parallel Demo   ");
Console.WriteLine("=============================================\n");

// ============================================
// Step 1: Initialize distributed environment
// ============================================
Console.WriteLine("Step 1: Initializing distributed environment...");

var distConfig = new DistributedConfig
{
    // Auto-detect from environment variables
    MasterAddress = Environment.GetEnvironmentVariable("MASTER_ADDR") ?? "localhost",
    MasterPort = int.TryParse(Environment.GetEnvironmentVariable("MASTER_PORT"), out var port) ? port : 29500,
    WorldSize = int.TryParse(Environment.GetEnvironmentVariable("WORLD_SIZE"), out var world) ? world : 1,
    Rank = int.TryParse(Environment.GetEnvironmentVariable("RANK"), out var rank) ? rank : 0,
    LocalRank = int.TryParse(Environment.GetEnvironmentVariable("LOCAL_RANK"), out var local) ? local : 0,

    // Backend configuration
    Backend = DistributedBackend.NCCL, // NCCL for GPU, Gloo for CPU

    // Timeout settings
    InitTimeout = TimeSpan.FromMinutes(5),
    OperationTimeout = TimeSpan.FromMinutes(30)
};

// Initialize process group
using var distContext = DistributedContext.Initialize(distConfig);

Console.WriteLine($"  Master: {distConfig.MasterAddress}:{distConfig.MasterPort}");
Console.WriteLine($"  World size: {distContext.WorldSize}");
Console.WriteLine($"  Rank: {distContext.Rank}");
Console.WriteLine($"  Local rank: {distContext.LocalRank}");
Console.WriteLine($"  Device: GPU {distContext.LocalRank}");

// ============================================
// Step 2: Configure FSDP
// ============================================
Console.WriteLine("\nStep 2: Configuring FSDP...");

var fsdpConfig = new FSDPConfig<float>
{
    // Sharding strategy
    ShardingStrategy = ShardingStrategy.FullShard, // Maximum memory savings

    // CPU offloading (optional, for even larger models)
    CpuOffload = new CpuOffloadConfig
    {
        Enabled = false, // Enable for models larger than GPU memory
        OffloadParameters = false,
        OffloadGradients = false
    },

    // Mixed precision training
    MixedPrecision = new FSDPMixedPrecisionConfig
    {
        Enabled = true,
        ParameterDtype = DataType.Float32,    // Keep params in FP32
        ReduceDtype = DataType.Float32,        // Reduce in FP32
        BufferDtype = DataType.BFloat16        // Buffers in BF16
    },

    // Activation checkpointing for memory efficiency
    ActivationCheckpointing = new ActivationCheckpointingConfig
    {
        Enabled = true,
        CheckpointInterval = 2 // Checkpoint every 2 layers
    },

    // Backward prefetching for better performance
    BackwardPrefetch = BackwardPrefetch.BackwardPost,

    // Forward prefetching
    ForwardPrefetch = true,

    // Limit all-gather operations
    LimitAllGathers = true,

    // Use original parameters in forward (for certain optimizations)
    UseOrigParamsInForward = false
};

Console.WriteLine($"  Sharding strategy: {fsdpConfig.ShardingStrategy}");
Console.WriteLine($"  Mixed precision: {fsdpConfig.MixedPrecision.Enabled}");
Console.WriteLine($"  Activation checkpointing: {fsdpConfig.ActivationCheckpointing.Enabled}");
Console.WriteLine($"  CPU offload: {fsdpConfig.CpuOffload.Enabled}");

// ============================================
// Step 3: Build model
// ============================================
Console.WriteLine("\nStep 3: Building large Transformer model...");

// Configuration for a large model (similar to GPT-2 Medium)
var modelConfig = new TransformerConfig<float>
{
    VocabSize = 50257,
    MaxSequenceLength = 1024,
    EmbeddingDim = 1024,
    NumHeads = 16,
    NumLayers = 24,
    FeedForwardDim = 4096,
    DropoutRate = 0.1f,
    UseRelativePositionalEncoding = true
};

var model = new TransformerLM<float>(modelConfig);
long totalParams = model.GetParameterCount();
float paramMemoryGB = totalParams * 4 / (1024f * 1024f * 1024f);

Console.WriteLine($"  Model: Transformer LM");
Console.WriteLine($"  Layers: {modelConfig.NumLayers}");
Console.WriteLine($"  Hidden size: {modelConfig.EmbeddingDim}");
Console.WriteLine($"  Attention heads: {modelConfig.NumHeads}");
Console.WriteLine($"  Total parameters: {totalParams:N0}");
Console.WriteLine($"  Parameter memory (FP32): {paramMemoryGB:F2} GB");

// ============================================
// Step 4: Wrap model with FSDP
// ============================================
Console.WriteLine("\nStep 4: Wrapping model with FSDP...");

// Define which modules to wrap with FSDP
var wrapPolicy = new FSDPWrapPolicy<float>
{
    // Wrap transformer blocks individually
    WrapModuleTypes = [typeof(TransformerBlock<float>)],

    // Minimum parameter threshold for wrapping
    MinNumParams = 1_000_000,

    // Custom wrapping decisions
    CustomWrapPredicate = (module, name) =>
    {
        // Always wrap embedding layers
        if (name.Contains("embedding")) return true;
        // Always wrap output layer
        if (name.Contains("output") || name.Contains("lm_head")) return true;
        return false;
    }
};

var fsdpModel = FSDP<float>.Wrap(model, fsdpConfig, wrapPolicy);

// Calculate memory savings
float shardedMemoryGB = paramMemoryGB / distContext.WorldSize;
Console.WriteLine($"  FSDP wrapping complete!");
Console.WriteLine($"  Original memory per GPU: {paramMemoryGB:F2} GB");
Console.WriteLine($"  Sharded memory per GPU: {shardedMemoryGB:F2} GB");
Console.WriteLine($"  Memory reduction: {(1 - shardedMemoryGB / paramMemoryGB) * 100:F1}%");

// ============================================
// Step 5: Configure optimizer with FSDP
// ============================================
Console.WriteLine("\nStep 5: Configuring optimizer...");

// Use AdamW optimizer (parameters are already sharded)
var optimizerConfig = new AdamWOptimizerConfig<float>
{
    LearningRate = 1e-4f,
    Beta1 = 0.9f,
    Beta2 = 0.999f,
    WeightDecay = 0.01f,
    Epsilon = 1e-8f
};

var optimizer = fsdpModel.CreateOptimizer(optimizerConfig);

// Learning rate scheduler
var scheduler = new CosineAnnealingScheduler<float>(
    optimizer,
    totalSteps: 10000,
    warmupSteps: 500,
    minLearningRate: 1e-6f
);

Console.WriteLine($"  Optimizer: AdamW");
Console.WriteLine($"  Learning rate: {optimizerConfig.LearningRate}");
Console.WriteLine($"  Weight decay: {optimizerConfig.WeightDecay}");
Console.WriteLine($"  Scheduler: Cosine annealing with warmup");

// ============================================
// Step 6: Generate synthetic training data
// ============================================
Console.WriteLine("\nStep 6: Generating synthetic training data...");

const int totalSamples = 10000;
const int seqLength = 256;
const int batchSize = 8; // Per-GPU batch size
const int gradientAccumulationSteps = 4;
const int effectiveBatchSize = batchSize * gradientAccumulationSteps * distContext.WorldSize;

var random = new Random(42 + distContext.Rank); // Different seed per rank

// Generate random token sequences
var trainData = new List<int[]>();
for (int i = 0; i < totalSamples / distContext.WorldSize; i++)
{
    var sequence = new int[seqLength];
    for (int j = 0; j < seqLength; j++)
    {
        sequence[j] = random.Next(modelConfig.VocabSize);
    }
    trainData.Add(sequence);
}

Console.WriteLine($"  Total samples: {totalSamples}");
Console.WriteLine($"  Samples per GPU: {trainData.Count}");
Console.WriteLine($"  Sequence length: {seqLength}");
Console.WriteLine($"  Per-GPU batch size: {batchSize}");
Console.WriteLine($"  Gradient accumulation steps: {gradientAccumulationSteps}");
Console.WriteLine($"  Effective batch size: {effectiveBatchSize}");

// ============================================
// Step 7: Training loop
// ============================================
Console.WriteLine("\nStep 7: Starting FSDP training...\n");

var lossFunction = new CrossEntropyLoss<float>();
const int numEpochs = 3;
const int logInterval = 10;

var trainingLosses = new List<float>();
int globalStep = 0;

for (int epoch = 0; epoch < numEpochs; epoch++)
{
    fsdpModel.Train();
    float epochLoss = 0;
    int numBatches = 0;

    // Shuffle data
    var shuffledData = trainData.OrderBy(_ => random.Next()).ToList();

    // Create data sampler (ensures different data on each GPU)
    var sampler = new DistributedSampler<int[]>(
        shuffledData,
        numReplicas: distContext.WorldSize,
        rank: distContext.Rank,
        shuffle: true,
        seed: epoch
    );

    optimizer.ZeroGrad();

    int accumSteps = 0;
    for (int batchStart = 0; batchStart < sampler.Count; batchStart += batchSize)
    {
        int actualBatchSize = Math.Min(batchSize, sampler.Count - batchStart);

        // Prepare batch
        var inputIds = new Tensor<int>([actualBatchSize, seqLength]);
        var targetIds = new Tensor<int>([actualBatchSize, seqLength]);

        for (int i = 0; i < actualBatchSize; i++)
        {
            var sequence = sampler[batchStart + i];
            for (int j = 0; j < seqLength - 1; j++)
            {
                inputIds[[i, j]] = sequence[j];
                targetIds[[i, j]] = sequence[j + 1];
            }
            inputIds[[i, seqLength - 1]] = sequence[seqLength - 1];
            targetIds[[i, seqLength - 1]] = sequence[0]; // Wrap around
        }

        // Forward pass (parameters are gathered automatically)
        var logits = fsdpModel.Forward(inputIds);

        // Compute loss
        float loss = lossFunction.Compute(logits, targetIds);
        loss /= gradientAccumulationSteps; // Scale for accumulation

        epochLoss += loss * gradientAccumulationSteps;
        numBatches++;

        // Backward pass (gradients are sharded and synchronized)
        fsdpModel.Backward(loss);

        accumSteps++;

        // Update weights after accumulation steps
        if (accumSteps >= gradientAccumulationSteps)
        {
            // Gradient clipping (applied to sharded gradients)
            fsdpModel.ClipGradNorm(1.0f);

            // Optimizer step
            optimizer.Step();
            scheduler.Step();
            optimizer.ZeroGrad();

            globalStep++;
            accumSteps = 0;

            // Log progress (only on rank 0)
            if (distContext.Rank == 0 && globalStep % logInterval == 0)
            {
                float avgLoss = epochLoss / numBatches;
                float lr = scheduler.GetLastLearningRate();
                Console.WriteLine($"  Epoch {epoch + 1}, Step {globalStep}: Loss = {avgLoss:F4}, LR = {lr:E2}");
            }
        }
    }

    // Synchronize metrics across all GPUs
    float globalAvgLoss = distContext.AllReduce(epochLoss / numBatches, ReduceOp.Mean);
    trainingLosses.Add(globalAvgLoss);

    if (distContext.Rank == 0)
    {
        Console.WriteLine($"\n  Epoch {epoch + 1} complete: Avg Loss = {globalAvgLoss:F4}");
    }

    // Save checkpoint (only on rank 0, FSDP handles gathering)
    if (distContext.Rank == 0 && (epoch + 1) % 1 == 0)
    {
        Console.WriteLine($"  Saving checkpoint...");
        fsdpModel.SaveCheckpoint($"fsdp_checkpoint_epoch_{epoch + 1}.pt");
    }

    // Barrier to ensure all ranks are synchronized
    distContext.Barrier();
}

// ============================================
// Step 8: Evaluation
// ============================================
if (distContext.Rank == 0)
{
    Console.WriteLine("\nStep 8: Evaluation...");
}

fsdpModel.Eval();

// Generate sample text
if (distContext.Rank == 0)
{
    Console.WriteLine("  Generating sample text...");

    var prompt = new int[] { 50256 }; // Start token
    var generated = fsdpModel.Generate(
        prompt,
        maxLength: 50,
        temperature: 0.8f,
        topK: 50,
        topP: 0.9f
    );

    Console.WriteLine($"  Generated {generated.Length} tokens");
}

// ============================================
// Step 9: Memory profiling
// ============================================
if (distContext.Rank == 0)
{
    Console.WriteLine("\nStep 9: Memory profiling...");
}

var memStats = fsdpModel.GetMemoryStats();

if (distContext.Rank == 0)
{
    Console.WriteLine($"  Peak memory allocated: {memStats.PeakAllocatedGB:F2} GB");
    Console.WriteLine($"  Current memory allocated: {memStats.CurrentAllocatedGB:F2} GB");
    Console.WriteLine($"  Memory reserved: {memStats.ReservedGB:F2} GB");
}

// Collect stats from all ranks
var allPeakMemory = distContext.AllGather(memStats.PeakAllocatedGB);

if (distContext.Rank == 0)
{
    Console.WriteLine($"\n  Memory usage per GPU:");
    for (int i = 0; i < allPeakMemory.Length; i++)
    {
        Console.WriteLine($"    GPU {i}: {allPeakMemory[i]:F2} GB");
    }
    Console.WriteLine($"  Total peak memory: {allPeakMemory.Sum():F2} GB");
}

// ============================================
// Summary
// ============================================
distContext.Barrier();

if (distContext.Rank == 0)
{
    Console.WriteLine("\n=============================================");
    Console.WriteLine("  FSDP Training Complete!");
    Console.WriteLine("=============================================");
    Console.WriteLine($"\n  Configuration:");
    Console.WriteLine($"    - World size: {distContext.WorldSize} GPUs");
    Console.WriteLine($"    - Sharding strategy: {fsdpConfig.ShardingStrategy}");
    Console.WriteLine($"    - Model parameters: {totalParams:N0}");
    Console.WriteLine($"\n  Memory efficiency:");
    Console.WriteLine($"    - Without FSDP: {paramMemoryGB:F2} GB/GPU");
    Console.WriteLine($"    - With FSDP: {shardedMemoryGB:F2} GB/GPU");
    Console.WriteLine($"    - Savings: {(1 - shardedMemoryGB / paramMemoryGB) * 100:F1}%");
    Console.WriteLine($"\n  Training:");
    Console.WriteLine($"    - Epochs: {numEpochs}");
    Console.WriteLine($"    - Final loss: {trainingLosses.Last():F4}");
    Console.WriteLine($"\n  FSDP enables training models {distContext.WorldSize}x larger!");
}

// Cleanup
distContext.Barrier();
fsdpModel.Dispose();
