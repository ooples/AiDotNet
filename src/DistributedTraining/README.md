# AiDotNet Distributed Training (FSDP-Inspired)

## Overview

This module implements a **Fully Sharded Data Parallelism (FSDP)** framework for AiDotNet, enabling training of models that are too large to fit on a single GPU. Our implementation is inspired by PyTorch's FSDP but designed with a simpler, more intuitive API specifically for .NET.

### For Beginners: What is Distributed Training?

Imagine you have a giant puzzle that's too big for one table. Distributed training is like having multiple tables (GPUs/machines) working together:

- **Normal Training**: One person works on the entire puzzle alone
- **Distributed Training**: Multiple people each work on part of the puzzle, sharing pieces when needed and coordinating their work

In machine learning terms:
- Instead of loading all model parameters on one GPU (which might run out of memory)
- We split parameters across multiple GPUs
- Each GPU holds only part of the model, but they work together to train it

## Key Features

✅ **Fully Sharded Data Parallelism**: Parameters are split across processes
✅ **Simple `.AsDistributed()` API**: One-line conversion to distributed training
✅ **Automatic Gradient Synchronization**: Gradients are averaged across processes automatically
✅ **Smart Parameter Grouping**: Small parameters are grouped to reduce communication overhead
✅ **Multiple Communication Backends**: In-memory for testing, MPI.NET for production
✅ **Type-Safe**: Uses `INumericOperations<T>` for all arithmetic
✅ **Beginner-Friendly**: Extensive documentation and examples

## Quick Start

### 1. Basic Usage

```csharp
using AiDotNet.DistributedTraining;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

// Create your model as usual
var model = new NeuralNetworkModel<double>(...);

// Create a communication backend
// For testing: Use InMemoryCommunicationBackend
var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);
backend.Initialize();

// Make it distributed with ONE LINE!
var distributedModel = model.AsDistributed(backend);

// Train as usual - distributed magic happens automatically!
distributedModel.Train(trainingInputs, trainingOutputs);

// Make predictions
var predictions = distributedModel.Predict(testInputs);

// Cleanup
backend.Shutdown();
```

### 2. Advanced Usage with Custom Configuration

```csharp
// Create custom configuration for low-bandwidth networks
var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 8);
backend.Initialize();

var config = new ShardingConfiguration<double>(backend)
{
    AutoSyncGradients = true,              // Sync after each training step
    MinimumParameterGroupSize = 4096,      // Group small parameters
    EnableGradientCompression = true       // Compress for slow networks
};

// Or use preset configurations
var configForHighBandwidth = ShardingConfiguration<double>.CreateForHighBandwidth(backend);
var configForLowBandwidth = ShardingConfiguration<double>.CreateForLowBandwidth(backend);

// Apply configuration
var distributedModel = model.AsDistributed(config);
```

### 3. Using with AiModelBuilder

```csharp
var result = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model.AsDistributed(backend))
    .ConfigureOptimizer(optimizer.AsDistributed(backend))
    .Build(xTrain, yTrain);

var predictions = result.Predict(xTest);
```

## Architecture

### Components

#### 1. Communication Layer

**ICommunicationBackend<T>**
- Defines the contract for distributed communication
- Supports: `AllReduce`, `AllGather`, `Broadcast`, `Scatter`, `ReduceScatter`, `Barrier`

**CommunicationManager**
- Static manager for global communication operations
- Thread-safe, singleton-style access

**InMemoryCommunicationBackend<T>**
- In-memory implementation for testing and single-machine scenarios
- Simulates multi-process behavior using shared memory
- Perfect for development and testing without MPI

#### 2. Model Sharding

**IShardedModel<T, TInput, TOutput>**
- Extends `IFullModel` with sharding capabilities
- Automatically distributes parameters across processes

**ShardedModel<T, TInput, TOutput>**
- Wraps any `IFullModel` and adds distributed capabilities
- Handles parameter sharding, gathering, and gradient synchronization

#### 3. Optimizer Sharding

**IShardedOptimizer<T, TInput, TOutput>**
- Extends `IOptimizer` with distributed coordination

**ShardedOptimizer<T, TInput, TOutput>**
- Wraps any `IOptimizer` for distributed optimization
- Synchronizes optimizer state across processes

#### 4. Smart Improvements

**ParameterAnalyzer<T>**
- Analyzes models and creates optimized parameter groupings
- Reduces communication overhead by grouping small parameters
- Ensures even distribution across processes

**DistributedExtensions**
- Provides `.AsDistributed()` extension methods
- Simple one-line API for enabling distributed training

## How It Works

### Parameter Sharding

```
Model Parameters: [P0, P1, P2, P3, P4, P5, P6, P7]

With 4 GPUs:
GPU 0: [P0, P1]
GPU 1: [P2, P3]
GPU 2: [P4, P5]
GPU 3: [P6, P7]
```

### Forward Pass (AllGather)

```
Before AllGather:
GPU 0: [P0, P1, ??, ??, ??, ??, ??, ??]
GPU 1: [??, ??, P2, P3, ??, ??, ??, ??]
GPU 2: [??, ??, ??, ??, P4, P5, ??, ??]
GPU 3: [??, ??, ??, ??, ??, ??, P6, P7]

After AllGather (everyone has full parameters):
All GPUs: [P0, P1, P2, P3, P4, P5, P6, P7]
```

### Backward Pass (AllReduce)

```
Each GPU calculates gradients on its data:
GPU 0: [G0_0, G0_1, G0_2, ...]  (gradients from GPU 0's data)
GPU 1: [G1_0, G1_1, G1_2, ...]  (gradients from GPU 1's data)
GPU 2: [G2_0, G2_1, G2_2, ...]  (gradients from GPU 2's data)
GPU 3: [G3_0, G3_1, G3_2, ...]  (gradients from GPU 3's data)

After AllReduce (average gradients):
All GPUs: [Avg(G*_0), Avg(G*_1), Avg(G*_2), ...]
```

## Communication Operations

### AllReduce
Combines values from all processes and distributes result to all.

```csharp
var gradients = new Vector<double>([1.0, 2.0, 3.0]);
CommunicationManager.AllReduce<double>(gradients, ReductionOperation.Average);
// All processes now have averaged gradients
```

### AllGather
Gathers data from all processes and concatenates.

```csharp
var localShard = new Vector<double>([1.0, 2.0]);
var fullParams = CommunicationManager.AllGather<double>(localShard);
// Returns concatenation of all shards
```

### Broadcast
Sends data from one process (root) to all others.

```csharp
var initialParams = new Vector<double>([1.0, 2.0, 3.0]);
var params = CommunicationManager.Broadcast<double>(initialParams, root: 0);
// All processes receive params from root
```

### Barrier
Synchronization point - all processes must reach before any can continue.

```csharp
CommunicationManager.Barrier<double>();
// All processes wait here until everyone arrives
```

## Launcher Scripts

Use the provided scripts to launch distributed training across multiple processes:

### Bash (Linux/macOS)

```bash
cd scripts
./launch-distributed-training.sh 4 ./MyTrainingApp --epochs 100
```

### PowerShell (Windows)

```powershell
cd scripts
.\launch-distributed-training.ps1 -NumProcesses 4 -Program ".\MyTrainingApp.exe" -ProgramArgs "--epochs 100"
```

## Configuration Options

### ShardingConfiguration Properties

| Property | Description | Default | When to Change |
|----------|-------------|---------|----------------|
| `AutoSyncGradients` | Automatically synchronize after training | `true` | Rarely. Keep true for standard training |
| `MinimumParameterGroupSize` | Minimum parameters per communication | `1024` | Increase for slow networks, decrease for fast ones |
| `EnableGradientCompression` | Compress gradients before sending | `false` | Enable on slow networks to reduce bandwidth |

### Preset Configurations

```csharp
// For high-speed GPU interconnects (NVLink, InfiniBand)
var config = ShardingConfiguration<double>.CreateForHighBandwidth(backend);

// For slower networks (Ethernet)
var config = ShardingConfiguration<double>.CreateForLowBandwidth(backend);

// Default (balanced settings)
var config = ShardingConfiguration<double>.CreateDefault(backend);
```

## Performance Considerations

### When to Use Distributed Training

✅ **Good Use Cases:**
- Model too large for single GPU memory
- Very large batch sizes
- Multiple GPUs/machines available
- Training time is bottleneck

❌ **Poor Use Cases:**
- Model fits comfortably on one GPU
- Small models (< 100MB)
- Communication cost exceeds computation savings
- Limited network bandwidth

### Optimization Tips

1. **Batch Size**: Increase batch size to reduce communication frequency
2. **Parameter Grouping**: Larger groups reduce message count but increase latency
3. **Gradient Compression**: Helpful on slow networks, but adds CPU overhead
4. **Network Topology**: Use high-bandwidth interconnects when possible

## Examples

### Example 1: Simple Linear Regression

```csharp
// Create model
var coefficients = new Vector<double>(new double[1000]);  // 1000 parameters
var model = new VectorModel<double>(coefficients);

// Setup distributed training (4 processes)
var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);
backend.Initialize();

var distributedModel = model.AsDistributed(backend);

// Each process gets 250 parameters (1000 / 4)
Console.WriteLine($"Shard size: {distributedModel.LocalParameterShard.Length}");  // 250

// Train
distributedModel.Train(inputs, outputs);

// Cleanup
backend.Shutdown();
```

### Example 2: Neural Network with Custom Optimizer

```csharp
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

// Create neural network
var network = new NeuralNetworkModel<double>(...);
var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(network, options);

// Make both distributed
var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 8);
backend.Initialize();

var distributedNetwork = network.AsDistributed(backend);
var distributedOptimizer = optimizer.AsDistributed(backend);

// Optimize with distributed setup
var inputData = new OptimizationInputData<double, Tensor<double>, Tensor<double>>
{
    Model = distributedNetwork,
    XTrain = xTrain,
    YTrain = yTrain
};

var result = distributedOptimizer.Optimize(inputData);

backend.Shutdown();
```

## Testing

Comprehensive unit tests ensure correctness:

```bash
cd tests
dotnet test --filter "DistributedTraining"
```

### Key Tests

- **Numerical Equivalence**: Distributed training produces same results as single-process
- **Communication Correctness**: AllReduce, AllGather work correctly
- **Parameter Sharding**: Parameters distributed evenly and completely
- **Gradient Synchronization**: Gradients averaged correctly

## Future Enhancements

### Phase 5 (Future Work)

- [ ] MPI.NET backend implementation for true multi-machine training
- [ ] NCCL backend for NVIDIA GPUs
- [ ] Gradient compression algorithms (quantization, sparsification)
- [ ] Zero Redundancy Optimizer (ZeRO) stages
- [ ] Mixed precision training support
- [ ] Checkpoint/resume functionality
- [ ] Performance profiling and monitoring

## Contributing

When contributing to distributed training:

1. **Follow the pattern**: Interface → Base class → Concrete implementations
2. **Use INumericOperations<T>**: Never hardcode numeric types
3. **Add "For Beginners" documentation**: Make it accessible
4. **Write tests**: Especially numerical equivalence tests
5. **Consider communication cost**: Minimize AllReduce/AllGather calls

## FAQ

### Q: Can I use this without MPI?

**A:** Yes! Use `InMemoryCommunicationBackend` for single-machine scenarios. It simulates distributed behavior without requiring MPI.

### Q: How do I know if my model will benefit from distributed training?

**A:** If your model doesn't fit in GPU memory OR training time is a major bottleneck, distributed training can help. For small models that fit easily on one GPU, the communication overhead may not be worth it.

### Q: What's the difference between FSDP and data parallelism?

**A:**
- **Data Parallelism**: Full model copy on each GPU, only data is split
- **FSDP**: Model parameters are split across GPUs, saving memory

FSDP enables training larger models at the cost of more communication.

### Q: Do I need special hardware?

**A:** No! You can start with `InMemoryCommunicationBackend` for testing. For production, any GPUs or machines that can run .NET and communicate over network will work. High-bandwidth interconnects (NVLink, InfiniBand) improve performance.

### Q: How does this compare to PyTorch FSDP?

**A:** We're inspired by PyTorch FSDP but optimized for .NET:
- Simpler API (`.AsDistributed()`)
- Automatic parameter grouping
- Type-safe numeric operations
- Beginner-friendly documentation

## License

This module is part of AiDotNet and follows the same Apache 2.0 license.

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [MPI.NET Project](https://github.com/microsoft/MPI.NET)
- [Distributed Training Concepts](https://huggingface.co/docs/transformers/perf_train_gpu_many)
