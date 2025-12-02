using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements 3D Parallelism optimizer - coordinates across data, tensor, and pipeline dimensions.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// 3D Parallelism optimizer coordinates optimization across all three parallelism dimensions:
/// - Data parallel: synchronizes gradients across data-parallel replicas
/// - Tensor parallel: synchronizes within tensor-parallel groups
/// - Pipeline parallel: handles gradient accumulation across micro-batches
///
/// This requires managing separate communication groups for each dimension and ensuring
/// proper synchronization order to maintain correctness and efficiency.
/// </para>
/// <para><b>For Beginners:</b>
/// This is the most complex optimizer, coordinating all three types of parallelism.
/// It needs to handle:
/// 1. Averaging gradients across data-parallel replicas (GPUs processing different batches)
/// 2. Synchronizing tensor-parallel groups (GPUs sharing layer computations)
/// 3. Accumulating gradients from pipeline micro-batches
///
/// Think of it like coordinating a massive team split into departments (pipeline stages),
/// work groups (tensor parallel), and shifts (data parallel) - all need to sync at the right times.
/// </para>
/// <para><b>Use Cases:</b>
/// - Frontier-scale models (100B+ parameters)
/// - 100s to 1000s of GPUs
/// - Works with HybridShardedModel
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent - exploits all dimensions
/// - Communication: Complex - multiple sync patterns
/// - Complexity: Very High - most complex optimizer
/// - Best for: Largest scale training
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class HybridShardedOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    private readonly int _pipelineParallelSize;
    private readonly int _tensorParallelSize;
    private readonly int _dataParallelSize;
    private readonly int _pipelineRank;
    private readonly int _tensorRank;
    private readonly int _dataRank;
    private readonly List<int> _tensorParallelGroup;
    private readonly List<int> _dataParallelGroup;

    public HybridShardedOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config,
        int pipelineParallelSize = 1,
        int tensorParallelSize = 1,
        int dataParallelSize = -1)
        : base(wrappedOptimizer, config)
    {
        _pipelineParallelSize = pipelineParallelSize;
        _tensorParallelSize = tensorParallelSize;

        if (dataParallelSize == -1)
        {
            _dataParallelSize = WorldSize / (pipelineParallelSize * tensorParallelSize);
        }
        else
        {
            _dataParallelSize = dataParallelSize;
        }

        if (_pipelineParallelSize * _tensorParallelSize * _dataParallelSize != WorldSize)
        {
            throw new ArgumentException(
                $"Pipeline ({_pipelineParallelSize}) × Tensor ({_tensorParallelSize}) × " +
                $"Data ({_dataParallelSize}) must equal WorldSize ({WorldSize})");
        }

        // Calculate this rank's position in the 3D grid
        // Layout: rank = pipeline_rank * (tensor_size * data_size) + tensor_rank * data_size + data_rank
        int tensorDataPlane = _tensorParallelSize * _dataParallelSize;
        _pipelineRank = Rank / tensorDataPlane;
        int remainder = Rank % tensorDataPlane;
        _tensorRank = remainder / _dataParallelSize;
        _dataRank = remainder % _dataParallelSize;

        // Build tensor-parallel group: all ranks with same pipeline and data rank
        _tensorParallelGroup = new List<int>();
        for (int tp = 0; tp < _tensorParallelSize; tp++)
        {
            int groupRank = _pipelineRank * tensorDataPlane + tp * _dataParallelSize + _dataRank;
            _tensorParallelGroup.Add(groupRank);
        }

        // Build data-parallel group: all ranks with same pipeline and tensor rank
        _dataParallelGroup = new List<int>();
        for (int dp = 0; dp < _dataParallelSize; dp++)
        {
            int groupRank = _pipelineRank * tensorDataPlane + _tensorRank * _dataParallelSize + dp;
            _dataParallelGroup.Add(groupRank);
        }
    }

    /// <summary>
    /// Performs AllReduce within a subgroup of ranks.
    /// </summary>
    /// <remarks>
    /// This implements subgroup AllReduce using one of two strategies:
    /// 1. Point-to-point (Send/Receive): Efficient but requires P2P support (MPI, Gloo TCP)
    /// 2. Global collective fallback: Works with any backend (NCCL, Gloo) but less efficient
    ///
    /// The fallback uses global AllReduce with masking: ranks outside the subgroup contribute zeros,
    /// then all ranks receive the result but only subgroup members use it.
    /// </remarks>
    private void SubgroupAllReduce(Vector<T> data, List<int> groupRanks, ReductionOperation operation)
    {
        if (groupRanks.Count == 1)
        {
            // Single rank in group - no communication needed
            return;
        }

        // Check if this rank is in the subgroup
        bool inGroup = groupRanks.Contains(Rank);

        // Try point-to-point approach first (efficient)
        try
        {
            SubgroupAllReduceP2P(data, groupRanks, operation, inGroup);
        }
        catch (NotSupportedException)
        {
            // Fallback to global collective approach (works with any backend)
            SubgroupAllReduceGlobal(data, groupRanks, operation, inGroup);
        }
    }

    /// <summary>
    /// Implements subgroup AllReduce using point-to-point Send/Receive (efficient).
    /// </summary>
    private void SubgroupAllReduceP2P(Vector<T> data, List<int> groupRanks, ReductionOperation operation, bool inGroup)
    {
        if (!inGroup)
        {
            // This rank is not in the subgroup - no participation needed
            return;
        }

        int groupRoot = groupRanks[0];
        bool isGroupRoot = (Rank == groupRoot);

        if (isGroupRoot)
        {
            // Root collects from all other ranks in group
            var allData = new List<Vector<T>> { data.Clone() };

            for (int i = 1; i < groupRanks.Count; i++)
            {
                int sourceRank = groupRanks[i];
                var receivedData = Config.CommunicationBackend.Receive(sourceRank, data.Length, tag: 0);
                allData.Add(receivedData);
            }

            // Perform reduction on collected data
            var result = PerformReduction(allData, operation);

            // Copy result back to data
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = result[i];
            }

            // Send result back to all other ranks in group
            for (int i = 1; i < groupRanks.Count; i++)
            {
                int destRank = groupRanks[i];
                Config.CommunicationBackend.Send(data, destRank, tag: 0);
            }
        }
        else
        {
            // Non-root sends data to root
            Config.CommunicationBackend.Send(data, groupRoot, tag: 0);

            // Receive result from root
            var result = Config.CommunicationBackend.Receive(groupRoot, data.Length, tag: 0);

            // Copy result back to data
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = result[i];
            }
        }
    }

    /// <summary>
    /// Implements subgroup AllReduce using global collective operations (fallback for NCCL, Gloo).
    /// </summary>
    /// <remarks>
    /// This fallback works by:
    /// 1. Ranks outside subgroup zero their data
    /// 2. Global AllReduce across all ranks (Sum operation)
    /// 3. For Average operation, divide by subgroup size (not world size)
    /// 4. All ranks receive result, but only subgroup members use it
    ///
    /// This is less efficient than P2P (communicates across all ranks) but works with any backend.
    /// </remarks>
    private void SubgroupAllReduceGlobal(Vector<T> data, List<int> groupRanks, ReductionOperation operation, bool inGroup)
    {
        var workingData = data.Clone();

        // Ranks outside the subgroup contribute zeros
        if (!inGroup)
        {
            for (int i = 0; i < workingData.Length; i++)
            {
                workingData[i] = NumOps.FromDouble(0);
            }
        }

        // Perform global AllReduce with Sum operation
        // (we'll handle Average separately by dividing by subgroup size, not world size)
        var globalOp = (operation == ReductionOperation.Average) ? ReductionOperation.Sum : operation;
        Config.CommunicationBackend.AllReduce(workingData, globalOp);

        // For Average, divide by subgroup size (not world size)
        if (operation == ReductionOperation.Average)
        {
            var groupSize = NumOps.FromDouble(groupRanks.Count);
            for (int i = 0; i < workingData.Length; i++)
            {
                workingData[i] = NumOps.Divide(workingData[i], groupSize);
            }
        }

        // Copy result back to data (all ranks get the result, but only subgroup members will use it)
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = workingData[i];
        }
    }

    /// <summary>
    /// Performs the actual reduction operation on a list of vectors.
    /// </summary>
    private Vector<T> PerformReduction(List<Vector<T>> vectors, ReductionOperation operation)
    {
        if (vectors.Count == 0)
            throw new ArgumentException("Cannot reduce empty vector list");

        int length = vectors[0].Length;
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            T value = vectors[0][i];

            for (int j = 1; j < vectors.Count; j++)
            {
                switch (operation)
                {
                    case ReductionOperation.Sum:
                        value = NumOps.Add(value, vectors[j][i]);
                        break;
                    case ReductionOperation.Average:
                        value = NumOps.Add(value, vectors[j][i]);
                        break;
                    case ReductionOperation.Max:
                        if (NumOps.GreaterThan(vectors[j][i], value))
                            value = vectors[j][i];
                        break;
                    case ReductionOperation.Min:
                        if (NumOps.LessThan(vectors[j][i], value))
                            value = vectors[j][i];
                        break;
                    case ReductionOperation.Product:
                        value = NumOps.Multiply(value, vectors[j][i]);
                        break;
                }
            }

            // For Average, divide by count
            if (operation == ReductionOperation.Average)
            {
                var count = NumOps.FromDouble(vectors.Count);
                value = NumOps.Divide(value, count);
            }

            result[i] = value;
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // CRITICAL: Opening barrier must execute BEFORE any divergent logic to synchronize all workers.
        // This prevents deadlock if some workers throw exceptions while others continue.
        Config.CommunicationBackend.Barrier();

        try
        {
            // Null check happens AFTER opening barrier but INSIDE try block.
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));

            // Check if wrapped optimizer supports gradient operations
            var gradientOptimizer = WrappedOptimizer as IGradientBasedOptimizer<T, TInput, TOutput>;
            if (Config.AutoSyncGradients && gradientOptimizer == null)
            {
                throw new InvalidOperationException(
                    "HybridShardedOptimizer with AutoSyncGradients=true requires a gradient-based optimizer. " +
                    $"Received {WrappedOptimizer.GetType().Name} which does not implement IGradientBasedOptimizer.");
            }

            // Populate InitialSolution from wrapped optimizer's model if not already set
            if (inputData.InitialSolution == null && WrappedOptimizer is OptimizerBase<T, TInput, TOutput> baseOptimizer && baseOptimizer.Model != null)
            {
                inputData.InitialSolution = baseOptimizer.Model.Clone();
            }

            // CRITICAL: Save parameters BEFORE local optimization
            // This allows us to discard the local update and apply only the synchronized gradients
            Vector<T>? savedParameters = null;
            if (Config.AutoSyncGradients && inputData.InitialSolution != null)
            {
                savedParameters = inputData.InitialSolution.GetParameters();
            }

            // Step 1: Optimize locally to compute gradients
            var localResult = WrappedOptimizer.Optimize(inputData);

            // Step 2: Synchronize gradients across 3D parallelism dimensions
            if (Config.AutoSyncGradients && localResult.BestSolution != null && savedParameters != null && gradientOptimizer != null)
            {
                var localGradients = gradientOptimizer.LastComputedGradients;

                if (localGradients != null && localGradients.Length > 0)
                {
                    // 3D parallelism gradient synchronization:
                    // 1. First reduce within tensor-parallel group (sum partial tensor results)
                    // 2. Then reduce across data-parallel replicas (average gradients)
                    // 3. Pipeline stages handle their own gradient accumulation separately

                    // Step 2a: Tensor-parallel reduction (SUM operation)
                    // Tensor-parallel ranks have partial results that need to be summed
                    if (_tensorParallelSize > 1)
                    {
                        SubgroupAllReduce(localGradients, _tensorParallelGroup, ReductionOperation.Sum);
                    }

                    // Step 2b: Data-parallel reduction (AVERAGE operation)
                    // Data-parallel ranks have independent gradients that need to be averaged
                    if (_dataParallelSize > 1)
                    {
                        SubgroupAllReduce(localGradients, _dataParallelGroup, ReductionOperation.Average);
                    }

                    // Apply the synchronized gradients using the safe 3-parameter overload
                    // This explicitly passes savedParameters (pre-update state) to prevent double-stepping
                    // Works for ANY optimizer (SGD, Adam, RMSprop, etc.) because ApplyGradients
                    // handles optimizer-specific state (momentum, variance, etc.)
                    var finalModel = gradientOptimizer.ApplyGradients(savedParameters, localGradients, localResult.BestSolution);
                    localResult.BestSolution = finalModel;
                }
            }

            return localResult;
        }
        finally
        {
            // CRITICAL: Closing barrier ALWAYS executes to prevent deadlock,
            // even if null check, WrappedOptimizer.Optimize, or other operations throw.
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // In 3D parallelism, optimizer state management is complex:
        // - Pipeline stages: independent states (no sync)
        // - Tensor parallel: states partitioned by layer slice (sync within group)
        // - Data parallel: states replicated (no sync needed)

        // Full implementation would use process groups for each dimension
        // Framework placeholder
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize hybrid-specific configuration
        writer.Write(_pipelineParallelSize);
        writer.Write(_tensorParallelSize);
        writer.Write(_dataParallelSize);

        // Serialize sharding configuration info
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(Config.AutoSyncGradients);
        writer.Write(Config.MinimumParameterGroupSize);
        writer.Write(Config.EnableGradientCompression);

        // Serialize wrapped optimizer
        var optimizerData = WrappedOptimizer.Serialize();
        writer.Write(optimizerData.Length);
        writer.Write(optimizerData);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read hybrid-specific configuration
        int savedPipelineParallelSize = reader.ReadInt32();
        int savedTensorParallelSize = reader.ReadInt32();
        int savedDataParallelSize = reader.ReadInt32();

        // Read sharding configuration (for validation)
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        reader.ReadBoolean(); // AutoSyncGradients
        reader.ReadInt32(); // MinimumParameterGroupSize
        reader.ReadBoolean(); // EnableGradientCompression

        if (savedWorldSize != WorldSize)
        {
            throw new InvalidOperationException(
                $"World size mismatch. Optimizer was saved with {savedWorldSize} processes, " +
                $"but current configuration has {WorldSize} processes.");
        }

        if (savedRank != Rank)
        {
            throw new InvalidOperationException(
                $"Rank mismatch. Optimizer was saved on rank {savedRank}, " +
                $"but is being loaded on rank {Rank}. This could indicate a configuration error.");
        }

        if (savedPipelineParallelSize != _pipelineParallelSize ||
            savedTensorParallelSize != _tensorParallelSize ||
            savedDataParallelSize != _dataParallelSize)
        {
            throw new InvalidOperationException(
                $"Hybrid parallelism configuration mismatch. Optimizer was saved with " +
                $"pipeline={savedPipelineParallelSize}, tensor={savedTensorParallelSize}, data={savedDataParallelSize}, " +
                $"but current configuration has pipeline={_pipelineParallelSize}, " +
                $"tensor={_tensorParallelSize}, data={_dataParallelSize}.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }
}
