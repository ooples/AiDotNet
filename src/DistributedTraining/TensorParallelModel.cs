using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Tensor Parallel model wrapper - splits individual layers across ranks (Megatron-LM style).
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Tensor Parallelism (Megatron-LM style) partitions individual layers horizontally across processes.
/// For example, a large matrix multiplication is split so each GPU computes only a portion of the output,
/// then results are combined. This is particularly effective for transformer models where attention and
/// feed-forward layers can be partitioned along specific dimensions (column-parallel and row-parallel).
/// </para>
/// <para><b>For Beginners:</b>
/// Tensor parallelism is like splitting a single large calculation across multiple workers.
/// Imagine a huge spreadsheet calculation - instead of one person doing all the math, we divide
/// the spreadsheet columns across multiple people, each computing their portion simultaneously.
///
/// For example, in a neural network layer with a 10000x10000 weight matrix:
/// - GPU 0 handles columns 0-2499
/// - GPU 1 handles columns 2500-4999
/// - GPU 2 handles columns 5000-7499
/// - GPU 3 handles columns 7500-9999
///
/// They compute in parallel, then combine results.
/// </para>
/// <para><b>Use Cases:</b>
/// - Very wide models (large hidden dimensions)
/// - Transformer models (BERT, GPT) with large attention/FFN layers
/// - When individual layers are too large for single GPU
/// - Often combined with pipeline parallelism for maximum scalability
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent for wide layers - each rank stores only portion of weights
/// - Communication: High - requires AllReduce or AllGather within each layer
/// - Complexity: Very High - requires model-aware partitioning, specific to layer types
/// - Best for: Transformer models, very wide layers, fast interconnects (NVLink)
/// - Limitation: Requires fast communication (high overhead on slow networks)
/// </para>
/// <para><b>Implementation Note:</b>
/// This is a production-ready framework implementation. Full tensor parallelism requires
/// model-specific layer partitioning (column-parallel vs row-parallel strategy for different
/// layer types). This implementation provides the infrastructure. For production use with
/// specific models (e.g., transformers), extend this class with layer-aware partitioning.
/// </para>
/// <para><b>⚠️ IMPORTANT LIMITATION - Memory Efficiency:</b>
/// This implementation gathers the full parameter vector on every Train() and Predict() call
/// (via GatherFullParameters and SetParameters), which defeats the memory-saving purpose of
/// true tensor parallelism. While parameters are sharded across ranks for storage, they are
/// reconstructed into the full vector for each forward/backward pass. This means:
/// - Memory savings are minimal compared to data-parallel training
/// - Communication overhead is high (AllGather on every forward pass)
/// - This wrapper primarily provides gradient synchronization, not memory-efficient tensor parallelism
///
/// For true memory-efficient tensor parallelism, you would need layer-aware implementations where
/// each rank only loads its parameter shard and performs partial matrix multiplications without
/// ever reconstructing the full parameter vector. This simplified implementation is suitable for:
/// - Testing and development of distributed training infrastructure
/// - Scenarios where gradient synchronization is more important than memory efficiency
/// - Models where memory is not the primary constraint
///
/// If memory efficiency is critical, consider using FSDP (Fully Sharded Data Parallel) or ZeRO-3
/// instead, which shard parameters more aggressively and avoid full parameter reconstruction.
/// </para>
/// <para>
/// Example:
/// <code>
/// var model = new TransformerModel&lt;double&gt;(...); // Large transformer
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// // Each rank handles 1/4 of each layer's width
/// var tensorParallelModel = new TensorParallelModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     model, config);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class TensorParallelModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private int _tensorParallelSize;
    private List<int> _tensorParallelGroup = new();

    /// <summary>
    /// Creates a new Tensor Parallel model.
    /// </summary>
    /// <param name="wrappedModel">The model to partition with tensor parallelism</param>
    /// <param name="config">Configuration for sharding and communication</param>
    public TensorParallelModel(
        IFullModel<T, TInput, TOutput> wrappedModel,
        IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
        // Note: _tensorParallelSize and _tensorParallelGroup are set in OnBeforeInitializeSharding
        // which is called by the base constructor before this body runs.
    }

    /// <summary>
    /// Called before InitializeSharding to set up derived class state.
    /// </summary>
    protected override void OnBeforeInitializeSharding()
    {
        _tensorParallelSize = Config.CommunicationBackend.WorldSize;

        // Build tensor-parallel group (all ranks in this world are in the same TP group)
        _tensorParallelGroup = new List<int>();
        for (int i = 0; i < _tensorParallelSize; i++)
        {
            _tensorParallelGroup.Add(i);
        }
    }

    /// <summary>
    /// Initializes tensor parallelism by partitioning layer weights.
    /// </summary>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        // In tensor parallelism, we partition weights within layers
        // For this framework implementation, we use a simplified column-wise partitioning
        // Production usage would require layer-specific partitioning logic

        int baseShardSize = totalParams / _tensorParallelSize;
        int remainder = totalParams % _tensorParallelSize;

        ShardSize = baseShardSize + (Rank < remainder ? 1 : 0);
        ShardStartIndex = Rank * baseShardSize + Math.Min(Rank, remainder);

        var shardData = new T[ShardSize];
        Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
        LocalShard = new Vector<T>(shardData);

        CachedFullParameters = null;
    }

    /// <summary>
    /// Performs AllReduce within the tensor-parallel subgroup.
    /// </summary>
    private void SubgroupAllReduce(Vector<T> data, ReductionOperation operation)
    {
        if (_tensorParallelGroup.Count == 1)
        {
            // Single rank in group - no communication needed
            return;
        }

        // Check if this rank is in the subgroup
        bool inGroup = _tensorParallelGroup.Contains(Rank);

        // Try point-to-point approach first (efficient)
        try
        {
            SubgroupAllReduceP2P(data, operation, inGroup);
        }
        catch (NotSupportedException)
        {
            // Fallback to global collective approach (works with any backend)
            SubgroupAllReduceGlobal(data, operation, inGroup);
        }
    }

    /// <summary>
    /// Implements subgroup AllReduce using point-to-point Send/Receive (efficient).
    /// </summary>
    private void SubgroupAllReduceP2P(Vector<T> data, ReductionOperation operation, bool inGroup)
    {
        if (!inGroup)
        {
            // This rank is not in the subgroup - no participation needed
            return;
        }

        int groupRoot = _tensorParallelGroup[0];
        bool isGroupRoot = (Rank == groupRoot);

        if (isGroupRoot)
        {
            // Root collects from all other ranks in group
            var allData = new List<Vector<T>> { data.Clone() };

            for (int i = 1; i < _tensorParallelGroup.Count; i++)
            {
                int sourceRank = _tensorParallelGroup[i];
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
            for (int i = 1; i < _tensorParallelGroup.Count; i++)
            {
                int destRank = _tensorParallelGroup[i];
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
    private void SubgroupAllReduceGlobal(Vector<T> data, ReductionOperation operation, bool inGroup)
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
        var globalOp = (operation == ReductionOperation.Average) ? ReductionOperation.Sum : operation;
        Config.CommunicationBackend.AllReduce(workingData, globalOp);

        // For Average, divide by subgroup size (not world size)
        if (operation == ReductionOperation.Average)
        {
            var groupSize = NumOps.FromDouble(_tensorParallelGroup.Count);
            for (int i = 0; i < workingData.Length; i++)
            {
                workingData[i] = NumOps.Divide(workingData[i], groupSize);
            }
        }

        // Copy result back to data
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

    /// <summary>
    /// Synchronizes tensor-parallel computation results.
    /// </summary>
    /// <remarks>
    /// In tensor parallelism, different layers require different synchronization patterns:
    /// - Column-parallel layers: AllReduce after computation
    /// - Row-parallel layers: AllGather before computation
    /// This implementation uses subgroup AllReduce within the tensor-parallel group.
    /// </remarks>
    public override void SynchronizeGradients()
    {
        // Note: Gradient synchronization is handled within Train() by averaging
        // the full gradient vector across ranks. This method is provided for API
        // compatibility but is a no-op in the current implementation.
        //
        // True tensor-parallel semantics would require layer-aware gradient computation
        // where each rank only computes gradients for its parameter shard. The current
        // implementation uses data-parallel gradient averaging, which is correct for
        // distributed training where all ranks train on different data.
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Tensor parallel training:
        // 1. Each rank has a slice of the layer weights
        // 2. Forward: compute partial outputs, then AllReduce or AllGather depending on layer type
        // 3. Backward: similar communication pattern in reverse

        // For this framework implementation, we provide simplified pattern
        // Production usage would implement layer-specific forward/backward logic

        // Gather full parameters before training
        var originalParams = GatherFullParameters();
        WrappedModel.SetParameters(originalParams);

        // Compute true gradients using the model's gradient computation
        // This provides accurate gradients before optimizer updates are applied
        var gradVec = WrappedModel.ComputeGradients(input, expectedOutput);

        // Synchronize gradients across tensor-parallel group
        if (Config.AutoSyncGradients)
        {
            // Average gradients across ranks in tensor-parallel group
            SubgroupAllReduce(gradVec, ReductionOperation.Average);
        }

        // Apply averaged gradients to parameters using the configured learning rate
        // In tensor parallelism, we use a simple SGD-style update: θ = θ - lr * gradients
        // For more sophisticated optimization, wrap this model with a gradient-based optimizer
        WrappedModel.ApplyGradients(gradVec, Config.LearningRate);

        // Get updated parameters after applying gradients
        var updatedParams = WrappedModel.GetParameters();

        // Extract shard from final parameters
        UpdateLocalShardFromFull(updatedParams);
        InvalidateCache();
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        // Tensor parallel inference
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);
        return WrappedModel.Predict(input);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "TensorParallel");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("TensorParallelSize", _tensorParallelSize);
        metadata.SetProperty("PartitioningStyle", "Column-wise (simplified)");
        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new TensorParallelModel<T, TInput, TOutput>(
            WrappedModel.WithParameters(parameters), Config);
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(Config.AutoSyncGradients);
        writer.Write(Config.MinimumParameterGroupSize);
        writer.Write(Config.EnableGradientCompression);
        var modelData = WrappedModel.Serialize();
        writer.Write(modelData.Length);
        writer.Write(modelData);
        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        reader.ReadBoolean();
        reader.ReadInt32();
        reader.ReadBoolean();

        if (savedWorldSize != WorldSize)
            throw new InvalidOperationException($"World size mismatch: {savedWorldSize} vs {WorldSize}");
        if (savedRank != Rank)
            throw new InvalidOperationException($"Rank mismatch: {savedRank} vs {Rank}");

        int modelDataLength = reader.ReadInt32();
        byte[] modelData = reader.ReadBytes(modelDataLength);
        WrappedModel.Deserialize(modelData);
        InitializeSharding();
    }

    /// <inheritdoc/>
    public override void SaveModel(string filePath)
    {
        Config.CommunicationBackend.Barrier();
        try
        {
            if (Rank == 0)
                File.WriteAllBytes(filePath, Serialize());
        }
        finally
        {
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void LoadModel(string filePath)
    {
        Config.CommunicationBackend.Barrier();
        try
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        finally
        {
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> Clone()
    {
        return new TensorParallelModel<T, TInput, TOutput>(WrappedModel.Clone(), Config);
    }
}
