using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NumericOperations;
using AiDotNet.Helpers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements a distributed model wrapper that shards parameters across multiple processes.
///
/// For Beginners:
/// This class wraps any existing model and makes it work across multiple GPUs or machines.
/// It automatically handles:
/// - Splitting parameters across processes (sharding)
/// - Gathering parameters when needed for forward pass
/// - Averaging gradients across all processes during training
///
/// Think of it like a team project where each person holds part of the solution.
/// When you need the full solution, everyone shares their part (AllGather).
/// When everyone learns something new, they share and average their learnings (AllReduce).
///
/// Example:
/// <code>
/// // Original model
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
///
/// // Wrap it for distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var distributedModel = new ShardedModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
///
/// // Now train as usual - distributed magic happens automatically!
/// distributedModel.Train(inputs, outputs);
/// </code>
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ShardedModel<T, TInput, TOutput> : IShardedModel<T, TInput, TOutput> where T : struct
{
    private readonly IFullModel<T, TInput, TOutput> _wrappedModel;
    private readonly IShardingConfiguration<T> _config;
    private readonly INumericOperations<T> _numOps;
    private Vector<T> _localParameterShard;
    private Vector<T>? _cachedFullParameters;
    private int _shardStartIndex;
    private int _shardSize;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> WrappedModel => _wrappedModel;

    /// <inheritdoc/>
    public int Rank => _config.CommunicationBackend.Rank;

    /// <inheritdoc/>
    public int WorldSize => _config.CommunicationBackend.WorldSize;

    /// <inheritdoc/>
    public Vector<T> LocalParameterShard => _localParameterShard;

    /// <inheritdoc/>
    public IShardingConfiguration<T> ShardingConfiguration => _config;

    /// <inheritdoc/>
    public int ParameterCount => _wrappedModel.ParameterCount;

    /// <summary>
    /// Creates a new sharded model wrapping an existing model.
    ///
    /// For Beginners:
    /// This constructor takes your existing model and makes it distributed.
    /// You provide:
    /// 1. The model you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    ///
    /// The constructor automatically:
    /// - Splits the model's parameters across all processes
    /// - Sets up communication channels
    /// - Prepares everything for distributed training
    /// </summary>
    /// <param name="wrappedModel">The model to wrap with distributed capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if model or config is null</exception>
    public ShardedModel(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
    {
        _wrappedModel = wrappedModel ?? throw new ArgumentNullException(nameof(wrappedModel));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize the communication backend if not already done
        if (!_config.CommunicationBackend.IsInitialized)
        {
            _config.CommunicationBackend.Initialize();
        }

        // Shard the model parameters across processes
        InitializeSharding();
    }

    /// <summary>
    /// Initializes parameter sharding by dividing parameters across processes.
    ///
    /// For Beginners:
    /// This method splits the model's parameters into chunks and gives each
    /// process its own chunk to manage. It's like dividing a deck of cards
    /// evenly among players.
    /// </summary>
    private void InitializeSharding()
    {
        var fullParameters = _wrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        // Calculate shard size for this process
        int baseShardSize = totalParams / WorldSize;
        int remainder = totalParams % WorldSize;

        // Distribute remainder among first 'remainder' processes
        _shardSize = baseShardSize + (Rank < remainder ? 1 : 0);
        _shardStartIndex = Rank * baseShardSize + Math.Min(Rank, remainder);

        // Extract local shard
        var shardData = new T[_shardSize];
        Array.Copy(fullParameters.ToArray(), _shardStartIndex, shardData, 0, _shardSize);
        _localParameterShard = new Vector<T>(shardData);

        // Cache invalidated
        _cachedFullParameters = null;
    }

    /// <inheritdoc/>
    public Vector<T> GatherFullParameters()
    {
        // Use cached version if available
        if (_cachedFullParameters != null)
        {
            return _cachedFullParameters;
        }

        // Gather parameters from all processes
        var gathered = _config.CommunicationBackend.AllGather(_localParameterShard);
        _cachedFullParameters = gathered;
        return gathered;
    }

    /// <inheritdoc/>
    public void SynchronizeGradients()
    {
        // Gather full parameters from all shards since shard sizes differ when
        // ParameterCount % WorldSize != 0 (cannot AllReduce different-sized vectors)
        var fullParameters = GatherFullParameters();

        // AllReduce the full parameter vector to average across all ranks
        // This ensures all ranks end up with the same averaged parameters
        _config.CommunicationBackend.AllReduce(fullParameters, ReductionOperation.Average);

        // Update local shard from the reduced parameters
        var shardData = new T[_shardSize];
        Array.Copy(fullParameters.ToArray(), _shardStartIndex, shardData, 0, _shardSize);
        _localParameterShard = new Vector<T>(shardData);

        // Update cache with synchronized parameters
        _cachedFullParameters = fullParameters;

        // Update wrapped model with synchronized parameters
        _wrappedModel.SetParameters(fullParameters);
    }

    /// <inheritdoc/>
    public void Train(TInput input, TOutput expectedOutput)
    {
        // Gather full parameters for training
        var fullParams = GatherFullParameters();
        _wrappedModel.SetParameters(fullParams);

        // Train the wrapped model
        _wrappedModel.Train(input, expectedOutput);

        // Get updated parameters
        var updatedParams = _wrappedModel.GetParameters();

        // Update local shard
        var shardData = new T[_shardSize];
        Array.Copy(updatedParams.ToArray(), _shardStartIndex, shardData, 0, _shardSize);
        _localParameterShard = new Vector<T>(shardData);

        // Synchronize gradients if auto-sync is enabled
        if (_config.AutoSyncGradients)
        {
            SynchronizeGradients();

            // Apply synchronized parameters back to the model
            fullParams = GatherFullParameters();
            _wrappedModel.SetParameters(fullParams);
        }

        // Invalidate cache
        _cachedFullParameters = null;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Gather full parameters for prediction
        var fullParams = GatherFullParameters();
        _wrappedModel.SetParameters(fullParams);

        // Use wrapped model for prediction
        return _wrappedModel.Predict(input);
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        var metadata = _wrappedModel.GetModelMetadata();

        // Add distributed training info
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("ShardSize", _shardSize);

        return metadata;
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        return GatherFullParameters();
    }

    /// <inheritdoc/>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Parameter count mismatch. Expected {ParameterCount}, got {parameters.Length}.");
        }

        // Update local shard
        var shardData = new T[_shardSize];
        Array.Copy(parameters.ToArray(), _shardStartIndex, shardData, 0, _shardSize);
        _localParameterShard = new Vector<T>(shardData);

        // Invalidate cache
        _cachedFullParameters = null;

        // Update wrapped model
        _wrappedModel.SetParameters(parameters);
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = _wrappedModel.WithParameters(parameters);
        return new ShardedModel<T, TInput, TOutput>(newModel, _config);
    }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize sharding configuration info
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(_config.AutoSyncGradients);
        writer.Write(_config.MinimumParameterGroupSize);
        writer.Write(_config.EnableGradientCompression);

        // Serialize wrapped model
        var modelData = _wrappedModel.Serialize();
        writer.Write(modelData.Length);
        writer.Write(modelData);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read sharding configuration (for validation)
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        reader.ReadBoolean(); // AutoSyncGradients
        reader.ReadInt32(); // MinimumParameterGroupSize
        reader.ReadBoolean(); // EnableGradientCompression

        if (savedWorldSize != WorldSize)
        {
            throw new InvalidOperationException(
                $"World size mismatch. Model was trained with {savedWorldSize} processes, " +
                $"but current configuration has {WorldSize} processes.");
        }

        // Read wrapped model
        int modelDataLength = reader.ReadInt32();
        byte[] modelData = reader.ReadBytes(modelDataLength);
        _wrappedModel.Deserialize(modelData);

        // Re-initialize sharding
        InitializeSharding();
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        var data = Serialize();
        File.WriteAllBytes(filePath, data);
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        var data = File.ReadAllBytes(filePath);
        Deserialize(data);
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Clone()
    {
        var clonedWrappedModel = _wrappedModel.Clone();
        return new ShardedModel<T, TInput, TOutput>(clonedWrappedModel, _config);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> GetFeatureImportance()
    {
        return _wrappedModel.GetFeatureImportance();
    }

    /// <inheritdoc/>
    public List<string>? GetFeatureNames()
    {
        return _wrappedModel.GetFeatureNames();
    }

    /// <inheritdoc/>
    public void SetFeatureNames(List<string>? featureNames)
    {
        _wrappedModel.SetFeatureNames(featureNames);
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        var deepCopiedWrappedModel = _wrappedModel.DeepCopy();
        return new ShardedModel<T, TInput, TOutput>(deepCopiedWrappedModel, _config);
    }

    /// <inheritdoc/>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        return _wrappedModel.GetActiveFeatureIndices();
    }

    /// <inheritdoc/>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _wrappedModel.SetActiveFeatureIndices(featureIndices);
    }

    /// <inheritdoc/>
    public bool IsFeatureUsed(int featureIndex)
    {
        return _wrappedModel.IsFeatureUsed(featureIndex);
    }
}
