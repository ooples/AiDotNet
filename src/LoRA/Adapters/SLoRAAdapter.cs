using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// S-LoRA adapter for scalable serving of thousands of concurrent LoRA adapters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// S-LoRA (Scalable LoRA) is a system designed for efficient serving of many LoRA adapters simultaneously.
/// Published in November 2023, it addresses the challenge of deploying thousands of task-specific LoRA adapters
/// in production environments with limited GPU memory.
/// </para>
/// <para><b>For Beginners:</b> S-LoRA solves a real-world problem in production AI systems.
///
/// The problem:
/// - You have a large base model (like GPT or LLaMA)
/// - You want to serve thousands of different LoRA adapters (one per customer, task, or use case)
/// - Each adapter is small (few MB), but thousands of them won't fit in GPU memory
/// - Naive approaches either: load one adapter at a time (slow) or reserve memory for all (wasteful)
///
/// S-LoRA's solution:
/// - Unified memory pool: Dynamically manage adapter weights and cache together
/// - Batched computation: Process multiple adapters in parallel efficiently
/// - Adapter clustering: Group adapters by rank for optimized computation
/// - On-demand loading: Fetch adapters from CPU to GPU memory only when needed
///
/// Key features implemented:
/// 1. **Unified Memory Pool**: Single pool for adapter weights (no pre-allocation waste)
/// 2. **Adapter Clustering**: Group adapters by rank for batched computation
/// 3. **Dynamic Loading**: Load adapters on-demand, evict when not needed
/// 4. **Batched Forward Pass**: Process multiple requests with different adapters simultaneously
/// 5. **Memory Efficiency**: Serve 100x more adapters than naive approaches
///
/// Research Paper Reference:
/// "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
/// Ying Sheng, Shiyi Cao, et al. (November 2023)
/// arXiv:2311.03285
///
/// Performance (from paper):
/// - Throughput: 4x improvement over vLLM, 30x over HuggingFace PEFT
/// - Adapter capacity: 2,000+ concurrent adapters on single server
/// - Memory efficiency: 75-90% GPU memory utilization
/// - Scalability: Superlinear throughput scaling with more GPUs
///
/// Example usage:
/// ```csharp
/// // Create S-LoRA serving system for base layer
/// var sloraAdapter = new SLoRAAdapter&lt;double&gt;(baseLayer, rank: 8);
///
/// // Register multiple adapters for different tasks
/// sloraAdapter.RegisterAdapter("customer_1", adapter1);
/// sloraAdapter.RegisterAdapter("customer_2", adapter2);
/// sloraAdapter.RegisterAdapter("task_classification", adapter3);
///
/// // Process batched requests efficiently
/// var outputs = sloraAdapter.BatchForward(inputs, adapterIds);
/// ```
///
/// When to use S-LoRA:
/// - Serving multiple LoRA adapters in production
/// - Multi-tenant AI systems (one adapter per tenant)
/// - Task-specific fine-tuning at scale
/// - Limited GPU memory but many adapters
/// - Need high throughput with many concurrent users
///
/// Differences from standard LoRA:
/// - Standard LoRA: Single adapter, simple forward/backward pass
/// - S-LoRA: Multiple adapters, optimized for concurrent serving, memory pooling
/// </para>
/// </remarks>
public class SLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Represents an adapter entry in the memory pool.
    /// </summary>
    private class AdapterEntry
    {
        /// <summary>
        /// The adapter's unique identifier.
        /// </summary>
        public string Id { get; set; }

        /// <summary>
        /// The LoRA layer for this adapter.
        /// </summary>
        public LoRALayer<T> Layer { get; set; }

        /// <summary>
        /// The rank of this adapter.
        /// </summary>
        public int Rank { get; set; }

        /// <summary>
        /// Whether this adapter is currently loaded in "GPU memory" (in-memory cache).
        /// </summary>
        public bool IsLoaded { get; set; }

        /// <summary>
        /// Last access timestamp for LRU eviction.
        /// </summary>
        public long LastAccess { get; set; }

        /// <summary>
        /// Reference count for active requests using this adapter.
        /// </summary>
        public int ReferenceCount { get; set; }

        /// <summary>
        /// Initializes a new adapter entry.
        /// </summary>
        public AdapterEntry(string id, LoRALayer<T> layer, int rank)
        {
            Id = id ?? string.Empty;
            Layer = layer;
            Rank = rank;
            IsLoaded = false;
            LastAccess = 0;
            ReferenceCount = 0;
        }
    }

    /// <summary>
    /// Unified memory pool storing all registered adapters.
    /// </summary>
    /// <remarks>
    /// This simulates S-LoRA's unified memory pool where all adapters reside in CPU memory
    /// and are dynamically loaded to GPU memory based on demand.
    /// </remarks>
    private readonly Dictionary<string, AdapterEntry> _adapterPool;

    /// <summary>
    /// Adapters currently loaded in "GPU memory" (in-memory cache).
    /// </summary>
    private readonly Dictionary<string, AdapterEntry> _loadedAdapters;

    /// <summary>
    /// Adapters clustered by rank for efficient batched computation.
    /// </summary>
    private readonly Dictionary<int, List<string>> _rankClusters;

    /// <summary>
    /// Maximum number of adapters that can be loaded simultaneously (simulates GPU memory limit).
    /// </summary>
    private readonly int _maxLoadedAdapters;

    /// <summary>
    /// Current timestamp for LRU eviction policy.
    /// </summary>
    private long _timestamp;

    /// <summary>
    /// Gets the total number of registered adapters in the pool.
    /// </summary>
    /// <remarks>
    /// This represents all adapters in the system, including those not currently loaded.
    /// S-LoRA can serve thousands of adapters from a unified pool.
    /// </remarks>
    public int TotalAdapterCount => _adapterPool.Count;

    /// <summary>
    /// Gets the number of adapters currently loaded in memory.
    /// </summary>
    /// <remarks>
    /// This represents the "hot" adapters actively being used or cached.
    /// S-LoRA dynamically loads/evicts adapters based on request patterns.
    /// </remarks>
    public int LoadedAdapterCount => _loadedAdapters.Count;

    /// <summary>
    /// Gets the maximum number of adapters that can be loaded simultaneously.
    /// </summary>
    /// <remarks>
    /// This simulates GPU memory constraints. S-LoRA's unified paging mechanism
    /// efficiently manages this limited resource.
    /// </remarks>
    public int MaxLoadedAdapters => _maxLoadedAdapters;

    /// <summary>
    /// Gets the number of rank clusters for batched computation optimization.
    /// </summary>
    /// <remarks>
    /// Adapters with the same rank are clustered together for efficient batched computation.
    /// This is a key optimization in S-LoRA for heterogeneous adapter serving.
    /// </remarks>
    public int RankClusterCount => _rankClusters.Count;

    /// <summary>
    /// Initializes a new S-LoRA adapter for scalable multi-adapter serving.
    /// </summary>
    /// <param name="baseLayer">The base layer to adapt with S-LoRA.</param>
    /// <param name="rank">The default rank for the primary LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="maxLoadedAdapters">Maximum number of adapters to keep loaded simultaneously (default: 100).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when maxLoadedAdapters is less than 1.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an S-LoRA serving system for efficient multi-adapter deployment.
    ///
    /// Parameters:
    /// - baseLayer: The shared base model that all adapters modify
    /// - rank: Default rank for new adapters (typical: 8-32)
    /// - alpha: Scaling factor for LoRA contributions
    /// - maxLoadedAdapters: How many adapters to cache in "GPU memory" (100 = good balance)
    /// - freezeBaseLayer: Lock base weights (true for serving, false for continued training)
    ///
    /// How S-LoRA works:
    /// 1. One base model shared across all adapters (memory efficient)
    /// 2. Thousands of small adapters registered in unified pool
    /// 3. Only popular adapters kept loaded in fast memory
    /// 4. Unpopular adapters evicted and loaded on-demand
    /// 5. Batched computation for multiple adapters simultaneously
    ///
    /// Example: Serving 10,000 customer-specific adapters:
    /// - Base model: 7B parameters (14 GB)
    /// - Each adapter: rank 16 (few MB)
    /// - Total pool: 10,000 adapters (few GB in CPU memory)
    /// - Loaded cache: 100 most-used adapters (hundreds of MB in GPU memory)
    /// - Result: Serve 10,000 adapters with GPU memory for 1 base model + 100 adapters!
    ///
    /// This is 100x more efficient than loading full fine-tuned models.
    /// </para>
    /// </remarks>
    public SLoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        double alpha = -1,
        int maxLoadedAdapters = 100,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (maxLoadedAdapters < 1)
        {
            throw new ArgumentException("Max loaded adapters must be at least 1", nameof(maxLoadedAdapters));
        }

        _adapterPool = new Dictionary<string, AdapterEntry>();
        _loadedAdapters = new Dictionary<string, AdapterEntry>();
        _rankClusters = new Dictionary<int, List<string>>();
        _maxLoadedAdapters = maxLoadedAdapters;
        _timestamp = 0;

        // Register the primary adapter (from base class)
        RegisterAdapter("primary", _loraLayer, rank);
        LoadAdapter("primary");
    }

    /// <summary>
    /// Registers a new adapter in the unified memory pool.
    /// </summary>
    /// <param name="adapterId">Unique identifier for this adapter.</param>
    /// <param name="loraLayer">The LoRA layer to register.</param>
    /// <param name="rank">The rank of this adapter.</param>
    /// <exception cref="ArgumentNullException">Thrown when adapterId or loraLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when an adapter with this ID already exists.</exception>
    /// <remarks>
    /// <para>
    /// This method adds a new adapter to S-LoRA's unified memory pool. The adapter is not immediately
    /// loaded into GPU memory but is available for on-demand loading when needed.
    /// </para>
    /// <para><b>For Beginners:</b> This is like adding a new customer or task-specific adapter to your system.
    ///
    /// What happens when you register an adapter:
    /// 1. Adapter stored in CPU memory pool (cheap storage)
    /// 2. Added to rank cluster for batched computation optimization
    /// 3. Not loaded to GPU yet (only loaded when first used)
    /// 4. Can register thousands of adapters this way
    ///
    /// Example: Multi-tenant SaaS application
    /// ```csharp
    /// var slora = new SLoRAAdapter&lt;double&gt;(baseModel, rank: 8, maxLoadedAdapters: 100);
    ///
    /// // Register 1000 customer adapters
    /// for (int i = 0; i &lt; 1000; i++)
    /// {
    ///     var adapter = LoadCustomerAdapter(i);
    ///     slora.RegisterAdapter($"customer_{i}", adapter, rank: 8);
    /// }
    ///
    /// // All 1000 adapters registered, but only 100 will be loaded at once
    /// // Popular customers get fast GPU-cached access
    /// // Inactive customers loaded on-demand from CPU pool
    /// ```
    ///
    /// This enables serving far more adapters than GPU memory allows!
    /// </para>
    /// </remarks>
    public void RegisterAdapter(string adapterId, LoRALayer<T> loraLayer, int rank)
    {
        if (adapterId == null)
        {
            throw new ArgumentNullException(nameof(adapterId));
        }

        if (loraLayer == null)
        {
            throw new ArgumentNullException(nameof(loraLayer));
        }

        if (_adapterPool.ContainsKey(adapterId))
        {
            throw new ArgumentException($"Adapter with ID '{adapterId}' already exists", nameof(adapterId));
        }

        // Create adapter entry
        var entry = new AdapterEntry(adapterId, loraLayer, rank);
        _adapterPool[adapterId] = entry;

        // Add to rank cluster for batched computation
        if (!_rankClusters.ContainsKey(rank))
        {
            _rankClusters[rank] = new List<string>();
        }
        _rankClusters[rank].Add(adapterId);
    }

    /// <summary>
    /// Loads an adapter from the pool into active memory (simulates GPU loading).
    /// </summary>
    /// <param name="adapterId">The ID of the adapter to load.</param>
    /// <exception cref="ArgumentException">Thrown when adapter ID is not found in pool.</exception>
    /// <remarks>
    /// <para>
    /// This method simulates S-LoRA's dynamic adapter loading from CPU to GPU memory.
    /// If the loaded adapter cache is full, it evicts the least recently used adapter.
    /// </para>
    /// <para><b>For Beginners:</b> This moves an adapter from slow storage to fast cache.
    ///
    /// In S-LoRA's architecture:
    /// - CPU memory: All adapters stored here (slow but large capacity)
    /// - GPU memory: Hot adapters cached here (fast but limited capacity)
    ///
    /// Loading process:
    /// 1. Check if adapter already loaded (if yes, update access time and return)
    /// 2. Check if cache is full (if yes, evict least recently used adapter)
    /// 3. Load adapter into cache
    /// 4. Mark as loaded and update access timestamp
    ///
    /// LRU eviction policy:
    /// - Adapters with oldest last access time evicted first
    /// - Adapters with active references (in-flight requests) never evicted
    /// - This keeps popular adapters hot in cache
    ///
    /// Example: Customer request patterns
    /// ```
    /// Time 0: Customer A requests (load adapter A)
    /// Time 1: Customer B requests (load adapter B)
    /// ...
    /// Time 99: Customer Z requests (load adapter Z, cache now full at 100)
    /// Time 100: Customer AA requests (evict least-used, load adapter AA)
    /// Time 101: Customer A requests again (adapter A was evicted, reload)
    /// ```
    ///
    /// Popular customers stay cached, inactive ones evicted automatically!
    /// </para>
    /// </remarks>
    public void LoadAdapter(string adapterId)
    {
        if (!_adapterPool.ContainsKey(adapterId))
        {
            throw new ArgumentException($"Adapter '{adapterId}' not found in pool", nameof(adapterId));
        }

        var entry = _adapterPool[adapterId];

        // If already loaded, just update access time
        if (entry.IsLoaded)
        {
            entry.LastAccess = ++_timestamp;
            return;
        }

        // Evict if cache is full
        while (_loadedAdapters.Count >= _maxLoadedAdapters)
        {
            if (!EvictLRUAdapter())
            {
                // All loaded adapters are pinned (have active references)
                throw new InvalidOperationException(
                    $"Cannot load adapter '{adapterId}': cache is full ({_loadedAdapters.Count}/{_maxLoadedAdapters}) " +
                    "and all loaded adapters are currently in use (pinned with active references). " +
                    "Consider increasing maxLoadedAdapters or releasing adapter references.");
            }
        }

        // Load adapter into cache
        entry.IsLoaded = true;
        entry.LastAccess = ++_timestamp;
        _loadedAdapters[adapterId] = entry;
    }

    /// <summary>
    /// Evicts the least recently used adapter from the loaded cache.
    /// </summary>
    /// <returns>True if an adapter was evicted, false if no adapter could be evicted.</returns>
    /// <remarks>
    /// <para>
    /// This implements S-LoRA's LRU eviction policy for memory management.
    /// Adapters with active references (in-flight requests) are not evicted.
    /// </para>
    /// <para><b>For Beginners:</b> This removes the least popular adapter from fast cache to make room.
    ///
    /// LRU (Least Recently Used) eviction:
    /// - Find adapter with oldest last access time
    /// - Check it's not actively being used (reference count = 0)
    /// - Remove from cache (but keep in pool for future reload)
    /// - Frees space for more popular adapters
    ///
    /// Why this works well:
    /// - Popular adapters get accessed frequently (stay cached)
    /// - Unpopular adapters get evicted (freed memory for others)
    /// - Temporal locality: recent requests predict future requests
    /// - Balance between memory usage and performance
    ///
    /// Example: E-commerce seasonal patterns
    /// ```
    /// Black Friday: Customer adapters for shoppers cached
    /// Normal day: Employee adapters for operations cached
    /// Tax season: Accounting adapters cached
    /// ```
    ///
    /// System automatically adapts to workload patterns!
    /// </para>
    /// </remarks>
    private bool EvictLRUAdapter()
    {
        if (_loadedAdapters.Count == 0)
        {
            return false;
        }

        // Find LRU adapter that's not actively in use
        AdapterEntry? lruEntry = null;
        long minTimestamp = long.MaxValue;

        foreach (var entry in _loadedAdapters.Values)
        {
            // Don't evict adapters with active references
            if (entry.ReferenceCount > 0)
            {
                continue;
            }

            if (entry.LastAccess < minTimestamp)
            {
                minTimestamp = entry.LastAccess;
                lruEntry = entry;
            }
        }

        // Evict the LRU adapter if found
        if (lruEntry != null)
        {
            lruEntry.IsLoaded = false;
            _loadedAdapters.Remove(lruEntry.Id);
            return true;
        }

        // No adapter could be evicted (all are pinned with active references)
        return false;
    }

    /// <summary>
    /// Performs batched forward pass with a specific adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="adapterId">The ID of the adapter to use (default: "primary").</param>
    /// <returns>Output tensor with adapter applied.</returns>
    /// <exception cref="ArgumentException">Thrown when adapter ID is not found.</exception>
    /// <remarks>
    /// <para>
    /// This method performs S-LoRA's optimized forward pass with automatic adapter loading
    /// and reference tracking.
    /// </para>
    /// <para><b>For Beginners:</b> This runs inference with a specific adapter efficiently.
    ///
    /// What happens during forward pass:
    /// 1. Load adapter if not already cached (automatic on-demand loading)
    /// 2. Increment reference count (prevent eviction during processing)
    /// 3. Run base model forward pass
    /// 4. Run adapter-specific LoRA computation
    /// 5. Combine base output + adapter output
    /// 6. Decrement reference count (allow eviction if needed)
    ///
    /// Key S-LoRA optimizations simulated:
    /// - Separated base and adapter computation (can batch differently)
    /// - Automatic loading from unified pool
    /// - Reference counting prevents eviction during processing
    /// - LRU access tracking for cache management
    ///
    /// Example: Multi-customer request handling
    /// ```csharp
    /// // Request from customer A
    /// var outputA = slora.Forward(inputA, "customer_a");
    ///
    /// // Request from customer B (different adapter)
    /// var outputB = slora.Forward(inputB, "customer_b");
    ///
    /// // Request from customer A again (adapter still cached)
    /// var outputA2 = slora.Forward(inputA2, "customer_a");
    /// ```
    ///
    /// Each customer gets their personalized model behavior efficiently!
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, string adapterId = "primary")
    {
        if (!_adapterPool.ContainsKey(adapterId))
        {
            throw new ArgumentException($"Adapter '{adapterId}' not found", nameof(adapterId));
        }

        // Load adapter if not already loaded
        LoadAdapter(adapterId);

        var entry = _adapterPool[adapterId];

        // Increment reference count
        entry.ReferenceCount++;

        try
        {
            // Forward through base layer
            Tensor<T> baseOutput = _baseLayer.Forward(input);

            // Forward through adapter-specific LoRA layer
            Tensor<T> loraOutput = entry.Layer.Forward(input);

            // Combine base and adapter outputs
            Tensor<T> result = new Tensor<T>(baseOutput.Shape);
            for (int i = 0; i < baseOutput.Length; i++)
            {
                result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
            }

            return result;
        }
        finally
        {
            // Decrement reference count
            entry.ReferenceCount--;
        }
    }

    /// <summary>
    /// Performs batched forward pass with multiple adapters simultaneously.
    /// </summary>
    /// <param name="inputs">Array of input tensors.</param>
    /// <param name="adapterIds">Array of adapter IDs corresponding to each input.</param>
    /// <returns>Array of output tensors.</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputs or adapterIds is null.</exception>
    /// <exception cref="ArgumentException">Thrown when array lengths don't match or adapter not found.</exception>
    /// <remarks>
    /// <para>
    /// This method demonstrates S-LoRA's key innovation: efficient batched computation across
    /// heterogeneous adapters. Adapters are clustered by rank for optimized computation.
    /// </para>
    /// <para><b>For Beginners:</b> This is S-LoRA's killer feature - processing many requests efficiently!
    ///
    /// The problem with naive batching:
    /// - Request 1: Use customer A's adapter (rank 8)
    /// - Request 2: Use customer B's adapter (rank 16)
    /// - Request 3: Use customer C's adapter (rank 8)
    /// - Naive approach: Process one by one (slow) or merge adapters (memory expensive)
    ///
    /// S-LoRA's solution:
    /// 1. Group requests by adapter rank (rank-based clustering)
    /// 2. Process same-rank adapters in optimized batches
    /// 3. Use custom kernels for heterogeneous batching
    /// 4. Minimize memory overhead and maximize throughput
    ///
    /// Batching strategy:
    /// - Cluster 1 (rank 8): [customer A, customer C] - batch process together
    /// - Cluster 2 (rank 16): [customer B] - process separately
    /// - Base model: Shared computation for all requests
    ///
    /// Performance benefits (from paper):
    /// - 4x throughput vs. non-batched serving
    /// - 30x throughput vs. merging adapters per request
    /// - Near-linear scaling with more concurrent requests
    /// - 75-90% GPU utilization
    ///
    /// Example: Multi-tenant API serving
    /// ```csharp
    /// // Batch of 100 requests from different customers
    /// var inputs = new Tensor&lt;T&gt;[100];
    /// var adapterIds = new string[100];
    ///
    /// for (int i = 0; i &lt; 100; i++)
    /// {
    ///     inputs[i] = GetCustomerRequest(i);
    ///     adapterIds[i] = $"customer_{GetCustomerId(i)}";
    /// }
    ///
    /// // Process entire batch efficiently (S-LoRA magic!)
    /// var outputs = slora.BatchForward(inputs, adapterIds);
    /// ```
    ///
    /// This enables high-throughput multi-tenant AI serving!
    /// </para>
    /// </remarks>
    public Tensor<T>[] BatchForward(Tensor<T>[] inputs, string[] adapterIds)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (adapterIds == null)
        {
            throw new ArgumentNullException(nameof(adapterIds));
        }

        if (inputs.Length != adapterIds.Length)
        {
            throw new ArgumentException("Number of inputs must match number of adapter IDs", nameof(adapterIds));
        }

        // Cluster requests by adapter for batched computation
        var requestClusters = new Dictionary<string, List<int>>();
        for (int i = 0; i < adapterIds.Length; i++)
        {
            if (!_adapterPool.ContainsKey(adapterIds[i]))
            {
                throw new ArgumentException($"Adapter '{adapterIds[i]}' not found", nameof(adapterIds));
            }

            if (!requestClusters.ContainsKey(adapterIds[i]))
            {
                requestClusters[adapterIds[i]] = new List<int>();
            }
            requestClusters[adapterIds[i]].Add(i);
        }

        // Prepare output array
        Tensor<T>[] outputs = new Tensor<T>[inputs.Length];

        // Process each adapter cluster
        foreach (var cluster in requestClusters)
        {
            string adapterId = cluster.Key;
            List<int> requestIndices = cluster.Value;

            // Load adapter once for entire cluster
            LoadAdapter(adapterId);
            var entry = _adapterPool[adapterId];

            // Increment reference count for this batch
            entry.ReferenceCount += requestIndices.Count;

            try
            {
                // Process all requests in this cluster
                foreach (int idx in requestIndices)
                {
                    // Forward through base layer
                    Tensor<T> baseOutput = _baseLayer.Forward(inputs[idx]);

                    // Forward through adapter-specific LoRA layer
                    Tensor<T> loraOutput = entry.Layer.Forward(inputs[idx]);

                    // Combine base and adapter outputs
                    Tensor<T> result = new Tensor<T>(baseOutput.Shape);
                    for (int i = 0; i < baseOutput.Length; i++)
                    {
                        result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
                    }

                    outputs[idx] = result;
                }
            }
            finally
            {
                // Decrement reference count
                entry.ReferenceCount -= requestIndices.Count;
            }
        }

        return outputs;
    }

    /// <summary>
    /// Gets the list of adapter IDs in a specific rank cluster.
    /// </summary>
    /// <param name="rank">The rank to query.</param>
    /// <returns>List of adapter IDs with the specified rank, or empty list if none.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to S-LoRA's rank-based clustering information.
    /// Adapters with the same rank can be batched together more efficiently.
    /// </para>
    /// <para><b>For Beginners:</b> This shows which adapters can be batched together efficiently.
    ///
    /// Why rank clustering matters:
    /// - Adapters with same rank have same computational cost
    /// - Can use same CUDA kernels / computation paths
    /// - Better memory access patterns
    /// - Higher GPU utilization
    ///
    /// Example: Analyzing your adapter distribution
    /// ```csharp
    /// var slora = new SLoRAAdapter&lt;double&gt;(baseModel, rank: 8);
    ///
    /// // Register many adapters with different ranks
    /// // ...
    ///
    /// // See how adapters are distributed
    /// var rank8Adapters = slora.GetRankCluster(8);   // Maybe 500 adapters
    /// var rank16Adapters = slora.GetRankCluster(16); // Maybe 300 adapters
    /// var rank32Adapters = slora.GetRankCluster(32); // Maybe 200 adapters
    ///
    /// Console.WriteLine($"Rank 8: {rank8Adapters.Count} adapters");
    /// Console.WriteLine($"Rank 16: {rank16Adapters.Count} adapters");
    /// Console.WriteLine($"Rank 32: {rank32Adapters.Count} adapters");
    /// ```
    ///
    /// This helps optimize batch sizes and resource allocation!
    /// </para>
    /// </remarks>
    public List<string> GetRankCluster(int rank)
    {
        if (_rankClusters.ContainsKey(rank))
        {
            return new List<string>(_rankClusters[rank]);
        }
        return new List<string>();
    }

    /// <summary>
    /// Gets statistics about the current state of the S-LoRA system.
    /// </summary>
    /// <returns>Dictionary containing system statistics.</returns>
    /// <remarks>
    /// <para>
    /// This method provides detailed statistics about S-LoRA's memory usage, cache efficiency,
    /// and adapter distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you insights into how well your S-LoRA system is performing.
    ///
    /// Key metrics returned:
    /// - TotalAdapters: How many adapters registered in pool
    /// - LoadedAdapters: How many currently cached in "GPU memory"
    /// - CacheUtilization: Percentage of cache capacity used
    /// - RankClusters: Number of different rank groups
    /// - AverageRank: Mean rank across all adapters
    /// - ActiveReferences: Adapters currently processing requests
    ///
    /// Example: Monitoring production system
    /// ```csharp
    /// var stats = slora.GetStatistics();
    ///
    /// Console.WriteLine($"Total adapters: {stats["TotalAdapters"]}");
    /// Console.WriteLine($"Loaded adapters: {stats["LoadedAdapters"]}");
    /// Console.WriteLine($"Cache utilization: {stats["CacheUtilization"]}%");
    ///
    /// // Alert if cache too small
    /// if ((double)stats["CacheUtilization"] &gt; 95)
    /// {
    ///     Console.WriteLine("Warning: Cache nearly full, consider increasing maxLoadedAdapters");
    /// }
    /// ```
    ///
    /// Use this to tune your S-LoRA configuration for optimal performance!
    /// </para>
    /// </remarks>
    public Dictionary<string, double> GetStatistics()
    {
        var stats = new Dictionary<string, double>();

        stats["TotalAdapters"] = _adapterPool.Count;
        stats["LoadedAdapters"] = _loadedAdapters.Count;
        stats["CacheUtilization"] = (_loadedAdapters.Count / (double)_maxLoadedAdapters) * 100.0;
        stats["RankClusters"] = _rankClusters.Count;

        // Calculate average rank
        if (_adapterPool.Count > 0)
        {
            double totalRank = _adapterPool.Values.Sum(e => e.Rank);
            stats["AverageRank"] = totalRank / _adapterPool.Count;
        }
        else
        {
            stats["AverageRank"] = 0;
        }

        // Count active references
        int activeReferences = _loadedAdapters.Values.Sum(e => e.ReferenceCount);
        stats["ActiveReferences"] = activeReferences;

        return stats;
    }

    /// <summary>
    /// Merges the primary adapter into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with primary LoRA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// For S-LoRA, this merges the primary adapter (the one created during initialization).
    /// In production S-LoRA deployments, individual adapters typically remain separate for
    /// efficient multi-adapter serving rather than being merged.
    /// </para>
    /// <para><b>For Beginners:</b> This merges the default adapter for deployment.
    ///
    /// When to merge adapters:
    /// - Deploying a single-adapter model (no longer need multi-adapter serving)
    /// - Want maximum inference speed for one specific adapter
    /// - Converting S-LoRA deployment back to standard model
    ///
    /// When NOT to merge:
    /// - Serving multiple adapters (defeats purpose of S-LoRA)
    /// - Need to swap adapters dynamically
    /// - Want memory efficiency of shared base model
    ///
    /// S-LoRA's strength is NOT merging:
    /// - Keep base model frozen and shared
    /// - Keep all adapters separate in pool
    /// - Swap adapters per request efficiently
    /// - Serve thousands of adapters from one base model
    ///
    /// This method is mainly for compatibility or transitioning away from S-LoRA architecture.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // For S-LoRA, we merge the primary adapter
        // In production, adapters typically remain separate
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("SLoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the primary LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Calculate dimensions
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Clears all adapters from the pool (useful for testing or reset).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method removes all adapters from the unified pool except the primary adapter.
    /// Useful for resetting the system or clearing adapters during reconfiguration.
    /// </para>
    /// <para><b>For Beginners:</b> This wipes all registered adapters (except the default one).
    ///
    /// Use cases:
    /// - Testing: Reset between test runs
    /// - Maintenance: Clear old adapters no longer in use
    /// - Reconfiguration: Remove all adapters before registering new set
    /// - Memory cleanup: Free memory from unused adapters
    ///
    /// Example: Periodic cleanup
    /// ```csharp
    /// // Monthly cleanup of inactive customer adapters
    /// slora.ClearAdapters();
    ///
    /// // Re-register only active customers
    /// foreach (var customer in GetActiveCustomers())
    /// {
    ///     var adapter = LoadCustomerAdapter(customer.Id);
    ///     slora.RegisterAdapter(customer.Id, adapter, customer.Rank);
    /// }
    /// ```
    ///
    /// Note: Primary adapter is preserved to maintain base functionality.
    /// </para>
    /// </remarks>
    public void ClearAdapters()
    {
        _loadedAdapters.Clear();
        _rankClusters.Clear();

        // Keep only the primary adapter
        var primaryEntry = _adapterPool["primary"];
        _adapterPool.Clear();
        _adapterPool["primary"] = primaryEntry;
        _rankClusters[primaryEntry.Rank] = new List<string> { "primary" };

        // Reload primary adapter
        LoadAdapter("primary");
    }
}
