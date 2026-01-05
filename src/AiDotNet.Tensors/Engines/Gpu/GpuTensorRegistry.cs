using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Manages the lifecycle of GPU tensors with memory pressure handling.
/// Tracks tensor usage and can evict least-recently-used tensors when memory is low.
/// </summary>
/// <remarks>
/// <para><b>Purpose:</b></para>
/// <para>
/// The registry provides centralized management of GPU memory for tensors.
/// It tracks which tensors are currently on GPU and can make eviction decisions
/// when memory pressure is detected.
/// </para>
/// <para><b>Eviction Strategy:</b></para>
/// <list type="bullet">
/// <item>Weights/Biases are never evicted (permanent residents)</item>
/// <item>Activations are evicted first (can be recomputed)</item>
/// <item>Intermediates are evicted next (temporary by definition)</item>
/// <item>LRU ordering within each priority tier</item>
/// </list>
/// </remarks>
public sealed class GpuTensorRegistry : IDisposable
{
    private readonly IDirectGpuBackend _backend;
    private readonly GpuExecutionOptions _options;
    private readonly ConcurrentDictionary<int, TensorEntry> _tensors = new();
    private readonly object _evictionLock = new();
    private int _nextId;
    private long _totalAllocatedBytes;
    private long _maxMemoryBytes;
    private bool _disposed;

    /// <summary>
    /// Gets the total number of registered tensors.
    /// </summary>
    public int TensorCount => _tensors.Count;

    /// <summary>
    /// Gets the total allocated memory in bytes.
    /// </summary>
    public long TotalAllocatedBytes => _totalAllocatedBytes;

    /// <summary>
    /// Gets the maximum memory usage limit in bytes.
    /// </summary>
    public long MaxMemoryBytes => _maxMemoryBytes;

    /// <summary>
    /// Gets the current memory usage as a fraction of the maximum.
    /// </summary>
    public double MemoryUsage => _maxMemoryBytes > 0 ? (double)_totalAllocatedBytes / _maxMemoryBytes : 0;

    /// <summary>
    /// Gets whether memory pressure is high (above threshold).
    /// </summary>
    public bool IsUnderMemoryPressure => MemoryUsage > _options.MaxMemoryUsage;

    /// <summary>
    /// Creates a new tensor registry for the given backend.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="options">Execution options controlling memory behavior.</param>
    public GpuTensorRegistry(IDirectGpuBackend backend, GpuExecutionOptions? options = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _options = options ?? new GpuExecutionOptions();
        _maxMemoryBytes = (long)(backend.GlobalMemoryBytes * _options.MaxMemoryUsage);
    }

    /// <summary>
    /// Registers a tensor with the registry for lifecycle management.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="tensor">The tensor to register.</param>
    /// <returns>A registration handle that should be kept while the tensor is in use.</returns>
    public TensorRegistration Register<T>(IGpuTensor<T> tensor)
    {
        ThrowIfDisposed();

        int id = Interlocked.Increment(ref _nextId);
        var entry = new TensorEntry(id, tensor.Buffer, tensor.Role, tensor.Buffer.SizeInBytes);
        _tensors.TryAdd(id, entry);

        Interlocked.Add(ref _totalAllocatedBytes, entry.SizeInBytes);

        // Check memory pressure after registration
        if (IsUnderMemoryPressure)
        {
            TryEvictLeastUsed();
        }

        return new TensorRegistration(this, id);
    }

    /// <summary>
    /// Marks a tensor as recently used to prevent eviction.
    /// </summary>
    /// <param name="registrationId">The registration ID.</param>
    public void Touch(int registrationId)
    {
        if (_tensors.TryGetValue(registrationId, out var entry))
        {
            entry.LastAccessTime = DateTime.UtcNow;
            entry.AccessCount++;
        }
    }

    /// <summary>
    /// Unregisters a tensor, releasing its registration.
    /// Does not dispose the tensor itself.
    /// </summary>
    /// <param name="registrationId">The registration ID.</param>
    public void Unregister(int registrationId)
    {
        if (_tensors.TryRemove(registrationId, out var entry))
        {
            Interlocked.Add(ref _totalAllocatedBytes, -entry.SizeInBytes);
        }
    }

    /// <summary>
    /// Tries to evict tensors to free memory.
    /// </summary>
    /// <param name="bytesNeeded">The number of bytes to try to free.</param>
    /// <returns>The number of bytes actually freed.</returns>
    public long TryEvict(long bytesNeeded)
    {
        ThrowIfDisposed();

        lock (_evictionLock)
        {
            long freedBytes = 0;

            // Get eviction candidates sorted by priority
            var candidates = GetEvictionCandidates();

            foreach (var candidate in candidates)
            {
                if (freedBytes >= bytesNeeded)
                {
                    break;
                }

                if (TryEvictEntry(candidate))
                {
                    freedBytes += candidate.SizeInBytes;
                }
            }

            return freedBytes;
        }
    }

    /// <summary>
    /// Gets statistics about registered tensors.
    /// </summary>
    /// <returns>A dictionary of role to count and bytes.</returns>
    public Dictionary<GpuTensorRole, (int Count, long Bytes)> GetStatistics()
    {
        var stats = new Dictionary<GpuTensorRole, (int Count, long Bytes)>();

        foreach (GpuTensorRole role in Enum.GetValues(typeof(GpuTensorRole)))
        {
            stats[role] = (0, 0);
        }

        foreach (var entry in _tensors.Values)
        {
            var current = stats[entry.Role];
            stats[entry.Role] = (current.Count + 1, current.Bytes + entry.SizeInBytes);
        }

        return stats;
    }

    /// <summary>
    /// Clears all registered tensors.
    /// </summary>
    public void Clear()
    {
        lock (_evictionLock)
        {
            _tensors.Clear();
            Interlocked.Exchange(ref _totalAllocatedBytes, 0);
        }
    }

    private void TryEvictLeastUsed()
    {
        // Only evict if significantly over threshold
        if (MemoryUsage < _options.MaxMemoryUsage + 0.1)
        {
            return;
        }

        lock (_evictionLock)
        {
            // Try to get back under the threshold
            // Note: _maxMemoryBytes is already calculated as backend.GlobalMemoryBytes * _options.MaxMemoryUsage
            // in the constructor, so we use it directly without multiplying again
            long targetBytes = _maxMemoryBytes;
            long bytesToFree = _totalAllocatedBytes - targetBytes;

            if (bytesToFree > 0)
            {
                TryEvict(bytesToFree);
            }
        }
    }

    private IEnumerable<TensorEntry> GetEvictionCandidates()
    {
        // Eviction priority:
        // 1. Intermediate tensors (temporary)
        // 2. Activation tensors (can be recomputed)
        // 3. Statistics (can be recomputed)
        // 4. Gradient tensors (only during training)
        // Never evict: Weights, Biases, Input, Output, AttentionCache

        var neverEvict = new HashSet<GpuTensorRole>
        {
            GpuTensorRole.Weight,
            GpuTensorRole.Bias,
            GpuTensorRole.Input,
            GpuTensorRole.Output,
            GpuTensorRole.AttentionCache
        };

        var candidates = _tensors.Values
            .Where(e => !neverEvict.Contains(e.Role))
            .OrderBy(GetEvictionPriority)
            .ThenBy(e => e.LastAccessTime)
            .ToList();

        return candidates;
    }

    private static int GetEvictionPriority(TensorEntry entry)
    {
        return entry.Role switch
        {
            GpuTensorRole.Intermediate => 0, // Evict first
            GpuTensorRole.Activation => 1,
            GpuTensorRole.Statistics => 2,
            GpuTensorRole.Gradient => 3,
            GpuTensorRole.General => 4,
            _ => 10 // Never evict
        };
    }

    private bool TryEvictEntry(TensorEntry entry)
    {
        // Try to remove from registry
        if (_tensors.TryRemove(entry.Id, out _))
        {
            Interlocked.Add(ref _totalAllocatedBytes, -entry.SizeInBytes);

            // Note: We don't dispose the buffer here because the GpuTensor owns it.
            // The tensor will be disposed when it goes out of scope.
            // For actual eviction (downloading to CPU and freeing GPU memory),
            // the caller should handle that based on notifications.

            return true;
        }

        return false;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuTensorRegistry));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Clear();
    }

    /// <summary>
    /// Internal entry tracking a registered tensor.
    /// </summary>
    private sealed class TensorEntry
    {
        public int Id { get; }
        public IGpuBuffer Buffer { get; }
        public GpuTensorRole Role { get; }
        public long SizeInBytes { get; }
        public DateTime CreationTime { get; }
        public DateTime LastAccessTime { get; set; }
        public int AccessCount { get; set; }

        public TensorEntry(int id, IGpuBuffer buffer, GpuTensorRole role, long sizeInBytes)
        {
            Id = id;
            Buffer = buffer;
            Role = role;
            SizeInBytes = sizeInBytes;
            CreationTime = DateTime.UtcNow;
            LastAccessTime = DateTime.UtcNow;
            AccessCount = 0;
        }
    }
}

/// <summary>
/// Handle for a tensor registration that automatically unregisters when disposed.
/// </summary>
public sealed class TensorRegistration : IDisposable
{
    private readonly GpuTensorRegistry _registry;
    private readonly int _id;
    private bool _disposed;

    /// <summary>
    /// Gets the registration ID.
    /// </summary>
    public int Id => _id;

    internal TensorRegistration(GpuTensorRegistry registry, int id)
    {
        _registry = registry;
        _id = id;
    }

    /// <summary>
    /// Marks the tensor as recently used.
    /// </summary>
    public void Touch()
    {
        if (!_disposed)
        {
            _registry.Touch(_id);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _registry.Unregister(_id);
    }
}
