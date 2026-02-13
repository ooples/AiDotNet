

namespace AiDotNet.Inference;

/// <summary>
/// Key-Value cache for efficient autoregressive inference in transformer models.
/// </summary>
/// <remarks>
/// <para>
/// The KV-Cache stores computed Key and Value projections from attention layers,
/// enabling efficient incremental generation where each new token only needs to
/// compute attention against cached keys/values rather than recomputing everything.
/// </para>
/// <para><b>For Beginners:</b> KV-Cache is like a memory bank for transformers.
///
/// When generating text:
/// 1. First token: Compute and cache K, V for position 0
/// 2. Second token: Compute K, V for position 1, append to cache, attend to positions 0-1
/// 3. Third token: Compute K, V for position 2, append to cache, attend to positions 0-2
/// ... and so on
///
/// Without caching, token N would require recomputing K, V for positions 0 to N-1.
/// With caching, we only compute K, V for the new position and look up the rest.
///
/// This provides massive speedup for autoregressive generation:
/// - Without cache: O(N^2) total compute for N tokens
/// - With cache: O(N) total compute for N tokens
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for cache storage (typically float or double).</typeparam>
internal class KVCache<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly KVCacheConfig _config;

    // Cache storage: [layer][batch, heads, seq, headDim]
    private readonly Tensor<T>[] _keyCache;
    private readonly Tensor<T>[] _valueCache;

    // Optional FP16 cache storage (used when Config.DataType == Float16 and T is float/double)
    private readonly Tensor<Half>[]? _keyCacheFp16;
    private readonly Tensor<Half>[]? _valueCacheFp16;
    private readonly bool _useFp16Storage;
    private readonly Func<T, Half>? _toHalf;
    private readonly Func<Half, T>? _fromHalf;

    // Optional int8 quantized cache storage (used when Config.DataType == Int8).
    private readonly Tensor<sbyte>[]? _keyCacheInt8;
    private readonly Tensor<sbyte>[]? _valueCacheInt8;
    private readonly bool _useInt8Storage;
    private readonly float[]? _keyScaleInt8;
    private readonly float[]? _valueScaleInt8;
    private readonly Func<T, float>? _toFloat;
    private readonly Func<float, T>? _fromFloat;

    // Current sequence length for each layer and batch item: [layer][batch]
    private readonly int[][] _sequenceLengths;

    // Statistics
    private long _cacheHits;
    private long _cacheMisses;
    private long _evictions;

    /// <summary>
    /// Gets the configuration used for this cache.
    /// </summary>
    public KVCacheConfig Config => _config;

    /// <summary>
    /// Gets the current number of cached tokens for batch item 0.
    /// </summary>
    public int CurrentLength => _sequenceLengths.Length > 0 ? _sequenceLengths[0][0] : 0;

    /// <summary>
    /// Gets the maximum sequence length this cache can hold.
    /// </summary>
    public int MaxLength => _config.MaxSequenceLength;

    /// <summary>
    /// Gets the number of cache hits (successful lookups).
    /// </summary>
    public long CacheHits => _cacheHits;

    /// <summary>
    /// Gets the number of cache misses (new computations needed).
    /// </summary>
    public long CacheMisses => _cacheMisses;

    /// <summary>
    /// Gets the number of evicted entries (due to sliding window).
    /// </summary>
    public long Evictions => _evictions;

    /// <summary>
    /// Creates a new KV-Cache with the specified configuration.
    /// </summary>
    /// <param name="config">Cache configuration.</param>
    public KVCache(KVCacheConfig config)
    {
        Guard.NotNull(config);
        _config = config;

        _keyCache = new Tensor<T>[config.NumLayers];
        _valueCache = new Tensor<T>[config.NumLayers];

        if (config.DataType == CacheDataType.Int8)
        {
            // Only enable int8 storage when we can safely convert between T and float.
            if (typeof(T) == typeof(float))
            {
                _useInt8Storage = true;
                _toFloat = value => (float)(object)value!;
                _fromFloat = value => (T)(object)value;
            }
            else if (typeof(T) == typeof(double))
            {
                _useInt8Storage = true;
                _toFloat = value => (float)(double)(object)value!;
                _fromFloat = value => (T)(object)(double)value;
            }
            else if (typeof(T) == typeof(Half))
            {
                _useInt8Storage = true;
                _toFloat = value => (float)(Half)(object)value!;
                _fromFloat = value => (T)(object)(Half)value;
            }
        }

        if (config.DataType == CacheDataType.Float16 && typeof(T) != typeof(Half))
        {
            // Only enable FP16 storage when we can safely convert between T and Half.
            if (typeof(T) == typeof(float))
            {
                _useFp16Storage = true;
                _toHalf = value => (Half)(float)(object)value!;
                _fromHalf = value => (T)(object)(float)value;
            }
            else if (typeof(T) == typeof(double))
            {
                _useFp16Storage = true;
                _toHalf = value => (Half)(double)(object)value!;
                _fromHalf = value => (T)(object)(double)(float)value;
            }
        }

        if (_useFp16Storage)
        {
            _keyCacheFp16 = new Tensor<Half>[config.NumLayers];
            _valueCacheFp16 = new Tensor<Half>[config.NumLayers];
        }

        if (_useInt8Storage)
        {
            _keyCacheInt8 = new Tensor<sbyte>[config.NumLayers];
            _valueCacheInt8 = new Tensor<sbyte>[config.NumLayers];
            _keyScaleInt8 = new float[config.NumLayers];
            _valueScaleInt8 = new float[config.NumLayers];
        }

        _sequenceLengths = new int[config.NumLayers][];
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            _sequenceLengths[layer] = new int[config.MaxBatchSize];
        }

        if (config.PreAllocate)
        {
            AllocateCaches();
        }
    }

    /// <summary>
    /// Creates a new KV-Cache with default configuration.
    /// </summary>
    public KVCache(int numLayers, int numHeads, int headDim, int maxSeqLen, int maxBatchSize = 1)
        : this(new KVCacheConfig
        {
            NumLayers = numLayers,
            NumHeads = numHeads,
            HeadDimension = headDim,
            MaxSequenceLength = maxSeqLen,
            MaxBatchSize = maxBatchSize
        })
    {
    }

    private void AllocateCaches()
    {
        var shape = new[]
        {
            _config.MaxBatchSize,
            _config.NumHeads,
            _config.MaxSequenceLength,
            _config.HeadDimension
        };

        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            if (_useInt8Storage)
            {
                _keyCacheInt8![layer] = new Tensor<sbyte>(shape);
                _valueCacheInt8![layer] = new Tensor<sbyte>(shape);
                _keyScaleInt8![layer] = 0f;
                _valueScaleInt8![layer] = 0f;
            }
            else if (_useFp16Storage)
            {
                _keyCacheFp16![layer] = new Tensor<Half>(shape);
                _valueCacheFp16![layer] = new Tensor<Half>(shape);
            }
            else
            {
                _keyCache[layer] = new Tensor<T>(shape);
                _valueCache[layer] = new Tensor<T>(shape);
            }
        }
    }

    /// <summary>
    /// Appends new key-value pairs to the cache for a specific layer.
    /// </summary>
    /// <param name="layerIndex">The transformer layer index (0-based).</param>
    /// <param name="newKeys">New keys to append, shape [batch, heads, newSeqLen, headDim].</param>
    /// <param name="newValues">New values to append, shape [batch, heads, newSeqLen, headDim].</param>
    /// <returns>Tuple of (allKeys, allValues) including cached and new entries.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds new K, V entries to the cache.
    ///
    /// During generation:
    /// - newKeys/newValues have shape [..., 1, ...] for single new token
    /// - Returns full sequence including all previously cached tokens
    ///
    /// Example for generating token 5:
    /// - Cache has tokens 0-4 cached
    /// - newKeys/newValues contain K, V for token 5
    /// - Returns K, V for tokens 0-5 (cached + new)
    /// </para>
    /// </remarks>
    public (Tensor<T> Keys, Tensor<T> Values) Append(
        int layerIndex,
        Tensor<T> newKeys,
        Tensor<T> newValues)
    {
        ValidateLayerIndex(layerIndex);
        ValidateInputShapes(newKeys, newValues);

        int batchSize = newKeys.Shape[0];
        int newSeqLen = newKeys.Shape[2];

        // Ensure cache is allocated
        EnsureCacheAllocated(layerIndex);

        // Check if we need sliding window eviction
        if (_config.UseSlidingWindow)
        {
            HandleSlidingWindowEviction(layerIndex, batchSize, newSeqLen);
        }

        if (_useInt8Storage)
        {
            EnsureInt8Scales(layerIndex, newKeys, newValues, batchSize, newSeqLen);
        }

        // Append new entries
        for (int b = 0; b < batchSize; b++)
        {
            int currentLen = _sequenceLengths[layerIndex][b];
            int newLen = currentLen + newSeqLen;

            if (newLen > _config.MaxSequenceLength)
            {
                throw new InvalidOperationException(
                    $"Cache overflow: attempting to store {newLen} tokens but max is {_config.MaxSequenceLength}. " +
                    "Consider enabling sliding window or increasing MaxSequenceLength.");
            }

            // Copy new keys and values to cache
            for (int h = 0; h < _config.NumHeads; h++)
            {
                for (int s = 0; s < newSeqLen; s++)
                {
                    int targetPos = currentLen + s;
                    for (int d = 0; d < _config.HeadDimension; d++)
                    {
                        if (_useInt8Storage)
                        {
                            var keyScale = _keyScaleInt8![layerIndex];
                            var valueScale = _valueScaleInt8![layerIndex];
                            _keyCacheInt8![layerIndex][new[] { b, h, targetPos, d }] = QuantizeToInt8(_toFloat!(newKeys[new[] { b, h, s, d }]), keyScale);
                            _valueCacheInt8![layerIndex][new[] { b, h, targetPos, d }] = QuantizeToInt8(_toFloat!(newValues[new[] { b, h, s, d }]), valueScale);
                        }
                        else if (_useFp16Storage)
                        {
                            _keyCacheFp16![layerIndex][new[] { b, h, targetPos, d }] = _toHalf!(newKeys[new[] { b, h, s, d }]);
                            _valueCacheFp16![layerIndex][new[] { b, h, targetPos, d }] = _toHalf!(newValues[new[] { b, h, s, d }]);
                        }
                        else
                        {
                            _keyCache[layerIndex][new[] { b, h, targetPos, d }] = newKeys[new[] { b, h, s, d }];
                            _valueCache[layerIndex][new[] { b, h, targetPos, d }] = newValues[new[] { b, h, s, d }];
                        }
                    }
                }
            }

            _sequenceLengths[layerIndex][b] = newLen;
            _cacheMisses += newSeqLen;
        }

        // Return full cached sequence
        return GetCached(layerIndex, batchSize);
    }

    /// <summary>
    /// Gets cached keys and values for a specific layer up to the current sequence length.
    /// </summary>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <param name="batchSize">Batch size to return (must be <= MaxBatchSize).</param>
    /// <returns>Tuple of (keys, values) tensors containing cached entries.</returns>
    public (Tensor<T> Keys, Tensor<T> Values) GetCached(int layerIndex, int batchSize = 1)
    {
        ValidateLayerIndex(layerIndex);

        if (!IsLayerAllocated(layerIndex))
        {
            throw new InvalidOperationException($"Layer {layerIndex} cache not initialized. Call Append first.");
        }

        // Find max sequence length across batch
        int maxLen = 0;
        for (int b = 0; b < batchSize; b++)
        {
            if (_sequenceLengths[layerIndex][b] > maxLen) maxLen = _sequenceLengths[layerIndex][b];
        }

        if (maxLen == 0)
        {
            // Return empty tensors
            var emptyShape = new[] { batchSize, _config.NumHeads, 0, _config.HeadDimension };
            return (new Tensor<T>(emptyShape), new Tensor<T>(emptyShape));
        }

        // Create output tensors
        var keyShape = new[] { batchSize, _config.NumHeads, maxLen, _config.HeadDimension };
        var keys = new Tensor<T>(keyShape);
        var values = new Tensor<T>(keyShape);

        // Copy cached values
        for (int b = 0; b < batchSize; b++)
        {
            int seqLen = _sequenceLengths[layerIndex][b];
            for (int h = 0; h < _config.NumHeads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < _config.HeadDimension; d++)
                    {
                        if (_useInt8Storage)
                        {
                            float keyScale = _keyScaleInt8![layerIndex];
                            float valueScale = _valueScaleInt8![layerIndex];
                            keys[new[] { b, h, s, d }] = _fromFloat!(DequantizeInt8(_keyCacheInt8![layerIndex][new[] { b, h, s, d }], keyScale));
                            values[new[] { b, h, s, d }] = _fromFloat!(DequantizeInt8(_valueCacheInt8![layerIndex][new[] { b, h, s, d }], valueScale));
                        }
                        else if (_useFp16Storage)
                        {
                            keys[new[] { b, h, s, d }] = _fromHalf!(_keyCacheFp16![layerIndex][new[] { b, h, s, d }]);
                            values[new[] { b, h, s, d }] = _fromHalf!(_valueCacheFp16![layerIndex][new[] { b, h, s, d }]);
                        }
                        else
                        {
                            keys[new[] { b, h, s, d }] = _keyCache[layerIndex][new[] { b, h, s, d }];
                            values[new[] { b, h, s, d }] = _valueCache[layerIndex][new[] { b, h, s, d }];
                        }
                    }
                }
            }
        }

        _cacheHits += (long)batchSize * maxLen;
        return (keys, values);
    }

    /// <summary>
    /// Updates cached keys and values at specific positions (for speculative decoding).
    /// </summary>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <param name="positions">Positions to update, shape [batch, numPositions].</param>
    /// <param name="keys">New keys for the positions.</param>
    /// <param name="values">New values for the positions.</param>
    public void Update(int layerIndex, int[] positions, Tensor<T> keys, Tensor<T> values)
    {
        ValidateLayerIndex(layerIndex);

        int batchSize = keys.Shape[0];
        int numPositions = positions.Length;

        if (_useInt8Storage)
        {
            EnsureCacheAllocated(layerIndex);
            EnsureInt8Scales(layerIndex, keys, values, batchSize, numPositions);
        }

        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPositions; p++)
            {
                int pos = positions[p];
                if (pos < 0 || pos >= _config.MaxSequenceLength)
                {
                    throw new ArgumentOutOfRangeException(nameof(positions),
                        $"Position {pos} is out of range [0, {_config.MaxSequenceLength})");
                }

                for (int h = 0; h < _config.NumHeads; h++)
                {
                    for (int d = 0; d < _config.HeadDimension; d++)
                    {
                        if (_useInt8Storage)
                        {
                            var keyScale = _keyScaleInt8![layerIndex];
                            var valueScale = _valueScaleInt8![layerIndex];
                            _keyCacheInt8![layerIndex][new[] { b, h, pos, d }] = QuantizeToInt8(_toFloat!(keys[new[] { b, h, p, d }]), keyScale);
                            _valueCacheInt8![layerIndex][new[] { b, h, pos, d }] = QuantizeToInt8(_toFloat!(values[new[] { b, h, p, d }]), valueScale);
                        }
                        else if (_useFp16Storage)
                        {
                            _keyCacheFp16![layerIndex][new[] { b, h, pos, d }] = _toHalf!(keys[new[] { b, h, p, d }]);
                            _valueCacheFp16![layerIndex][new[] { b, h, pos, d }] = _toHalf!(values[new[] { b, h, p, d }]);
                        }
                        else
                        {
                            _keyCache[layerIndex][new[] { b, h, pos, d }] = keys[new[] { b, h, p, d }];
                            _valueCache[layerIndex][new[] { b, h, pos, d }] = values[new[] { b, h, p, d }];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Truncates the cache to a specific length (for beam search or rejection).
    /// </summary>
    /// <param name="newLength">New sequence length to truncate to.</param>
    /// <param name="batchIndex">Batch index to truncate (-1 for all).</param>
    public void Truncate(int newLength, int batchIndex = -1)
    {
        if (newLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(newLength), "Length cannot be negative");
        }

        if (batchIndex == -1)
        {
            for (int layer = 0; layer < _sequenceLengths.Length; layer++)
            {
                for (int b = 0; b < _sequenceLengths[layer].Length; b++)
                {
                    _sequenceLengths[layer][b] = Math.Min(_sequenceLengths[layer][b], newLength);
                }
            }
        }
        else
        {
            if (batchIndex < 0 || (_sequenceLengths.Length > 0 && batchIndex >= _sequenceLengths[0].Length))
            {
                throw new ArgumentOutOfRangeException(nameof(batchIndex));
            }

            for (int layer = 0; layer < _sequenceLengths.Length; layer++)
            {
                _sequenceLengths[layer][batchIndex] = Math.Min(_sequenceLengths[layer][batchIndex], newLength);
            }
        }
    }

    /// <summary>
    /// Clears all cached entries.
    /// </summary>
    public void Clear()
    {
        for (int layer = 0; layer < _sequenceLengths.Length; layer++)
        {
            for (int b = 0; b < _sequenceLengths[layer].Length; b++)
            {
                _sequenceLengths[layer][b] = 0;
            }

            if (_useInt8Storage)
            {
                _keyScaleInt8![layer] = 0f;
                _valueScaleInt8![layer] = 0f;
            }
        }

        // Reset statistics
        _cacheHits = 0;
        _cacheMisses = 0;
        _evictions = 0;
    }

    /// <summary>
    /// Clears cache for a specific batch index.
    /// </summary>
    public void Clear(int batchIndex)
    {
        if (batchIndex < 0 || (_sequenceLengths.Length > 0 && batchIndex >= _sequenceLengths[0].Length))
        {
            throw new ArgumentOutOfRangeException(nameof(batchIndex));
        }

        for (int layer = 0; layer < _sequenceLengths.Length; layer++)
        {
            _sequenceLengths[layer][batchIndex] = 0;
        }
    }

    /// <summary>
    /// Gets the current sequence length for a batch item.
    /// </summary>
    public int GetSequenceLength(int batchIndex = 0)
    {
        if (batchIndex < 0 || (_sequenceLengths.Length > 0 && batchIndex >= _sequenceLengths[0].Length))
        {
            throw new ArgumentOutOfRangeException(nameof(batchIndex));
        }

        return _sequenceLengths.Length > 0 ? _sequenceLengths[0][batchIndex] : 0;
    }

    /// <summary>
    /// Gets the current memory usage of the cache in bytes.
    /// </summary>
    public long GetCurrentMemoryUsage()
    {
        long totalElements = 0;
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            if (IsLayerAllocated(layer))
            {
                if (_useInt8Storage)
                {
                    totalElements += _keyCacheInt8![layer].Length + _valueCacheInt8![layer].Length;
                }
                else if (_useFp16Storage)
                {
                    totalElements += _keyCacheFp16![layer].Length + _valueCacheFp16![layer].Length;
                }
                else
                {
                    totalElements += _keyCache[layer].Length + _valueCache[layer].Length;
                }
            }
        }

        int bytesPerElement = _config.DataType switch
        {
            CacheDataType.Int8 => 1,
            CacheDataType.Float16 => 2,
            CacheDataType.Float32 => 4,
            CacheDataType.Float64 => 8,
            CacheDataType.BFloat16 => 2,
            _ => 4
        };

        return totalElements * bytesPerElement;
    }

    /// <summary>
    /// Gets cache statistics as a dictionary.
    /// </summary>
    public Dictionary<string, object> GetStatistics()
    {
        return new Dictionary<string, object>
        {
            ["DataType"] = _config.DataType.ToString(),
            ["UseInt8Storage"] = _useInt8Storage,
            ["UseFp16Storage"] = _useFp16Storage,
            ["CacheHits"] = _cacheHits,
            ["CacheMisses"] = _cacheMisses,
            ["Evictions"] = _evictions,
            ["HitRate"] = _cacheHits + _cacheMisses > 0
                ? (double)_cacheHits / (_cacheHits + _cacheMisses)
                : 0.0,
            ["CurrentMemoryMB"] = GetCurrentMemoryUsage() / (1024.0 * 1024.0),
            ["MaxMemoryMB"] = _config.EstimateMemoryBytes() / (1024.0 * 1024.0),
            ["SequenceLengths"] = _sequenceLengths.Length > 0
                ? _sequenceLengths[0].ToArray()
                : Array.Empty<int>()
        };
    }

    /// <summary>
    /// Copies cache state from one batch index to another (for beam search).
    /// </summary>
    public void CopyBatchState(int sourceBatch, int destBatch)
    {
        if (sourceBatch < 0 || sourceBatch >= _config.MaxBatchSize)
            throw new ArgumentOutOfRangeException(nameof(sourceBatch));
        if (destBatch < 0 || destBatch >= _config.MaxBatchSize)
            throw new ArgumentOutOfRangeException(nameof(destBatch));

        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            if (!IsLayerAllocated(layer)) continue;

            int seqLen = _sequenceLengths[layer][sourceBatch];

            for (int h = 0; h < _config.NumHeads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < _config.HeadDimension; d++)
                    {
                        if (_useInt8Storage)
                        {
                            _keyCacheInt8![layer][new[] { destBatch, h, s, d }] =
                                _keyCacheInt8![layer][new[] { sourceBatch, h, s, d }];
                            _valueCacheInt8![layer][new[] { destBatch, h, s, d }] =
                                _valueCacheInt8![layer][new[] { sourceBatch, h, s, d }];
                        }
                        else if (_useFp16Storage)
                        {
                            _keyCacheFp16![layer][new[] { destBatch, h, s, d }] =
                                _keyCacheFp16![layer][new[] { sourceBatch, h, s, d }];
                            _valueCacheFp16![layer][new[] { destBatch, h, s, d }] =
                                _valueCacheFp16![layer][new[] { sourceBatch, h, s, d }];
                        }
                        else
                        {
                            _keyCache[layer][new[] { destBatch, h, s, d }] =
                                _keyCache[layer][new[] { sourceBatch, h, s, d }];
                            _valueCache[layer][new[] { destBatch, h, s, d }] =
                                _valueCache[layer][new[] { sourceBatch, h, s, d }];
                        }
                    }
                }
            }

            _sequenceLengths[layer][destBatch] = seqLen;
        }
    }

    private void EnsureInt8Scales(int layerIndex, Tensor<T> newKeys, Tensor<T> newValues, int batchSize, int newSeqLen)
    {
        if (!_useInt8Storage)
        {
            return;
        }

        float maxAbsK = 0f;
        float maxAbsV = 0f;

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _config.NumHeads; h++)
            {
                for (int s = 0; s < newSeqLen; s++)
                {
                    for (int d = 0; d < _config.HeadDimension; d++)
                    {
                        float k = _toFloat!(newKeys[new[] { b, h, s, d }]);
                        float v = _toFloat!(newValues[new[] { b, h, s, d }]);
                        float ak = Math.Abs(k);
                        float av = Math.Abs(v);
                        if (ak > maxAbsK) maxAbsK = ak;
                        if (av > maxAbsV) maxAbsV = av;
                    }
                }
            }
        }

        EnsureInt8ScaleForLayer(layerIndex, isKey: true, maxAbs: maxAbsK);
        EnsureInt8ScaleForLayer(layerIndex, isKey: false, maxAbs: maxAbsV);
    }

    private void EnsureInt8ScaleForLayer(int layerIndex, bool isKey, float maxAbs)
    {
        float requiredScale = maxAbs > 0f ? (maxAbs / 127f) : 1f;
        if (requiredScale <= 0f) requiredScale = 1f;

        float currentScale = isKey ? _keyScaleInt8![layerIndex] : _valueScaleInt8![layerIndex];

        if (currentScale <= 0f)
        {
            if (isKey) _keyScaleInt8![layerIndex] = requiredScale;
            else _valueScaleInt8![layerIndex] = requiredScale;
            return;
        }

        if (requiredScale > currentScale)
        {
            RescaleInt8Layer(layerIndex, isKey, currentScale, requiredScale);
            if (isKey) _keyScaleInt8![layerIndex] = requiredScale;
            else _valueScaleInt8![layerIndex] = requiredScale;
        }
    }

    private void RescaleInt8Layer(int layerIndex, bool isKey, float oldScale, float newScale)
    {
        if (oldScale <= 0f || newScale <= 0f || Math.Abs(newScale - oldScale) < float.Epsilon)
        {
            return;
        }

        var cache = isKey ? _keyCacheInt8![layerIndex] : _valueCacheInt8![layerIndex];

        for (int b = 0; b < _sequenceLengths[layerIndex].Length; b++)
        {
            int seqLen = _sequenceLengths[layerIndex][b];
            for (int h = 0; h < _config.NumHeads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < _config.HeadDimension; d++)
                    {
                        sbyte q = cache[new[] { b, h, s, d }];
                        float value = q * oldScale;
                        cache[new[] { b, h, s, d }] = QuantizeToInt8(value, newScale);
                    }
                }
            }
        }
    }

    private static sbyte QuantizeToInt8(float value, float scale)
    {
        if (scale <= 0f) scale = 1f;
        int q = (int)Math.Round(value / scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        return (sbyte)q;
    }

    private static float DequantizeInt8(sbyte value, float scale)
    {
        if (scale <= 0f) scale = 1f;
        return value * scale;
    }

    private void ValidateLayerIndex(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= _config.NumLayers)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index {layerIndex} is out of range [0, {_config.NumLayers})");
        }
    }

    private void ValidateInputShapes(Tensor<T> keys, Tensor<T> values)
    {
        if (keys.Shape.Length != 4 || values.Shape.Length != 4)
        {
            throw new ArgumentException("Keys and values must be 4D tensors [batch, heads, seq, dim]");
        }

        if (keys.Shape[0] != values.Shape[0] ||
            keys.Shape[1] != values.Shape[1] ||
            keys.Shape[2] != values.Shape[2] ||
            keys.Shape[3] != values.Shape[3])
        {
            throw new ArgumentException("Keys and values must have matching shapes");
        }

        if (keys.Shape[1] != _config.NumHeads)
        {
            throw new ArgumentException(
                $"Number of heads mismatch: expected {_config.NumHeads}, got {keys.Shape[1]}");
        }

        if (keys.Shape[3] != _config.HeadDimension)
        {
            throw new ArgumentException(
                $"Head dimension mismatch: expected {_config.HeadDimension}, got {keys.Shape[3]}");
        }
    }

    private void EnsureCacheAllocated(int layerIndex)
    {
        if (!IsLayerAllocated(layerIndex))
        {
            var shape = new[]
            {
                _config.MaxBatchSize,
                _config.NumHeads,
                _config.MaxSequenceLength,
                _config.HeadDimension
            };

            if (_useFp16Storage)
            {
                _keyCacheFp16![layerIndex] = new Tensor<Half>(shape);
                _valueCacheFp16![layerIndex] = new Tensor<Half>(shape);
            }
            else if (_useInt8Storage)
            {
                _keyCacheInt8![layerIndex] = new Tensor<sbyte>(shape);
                _valueCacheInt8![layerIndex] = new Tensor<sbyte>(shape);
                _keyScaleInt8![layerIndex] = 0f;
                _valueScaleInt8![layerIndex] = 0f;
            }
            else
            {
                _keyCache[layerIndex] = new Tensor<T>(shape);
                _valueCache[layerIndex] = new Tensor<T>(shape);
            }
        }
    }

    private void HandleSlidingWindowEviction(int layerIndex, int batchSize, int newSeqLen)
    {
        for (int b = 0; b < batchSize; b++)
        {
            int currentLen = _sequenceLengths[layerIndex][b];
            int newLen = currentLen + newSeqLen;

            if (newLen > _config.WindowSize)
            {
                int evictCount = newLen - _config.WindowSize;

                // Shift cache entries
                int keepCount = currentLen - evictCount;
                if (keepCount > 0)
                {
                    for (int h = 0; h < _config.NumHeads; h++)
                    {
                        for (int s = 0; s < keepCount; s++)
                        {
                            int srcPos = evictCount + s;
                            for (int d = 0; d < _config.HeadDimension; d++)
                            {
                                if (_useInt8Storage)
                                {
                                    _keyCacheInt8![layerIndex][new[] { b, h, s, d }] =
                                        _keyCacheInt8![layerIndex][new[] { b, h, srcPos, d }];
                                    _valueCacheInt8![layerIndex][new[] { b, h, s, d }] =
                                        _valueCacheInt8![layerIndex][new[] { b, h, srcPos, d }];
                                }
                                else if (_useFp16Storage)
                                {
                                    _keyCacheFp16![layerIndex][new[] { b, h, s, d }] =
                                        _keyCacheFp16![layerIndex][new[] { b, h, srcPos, d }];
                                    _valueCacheFp16![layerIndex][new[] { b, h, s, d }] =
                                        _valueCacheFp16![layerIndex][new[] { b, h, srcPos, d }];
                                }
                                else
                                {
                                    _keyCache[layerIndex][new[] { b, h, s, d }] =
                                        _keyCache[layerIndex][new[] { b, h, srcPos, d }];
                                    _valueCache[layerIndex][new[] { b, h, s, d }] =
                                        _valueCache[layerIndex][new[] { b, h, srcPos, d }];
                                }
                            }
                        }
                    }
                }

                _sequenceLengths[layerIndex][b] = keepCount;
                _evictions += evictCount;
            }
        }
    }

    private bool IsLayerAllocated(int layerIndex)
    {
        if (_useInt8Storage)
            return _keyCacheInt8![layerIndex] != null;

        return _useFp16Storage ? _keyCacheFp16![layerIndex] != null : _keyCache[layerIndex] != null;
    }
}
