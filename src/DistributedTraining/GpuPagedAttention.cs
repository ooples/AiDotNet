using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Thin wrapper over the Tensors GPU paged-attention kernels for ONE (rank, layer) head-group: it owns a
/// <see cref="DevicePagedKVCache"/> (device-resident KV), appends each step's K/V, and runs scaled-dot-product
/// attention on the GPU for a decode step (single query) or a prefill chunk (causal, multiple queries), returning
/// the attention context. Used by <see cref="TensorParallelPagedModel{T}"/>'s GPU execution mode so each rank's
/// head-group attention runs on the device.
/// </summary>
/// <remarks>
/// Tensors 0.115.1 exposes <c>PagedAttentionDecode/Prefill</c> on the concrete backends but NOT via a shared
/// interface (the <c>IPagedAttentionBackend</c> capability of #806 is not in this release), so the kernels are
/// invoked by cached reflection off the resolved backend. The kernels RETURN the output buffer (the query buffer
/// is input-only). Attention runs in FP32 (the GPU kernels' precision), so results match a double CPU model only
/// within float tolerance — the industry-standard serving precision.
/// </remarks>
internal sealed class GpuPagedAttention : IDisposable
{
    private static MethodInfo? _decode;
    private static MethodInfo? _prefill;
    private static Type? _resolvedBackendType;

    private readonly IDirectGpuBackend _backend;
    private readonly DevicePagedKVCache _cache;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _dim;
    private readonly int _blockSize;
    private readonly float _scale;

    private GpuPagedAttention(IDirectGpuBackend backend, int heads, int headDim, int blockSize, int maxBlocks)
    {
        _backend = backend;
        _heads = heads;
        _headDim = headDim;
        _dim = heads * headDim;
        _blockSize = blockSize;
        _scale = 1.0f / (float)Math.Sqrt(headDim);
        _cache = new DevicePagedKVCache(backend, maxBlocks, blockSize, heads, headDim);
    }

    /// <summary>Whether a GPU engine with paged-attention kernels is active in this process.</summary>
    public static bool IsAvailable => TryGetBackend(out _);

    /// <summary>Creates a GPU paged-attention head-group, or null when no compatible GPU backend is active.</summary>
    public static GpuPagedAttention? TryCreate(int heads, int headDim, int blockSize, int maxBlocks)
    {
        if (!TryGetBackend(out var backend) || backend is null) return null;
        try
        {
            return new GpuPagedAttention(backend, heads, headDim, blockSize, maxBlocks);
        }
        catch
        {
            return null;
        }
    }

    private static bool TryGetBackend(out IDirectGpuBackend? backend)
    {
        backend = null;
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu || !gpu.IsGpuAvailable) return false;
        var b = gpu.GetBackend();
        if (b is null) return false;
        var type = b.GetType();
        if (!ReferenceEquals(type, _resolvedBackendType))
        {
            _decode = type.GetMethod("PagedAttentionDecode");
            _prefill = type.GetMethod("PagedAttentionPrefill");
            _resolvedBackendType = type;
        }
        if (_decode is null || _prefill is null) return false;
        backend = b;
        return true;
    }

    /// <summary>Appends one token's K/V (each <c>heads*headDim</c>) for the sequence at its next position.</summary>
    public void Append(int seqId, float[] key, float[] value) => _cache.Append(seqId, key, value);

    /// <summary>Frees this head-group's device KV for the sequence.</summary>
    public void Free(int seqId) => _cache.Free(seqId);

    /// <summary>
    /// Decode attention for a single query (<c>heads*headDim</c>) over the sequence's cached K/V (length
    /// <paramref name="seqLen"/>). Returns the context (<c>heads*headDim</c>).
    /// </summary>
    public float[] Decode(int seqId, float[] query, int seqLen)
    {
        var qBuf = _backend.AllocateBuffer(query);
        var ret = _decode!.Invoke(_backend, new object[]
        {
            qBuf, _cache.KeyBlocks, _cache.ValueBlocks, _cache.GetBlockTableBuffer(seqId),
            _heads, _headDim, _blockSize, seqLen, _scale
        });
        var outBuf = ret as IGpuBuffer ?? qBuf;
        return _backend.DownloadBuffer(outBuf);
    }

    /// <summary>
    /// Prefill / multi-query causal attention: <paramref name="numQueries"/> queries laid out
    /// <c>[numQueries, heads, headDim]</c> at logical positions <paramref name="startPos"/>..startPos+numQueries-1.
    /// Returns the context <c>[numQueries, heads, headDim]</c>.
    /// </summary>
    public float[] Prefill(int seqId, float[] queries, int numQueries, int startPos)
    {
        var qBuf = _backend.AllocateBuffer(queries);
        var ret = _prefill!.Invoke(_backend, new object[]
        {
            qBuf, _cache.KeyBlocks, _cache.ValueBlocks, _cache.GetBlockTableBuffer(seqId),
            _heads, _headDim, _blockSize, numQueries, startPos, _scale
        });
        var outBuf = ret as IGpuBuffer ?? qBuf;
        return _backend.DownloadBuffer(outBuf);
    }

    public void Dispose()
    {
        if (_cache is IDisposable d) d.Dispose();
    }
}
