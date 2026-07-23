using System;
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
/// The kernels are invoked through the public <see cref="IDirectGpuBackend"/> interface
/// (<c>PagedAttentionDecode/Prefill</c>, exposed on the interface as of AiDotNet.Tensors 0.116.0) and RETURN the
/// output buffer (the query buffer is input-only). Attention runs in FP32 (the GPU kernels' precision), so results
/// match a double CPU model only within float tolerance — the industry-standard serving precision.
/// </remarks>
internal sealed class GpuPagedAttention : IDisposable
{
    private readonly IDirectGpuBackend _backend;
    private readonly DevicePagedKVCache _cache;
    private readonly int _heads;    // query heads
    private readonly int _kvHeads;  // KV heads (== query heads for plain MHA; < for grouped-query attention)
    private readonly int _headDim;
    private readonly int _blockSize;
    private readonly float _scale;
    private readonly bool _gqa;

    private GpuPagedAttention(IDirectGpuBackend backend, int heads, int kvHeads, int headDim, int blockSize, int maxBlocks)
    {
        _backend = backend;
        _heads = heads;
        _kvHeads = kvHeads;
        _headDim = headDim;
        _blockSize = blockSize;
        _scale = 1.0f / (float)Math.Sqrt(headDim);
        _gqa = kvHeads != heads;
        // The device cache stores the KV heads' K/V.
        _cache = new DevicePagedKVCache(backend, maxBlocks, blockSize, kvHeads, headDim);
    }

    /// <summary>Whether a GPU engine with paged-attention kernels is active in this process.</summary>
    public static bool IsAvailable => TryGetBackend(out _);

    /// <summary>The number of GPU devices available for multi-GPU tensor-parallel placement (0 when none).</summary>
    public static int DeviceCount
    {
        get
        {
            try { return AiDotNet.Tensors.Engines.DirectGpu.DirectGpuBackendFactory.GetAvailableGpus().Length; }
            catch { return 0; }
        }
    }

    /// <summary>
    /// Creates a fresh GPU backend bound to <paramref name="deviceIndex"/> (its own context) so a tensor-parallel
    /// rank can run on a distinct device. Returns null when no GPU is available. The caller owns/disposes it.
    /// </summary>
    public static IDirectGpuBackend? CreateDeviceBackend(int deviceIndex)
    {
        try
        {
            var b = AiDotNet.Tensors.Engines.DirectGpu.DirectGpuBackendFactory.CreateNew(
                AiDotNet.Tensors.Engines.DirectGpu.GpuBackendPreference.Auto, deviceIndex);
            return (b is not null && b.IsAvailable) ? b : null;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Creates a GPU paged-attention head-group, or null when no compatible GPU backend is active.</summary>
    public static GpuPagedAttention? TryCreate(int heads, int kvHeads, int headDim, int blockSize, int maxBlocks)
    {
        if (!TryGetBackend(out var backend) || backend is null) return null;
        return Create(backend, heads, kvHeads, headDim, blockSize, maxBlocks);
    }

    /// <summary>Creates a GPU paged-attention head-group on an explicit (per-device) backend.</summary>
    public static GpuPagedAttention? Create(IDirectGpuBackend backend, int heads, int kvHeads, int headDim, int blockSize, int maxBlocks)
    {
        if (backend is null) return null;
        try
        {
            return new GpuPagedAttention(backend, heads, kvHeads, headDim, blockSize, maxBlocks);
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
        IGpuBuffer? outBuf = null;
        try
        {
            outBuf = _gqa
                ? _backend.PagedAttentionDecodeGqa(
                    qBuf, _cache.KeyBlocks, _cache.ValueBlocks, _cache.GetBlockTableBuffer(seqId),
                    _heads, _kvHeads, _headDim, _blockSize, seqLen, _scale)
                : _backend.PagedAttentionDecode(
                    qBuf, _cache.KeyBlocks, _cache.ValueBlocks, _cache.GetBlockTableBuffer(seqId),
                    _heads, _headDim, _blockSize, seqLen, _scale);
            return _backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf.Dispose();
            if (outBuf is not null && !ReferenceEquals(outBuf, qBuf)) outBuf.Dispose();
        }
    }

    /// <summary>
    /// Prefill / multi-query causal attention: <paramref name="numQueries"/> queries laid out
    /// <c>[numQueries, heads, headDim]</c> at logical positions <paramref name="startPos"/>..startPos+numQueries-1.
    /// Returns the context <c>[numQueries, heads, headDim]</c>.
    /// </summary>
    public float[] Prefill(int seqId, float[] queries, int numQueries, int startPos)
    {
        var qBuf = _backend.AllocateBuffer(queries);
        IGpuBuffer? outBuf = null;
        try
        {
            outBuf = _gqa
                ? _backend.PagedAttentionPrefillGqa(
                    qBuf, _cache.KeyBlocks, _cache.ValueBlocks, _cache.GetBlockTableBuffer(seqId),
                    _heads, _kvHeads, _headDim, _blockSize, numQueries, startPos, _scale)
                : _backend.PagedAttentionPrefill(
                    qBuf, _cache.KeyBlocks, _cache.ValueBlocks, _cache.GetBlockTableBuffer(seqId),
                    _heads, _headDim, _blockSize, numQueries, startPos, _scale);
            return _backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf.Dispose();
            if (outBuf is not null && !ReferenceEquals(outBuf, qBuf)) outBuf.Dispose();
        }
    }

    public void Dispose()
    {
        if (_cache is IDisposable d) d.Dispose();
    }
}
