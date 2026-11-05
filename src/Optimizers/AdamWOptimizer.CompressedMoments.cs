using System.Collections.Concurrent;
using AiDotNet.Enums;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Optimizers;

/// <summary>
/// AdamW with compressed moments (<see cref="AdamWOptimizerOptions{T, TInput, TOutput}.MomentStorage"/>): bfloat16 or
/// 8-bit block-wise moments, the same state on the device and the host.
/// </summary>
/// <remarks>
/// <para>Each parameter's moments live in one <see cref="CompressedMomentSlot"/>, either as device buffers (the GPU-resident
/// step, <see cref="GpuOptimizer.TryAdamBf16Step"/> / <c>GpuOptimizer.TryAdam8BitStep</c>) or as host arrays (the host
/// step, <see cref="HostCompressedMomentAdam"/>). Both use one byte layout, so a slot moves between them by upload or
/// download, and the two steps produce the same bits.</para>
/// <para>Checkpoints reuse the tape-state serializer, packing and releasing one tensor at a time as exact float
/// values (bf16 widens exactly; codebook indices 0-255 are exact; block scales occupy the leading entries).
/// A versioned header validates format-affecting options before restored state is committed.</para>
/// </remarks>
public partial class AdamWOptimizer<T, TInput, TOutput>
{
    // Persisted form of the compressed moments (filled only while serializing, or by a restore). Their field type is
    // what the base tape-state serializer persists by parameter index.
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeCompressedM = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeCompressedV = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeCompressedScales = new(TensorReferenceComparer<Tensor<T>>.Instance);

    // Live compressed state, one slot per parameter.
    private readonly ConcurrentDictionary<Tensor<T>, CompressedMomentSlot> _compressedSlots = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private sealed class CompressedMomentSlot
    {
        public CompressedMomentSlot(int length, bool int8, int blockSize)
        {
            Length = length;
            Int8 = int8;
            BlockSize = blockSize;
            Blocks = int8 ? HostCompressedMomentAdam.BlockCount(length, blockSize) : 0;
            int momentBytes = int8 ? length : checked(length * 2);
            M = new byte[momentBytes];
            V = new byte[momentBytes];
            MScales = new float[Blocks];
            VScales = new float[Blocks];
        }

        public int Length { get; }
        public bool Int8 { get; }
        public int BlockSize { get; }
        public int Blocks { get; }
        public int MomentBytes => Int8 ? Length : Length * 2;

        // Host state: authoritative while DeviceM is null.
        public byte[] M;
        public byte[] V;
        public float[] MScales;
        public float[] VScales;

        // Device state: authoritative while non-null.
        public IGpuBuffer? DeviceM, DeviceV, DeviceMScales, DeviceVScales;

        public bool OnDevice => DeviceM is not null;

        /// <summary>Copies the device state into the host arrays and frees the device buffers.</summary>
        public void MoveToHost()
        {
            if (!OnDevice) return;
            if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu || gpu.GetBackend() is not { } backend)
                throw new InvalidOperationException(
                    "AdamW's compressed moments are on a GPU device, but the active engine is no longer that GPU engine; " +
                    "they cannot be read back.");
            M = backend.DownloadByteBuffer(DeviceM!, MomentBytes);
            V = backend.DownloadByteBuffer(DeviceV!, MomentBytes);
            if (Int8)
            {
                MScales = Exact(backend.DownloadBuffer(DeviceMScales!), Blocks);
                VScales = Exact(backend.DownloadBuffer(DeviceVScales!), Blocks);
            }
            FreeDevice();
        }

        /// <summary>Puts the host state on the active GPU engine's device. False when it has no compressed-moment kernels.</summary>
        public bool TryMoveToDevice()
        {
            if (OnDevice) return true;
            if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu || gpu.GetBackend() is not { } backend) return false;
            if (Int8)
            {
                if (!GpuOptimizer.TryAllocAdam8BitState(Length, BlockSize, out var mq, out var vq, out var ms, out var vs))
                    return false;
                DeviceM = mq; DeviceV = vq; DeviceMScales = ms; DeviceVScales = vs;
                backend.UploadByteBuffer(DeviceM!, M);
                backend.UploadByteBuffer(DeviceV!, V);
                if (Blocks > 0)
                {
                    GpuOptimizer.FreeGpuBuffer(DeviceMScales); GpuOptimizer.FreeGpuBuffer(DeviceVScales);
                    DeviceMScales = backend.AllocateBuffer((float[])MScales.Clone());
                    DeviceVScales = backend.AllocateBuffer((float[])VScales.Clone());
                }
            }
            else
            {
                if (!GpuOptimizer.TryAllocBf16MomentState(Length, out var m, out var v)) return false;
                DeviceM = m; DeviceV = v;
                backend.UploadByteBuffer(DeviceM!, M);
                backend.UploadByteBuffer(DeviceV!, V);
            }
            return true;
        }

        public void FreeDevice()
        {
            GpuOptimizer.FreeGpuBuffer(DeviceM); GpuOptimizer.FreeGpuBuffer(DeviceV);
            GpuOptimizer.FreeGpuBuffer(DeviceMScales); GpuOptimizer.FreeGpuBuffer(DeviceVScales);
            DeviceM = DeviceV = DeviceMScales = DeviceVScales = null;
        }

        private static float[] Exact(float[] values, int count)
        {
            if (values.Length == count) return values;
            var exact = new float[count];
            Array.Copy(values, exact, count);
            return exact;
        }
    }

    /// <summary>
    /// Whether <paramref name="param"/>'s moments are compressed: any length for bf16; for 8-bit, tensors at least
    /// <see cref="AdamWOptimizerOptions{T, TInput, TOutput}.Int8MinQuantizedLength"/> long (shorter ones keep fp32).
    /// </summary>
    private bool UsesCompressedMoments(Tensor<T> param)
    {
        var storage = _options.MomentStorage;
        if (storage == OptimizerMomentStorage.Float32) return false;
        if (typeof(T) != typeof(float))
            throw new NotSupportedException(
                $"AdamW {storage} moments are implemented for float models; this model uses {typeof(T).Name}.");
        if (_options.UseAMSGrad)
            throw new NotSupportedException($"AdamW {storage} moments do not support AMSGrad (its running max is a third moment).");
        if (_options.WeightDecayMask is not null)
            throw new NotSupportedException($"AdamW {storage} moments do not support a per-element weight-decay mask.");
        if (storage == OptimizerMomentStorage.Int8Blockwise)
        {
            if (_options.Int8MomentBlockSize <= 0)
                throw new InvalidOperationException($"Int8MomentBlockSize must be positive (was {_options.Int8MomentBlockSize}).");
            // The persisted form keeps both moments' block scales in one parameter-length tensor.
            int blocks = HostCompressedMomentAdam.BlockCount(param.Length, _options.Int8MomentBlockSize);
            return param.Length >= Math.Max(1, _options.Int8MinQuantizedLength) && 2 * blocks <= param.Length;
        }
        return true;
    }

    /// <summary>
    /// One compressed-moment AdamW step for <paramref name="param"/>: on the device when the engine is a GPU engine and
    /// the tensors are resident there, otherwise on the host with the same state.
    /// </summary>
    private void CompressedMomentStep(Tensor<T> param, Tensor<T> grad, bool gpu)
    {
        var slot = GetOrCreateCompressedSlot(param);
        float lr = (float)NumOps.ToDouble(CurrentLearningRate);
        float b1 = (float)_options.Beta1, b2 = (float)_options.Beta2, eps = (float)_options.Epsilon, wd = (float)_options.WeightDecay;
        var p = (Tensor<float>)(object)param;
        var g = (Tensor<float>)(object)grad;

        if (gpu && p.HasPendingGpuData && g.HasPendingGpuData && slot.TryMoveToDevice())
        {
            bool done = slot.Int8
                ? GpuOptimizer.TryAdam8BitStep(p, g, slot.DeviceM, slot.DeviceV, slot.DeviceMScales, slot.DeviceVScales,
                    lr, b1, b2, eps, 1f - (float)Math.Pow(b1, _tapeStep), 1f - (float)Math.Pow(b2, _tapeStep), slot.BlockSize,
                    wd, decoupledWeightDecay: true)
                : GpuOptimizer.TryAdamBf16Step(p, g, slot.DeviceM, slot.DeviceV, lr, b1, b2, eps, wd, _tapeStep,
                    decoupledWeightDecay: true);
            if (done) return;
        }

        // Host step on the same state (a tensor that is not device-resident, or a CPU engine).
        slot.MoveToHost();
        var pHost = p.ToArray();
        var gHost = g.ToArray();
        if (slot.Int8)
            HostCompressedMomentAdam.Int8Step(pHost, gHost, slot.M, slot.V, slot.MScales, slot.VScales, slot.Length,
                slot.BlockSize, lr, b1, b2, eps, wd, _tapeStep, decoupledWeightDecay: true);
        else
            HostCompressedMomentAdam.Bf16Step(pHost, gHost, slot.M, slot.V, slot.Length, lr, b1, b2, eps, wd, _tapeStep,
                decoupledWeightDecay: true);
        pHost.AsSpan().CopyTo(p.AsWritableSpan());
    }

    private CompressedMomentSlot GetOrCreateCompressedSlot(Tensor<T> param)
    {
        bool int8 = _options.MomentStorage == OptimizerMomentStorage.Int8Blockwise;
        int blockSize = int8 ? _options.Int8MomentBlockSize : 0;
        // A restored checkpoint left its state in the persisted dictionaries: it replaces whatever this slot held.
        if (_tapeCompressedM.TryRemove(param, out var restoredM))
        {
            _tapeCompressedV.TryRemove(param, out var restoredV);
            _tapeCompressedScales.TryRemove(param, out var restoredScales);
            if (_compressedSlots.TryRemove(param, out var stale)) stale.FreeDevice();
            var restored = new CompressedMomentSlot(param.Length, int8, blockSize);
            Unpack(restored, restoredM, restoredV ?? throw MissingState("second moment"),
                int8 ? restoredScales ?? throw MissingState("block scales") : null);
            _compressedSlots[param] = restored;
            return restored;
        }

        if (_compressedSlots.TryGetValue(param, out var slot))
        {
            if (slot.Int8 != int8 || slot.BlockSize != blockSize || slot.Length != param.Length)
                throw new InvalidOperationException(
                    "AdamW's MomentStorage or Int8MomentBlockSize changed after training started; Reset() the optimizer first.");
            return slot;
        }

        if (_tapeM.ContainsKey(param))
            throw new InvalidOperationException(
                $"This AdamW optimizer holds 32-bit moments for a parameter now trained with {_options.MomentStorage} moments " +
                "(a checkpoint saved with different MomentStorage, or the option changed mid-run). Reset() the optimizer or " +
                "restore with the MomentStorage the checkpoint was saved with.");
        slot = new CompressedMomentSlot(param.Length, int8, blockSize);
        _compressedSlots[param] = slot;
        return slot;

        static InvalidOperationException MissingState(string what) => new(
            $"The checkpoint's AdamW compressed-moment state is missing its {what}.");
    }

    /// <summary>Packs one checkpoint tensor on demand, avoiding full-model widened moment copies.</summary>
    private Tensor<T> PackCompressedSlotTensor(Tensor<T> param, CompressedMomentSlot slot, string fieldName)
    {
        slot.MoveToHost();
        var packed = new Tensor<T>(param._shape);
        try
        {
            var values = ((Tensor<float>)(object)packed).AsWritableSpan();
            if (fieldName == nameof(_tapeCompressedScales))
            {
                slot.MScales.AsSpan().CopyTo(values);
                slot.VScales.AsSpan().CopyTo(values.Slice(slot.Blocks));
            }
            else
            {
                var bytes = fieldName == nameof(_tapeCompressedM) ? slot.M : slot.V;
                for (int i = 0; i < slot.Length; i++)
                    values[i] = slot.Int8 ? bytes[i] : Bf16ToFloat(bytes, i);
            }
            return packed;
        }
        catch { packed.Dispose(); throw; }
    }

    private protected override IEnumerable<(Tensor<T> Parameter, Func<Tensor<T>> Create)> TransientTapeTensorStates(string fieldName)
    {
        if (fieldName != nameof(_tapeCompressedM) && fieldName != nameof(_tapeCompressedV)
            && fieldName != nameof(_tapeCompressedScales)) yield break;
        foreach (var entry in _compressedSlots.Where(entry => fieldName != nameof(_tapeCompressedScales) || entry.Value.Int8))
        {
            yield return (entry.Key, () => PackCompressedSlotTensor(entry.Key, entry.Value, fieldName));
        }
    }


    private void ClearPackedCompressedState()
    {
        _tapeCompressedM.Clear();
        _tapeCompressedV.Clear();
        _tapeCompressedScales.Clear();
    }

    private static void Unpack(CompressedMomentSlot slot, Tensor<T> m, Tensor<T> v, Tensor<T>? scales)
    {
        var ms = ((Tensor<float>)(object)m).AsSpan();
        var vs = ((Tensor<float>)(object)v).AsSpan();
        for (int i = 0; i < slot.Length; i++)
        {
            if (slot.Int8)
            {
                slot.M[i] = ToByte(ms[i]);
                slot.V[i] = ToByte(vs[i]);
            }
            else
            {
                FloatToBf16(ms[i], slot.M, i);
                FloatToBf16(vs[i], slot.V, i);
            }
        }
        if (slot.Int8)
        {
            var sc = ((Tensor<float>)(object)scales!).AsSpan();
            sc.Slice(0, slot.Blocks).CopyTo(slot.MScales);
            sc.Slice(slot.Blocks, slot.Blocks).CopyTo(slot.VScales);
        }

        static byte ToByte(float value)
        {
            if (!(value >= 0f && value <= 255f) || !Math.Floor(value).Equals((double)value))
                throw new InvalidOperationException($"Corrupt AdamW 8-bit moment state: {value} is not a codebook index.");
            return (byte)value;
        }
    }

    // bfloat16 is the top 16 bits of a float, so the widened value is exact and narrowing it back is lossless.
    private static float Bf16ToFloat(byte[] bytes, int i)
        => new AdamWMomentFloatIntBits { Int32 = (bytes[2 * i] | (bytes[2 * i + 1] << 8)) << 16 }.Single;

    private static void FloatToBf16(float value, byte[] bytes, int i)
    {
        int bits = new AdamWMomentFloatIntBits { Single = value }.Int32;
        if ((bits & 0xFFFF) != 0)
            throw new InvalidOperationException($"Corrupt AdamW bf16 moment state: {value} is not a bfloat16 value.");
        bytes[2 * i] = (byte)(bits >> 16);
        bytes[2 * i + 1] = (byte)(bits >> 24);
    }

    private void ResetCompressedMoments()
    {
        foreach (var slot in _compressedSlots.Values) slot.FreeDevice();
        _compressedSlots.Clear();
        ClearPackedCompressedState();
    }

    /// <inheritdoc />
    private protected override void SerializeExtensionData(BinaryWriter writer)
    {
        base.SerializeExtensionData(writer);
        AdamWMomentFormatHeader.Write(writer, _options.MomentStorage, _options.Int8MomentBlockSize, _options.Int8MinQuantizedLength);
    }

    /// <inheritdoc />
    private protected override Action StageExtensionData(BinaryReader reader, out bool hasFallibleCommit)
    {
        var applyBaseState = base.StageExtensionData(reader, out hasFallibleCommit);
        AdamWMomentFormatHeader.Validate(reader, _options.MomentStorage, _options.Int8MomentBlockSize, _options.Int8MinQuantizedLength);
        return () =>
        {
            applyBaseState();
            foreach (var slot in _compressedSlots.Values) slot.FreeDevice();
            _compressedSlots.Clear();
            ClearPackedCompressedState();
        };
    }
}

// A non-generic type: explicit layout is invalid even for a struct nested in a generic class.
[System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit)]
internal struct AdamWMomentFloatIntBits
{
    [System.Runtime.InteropServices.FieldOffset(0)] public float Single;
    [System.Runtime.InteropServices.FieldOffset(0)] public int Int32;
}
