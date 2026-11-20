using System.Threading;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TextToSpeech.Vocoders;

internal sealed partial class HiFiGanGenerator<T>
{
    // Reuse five exclusively owned activation slots. Bound batching by bytes so a long
    // waveform cannot retain scratch proportional to the public batch size.
    private const int PrivateCpuBatchSlotElements = 16 * 1024 * 1024 / sizeof(float);
    private readonly int _privateCpuBatchLimit = ReadPrivateCpuBatchLimit();

    private static int ReadPrivateCpuBatchLimit()
    {
        return int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_HIFI_CPU_INFERENCE_BATCH_LIMIT"), out int limit) && limit > 0
            ? limit : int.MaxValue;
    }
    private float[][]? _privateCpuBuffers;
    private int _privateCpuBusy;
    private long _privateCpuCalls;
    internal long PrivateCpuInferenceCalls => Interlocked.Read(ref _privateCpuCalls);

    private readonly struct PrivateCpuActivation(Tensor<T> logical, Tensor<T> nchw, int channels, int time)
    {
        internal Tensor<T> Logical { get; } = logical;
        internal Tensor<T> Nchw { get; } = nchw;
        internal int Channels { get; } = channels;
        internal int Time { get; } = time;
    }

    private bool TryForwardPrivateCpu(Tensor<T> features, Tensor<T>? condition, out Tensor<T> output)
    {
        output = null!;
        if (typeof(T) != typeof(float) || !InferenceArenaSettings.Enabled || !InferenceArenaSettings.RecycleLayerScratch
            || TensorArena.Current is null
            || !_resblockType1 || condition is not null || _outputReflectPad != 0 || _blocks.Any(stage => stage.Count <= 1)
            || !NoGradScope<T>.IsSuppressed || GradientTape<T>.Current is not null || GradientRecording.IsRecording<T>()
            || GraphMode.IsActive || AiDotNet.NeuralNetworks.GraphCaptureCompatibility.IsActive
            || AiDotNet.Tensors.Engines.Gpu.AutocastScope.IsEnabled
            || AiDotNet.NeuralNetworks.Graph.LayerForwardObserver<T>.Current is not null
            || _engine.Unwrap().GetType() != typeof(CpuEngine) || !ReferenceEquals(_engine, _engine.Unwrap())
            || features.Rank != 3 || !features.IsContiguous || features.Shape[0] <= 0 || features.Shape[2] <= 0
            || !_layers.All(layer => layer is NormedConv1DLayer<T> convolution && convolution.SupportsPrivateCpuInference))
            return false;
        int fullBatch = features.Shape[0], time = features.Shape[2];
        int inputCount = checked(features.Shape[1] * time);
        if (!_pre.TryPrivateCpuOutputTime(time, out time)) return false;
        long maximum = Math.Max(inputCount, (long)_pre.OutputChannels * time);
        foreach (var up in _ups)
        {
            if (!up.TryPrivateCpuOutputTime(time, out time)) return false;
            maximum = Math.Max(maximum, (long)up.OutputChannels * time);
        }
        if (!_post.TryPrivateCpuOutputTime(time, out int outputTime)) return false;
        long sampleOutputCount = (long)_post.OutputChannels * outputTime;
        maximum = Math.Max(maximum, sampleOutputCount);
        if (sampleOutputCount <= 0 || sampleOutputCount * fullBatch > int.MaxValue) return false;
        if (maximum <= 0 || maximum > int.MaxValue || Interlocked.CompareExchange(ref _privateCpuBusy, 1, 0) != 0) return false;
        int chunkLimit = Math.Min(fullBatch, _privateCpuBatchLimit);
        chunkLimit = Math.Min(chunkLimit, Math.Max(1, PrivateCpuBatchSlotElements / (int)maximum));
        int slotElements = checked(chunkLimit * (int)maximum);
        int batch = chunkLimit;
        Dictionary<(int Slot, int Batch, int Channels, int Time), PrivateCpuActivation>? views = null;
        try
        {
            var buffers = _privateCpuBuffers ??= new float[5][];
            for (int i = 0; i < buffers.Length; i++)
                if (buffers[i] is null || buffers[i].Length < slotElements)
                    buffers[i] = CpuStorageAllocationDiagnostics.NewNumericArray<float>(slotElements, "workspace.hifi_cpu_private_activation");
            Interlocked.Increment(ref _privateCpuCalls);
            var cpu = _engine.Cast<CpuEngine>();
            var activeViews = new Dictionary<(int Slot, int Batch, int Channels, int Time), PrivateCpuActivation>();
            views = activeViews;
            var bindings = new Dictionary<NormedConv1DLayer<T>, NormedConv1DLayer<T>.CpuInferenceBinding>(AiDotNet.Helpers.TensorReferenceComparer<NormedConv1DLayer<T>>.Instance);
            var numOps = MathHelper.GetNumericOperations<T>();
            T slope = numOps.FromDouble(Slope), finalSlope = numOps.FromDouble(0.01);
            PrivateCpuActivation View(int slot, int channels, int length)
            {
                var key = (slot, batch, channels, length);
                if (activeViews.TryGetValue(key, out var cached)) return cached;
                int count = checked(batch * channels * length);
                var memory = ((T[])(object)buffers[slot]).AsMemory(0, count);
                // Separate borrowed storages keep the paired views from introducing a
                // copy-on-write storage alias. Both remain inside this exclusive call.
                var logical = Tensor<T>.FromMemory(memory, new[] { batch, channels, length });
                Tensor<T>? nchw = null;
                try
                {
                    nchw = Tensor<T>.FromMemory(memory, new[] { batch, channels, 1, length });
                    var value = new PrivateCpuActivation(logical, nchw, channels, length);
                    activeViews.Add(key, value);
                    return value;
                }
                catch { logical.Dispose(); nchw?.Dispose(); throw; }
            }
            PrivateCpuActivation Conv(NormedConv1DLayer<T> layer, PrivateCpuActivation input, int slot,
                PrivateCpuActivation? residual = null, bool activate = false, bool post = false)
            {
                if (!layer.TryPrivateCpuOutputTime(input.Time, out int length)) throw new InvalidOperationException("Private CPU output geometry changed.");
                var target = View(slot, layer.OutputChannels, length);
                var residualTensor = residual?.Logical;
                T alpha = post ? numOps.Zero : slope;
                if (!bindings.TryGetValue(layer, out var binding)
                    || !binding.Matches(input.Logical, input.Nchw, target.Nchw, residualTensor, activate, alpha))
                {
                    binding = layer.BindPrivateCpuInference(input.Logical, input.Nchw, target.Nchw, residualTensor, activate, alpha);
                    bindings[layer] = binding;
                }
                binding.Execute();
                return target;
            }
            PrivateCpuActivation Activate(PrivateCpuActivation input, int slot, T alpha)
            {
                var target = View(slot, input.Channels, input.Time);
                cpu.LeakyReLUInto(target.Logical, input.Logical, alpha);
                return target;
            }
            // Predict performs the single durable public copy. Keep this complete result
            // arena-owned while independent chunks reuse the same private activation storage.
            var complete = TensorAllocator.RentUninitialized<T>(new[] { fullBatch, _post.OutputChannels, outputTime });
            using var sourceKeep = new KeepAliveScope(features);
            using var destinationKeep = new KeepAliveScope(complete);
            var source = features.AsSpanUnmarked();
            var destination = complete.AsWritableSpanUnmarked();
            for (int example = 0; example < fullBatch; example += batch)
            {
                batch = Math.Min(chunkLimit, fullBatch - example);
                int chunkInputs = checked(batch * inputCount);
                source.Slice(example * inputCount, chunkInputs).CopyTo(((T[])(object)buffers[4]).AsSpan(0, chunkInputs));
                var sample = View(4, features.Shape[1], features.Shape[2]);
                var x = Conv(_pre, sample, 0);
                for (int stage = 0; stage < _ups.Count; stage++)
                {
                    var activatedInput = Activate(x, 1, slope);
                    x = Conv(_ups[stage], activatedInput, 0);
                    PrivateCpuActivation? sum = null;
                    foreach (var (first, second) in _blocks[stage])
                    {
                        var h = x;
                        int residualSlot = 0;
                        for (int pair = 0; pair < first.Count; pair++)
                        {
                            int nextSlot = residualSlot == 1 ? 2 : 1;
                            var activatedResidual = Activate(h, nextSlot, slope);
                            var middle = Conv(first[pair], activatedResidual, 3, activate: true);
                            h = Conv(second[pair], middle, nextSlot, h);
                            residualSlot = nextSlot;
                        }
                        if (sum is null)
                        {
                            sum = View(4, h.Channels, h.Time);
                            h.Logical.AsSpanUnmarked().CopyTo(sum.Value.Logical.AsWritableSpanUnmarked());
                        }
                        else cpu.TensorAddInPlace(sum.Value.Logical, h.Logical);
                    }
                    if (sum is null) throw new InvalidOperationException("Private CPU residual stage is empty.");
                    cpu.TensorMultiplyScalarInPlace(sum.Value.Logical, _residualAverageScale);
                    x = sum.Value;
                }
                var finalActivation = Activate(x, 1, finalSlope);
                // The stage input is dead after the final activation. Its slot can hold
                // the chunk's waveforms until they are copied into the public batch.
                var final = Conv(_post, finalActivation, 0, post: true);
                if (_tanhOutput) cpu.TanhInto(final.Nchw, final.Nchw);
                final.Logical.AsSpanUnmarked().CopyTo(destination.Slice(checked(example * (int)sampleOutputCount), checked(batch * (int)sampleOutputCount)));
            }
            output = complete;
            return true;
        }
        finally
        {
            if (views is not null)
                foreach (var value in views.Values) { value.Logical.Dispose(); value.Nchw.Dispose(); }
            Volatile.Write(ref _privateCpuBusy, 0);
        }
    }
}
