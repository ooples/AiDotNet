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
    // Exclusive per-generator ownership bounds inference to five single-example activation
    // buffers. Stream independent examples so batch growth does not retain batch-sized scratch.
    // Concurrent/reentrant calls retain the ordinary execution path.
    private float[][]? _privateCpuBuffers;
    private int _privateCpuBusy;
    private long _privateCpuCalls;
    internal long PrivateCpuInferenceCalls => Interlocked.Read(ref _privateCpuCalls);

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
        int fullBatch = features.Shape[0], batch = 1, time = features.Shape[2];
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
        try
        {
            var buffers = _privateCpuBuffers ??= new float[5][];
            for (int i = 0; i < buffers.Length; i++)
                if (buffers[i] is null || buffers[i].Length < maximum)
                    buffers[i] = CpuStorageAllocationDiagnostics.NewNumericArray<float>((int)maximum, "workspace.hifi_cpu_private_activation");
            Interlocked.Increment(ref _privateCpuCalls);
            var cpu = _engine.Cast<CpuEngine>();
            Tensor<T> View(int slot, int channels, int length, bool nchw)
            {
                int count = checked(batch * channels * length);
                return Tensor<T>.FromMemory(((T[])(object)buffers[slot]).AsMemory(0, count),
                    nchw ? new[] { batch, channels, 1, length } : new[] { batch, channels, length });
            }
            Tensor<T> Conv(NormedConv1DLayer<T> layer, Tensor<T> input, int slot, Tensor<T>? residual = null, bool activate = false)
            {
                if (!layer.TryPrivateCpuOutputTime(input.Shape[2], out int length)) throw new InvalidOperationException("Private CPU output geometry changed.");
                using var target = View(slot, layer.OutputChannels, length, true);
                layer.ForwardPrivateCpuInto(input, target, residual, activate,
                    MathHelper.GetNumericOperations<T>().FromDouble(Slope));
                return _engine.Reshape(target, new[] { batch, layer.OutputChannels, length });
            }
            Tensor<T> Activate(Tensor<T> input, int slot, double slope)
            {
                var target = View(slot, input.Shape[1], input.Shape[2], false);
                cpu.LeakyReLUInto(target, input, MathHelper.GetNumericOperations<T>().FromDouble(slope));
                return target;
            }
            // Predict performs the single durable public copy. Keep this complete result
            // arena-owned while each example reuses the same private activation storage.
            var complete = TensorAllocator.RentUninitialized<T>(new[] { fullBatch, _post.OutputChannels, outputTime });
            using var sourceKeep = new KeepAliveScope(features);
            using var destinationKeep = new KeepAliveScope(complete);
            var source = features.AsSpanUnmarked();
            var destination = complete.AsWritableSpanUnmarked();
            for (int example = 0; example < fullBatch; example++)
            {
                source.Slice(example * inputCount, inputCount).CopyTo(((T[])(object)buffers[4]).AsSpan(0, inputCount));
                using var sample = View(4, features.Shape[1], features.Shape[2], false);
                var x = Conv(_pre, sample, 0);
                for (int stage = 0; stage < _ups.Count; stage++)
                {
                    using var activatedInput = Activate(x, 1, Slope);
                    x = Conv(_ups[stage], activatedInput, 0);
                    Tensor<T>? sum = null;
                    foreach (var (first, second) in _blocks[stage])
                    {
                        var h = x;
                        int residualSlot = 0;
                        for (int pair = 0; pair < first.Count; pair++)
                        {
                            int nextSlot = residualSlot == 1 ? 2 : 1;
                            using var activatedResidual = Activate(h, nextSlot, Slope);
                            using var middle = Conv(first[pair], activatedResidual, 3, activate: true);
                            h = Conv(second[pair], middle, nextSlot, h);
                            residualSlot = nextSlot;
                        }
                        if (sum is null)
                        {
                            sum = View(4, h.Shape[1], h.Shape[2], false);
                            h.AsSpanUnmarked().CopyTo(sum.AsWritableSpanUnmarked());
                        }
                        else cpu.TensorAddInPlace(sum, h);
                    }
                    if (sum is null) throw new InvalidOperationException("Private CPU residual stage is empty.");
                    cpu.TensorMultiplyScalarInPlace(sum, _residualAverageScale);
                    x = sum;
                }
                using var finalActivation = Activate(x, 1, 0.01);
                if (!_post.TryPrivateCpuOutputTime(finalActivation.Shape[2], out int finalTime)) throw new InvalidOperationException("Private CPU post geometry changed.");
                // Slot0's stage input is dead after finalActivation; it can hold this example's
                // waveform without another backing array or a view into a public output buffer.
                using var final4 = View(0, _post.OutputChannels, finalTime, true);
                _post.ForwardPrivateCpuInto(finalActivation, final4, null, false, MathHelper.GetNumericOperations<T>().Zero);
                if (_tanhOutput) cpu.TanhInto(final4, final4);
                final4.AsSpanUnmarked().CopyTo(destination.Slice(checked(example * (int)sampleOutputCount), (int)sampleOutputCount));
            }
            output = complete;
            return true;
        }
        finally { Volatile.Write(ref _privateCpuBusy, 0); }
    }
}
