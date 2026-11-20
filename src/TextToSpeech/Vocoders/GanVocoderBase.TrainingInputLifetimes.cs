using AiDotNet.Tensors.Engines;

namespace AiDotNet.TextToSpeech.Vocoders;

public abstract partial class GanVocoderBase<T>
{
    private readonly bool _ownedTrainingInputLifetimes =
        Environment.GetEnvironmentVariable("AIDOTNET_GAN_TRAINING_INPUT_LIFETIMES") == "1";
    private Dictionary<string, Tensor<T>>? _trainingInputStages;

    private Tensor<T> StageTrainingInput(string role, Tensor<T> source)
    {
        // A closure plan retains input tensor identities. Keep one owned destination per role and segment
        // geometry, so replacing a crop neither invalidates a plan nor strands its uploaded device buffer.
        string key = role + ":" + string.Join("x", source._shape);
        var stages = _trainingInputStages ??= new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        using var noScope = TensorScope.Suspend();
        if (!stages.TryGetValue(key, out var destination))
        {
            destination = new Tensor<T>(source._shape.ToArray());
            stages.Add(key, destination);
        }
        Engine.TensorCopy(source, destination);
        return destination;
    }

    private Tensor<T> StageDetachedTrainingInput(Tensor<T> generated, Tensor<T> mel, Tensor<T> segment)
    {
        try { return StageTrainingInput("detached", generated); }
        finally { ReleaseOwnedTrainingInput(Engine, generated, mel, segment); }
    }

    private void DisposeTrainingInputStages()
    {
        var stages = _trainingInputStages;
        _trainingInputStages = null;
        if (stages is null) return;
        foreach (var value in stages.Values)
        {
            try
            {
                if (Engine.TryGet<DirectGpuTensorEngine>(out var gpu))
                    gpu.ReleaseDeadDeviceStorage(value, afterQueuedWork: true);
            }
            finally { value.Dispose(); }
        }
    }

    // The supplied recordings belong to the caller. AlignedSegment's crops and a newly generated detached
    // discriminator waveform belong to this training call; a compiled plan may upload either after construction.
    // Release that private device binding in queue order before disposing its tensor, including when an upload
    // was attached to an originally CPU-owned tensor by EnsureResidentInput.
    internal static void ReleaseOwnedTrainingInput(IEngine engine, Tensor<T> value, Tensor<T> callerInput, Tensor<T> callerTarget)
    {
        if (ReferenceEquals(value, callerInput) || ReferenceEquals(value, callerTarget)) return;
        try
        {
            bool aliasesCaller = value.SharesStorageWith(callerInput) || value.SharesStorageWith(callerTarget);
            if (!aliasesCaller && engine.TryGet<DirectGpuTensorEngine>(out var gpu))
            {
                var buffer = value.DataVector._deviceState?.Buffer;
                // Shared device state can also be keyed on a caller array wrapped by a different tensor storage.
                if (buffer is not null)
                {
                    var root = AiDotNet.Tensors.Engines.DirectGpu.GpuBufferViewRoot.Of(buffer);
                    foreach (var boundary in new[] { callerInput, callerTarget })
                    {
                        var other = boundary.DataVector._deviceState?.Buffer ?? boundary._gpuBuffer;
                        if (other is null) continue;
                        var otherRoot = AiDotNet.Tensors.Engines.DirectGpu.GpuBufferViewRoot.Of(other);
                        if (ReferenceEquals(root, otherRoot) || (root.Handle != IntPtr.Zero && root.Handle == otherRoot.Handle))
                        {
                            aliasesCaller = true;
                            break;
                        }
                    }
                }
                if (!aliasesCaller)
                {
                    if (buffer is not null)
                        gpu.ReleaseDeadDeviceStorage(value, afterQueuedWork: true);
                    else
                    {
                        object vector = value.DataVector;
                        var array = value.GetBackingArrayForCacheLookupUnsafe();
                        gpu.EvictActivationsCreatedAfter(0, protect: null, mode: ActivationReleaseMode.Release,
                            releasableOnly: key => ReferenceEquals(key, vector) || (array is not null && ReferenceEquals(key, array)),
                            afterQueuedWork: true);
                    }
                }
            }
        }
        finally { value.Dispose(); }
    }
}
