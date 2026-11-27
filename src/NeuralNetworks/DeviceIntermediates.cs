using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Frees the device memory of an inference intermediate at its last use, instead of when the enclosing
/// <see cref="TensorScope"/> ends. A scope hands a kept result to the scope enclosing it, so a chain of results that
/// each replace the one before (a dense block's running concatenation, the output of each layer of a sequential stack)
/// stays alive whole until the outermost scope ends unless the replaced ones are released as the chain advances.
/// </summary>
internal static class DeviceIntermediates
{
    /// <summary>
    /// Releases <paramref name="consumed"/>, a result created after <paramref name="activationsBefore"/> (an
    /// <see cref="DirectGpuTensorEngine.ActivationCacheTimestampSnapshot"/>) that nothing reads again, unless it shares
    /// storage with one of <paramref name="live"/>. The release is ordered behind the queued device work that still
    /// reads it, and a later host read of it throws instead of returning stale data.
    /// </summary>
    internal static void ReleaseConsumed<T>(DirectGpuTensorEngine gpu, Tensor<T> consumed, long activationsBefore,
        params Tensor<T>[] live)
    {
        foreach (var kept in live)
        {
            if (ReferenceEquals(consumed, kept) || consumed.SharesStorageWith(kept)) return;
        }

        if (consumed.DataVector._deviceState is { Buffer: not null })
        {
            // The tensor owns its buffer: the stream-ordered release a TensorScope runs for the results it frees.
            gpu.ReleaseDeadDeviceStorage(consumed, afterQueuedWork: true);
            return;
        }

        // A result held by the engine's activation cache instead of by its vector: release this result's entries only.
        object vector = consumed.DataVector;
        var array = consumed.GetBackingArrayForCacheLookupUnsafe();
        gpu.EvictActivationsCreatedAfter(activationsBefore, protect: null, mode: ActivationReleaseMode.Release,
            releasableOnly: key => ReferenceEquals(key, vector) || (array is not null && ReferenceEquals(key, array)),
            afterQueuedWork: true);
    }
}
