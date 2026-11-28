using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Optimizers;

/// <summary>
/// Fixed-address GPU scalar buffer holding per-step optimizer hyperparameters and a device-side NaN/Inf anomaly flag
/// (<c>[LearningRate, StepCount, WeightDecay, Beta1, Beta2, Epsilon, MaxGradNorm, AnomalyFlag]</c>) so full-step
/// CUDA Graphs (<c>Forward + Loss + Backward + Multi-Tensor Clip + Optimizer.Step</c>) bind one constant device
/// pointer at capture time while learning-rate warmup / cosine decay updates the scalar values in place without
/// invalidating the captured graph.
/// </summary>
internal sealed class GpuOptimizerScalarBuffer : IDisposable
{
    internal const int SlotLearningRate = 0;
    internal const int SlotStepCount = 1;
    internal const int SlotWeightDecay = 2;
    internal const int SlotBeta1 = 3;
    internal const int SlotBeta2 = 4;
    internal const int SlotEpsilon = 5;
    internal const int SlotMaxGradNorm = 6;
    internal const int SlotAnomalyFlag = 7;
    internal const int ScalarCount = 8;

    private readonly object _gate = new();
    private readonly float[] _hostStaging = new float[ScalarCount];
    private readonly byte[] _hostStagingBytes = new byte[ScalarCount * sizeof(float)];
    private IDirectGpuBackend? _backend;
    private IGpuBuffer? _deviceBuffer;
    private bool _hasUploadedInitialState;
    private bool _disposed;

    /// <summary>Gets the live fixed-address GPU scalar buffer, or <c>null</c> when no GPU step has initialized it.</summary>
    internal IGpuBuffer? DeviceBuffer => _deviceBuffer;

    /// <summary>Gets the most recently staged learning rate.</summary>
    internal float LastStagedLearningRate => _hostStaging[SlotLearningRate];

    /// <summary>Gets the most recently staged 1-based optimizer step count.</summary>
    internal long LastStagedStepCount => (long)_hostStaging[SlotStepCount];

    /// <summary>
    /// Ensures the fixed-address 8-float GPU scalar buffer exists on <paramref name="backend"/> and updates its
    /// contents in place whenever any hyperparameter or step counter changed.
    /// </summary>
    internal IGpuBuffer? EnsureUpdated(
        IDirectGpuBackend backend,
        float learningRate,
        long stepCount,
        float weightDecay,
        float beta1,
        float beta2,
        float epsilon,
        float maxGradNorm,
        bool isStreamCapturing = false)
    {
        if (backend is null)
            throw new ArgumentNullException(nameof(backend));

        float stepCountFloat = stepCount;
        lock (_gate)
        {
            if (_disposed)
                return null;

            bool needRealloc = _deviceBuffer is null
                || _deviceBuffer.Handle == IntPtr.Zero
                || !ReferenceEquals(_backend, backend);

            if (needRealloc)
            {
                if (isStreamCapturing)
                {
                    // Never allocate a brand-new persistent buffer mid-capture; warmup steps allocate it first.
                    return _deviceBuffer;
                }

                _deviceBuffer?.Dispose();
                _deviceBuffer = null;
                _backend = null;
                // Not uploaded until AllocateBuffer succeeds: a throw leaves the slot reallocating on the next call.
                _hasUploadedInitialState = false;
                _hostStaging[SlotLearningRate] = learningRate;
                _hostStaging[SlotStepCount] = stepCountFloat;
                _hostStaging[SlotWeightDecay] = weightDecay;
                _hostStaging[SlotBeta1] = beta1;
                _hostStaging[SlotBeta2] = beta2;
                _hostStaging[SlotEpsilon] = epsilon;
                _hostStaging[SlotMaxGradNorm] = maxGradNorm;
                _hostStaging[SlotAnomalyFlag] = 0f;

                _deviceBuffer = backend.AllocateBuffer(_hostStaging);
                _backend = backend;
                _hasUploadedInitialState = true;
                return _deviceBuffer;
            }

            bool changed = !_hasUploadedInitialState
                || _hostStaging[SlotLearningRate] != learningRate
                || _hostStaging[SlotStepCount] != stepCountFloat
                || _hostStaging[SlotWeightDecay] != weightDecay
                || _hostStaging[SlotBeta1] != beta1
                || _hostStaging[SlotBeta2] != beta2
                || _hostStaging[SlotEpsilon] != epsilon
                || _hostStaging[SlotMaxGradNorm] != maxGradNorm;

            if (changed && !isStreamCapturing && _deviceBuffer is { } liveBuffer)
            {
                _hostStaging[SlotLearningRate] = learningRate;
                _hostStaging[SlotStepCount] = stepCountFloat;
                _hostStaging[SlotWeightDecay] = weightDecay;
                _hostStaging[SlotBeta1] = beta1;
                _hostStaging[SlotBeta2] = beta2;
                _hostStaging[SlotEpsilon] = epsilon;
                _hostStaging[SlotMaxGradNorm] = maxGradNorm;
                _hostStaging[SlotAnomalyFlag] = 0f;
                Buffer.BlockCopy(_hostStaging, 0, _hostStagingBytes, 0, _hostStagingBytes.Length);
                // Mark dirty before the upload and complete only after it returns: the staging array already holds
                // the requested values, so a failed upload must not let a retry with the same values skip the copy.
                _hasUploadedInitialState = false;
                backend.UploadByteBuffer(liveBuffer, _hostStagingBytes);
                _hasUploadedInitialState = true;
            }

            return _deviceBuffer;
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
                return;
            _disposed = true;
            _deviceBuffer?.Dispose();
            _deviceBuffer = null;
            _backend = null;
            _hasUploadedInitialState = false;
        }
    }
}
