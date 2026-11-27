using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Per-parameter 8-bit (or BF16-moment) Adam/AdamW optimizer state for the memory-bounded streaming path,
/// with zero-copy GPU-resident execution via <see cref="GpuOptimizer.TryAdam8BitStep(Tensor{float}, Tensor{float}, IGpuBuffer?, IGpuBuffer?, IGpuBuffer?, IGpuBuffer?, float, float, float, float, float, float, int, float, bool)"/>
/// and <see cref="GpuOptimizer.TryAdamBf16Step"/> across all supported GPU backends.
/// </summary>
internal class StreamingAdam8Bit<T> : BlockQuantizedStreamingOptimizer<T>, IDisposable
{
    private sealed class GpuMomentState : IDisposable
    {
        public int Length;
        public IGpuBuffer? MQuant;
        public IGpuBuffer? VQuant;
        public IGpuBuffer? MScales;
        public IGpuBuffer? VScales;
        public IGpuBuffer? MBf16;
        public IGpuBuffer? VBf16;

        public void Dispose()
        {
            GpuOptimizer.FreeGpuBuffer(MQuant);
            GpuOptimizer.FreeGpuBuffer(VQuant);
            GpuOptimizer.FreeGpuBuffer(MScales);
            GpuOptimizer.FreeGpuBuffer(VScales);
            GpuOptimizer.FreeGpuBuffer(MBf16);
            GpuOptimizer.FreeGpuBuffer(VBf16);
            MQuant = VQuant = MScales = VScales = MBf16 = VBf16 = null;
        }
    }

    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;
    private readonly bool _decoupledWeightDecay;
    // The path each parameter's moments live on, fixed by its first step. The GPU state and the base class's CPU state
    // are separate representations, so a parameter that switched paths would continue from moments that miss every
    // update made on the other path.
    private readonly Dictionary<Tensor<T>, bool> _keepsMomentsOnGpu =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly Dictionary<Tensor<T>, GpuMomentState> _gpuState =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly Dictionary<Tensor<float>, Tensor<float>[]> _gpuFp32SmallState =
        new(TensorReferenceComparer<Tensor<float>>.Instance);

    public StreamingAdam8Bit(
        double learningRate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8,
        double weightDecay = 0.0,
        int blockSize = 2048,
        double maxUpdateRatio = 5.0,
        bool decoupledWeightDecay = false,
        bool useBf16Moments = false,
        int min8BitSize = 0)
        : base(nameof(StreamingAdam8Bit<T>), learningRate, new[] { true, false }, blockSize, maxUpdateRatio,
            useBf16Moments, min8BitSize)
    {
        if (double.IsNaN(weightDecay) || double.IsInfinity(weightDecay) || weightDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(weightDecay), weightDecay,
                "Weight decay must be a finite, non-negative value.");
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _decoupledWeightDecay = decoupledWeightDecay;
    }

    protected override bool TryApplyBatchOnGpu(IReadOnlyList<Tensor<T>> parameters, IReadOnlyList<Tensor<T>> gradients)
    {
        string listMethod = _decoupledWeightDecay ? "TryAdamWStepMultiTensor" : "TryAdamStepMultiTensor";
        return TryApplySmallParametersGpuListStep(
            parameters,
            gradients,
            _keepsMomentsOnGpu,
            _gpuFp32SmallState,
            listMethod,
            stateCount: 2,
            (float)LearningRate,
            (float)_beta1,
            (float)_beta2,
            (float)_epsilon,
            (float)_weightDecay,
            Math.Max(1, Step));
    }

    public override void Apply(Tensor<T> param, Tensor<T> grad)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));

        bool gpuEligible = typeof(T) == typeof(float)
            && param.Length > 0
            && grad.Length == param.Length
            && AiDotNetEngine.Current is DirectGpuTensorEngine;

        if (_keepsMomentsOnGpu.TryGetValue(param, out bool onGpu))
        {
            if (!onGpu)
            {
                base.Apply(param, grad);
                return;
            }
            if (gpuEligible && TryApplyOnGpu(param, grad))
                return;
            throw new InvalidOperationException(
                $"{nameof(StreamingAdam8Bit<T>)}: the GPU step failed for a parameter whose optimizer moments are on the " +
                "GPU from earlier steps. Continuing on the CPU would restart it from zero moments, so the step is not taken.");
        }

        // The first step picks the path for the parameter's lifetime: the GPU when it is available and its state
        // allocates, the CPU (in the configured 8-bit, BF16 or FP32 layout) otherwise.
        if (gpuEligible && TryApplyOnGpu(param, grad))
        {
            _keepsMomentsOnGpu[param] = true;
            return;
        }
        _keepsMomentsOnGpu[param] = false;
        base.Apply(param, grad);
    }

    private bool TryApplyOnGpu(Tensor<T> param, Tensor<T> grad)
    {
        var pFloat = Unsafe.As<Tensor<T>, Tensor<float>>(ref param);
        var gFloat = Unsafe.As<Tensor<T>, Tensor<float>>(ref grad);
        int length = pFloat.Length;

        float lr = (float)LearningRate;
        float b1 = (float)_beta1;
        float b2 = (float)_beta2;
        float eps = (float)_epsilon;
        float wd = (float)_weightDecay;
        int step = Math.Max(1, Step);

        if (!UseBf16Moments && length < Min8BitSize)
        {
            if (!_gpuFp32SmallState.TryGetValue(pFloat, out var fp) || fp.Length < 2 || fp[0].Length != length)
            {
                fp = new[]
                {
                    GpuOptimizer.CreateStateTensor(pFloat._shape),
                    GpuOptimizer.CreateStateTensor(pFloat._shape)
                };
                _gpuFp32SmallState[pFloat] = fp;
            }

            return _decoupledWeightDecay
                ? GpuOptimizer.TryAdamWStep(pFloat, gFloat, fp[0], fp[1], lr, b1, b2, eps, wd, step)
                : GpuOptimizer.TryAdamStep(pFloat, gFloat, fp[0], fp[1], lr, b1, b2, eps, wd, step);
        }

        if (!_gpuState.TryGetValue(param, out var state) || state is null || state.Length != length)
        {
            // Unmap the stale state before freeing it, so a failed allocation below never leaves a disposed
            // state mapped to this parameter, and free a partial allocation instead of leaking it.
            _gpuState.Remove(param);
            state?.Dispose();
            var fresh = new GpuMomentState { Length = length };
            bool allocated = UseBf16Moments
                ? GpuOptimizer.TryAllocBf16MomentState(length, out fresh.MBf16, out fresh.VBf16)
                : GpuOptimizer.TryAllocAdam8BitState(
                    length, BlockSize, out fresh.MQuant, out fresh.VQuant, out fresh.MScales, out fresh.VScales);
            if (!allocated)
            {
                fresh.Dispose();
                return false;
            }
            _gpuState[param] = fresh;
            state = fresh;
        }

        if (UseBf16Moments)
        {
            return GpuOptimizer.TryAdamBf16Step(
                pFloat, gFloat, state.MBf16, state.VBf16, lr, b1, b2, eps, wd, step, _decoupledWeightDecay);
        }

        float bc1 = (float)(1.0 - Math.Pow(_beta1, step));
        float bc2 = (float)(1.0 - Math.Pow(_beta2, step));
        if (bc1 <= 0f) bc1 = 1f;
        if (bc2 <= 0f) bc2 = 1f;

        return GpuOptimizer.TryAdam8BitStep(
            pFloat, gFloat,
            state.MQuant, state.VQuant, state.MScales, state.VScales,
            lr, b1, b2, eps, bc1, bc2, BlockSize,
            wd, _decoupledWeightDecay);
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double effectiveGrad = _decoupledWeightDecay || _weightDecay <= 0.0
            ? gradient
            : gradient + _weightDecay * parameter;

        double m = _beta1 * moments[0] + (1.0 - _beta1) * effectiveGrad;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * effectiveGrad * effectiveGrad;
        nextMoments[0] = m;
        nextMoments[1] = v;

        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, Step);
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0.0) biasCorr2 = 1.0;

        if (_decoupledWeightDecay)
            parameter = ApplyDecoupledWeightDecay(parameter, _weightDecay);
        double update = LearningRate * (m / biasCorr1) / (Math.Sqrt(v / biasCorr2) + _epsilon);
        return parameter - ClampUpdate(update);
    }

    public void Dispose()
    {
        foreach (var entry in _gpuState.Values)
            entry.Dispose();
        _gpuState.Clear();
        foreach (var fp in _gpuFp32SmallState.Values)
        {
            for (int s = 0; s < fp.Length; s++)
                fp[s]?.Dispose();
        }
        _gpuFp32SmallState.Clear();
        _keepsMomentsOnGpu.Clear();
    }
}

internal static class GpuMultiTensorAdamBridge
{
    public delegate bool MultiTensorAdamStepDelegate(
        IReadOnlyList<Tensor<float>> parameters,
        IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> firstMoments,
        IReadOnlyList<Tensor<float>> secondMoments,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step);

    public delegate bool MultiTensorScaleDelegate(
        IReadOnlyList<Tensor<float>> gradients,
        float scale);

    public static readonly MultiTensorAdamStepDelegate? TryAdamMultiTensor = ResolveAdamDelegate("TryAdamStepMultiTensor");
    public static readonly MultiTensorAdamStepDelegate? TryAdamWMultiTensor = ResolveAdamDelegate("TryAdamWStepMultiTensor");
    public static readonly MultiTensorScaleDelegate? TryScaleGradientsMultiTensor = ResolveScaleDelegate();

    private static MultiTensorAdamStepDelegate? ResolveAdamDelegate(string methodName)
    {
        try
        {
            var method = typeof(GpuOptimizer).GetMethod(
                methodName,
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static,
                binder: null,
                types: new[]
                {
                    typeof(IReadOnlyList<Tensor<float>>),
                    typeof(IReadOnlyList<Tensor<float>>),
                    typeof(IReadOnlyList<Tensor<float>>),
                    typeof(IReadOnlyList<Tensor<float>>),
                    typeof(float), typeof(float), typeof(float), typeof(float), typeof(float), typeof(int)
                },
                modifiers: null);
            if (method is null) return null;
            return (MultiTensorAdamStepDelegate)Delegate.CreateDelegate(typeof(MultiTensorAdamStepDelegate), method);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"[AiDotNet] GpuMultiTensorAdamBridge failed to bind {methodName}: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static MultiTensorScaleDelegate? ResolveScaleDelegate()
    {
        try
        {
            var method = typeof(GpuOptimizer).GetMethod(
                "TryScaleGradientsMultiTensor",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static,
                binder: null,
                types: new[]
                {
                    typeof(IReadOnlyList<Tensor<float>>),
                    typeof(float)
                },
                modifiers: null);
            if (method is null) return null;
            return (MultiTensorScaleDelegate)Delegate.CreateDelegate(typeof(MultiTensorScaleDelegate), method);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"[AiDotNet] GpuMultiTensorAdamBridge failed to bind TryScaleGradientsMultiTensor: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }
}
