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
    private readonly bool _useBf16Moments;
    private readonly int _min8BitSize;
    private readonly Dictionary<Tensor<T>, GpuMomentState> _gpuState =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly Dictionary<Tensor<float>, (Tensor<float> M, Tensor<float> V)> _gpuFp32SmallState =
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
        : base(nameof(StreamingAdam8Bit<T>), learningRate, new[] { true, false }, blockSize, maxUpdateRatio)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _decoupledWeightDecay = decoupledWeightDecay;
        _useBf16Moments = useBf16Moments;
        _min8BitSize = Math.Max(0, min8BitSize);
    }

    public override void Apply(Tensor<T> param, Tensor<T> grad)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));

        if (typeof(T) == typeof(float)
            && param.Length > 0
            && grad.Length == param.Length
            && AiDotNetEngine.Current is DirectGpuTensorEngine
            && TryApplyOnGpu(param, grad))
        {
            return;
        }

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

        if (!_useBf16Moments && length < _min8BitSize)
        {
            if (!_gpuFp32SmallState.TryGetValue(pFloat, out var fp) || fp.M.Length != length)
            {
                fp = (GpuOptimizer.CreateStateTensor(pFloat._shape), GpuOptimizer.CreateStateTensor(pFloat._shape));
                _gpuFp32SmallState[pFloat] = fp;
            }

            return _decoupledWeightDecay
                ? GpuOptimizer.TryAdamWStep(pFloat, gFloat, fp.M, fp.V, lr, b1, b2, eps, wd, step)
                : GpuOptimizer.TryAdamStep(pFloat, gFloat, fp.M, fp.V, lr, b1, b2, eps, wd, step);
        }

        if (!_gpuState.TryGetValue(param, out var state) || state.Length != length)
        {
            state?.Dispose();
            state = new GpuMomentState { Length = length };
            if (_useBf16Moments)
            {
                if (!GpuOptimizer.TryAllocBf16MomentState(length, out state.MBf16, out state.VBf16))
                    return false;
            }
            else
            {
                if (!GpuOptimizer.TryAllocAdam8BitState(
                    length, BlockSize, out state.MQuant, out state.VQuant, out state.MScales, out state.VScales))
                {
                    return false;
                }
            }
            _gpuState[param] = state;
        }

        if (_useBf16Moments)
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
        double effectiveGrad = _decoupledWeightDecay || _weightDecay == 0.0
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
        _gpuFp32SmallState.Clear();
    }
}
