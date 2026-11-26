using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// #1662 lever #1: full-precision per-parameter Adam for the streaming optimizer-in-backward
/// path. Bit-identical to the classic <c>AdamOptimizer</c> whole-vector step (same formula,
/// bias-correction, epsilon placement, and once-per-step <c>t</c>) — NO update clamping and NO
/// 8-bit moment quantization, unlike <see cref="StreamingAdam8Bit{T}"/>. Used when the streaming
/// fused path engages for a model that FITS in memory (the common-case throughput/locality win),
/// where matching the classic trajectory exactly is required.
/// </summary>
internal sealed class FullPrecisionStreamingAdam<T> : FullPrecisionStreamingOptimizer<T>, IDisposable
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;
    private readonly Dictionary<Tensor<T>, bool> _keepsMomentsOnGpu =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly Dictionary<Tensor<float>, (Tensor<float> M, Tensor<float> V)> _gpuState =
        new(TensorReferenceComparer<Tensor<float>>.Instance);

    public FullPrecisionStreamingAdam(
        double learningRate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8,
        double weightDecay = 0.0)
        : base(nameof(FullPrecisionStreamingAdam<T>), learningRate, momentCount: 2)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
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
            if (onGpu)
            {
                if (gpuEligible && TryApplyOnGpu(param, grad))
                    return;
                MigrateGpuMomentsToHost(param);
            }
            else if (gpuEligible && TryApplyOnGpu(param, grad))
            {
                _keepsMomentsOnGpu[param] = true;
                return;
            }
            base.Apply(param, grad);
            return;
        }

        if (gpuEligible && TryApplyOnGpu(param, grad))
        {
            _keepsMomentsOnGpu[param] = true;
            return;
        }
        _keepsMomentsOnGpu[param] = false;
        base.Apply(param, grad);
    }

    public override void ApplyBatch(IReadOnlyList<Tensor<T>> parameters, IReadOnlyList<Tensor<T>> gradients)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        int count = parameters.Count;
        if (gradients.Count != count)
            throw new ArgumentException("Parameters and gradients must have identical counts.", nameof(gradients));
        if (count == 0) return;

        var multiFn = GpuMultiTensorAdamBridge.TryAdamWMultiTensor;
        if (typeof(T) != typeof(float) || multiFn is null || AiDotNetEngine.Current is not DirectGpuTensorEngine)
        {
            base.ApplyBatch(parameters, gradients);
            return;
        }

        var mtParams = new List<Tensor<float>>(count);
        var mtGrads = new List<Tensor<float>>(count);
        var mtMs = new List<Tensor<float>>(count);
        var mtVs = new List<Tensor<float>>(count);
        var mtSourceRefs = new List<Tensor<T>>(count);
        var mtOrigGrads = new List<Tensor<T>>(count);
        List<int>? deferredIndices = null;

        for (int i = 0; i < count; i++)
        {
            var p = parameters[i];
            var g = gradients[i];
            if (p is null) throw new ArgumentNullException(nameof(parameters));
            if (g is null) throw new ArgumentNullException(nameof(gradients));
            int len = p.Length;
            bool eligible = len > 0 && g.Length == len;
            if (!eligible)
            {
                (deferredIndices ??= new List<int>()).Add(i);
                continue;
            }

            var pFloat = Unsafe.As<Tensor<T>, Tensor<float>>(ref p);
            var gFloat = Unsafe.As<Tensor<T>, Tensor<float>>(ref g);
            var fp = GetOrCreateGpuMoments(p, pFloat, len);

            mtParams.Add(pFloat);
            mtGrads.Add(gFloat);
            mtMs.Add(fp.M);
            mtVs.Add(fp.V);
            mtSourceRefs.Add(p);
            mtOrigGrads.Add(g);
        }

        if (mtParams.Count > 0)
        {
            int step = Math.Max(1, Step);
            if (multiFn(mtParams, mtGrads, mtMs, mtVs, (float)LearningRate, (float)_beta1, (float)_beta2, (float)_epsilon, (float)_weightDecay, step))
            {
                for (int i = 0; i < mtSourceRefs.Count; i++)
                    _keepsMomentsOnGpu[mtSourceRefs[i]] = true;
            }
            else
            {
                for (int i = 0; i < mtSourceRefs.Count; i++)
                    Apply(mtSourceRefs[i], mtOrigGrads[i]);
            }
        }

        if (deferredIndices is not null)
        {
            for (int i = 0; i < deferredIndices.Count; i++)
            {
                int idx = deferredIndices[i];
                Apply(parameters[idx], gradients[idx]);
            }
        }
    }

    private (Tensor<float> M, Tensor<float> V) GetOrCreateGpuMoments(Tensor<T> param, Tensor<float> pFloat, int len)
    {
        if (_gpuState.TryGetValue(pFloat, out var fp) && fp.M.Length == len)
            return fp;

        if (TryTakeHostMoments(param, out var hostMoments)
            && hostMoments is not null
            && hostMoments.Length >= 2
            && hostMoments[0].Length == len
            && hostMoments[1].Length == len)
        {
            var mArr = new float[len];
            var vArr = new float[len];
            for (int i = 0; i < len; i++)
            {
                mArr[i] = (float)hostMoments[0][i];
                vArr[i] = (float)hostMoments[1][i];
            }
            fp = (
                new Tensor<float>(pFloat._shape, new Vector<float>(mArr)),
                new Tensor<float>(pFloat._shape, new Vector<float>(vArr)));
        }
        else
        {
            fp = (GpuOptimizer.CreateStateTensor(pFloat._shape), GpuOptimizer.CreateStateTensor(pFloat._shape));
        }

        _gpuState[pFloat] = fp;
        return fp;
    }

    private void MigrateGpuMomentsToHost(Tensor<T> param)
    {
        var pFloat = Unsafe.As<Tensor<T>, Tensor<float>>(ref param);
        if (!_gpuState.TryGetValue(pFloat, out var fp))
        {
            _keepsMomentsOnGpu[param] = false;
            return;
        }

        try
        {
            int len = param.Length;
            var mSpan = fp.M.AsSpan();
            var vSpan = fp.V.AsSpan();
            if (mSpan.Length == len && vSpan.Length == len)
            {
                var mHost = new double[len];
                var vHost = new double[len];
                for (int i = 0; i < len; i++)
                {
                    mHost[i] = mSpan[i];
                    vHost[i] = vSpan[i];
                }
                SetHostMoments(param, new[] { mHost, vHost });
            }
            _keepsMomentsOnGpu[param] = false;
        }
        catch
        {
            // If the GPU context is unrecoverably lost and device moments cannot be downloaded,
            // clear the GPU ownership entry before rethrowing so subsequent CPU steps are not poisoned.
            _keepsMomentsOnGpu.Remove(param);
            throw;
        }
        finally
        {
            _gpuState.Remove(pFloat);
            try { fp.M.Dispose(); } catch { }
            try { fp.V.Dispose(); } catch { }
        }
    }

    private bool TryApplyOnGpu(Tensor<T> param, Tensor<T> grad)
    {
        var pFloat = Unsafe.As<Tensor<T>, Tensor<float>>(ref param);
        var gFloat = Unsafe.As<Tensor<T>, Tensor<float>>(ref grad);
        int len = pFloat.Length;
        var fp = GetOrCreateGpuMoments(param, pFloat, len);
        return GpuOptimizer.TryAdamWStep(
            pFloat, gFloat, fp.M, fp.V,
            (float)LearningRate, (float)_beta1, (float)_beta2, (float)_epsilon, (float)_weightDecay, Math.Max(1, Step));
    }

    public void Dispose()
    {
        foreach (var fp in _gpuState.Values)
        {
            fp.M.Dispose();
            fp.V.Dispose();
        }
        _gpuState.Clear();
        _keepsMomentsOnGpu.Clear();
    }

    protected override double UpdateElement(
        double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        // m = beta1*m + (1-beta1)*g ;  v = beta2*v + (1-beta2)*g^2
        double m = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * gradient * gradient;
        nextMoments[0] = m;
        nextMoments[1] = v;

        // Bias correction with the global step t (advanced once per BeginStep, matching
        // AdamOptimizer._t). For t >= 1 these are always > 0, so no guard is needed.
        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, Step);

        parameter = ApplyDecoupledWeightDecay(parameter, _weightDecay);
        // update = lr * mHat / (sqrt(vHat) + eps) ; NO clamping (classic Adam does not clamp).
        double update = LearningRate * (m / biasCorr1) / (Math.Sqrt(v / biasCorr2) + _epsilon);
        return parameter - update;
    }
}
