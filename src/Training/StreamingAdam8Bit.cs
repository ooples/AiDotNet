using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Per-parameter 8-bit Adam(W) optimizer state for the memory-bounded streaming
/// training path. Each parameter's first/second moments are stored block-wise
/// quantized to 8 bits — roughly 16× smaller than fp64 moments — so a model
/// whose full-precision Adam state would not fit in RAM can still take a real,
/// Adam-faithful optimizer step. Each parameter's state is created lazily and
/// updated + applied IN PLACE inside the gradient-streaming callback, right
/// after that parameter's gradient is produced and before it is freed
/// (optimizer-in-backward).
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the block-wise quantization scheme of
/// <see cref="AiDotNet.Optimizers.Adam8BitOptimizer{T, TInput, TOutput}"/>
/// (8-bit signed first moment, 8-bit unsigned second moment, one scale per
/// block) but is scoped to a single parameter tensor so it can run incrementally
/// during the streaming backward rather than over one flat model-wide vector.
/// All arithmetic is performed in <see cref="double"/> and converted at the
/// tensor boundary, matching the rest of the framework's numeric pattern.
/// </para>
/// <para>
/// State persists across <c>Train</c> calls (keyed by parameter-tensor
/// reference) so the moments accumulate correctly over a multi-step training
/// run; <see cref="BeginStep"/> advances the shared timestep used for Adam bias
/// correction once per backward pass.
/// </para>
/// </remarks>
internal sealed class StreamingAdam8Bit<T>
{
    private sealed class MomentState
    {
        public readonly byte[] MQuant;     // signed first moment, 128 == 0
        public readonly byte[] VQuant;     // unsigned second moment
        public readonly double[] MScale;   // per block
        public readonly double[] VScale;   // per block

        public MomentState(int length, int numBlocks)
        {
            MQuant = new byte[length];
            VQuant = new byte[length];
            MScale = new double[numBlocks];
            VScale = new double[numBlocks];
            // 128 maps to 0 for the signed first moment; scales start at 1.0
            // so the zero-initialized state dequantizes to exactly zero.
            for (int i = 0; i < length; i++) MQuant[i] = 128;
            for (int b = 0; b < numBlocks; b++) { MScale[b] = 1.0; VScale[b] = 1.0; }
        }
    }

    private readonly Dictionary<Tensor<T>, MomentState> _state =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();

    private readonly int _blockSize;
    private readonly double _lr;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;
    private readonly double _maxUpdateRatio;

    private int _t; // shared Adam timestep (bias correction), advanced per backward pass

    // Reusable one-block-wide scratch for the updated full-precision moments,
    // used by the serial generic fallback (ApplyGeneric) only. The hot double/
    // float paths parallelize the block loop and allocate per-worker scratch via
    // ParallelForOrSerial's localInit, so this shared buffer must not be touched
    // from them (it is not thread-safe). ApplyGeneric runs serial, so sharing is
    // safe there.
    private readonly double[] _mScratch;
    private readonly double[] _vScratch;

    public StreamingAdam8Bit(
        double learningRate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8,
        double weightDecay = 0.0,
        int blockSize = 2048,
        double maxUpdateRatio = 5.0)
    {
        _lr = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _blockSize = Math.Max(1, blockSize);
        _maxUpdateRatio = maxUpdateRatio > 0 ? maxUpdateRatio : 5.0;
        _mScratch = new double[_blockSize];
        _vScratch = new double[_blockSize];
    }

    /// <summary>Advances the shared Adam timestep. Call once per backward pass,
    /// before streaming that pass's parameter gradients.</summary>
    public void BeginStep() => _t++;

    /// <summary>
    /// Applies one Adam(W) update to <paramref name="param"/> in place using the
    /// just-computed <paramref name="grad"/>, maintaining this parameter's 8-bit
    /// moment state. Safe to call exactly once per parameter per backward pass.
    /// </summary>
    public void Apply(Tensor<T> param, Tensor<T> grad)
    {
        // Apply is on the hot training path — a null tensor or length mismatch is a
        // hard correctness bug in the backward pass (gradient missing for a registered
        // source, or a shape regression between forward and backward). Silently
        // dropping the update hides the bug and leaves the parameter forever frozen,
        // which is worse than failing fast.
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        int length = param.Length;
        if (length == 0)
            throw new ArgumentException("StreamingAdam8Bit.Apply: param is empty (length == 0); " +
                "a zero-length parameter should never be registered as a training source.", nameof(param));
        if (grad.Length != length)
            throw new ArgumentException(
                $"StreamingAdam8Bit.Apply: gradient length {grad.Length} does not match param length {length}. " +
                "This indicates a shape mismatch between the forward pass and the tape backward — " +
                "fix the source of the size drift rather than letting Apply silently no-op.",
                nameof(grad));

        int numBlocks = (length + _blockSize - 1) / _blockSize;
        if (!_state.TryGetValue(param, out var st))
        {
            st = new MomentState(length, numBlocks);
            _state[param] = st;
        }

        double biasCorr1 = 1.0 - Math.Pow(_beta1, _t);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, _t);
        if (biasCorr1 <= 0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0) biasCorr2 = 1.0;

        // Fast path: raw double spans (no per-element Tensor indexer / NumOps
        // virtual dispatch). This is the dominant cost at foundation scale —
        // billions of elements — so the spanned path is ~10× the generic one and
        // the JIT auto-vectorizes the FMA-heavy moment math. Only valid for
        // non-view, full-storage parameter tensors (layer weights are exactly
        // that); anything else falls through to the generic path.
        if (typeof(T) == typeof(double)
            && (object)param is Tensor<double> pTen
            && (object)grad is Tensor<double> gTen)
        {
            var pMem = pTen.Data;
            var gMem = gTen.Data;
            if (pMem.Length >= length && gMem.Length >= length)
            {
                ApplyDouble(pMem, gMem, st, numBlocks, length, biasCorr1, biasCorr2);
                return;
            }
        }
        else if (typeof(T) == typeof(float)
            && (object)param is Tensor<float> pTenF
            && (object)grad is Tensor<float> gTenF)
        {
            var pMem = pTenF.Data;
            var gMem = gTenF.Data;
            if (pMem.Length >= length && gMem.Length >= length)
            {
                ApplyFloat(pMem, gMem, st, numBlocks, length, biasCorr1, biasCorr2);
                return;
            }
        }

        ApplyGeneric(param, grad, st, numBlocks, length, biasCorr1, biasCorr2);
    }

    private void ApplyDouble(
        Memory<double> pMem, ReadOnlyMemory<double> gMem, MomentState st,
        int numBlocks, int length, double biasCorr1, double biasCorr2)
    {
        double beta1 = _beta1, beta2 = _beta2, oneMinusB1 = 1.0 - _beta1, oneMinusB2 = 1.0 - _beta2;
        double lr = _lr, eps = _epsilon, wd = _weightDecay, maxStep = _lr * _maxUpdateRatio;
        double invBc1 = 1.0 / biasCorr1, invBc2 = 1.0 / biasCorr2;
        byte[] mQ = st.MQuant, vQ = st.VQuant;
        double[] mScaleArr = st.MScale, vScaleArr = st.VScale;
        int blockSize = _blockSize;

        // Each block owns a disjoint slice of p / mQ / vQ and its own per-block
        // scale, so the block loop is embarrassingly parallel. Per-worker scratch
        // (one block wide) is allocated once in localInit to keep the steady state
        // alloc-free; ParallelForOrSerial gates on DefaultSerialGrainSize so small
        // parameters still run serial (no thread-dispatch overhead). At foundation
        // scale this was ~28% of the training step and previously single-threaded.
        CpuParallelSettings.ParallelForOrSerial<(double[] m, double[] v)>(
            0, numBlocks, (long)length,
            () => (new double[blockSize], new double[blockSize]),
            (b, _, scratch) =>
            {
                double[] mNew = scratch.m, vNew = scratch.v;
                var p = pMem.Span;
                var g = gMem.Span;
                int start = b * blockSize;
                int end = Math.Min(start + blockSize, length);
                double mScale = mScaleArr[b];
                double vScale = vScaleArr[b];
                double newMMaxAbs = 0.0, newVMax = 0.0;

                for (int i = start; i < end; i++)
                {
                    int li = i - start;
                    double gi = g[i];
                    double mPrev = (mQ[i] - 128) * mScale;   // signed dequant
                    double vPrev = vQ[i] * vScale;            // unsigned dequant

                    if (double.IsNaN(gi) || double.IsInfinity(gi))
                    {
                        // Skip non-finite gradient: keep prior moments + weight.
                        mNew[li] = mPrev; vNew[li] = vPrev;
                    }
                    else
                    {
                        double m = beta1 * mPrev + oneMinusB1 * gi;
                        double v = beta2 * vPrev + oneMinusB2 * gi * gi;
                        double mHat = m * invBc1;
                        double vHat = v * invBc2;

                        double pv = p[i];
                        if (wd != 0.0) pv -= lr * wd * pv;
                        double update = lr * mHat / (Math.Sqrt(vHat) + eps);
                        // Trust bound: 8-bit quantization can round vHat→0, which
                        // would blow up the step; clamp to a small multiple of lr
                        // (real Adam steps are ~lr) and catch any residual NaN.
                        if (!(update >= -maxStep)) update = update > 0 ? maxStep : -maxStep;
                        else if (update > maxStep) update = maxStep;
                        p[i] = pv - update;
                        mNew[li] = m; vNew[li] = v;
                    }

                    double am = Math.Abs(mNew[li]);
                    if (am > newMMaxAbs) newMMaxAbs = am;
                    if (vNew[li] > newVMax) newVMax = vNew[li];
                }

                double newMScale = newMMaxAbs / 127.0; if (newMScale < 1e-10) newMScale = 1e-10;
                double newVScale = newVMax / 255.0;     if (newVScale < 1e-10) newVScale = 1e-10;
                mScaleArr[b] = newMScale; vScaleArr[b] = newVScale;
                double invM = 1.0 / newMScale, invV = 1.0 / newVScale;

                for (int i = start; i < end; i++)
                {
                    int li = i - start;
                    int mq = (int)Math.Round(mNew[li] * invM);
                    if (mq < -127) mq = -127; else if (mq > 127) mq = 127;
                    mQ[i] = (byte)(mq + 128);
                    int vq = (int)Math.Round(vNew[li] * invV);
                    if (vq < 0) vq = 0; else if (vq > 255) vq = 255;
                    vQ[i] = (byte)vq;
                }

                return scratch;
            },
            _ => { });
    }

    private void ApplyFloat(
        Memory<float> pMem, ReadOnlyMemory<float> gMem, MomentState st,
        int numBlocks, int length, double biasCorr1, double biasCorr2)
    {
        double beta1 = _beta1, beta2 = _beta2, oneMinusB1 = 1.0 - _beta1, oneMinusB2 = 1.0 - _beta2;
        double lr = _lr, eps = _epsilon, wd = _weightDecay, maxStep = _lr * _maxUpdateRatio;
        double invBc1 = 1.0 / biasCorr1, invBc2 = 1.0 / biasCorr2;
        byte[] mQ = st.MQuant, vQ = st.VQuant;
        double[] mScaleArr = st.MScale, vScaleArr = st.VScale;
        int blockSize = _blockSize;

        // Parallel block loop with per-worker scratch — see ApplyDouble for the
        // independence + grain-gating rationale.
        CpuParallelSettings.ParallelForOrSerial<(double[] m, double[] v)>(
            0, numBlocks, (long)length,
            () => (new double[blockSize], new double[blockSize]),
            (b, _, scratch) =>
            {
                double[] mNew = scratch.m, vNew = scratch.v;
                var p = pMem.Span;
                var g = gMem.Span;
                int start = b * blockSize;
                int end = Math.Min(start + blockSize, length);
                double mScale = mScaleArr[b];
                double vScale = vScaleArr[b];
                double newMMaxAbs = 0.0, newVMax = 0.0;

                for (int i = start; i < end; i++)
                {
                    int li = i - start;
                    double gi = g[i];
                    double mPrev = (mQ[i] - 128) * mScale;
                    double vPrev = vQ[i] * vScale;

                    if (double.IsNaN(gi) || double.IsInfinity(gi))
                    {
                        mNew[li] = mPrev; vNew[li] = vPrev;
                    }
                    else
                    {
                        double m = beta1 * mPrev + oneMinusB1 * gi;
                        double v = beta2 * vPrev + oneMinusB2 * gi * gi;
                        double mHat = m * invBc1;
                        double vHat = v * invBc2;
                        double pv = p[i];
                        if (wd != 0.0) pv -= lr * wd * pv;
                        double update = lr * mHat / (Math.Sqrt(vHat) + eps);
                        if (!(update >= -maxStep)) update = update > 0 ? maxStep : -maxStep;
                        else if (update > maxStep) update = maxStep;
                        p[i] = (float)(pv - update);
                        mNew[li] = m; vNew[li] = v;
                    }
                    double am = Math.Abs(mNew[li]);
                    if (am > newMMaxAbs) newMMaxAbs = am;
                    if (vNew[li] > newVMax) newVMax = vNew[li];
                }

                double newMScale = newMMaxAbs / 127.0; if (newMScale < 1e-10) newMScale = 1e-10;
                double newVScale = newVMax / 255.0;     if (newVScale < 1e-10) newVScale = 1e-10;
                mScaleArr[b] = newMScale; vScaleArr[b] = newVScale;
                double invM = 1.0 / newMScale, invV = 1.0 / newVScale;

                for (int i = start; i < end; i++)
                {
                    int li = i - start;
                    int mq = (int)Math.Round(mNew[li] * invM);
                    if (mq < -127) mq = -127; else if (mq > 127) mq = 127;
                    mQ[i] = (byte)(mq + 128);
                    int vq = (int)Math.Round(vNew[li] * invV);
                    if (vq < 0) vq = 0; else if (vq > 255) vq = 255;
                    vQ[i] = (byte)vq;
                }

                return scratch;
            },
            _ => { });
    }

    private void ApplyGeneric(
        Tensor<T> param, Tensor<T> grad, MomentState st,
        int numBlocks, int length, double biasCorr1, double biasCorr2)
    {
        double[] mNew = _mScratch, vNew = _vScratch;
        double maxStep = _lr * _maxUpdateRatio;

        for (int b = 0; b < numBlocks; b++)
        {
            int start = b * _blockSize;
            int end = Math.Min(start + _blockSize, length);
            double mScale = st.MScale[b];
            double vScale = st.VScale[b];
            double newMMaxAbs = 0.0, newVMax = 0.0;

            for (int i = start; i < end; i++)
            {
                int li = i - start;
                double g = _ops.ToDouble(grad[i]);
                double mPrev = (st.MQuant[i] - 128) * mScale;
                double vPrev = st.VQuant[i] * vScale;

                if (double.IsNaN(g) || double.IsInfinity(g))
                {
                    mNew[li] = mPrev; vNew[li] = vPrev;
                }
                else
                {
                    double m = _beta1 * mPrev + (1.0 - _beta1) * g;
                    double v = _beta2 * vPrev + (1.0 - _beta2) * g * g;
                    double mHat = m / biasCorr1;
                    double vHat = v / biasCorr2;
                    double p = _ops.ToDouble(param[i]);
                    if (_weightDecay != 0.0) p -= _lr * _weightDecay * p;
                    double update = _lr * mHat / (Math.Sqrt(vHat) + _epsilon);
                    if (!(update >= -maxStep)) update = update > 0 ? maxStep : -maxStep;
                    else if (update > maxStep) update = maxStep;
                    param[i] = _ops.FromDouble(p - update);
                    mNew[li] = m; vNew[li] = v;
                }
                double am = Math.Abs(mNew[li]);
                if (am > newMMaxAbs) newMMaxAbs = am;
                if (vNew[li] > newVMax) newVMax = vNew[li];
            }

            double newMScale = newMMaxAbs / 127.0; if (newMScale < 1e-10) newMScale = 1e-10;
            double newVScale = newVMax / 255.0;     if (newVScale < 1e-10) newVScale = 1e-10;
            st.MScale[b] = newMScale; st.VScale[b] = newVScale;

            for (int i = start; i < end; i++)
            {
                int li = i - start;
                int mq = (int)Math.Round(mNew[li] / newMScale);
                if (mq < -127) mq = -127; else if (mq > 127) mq = 127;
                st.MQuant[i] = (byte)(mq + 128);
                int vq = (int)Math.Round(vNew[li] / newVScale);
                if (vq < 0) vq = 0; else if (vq > 255) vq = 255;
                st.VQuant[i] = (byte)vq;
            }
        }
    }
}
