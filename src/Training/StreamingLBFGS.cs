// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Memory-bounded streaming L-BFGS — the second-order entry in the streaming optimizer family.
/// </summary>
/// <remarks>
/// <para>
/// Unlike the first-order streaming optimizers (which update each parameter in place as its
/// gradient is produced), L-BFGS computes a search direction from the WHOLE gradient plus a
/// curvature history, so it can't update per-parameter-as-gradient-arrives. Instead it buffers
/// the per-parameter gradients/params into flat vectors during <see cref="Apply"/> and performs
/// the two-loop recursion + parameter update once per step in <see cref="EndStep"/>.
/// </para>
/// <para>
/// <b>The memory win:</b> the (s, y) curvature history — <c>MemorySize</c> vectors of length n
/// each, O(m·n) — is the dominant memory cost of L-BFGS. Here it's stored 8-bit block-quantized
/// (one signed int8 per element + one fp64 scale per block), ~8× smaller than fp64. The history
/// is dequantized ON DEMAND, one vector at a time into a reused O(n) buffer, so the two-loop's
/// transient memory stays O(n) rather than O(m·n). Full BFGS/Newton (O(n²) / dense Hessian) are
/// intentionally not provided — they don't fit memory-bounded large-model training.
/// </para>
/// </remarks>
internal sealed class StreamingLBFGS<T> : IStreamingOptimizer<T>, IStreamingOptimizerLearningRate
{
    private readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();
    private readonly int _memorySize;
    private readonly int _blockSize;
    private double _learningRate;

    // Stable per-step layout of the parameter tensors + their slices in the flat buffers.
    private readonly List<(Tensor<T> Param, int Offset, int Length)> _layout = new();
    private double[] _gradFlat = Array.Empty<double>();
    private double[] _paramFlat = Array.Empty<double>();
    private int _writeOffset;

    // 8-bit block-quantized history (the memory win) + the O(n) previous gradient for y = g - g_prev.
    private readonly List<QuantizedVector> _sHist = new();
    private readonly List<QuantizedVector> _yHist = new();
    private double[]? _prevGrad;

    // O(n) reused scratch so the two-loop's dequant stays O(n) transient (not O(m·n)).
    private double[] _q = Array.Empty<double>();
    private double[] _tmpS = Array.Empty<double>();
    private double[] _tmpY = Array.Empty<double>();

    public StreamingLBFGS(double learningRate, int memorySize = 10, int blockSize = 2048)
    {
        _learningRate = learningRate;
        _memorySize = Math.Max(1, memorySize);
        _blockSize = Math.Max(1, blockSize);
    }

    public void SetLearningRate(double learningRate)
    {
        if (learningRate > 0 && !double.IsNaN(learningRate) && !double.IsInfinity(learningRate))
            _learningRate = learningRate;
    }

    public void BeginStep()
    {
        _layout.Clear();
        _writeOffset = 0;
    }

    public void Apply(Tensor<T> param, Tensor<T> grad)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        int len = param.Length;
        if (grad.Length != len)
            throw new ArgumentException($"StreamingLBFGS.Apply: gradient length {grad.Length} does not match param length {len}.", nameof(grad));

        EnsureFlatCapacity(_writeOffset + len);
        _layout.Add((param, _writeOffset, len));
        for (int i = 0; i < len; i++)
        {
            _gradFlat[_writeOffset + i] = _ops.ToDouble(grad[i]);
            _paramFlat[_writeOffset + i] = _ops.ToDouble(param[i]);
        }
        _writeOffset += len;
    }

    public void EndStep()
    {
        int n = _writeOffset;
        if (n == 0) return;
        EnsureScratch(n);

        // Search direction d = H_k * g (gradient descent on the first step, before any history).
        var d = TwoLoopDirection(n);

        // s = x_new - x = -lr*d (computed into _tmpS); write x_new = x + s back to the live tensors.
        for (int i = 0; i < n; i++)
        {
            double step = _learningRate * d[i];
            _tmpS[i] = (double.IsNaN(step) || double.IsInfinity(step)) ? 0.0 : -step;
        }
        Scatter(_tmpS);

        // Curvature pair: y = g - g_prev. Only store a positive-curvature (s·y > 0) pair so the
        // implicit Hessian approximation stays positive-definite (standard L-BFGS skip rule).
        if (_prevGrad is not null)
        {
            for (int i = 0; i < n; i++) _tmpY[i] = _gradFlat[i] - _prevGrad[i];
            if (Dot(_tmpS, _tmpY, n) > 1e-12)
                PushHistory(_tmpS, _tmpY, n);
        }

        if (_prevGrad is null || _prevGrad.Length < n) _prevGrad = new double[n];
        Array.Copy(_gradFlat, _prevGrad, n);
    }

    private double[] TwoLoopDirection(int n)
    {
        Array.Copy(_gradFlat, _q, n); // q = g
        int k = _sHist.Count;
        if (k == 0)
        {
            var d0 = new double[n];
            Array.Copy(_q, d0, n);
            return d0; // plain gradient descent until curvature is available
        }

        var alpha = new double[k];
        var rho = new double[k];

        for (int i = k - 1; i >= 0; i--) // recent → old
        {
            _sHist[i].Dequantize(_tmpS, n);
            _yHist[i].Dequantize(_tmpY, n);
            double ys = Dot(_tmpY, _tmpS, n);
            rho[i] = ys != 0.0 ? 1.0 / ys : 0.0;
            alpha[i] = rho[i] * Dot(_tmpS, _q, n);
            Axpy(_q, _tmpY, -alpha[i], n); // q -= alpha_i * y_i
        }

        // H0 = gamma·I, gamma = (s·y)/(y·y) of the most recent pair (Nocedal & Wright).
        _sHist[k - 1].Dequantize(_tmpS, n);
        _yHist[k - 1].Dequantize(_tmpY, n);
        double yy = Dot(_tmpY, _tmpY, n);
        double gamma = yy > 0.0 ? Dot(_tmpS, _tmpY, n) / yy : 1.0;
        if (gamma <= 0.0 || double.IsNaN(gamma) || double.IsInfinity(gamma)) gamma = 1.0;

        var r = new double[n];
        for (int i = 0; i < n; i++) r[i] = gamma * _q[i];

        for (int i = 0; i < k; i++) // old → recent
        {
            _sHist[i].Dequantize(_tmpS, n);
            _yHist[i].Dequantize(_tmpY, n);
            double beta = rho[i] * Dot(_tmpY, r, n);
            Axpy(r, _tmpS, alpha[i] - beta, n); // r += (alpha_i - beta) * s_i
        }
        return r;
    }

    private void Scatter(double[] delta)
    {
        foreach (var (param, offset, length) in _layout)
        {
            for (int i = 0; i < length; i++)
            {
                double next = _ops.ToDouble(param[i]) + delta[offset + i];
                param[i] = _ops.FromDouble(next);
            }
        }
    }

    private void PushHistory(double[] s, double[] y, int n)
    {
        _sHist.Add(QuantizedVector.Quantize(s, n, _blockSize));
        _yHist.Add(QuantizedVector.Quantize(y, n, _blockSize));
        while (_sHist.Count > _memorySize)
        {
            _sHist.RemoveAt(0);
            _yHist.RemoveAt(0);
        }
    }

    private static double Dot(double[] a, double[] b, int n)
    {
        double s = 0.0;
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    private static void Axpy(double[] y, double[] x, double a, int n)
    {
        for (int i = 0; i < n; i++) y[i] += a * x[i];
    }

    private void EnsureFlatCapacity(int needed)
    {
        if (_gradFlat.Length < needed)
        {
            int cap = Math.Max(needed, _gradFlat.Length == 0 ? needed : _gradFlat.Length * 2);
            Array.Resize(ref _gradFlat, cap);
            Array.Resize(ref _paramFlat, cap);
        }
    }

    private void EnsureScratch(int n)
    {
        if (_q.Length < n)
        {
            _q = new double[n];
            _tmpS = new double[n];
            _tmpY = new double[n];
        }
    }
}
