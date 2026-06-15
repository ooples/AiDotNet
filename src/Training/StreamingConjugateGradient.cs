using AiDotNet.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Memory-bounded streaming nonlinear Conjugate Gradient — a Hessian-free, second-order-like entry
/// in the streaming optimizer family.
/// </summary>
/// <remarks>
/// <para>
/// Like streaming L-BFGS, CG is a full-gradient method: the search direction depends on the WHOLE
/// gradient plus the previous direction, so it can't update per-parameter-as-gradient-arrives. It
/// buffers the per-parameter gradients into a flat vector during <see cref="Apply"/> and performs
/// the direction computation + parameter update once per step in <see cref="EndStep"/>.
/// </para>
/// <para>
/// Direction: Polak–Ribière+ with automatic restart — <c>d_k = -g_k + β_k · d_{k-1}</c>,
/// <c>β_k = max(0, g_k·(g_k - g_{k-1}) / (g_{k-1}·g_{k-1}))</c>. Clamping β at 0 restarts the search
/// along steepest descent whenever conjugacy degrades, which keeps the iteration stable without a
/// line search. The first step is plain steepest descent.
/// </para>
/// <para>
/// <b>The memory win:</b> CG's only persistent state is the previous gradient and previous
/// direction — O(n) each. Both are stored 8-bit block-quantized via <see cref="QuantizedVector"/>
/// (~4× smaller than fp32) and dequantized on demand into reused O(n) scratch, so the method's
/// transient footprint stays O(n).
/// </para>
/// </remarks>
internal sealed class StreamingConjugateGradient<T> : IStreamingOptimizer<T>, IStreamingOptimizerLearningRate
{
    private readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();
    private readonly int _blockSize;
    private double _learningRate;

    // Stable per-step layout of the parameter tensors + their slices in the flat gradient buffer.
    private readonly List<(Tensor<T> Param, int Offset, int Length)> _layout = new();
    private double[] _gradFlat = Array.Empty<double>();
    private int _writeOffset;

    // 8-bit block-quantized O(n) state (the memory win): previous gradient + previous direction.
    private QuantizedVector? _prevGrad;
    private QuantizedVector? _prevDir;

    // Reused O(n) scratch so dequantization stays O(n) transient.
    private double[] _dir = Array.Empty<double>();
    private double[] _tmpG = Array.Empty<double>();
    private double[] _tmpD = Array.Empty<double>();

    public StreamingConjugateGradient(double learningRate, int blockSize = 2048)
    {
        _learningRate = learningRate;
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
            throw new ArgumentException($"StreamingConjugateGradient.Apply: gradient length {grad.Length} does not match param length {len}.", nameof(grad));

        EnsureFlatCapacity(_writeOffset + len);
        _layout.Add((param, _writeOffset, len));
        for (int i = 0; i < len; i++)
            _gradFlat[_writeOffset + i] = _ops.ToDouble(grad[i]);
        _writeOffset += len;
    }

    public void EndStep()
    {
        int n = _writeOffset;
        if (n == 0) return;
        EnsureScratch(n);

        if (_prevGrad is null || _prevDir is null)
        {
            // First step: steepest descent.
            for (int i = 0; i < n; i++) _dir[i] = -_gradFlat[i];
        }
        else
        {
            _prevGrad.Dequantize(_tmpG, n); // g_{k-1}
            _prevDir.Dequantize(_tmpD, n);  // d_{k-1}
            double denom = Dot(_tmpG, _tmpG, n);
            double num = 0.0;
            for (int i = 0; i < n; i++) num += _gradFlat[i] * (_gradFlat[i] - _tmpG[i]);
            double beta = denom > 1e-30 ? num / denom : 0.0;
            if (beta < 0.0 || double.IsNaN(beta) || double.IsInfinity(beta)) beta = 0.0; // PR+ restart
            for (int i = 0; i < n; i++) _dir[i] = -_gradFlat[i] + beta * _tmpD[i];
        }

        // x <- x + lr * d, written back to the live tensors (guarded against non-finite steps).
        Scatter(_dir);

        // Roll history forward, 8-bit quantized.
        _prevGrad = QuantizedVector.Quantize(_gradFlat, n, _blockSize);
        _prevDir = QuantizedVector.Quantize(_dir, n, _blockSize);
    }

    private void Scatter(double[] dir)
    {
        foreach (var (param, offset, length) in _layout)
        {
            for (int i = 0; i < length; i++)
            {
                double step = _learningRate * dir[offset + i];
                if (double.IsNaN(step) || double.IsInfinity(step)) continue;
                double next = _ops.ToDouble(param[i]) + step;
                if (!double.IsNaN(next) && !double.IsInfinity(next))
                    param[i] = _ops.FromDouble(next);
            }
        }
    }

    private static double Dot(double[] a, double[] b, int n)
    {
        double s = 0.0;
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    private void EnsureFlatCapacity(int needed)
    {
        if (_gradFlat.Length < needed)
        {
            int cap = Math.Max(needed, _gradFlat.Length == 0 ? needed : _gradFlat.Length * 2);
            Array.Resize(ref _gradFlat, cap);
        }
    }

    private void EnsureScratch(int n)
    {
        if (_dir.Length < n)
        {
            _dir = new double[n];
            _tmpG = new double[n];
            _tmpD = new double[n];
        }
    }
}
