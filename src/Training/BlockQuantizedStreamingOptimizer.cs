using AiDotNet.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Shared per-parameter block-quantized state machinery for streaming optimizers.
/// Concrete optimizers only provide the per-element update rule and moment layout.
/// </summary>
internal abstract class BlockQuantizedStreamingOptimizer<T> : IStreamingOptimizer<T>, IStreamingOptimizerLearningRate
{
    protected sealed class QuantizedState
    {
        public readonly byte[][] Quantized;
        public readonly double[][] Scales;
        public readonly int Length;
        public readonly int NumBlocks;

        public QuantizedState(int momentCount, int length, int numBlocks, IReadOnlyList<bool> signedMoments)
        {
            Length = length;
            NumBlocks = numBlocks;
            Quantized = new byte[momentCount][];
            Scales = new double[momentCount][];

            for (int m = 0; m < momentCount; m++)
            {
                Quantized[m] = new byte[length];
                Scales[m] = new double[numBlocks];
                // net471 has no Array.Fill — initialize with loops. Signed moments quantize 0 as
                // byte 128 (the zero point); unsigned default to 0. All block scales start at 1.0.
                if (signedMoments[m])
                {
                    for (int i = 0; i < length; i++) Quantized[m][i] = 128;
                }

                for (int i = 0; i < numBlocks; i++) Scales[m][i] = 1.0;
            }
        }
    }

    private readonly Dictionary<Tensor<T>, QuantizedState> _state =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();
    private readonly bool[] _signedMoments;
    private readonly double[] _previousMomentValues;
    private readonly double[] _nextMomentValues;
    private readonly double[][] _momentScratch;
    private readonly double[] _momentMax;
    private readonly string _optimizerName;

    protected readonly int BlockSize;
    protected readonly double MaxUpdateRatio;
    protected int Step { get; private set; }
    protected double LearningRate { get; private set; }

    protected BlockQuantizedStreamingOptimizer(
        string optimizerName,
        double learningRate,
        bool[] signedMoments,
        int blockSize = 2048,
        double maxUpdateRatio = 5.0)
    {
        _optimizerName = optimizerName;
        LearningRate = learningRate;
        _signedMoments = signedMoments;
        BlockSize = Math.Max(1, blockSize);
        MaxUpdateRatio = maxUpdateRatio > 0 ? maxUpdateRatio : 5.0;

        _previousMomentValues = new double[signedMoments.Length];
        _nextMomentValues = new double[signedMoments.Length];
        _momentScratch = new double[signedMoments.Length][];
        _momentMax = new double[signedMoments.Length];
        for (int i = 0; i < signedMoments.Length; i++)
        {
            _momentScratch[i] = new double[BlockSize];
        }
    }

    public void SetLearningRate(double learningRate)
    {
        if (learningRate > 0.0 && !double.IsNaN(learningRate) && !double.IsInfinity(learningRate))
        {
            LearningRate = learningRate;
        }
    }

    public void BeginStep()
    {
        Step++;
        OnBeginStep();
    }

    protected virtual void OnBeginStep()
    {
    }

    // First-order optimizers update in place during Apply, so the post-step hook is a no-op.
    public virtual void EndStep()
    {
    }

    public virtual void Apply(Tensor<T> param, Tensor<T> grad)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));

        int length = param.Length;
        if (length == 0)
        {
            throw new ArgumentException(
                $"{_optimizerName}.Apply: param is empty (length == 0); a zero-length parameter should never be registered as a training source.",
                nameof(param));
        }

        if (grad.Length != length)
        {
            throw new ArgumentException(
                $"{_optimizerName}.Apply: gradient length {grad.Length} does not match param length {length}.",
                nameof(grad));
        }

        int numBlocks = (length + BlockSize - 1) / BlockSize;
        if (!_state.TryGetValue(param, out var state) || state.Length != length)
        {
            state = new QuantizedState(_signedMoments.Length, length, numBlocks, _signedMoments);
            _state[param] = state;
        }

        PrepareParameter(param, grad, state, length);
        ApplyBlocks(param, grad, state, length);
    }

    protected virtual void PrepareParameter(Tensor<T> param, Tensor<T> grad, QuantizedState state, int length)
    {
    }

    protected abstract double UpdateElement(
        double parameter,
        double gradient,
        ReadOnlySpan<double> moments,
        Span<double> nextMoments,
        int index);

    protected double ToDouble(T value) => _ops.ToDouble(value);

    protected T FromDouble(double value) => _ops.FromDouble(value);

    protected double Dequantize(QuantizedState state, int moment, int index)
    {
        int block = index / BlockSize;
        double scale = state.Scales[moment][block];
        byte q = state.Quantized[moment][index];
        return _signedMoments[moment] ? (q - 128) * scale : q * scale;
    }

    protected double L2Norm(Tensor<T> tensor)
    {
        double sum = 0.0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = ToDouble(tensor[i]);
            if (double.IsNaN(v) || double.IsInfinity(v)) continue;
            sum += v * v;
        }

        return Math.Sqrt(sum);
    }

    protected double ApplyDecoupledWeightDecay(double parameter, double weightDecay)
    {
        return weightDecay == 0.0 ? parameter : parameter - LearningRate * weightDecay * parameter;
    }

    protected double ClampUpdate(double update)
    {
        double maxStep = LearningRate * MaxUpdateRatio;
        if (maxStep <= 0.0 || double.IsNaN(maxStep) || double.IsInfinity(maxStep))
            return update;

        if (double.IsNaN(update) || double.IsInfinity(update))
            return update > 0 ? maxStep : -maxStep;
        if (update > maxStep) return maxStep;
        if (update < -maxStep) return -maxStep;
        return update;
    }

    // Only parallelize the block loop when there is enough work to amortize thread
    // dispatch. Small parameters (biases, norms) stay on the serial, zero-dispatch path.
    private const int ParallelApplyMinLength = 1 << 15;

    private void ApplyBlocks(Tensor<T> param, Tensor<T> grad, QuantizedState state, int length)
    {
        int momentCount = _signedMoments.Length;

        // Force any lazy-graph realization / streaming-weight rehydration ONCE on this thread,
        // then try to grab the LIVE contiguous backing arrays. For trainable weights and their
        // gradients (simple, contiguous, CPU-resident storage) this lets us update the parameter
        // array IN PLACE with raw indexing — no per-element tensor-indexer overhead (each
        // GetFlat/SetFlat re-checks materialization, bounds, and does an Interlocked version
        // bump) — and bump the version exactly ONCE at the end. Views / GPU-resident / lazy
        // tensors return null here and fall back to the per-element indexer. Either way each
        // block updates a DISJOINT parameter/gradient/state slice, so the loop parallelizes
        // with per-worker scratch and stays deterministic (no O(n) staging buffer, so the
        // memory-bounded design is preserved).
        _ = _ops.ToDouble(param[0]);
        _ = _ops.ToDouble(grad[0]);
        T[]? paramArr = param.GetLiveBackingArrayOrNull();
        T[]? gradArr = grad.GetLiveBackingArrayOrNull();

        if (length < ParallelApplyMinLength || state.NumBlocks < 2 || Environment.ProcessorCount < 2)
        {
            for (int b = 0; b < state.NumBlocks; b++)
            {
                ApplyOneBlock(b, param, grad, paramArr, gradArr, state, length, momentCount,
                    _previousMomentValues, _nextMomentValues, _momentScratch, _momentMax);
            }
        }
        else
        {
            System.Threading.Tasks.Parallel.For(
                0,
                state.NumBlocks,
                () => new BlockScratch(momentCount, BlockSize),
                (b, _, scratch) =>
                {
                    ApplyOneBlock(b, param, grad, paramArr, gradArr, state, length, momentCount,
                        scratch.Previous, scratch.Next, scratch.MomentScratch, scratch.MomentMax);
                    return scratch;
                },
                _ => { });
        }

        // The raw-array write path bypasses the indexer's per-element version bump, so bump
        // once here to invalidate any cached GPU buffer for the now-mutated parameter.
        if (paramArr is not null) param.IncrementVersion();
    }

    private void ApplyOneBlock(
        int b, Tensor<T> param, Tensor<T> grad, T[]? paramArr, T[]? gradArr,
        QuantizedState state, int length, int momentCount,
        double[] previous, double[] next, double[][] momentScratch, double[] momentMax)
    {
        int start = b * BlockSize;
        int end = Math.Min(start + BlockSize, length);
        for (int m = 0; m < momentCount; m++) momentMax[m] = 0.0;

        var previousSpan = previous.AsSpan(0, momentCount);
        var nextSpan = next.AsSpan(0, momentCount);

        for (int i = start; i < end; i++)
        {
            int local = i - start;
            for (int m = 0; m < momentCount; m++)
            {
                double value = Dequantize(state, m, i);
                previous[m] = value;
                next[m] = value;
            }

            double gradientValue = gradArr is not null ? ToDouble(gradArr[i]) : ToDouble(grad[i]);
            if (!double.IsNaN(gradientValue) && !double.IsInfinity(gradientValue))
            {
                double parameterValue = paramArr is not null ? ToDouble(paramArr[i]) : ToDouble(param[i]);
                double nextParameter = UpdateElement(parameterValue, gradientValue, previousSpan, nextSpan, i);
                if (!double.IsNaN(nextParameter) && !double.IsInfinity(nextParameter))
                {
                    if (paramArr is not null) paramArr[i] = FromDouble(nextParameter);
                    else param[i] = FromDouble(nextParameter);
                }
            }

            for (int m = 0; m < momentCount; m++)
            {
                double value = next[m];
                momentScratch[m][local] = value;
                double magnitude = _signedMoments[m] ? Math.Abs(value) : Math.Max(0.0, value);
                if (magnitude > momentMax[m]) momentMax[m] = magnitude;
            }
        }

        for (int m = 0; m < momentCount; m++)
        {
            double divisor = _signedMoments[m] ? 127.0 : 255.0;
            double scale = momentMax[m] / divisor;
            if (scale < 1e-10 || double.IsNaN(scale) || double.IsInfinity(scale))
                scale = 1e-10;
            state.Scales[m][b] = scale;

            double invScale = 1.0 / scale;
            for (int i = start; i < end; i++)
            {
                int local = i - start;
                double value = momentScratch[m][local];
                if (_signedMoments[m])
                {
                    int q = (int)Math.Round(value * invScale);
                    if (q < -127) q = -127;
                    else if (q > 127) q = 127;
                    state.Quantized[m][i] = (byte)(q + 128);
                }
                else
                {
                    int q = (int)Math.Round(Math.Max(0.0, value) * invScale);
                    if (q < 0) q = 0;
                    else if (q > 255) q = 255;
                    state.Quantized[m][i] = (byte)q;
                }
            }
        }
    }

    /// <summary>
    /// Per-worker moment scratch for the parallel block loop. Each parallel worker owns one
    /// instance (allocated once per worker via <c>Parallel.For</c> localInit, reused across the
    /// blocks that worker handles), so no two threads share the requantization scratch.
    /// </summary>
    private sealed class BlockScratch
    {
        public readonly double[] Previous;
        public readonly double[] Next;
        public readonly double[][] MomentScratch;
        public readonly double[] MomentMax;

        public BlockScratch(int momentCount, int blockSize)
        {
            Previous = new double[momentCount];
            Next = new double[momentCount];
            MomentMax = new double[momentCount];
            MomentScratch = new double[momentCount][];
            for (int m = 0; m < momentCount; m++)
                MomentScratch[m] = new double[blockSize];
        }
    }
}

