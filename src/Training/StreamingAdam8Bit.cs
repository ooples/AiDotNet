using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Optimizer-in-backward contract for the memory-bounded streaming training path.
/// </summary>
internal interface IStreamingOptimizer<T>
{
    void BeginStep();
    void Apply(Tensor<T> param, Tensor<T> grad);
}

internal interface IStreamingOptimizerLearningRate
{
    void SetLearningRate(double learningRate);
}

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
                if (signedMoments[m])
                {
                    Array.Fill(Quantized[m], (byte)128);
                }

                Array.Fill(Scales[m], 1.0);
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

    private void ApplyBlocks(Tensor<T> param, Tensor<T> grad, QuantizedState state, int length)
    {
        int momentCount = _signedMoments.Length;
        var previous = _previousMomentValues.AsSpan(0, momentCount);
        var next = _nextMomentValues.AsSpan(0, momentCount);

        for (int b = 0; b < state.NumBlocks; b++)
        {
            int start = b * BlockSize;
            int end = Math.Min(start + BlockSize, length);
            Array.Clear(_momentMax, 0, _momentMax.Length);

            for (int i = start; i < end; i++)
            {
                int local = i - start;
                for (int m = 0; m < momentCount; m++)
                {
                    double value = Dequantize(state, m, i);
                    _previousMomentValues[m] = value;
                    _nextMomentValues[m] = value;
                }

                double gradientValue = ToDouble(grad[i]);
                if (!double.IsNaN(gradientValue) && !double.IsInfinity(gradientValue))
                {
                    double parameterValue = ToDouble(param[i]);
                    double nextParameter = UpdateElement(parameterValue, gradientValue, previous, next, i);
                    if (!double.IsNaN(nextParameter) && !double.IsInfinity(nextParameter))
                    {
                        param[i] = FromDouble(nextParameter);
                    }
                }

                for (int m = 0; m < momentCount; m++)
                {
                    double value = _nextMomentValues[m];
                    _momentScratch[m][local] = value;
                    double magnitude = _signedMoments[m] ? Math.Abs(value) : Math.Max(0.0, value);
                    if (magnitude > _momentMax[m]) _momentMax[m] = magnitude;
                }
            }

            for (int m = 0; m < momentCount; m++)
            {
                double divisor = _signedMoments[m] ? 127.0 : 255.0;
                double scale = _momentMax[m] / divisor;
                if (scale < 1e-10 || double.IsNaN(scale) || double.IsInfinity(scale))
                    scale = 1e-10;
                state.Scales[m][b] = scale;

                double invScale = 1.0 / scale;
                for (int i = start; i < end; i++)
                {
                    int local = i - start;
                    double value = _momentScratch[m][local];
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
    }
}

internal sealed class StreamingSgd8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    public StreamingSgd8Bit(double learningRate)
        : base(nameof(StreamingSgd8Bit<T>), learningRate, Array.Empty<bool>())
    {
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
        => parameter - ClampUpdate(LearningRate * gradient);
}

internal sealed class StreamingMomentum8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _momentum;

    public StreamingMomentum8Bit(double learningRate, double momentum)
        : base(nameof(StreamingMomentum8Bit<T>), learningRate, new[] { true })
    {
        _momentum = momentum;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double velocity = _momentum * moments[0] + LearningRate * gradient;
        nextMoments[0] = velocity;
        return parameter - ClampUpdate(velocity);
    }
}

internal sealed class StreamingNesterov8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _momentum;

    public StreamingNesterov8Bit(double learningRate, double momentum)
        : base(nameof(StreamingNesterov8Bit<T>), learningRate, new[] { true })
    {
        _momentum = momentum;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double velocity = _momentum * moments[0] + LearningRate * gradient;
        double nesterovUpdate = _momentum * velocity + LearningRate * gradient;
        nextMoments[0] = velocity;
        return parameter - ClampUpdate(nesterovUpdate);
    }
}

internal sealed class StreamingRmsProp8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _decay;
    private readonly double _epsilon;

    public StreamingRmsProp8Bit(double learningRate, double decay, double epsilon)
        : base(nameof(StreamingRmsProp8Bit<T>), learningRate, new[] { false })
    {
        _decay = decay;
        _epsilon = epsilon;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double sq = _decay * moments[0] + (1.0 - _decay) * gradient * gradient;
        nextMoments[0] = sq;
        double update = LearningRate * gradient / (Math.Sqrt(sq) + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingAdagrad8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _epsilon;

    public StreamingAdagrad8Bit(double learningRate, double epsilon)
        : base(nameof(StreamingAdagrad8Bit<T>), learningRate, new[] { false })
    {
        _epsilon = epsilon;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double acc = moments[0] + gradient * gradient;
        nextMoments[0] = acc;
        double update = LearningRate * gradient / (Math.Sqrt(acc) + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingAdaDelta8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _rho;
    private readonly double _epsilon;

    public StreamingAdaDelta8Bit(double learningRate, double rho, double epsilon)
        : base(nameof(StreamingAdaDelta8Bit<T>), learningRate, new[] { false, false })
    {
        _rho = rho;
        _epsilon = epsilon;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double accGrad = _rho * moments[0] + (1.0 - _rho) * gradient * gradient;
        double delta = Math.Sqrt(moments[1] + _epsilon) / Math.Sqrt(accGrad + _epsilon) * gradient;
        double accUpdate = _rho * moments[1] + (1.0 - _rho) * delta * delta;

        nextMoments[0] = accGrad;
        nextMoments[1] = accUpdate;
        return parameter - ClampUpdate(LearningRate * delta);
    }
}

/// <summary>
/// Per-parameter 8-bit Adam optimizer state for the memory-bounded streaming path.
/// </summary>
internal class StreamingAdam8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;

    public StreamingAdam8Bit(
        double learningRate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8,
        double weightDecay = 0.0,
        int blockSize = 2048,
        double maxUpdateRatio = 5.0)
        : base(nameof(StreamingAdam8Bit<T>), learningRate, new[] { true, false }, blockSize, maxUpdateRatio)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double m = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * gradient * gradient;
        nextMoments[0] = m;
        nextMoments[1] = v;

        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, Step);
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0.0) biasCorr2 = 1.0;

        parameter = ApplyDecoupledWeightDecay(parameter, _weightDecay);
        double update = LearningRate * (m / biasCorr1) / (Math.Sqrt(v / biasCorr2) + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingAdamW8Bit<T> : StreamingAdam8Bit<T>
{
    public StreamingAdamW8Bit(
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        double weightDecay)
        : base(learningRate, beta1, beta2, epsilon, weightDecay)
    {
    }
}

internal sealed class StreamingAMSGrad8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;

    public StreamingAMSGrad8Bit(double learningRate, double beta1, double beta2, double epsilon, double weightDecay)
        : base(nameof(StreamingAMSGrad8Bit<T>), learningRate, new[] { true, false, false })
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double m = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * gradient * gradient;
        double vMax = Math.Max(moments[2], v);
        nextMoments[0] = m;
        nextMoments[1] = v;
        nextMoments[2] = vMax;

        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, Step);
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0.0) biasCorr2 = 1.0;

        parameter = ApplyDecoupledWeightDecay(parameter, _weightDecay);
        double update = LearningRate * (m / biasCorr1) / (Math.Sqrt(vMax / biasCorr2) + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingNadam8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    public StreamingNadam8Bit(double learningRate, double beta1, double beta2, double epsilon)
        : base(nameof(StreamingNadam8Bit<T>), learningRate, new[] { true, false })
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double m = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * gradient * gradient;
        nextMoments[0] = m;
        nextMoments[1] = v;

        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, Step);
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0.0) biasCorr2 = 1.0;

        double mHat = m / biasCorr1;
        double vHat = v / biasCorr2;
        double nesterov = _beta1 * mHat + (1.0 - _beta1) * gradient / biasCorr1;
        double update = LearningRate * nesterov / (Math.Sqrt(vHat) + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingAdaMax8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    public StreamingAdaMax8Bit(double learningRate, double beta1, double beta2, double epsilon)
        : base(nameof(StreamingAdaMax8Bit<T>), learningRate, new[] { true, false })
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double m = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double u = Math.Max(_beta2 * moments[1], Math.Abs(gradient));
        nextMoments[0] = m;
        nextMoments[1] = u;

        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;

        double update = (LearningRate / biasCorr1) * m / (u + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingLion8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _weightDecay;

    public StreamingLion8Bit(double learningRate, double beta1, double beta2, double weightDecay)
        : base(nameof(StreamingLion8Bit<T>), learningRate, new[] { true })
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _weightDecay = weightDecay;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double updateDirection = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double momentum = _beta2 * moments[0] + (1.0 - _beta2) * gradient;
        nextMoments[0] = momentum;

        parameter = ApplyDecoupledWeightDecay(parameter, _weightDecay);
        double update = LearningRate * Math.Sign(updateDirection);
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingLars8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _momentum;
    private readonly double _weightDecay;
    private readonly double _trustCoefficient;
    private readonly double _epsilon;
    private readonly bool _useNesterov;
    private double _localLearningRate;

    public StreamingLars8Bit(
        double learningRate,
        double momentum,
        double weightDecay,
        double trustCoefficient,
        double epsilon,
        bool useNesterov)
        : base(nameof(StreamingLars8Bit<T>), learningRate, new[] { true })
    {
        _momentum = momentum;
        _weightDecay = weightDecay;
        _trustCoefficient = trustCoefficient;
        _epsilon = epsilon;
        _useNesterov = useNesterov;
        _localLearningRate = learningRate;
    }

    protected override void PrepareParameter(Tensor<T> param, Tensor<T> grad, QuantizedState state, int length)
    {
        double paramNorm = L2Norm(param);
        double gradNorm = L2Norm(grad);
        if (paramNorm > 0.0 && gradNorm > 0.0)
        {
            double denom = gradNorm + _weightDecay * paramNorm + _epsilon;
            _localLearningRate = LearningRate * _trustCoefficient * paramNorm / denom;
        }
        else
        {
            _localLearningRate = LearningRate;
        }
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double gradWithDecay = gradient + _weightDecay * parameter;
        double scaledGrad = _localLearningRate * gradWithDecay;
        double velocity = _momentum * moments[0] + scaledGrad;
        nextMoments[0] = velocity;

        double update = _useNesterov ? _momentum * velocity + scaledGrad : velocity;
        return parameter - ClampUpdate(update);
    }
}

internal sealed class StreamingLamb8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;
    private readonly bool _clipTrustRatio;
    private readonly double _maxTrustRatio;
    private readonly bool _useBiasCorrection;
    private double _trustRatio = 1.0;

    public StreamingLamb8Bit(
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        double weightDecay,
        bool clipTrustRatio,
        double maxTrustRatio,
        bool useBiasCorrection)
        : base(nameof(StreamingLamb8Bit<T>), learningRate, new[] { true, false })
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _clipTrustRatio = clipTrustRatio;
        _maxTrustRatio = maxTrustRatio;
        _useBiasCorrection = useBiasCorrection;
    }

    protected override void PrepareParameter(Tensor<T> param, Tensor<T> grad, QuantizedState state, int length)
    {
        double paramNormSq = 0.0;
        double updateNormSq = 0.0;
        double biasCorr1 = _useBiasCorrection ? 1.0 - Math.Pow(_beta1, Step) : 1.0;
        double biasCorr2 = _useBiasCorrection ? 1.0 - Math.Pow(_beta2, Step) : 1.0;
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0.0) biasCorr2 = 1.0;

        for (int i = 0; i < length; i++)
        {
            double p = ToDouble(param[i]);
            double g = ToDouble(grad[i]);
            if (double.IsNaN(g) || double.IsInfinity(g)) continue;

            double m = _beta1 * Dequantize(state, 0, i) + (1.0 - _beta1) * g;
            double v = _beta2 * Dequantize(state, 1, i) + (1.0 - _beta2) * g * g;
            double update = (m / biasCorr1) / (Math.Sqrt(v / biasCorr2) + _epsilon) + _weightDecay * p;
            paramNormSq += p * p;
            updateNormSq += update * update;
        }

        double paramNorm = Math.Sqrt(paramNormSq);
        double updateNorm = Math.Sqrt(updateNormSq);
        if (paramNorm > 0.0 && updateNorm > 0.0)
        {
            _trustRatio = paramNorm / updateNorm;
            if (_clipTrustRatio && _trustRatio > _maxTrustRatio)
                _trustRatio = _maxTrustRatio;
        }
        else
        {
            _trustRatio = 1.0;
        }
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double m = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * gradient * gradient;
        nextMoments[0] = m;
        nextMoments[1] = v;

        double biasCorr1 = _useBiasCorrection ? 1.0 - Math.Pow(_beta1, Step) : 1.0;
        double biasCorr2 = _useBiasCorrection ? 1.0 - Math.Pow(_beta2, Step) : 1.0;
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0.0) biasCorr2 = 1.0;

        double update = (m / biasCorr1) / (Math.Sqrt(v / biasCorr2) + _epsilon) + _weightDecay * parameter;
        return parameter - ClampUpdate(LearningRate * _trustRatio * update);
    }
}

internal sealed class StreamingFtrl8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _alpha;
    private readonly double _beta;
    private readonly double _lambda1;
    private readonly double _lambda2;

    public StreamingFtrl8Bit(double alpha, double beta, double lambda1, double lambda2)
        : base(nameof(StreamingFtrl8Bit<T>), alpha, new[] { true, false })
    {
        _alpha = alpha;
        _beta = beta;
        _lambda1 = lambda1;
        _lambda2 = lambda2;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double z = moments[0];
        double n = moments[1];
        double nNew = n + gradient * gradient;
        double sigma = (Math.Sqrt(nNew) - Math.Sqrt(n)) / _alpha;
        double zNew = z + gradient - sigma * parameter;

        nextMoments[0] = zNew;
        nextMoments[1] = nNew;

        if (Math.Abs(zNew) <= _lambda1)
            return 0.0;

        double numerator = -Math.Sign(zNew) * (Math.Abs(zNew) - _lambda1);
        double denominator = _lambda2 + (Math.Sqrt(nNew) + _beta) / _alpha;
        return numerator / denominator;
    }
}

internal static class StreamingOptimizerResolver<T>
{
    public static string BuildKey(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        bool useStreamingDefaults,
        double streamingWeightDecay)
    {
        string type = useStreamingDefaults ? "DefaultAdam" : optimizer.GetType().FullName ?? optimizer.GetType().Name;
        object? options = GetOptionsObject(optimizer);
        return string.Join("|",
            type,
            ReadDouble(options, "Beta1", 0.9),
            ReadDouble(options, "Beta2", 0.999),
            ReadDouble(options, "Epsilon", 1e-8),
            ReadDouble(options, "Decay", 0.9),
            ReadDouble(options, "Rho", 0.95),
            ReadDouble(options, "InitialMomentum", ReadDouble(options, "Momentum", 0.9)),
            ReadDouble(options, "WeightDecay", streamingWeightDecay),
            ReadDouble(options, "TrustCoefficient", 0.001),
            ReadDouble(options, "Alpha", 0.005),
            ReadDouble(options, "Lambda1", 1.0),
            ReadDouble(options, "Lambda2", 1.0),
            ReadBool(options, "UseAMSGrad", false),
            ReadBool(options, "UseNesterov", false),
            ReadBool(options, "ClipTrustRatio", true),
            ReadDouble(options, "MaxTrustRatio", 10.0),
            ReadBool(options, "UseBiasCorrection", true));
    }

    public static IStreamingOptimizer<T> Create(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        bool useStreamingDefaults,
        double fallbackLearningRate,
        double fallbackWeightDecay)
    {
        object? options = GetOptionsObject(optimizer);
        double lr = useStreamingDefaults ? fallbackLearningRate : ResolveLearningRate(optimizer, fallbackLearningRate);

        if (useStreamingDefaults)
        {
            return new StreamingAdam8Bit<T>(lr, weightDecay: fallbackWeightDecay);
        }

        switch (optimizer)
        {
            case AdamWOptimizer<T, Tensor<T>, Tensor<T>>:
                if (ReadBool(options, "UseAMSGrad", false))
                    return new StreamingAMSGrad8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-8), ReadDouble(options, "WeightDecay", 0.01));
                return new StreamingAdamW8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-8), ReadDouble(options, "WeightDecay", 0.01));
            case AdamOptimizer<T, Tensor<T>, Tensor<T>>:
                if (ReadBool(options, "UseAMSGrad", false))
                    return new StreamingAMSGrad8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-8), fallbackWeightDecay);
                return new StreamingAdam8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-8), fallbackWeightDecay);
            case AMSGradOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingAMSGrad8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-8), 0.0);
            case NadamOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingNadam8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-8));
            case AdaMaxOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingAdaMax8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-8));
            case LionOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingLion8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.99), ReadDouble(options, "WeightDecay", 0.0));
            case LAMBOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingLamb8Bit<T>(lr, ReadDouble(options, "Beta1", 0.9), ReadDouble(options, "Beta2", 0.999), ReadDouble(options, "Epsilon", 1e-6), ReadDouble(options, "WeightDecay", 0.01), ReadBool(options, "ClipTrustRatio", true), ReadDouble(options, "MaxTrustRatio", 10.0), ReadBool(options, "UseBiasCorrection", true));
            case LARSOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingLars8Bit<T>(lr, ReadDouble(options, "Momentum", 0.9), ReadDouble(options, "WeightDecay", 1e-4), ReadDouble(options, "TrustCoefficient", 0.001), ReadDouble(options, "Epsilon", 1e-8), ReadBool(options, "UseNesterov", false));
            case FTRLOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingFtrl8Bit<T>(ReadDouble(options, "Alpha", lr), ReadDouble(options, "Beta", 1.0), ReadDouble(options, "Lambda1", 1.0), ReadDouble(options, "Lambda2", 1.0));
            case AdaDeltaOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingAdaDelta8Bit<T>(lr, ReadDouble(options, "Rho", 0.95), ReadDouble(options, "Epsilon", 1e-6));
            case AdagradOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingAdagrad8Bit<T>(lr, ReadDouble(options, "Epsilon", 1e-8));
            case RootMeanSquarePropagationOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingRmsProp8Bit<T>(lr, ReadDouble(options, "Decay", 0.9), ReadDouble(options, "Epsilon", 1e-8));
            case MomentumOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingMomentum8Bit<T>(lr, ReadDouble(options, "InitialMomentum", 0.9));
            case NesterovAcceleratedGradientOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingNesterov8Bit<T>(lr, ReadDouble(options, "InitialMomentum", 0.9));
            case GradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
            case StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
            case MiniBatchGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
            case ProximalGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingSgd8Bit<T>(lr);
            default:
                return new StreamingAdam8Bit<T>(fallbackLearningRate, weightDecay: fallbackWeightDecay);
        }
    }

    public static void RefreshLearningRate(
        IStreamingOptimizer<T> streamingOptimizer,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        bool useStreamingDefaults,
        double fallbackLearningRate)
    {
        if (streamingOptimizer is IStreamingOptimizerLearningRate mutable)
        {
            mutable.SetLearningRate(useStreamingDefaults
                ? fallbackLearningRate
                : ResolveLearningRate(optimizer, fallbackLearningRate));
        }
    }

    private static double ResolveLearningRate(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        double fallbackLearningRate)
    {
        if (optimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> gradientBased)
        {
            return gradientBased.GetCurrentLearningRate();
        }

        return fallbackLearningRate;
    }

    private static object? GetOptionsObject(object optimizer)
    {
        Type? type = optimizer.GetType();
        while (type is not null)
        {
            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.NonPublic))
            {
                if (!field.Name.Contains("options", StringComparison.OrdinalIgnoreCase))
                    continue;
                object? value = field.GetValue(optimizer);
                if (value is not null)
                    return value;
            }

            type = type.BaseType;
        }

        return null;
    }

    private static double ReadDouble(object? options, string propertyName, double fallback)
    {
        object? value = ReadProperty(options, propertyName);
        if (value is null) return fallback;
        try
        {
            return Convert.ToDouble(value);
        }
        catch
        {
            return fallback;
        }
    }

    private static bool ReadBool(object? options, string propertyName, bool fallback)
    {
        object? value = ReadProperty(options, propertyName);
        return value is bool b ? b : fallback;
    }

    private static object? ReadProperty(object? options, string propertyName)
    {
        if (options is null) return null;
        var property = options.GetType().GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public);
        return property?.GetValue(options);
    }
}
