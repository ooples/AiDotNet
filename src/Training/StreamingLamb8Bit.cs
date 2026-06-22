using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

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

