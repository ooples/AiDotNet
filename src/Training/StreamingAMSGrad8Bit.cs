namespace AiDotNet.Training;

internal sealed class StreamingAMSGrad8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;
    private readonly bool _decoupledWeightDecay;

    public StreamingAMSGrad8Bit(
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        double weightDecay,
        bool decoupledWeightDecay = true,
        int blockSize = 2048,
        bool useBf16Moments = false,
        int min8BitSize = 0)
        : base(nameof(StreamingAMSGrad8Bit<T>), learningRate, new[] { true, false, false }, blockSize,
            useBf16Moments: useBf16Moments, min8BitSize: min8BitSize)
    {
        _decoupledWeightDecay = decoupledWeightDecay;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double effectiveGrad = _decoupledWeightDecay || _weightDecay <= 0.0
            ? gradient
            : gradient + _weightDecay * parameter;

        double m = _beta1 * moments[0] + (1.0 - _beta1) * effectiveGrad;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * effectiveGrad * effectiveGrad;
        double vMax = Math.Max(moments[2], v);
        nextMoments[0] = m;
        nextMoments[1] = v;
        nextMoments[2] = vMax;

        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, Step);
        if (biasCorr1 <= 0.0) biasCorr1 = 1.0;
        if (biasCorr2 <= 0.0) biasCorr2 = 1.0;

        if (_decoupledWeightDecay)
            parameter = ApplyDecoupledWeightDecay(parameter, _weightDecay);
        double update = LearningRate * (m / biasCorr1) / (Math.Sqrt(vMax / biasCorr2) + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

