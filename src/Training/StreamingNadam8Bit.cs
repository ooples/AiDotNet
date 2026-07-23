namespace AiDotNet.Training;

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

