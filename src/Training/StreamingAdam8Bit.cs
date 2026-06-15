namespace AiDotNet.Training;

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

