namespace AiDotNet.Training;

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

