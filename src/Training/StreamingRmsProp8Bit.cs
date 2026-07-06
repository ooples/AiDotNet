namespace AiDotNet.Training;

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

