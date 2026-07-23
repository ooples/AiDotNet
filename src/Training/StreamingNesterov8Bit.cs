namespace AiDotNet.Training;

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

