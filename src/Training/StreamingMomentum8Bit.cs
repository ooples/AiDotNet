namespace AiDotNet.Training;

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

