namespace AiDotNet.Training;

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

