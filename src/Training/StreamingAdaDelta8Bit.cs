namespace AiDotNet.Training;

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

