namespace AiDotNet.Training;

internal sealed class StreamingFtrl8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _alpha;
    private readonly double _beta;
    private readonly double _lambda1;
    private readonly double _lambda2;

    public StreamingFtrl8Bit(double alpha, double beta, double lambda1, double lambda2)
        : base(nameof(StreamingFtrl8Bit<T>), alpha, new[] { true, false })
    {
        _alpha = alpha;
        _beta = beta;
        _lambda1 = lambda1;
        _lambda2 = lambda2;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double z = moments[0];
        double n = moments[1];
        double nNew = n + gradient * gradient;
        double sigma = (Math.Sqrt(nNew) - Math.Sqrt(n)) / _alpha;
        double zNew = z + gradient - sigma * parameter;

        nextMoments[0] = zNew;
        nextMoments[1] = nNew;

        if (Math.Abs(zNew) <= _lambda1)
            return 0.0;

        double numerator = -Math.Sign(zNew) * (Math.Abs(zNew) - _lambda1);
        double denominator = _lambda2 + (Math.Sqrt(nNew) + _beta) / _alpha;
        return numerator / denominator;
    }
}

