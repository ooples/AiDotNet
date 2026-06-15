using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

internal sealed class StreamingLars8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _momentum;
    private readonly double _weightDecay;
    private readonly double _trustCoefficient;
    private readonly double _epsilon;
    private readonly bool _useNesterov;
    private double _localLearningRate;

    public StreamingLars8Bit(
        double learningRate,
        double momentum,
        double weightDecay,
        double trustCoefficient,
        double epsilon,
        bool useNesterov)
        : base(nameof(StreamingLars8Bit<T>), learningRate, new[] { true })
    {
        _momentum = momentum;
        _weightDecay = weightDecay;
        _trustCoefficient = trustCoefficient;
        _epsilon = epsilon;
        _useNesterov = useNesterov;
        _localLearningRate = learningRate;
    }

    protected override void PrepareParameter(Tensor<T> param, Tensor<T> grad, QuantizedState state, int length)
    {
        double paramNorm = L2Norm(param);
        double gradNorm = L2Norm(grad);
        if (paramNorm > 0.0 && gradNorm > 0.0)
        {
            double denom = gradNorm + _weightDecay * paramNorm + _epsilon;
            _localLearningRate = LearningRate * _trustCoefficient * paramNorm / denom;
        }
        else
        {
            _localLearningRate = LearningRate;
        }
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double gradWithDecay = gradient + _weightDecay * parameter;
        double scaledGrad = _localLearningRate * gradWithDecay;
        double velocity = _momentum * moments[0] + scaledGrad;
        nextMoments[0] = velocity;

        double update = _useNesterov ? _momentum * velocity + scaledGrad : velocity;
        return parameter - ClampUpdate(update);
    }
}

