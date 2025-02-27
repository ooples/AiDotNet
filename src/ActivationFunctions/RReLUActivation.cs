namespace AiDotNet.ActivationFunctions;

public class RReLUActivation<T> : ActivationFunctionBase<T>
{
    private readonly Random _random;
    private readonly T _lowerBound;
    private readonly T _upperBound;
    private T _alpha;
    private bool _isTraining;

    public RReLUActivation(double lowerBound = 1.0 / 8, double upperBound = 1.0 / 3)
    {
        _random = new Random();
        _lowerBound = NumOps.FromDouble(lowerBound);
        _upperBound = NumOps.FromDouble(upperBound);
        _alpha = NumOps.FromDouble((_random.NextDouble() * (upperBound - lowerBound)) + lowerBound);
        _isTraining = true;
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        if (_isTraining)
        {
            _alpha = NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), NumOps.Add(NumOps.Subtract(_upperBound, _lowerBound), _lowerBound));
        }

        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return input;
        }
        else
        {
            return NumOps.Multiply(_alpha, input);
        }
    }

    public override T Derivative(T input)
    {
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return NumOps.One;
        }
        else
        {
            return _alpha;
        }
    }

    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
        if (!_isTraining)
        {
            // Set alpha to the average of lower and upper bounds for inference
            _alpha = NumOps.Divide(NumOps.Add(_lowerBound, _upperBound), NumOps.FromDouble(2));
        }
    }
}