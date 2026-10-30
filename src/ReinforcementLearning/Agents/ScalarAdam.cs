namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Adam (Kingma &amp; Ba 2015) for a single scalar parameter, such as SAC's log-temperature or CQL's
/// log-Lagrange multiplier, which the reference implementations train with their own Adam optimizer.
/// </summary>
internal sealed class ScalarAdam
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private double _firstMoment;
    private double _secondMoment;
    private int _step;

    /// <param name="learningRate">The step size; must be positive and finite.</param>
    /// <param name="beta1">Decay of the first-moment estimate.</param>
    /// <param name="beta2">Decay of the second-moment estimate.</param>
    /// <param name="epsilon">Denominator offset.</param>
    public ScalarAdam(double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        if (!(learningRate > 0.0) || double.IsInfinity(learningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), learningRate, "The learning rate must be positive and finite.");
        }

        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    /// <summary>Returns <paramref name="value"/> after one Adam descent step along <paramref name="gradient"/>.</summary>
    public double Step(double value, double gradient)
    {
        if (double.IsNaN(gradient) || double.IsInfinity(gradient))
        {
            return value;
        }

        _step++;
        _firstMoment = _beta1 * _firstMoment + (1.0 - _beta1) * gradient;
        _secondMoment = _beta2 * _secondMoment + (1.0 - _beta2) * gradient * gradient;
        double firstUnbiased = _firstMoment / (1.0 - Math.Pow(_beta1, _step));
        double secondUnbiased = _secondMoment / (1.0 - Math.Pow(_beta2, _step));
        return value - _learningRate * firstUnbiased / (Math.Sqrt(secondUnbiased) + _epsilon);
    }
}
