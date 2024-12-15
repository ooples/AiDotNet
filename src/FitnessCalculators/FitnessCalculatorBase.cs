namespace AiDotNet.FitnessCalculators;

public abstract class FitnessCalculatorBase<T> : IFitnessCalculator<T>
{
    protected bool _isHigherScoreBetter;
    protected readonly INumericOperations<T> _numOps;

    protected FitnessCalculatorBase(bool isHigherScoreBetter)
    {
        _isHigherScoreBetter = isHigherScoreBetter;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public abstract T CalculateFitnessScore(
        ErrorStats<T> errorStats,
        BasicStats<T> basicStats,
        BasicStats<T> predictedStats,
        Vector<T> actualValues,
        Vector<T> predictedValues,
        Matrix<T> features,
        PredictionStats<T> predictionStats);

    public bool IsHigherScoreBetter => _isHigherScoreBetter;

    public bool IsBetterFitness(T newScore, T currentBestScore)
    {
        return _isHigherScoreBetter
            ? _numOps.GreaterThan(newScore, currentBestScore)
            : _numOps.LessThan(newScore, currentBestScore);
    }
}