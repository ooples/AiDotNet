namespace AiDotNet.LinearAlgebra;

public abstract class FitnessCalculatorBase : IFitnessCalculator
{
    protected bool _isHigherScoreBetter;

    protected FitnessCalculatorBase(bool isHigherScoreBetter)
    {
        _isHigherScoreBetter = isHigherScoreBetter;
    }

    public abstract double CalculateFitnessScore(
        ErrorStats errorStats, 
        BasicStats basicStats, 
        Vector<double> actualValues, 
        Vector<double> predictedValues,
        Matrix<double> features);

    public bool IsHigherScoreBetter => _isHigherScoreBetter;

    public bool IsBetterFitness(double newScore, double currentBestScore)
    {
        return _isHigherScoreBetter ? newScore > currentBestScore : newScore < currentBestScore;
    }
}