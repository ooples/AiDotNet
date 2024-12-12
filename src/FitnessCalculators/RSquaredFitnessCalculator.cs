namespace AiDotNet.FitnessCalculators;

public class RSquaredFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public RSquaredFitnessCalculator() : base(isHigherScoreBetter: true)
    {
    }

    public override T CalculateFitnessScore(
        ErrorStats<T> errorStats,
        BasicStats<T> basicStats,
        BasicStats<T> targetStats,
        Vector<T> actualValues,
        Vector<T> predictedValues,
        Matrix<T> features,
        PredictionStats<T> predictionStats)
    {
        return predictionStats.R2;
    }
}