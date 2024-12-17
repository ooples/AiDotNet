namespace AiDotNet.FitnessCalculators;

public class RMSEFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public RMSEFitnessCalculator() : base(isHigherScoreBetter: false)
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
        return errorStats.RMSE;
    }
}