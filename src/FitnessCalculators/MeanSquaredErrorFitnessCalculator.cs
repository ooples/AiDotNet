namespace AiDotNet.FitnessCalculators;

public class MeanSquaredErrorFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public MeanSquaredErrorFitnessCalculator() : base(isHigherScoreBetter: false)
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
        return errorStats.MSE;
    }
}