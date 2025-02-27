namespace AiDotNet.FitnessCalculators;

public class AdjustedRSquaredFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public AdjustedRSquaredFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return dataSet.PredictionStats.AdjustedR2;
    }
}