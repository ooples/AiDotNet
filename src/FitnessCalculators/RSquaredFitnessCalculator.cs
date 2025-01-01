namespace AiDotNet.FitnessCalculators;

public class RSquaredFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public RSquaredFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return dataSet.PredictionStats.R2;
    }
}