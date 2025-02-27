namespace AiDotNet.FitnessCalculators;

public class RootMeanSquaredErrorFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public RootMeanSquaredErrorFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return dataSet.ErrorStats.RMSE;
    }
}