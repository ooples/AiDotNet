namespace AiDotNet.FitnessCalculators;

public class MeanSquaredErrorFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public MeanSquaredErrorFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return dataSet.ErrorStats.MSE;
    }
}