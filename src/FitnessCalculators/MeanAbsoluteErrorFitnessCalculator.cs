namespace AiDotNet.FitnessCalculators;

public class MeanAbsoluteErrorFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public MeanAbsoluteErrorFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return dataSet.ErrorStats.MAE;
    }
}