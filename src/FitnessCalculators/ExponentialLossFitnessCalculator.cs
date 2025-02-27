namespace AiDotNet.FitnessCalculators;

public class ExponentialLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public ExponentialLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.ExponentialLoss(dataSet.Predicted, dataSet.Actual);
    }
}