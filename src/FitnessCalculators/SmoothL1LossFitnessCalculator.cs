namespace AiDotNet.FitnessCalculators;

public class SmoothL1LossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public SmoothL1LossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.SmoothL1Loss(dataSet.Predicted, dataSet.Actual);
    }
}