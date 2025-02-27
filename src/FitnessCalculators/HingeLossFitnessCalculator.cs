namespace AiDotNet.FitnessCalculators;

public class HingeLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public HingeLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.HingeLoss(dataSet.Predicted, dataSet.Actual);
    }
}