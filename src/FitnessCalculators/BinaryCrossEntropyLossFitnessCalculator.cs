namespace AiDotNet.FitnessCalculators;

public class BinaryCrossEntropyLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public BinaryCrossEntropyLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.BinaryCrossEntropyLoss(dataSet.Predicted, dataSet.Actual);
    }
}