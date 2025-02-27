namespace AiDotNet.FitnessCalculators;

public class CrossEntropyLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public CrossEntropyLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.CrossEntropyLoss(dataSet.Predicted, dataSet.Actual);
    }
}