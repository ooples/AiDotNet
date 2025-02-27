namespace AiDotNet.FitnessCalculators;

public class DiceLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public DiceLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.DiceLoss(dataSet.Predicted, dataSet.Actual);
    }
}