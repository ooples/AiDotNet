namespace AiDotNet.FitnessCalculators;

public class SquaredHingeLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public SquaredHingeLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.SquaredHingeLoss(dataSet.Predicted, dataSet.Actual);
    }
}