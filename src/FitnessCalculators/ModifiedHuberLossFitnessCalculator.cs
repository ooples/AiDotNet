namespace AiDotNet.FitnessCalculators;

public class ModifiedHuberLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public ModifiedHuberLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.ModifiedHuberLoss(dataSet.Predicted, dataSet.Actual);
    }
}