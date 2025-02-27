namespace AiDotNet.FitnessCalculators;

public class PoissonLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public PoissonLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.PoissonLoss(dataSet.Predicted, dataSet.Actual);
    }
}