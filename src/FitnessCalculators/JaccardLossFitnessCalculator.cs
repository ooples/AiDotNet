namespace AiDotNet.FitnessCalculators;

public class JaccardLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public JaccardLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.JaccardLoss(dataSet.Predicted, dataSet.Actual);
    }
}