namespace AiDotNet.FitnessCalculators;

public class LogCoshLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public LogCoshLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.LogCoshLoss(dataSet.Predicted, dataSet.Actual);
    }
}