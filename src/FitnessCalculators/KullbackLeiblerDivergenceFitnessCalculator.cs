namespace AiDotNet.FitnessCalculators;

public class KullbackLeiblerDivergenceFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public KullbackLeiblerDivergenceFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.KullbackLeiblerDivergence(dataSet.Predicted, dataSet.Actual);
    }
}