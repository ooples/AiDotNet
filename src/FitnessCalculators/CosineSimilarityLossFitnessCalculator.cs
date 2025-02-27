namespace AiDotNet.FitnessCalculators;

public class CosineSimilarityLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public CosineSimilarityLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.CosineSimilarityLoss(dataSet.Predicted, dataSet.Actual);
    }
}