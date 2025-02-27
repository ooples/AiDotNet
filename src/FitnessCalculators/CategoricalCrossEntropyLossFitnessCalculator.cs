namespace AiDotNet.FitnessCalculators;

public class CategoricalCrossEntropyLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    public CategoricalCrossEntropyLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.CategoricalCrossEntropyLoss(
            Matrix<T>.FromVector(dataSet.Predicted),
            Matrix<T>.FromVector(dataSet.Actual)
        );
    }
}