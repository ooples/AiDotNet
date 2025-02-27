namespace AiDotNet.FitnessCalculators;

public class WeightedCrossEntropyLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private Vector<T>? _weights;

    public WeightedCrossEntropyLossFitnessCalculator(Vector<T>? weights = default, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _weights = weights;
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        if (_weights == null || _weights.Length != dataSet.Actual.Length)
        {
            _weights = new Vector<T>(dataSet.Actual.Length);
        }

        return NeuralNetworkHelper<T>.WeightedCrossEntropyLoss(dataSet.Predicted, dataSet.Actual, _weights);
    }
}