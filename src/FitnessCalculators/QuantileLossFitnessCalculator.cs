namespace AiDotNet.FitnessCalculators;

public class QuantileLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private readonly T _quantile;

    public QuantileLossFitnessCalculator(T? quantile = default, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _quantile = quantile ?? _numOps.FromDouble(0.5);
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.QuantileLoss(dataSet.Predicted, dataSet.Actual, _quantile);
    }
}