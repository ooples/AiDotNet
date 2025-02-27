namespace AiDotNet.FitnessCalculators;

public class HuberLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private readonly T _delta;

    public HuberLossFitnessCalculator(T? delta = default, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _delta = delta ?? _numOps.FromDouble(1.0);
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.HuberLoss(dataSet.Predicted, dataSet.Actual, _delta);
    }
}