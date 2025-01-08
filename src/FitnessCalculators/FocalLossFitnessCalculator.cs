namespace AiDotNet.FitnessCalculators;

public class FocalLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private readonly T _gamma;
    private readonly T _alpha;

    public FocalLossFitnessCalculator(T? gamma = default, T? alpha = default, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _gamma = gamma ?? _numOps.FromDouble(2.0);
        _alpha = alpha ?? _numOps.FromDouble(0.25);
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.FocalLoss(dataSet.Predicted, dataSet.Actual, _gamma, _alpha);
    }
}