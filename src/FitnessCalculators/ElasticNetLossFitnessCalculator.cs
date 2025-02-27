namespace AiDotNet.FitnessCalculators;

public class ElasticNetLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private readonly T _l1Ratio;
    private readonly T _alpha;

    public ElasticNetLossFitnessCalculator(T? l1Ratio = default, T? alpha = default, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _l1Ratio = l1Ratio ?? _numOps.FromDouble(0.5);
        _alpha = alpha ?? _numOps.FromDouble(1.0);
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        return NeuralNetworkHelper<T>.ElasticNetLoss(dataSet.Predicted, dataSet.Actual, _l1Ratio, _alpha);
    }
}