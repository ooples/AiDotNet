namespace AiDotNet.FitnessCalculators;

public abstract class FitnessCalculatorBase<T> : IFitnessCalculator<T>
{
    protected bool _isHigherScoreBetter;
    protected readonly INumericOperations<T> _numOps;
    protected readonly DataSetType _dataSetType;

    protected FitnessCalculatorBase(bool isHigherScoreBetter, DataSetType dataSetType = DataSetType.Validation)
    {
        _isHigherScoreBetter = isHigherScoreBetter;
        _numOps = MathHelper.GetNumericOperations<T>();
        _dataSetType = dataSetType;
    }

    public T CalculateFitnessScore(ModelEvaluationData<T> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        DataSetStats<T> dataSet = _dataSetType switch
        {
            DataSetType.Training => evaluationData.TrainingSet,
            DataSetType.Validation => evaluationData.ValidationSet,
            DataSetType.Testing => evaluationData.TestSet,
            _ => throw new ArgumentException($"Unsupported DataSetType: {_dataSetType}")
        };

        return dataSet == null
            ? throw new InvalidOperationException($"The {_dataSetType} dataset is not available in the provided ModelEvaluationData.")
            : GetFitnessScore(dataSet);
    }

    protected abstract T GetFitnessScore(DataSetStats<T> dataSet);

    public bool IsHigherScoreBetter => _isHigherScoreBetter;

    public bool IsBetterFitness(T newScore, T currentBestScore)
    {
        return _isHigherScoreBetter
            ? _numOps.GreaterThan(newScore, currentBestScore)
            : _numOps.LessThan(newScore, currentBestScore);
    }
}