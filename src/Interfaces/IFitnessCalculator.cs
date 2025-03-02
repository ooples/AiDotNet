namespace AiDotNet.Interfaces;

public interface IFitnessCalculator<T>
{
    T CalculateFitnessScore(ModelEvaluationData<T> evaluationData);

    T CalculateFitnessScore(DataSetStats<T> dataSet);

    bool IsHigherScoreBetter { get; }

    bool IsBetterFitness(T currentFitness, T bestFitness);
}