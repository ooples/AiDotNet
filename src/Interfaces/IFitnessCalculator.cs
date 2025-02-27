namespace AiDotNet.Interfaces;

public interface IFitnessCalculator<T>
{
    T CalculateFitnessScore(ModelEvaluationData<T> evaluationData);

    bool IsHigherScoreBetter { get; }

    bool IsBetterFitness(T currentFitness, T bestFitness);
}