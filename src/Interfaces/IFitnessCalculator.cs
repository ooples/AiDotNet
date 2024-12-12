namespace AiDotNet.Interfaces;

public interface IFitnessCalculator<T>
{
    T CalculateFitnessScore(
        ErrorStats<T> errorStats,
        BasicStats<T> actualBasicStats,
        BasicStats<T> predictedBasicStats,
        Vector<T> actualValues,
        Vector<T> predictedValues,
        Matrix<T> features,
        PredictionStats<T> predictionStats);

    bool IsHigherScoreBetter { get; }

    bool IsBetterFitness(T currentFitness, T bestFitness);
}