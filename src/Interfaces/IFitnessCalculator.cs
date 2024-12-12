namespace AiDotNet.Interfaces;

public interface IFitnessCalculator
{
    double CalculateFitnessScore(
        ErrorStats errorStats, 
        BasicStats basicStats, 
        Vector<double> actualValues, 
        Vector<double> predictedValues,
        Matrix<double> features);

    bool IsHigherScoreBetter { get; }

    bool IsBetterFitness(double currentFitness, double bestFitness);
}