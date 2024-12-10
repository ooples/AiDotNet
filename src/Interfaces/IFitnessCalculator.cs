namespace AiDotNet.Interfaces;

public interface IFitnessCalculator
{
    double CalculateFitnessScore(Vector<double> actualYValues, Vector<double> predictedYValues);
}