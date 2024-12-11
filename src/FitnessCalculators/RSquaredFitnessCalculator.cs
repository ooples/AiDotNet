namespace AiDotNet.FitnessCalculators;

public class RSquaredFitnessCalculator : FitnessCalculatorBase
{
    public RSquaredFitnessCalculator() : base(isHigherScoreBetter: true)
    {
    }

    public override double CalculateFitnessScore(
        ErrorStats errorStats, 
        BasicStats basicStats, 
        Vector<double> actualValues, 
        Vector<double> predictedValues,
        Matrix<double> features)
    {
        return errorStats.R2;
    }
}