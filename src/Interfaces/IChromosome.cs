namespace AiDotNet.Interfaces;

public interface IChromosome<T, CType> : IComparable<T>
{
    double FitnessScore { get; }

    CType Chromosome { get; }

    public void Mutate();

    public CType Crossover(IChromosome<T, CType> chromosome);

    public CType Generate();

    public IChromosome<T, CType> Clone();

    public IChromosome<T, CType> CreateNew();

    public double CalculateFitness(IFitnessFunction fitnessFunction);
}