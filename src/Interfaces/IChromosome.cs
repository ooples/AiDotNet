namespace AiDotNet.Interfaces;

public interface IChromosome<T>
{
    double FitnessScore { get; }

    T Chromosome { get; }

    public void Mutate();

    public T Crossover(IChromosome<T> chromosome);

    public T Generate();

    public IChromosome<T> Clone();

    public IChromosome<T> CreateNew();

    public double CalculateFitnessScore();
}