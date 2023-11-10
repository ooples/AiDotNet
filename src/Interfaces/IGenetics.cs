namespace AiDotNet.Interfaces;

public interface IGenetics<T>
{
    public int PopulationSize { get; }
    public List<IChromosome<T>> Population { get; }
    public ISelectionMethod<T> SelectionMethod { get; }
    public double RandomSelectionPortion { get; }
    public bool AutoShuffle { get; }
    public double CrossoverRate { get; }
    public double CrossoverBalancer { get; }
    public double MutationRate { get; }
    public double MutationBalancer { get; }

    public void GeneratePopulation(IChromosome<T> chromosome);

    public void RunGeneration();

    public void Mutation();

    public void Crossover();

    public void Selection();
}