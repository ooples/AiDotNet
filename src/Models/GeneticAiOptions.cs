namespace AiDotNet.Models;

public class GeneticAiOptions<T>
{
    public int PopulationSize { get; set; } = 500;
    public double RandomSelectionPortion { get; set; } = 0.1;
    public bool AutoShuffle { get; set; } = false;
    public double CrossoverRate { get; set; } = 0.75;
    public double MutationRate { get; set; } = 0.1;
    public ISelectionMethod<T> SelectionMethod { get; set; } = new EliteSelection<T>();
}