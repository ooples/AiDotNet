namespace AiDotNet.Models;

public class ChromosomeOptions<T>
{
    public double CrossoverBalancer { get; set; } = 0.5;
    public double MutationBalancer { get; set; } = 0.5;
    public T Target { get; set; } = default!;
    public Random MutationMultiplierGenerator { get; set; } = new();
    public Random MutationAdditionGenerator { get; set; } = new();
    public Random RandomGenerator { get; set; } = new();
}