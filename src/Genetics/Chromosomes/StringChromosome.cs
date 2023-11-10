namespace AiDotNet.Genetics.Chromosomes;

public class StringChromosome : IChromosome<string>
{
    public double FitnessScore { get; }
    public string Chromosome { get; private set; }
    public int Size { get; }

    private double CrossoverBalancer { get; }
    private double MutationBalancer { get; }
    private string Target { get; }
    private string Genes { get; }
    private Random RandomGenerator { get; } = new();

    private StringChromosome(StringChromosome source)
    {
        Chromosome = (string)source.Chromosome.Clone();
        FitnessScore = source.FitnessScore;
        Target = source.Target;
        Genes = source.Genes;
        Size = source.Size;
        RandomGenerator = source.RandomGenerator;
        CrossoverBalancer = source.CrossoverBalancer;
        MutationBalancer = source.MutationBalancer;
    }

    public StringChromosome(string target, string genes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890, .-;:_!\"#%&/()=?@${[]}")
    {
        Chromosome = Generate();
        FitnessScore = CalculateFitnessScore();
        Target = target;
        Genes = genes;
    }

    public void Mutate()
    {
        var mutationGene = RandomGenerator.Next(Size);
        var tempArray = Chromosome.ToCharArray();

        if (RandomGenerator.NextDouble() < MutationBalancer)
        {
            tempArray[mutationGene] *= mutationMultiplierGenerator;
        }
        else
        {
            tempArray[mutationGene] += mutationAdditionGenerator.Generate();
        }

        Chromosome = tempArray.ToString() ?? Chromosome;
    }

    public string Generate()
    {
        var chromosome = string.Empty;
        for (var i = 0; i < Target.Length; i++)
        {
            chromosome += CreateRandomGene();
        }

        return chromosome;
    }

    private char CreateRandomGene()
    {
        var index = RandomGenerator.Next(0, Genes.Length);

        return Genes[index];
    }

    public IChromosome<string> Clone()
    {
        return new StringChromosome(this);
    }

    public IChromosome<string> CreateNew()
    {
        return new StringChromosome();
    }

    public double CalculateFitnessScore()
    {
        double fitnessScore = 0;
        for (var i = 0; i < Target.Length; i++)
        {
            fitnessScore += Target[i] != Chromosome[i] ? 1 : 0;
        }

        return fitnessScore;
    }

    public string Crossover(IChromosome<string> chromosome)
    {
        var newChromosome = string.Empty;
        for (var i = 0; i < Chromosome.Length; i++)
        {
            var probability = RandomGenerator.NextDouble();

            if (probability < CrossoverBalancer)
            {
                newChromosome += Chromosome[i];
            }
            else if (probability > CrossoverBalancer)
            {
                newChromosome += chromosome.Chromosome[i];
            }
            else
            {
                newChromosome += CreateRandomGene();
            }
        }

        return newChromosome;
    }

    public int CompareTo(IChromosome<string>? otherChromosome)
    {
        return FitnessScore.CompareTo(otherChromosome?.FitnessScore);
    }
}