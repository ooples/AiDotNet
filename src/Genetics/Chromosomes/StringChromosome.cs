namespace AiDotNet.Genetics.Chromosomes;

public class StringChromosome : IChromosome<string>
{
    public double FitnessScore { get; }
    public string Chromosome { get; private set; }
    public int Size { get; }

    private string Genes { get; }
    private ChromosomeOptions<string> ChromosomeOptions { get; }

    private StringChromosome(StringChromosome source)
    {
        Chromosome = (string)source.Chromosome.Clone();
        FitnessScore = source.FitnessScore;
        Genes = source.Genes;
        Size = source.Size;
        ChromosomeOptions = source.ChromosomeOptions;
    }

    public StringChromosome(ChromosomeOptions<string> chromosomeOptions, 
        string genes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890, .-;:_!\"#%&/()=?@${[]}")
    {
        ChromosomeOptions = chromosomeOptions;
        Chromosome = Generate();
        FitnessScore = CalculateFitnessScore();
        Genes = genes;
    }

    public void Mutate()
    {
        var mutationGene = ChromosomeOptions.RandomGenerator.Next(Size);
        var tempArray = Chromosome.ToCharArray();

        if (ChromosomeOptions.RandomGenerator.NextDouble() < ChromosomeOptions.MutationBalancer)
        {

            tempArray[mutationGene] = Genes[ChromosomeOptions.MutationMultiplierGenerator.Next(0, 1)];
        }
        else
        {
            tempArray[mutationGene] = Genes[ChromosomeOptions.MutationAdditionGenerator.Next(-1, 1)];
        }

        Chromosome = tempArray.ToString() ?? Chromosome;
    }

    public string Generate()
    {
        var chromosome = string.Empty;
        for (var i = 0; i < ChromosomeOptions.Target.Length; i++)
        {
            chromosome += CreateRandomGene();
        }

        return chromosome;
    }

    private char CreateRandomGene()
    {
        var index = ChromosomeOptions.RandomGenerator.Next(0, Genes.Length);

        return Genes[index];
    }

    public IChromosome<string> Clone()
    {
        return new StringChromosome(this);
    }

    public IChromosome<string> CreateNew()
    {
        return new StringChromosome(ChromosomeOptions);
    }

    public double CalculateFitnessScore()
    {
        double fitnessScore = 0;
        for (var i = 0; i < ChromosomeOptions.Target.Length; i++)
        {
            fitnessScore += ChromosomeOptions.Target[i] != Chromosome[i] ? 1 : 0;
        }

        return fitnessScore;
    }

    public string Crossover(IChromosome<string> chromosome)
    {
        var newChromosome = string.Empty;
        for (var i = 0; i < Chromosome.Length; i++)
        {
            var probability = ChromosomeOptions.RandomGenerator.NextDouble();

            if (probability < ChromosomeOptions.CrossoverBalancer)
            {
                newChromosome += Chromosome[i];
            }
            else if (probability > ChromosomeOptions.CrossoverBalancer)
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