namespace AiDotNet.Genetics.Chromosomes;

public class StringChromosome : IChromosome<string>
{
    public double FitnessScore { get; }
    public string Chromosome { get; private set; }

    private string Target = "I Love Genetics so much!";
    private const string Genes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890, .-;:_!\"#%&/()=?@${[]}";

    private StringChromosome(StringChromosome source)
    {
        Chromosome = (string)source.Chromosome.Clone();
        FitnessScore = source.FitnessScore;
        Target = source.Target;
    }

    public StringChromosome(string chromosome)
    {
        Chromosome = chromosome;
        FitnessScore = CalculateFitnessScore();
    }

    public StringChromosome()
    {
        Chromosome = Generate();
        FitnessScore = CalculateFitnessScore();
    }

    public void Mutate()
    {
        
    }

    public string Generate()
    {
        var chromosome = string.Empty;
        for (var i = 0; i < Target.Length; i++)
        {
            chromosome += StringChromosome.CreateRandomGene();
        }

        return chromosome;
    }

    private static char CreateRandomGene()
    {
        var random = new Random();
        var index = random.Next(0, Genes.Length);

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
            var probability = new Random().NextDouble();

            newChromosome += probability switch
            {
                // if prob is less than 0.45, insert gene
                // from parent 1 
                < 0.45 => Chromosome[i],
                // if prob is between 0.45 and 0.90, insert
                // gene from parent 2
                < 0.9 => chromosome.Chromosome[i],
                _ => CreateRandomGene()
            };
        }

        return newChromosome;
    }
}