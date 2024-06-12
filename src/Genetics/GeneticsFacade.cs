namespace AiDotNet.Genetics;

internal class GeneticsFacade<T> : IGenetics<T>
{
    private Random RandomGenerator { get; } = new();
    public int PopulationSize { get; }
    public List<IChromosome<T>> Population { get; }
    public ISelectionMethod<T> SelectionMethod { get; }
    public double RandomSelectionPortion { get; }
    public bool AutoShuffle { get; }
    public double CrossoverRate { get; }
    public double MutationRate { get; }

    public GeneticsFacade(IChromosome<T> chromosome, GeneticAiOptions<T> geneticAiOptions)
    {
        PopulationSize = geneticAiOptions.PopulationSize;
        RandomSelectionPortion = geneticAiOptions.RandomSelectionPortion;
        SelectionMethod = geneticAiOptions.SelectionMethod;
        AutoShuffle = geneticAiOptions.AutoShuffle;
        CrossoverRate = geneticAiOptions.CrossoverRate;
        MutationRate = geneticAiOptions.MutationRate;
        Population = new List<IChromosome<T>>(PopulationSize);

        GeneratePopulation(chromosome);
        RunGeneration();
    }

    public void GeneratePopulation(IChromosome<T> chromosome)
    {
        // add more chromosomes to the population
        for (var i = 0; i < PopulationSize; i++)
        {
            // create new chromosome
            var c = chromosome.CreateNew();
            // calculate it's fitness
            c.CalculateFitnessScore();
            // add it to population
            Population.Add(c);
        }
    }

    public void RunGeneration()
    {
        // do crossover
        Crossover();

        // do mutation
        Mutation();

        // do selection
        Selection();

        // shuffle population
        if (AutoShuffle) Population.Shuffle();
    }

    public void Mutation()
    {
        for (var i = 0; i < PopulationSize; i++)
        {
            // generate next random number and check if we need to do mutation
            if (RandomGenerator.NextDouble() > MutationRate) continue;

            // clone the chromosome
            var c = Population[i].Clone();
            // mutate it
            c.Mutate();
            // calculate fitness of the mutant
            c.CalculateFitnessScore();
            // add mutant to the population
            Population.Add(c);
        }
    }

    public void Crossover()
    {
        for (var i = 1; i < PopulationSize; i += 2)
        {
            // generate next random number and check if we need to do crossover
            if (RandomGenerator.NextDouble() > CrossoverRate) continue;

            // clone both ancestors
            var c1 = Population[i - 1].Clone();
            var c2 = Population[i].Clone();

            // do crossover
            c1.Crossover(c2);

            // calculate fitness of these two offsprings
            c1.CalculateFitnessScore();
            c2.CalculateFitnessScore();

            // add two new offsprings to the population
            Population.Add(c1);
            Population.Add(c2);
        }
    }

    public void Selection()
    {
        // amount of random chromosomes in the new population
        var randomAmount = (int)(RandomSelectionPortion * PopulationSize);

        // do selection
        SelectionMethod.ApplySelection(Population, PopulationSize - randomAmount);

        // add random chromosomes
        if (randomAmount <= 0) return;
        var chromosome = Population[0];

        for (var i = 0; i < randomAmount; i++)
        {
            // create new chromosome
            var c = chromosome.CreateNew();
            // calculate it's fitness
            c.CalculateFitnessScore();
            // add it to population
            Population.Add(c);
        }
    }
}