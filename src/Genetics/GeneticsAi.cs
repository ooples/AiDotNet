using AiDotNet.Genetics.SelectionMethods;
namespace AiDotNet.Genetics;

public class GeneticsAi<T>
{
    public List<IChromosome<T>> Population { get; private set; }
    private ISelectionMethod<T> SelectionMethod { get; }
    private int Size { get; }
    private double RandomSelectionPortion { get; }
    private bool AutoShuffle { get; }
    private double CrossoverRate { get; }
    private double MutationRate { get; }
    private Random RandomGenerator { get; } = new ();

    public GeneticsAi(IChromosome<T> chromosome, ISelectionMethod<T>? selectionMethod, int size, 
        double randomSelectionPortion = 0.1, bool autoShuffle = false, double crossoverRate = 0.75, double mutationRate = 0.1)
    {
        Size = size;
        RandomSelectionPortion = randomSelectionPortion;
        SelectionMethod = selectionMethod ?? new EliteSelection<T>();
        AutoShuffle = autoShuffle;
        CrossoverRate = crossoverRate;
        MutationRate = mutationRate;
        Population = new List<IChromosome<T>>(size);

        GeneratePopulation(chromosome);
        RunGeneration();

        var test = new GeneticsFacade();
    }

    private void GeneratePopulation(IChromosome<T> ancestor)
    {
        // add ancestor to the population
        ancestor.CalculateFitnessScore();
        Population.Add(ancestor.Clone());

        // add more chromosomes to the population
        for (var i = 1; i < Size; i++)
        {
            // create new chromosome
            var c = ancestor.CreateNew();
            // calculate it's fitness
            c.CalculateFitnessScore();
            // add it to population
            Population.Add(c);
        }
    }

    private void RunGeneration()
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

    private void Mutation()
    {
        for (var i = 0; i < Size; i++)
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

    private void Crossover()
    {
        for (var i = 1; i < Size; i += 2)
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

    private void Selection()
    {
        // amount of random chromosomes in the new population
        var randomAmount = (int)(RandomSelectionPortion * Size);

        // do selection
        SelectionMethod?.ApplySelection(Population, Size - randomAmount);

        // add random chromosomes
        if (randomAmount <= 0) return;
        var ancestor = Population[0];

        for (var i = 0; i < randomAmount; i++)
        {
            // create new chromosome
            var c = ancestor.CreateNew();
            // calculate it's fitness
            c.CalculateFitnessScore();
            // add it to population
            Population.Add(c);
        }
    }
}