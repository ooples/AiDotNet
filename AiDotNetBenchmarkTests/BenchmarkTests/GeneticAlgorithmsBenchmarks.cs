using AiDotNet.Genetics;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for Genetic Algorithms
/// Tests all genetic algorithm variants and their performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class GeneticAlgorithmsBenchmarks
{
    [Params(50, 200)]
    public int PopulationSize { get; set; }

    [Params(20)]
    public int ChromosomeLength { get; set; }

    [Params(10, 50)]
    public int Generations { get; set; }

    private List<Vector<double>> _population = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _population = new List<Vector<double>>();

        for (int i = 0; i < PopulationSize; i++)
        {
            var chromosome = new Vector<double>(ChromosomeLength);
            for (int j = 0; j < ChromosomeLength; j++)
            {
                chromosome[j] = random.NextDouble();
            }
            _population.Add(chromosome);
        }
    }

    [Benchmark(Baseline = true)]
    public StandardGeneticAlgorithm<double> GA_StandardInitialize()
    {
        var ga = new StandardGeneticAlgorithm<double>(
            populationSize: PopulationSize,
            chromosomeLength: ChromosomeLength,
            mutationRate: 0.01,
            crossoverRate: 0.7
        );
        return ga;
    }

    [Benchmark]
    public StandardGeneticAlgorithm<double> GA_StandardEvolve()
    {
        var ga = new StandardGeneticAlgorithm<double>(
            populationSize: PopulationSize,
            chromosomeLength: ChromosomeLength,
            mutationRate: 0.01,
            crossoverRate: 0.7
        );

        ga.Initialize();
        for (int i = 0; i < Generations; i++)
        {
            ga.Evolve();
        }

        return ga;
    }

    [Benchmark]
    public AdaptiveGeneticAlgorithm<double> GA_AdaptiveInitialize()
    {
        var ga = new AdaptiveGeneticAlgorithm<double>(
            populationSize: PopulationSize,
            chromosomeLength: ChromosomeLength,
            initialMutationRate: 0.01,
            initialCrossoverRate: 0.7
        );
        return ga;
    }

    [Benchmark]
    public AdaptiveGeneticAlgorithm<double> GA_AdaptiveEvolve()
    {
        var ga = new AdaptiveGeneticAlgorithm<double>(
            populationSize: PopulationSize,
            chromosomeLength: ChromosomeLength,
            initialMutationRate: 0.01,
            initialCrossoverRate: 0.7
        );

        ga.Initialize();
        for (int i = 0; i < Generations; i++)
        {
            ga.Evolve();
        }

        return ga;
    }

    [Benchmark]
    public SteadyStateGeneticAlgorithm<double> GA_SteadyStateInitialize()
    {
        var ga = new SteadyStateGeneticAlgorithm<double>(
            populationSize: PopulationSize,
            chromosomeLength: ChromosomeLength,
            mutationRate: 0.01,
            crossoverRate: 0.7,
            replacementSize: 2
        );
        return ga;
    }

    [Benchmark]
    public IslandModelGeneticAlgorithm<double> GA_IslandModelInitialize()
    {
        var ga = new IslandModelGeneticAlgorithm<double>(
            numIslands: 4,
            populationPerIsland: PopulationSize / 4,
            chromosomeLength: ChromosomeLength,
            mutationRate: 0.01,
            crossoverRate: 0.7,
            migrationRate: 0.1
        );
        return ga;
    }

    [Benchmark]
    public NonDominatedSortingGeneticAlgorithm<double> GA_NSGAInitialize()
    {
        var ga = new NonDominatedSortingGeneticAlgorithm<double>(
            populationSize: PopulationSize,
            chromosomeLength: ChromosomeLength,
            mutationRate: 0.01,
            crossoverRate: 0.7,
            numObjectives: 2
        );
        return ga;
    }

    [Benchmark]
    public double GA_CalculateFitness()
    {
        // Simple fitness function: sum of chromosome values
        double totalFitness = 0;
        foreach (var chromosome in _population)
        {
            double fitness = 0;
            for (int i = 0; i < chromosome.Length; i++)
            {
                fitness += chromosome[i];
            }
            totalFitness += fitness;
        }
        return totalFitness / PopulationSize;
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) GA_Crossover()
    {
        var parent1 = _population[0];
        var parent2 = _population[1];

        var offspring1 = new Vector<double>(ChromosomeLength);
        var offspring2 = new Vector<double>(ChromosomeLength);

        int crossoverPoint = ChromosomeLength / 2;

        for (int i = 0; i < ChromosomeLength; i++)
        {
            if (i < crossoverPoint)
            {
                offspring1[i] = parent1[i];
                offspring2[i] = parent2[i];
            }
            else
            {
                offspring1[i] = parent2[i];
                offspring2[i] = parent1[i];
            }
        }

        return (offspring1, offspring2);
    }

    [Benchmark]
    public Vector<double> GA_Mutation()
    {
        var chromosome = _population[0].Clone();
        var random = new Random(42);
        double mutationRate = 0.01;

        for (int i = 0; i < chromosome.Length; i++)
        {
            if (random.NextDouble() < mutationRate)
            {
                chromosome[i] = random.NextDouble();
            }
        }

        return chromosome;
    }
}
