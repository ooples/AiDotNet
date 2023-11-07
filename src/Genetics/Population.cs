namespace AiDotNet.Genetics;

internal class Population
{
    private readonly int _populationSize;
    private readonly int _geneSize;
    private readonly double _mutationRate;
    private readonly double _crossoverRate;
    private readonly int _tournamentSize;
    private readonly bool _elitism;
    private readonly Random _random;

    internal Population(int populationSize, int geneSize, double mutationRate, double crossoverRate, int tournamentSize,
        bool elitism)
    {
        _populationSize = populationSize;
        _geneSize = geneSize;
        _mutationRate = mutationRate;
        _crossoverRate = crossoverRate;
        _tournamentSize = tournamentSize;
        _elitism = elitism;
        _random = new Random();
    }

    internal Individual[] CreatePopulation()
    {
        var population = new Individual[_populationSize];
        for (var i = 0; i < _populationSize; i++)
        {
            population[i] = new Individual(_geneSize);
        }

        return population;
    }

    internal Individual[] EvolvePopulation(Individual[] population)
    {
        var newPopulation = new Individual[_populationSize];
        var elitismOffset = 0;

        if (_elitism)
        {
            newPopulation[0] = GetFittest(population);
            elitismOffset = 1;
        }

        for (var i = elitismOffset; i < _populationSize; i++)
        {
            var parent1 = TournamentSelection(population);
            var parent2 = TournamentSelection(population);
            var child = Crossover(parent1, parent2);
            newPopulation[i] = child;
        }

        for (var i = elitismOffset; i < _populationSize; i++)
        {
            Mutate(newPopulation[i]);
        }

        return newPopulation;
    }

    private Individual GetFittest(Individual[] population)
    {
        var fittest = population[0];
        for (var i = 1; i < population.Length; i++)
        {
            if (population[i].Fitness > fittest.Fitness)
            {
                fittest = population[i];
            }
        }

        return fittest;
    }

    private Individual TournamentSelection(Individual[] population)
    {
        var tournament = new Individual[_tournamentSize];
        for (var i = 0; i < _tournamentSize; i++)
        {
            var randomIndex = _random.Next(0, _populationSize);
            tournament[i]
        }
    }
}