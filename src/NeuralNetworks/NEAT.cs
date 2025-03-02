namespace AiDotNet.NeuralNetworks;

public class NEAT<T> : NeuralNetworkBase<T>
{
    private List<Genome<T>> Population;
    private int PopulationSize { get; set; }
    private double MutationRate { get; set; }
    private double CrossoverRate { get; set; }
    private int InnovationNumber;

    public NEAT(NeuralNetworkArchitecture<T> architecture, int populationSize, double mutationRate = 0.1, double crossoverRate = 0.75)
        : base(architecture)
    {
        PopulationSize = populationSize;
        MutationRate = mutationRate;
        CrossoverRate = crossoverRate;
        InnovationNumber = 0;
        Population = InitializePopulation();
    }

    protected override void InitializeLayers()
    {
        // NEAT doesn't use fixed layers, so we'll leave this empty
    }

    private List<Genome<T>> InitializePopulation()
    {
        var population = new List<Genome<T>>();
        for (int i = 0; i < PopulationSize; i++)
        {
            population.Add(CreateInitialGenome());
        }

        return population;
    }

    private Genome<T> CreateInitialGenome()
    {
        var genome = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);
        for (int i = 0; i < Architecture.InputSize; i++)
        {
            for (int j = 0; j < Architecture.OutputSize; j++)
            {
                genome.AddConnection(i, Architecture.InputSize + j, RandomWeight(), true, InnovationNumber++);
            }
        }

        return genome;
    }

    public void EvolvePopulation(Func<Genome<T>, T> fitnessFunction, int generations)
    {
        for (int gen = 0; gen < generations; gen++)
        {
            // Evaluate fitness
            foreach (var genome in Population)
            {
                genome.Fitness = fitnessFunction(genome);
            }

            // Sort population by fitness
            Population.Sort((a, b) => NumOps.GreaterThan(b.Fitness, a.Fitness) ? 1 : -1);

            // Create new population
            var newPopulation = new List<Genome<T>>();

            // Elitism: Keep the best individual
            newPopulation.Add(Population[0]);

            while (newPopulation.Count < PopulationSize)
            {
                if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(CrossoverRate)))
                {
                    var parent1 = SelectParent();
                    var parent2 = SelectParent();
                    var child = Crossover(parent1, parent2);
                    Mutate(child);
                    newPopulation.Add(child);
                }
                else
                {
                    var parent = SelectParent();
                    var child = parent.Clone();
                    Mutate(child);
                    newPopulation.Add(child);
                }
            }

            Population = newPopulation;
        }
    }

    private Genome<T> SelectParent()
    {
        // Tournament selection
        int tournamentSize = 3;
        var tournament = new List<Genome<T>>();
        for (int i = 0; i < tournamentSize; i++)
        {
            tournament.Add(Population[Random.Next(Population.Count)]);
        }

        return tournament.OrderByDescending(g => g.Fitness).First();
    }

    private Genome<T> Crossover(Genome<T> parent1, Genome<T> parent2)
    {
        var child = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);
        foreach (var conn in parent1.Connections.Concat(parent2.Connections))
        {
            if (!child.Connections.Any(c => c.Innovation == conn.Innovation))
            {
                child.AddConnection(conn.FromNode, conn.ToNode, conn.Weight, conn.IsEnabled, conn.Innovation);
            }
        }

        return child;
    }

    private void Mutate(Genome<T> genome)
    {
        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(MutationRate)))
        {
            // Add new node
            var connection = genome.Connections[Random.Next(genome.Connections.Count)];
            int newNodeId = genome.Connections.Max(c => Math.Max(c.FromNode, c.ToNode)) + 1;
            genome.DisableConnection(connection.Innovation);
            genome.AddConnection(connection.FromNode, newNodeId, NumOps.One, true, InnovationNumber++);
            genome.AddConnection(newNodeId, connection.ToNode, connection.Weight, true, InnovationNumber++);
        }

        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(MutationRate)))
        {
            // Add new connection
            int fromNode = Random.Next(Architecture.InputSize + Architecture.OutputSize);
            int toNode = Random.Next(Architecture.InputSize, Architecture.InputSize + Architecture.OutputSize);
            if (!genome.Connections.Any(c => c.FromNode == fromNode && c.ToNode == toNode))
            {
                genome.AddConnection(fromNode, toNode, RandomWeight(), true, InnovationNumber++);
            }
        }

        // Mutate weights
        foreach (var conn in genome.Connections)
        {
            if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(MutationRate)))
            {
                conn.Weight = NumOps.Add(conn.Weight, RandomWeight());
            }
        }
    }

    private T RandomWeight()
    {
        return NumOps.FromDouble(Random.NextDouble() * 2 - 1);
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        // Use the best genome for prediction
        var bestGenome = Population.OrderByDescending(g => g.Fitness).First();
        return bestGenome.Activate(input);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // NEAT doesn't use this method for parameter updates
        throw new NotImplementedException("NEAT doesn't support direct parameter updates.");
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(PopulationSize);
        writer.Write(MutationRate);
        writer.Write(CrossoverRate);
        writer.Write(InnovationNumber);

        writer.Write(Population.Count);
        foreach (var genome in Population)
        {
            genome.Serialize(writer);
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        PopulationSize = reader.ReadInt32();
        MutationRate = reader.ReadDouble();
        CrossoverRate = reader.ReadDouble();
        InnovationNumber = reader.ReadInt32();

        int populationCount = reader.ReadInt32();
        Population = new List<Genome<T>>();
        for (int i = 0; i < populationCount; i++)
        {
            var genome = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);
            genome.Deserialize(reader);
            Population.Add(genome);
        }
    }
}