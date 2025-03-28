namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a NeuroEvolution of Augmenting Topologies (NEAT) algorithm implementation, which evolves
/// neural networks through genetic algorithms.
/// </summary>
/// <remarks>
/// <para>
/// NEAT is an evolutionary algorithm that creates and evolves neural network topologies along with connection weights.
/// Unlike traditional neural networks with fixed structures, NEAT starts with simple networks and gradually adds
/// complexity through evolution. It uses genetic operators like mutation and crossover, along with speciation
/// to protect innovation, to evolve networks that solve specific problems without requiring manual design
/// of the network architecture.
/// </para>
/// <para><b>For Beginners:</b> NEAT is a way to grow neural networks through evolution rather than training them with fixed structures.
/// 
/// Think of NEAT like breeding plants to get better features:
/// - Instead of designing a neural network by hand, you start with simple networks
/// - These networks "reproduce" and "mutate" over generations
/// - Networks that perform better on your task are more likely to pass on their "genes"
/// - Over time, the networks evolve complex structures that solve your problem well
/// 
/// The key differences from traditional neural networks:
/// - The structure (connections between neurons) evolves along with the weights
/// - Networks can grow more complex over time by adding new neurons and connections
/// - You work with a population of many networks, not just one
/// - Instead of training with gradient descent, you use evolution to improve performance
/// 
/// NEAT is particularly good for:
/// - Problems where you don't know the ideal network structure
/// - Reinforcement learning tasks (like game playing)
/// - Finding novel solutions that a human designer might not think of
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NEAT<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the current population of genomes (neural network structures).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list contains all the current individuals (genomes) in the population. Each genome represents
    /// a different neural network structure with its own set of nodes and connections. The population evolves
    /// over time through the evolutionary process.
    /// </para>
    /// <para><b>For Beginners:</b> This is the collection of all neural networks currently in your evolving population.
    /// 
    /// Think of Population as:
    /// - A group of different neural networks (called "genomes")
    /// - Each genome represents a different way to solve your problem
    /// - Some genomes will perform better than others
    /// - The best genomes get to "reproduce" and pass on their characteristics
    /// 
    /// During evolution, this population changes as:
    /// - Higher-performing networks reproduce more often
    /// - New networks are created through crossover (combining two parent networks)
    /// - Mutations introduce new variations
    /// 
    /// This diversity in the population helps NEAT explore different possible solutions.
    /// </para>
    /// </remarks>
    private List<Genome<T>> _population;

    /// <summary>
    /// Gets or sets the size of the population (number of genomes).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The population size determines how many different network structures (genomes) are evolved in parallel.
    /// A larger population provides more genetic diversity and exploration of the solution space, but requires
    /// more computational resources. Typical values range from 50 to several hundred.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many different neural networks are in your evolving population.
    /// 
    /// Population size is important because:
    /// - A larger population (like 100-500) provides more diversity
    /// - More diversity helps explore more possible solutions
    /// - But larger populations require more computing power
    /// - Smaller populations evolve faster but might get stuck in suboptimal solutions
    /// 
    /// This is like having a larger or smaller gene pool in a biological population.
    /// The right size depends on your problem complexity and available computing resources.
    /// </para>
    /// </remarks>
    private int _populationSize { get; set; }

    /// <summary>
    /// Gets or sets the probability of mutation occurring during reproduction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mutation rate controls how frequently random changes occur in the network structure or weights
    /// during evolution. Higher mutation rates increase exploration of new structures but can disrupt good solutions.
    /// Lower rates provide more stability but may limit innovation. Typical values range from 0.05 to 0.3.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how often random changes (mutations) occur in the networks.
    /// 
    /// Think of mutation rate as controlling how much experimentation happens:
    /// - A value of 0.1 means there's a 10% chance of each type of mutation occurring
    /// - Higher values (like 0.3) cause more random changes and exploration
    /// - Lower values (like 0.05) cause fewer changes, keeping solutions more stable
    /// 
    /// Mutations include:
    /// - Adding new neurons
    /// - Adding new connections
    /// - Changing connection weights
    /// 
    /// Finding the right mutation rate is important:
    /// - Too high: networks change too randomly and can't preserve good solutions
    /// - Too low: networks don't explore enough new possibilities
    /// </para>
    /// </remarks>
    private double _mutationRate { get; set; }

    /// <summary>
    /// Gets or sets the probability of crossover occurring during reproduction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The crossover rate determines how often two parent genomes combine to create offspring versus
    /// simply cloning and mutating a single parent. Higher rates increase the mixing of genetic material
    /// but may disrupt successful network structures. Typical values range from 0.5 to 0.8.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how often two parent networks combine to create a child network.
    /// 
    /// Crossover is like breeding in biology:
    /// - A value of 0.75 means 75% of new networks come from combining two parent networks
    /// - The remaining 25% come from copying and mutating a single parent
    /// 
    /// During crossover:
    /// - Parts from two successful networks are combined
    /// - This helps mix good features from different networks
    /// - The child gets some connections from each parent
    /// 
    /// This balance is important because:
    /// - Too much crossover can break up good network structures
    /// - Too little crossover limits how well good features can be combined
    /// </para>
    /// </remarks>
    private double _crossoverRate { get; set; }

    /// <summary>
    /// Gets or sets the global innovation number counter used to track historical origins of genes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The innovation number is a key component of the NEAT algorithm that helps track the historical origin
    /// of each connection gene. Each new structural innovation (a new connection or node) receives a unique,
    /// incrementing ID. This historical marking allows NEAT to perform meaningful crossover between different
    /// network topologies by matching genes with the same origin.
    /// </para>
    /// <para><b>For Beginners:</b> This is a counter that gives each new connection or neuron a unique ID number.
    /// 
    /// Innovation numbers are important because:
    /// - They help NEAT know which parts of different networks correspond to each other
    /// - When two networks reproduce, we need to know which connections match up
    /// - Each time a new connection or neuron is created, it gets a new innovation number
    /// 
    /// Think of it like a family tree that helps track where each feature came from.
    /// This historical marking is one of the key innovations in NEAT that allows
    /// networks with different structures to be combined effectively.
    /// </para>
    /// </remarks>
    private int _innovationNumber;

    /// <summary>
    /// Initializes a new instance of the <see cref="NEAT{T}"/> class with the specified architecture and evolution parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input and output sizes.</param>
    /// <param name="populationSize">The number of individual genomes in the population.</param>
    /// <param name="mutationRate">The probability of mutation occurring during reproduction. Default is 0.1.</param>
    /// <param name="crossoverRate">The probability of crossover occurring during reproduction. Default is 0.75.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a NEAT algorithm with the specified parameters. Unlike traditional neural networks,
    /// NEAT doesn't use a fixed architecture with predefined layers. Instead, it uses the architecture primarily
    /// to determine input and output sizes. The constructor initializes a population of minimal network structures
    /// (genomes) that will evolve over time through the evolutionary process.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new NEAT system with your chosen settings.
    /// 
    /// When creating a NEAT system, you specify:
    /// 
    /// 1. Architecture: Mainly used to define how many inputs and outputs your networks need
    ///    - For example, if solving a game, inputs might be game state variables
    ///    - Outputs might be action choices the AI can make
    /// 
    /// 2. Population Size: How many different neural networks to evolve at once
    ///    - Larger populations explore more possibilities but need more computing power
    /// 
    /// 3. Mutation Rate: How often random changes occur (default 0.1 or 10%)
    ///    - Controls the amount of exploration vs. stability
    /// 
    /// 4. Crossover Rate: How often networks combine vs. just being copied (default 0.75 or 75%)
    ///    - Controls how much genetic material is mixed between solutions
    /// 
    /// After creation, NEAT initializes a starting population of very simple networks
    /// that will grow more complex through evolution.
    /// </para>
    /// </remarks>
    public NEAT(NeuralNetworkArchitecture<T> architecture, int populationSize, double mutationRate = 0.1, double crossoverRate = 0.75)
        : base(architecture)
    {
        _populationSize = populationSize;
        _mutationRate = mutationRate;
        _crossoverRate = crossoverRate;
        _innovationNumber = 0;
        _population = InitializePopulation();
    }

    /// <summary>
    /// Initializes the layers of the neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is intentionally left empty because NEAT does not use fixed layers like traditional neural networks.
    /// Instead, NEAT evolves the network structure dynamically through the evolutionary process, adding nodes
    /// and connections as needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method is intentionally empty because NEAT works differently from traditional neural networks.
    /// 
    /// In traditional neural networks:
    /// - You define specific layers (input, hidden, output)
    /// - Each layer has a fixed number of neurons
    /// - The connections between layers are predetermined
    /// 
    /// In NEAT:
    /// - Networks don't have fixed layers
    /// - The structure evolves dynamically
    /// - Neurons and connections are added gradually through evolution
    /// 
    /// This fundamental difference is why this method doesn't need to do anything in NEAT.
    /// The network structure is defined by the genome, not by predefined layers.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // NEAT doesn't use fixed layers, so we'll leave this empty
    }

    /// <summary>
    /// Creates the initial population of genomes with minimal network structures.
    /// </summary>
    /// <returns>A list of initialized genomes.</returns>
    /// <remarks>
    /// <para>
    /// This method creates the initial population of genomes. Each genome starts with a minimal structure
    /// consisting of only input and output nodes with direct connections between them. This follows the NEAT
    /// principle of starting with minimal structures and gradually adding complexity through evolution.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the starting population of simple neural networks.
    /// 
    /// When NEAT begins:
    /// - It creates a population of very simple neural networks
    /// - Each network starts with just input and output neurons
    /// - Each input neuron is connected directly to each output neuron
    /// - The connection weights are randomized
    /// 
    /// This minimal starting point is important because:
    /// - It follows the principle of starting simple and growing complexity as needed
    /// - It doesn't make assumptions about what structure might work best
    /// - It allows the evolutionary process to discover the right complexity
    /// 
    /// As evolution progresses, these simple networks will grow more complex
    /// by adding neurons and connections where they're beneficial.
    /// </para>
    /// </remarks>
    private List<Genome<T>> InitializePopulation()
    {
        var population = new List<Genome<T>>();
        for (int i = 0; i < _populationSize; i++)
        {
            population.Add(CreateInitialGenome());
        }

        return population;
    }

    /// <summary>
    /// Creates a single initial genome with connections from each input to each output.
    /// </summary>
    /// <returns>A newly created genome with minimal structure.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a single genome with a minimal structure. It initializes a network with the specified
    /// number of input and output nodes, and creates direct connections from each input node to each output node
    /// with random weights. Each connection is assigned a unique innovation number to track its historical origin.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates one simple starting neural network.
    /// 
    /// Each initial network:
    /// - Has the input neurons you specified (for your problem's inputs)
    /// - Has the output neurons you specified (for your problem's outputs)
    /// - Connects each input directly to each output
    /// - Assigns random weights to these connections
    /// 
    /// For example, if you have 3 inputs and 2 outputs:
    /// - You'll have 6 connections (3 inputs × 2 outputs)
    /// - Each connection gets a random weight between -1 and 1
    /// - Each connection gets a unique innovation number for tracking
    /// 
    /// This simple structure provides a starting point that evolution can build upon,
    /// adding complexity only where it's beneficial for solving your problem.
    /// </para>
    /// </remarks>
    private Genome<T> CreateInitialGenome()
    {
        var genome = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);
        for (int i = 0; i < Architecture.InputSize; i++)
        {
            for (int j = 0; j < Architecture.OutputSize; j++)
            {
                genome.AddConnection(i, Architecture.InputSize + j, RandomWeight(), true, _innovationNumber++);
            }
        }

        return genome;
    }

    /// <summary>
    /// Evolves the population over a specified number of generations using the provided fitness function.
    /// </summary>
    /// <param name="fitnessFunction">A function that evaluates the fitness of each genome.</param>
    /// <param name="generations">The number of generations to evolve.</param>
    /// <remarks>
    /// <para>
    /// This method drives the evolutionary process over multiple generations. For each generation, it evaluates
    /// the fitness of each genome using the provided fitness function, sorts the population by fitness,
    /// and creates a new population through selection, crossover, and mutation. Through this process,
    /// the networks evolve to better solve the specified problem.
    /// </para>
    /// <para><b>For Beginners:</b> This method runs the evolutionary process for a specified number of generations.
    /// 
    /// The evolution process works like this:
    /// 
    /// 1. Evaluate each network:
    ///    - The provided fitness function tests how well each network performs
    ///    - Higher fitness scores mean better performance
    /// 
    /// 2. Sort networks by fitness:
    ///    - The best-performing networks are prioritized for reproduction
    /// 
    /// 3. Create a new population:
    ///    - Keep the very best network unchanged (called "elitism")
    ///    - Create new networks through either:
    ///      a) Crossover: Combining two parent networks
    ///      b) Mutation: Copying and modifying a single parent
    /// 
    /// 4. Repeat for the specified number of generations
    /// 
    /// Each generation should produce slightly better networks as successful
    /// traits are selected for and new beneficial mutations occur.
    /// 
    /// The fitness function you provide is crucial - it's what defines "good performance"
    /// and guides the entire evolutionary process.
    /// </para>
    /// </remarks>
    public void EvolvePopulation(Func<Genome<T>, T> fitnessFunction, int generations)
    {
        for (int gen = 0; gen < generations; gen++)
        {
            // Evaluate fitness
            foreach (var genome in _population)
            {
                genome.Fitness = fitnessFunction(genome);
            }

            // Sort population by fitness
            _population.Sort((a, b) => NumOps.GreaterThan(b.Fitness, a.Fitness) ? 1 : -1);

            // Create new population
            var newPopulation = new List<Genome<T>>();

            // Elitism: Keep the best individual
            newPopulation.Add(_population[0]);

            while (newPopulation.Count < _populationSize)
            {
                if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(_crossoverRate)))
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

            _population = newPopulation;
        }
    }

    /// <summary>
    /// Selects a parent genome for reproduction using tournament selection.
    /// </summary>
    /// <returns>The selected parent genome.</returns>
    /// <remarks>
    /// <para>
    /// This method implements tournament selection, a common selection method in genetic algorithms.
    /// It randomly selects a small number of genomes from the population, then returns the one with the
    /// highest fitness. This process favors fitter individuals while still giving less fit individuals
    /// a chance to reproduce, maintaining genetic diversity.
    /// </para>
    /// <para><b>For Beginners:</b> This method selects a parent network for reproduction, favoring better-performing networks.
    /// 
    /// Tournament selection works like this:
    /// 1. Randomly pick a small group of networks (3 in this case)
    /// 2. Compare their fitness scores
    /// 3. Select the best one from this small group
    /// 
    /// This approach has advantages:
    /// - Better networks are more likely to be selected
    /// - But even lower-performing networks have some chance
    /// - This maintains diversity while still driving improvement
    /// 
    /// Think of it like a small competition where the winner gets to reproduce.
    /// By not always selecting the absolute best network, we avoid getting
    /// stuck in a single solution path too early.
    /// </para>
    /// </remarks>
    private Genome<T> SelectParent()
    {
        // Tournament selection
        int tournamentSize = 3;
        var tournament = new List<Genome<T>>();
        for (int i = 0; i < tournamentSize; i++)
        {
            tournament.Add(_population[Random.Next(_population.Count)]);
        }

        return tournament.OrderByDescending(g => g.Fitness).First();
    }

    /// <summary>
    /// Creates a new genome by combining genetic material from two parent genomes.
    /// </summary>
    /// <param name="parent1">The first parent genome.</param>
    /// <param name="parent2">The second parent genome.</param>
    /// <returns>A new child genome created through crossover.</returns>
    /// <remarks>
    /// <para>
    /// This method implements crossover between two parent genomes to create a child genome. It combines
    /// connection genes from both parents, with matching genes (those with the same innovation number) being
    /// inherited from either parent. This allows NEAT to meaningfully combine networks with different topologies
    /// by utilizing the historical markings provided by innovation numbers.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a new network by combining parts from two parent networks.
    /// 
    /// The crossover process:
    /// 1. Creates a new empty network (the child)
    /// 2. Looks at all connections from both parents
    /// 3. Adds each unique connection to the child
    /// 
    /// The key to making this work is the innovation number:
    /// - Each connection has a unique ID number (innovation number)
    /// - This lets NEAT identify which connections in different networks match
    /// - When the same connection exists in both parents, only one copy is added to the child
    /// 
    /// This process allows NEAT to meaningfully combine networks with different structures,
    /// which is one of its key advantages over other evolutionary methods.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Applies random mutations to a genome based on the mutation rate.
    /// </summary>
    /// <param name="genome">The genome to mutate.</param>
    /// <remarks>
    /// <para>
    /// This method applies various types of mutations to a genome with probability based on the mutation rate.
    /// Possible mutations include adding a new node (by splitting an existing connection), adding a new connection
    /// between existing nodes, and modifying connection weights. These mutations allow NEAT to explore different
    /// network structures and parameters to find better solutions.
    /// </para>
    /// <para><b>For Beginners:</b> This method introduces random changes to a network to explore new possibilities.
    /// 
    /// NEAT uses three main types of mutations:
    /// 
    /// 1. Add Node Mutation:
    ///    - Takes an existing connection and splits it by adding a new neuron in the middle
    ///    - The original connection is disabled
    ///    - Two new connections are created (from source to new node, and from new node to target)
    ///    - This allows the network to create more complex behaviors
    /// 
    /// 2. Add Connection Mutation:
    ///    - Creates a new connection between two previously unconnected neurons
    ///    - This allows the network to create new paths for information flow
    /// 
    /// 3. Weight Mutation:
    ///    - Changes the weights of existing connections
    ///    - This fine-tunes the network behavior without changing its structure
    /// 
    /// Each type of mutation happens randomly based on the mutation rate.
    /// These mutations are how NEAT explores different network structures
    /// and gradually adds complexity where it's beneficial.
    /// </para>
    /// </remarks>
    private void Mutate(Genome<T> genome)
    {
        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(_mutationRate)))
        {
            // Add new node
            var connection = genome.Connections[Random.Next(genome.Connections.Count)];
            int newNodeId = genome.Connections.Max(c => Math.Max(c.FromNode, c.ToNode)) + 1;
            genome.DisableConnection(connection.Innovation);
            genome.AddConnection(connection.FromNode, newNodeId, NumOps.One, true, _innovationNumber++);
            genome.AddConnection(newNodeId, connection.ToNode, connection.Weight, true, _innovationNumber++);
        }

        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(_mutationRate)))
        {
            // Add new connection
            int fromNode = Random.Next(Architecture.InputSize + Architecture.OutputSize);
            int toNode = Random.Next(Architecture.InputSize, Architecture.InputSize + Architecture.OutputSize);
            if (!genome.Connections.Any(c => c.FromNode == fromNode && c.ToNode == toNode))
            {
                genome.AddConnection(fromNode, toNode, RandomWeight(), true, _innovationNumber++);
            }
        }

        // Mutate weights
        foreach (var conn in genome.Connections)
        {
            if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), NumOps.FromDouble(_mutationRate)))
            {
                conn.Weight = NumOps.Add(conn.Weight, RandomWeight());
            }
        }
    }

    /// <summary>
    /// Generates a random weight value for neural network connections.
    /// </summary>
    /// <returns>A random weight value between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a random weight value between -1 and 1 for use in neural network connections.
    /// These random weights provide the initial diversity for the evolutionary process and are also used
    /// during weight mutation to introduce changes to connection strengths.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a random connection weight value between -1 and 1.
    /// 
    /// In neural networks:
    /// - Connections have weight values that determine their strength and effect
    /// - Positive weights are excitatory (they increase activation)
    /// - Negative weights are inhibitory (they decrease activation)
    /// 
    /// This method is used:
    /// - When creating initial networks with random weights
    /// - During mutation to change existing weights
    /// 
    /// Starting with random weights gives the evolutionary process diverse
    /// starting points to work with, increasing the chances of finding good solutions.
    /// </para>
    /// </remarks>
    private T RandomWeight()
    {
        return NumOps.FromDouble(Random.NextDouble() * 2 - 1);
    }

    /// <summary>
    /// Performs a forward pass through the best genome in the population to generate a prediction from an input vector.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the best-performing genome (neural network) in the population to make predictions.
    /// It identifies the genome with the highest fitness score and activates it with the provided input vector.
    /// The genome's neural network processes the input through its evolved topology to produce an output.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the best neural network from your population to make a prediction.
    /// 
    /// When you need to use your evolved neural network:
    /// 1. NEAT finds the network with the highest fitness score
    /// 2. It passes your input through this network
    /// 3. The network processes the input through its evolved structure
    /// 4. The result is returned as the prediction
    /// 
    /// The beauty of NEAT is that this network's structure was not designed manually
    /// but discovered through evolution. It may have a unique arrangement of neurons
    /// and connections that a human designer might not have thought of.
    /// 
    /// After many generations of evolution, this best network should perform well
    /// on your target problem.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        // Use the best genome for prediction
        var bestGenome = _population.OrderByDescending(g => g.Fitness).First();
        return bestGenome.Activate(input);
    }

    /// <summary>
    /// Not implemented for NEAT, as it evolves parameters through natural selection rather than direct updates.
    /// </summary>
    /// <param name="parameters">A vector containing parameters to update.</param>
    /// <exception cref="NotImplementedException">Always thrown, as this method is not applicable to NEAT.</exception>
    /// <remarks>
    /// <para>
    /// This method is not implemented for NEAT because NEAT evolves network parameters through the evolutionary
    /// process rather than through direct parameter updates. Instead of using gradient-based optimization or
    /// similar techniques, NEAT relies on selection, crossover, and mutation to improve parameters over generations.
    /// </para>
    /// <para><b>For Beginners:</b> This method is not used in NEAT because parameters are evolved, not directly updated.
    /// 
    /// In traditional neural networks:
    /// - Parameters (weights) are directly updated based on gradients
    /// - You can set exact values for each parameter
    /// 
    /// In NEAT:
    /// - Parameters evolve through natural selection
    /// - Better-performing networks reproduce more often
    /// - Parameters change through crossover and mutation
    /// 
    /// Since NEAT uses evolution instead of direct parameter updates,
    /// this method throws an exception if called. You should use
    /// the EvolvePopulation method instead to improve the networks.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // NEAT doesn't use this method for parameter updates
        throw new NotImplementedException("NEAT doesn't support direct parameter updates.");
    }

    /// <summary>
    /// Serializes the NEAT algorithm state to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized state to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the NEAT algorithm's state to a binary format that can be stored and later loaded.
    /// It writes the evolution parameters (population size, mutation rate, crossover rate), the current innovation
    /// number, and the entire population of genomes. Each genome is serialized individually, preserving the
    /// evolved network structures and their fitness values.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the current state of your NEAT system to a file.
    /// 
    /// When you save a NEAT system, several things are preserved:
    /// 
    /// 1. Configuration parameters:
    ///    - Population size
    ///    - Mutation rate
    ///    - Crossover rate
    ///    - Current innovation number counter
    /// 
    /// 2. The entire population:
    ///    - All evolved neural networks
    ///    - Their structures (neurons and connections)
    ///    - Their fitness scores
    /// 
    /// This comprehensive saving ensures that when you reload the system later,
    /// you can continue evolution from exactly where you left off, without
    /// losing any of the progress made so far.
    /// 
    /// This is useful for:
    /// - Pausing and resuming long evolutionary runs
    /// - Backing up particularly successful populations
    /// - Distributing evolved neural networks to others
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_populationSize);
        writer.Write(_mutationRate);
        writer.Write(_crossoverRate);
        writer.Write(_innovationNumber);

        writer.Write(_population.Count);
        foreach (var genome in _population)
        {
            genome.Serialize(writer);
        }
    }

    /// <summary>
    /// Deserializes the NEAT algorithm state from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized state from.</param>
    /// <remarks>
    /// <para>
    /// This method loads a previously serialized NEAT algorithm state from a binary format. It reads the
    /// evolution parameters (population size, mutation rate, crossover rate), the current innovation number,
    /// and reconstructs the entire population of genomes. Each genome is deserialized individually, restoring
    /// the evolved network structures and their fitness values.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved NEAT system from a file.
    /// 
    /// When loading a NEAT system, the method restores:
    /// 
    /// 1. Configuration parameters:
    ///    - Population size
    ///    - Mutation rate
    ///    - Crossover rate
    ///    - Current innovation number counter
    /// 
    /// 2. The entire population:
    ///    - All evolved neural networks
    ///    - Their structures (neurons and connections)
    ///    - Their fitness scores
    /// 
    /// This allows you to:
    /// - Resume evolution from exactly where you left off
    /// - Continue working with previously evolved networks
    /// - Import networks evolved by others
    /// 
    /// After loading, the NEAT system is in exactly the same state it was when saved,
    /// ready to continue evolution or to use the best-evolved network for predictions.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        _populationSize = reader.ReadInt32();
        _mutationRate = reader.ReadDouble();
        _crossoverRate = reader.ReadDouble();
        _innovationNumber = reader.ReadInt32();

        int populationCount = reader.ReadInt32();
        _population = new List<Genome<T>>();
        for (int i = 0; i < populationCount; i++)
        {
            var genome = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);
            genome.Deserialize(reader);
            _population.Add(genome);
        }
    }
}