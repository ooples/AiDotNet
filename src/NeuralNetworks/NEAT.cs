using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Options;

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
/// <example>
/// <code>
/// var options = new NEATOptions { InputSize = 4, OutputSize = 2, PopulationSize = 150 };
/// var model = new NEAT&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 4 });
/// var output = model.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Evolving Neural Networks through Augmenting Topologies", "https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf", Year = 2002, Authors = "Kenneth O. Stanley, Risto Miikkulainen")]
public class NEAT<T> : NeuralNetworkBase<T>
{
    private readonly NEATOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    private T _mutationRate { get; set; }

    /// <summary>
    /// Per-instance deterministic RNG for the evolutionary search (selection,
    /// crossover, mutation). Seeded from the architecture's RandomSeed when present.
    /// </summary>
    private readonly Random _rng;

    // Faithful NEAT weight-mutation constants (Stanley & Miikkulainen 2002 §3.1;
    // NEAT-Python's weight_mutate_power / weight_replace_rate / weight_max_value).
    // A small perturbation power plus a hard clamp keep connection weights bounded
    // so weights cannot random-walk to ever-larger magnitudes across generations.
    private const double WeightPerturbPower = 0.5;
    private const double WeightReplaceRate = 0.1;
    private const double WeightCap = 8.0;

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
    private T _crossoverRate { get; set; }

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

    // Issue #1392 perf: scratch HashSet reused across Crossover calls so we
    // don't allocate a fresh one for every offspring. EvolvePopulation is
    // single-threaded, so a per-instance buffer is safe; the .Clear() at the
    // top of Crossover resets it without releasing the underlying entries
    // table.
    private readonly HashSet<int> _crossoverSeen = new HashSet<int>();

    private const int DefaultInputSize = 10;
    private const int DefaultOutputSize = 1;
    private const int DefaultPopulationSize = 150;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public NEAT()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: DefaultInputSize,
            outputSize: DefaultOutputSize),
            populationSize: DefaultPopulationSize)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="NEAT{T}"/> class with the specified architecture and evolution parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input and output sizes.</param>
    /// <param name="populationSize">The number of individual genomes in the population.</param>
    /// <param name="mutationRate">The probability of mutation occurring during reproduction. Default is 0.1.</param>
    /// <param name="crossoverRate">The probability of crossover occurring during reproduction. Default is 0.75.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new NEAT system with your chosen settings.</para>
    /// </remarks>
    public NEAT(NeuralNetworkArchitecture<T> architecture, int populationSize, double mutationRate = 0.1, double crossoverRate = 0.75, ILossFunction<T>? lossFunction = null, NEATOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new NEATOptions();
        Options = _options;
        _populationSize = populationSize;
        _mutationRate = NumOps.FromDouble(mutationRate);
        _crossoverRate = NumOps.FromDouble(crossoverRate);
        _innovationNumber = 0;
        // Deterministic, per-instance evolution RNG. NEAT's selection / crossover /
        // mutation all draw from this stream; seeding it from the architecture's
        // RandomSeed makes a NEAT run reproducible (the standard NEAT-Python contract —
        // every config carries a seed). Without it the evolution drew from the
        // process-shared RandomHelper.ThreadSafeRandom, whose state advances with
        // unrelated prior work, so a NEAT invariant could pass in isolation yet flake
        // when interleaved with other tests. When no seed is set (the production
        // default) it falls back to the shared RNG, preserving non-reproducible
        // production behaviour.
        _rng = Architecture.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(Architecture.RandomSeed.Value)
            : RandomHelper.ThreadSafeRandom;
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
    /// Gets the total number of trainable parameters (connections) in the best genome.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In NEAT, parameters are the connection weights in a genome. This property returns
    /// the number of connections in the best-performing genome in the population.
    /// </para>
    /// </remarks>
    public override long ParameterCount
    {
        get
        {
            if (_population == null || _population.Count == 0)
                return 0;

            // Return the connection count from the best genome
            var bestGenome = GetBestGenome();
            return bestGenome?.Connections?.Count ?? 0;
        }
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

            // Sort population by fitness (descending)
            _population.Sort((a, b) =>
            {
                if (NumOps.GreaterThan(b.Fitness, a.Fitness)) return 1;
                if (NumOps.GreaterThan(a.Fitness, b.Fitness)) return -1;
                return 0;
            });

            // Create new population — issue #1392 perf: pre-size to the final
            // capacity so the underlying array doesn't walk the 0→4→8→…
            // capacity-doubling chain and memcpy on every grow. Population
            // size is fixed for the lifetime of a NEAT instance.
            var newPopulation = new List<Genome<T>>(_populationSize);

            // Elitism: Keep the best individual
            newPopulation.Add(_population[0]);

            while (newPopulation.Count < _populationSize)
            {
                if (NumOps.LessThan(NumOps.FromDouble(_rng.NextDouble()), _crossoverRate))
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
        // Tournament selection. Issue #1392 perf: the prior implementation
        // allocated a fresh List<Genome<T>> and an OrderByDescending LINQ
        // enumerator on every call. Across a Train run that's ~447 k allocs
        // (149 children × 50 generations × ~30 Train calls × 2 parents per
        // crossover). Tournament is fixed-size 3 — pick three random genomes
        // and keep the running argmax inline. No allocations, no LINQ.
        const int tournamentSize = 3;
        int n = _population.Count;
        var best = _population[_rng.Next(n)];
        var bestFitness = best.Fitness;
        for (int i = 1; i < tournamentSize; i++)
        {
            var candidate = _population[_rng.Next(n)];
            if (NumOps.GreaterThan(candidate.Fitness, bestFitness))
            {
                best = candidate;
                bestFitness = candidate.Fitness;
            }
        }

        return best;
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
        // Issue #1392 perf: prior implementation used Enumerable.Concat (one
        // enumerator alloc) and a fresh HashSet per call (rehash table alloc).
        // Reuse the per-instance _crossoverSeen HashSet and walk both parent
        // connection lists by index so the JIT can elide bounds checks. Child
        // connection list is pre-sized to the upper bound (parent1.Count +
        // parent2.Count) to skip the List capacity-doubling chain in the
        // common case where most innovations are unique.
        var p1 = parent1.Connections;
        var p2 = parent2.Connections;
        int p1Count = p1.Count;
        int p2Count = p2.Count;
        var child = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);
        int upperBound = p1Count + p2Count;
        if (upperBound > 0)
        {
            child.Connections.Capacity = upperBound;
        }

        var seen = _crossoverSeen;
        seen.Clear();
        var childConnections = child.Connections;
        for (int i = 0; i < p1Count; i++)
        {
            var c = p1[i];
            if (seen.Add(c.Innovation))
            {
                childConnections.Add(new Connection<T>(c.FromNode, c.ToNode, c.Weight, c.IsEnabled, c.Innovation));
            }
        }
        for (int i = 0; i < p2Count; i++)
        {
            var c = p2[i];
            if (seen.Add(c.Innovation))
            {
                childConnections.Add(new Connection<T>(c.FromNode, c.ToNode, c.Weight, c.IsEnabled, c.Innovation));
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
        // Issue #1392 perf: prior implementation used LINQ Max+Any on every
        // call, allocating a Func delegate + an enumerator each time. Walk
        // Connections by index. Add-node case derives the new node id from
        // an inline scan over (FromNode, ToNode) since the existing
        // CachedMaxNodeId on Genome covers max(referenced node id,
        // biasNodeId), which already gives us "first free node id - 1".
        var connections = genome.Connections;
        int count = connections.Count;

        if (NumOps.LessThan(NumOps.FromDouble(_rng.NextDouble()), _mutationRate) && count > 0)
        {
            // Add new node
            var connection = connections[_rng.Next(count)];

            // First-free-node-id scan including the bias node — without the
            // explicit biasNodeId floor below, the first add-node mutation
            // on an initial genome (where max(FromNode, ToNode) =
            // InputSize + OutputSize − 1) would produce newNodeId =
            // InputSize + OutputSize, which collides with biasNodeId.
            // ActivateGenome writes activations[biasNodeId] = NumOps.One
            // BEFORE the connection sweep, so any connection accumulating
            // into this hidden slot would corrupt the bias value — and
            // every connection targeting it would also read a polluted
            // pre-activation from the same slot.
            int biasNodeId = Architecture.InputSize + Architecture.OutputSize;
            int maxNodeId = biasNodeId;
            for (int i = 0; i < count; i++)
            {
                var c = connections[i];
                int hi = c.FromNode > c.ToNode ? c.FromNode : c.ToNode;
                if (hi > maxNodeId) maxNodeId = hi;
            }
            int newNodeId = maxNodeId + 1;

            genome.DisableConnection(connection.Innovation);
            genome.AddConnection(connection.FromNode, newNodeId, NumOps.One, true, _innovationNumber++);
            genome.AddConnection(newNodeId, connection.ToNode, connection.Weight, true, _innovationNumber++);

            // List grew under us; reload local references for downstream loops
            connections = genome.Connections;
            count = connections.Count;
        }

        if (NumOps.LessThan(NumOps.FromDouble(_rng.NextDouble()), _mutationRate))
        {
            // Add new connection
            int fromNode = _rng.Next(Architecture.InputSize + Architecture.OutputSize);
            int toNode = _rng.Next(Architecture.InputSize, Architecture.InputSize + Architecture.OutputSize);
            bool exists = false;
            for (int i = 0; i < count; i++)
            {
                var c = connections[i];
                if (c.FromNode == fromNode && c.ToNode == toNode)
                {
                    exists = true;
                    break;
                }
            }
            if (!exists)
            {
                genome.AddConnection(fromNode, toNode, RandomWeight(), true, _innovationNumber++);
                connections = genome.Connections;
                count = connections.Count;
            }
        }

        // Mutate weights — index loop so JIT can elide the List<T>.Enumerator
        // bounds check overhead the foreach pays per step.
        T perturbPower = NumOps.FromDouble(WeightPerturbPower);
        T replaceRate = NumOps.FromDouble(WeightReplaceRate);
        T weightCap = NumOps.FromDouble(WeightCap);
        T negWeightCap = NumOps.FromDouble(-WeightCap);
        for (int i = 0; i < count; i++)
        {
            if (NumOps.LessThan(NumOps.FromDouble(_rng.NextDouble()), _mutationRate))
            {
                var conn = connections[i];
                // Faithful NEAT weight mutation (Stanley & Miikkulainen 2002 §3.1; the
                // same scheme as NEAT-Python's weight_mutate_power / weight_replace_rate /
                // weight_max_value): with the per-connection mutation probability, EITHER
                // replace the weight with a fresh random value (small replace rate) OR
                // nudge it by a SMALL perturbation, then CLAMP to a bounded range. The
                // previous `weight += RandomWeight()` added a full uniform[-1,1] every
                // time — an UNBOUNDED RANDOM WALK whose variance grows with the number of
                // perturbations, so weight magnitudes drift without limit across
                // generations. A small perturbation power plus a hard clamp keep the
                // weights bounded, matching the paper's bounded-weight evolutionary search.
                if (NumOps.LessThan(NumOps.FromDouble(_rng.NextDouble()), replaceRate))
                {
                    conn.Weight = RandomWeight();
                }
                else
                {
                    T perturbation = NumOps.Multiply(
                        NumOps.FromDouble(_rng.NextDouble() * 2 - 1), perturbPower);
                    conn.Weight = NumOps.Add(conn.Weight, perturbation);
                }
                if (NumOps.GreaterThan(conn.Weight, weightCap)) conn.Weight = weightCap;
                else if (NumOps.LessThan(conn.Weight, negWeightCap)) conn.Weight = negWeightCap;
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
        return NumOps.FromDouble(_rng.NextDouble() * 2 - 1);
    }

    /// <summary>
    /// Updates the connection weights of the best genome using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing parameters to update.</param>
    /// <exception cref="InvalidOperationException">Thrown when the best genome has no connections.</exception>
    /// <exception cref="ArgumentException">Thrown when parameter vector length doesn't match connection count.</exception>
    /// <remarks>
    /// <para>
    /// This method allows direct parameter updates to the best genome's connection weights, enabling
    /// integration with external optimization or parameter management systems. Note that this bypasses
    /// NEAT's evolutionary mechanisms and should be used carefully.
    /// </para>
    /// <para><b>For Beginners:</b> This method allows direct weight updates when needed.
    ///
    /// In traditional NEAT:
    /// - Parameters evolve through natural selection
    /// - Better-performing networks reproduce more often
    /// - Parameters change through crossover and mutation
    ///
    /// However, this method allows you to:
    /// - Directly set connection weights on the best genome
    /// - Integrate with external optimization algorithms
    /// - Transfer parameters from other sources
    ///
    /// <b>Important:</b> Changes may be lost if the modified genome doesn't survive selection
    /// in subsequent evolution cycles. For typical NEAT training, use the EvolvePopulation method instead.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        var bestGenome = GetBestGenome();

        if (bestGenome.Connections.Count == 0)
        {
            throw new InvalidOperationException("Best genome has no connections to update.");
        }

        if (parameters.Length != bestGenome.Connections.Count)
        {
            throw new ArgumentException($"Parameter vector length mismatch. Expected {bestGenome.Connections.Count} parameters but got {parameters.Length}.", nameof(parameters));
        }

        for (int i = 0; i < bestGenome.Connections.Count; i++)
        {
            bestGenome.Connections[i].Weight = parameters[i];
        }
    }

    /// <summary>
    /// Predicts output values for input data using the best genome in the population.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the highest-fitness genome in the population to make predictions. It activates the 
    /// genome's neural network with the provided input data and returns the resulting output activations.
    /// For batch inputs, it processes each sample independently.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the best evolved network to make predictions.
    /// 
    /// When making a prediction:
    /// - NEAT uses the highest-performing network from the population
    /// - The input data is fed into this network
    /// - The network processes the data through its evolved structure
    /// - The resulting output values are returned
    /// 
    /// Unlike traditional neural networks with fixed structures, the network used here
    /// has evolved its structure through the evolutionary process, potentially developing
    /// complex and unique connection patterns that solve the problem effectively.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Get the best genome (the one with highest fitness)
        var bestGenome = GetBestGenome();

        // Treat ANY rank-2 input as batched, even when the batch size is 1.
        // Returning rank-1 for single-sample input was a real bug —
        // NEAT.Train downstream reads `expectedOutput.Shape[1]`, which throws
        // IndexOutOfRangeException for rank-1 targets, and EffectiveOutputShape
        // cached the wrong rank for the test's warm-up Predict path. Per the
        // implicit `(batch, features)` contract used by every other neural
        // network in the codebase, output rank should match input rank.
        //
        // Validate rank explicitly: NEAT genomes activate over a flat feature
        // vector, so anything beyond rank-2 (e.g., a stray rank-3 image
        // tensor or rank-4 video) cannot be unambiguously interpreted as
        // batched-features and would mis-index in the loop below. Fail
        // fast at the boundary instead of producing garbage outputs.
        if (input.Shape.Length < 1 || input.Shape.Length > 2)
        {
            throw new ArgumentException(
                $"NEAT.Predict expects rank-1 [features] or rank-2 [batch, features]; " +
                $"got rank {input.Shape.Length} (shape [{string.Join(",", input.Shape)}]).",
                nameof(input));
        }
        bool isBatch = input.Shape.Length == 2;

        if (isBatch)
        {
            // Process each input in the batch
            int batchSize = input.Shape[0];
            int featureSize = input.Shape[1];

            // Create output tensor with correct shape
            var output = TensorAllocator.Rent<T>(new int[] { batchSize, Architecture.OutputSize });

            // Process each sample
            for (int b = 0; b < batchSize; b++)
            {
                // Extract individual input
                var sampleInput = new Vector<T>(featureSize);
                for (int f = 0; f < featureSize; f++)
                {
                    sampleInput[f] = input[b, f];
                }

                // Get activations for this sample
                var activations = ActivateGenome(bestGenome, sampleInput);

                // Store output activations
                for (int o = 0; o < Architecture.OutputSize; o++)
                {
                    output[b, o] = activations[Architecture.InputSize + o];
                }
            }

            return output;
        }
        else
        {
            // Single input
            // Convert input tensor to vector
            var inputVector = input.ToVector();

            // Get activations
            var activations = ActivateGenome(bestGenome, inputVector);

            // Create output tensor
            var output = TensorAllocator.Rent<T>(new int[] { Architecture.OutputSize });
            for (int i = 0; i < Architecture.OutputSize; i++)
            {
                output[i] = activations[Architecture.InputSize + i];
            }

            return output;
        }
    }

    /// <summary>
    /// Gets the genome with the highest fitness from the population.
    /// </summary>
    /// <returns>The best genome in the population.</returns>
    /// <remarks>
    /// <para>
    /// This method sorts the population by fitness in descending order and returns the genome
    /// with the highest fitness score. If the population hasn't been evaluated yet, it assigns
    /// a default fitness value to each genome.
    /// </para>
    /// <para><b>For Beginners:</b> This finds the top-performing network from the population.
    /// 
    /// This method:
    /// - Sorts all networks based on their fitness scores (performance)
    /// - Returns the network with the highest score
    /// - If networks haven't been evaluated yet, it assigns a neutral score
    /// 
    /// The returned network is the "champion" of the population - the one that
    /// has evolved to best solve the problem you're working on.
    /// </para>
    /// </remarks>
    private Genome<T> GetBestGenome()
    {
        // Check if any genomes have fitness set
        bool anyFitnessSet = _population.Any(g => !NumOps.Equals(g.Fitness, NumOps.Zero));

        // If no fitness values are set, assign default
        if (!anyFitnessSet)
        {
            foreach (var genome in _population)
            {
                genome.Fitness = NumOps.One; // Neutral fitness
            }
        }

        // Find the genome with highest fitness (O(n) instead of O(n log n) sort)
        var best = _population[0];
        for (int i = 1; i < _population.Count; i++)
        {
            if (NumOps.GreaterThan(_population[i].Fitness, best.Fitness))
            {
                best = _population[i];
            }
        }

        return best;
    }

    /// <summary>
    /// Activates a genome's neural network with the given input.
    /// </summary>
    /// <param name="genome">The genome to activate.</param>
    /// <param name="input">The input values.</param>
    /// <returns>A dictionary mapping node IDs to their activation values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the genome's neural network. It initializes
    /// input nodes with the provided values, processes connections in a topologically sorted order,
    /// and applies activation functions to produce the final node activations.
    /// </para>
    /// <para><b>For Beginners:</b> This runs input data through the evolved neural network.
    /// 
    /// The activation process:
    /// 1. Sets the input nodes to the provided input values
    /// 2. Processes connections in the correct order (feed-forward)
    /// 3. Applies the activation function to each neuron
    /// 4. Returns all neuron activation values
    /// 
    /// This is how a NEAT network processes information, similar to a traditional
    /// neural network but with the specific connection structure that evolved
    /// during the evolutionary process.
    /// </para>
    /// </remarks>
    private T[] ActivateGenome(Genome<T> genome, Vector<T> input)
    {
        // Issue #1392 perf: switched from Dictionary<int, T> to flat T[] indexed
        // by node id. NEAT node ids are dense small integers (inputs 0..InputSize-1,
        // outputs InputSize..InputSize+OutputSize-1, bias = InputSize+OutputSize,
        // hidden > biasNodeId). The Dictionary form added 2-3 heap allocations per
        // call (Dictionary instance + internal buckets[] + entries[]) plus hash-
        // compute + bucket-walk on every node access. With a flat array each
        // activation is a single contiguous-memory store/load.
        //
        // Buffer size = max(node id observed in genome's connections, biasNodeId) + 1.
        // Indices not referenced by any connection stay at default(T) and are
        // never queried by callers, so over-provisioning is benign.

        int inputSize = Architecture.InputSize;
        int outputSize = Architecture.OutputSize;
        int biasNodeId = inputSize + outputSize;

        // Sort connections topologically for proper feed-forward activation.
        // Cached on the genome — weight-only mutations (the dominant case across
        // the 50 internal generations per Train call) keep the topology
        // signature unchanged, so the cache hits and skips the O(E²) sort.
        var sortedConnections = GetOrBuildSortedConnections(genome);

        int maxNodeId = GetOrBuildMaxNodeId(genome, biasNodeId);
        var activations = new T[maxNodeId + 1];

        // Set input nodes
        for (int i = 0; i < inputSize; i++)
        {
            activations[i] = input[i];
        }

        // Set bias node
        activations[biasNodeId] = NumOps.One;

        // Output nodes + every other slot are pre-zeroed by `new T[]`
        // (default(T) = NumOps.Zero for built-in numeric types) — no
        // explicit init loop needed, no per-connection ContainsKey gymnastics
        // since every FromNode/ToNode id is in range by construction of maxNodeId.

        // Process connections in topological order.
        foreach (var connection in sortedConnections)
        {
            if (!connection.IsEnabled) continue;
            T weightedInput = NumOps.Multiply(activations[connection.FromNode], connection.Weight);
            activations[connection.ToNode] = NumOps.Add(activations[connection.ToNode], weightedInput);
        }

        // Apply activation function to all non-input nodes the genome actually
        // references (cached on the genome alongside the sort + max-node-id).
        var nonInputNodes = GetOrBuildReferencedNonInputNodeIds(genome, inputSize);
        foreach (var nodeId in nonInputNodes)
        {
            activations[nodeId] = ApplySigmoid(activations[nodeId]);
        }

        return activations;
    }

    /// <summary>
    /// Issue #1392 perf helper: returns the cached topologically-sorted
    /// connection list for <paramref name="genome"/>, rebuilding only when
    /// the topology signature changed since the last call.
    /// </summary>
    private List<Connection<T>> GetOrBuildSortedConnections(Genome<T> genome)
    {
        int count = genome.Connections.Count;
        ulong signature = ComputeTopologySignature(genome.Connections);
        if (genome.CachedSortedConnections != null
            && genome.CachedTopologySignatureCount == count
            && genome.CachedTopologySignatureMask == signature)
        {
            return genome.CachedSortedConnections;
        }
        var sorted = SortConnectionsTopologically(genome);
        genome.CachedSortedConnections = sorted;
        genome.CachedTopologySignatureCount = count;
        genome.CachedTopologySignatureMask = signature;
        // Invalidate the dependent caches — their content depends on the
        // connection set, so a topology change forces a rebuild on the next
        // call.
        genome.CachedNonInputNodeIds = null;
        genome.CachedMaxNodeId = -1;
        return sorted;
    }

    /// <summary>
    /// Issue #1392 perf helper: returns max(FromNode, ToNode, biasNodeId)
    /// across the genome's enabled connections. Cached on the genome so the
    /// per-call O(E) scan only runs after topology mutations. Used to size
    /// <see cref="ActivateGenome"/>'s flat activation buffer.
    /// </summary>
    private static int GetOrBuildMaxNodeId(Genome<T> genome, int biasNodeId)
    {
        if (genome.CachedMaxNodeId >= 0)
        {
            return genome.CachedMaxNodeId;
        }
        int max = biasNodeId;
        var conns = genome.Connections;
        for (int i = 0; i < conns.Count; i++)
        {
            var c = conns[i];
            if (c.FromNode > max) max = c.FromNode;
            if (c.ToNode > max) max = c.ToNode;
        }
        genome.CachedMaxNodeId = max;
        return max;
    }

    /// <summary>
    /// Issue #1392 perf helper: caches the list of node IDs &gt;= InputSize that
    /// the sigmoid sweep should touch. Built directly from the connection list
    /// (any FromNode/ToNode &gt;= InputSize that the genome references) so we
    /// don't need to materialize a Dictionary first. Invalidated alongside
    /// <see cref="GetOrBuildSortedConnections"/>.
    /// </summary>
    private List<int> GetOrBuildReferencedNonInputNodeIds(Genome<T> genome, int inputSize)
    {
        if (genome.CachedNonInputNodeIds != null)
        {
            return genome.CachedNonInputNodeIds;
        }
        var seen = new HashSet<int>();
        var list = new List<int>();
        foreach (var c in genome.Connections)
        {
            if (!c.IsEnabled) continue;
            if (c.FromNode >= inputSize && seen.Add(c.FromNode)) list.Add(c.FromNode);
            if (c.ToNode >= inputSize && seen.Add(c.ToNode)) list.Add(c.ToNode);
        }
        // Also include output nodes (always activated even if no connection
        // wrote to them, because ActivateGenome zero-initializes the slot
        // and the sigmoid should still run on the zero value for caller
        // consistency with the pre-refactor Dictionary behavior).
        int outputSize = Architecture.OutputSize;
        for (int i = 0; i < outputSize; i++)
        {
            int outNode = inputSize + i;
            if (seen.Add(outNode)) list.Add(outNode);
        }
        // Bias node: the pre-refactor Dictionary implementation set
        // activations[InputSize+OutputSize] = NumOps.One BEFORE the sigmoid
        // sweep and then iterated `activations.Keys.Where(k >= InputSize)`,
        // so the bias slot was overwritten with Sigmoid(1) = ~0.731 by the
        // end. Preserve that exact behavior here so callers that read the
        // bias slot from the returned array see the same value they saw
        // before this refactor.
        int biasNodeId = inputSize + outputSize;
        if (seen.Add(biasNodeId)) list.Add(biasNodeId);
        genome.CachedNonInputNodeIds = list;
        return list;
    }

    /// <summary>
    /// O(N) FNV-1a hash over every connection slot's <c>(FromNode,
    /// ToNode, IsEnabled)</c> tuple in iteration order. Used together
    /// with <c>Connections.Count</c> as the cache key for the
    /// topologically-sorted connection list on <see cref="Genome{T}"/>.
    /// </summary>
    /// <remarks>
    /// Replaces an earlier bitmask-of-enabled-flags signature that
    /// missed two real edit patterns and let stale cached sorts /
    /// non-input-node sets / max-node-id leak back to callers:
    /// <list type="bullet">
    /// <item>Same-count rewires — swapping a connection's <c>FromNode</c>
    /// or <c>ToNode</c> for a different node without flipping any
    /// <c>IsEnabled</c> bit preserved both <c>Count</c> and the bitmask
    /// → cache hit on the WRONG topology.</item>
    /// <item>&gt;64-connection aliasing — the bitmask's <c>(i &amp; 63)</c>
    /// wrap collapsed slots 0/64/128/… onto the same bit, so a flip at
    /// slot 64 could XOR-cancel an earlier flip at slot 0 and leave the
    /// mask unchanged.</item>
    /// </list>
    /// Weight is deliberately excluded from the signature — weight-only
    /// mutations are the dominant case across the 50 internal
    /// generations per public <c>Train</c> call, and we WANT the cached
    /// topological sort to survive them.
    /// </remarks>
    private static ulong ComputeTopologySignature(List<Connection<T>> connections)
    {
        // FNV-1a 64-bit hash of every connection slot's
        // (FromNode, ToNode, IsEnabled) tuple in iteration order. The
        // earlier ComputeEnabledBitmask only hashed the enabled flags and
        // would alias same-count rewires/replacements (e.g. swapping a
        // connection's FromNode preserved both count and enabled-bitmask
        // → stale cache → wrong activation). Hashing the full tuple
        // also avoids the >64-connection aliasing the bitmask suffered
        // once the wrap-around in `(i & 63)` started folding bits.
        // Connection.Weight is intentionally excluded — weight-only
        // mutations are the dominant case across the 50 internal
        // generations per Train call and we WANT the cached topological
        // sort to survive them.
        const ulong FnvOffsetBasis = 14695981039346656037UL;
        const ulong FnvPrime = 1099511628211UL;
        ulong hash = FnvOffsetBasis;
        int n = connections.Count;
        for (int i = 0; i < n; i++)
        {
            var c = connections[i];
            hash ^= (ulong)(uint)c.FromNode;
            hash *= FnvPrime;
            hash ^= (ulong)(uint)c.ToNode;
            hash *= FnvPrime;
            hash ^= c.IsEnabled ? 1UL : 0UL;
            hash *= FnvPrime;
        }
        return hash;
    }

    /// <summary>
    /// Sorts connections in topological order for proper feed-forward activation.
    /// </summary>
    /// <param name="genome">The genome containing connections to sort.</param>
    /// <returns>A list of connections sorted in topological order.</returns>
    /// <remarks>
    /// <para>
    /// This method sorts the connections in a genome to ensure they are processed in the correct
    /// order during network activation. It creates layers of nodes based on their depth in the network
    /// and sorts connections accordingly.
    /// </para>
    /// <para><b>For Beginners:</b> This determines the correct order to process connections.
    /// 
    /// In a neural network:
    /// - Information flows from input to output
    /// - Connections must be processed in the correct order
    /// - Inputs need to be calculated before they can be used
    /// 
    /// This method:
    /// - Figures out which neurons depend on which other neurons
    /// - Sorts connections so inputs are always processed before outputs
    /// - Ensures the network processes information in a feed-forward manner
    /// 
    /// This is especially important in NEAT since the connection structure
    /// evolves and isn't fixed in predefined layers.
    /// </para>
    /// </remarks>
    private List<Connection<T>> SortConnectionsTopologically(Genome<T> genome)
    {
        // Create a dictionary to track nodes that feed into each node
        var incomingConnections = new Dictionary<int, List<Connection<T>>>();

        // Create a set of all nodes
        var allNodes = new HashSet<int>();

        // Populate incoming connections and collect all nodes
        foreach (var conn in genome.Connections)
        {
            if (!conn.IsEnabled) continue;

            allNodes.Add(conn.FromNode);
            allNodes.Add(conn.ToNode);

            if (!incomingConnections.ContainsKey(conn.ToNode))
            {
                incomingConnections[conn.ToNode] = new List<Connection<T>>();
            }

            incomingConnections[conn.ToNode].Add(conn);
        }

        // Create a dictionary to track processed nodes
        var processedNodes = new Dictionary<int, bool>();

        // Input nodes don't have incoming connections and are already processed
        for (int i = 0; i < Architecture.InputSize; i++)
        {
            processedNodes[i] = true;
        }

        // Sort connections
        var sortedConnections = new List<Connection<T>>();
        var sortedSet = new HashSet<int>(); // Track sorted connections by innovation for O(1) lookup

        // Pre-filter enabled connections to avoid repeated enumeration
        var enabledConnections = genome.Connections.Where(c => c.IsEnabled).ToList();
        int enabledCount = enabledConnections.Count;

        // Process until all connections are sorted
        while (sortedConnections.Count < enabledCount)
        {
            bool addedConnection = false;

            // Check each enabled connection
            foreach (var conn in enabledConnections)
            {
                // Skip if already in sorted list (O(1) with HashSet)
                if (sortedSet.Contains(conn.Innovation)) continue;

                // Check if from node is processed
                if (processedNodes.ContainsKey(conn.FromNode) && processedNodes[conn.FromNode])
                {
                    // Add connection to sorted list
                    sortedConnections.Add(conn);
                    sortedSet.Add(conn.Innovation);

                    // Mark to node as processed
                    processedNodes[conn.ToNode] = true;

                    addedConnection = true;
                }
            }

            // If no connections were added in this iteration, we might have a cycle
            if (!addedConnection) break;
        }

        return sortedConnections;
    }

    /// <summary>
    /// Applies the sigmoid activation function to a value.
    /// </summary>
    /// <param name="value">The input value.</param>
    /// <returns>The sigmoid of the input.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the sigmoid activation function (1 / (1 + e^-x)) to the input value.
    /// Sigmoid squashes input values to the range (0, 1), which is useful for producing
    /// normalized activation values in the network.
    /// </para>
    /// <para><b>For Beginners:</b> This transforms neuron values to a value between 0 and 1.
    /// 
    /// The sigmoid function:
    /// - Takes any input value (positive or negative)
    /// - Transforms it to a value between 0 and 1
    /// - Creates a smooth, non-linear response
    /// 
    /// This non-linearity is important because:
    /// - It allows the network to learn complex patterns
    /// - It prevents the network from just computing weighted sums
    /// - It gives neurons an "activation threshold" like biological neurons
    /// 
    /// Sigmoid is one of several possible activation functions used in neural networks.
    /// </para>
    /// </remarks>
    private T ApplySigmoid(T value)
    {
        // Sigmoid function: 1 / (1 + e^-x)
        T negValue = NumOps.Negate(value);
        T expNeg = NumOps.Exp(negValue);
        T denominator = NumOps.Add(NumOps.One, expNeg);

        return NumOps.Divide(NumOps.One, denominator);
    }

    /// <summary>
    /// Trains the NEAT system using supervised learning data.
    /// </summary>
    /// <param name="input">The input training data tensor.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// This method adapts NEAT to work with traditional supervised learning data. It creates a fitness
    /// function based on the mean squared error between network predictions and expected outputs,
    /// then evolves the population to minimize this error. This allows NEAT to be used in scenarios
    /// where traditional supervised learning would be applied.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the NEAT system using example input-output pairs.
    /// 
    /// Unlike traditional neural networks that use gradient descent, NEAT learns through evolution:
    /// 1. It creates a fitness function based on prediction error
    ///    - Networks that make more accurate predictions get higher fitness scores
    ///    - Networks with lower error perform better
    /// 
    /// 2. It evolves the population for several generations
    ///    - Better networks reproduce more often
    ///    - Genetic operators (crossover and mutation) create diversity
    ///    - The population gradually improves at the task
    /// 
    /// This allows NEAT to work with supervised learning data while using its
    /// evolutionary approach to discover effective network structures.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Check if input and output have compatible batch sizes
        if (input.Shape[0] != expectedOutput.Shape[0])
        {
            throw new ArgumentException("Input and expected output must have the same batch size");
        }

        int batchSize = input.Shape[0];

        // Convert input and expected output to a format suitable for the fitness function
        var trainingData = ExtractTrainingData(input, expectedOutput);

        // Create a fitness function that measures how well each genome performs on the training data
        T fitnessFunction(Genome<T> genome)
        {
            T totalError = NumOps.Zero;

            // Calculate error for each training example
            foreach (var (sampleInput, sampleExpected) in trainingData)
            {
                // Get actual output from genome
                var activations = ActivateGenome(genome, sampleInput);

                // Extract predicted values into a vector
                var predictedVector = new Vector<T>(Architecture.OutputSize);
                for (int i = 0; i < Architecture.OutputSize; i++)
                {
                    int outputNodeId = Architecture.InputSize + i;
                    predictedVector[i] = activations[outputNodeId];
                }

                // Use the loss function to calculate error for this sample
                T sampleLoss = LossFunction.CalculateLoss(predictedVector, sampleExpected);
                totalError = NumOps.Add(totalError, sampleLoss);
            }

            // Calculate average loss across all samples
            T averageLoss = NumOps.Divide(totalError, NumOps.FromDouble(batchSize));

            // Issue #1392 perf fix: removed the inline LastLoss assignment that
            // previously fired here gated on `genome == _population.OrderByDescending
            // (g => g.Fitness).FirstOrDefault()`. That branch had two problems:
            //
            // 1. Perf: it ran `OrderByDescending` on the entire population for
            //    EVERY genome's fitness eval, giving O(N²) work per generation
            //    (N=150 default → 22,500 fitness-compare + allocations per
            //    generation × 50 generations × 30 Train calls = ~34M ops just
            //    for picking the best, before any actual genome activation).
            //    On a tiny tabular input this dominated the per-Train wall time
            //    and pushed Training_ShouldReduceLoss past the 120 s CI budget.
            //
            // 2. Correctness: the reference-equality probe `genome == best`
            //    matched the PRE-EVALUATION best (Fitness values from the
            //    previous generation) which is essentially random for the
            //    current generation. The post-evolution recompute below
            //    (lines 1130+) does the work properly by re-evaluating the
            //    actual post-generation best genome, so the inline
            //    LastLoss assignment was already dead code.
            //
            // Net: deleting the branch is pure speedup with no behavior change.

            // Convert loss to fitness (higher is better, so invert loss)
            // Add small constant to avoid division by zero
            T fitness = NumOps.Divide(NumOps.One, NumOps.Add(averageLoss, NumOps.FromDouble(0.01)));

            return fitness;
        }

        // Evolve the population for multiple generations
        // The number of generations can be adjusted based on the problem complexity
        int generations = 50;
        EvolvePopulation(fitnessFunction, generations);

        // Re-evaluate the post-evolution best genome and record its loss
        // as LastLoss. The fitness-function-side LastLoss assignment uses
        // a reference-equality probe against the pre-generation best,
        // which silently misses the post-evolution best when the
        // population reshuffles (every Train call after the first hit
        // this — LastLoss stayed at its pre-Train value or NumOps.Zero,
        // producing the misleading "step 1=0.000000, step N=0.000000"
        // failure on LossStrictlyDecreasesOnMemorizationTask). Recompute
        // here so the public Train contract surfaces a real per-call loss.
        var postBest = GetBestGenome();
        if (postBest.Connections.Count > 0 && trainingData.Count > 0)
        {
            T totalErr = NumOps.Zero;
            foreach (var (sampleInput, sampleExpected) in trainingData)
            {
                var act = ActivateGenome(postBest, sampleInput);
                var pred = new Vector<T>(Architecture.OutputSize);
                for (int i = 0; i < Architecture.OutputSize; i++)
                {
                    int outputNodeId = Architecture.InputSize + i;
                    // ActivateGenome's flat array is sized to max(node id,
                    // biasNodeId); output node ids are guaranteed in range
                    // since biasNodeId = InputSize + OutputSize > any output id.
                    pred[i] = act[outputNodeId];
                }
                totalErr = NumOps.Add(totalErr, LossFunction.CalculateLoss(pred, sampleExpected));
            }
            LastLoss = NumOps.Divide(totalErr, NumOps.FromDouble(trainingData.Count));
        }
    }

    /// <summary>
    /// Extracts training data pairs from input and expected output tensors.
    /// </summary>
    /// <param name="input">The input training data tensor.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <returns>A list of input-output vector pairs.</returns>
    /// <remarks>
    /// <para>
    /// This method converts tensor-based training data into a list of vector pairs that can be
    /// more easily processed by the NEAT algorithm. Each pair consists of an input vector and
    /// its corresponding expected output vector.
    /// </para>
    /// <para><b>For Beginners:</b> This converts tensor-based training data into a format NEAT can use.
    /// 
    /// The conversion process:
    /// 1. Takes the tensor-based input and output data
    /// 2. Extracts each individual training example
    /// 3. Creates pairs of (input, expected output) vectors
    /// 4. Returns a list of these pairs for the fitness function to use
    /// 
    /// This preprocessing step allows NEAT to work with the same types of
    /// training data used by traditional neural networks, making it more
    /// versatile for different applications.
    /// </para>
    /// </remarks>
    private List<(Vector<T> input, Vector<T> expected)> ExtractTrainingData(Tensor<T> input, Tensor<T> expectedOutput)
    {
        int batchSize = input.Shape[0];
        int inputFeatures = input.Shape[1];
        int outputFeatures = expectedOutput.Shape[1];

        var trainingData = new List<(Vector<T> input, Vector<T> expected)>(batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract input vector
            var inputVector = new Vector<T>(inputFeatures);
            for (int i = 0; i < inputFeatures; i++)
            {
                inputVector[i] = input[b, i];
            }

            // Extract expected output vector
            var expectedVector = new Vector<T>(outputFeatures);
            for (int o = 0; o < outputFeatures; o++)
            {
                expectedVector[o] = expectedOutput[b, o];
            }

            // Add pair to training data
            trainingData.Add((inputVector, expectedVector));
        }

        return trainingData;
    }

    /// <summary>
    /// Gets metadata about the NEAT model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the NEAT model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the NEAT model, including its architecture,
    /// evolutionary parameters, and population statistics. This information is useful for model
    /// management, tracking experiments, and reporting results.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about your NEAT system.
    /// 
    /// The metadata includes:
    /// - What this model is and what it does
    /// - Population size and evolutionary parameters
    /// - Statistics about the current population
    /// - Information about the best-performing network
    /// 
    /// This information is useful for:
    /// - Tracking your experiments
    /// - Comparing different NEAT runs
    /// - Documenting your work
    /// - Understanding the evolved solution
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Get the best genome
        var bestGenome = GetBestGenome();

        // Count average number of connections and nodes in the population
        double avgConnections = _population.Average(g => g.Connections.Count);
        int maxConnections = _population.Max(g => g.Connections.Count);

        // Count nodes by finding the highest node ID in each genome
        var nodeCounts = _population.Select(g =>
            g.Connections.Any() ?
                g.Connections.Max(c => Math.Max(c.FromNode, c.ToNode)) + 1 :
                Architecture.InputSize + Architecture.OutputSize
        );
        double avgNodes = nodeCounts.Average();
        int maxNodes = nodeCounts.Max();

        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "PopulationSize", _populationSize },
                { "MutationRate", NumOps.ToDouble(_mutationRate) },
                { "CrossoverRate", NumOps.ToDouble(_crossoverRate) },
                { "InnovationNumber", _innovationNumber },
                { "AverageConnections", avgConnections },
                { "MaxConnections", maxConnections },
                { "AverageNodes", avgNodes },
                { "MaxNodes", maxNodes },
                { "BestGenomeFitness", Convert.ToDouble(bestGenome.Fitness) },
                { "BestGenomeConnections", bestGenome.Connections.Count },
                { "BestGenomeEnabledConnections", bestGenome.Connections.Count(c => c.IsEnabled) }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Gets the parameters (connection weights) of the best genome.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var bestGenome = GetBestGenome();
        if (bestGenome.Connections.Count == 0)
            return new Vector<T>(0);

        var parameters = new Vector<T>(bestGenome.Connections.Count);
        for (int i = 0; i < bestGenome.Connections.Count; i++)
        {
            parameters[i] = bestGenome.Connections[i].Weight;
        }

        return parameters;
    }

    /// <summary>
    /// Yields the best genome's connection weights as a single chunk so
    /// snapshot-based parameter-change probes (Training_ShouldChangeParameters,
    /// GradientFlow_ShouldBeNonZeroAndFinite) see real evolutionary
    /// updates. The base <see cref="NeuralNetworkBase{T}.GetParameterChunks"/>
    /// walks <see cref="Layers"/>, but NEAT populates Layers with a stub
    /// representation of the best genome and the actual trainable surface
    /// lives in the genome's <c>Connections</c> list — so the inherited
    /// chunk walk reported zero changes after Train and produced false
    /// "no parameters changed" failures (#1224 Cluster F). Yielding a
    /// genome-derived chunk surfaces the evolutionary delta.
    /// </summary>
    public override System.Collections.Generic.IEnumerable<Tensor<T>> GetParameterChunks()
    {
        var paramVec = GetParameters();
        if (paramVec.Length == 0) yield break;
        var chunk = new Tensor<T>(new[] { paramVec.Length });
        for (int i = 0; i < paramVec.Length; i++) chunk[i] = paramVec[i];
        yield return chunk;
    }

    /// <summary>
    /// Gets named activations from the best genome's network when processing input.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var bestGenome = GetBestGenome();
        var inputVector = input.ToVector();
        var activations = ActivateGenome(bestGenome, inputVector);

        var result = new Dictionary<string, Tensor<T>>();

        int inputSize = Architecture.InputSize;
        int outputSize = Architecture.OutputSize;
        int biasNodeId = inputSize + outputSize;

        // Issue #1392 perf: ActivateGenome now returns a flat T[] sized to
        // max(referenced node id, biasNodeId) + 1, so input + output + bias
        // slots are guaranteed in range. The ContainsKey gymnastics from the
        // Dictionary era are unnecessary — read straight by index.

        var inputActivation = new Tensor<T>(new int[] { inputSize });
        for (int i = 0; i < inputSize; i++)
        {
            inputActivation[i] = activations[i];
        }
        result["InputNodes"] = inputActivation;

        var outputActivation = new Tensor<T>(new int[] { outputSize });
        for (int i = 0; i < outputSize; i++)
        {
            outputActivation[i] = activations[inputSize + i];
        }
        result["OutputNodes"] = outputActivation;

        // Hidden nodes: walk the genome's cached non-input-node-id list and
        // pick out the entries beyond the bias slot. Sorted ascending by id
        // for stable result ordering, matching what the prior OrderBy(k => k)
        // chain produced.
        var nonInputNodeIds = GetOrBuildReferencedNonInputNodeIds(bestGenome, inputSize);
        var hiddenNodes = new List<int>();
        foreach (var nodeId in nonInputNodeIds)
        {
            if (nodeId > biasNodeId) hiddenNodes.Add(nodeId);
        }
        hiddenNodes.Sort();

        if (hiddenNodes.Count > 0)
        {
            var hiddenActivation = new Tensor<T>(new int[] { hiddenNodes.Count });
            for (int i = 0; i < hiddenNodes.Count; i++)
            {
                hiddenActivation[i] = activations[hiddenNodes[i]];
            }
            result["HiddenNodes"] = hiddenActivation;
        }

        return result;
    }

    /// <summary>
    /// Serializes NEAT-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the state of the NEAT model to a binary stream. It serializes the
    /// evolutionary parameters, innovation number, and all genomes in the population, allowing
    /// the complete state to be restored later.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the complete state of your NEAT system to a file.
    /// 
    /// When saving the NEAT model:
    /// - Population size, mutation rate, and crossover rate are saved
    /// - The current innovation number is saved
    /// - Every genome in the population is saved with all its connections
    /// 
    /// This allows you to:
    /// - Save your progress and continue evolution later
    /// - Share evolved populations with others
    /// - Keep records of particularly successful runs
    /// - Deploy evolved networks in applications
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save NEAT parameters
        writer.Write(_populationSize);
        writer.Write(NumOps.ToDouble(_mutationRate));
        writer.Write(NumOps.ToDouble(_crossoverRate));
        writer.Write(_innovationNumber);

        // Save population
        writer.Write(_population.Count);
        foreach (var genome in _population)
        {
            // Save genome fitness
            writer.Write(Convert.ToDouble(genome.Fitness));

            // Save connections
            writer.Write(genome.Connections.Count);
            foreach (var conn in genome.Connections)
            {
                writer.Write(conn.FromNode);
                writer.Write(conn.ToNode);
                writer.Write(Convert.ToDouble(conn.Weight));
                writer.Write(conn.IsEnabled);
                writer.Write(conn.Innovation);
            }
        }
    }

    /// <summary>
    /// Deserializes NEAT-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the state of a previously saved NEAT model from a binary stream. It restores
    /// the evolutionary parameters, innovation number, and all genomes in the population, allowing
    /// evolution to continue from exactly where it left off.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a complete NEAT system from a saved file.
    /// 
    /// When loading the NEAT model:
    /// - Population size, mutation rate, and crossover rate are restored
    /// - The innovation number is restored
    /// - Every genome in the population is recreated with all its connections
    /// 
    /// This lets you:
    /// - Continue evolution from where you left off
    /// - Use previously evolved populations
    /// - Compare or combine results from different runs
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Load NEAT parameters
        _populationSize = reader.ReadInt32();
        _mutationRate = NumOps.FromDouble(reader.ReadDouble());
        _crossoverRate = NumOps.FromDouble(reader.ReadDouble());
        _innovationNumber = reader.ReadInt32();

        // Load population
        int populationCount = reader.ReadInt32();
        _population = new List<Genome<T>>(populationCount);

        for (int i = 0; i < populationCount; i++)
        {
            // Create new genome
            var genome = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);

            // Load genome fitness
            genome.Fitness = NumOps.FromDouble(reader.ReadDouble());

            // Load connections
            int connectionCount = reader.ReadInt32();
            for (int j = 0; j < connectionCount; j++)
            {
                int fromNode = reader.ReadInt32();
                int toNode = reader.ReadInt32();
                T weight = NumOps.FromDouble(reader.ReadDouble());
                bool isEnabled = reader.ReadBoolean();
                int innovation = reader.ReadInt32();

                genome.AddConnection(fromNode, toNode, weight, isEnabled, innovation);
            }

            _population.Add(genome);
        }
    }

    /// <summary>
    /// Checks if the NEAT model is ready to make predictions.
    /// </summary>
    /// <returns>True if the model is ready; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the NEAT model has a population with at least one genome that can be
    /// used for making predictions. It's useful for determining if the model has been properly
    /// initialized and evolved.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if your NEAT system is ready to use.
    /// 
    /// It verifies that:
    /// - The population exists
    /// - There is at least one genome in the population
    /// - At least one genome has connections that can process inputs
    /// 
    /// This is helpful for error checking before trying to use the model
    /// for predictions or continuing evolution.
    /// </para>
    /// </remarks>
    public bool IsReadyToPredict()
    {
        return _population != null &&
               _population.Count > 0 &&
               _population.Any(g => g.Connections.Count > 0);
    }

    /// <summary>
    /// Creates a new instance of the NEAT model with the same architecture and evolutionary parameters.
    /// </summary>
    /// <returns>A new instance of the NEAT model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new NEAT model with the same architecture, population size, mutation rate,
    /// and crossover rate as the current instance. The new instance starts with a fresh population,
    /// making it useful for restarting evolution with the same parameters or for creating parallel
    /// evolutionary runs.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a brand new NEAT system with the same settings.
    /// 
    /// This is useful when you want to:
    /// - Start over with a fresh population but keep the same settings
    /// - Run multiple separate evolutions with identical parameters
    /// - Create a "clean slate" version of a successful setup
    /// 
    /// The new NEAT system will have:
    /// - The same number of inputs and outputs
    /// - The same population size and mutation/crossover rates
    /// - A brand new initial population (not copying any evolved networks)
    /// 
    /// This effectively creates a "twin" of your NEAT system, but at the starting point
    /// rather than with any of the evolved progress.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NEAT<T>(Architecture, _populationSize, NumOps.ToDouble(_mutationRate), NumOps.ToDouble(_crossoverRate));
    }
}
