using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.NeuralNetworks.Layers;

using AiDotNet.Models.Parameters;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// NeuroEvolution of Augmenting Topologies (Stanley &amp; Miikkulainen 2002): evolves the structure and the
/// weights of feed-forward networks with historical markings, speciation and explicit fitness sharing.
/// </summary>
/// <remarks>
/// <para>
/// The algorithm follows the paper:
/// </para>
/// <para>
/// - <b>Historical markings</b>: every structural innovation gets a global innovation number; the same
/// innovation within one generation (the same connection split, or the same new link) gets the same
/// numbers in every genome. Initial genomes share one set of markings.
/// </para>
/// <para>
/// - <b>Crossover</b>: genes are lined up by innovation number; matching genes are inherited at random,
/// disjoint and excess genes from the fitter parent (from both when fitness is equal), and a gene
/// disabled in either parent stays disabled with probability 0.75.
/// </para>
/// <para>
/// - <b>Speciation</b>: compatibility distance <c>delta = c1 E / N + c2 D / N + c3 W</c> (c1 = c2 = 1,
/// c3 = 0.4, delta_t = 3); each genome joins the first species whose representative (a random member of
/// the previous generation) is within delta_t.
/// </para>
/// <para>
/// - <b>Explicit fitness sharing</b>: adjusted fitness is fitness divided by species size, and each
/// species gets offspring in proportion to its summed adjusted fitness. Species that have not improved in
/// 15 generations do not reproduce; after 20 generations without population-wide improvement only the
/// top two species do. The lowest-performing members are eliminated before reproduction, and the
/// champion of every species with more than five members is copied unchanged.
/// </para>
/// <para>
/// - <b>Mutation</b>: weights are mutated with probability 0.8 (each perturbed with probability 0.9,
/// otherwise replaced); add-node (0.03) splits a connection; add-connection (0.05) links two unconnected
/// nodes without creating a cycle.
/// </para>
/// <para>
/// - <b>Activation</b>: the steepened sigmoid <c>1 / (1 + e^(-4.9 x))</c> on every hidden and output
/// node, evaluated layer by layer so hidden activations propagate after squashing.
/// </para>
/// <para><b>For Beginners:</b> NEAT is a way to grow neural networks through evolution rather than
/// training them with fixed structures. It starts with simple networks, and the networks that do better
/// on your task are more likely to pass on their "genes". Grouping similar networks into species gives
/// new structures time to improve before they have to compete with everything else.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 4 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new NEAT&lt;float&gt;())
///     .Build(trainX, trainY);
/// var output = result.Predict(input);
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
public partial class NEAT<T> : VectorModelLayoutBase<T>
{
    // Static numeric operations for the static helpers (the base class's NumOps is an instance field).
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly NEATOptions _options;

    /// <inheritdoc />
    /// <remarks>
    /// Every hidden and output node is passed through the steepened sigmoid, so supervised fitness
    /// targets must lie in the unit interval. NEAT has no ordinary final layer from which the base can
    /// discover this domain.
    /// </remarks>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) =>
        UnitIntervalTensorDomain.Value;

    // The current population of genomes.
    private List<Genome<T>> _population;

    // Species of the current population (members are the genomes of the last evaluated generation).
    private readonly List<Species> _species = new();

    // Best genome of the current fitness function: what Predict and the parameter surface use.
    private Genome<T>? _champion;

    private readonly int _populationSize;

    /// <summary>
    /// Per-instance deterministic RNG for the evolutionary search. Seeded from the architecture's
    /// RandomSeed when present; otherwise the shared thread-safe RNG.
    /// </summary>
    private readonly Random _rng;

    // Global historical markings.
    private int _innovationNumber;
    private int _nextNodeId;

    // Innovations of the current generation (paper Section 3.2): the same split / link gets the same
    // node id and innovation numbers in every genome that makes it during one generation.
    private readonly Dictionary<int, (int NodeId, int InInnovation, int OutInnovation)> _splitInnovations = new();
    private readonly Dictionary<(int From, int To), int> _linkInnovations = new();

    private int _generation;
    private int _nextSpeciesId;
    private int _populationLastImproved;
    private T _populationBestFitness;
    private bool _hasPopulationBest;

    /// <summary>The current population (test access).</summary>
    internal IReadOnlyList<Genome<T>> Population => _population;

    /// <summary>Number of species after the last speciation.</summary>
    internal int SpeciesCount => _species.Count;

    private const int DefaultInputSize = 10;
    private const int DefaultOutputSize = 1;
    private const int DefaultPopulationSize = 150;

    private sealed class Species
    {
        public Species(int id, Genome<T> representative, int generation)
        {
            Id = id;
            Representative = representative;
            BestFitness = representative.Fitness;
            LastImproved = generation;
        }

        public int Id { get; }

        public Genome<T> Representative { get; set; }

        public List<Genome<T>> Members { get; } = new();

        public T BestFitness { get; set; }

        public bool HasBest { get; set; }

        public int LastImproved { get; set; }
    }

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
    /// <param name="populationSize">The number of individual genomes in the population (the paper uses 150).</param>
    /// <param name="lossFunction">Loss used to turn supervised error into fitness.</param>
    /// <param name="options">Evolution parameters; defaults are the paper's.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new NEAT system with your chosen settings.</para>
    /// </remarks>
    public NEAT(NeuralNetworkArchitecture<T> architecture,
        int populationSize,
        ILossFunction<T>? lossFunction = null,
        NEATOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        options ??= new NEATOptions();
        options.Validate();
        _options = options;
        Options = _options;
        _populationSize = populationSize;
        _populationBestFitness = NumOps.Zero;
        _rng = Architecture.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(Architecture.RandomSeed.Value)
            : RandomHelper.ThreadSafeRandom;
        _nextNodeId = BiasNodeId + 1;
        _population = InitializePopulation();
    }

    private int BiasNodeId => Architecture.InputSize + Architecture.OutputSize;

    /// <summary>
    /// Initializes the layers of the neural network. NEAT evolves its structure, so there are no fixed layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // NEAT doesn't use fixed layers.
    }

    /// <summary>
    /// NEAT's parameters are the connection weights of the best genome.
    /// </summary>
    /// <remarks>
    /// A computed surface: the count legitimately changes as the topology evolves, and the count and the
    /// vector always agree because both read the same genome.
    /// </remarks>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(new DelegatingParameterSource<T>(
            () => GetBestGenome()?.Connections?.Count ?? 0,
            () => new Vector<T>(GetBestGenome()?.Connections.Select(c => c.Weight) ?? Enumerable.Empty<T>()),
            values =>
            {
                var connections = GetBestGenome()?.Connections;
                if (connections is null) return;
                var weights = values.ToArray();
                int n = Math.Min(connections.Count, weights.Length);
                for (int i = 0; i < n; i++) connections[i].Weight = weights[i];
            }));
    }

    /// <summary>
    /// NEAT evolves connection weights and does not compute gradient-descent derivatives.
    /// </summary>
    public override Vector<T> GetParameterGradients()
        => throw new NotSupportedException(
            "NEAT is an evolutionary optimizer and does not expose parameter gradients.");

    /// <summary>
    /// Creates the initial population: every genome is the same minimal network (inputs and bias fully
    /// connected to the outputs) with the same historical markings and independent random weights.
    /// </summary>
    private List<Genome<T>> InitializePopulation()
    {
        int inputs = Architecture.InputSize;
        int outputs = Architecture.OutputSize;
        var sources = Enumerable.Range(0, inputs).Concat(new[] { BiasNodeId }).ToArray();

        // One shared innovation number per initial link, so crossover can line the genomes up.
        var links = sources.SelectMany(s => Enumerable.Range(0, outputs).Select(o => (From: s, To: inputs + o))).ToArray();
        int firstInnovation = _innovationNumber;
        _innovationNumber += links.Length;

        var population = new List<Genome<T>>(_populationSize);
        for (int p = 0; p < _populationSize; p++)
        {
            var weights = RandomWeights(links.Length);
            var genome = new Genome<T>(inputs, outputs);
            for (int i = 0; i < links.Length; i++)
            {
                genome.AddConnection(links[i].From, links[i].To, weights[i], true, firstInnovation + i);
            }

            population.Add(genome);
        }

        return population;
    }

    /// <summary>Uniform random weights in [-1, 1), drawn by the engine with one seed from the RNG.</summary>
    private T[] RandomWeights(int count)
    {
        if (count == 0) return Array.Empty<T>();
        return Engine.TensorRandomUniformRange(new[] { count }, NumOps.FromDouble(-1.0), NumOps.One, _rng.Next()).ToArray();
    }

    /// <summary>
    /// Evolves the population over a specified number of generations using the provided fitness function.
    /// </summary>
    /// <param name="fitnessFunction">A function that evaluates the fitness of each genome (higher is better).</param>
    /// <param name="generations">The number of generations to evolve.</param>
    /// <remarks>
    /// Each generation evaluates every genome, speciates the population, allocates offspring by explicit
    /// fitness sharing, and breeds the next generation within species.
    /// </remarks>
    public void EvolvePopulation(Func<Genome<T>, T> fitnessFunction, int generations)
    {
        if (_population.Count == 0) return;
        SyncHistoricalMarkings();

        // The champion is re-scored under this fitness function so it only survives on merit.
        if (_champion is not null)
        {
            _champion.Fitness = fitnessFunction(_champion);
        }

        for (int gen = 0; gen < generations; gen++)
        {
            _generation++;
            _splitInnovations.Clear();
            _linkInnovations.Clear();

            foreach (var genome in _population)
            {
                genome.Fitness = fitnessFunction(genome);
            }

            var fitness = FitnessTensor(_population);
            UpdateChampion(fitness);
            Speciate();
            var offspring = AllocateOffspring();
            _population = Reproduce(offspring);
        }
    }

    /// <summary>Makes the global counters exceed every node id and innovation number in use.</summary>
    private void SyncHistoricalMarkings()
    {
        var all = _population.SelectMany(g => g.Connections).ToList();
        if (all.Count == 0) return;
        _innovationNumber = Math.Max(_innovationNumber, all.Max(c => c.Innovation) + 1);
        _nextNodeId = Math.Max(_nextNodeId, all.Max(c => Math.Max(c.FromNode, c.ToNode)) + 1);
    }

    private static Tensor<T> FitnessTensor(IReadOnlyList<Genome<T>> genomes)
        => new Tensor<T>(genomes.Select(g => g.Fitness).ToArray(), new[] { genomes.Count });

    /// <summary>Keeps a copy of the fittest genome evaluated so far under the current fitness function.</summary>
    private void UpdateChampion(Tensor<T> fitness)
    {
        int best = Engine.TensorArgMax(fitness, 0).GetFlat(0);
        var candidate = _population[best];
        bool populationImproved = !_hasPopulationBest || NumOps.GreaterThan(candidate.Fitness, _populationBestFitness);
        if (populationImproved)
        {
            _populationBestFitness = candidate.Fitness;
            _hasPopulationBest = true;
            _populationLastImproved = _generation;
        }

        if (_champion is null || NumOps.GreaterThan(candidate.Fitness, _champion.Fitness))
        {
            var copy = candidate.Clone();
            copy.Fitness = candidate.Fitness;
            _champion = copy;
        }
    }

    /// <summary>
    /// Compatibility distance <c>delta = c1 E / N + c2 D / N + c3 W</c>: E excess genes, D disjoint genes,
    /// W the mean absolute weight difference of matching genes (disabled genes included), N the gene
    /// count of the larger genome, or 1 when both are smaller than the normalization threshold.
    /// </summary>
    internal T CompatibilityDistance(Genome<T> a, Genome<T> b)
    {
        var genesA = ByInnovation(a);
        var genesB = ByInnovation(b);
        int maxA = genesA.Count == 0 ? -1 : genesA.Keys.Max();
        int maxB = genesB.Count == 0 ? -1 : genesB.Keys.Max();

        var matching = genesA.Keys.Where(genesB.ContainsKey).OrderBy(k => k).ToArray();
        var onlyA = genesA.Keys.Where(k => !genesB.ContainsKey(k)).ToArray();
        var onlyB = genesB.Keys.Where(k => !genesA.ContainsKey(k)).ToArray();
        int excess = onlyA.Count(k => k > maxB) + onlyB.Count(k => k > maxA);
        int disjoint = onlyA.Length + onlyB.Length - excess;

        int threshold = _options.GenomeSizeNormalizationThreshold;
        int larger = Math.Max(genesA.Count, genesB.Count);
        double n = genesA.Count < threshold && genesB.Count < threshold ? 1.0 : Math.Max(1, larger);

        T structural = NumOps.FromDouble(
            (_options.ExcessCoefficient * excess + _options.DisjointCoefficient * disjoint) / n);
        if (matching.Length == 0) return structural;

        var weightsA = new Tensor<T>(matching.Select(k => genesA[k].Weight).ToArray(), new[] { matching.Length });
        var weightsB = new Tensor<T>(matching.Select(k => genesB[k].Weight).ToArray(), new[] { matching.Length });
        T meanDifference = Engine.TensorMean(Engine.TensorAbs(Engine.TensorSubtract(weightsA, weightsB)));
        return NumOps.Add(structural, NumOps.Multiply(NumOps.FromDouble(_options.WeightDifferenceCoefficient), meanDifference));
    }

    private static Dictionary<int, Connection<T>> ByInnovation(Genome<T> genome)
        => genome.Connections.GroupBy(c => c.Innovation).ToDictionary(g => g.Key, g => g.First());

    private bool IsCompatible(Genome<T> a, Genome<T> b)
        => NumOps.LessThan(CompatibilityDistance(a, b), NumOps.FromDouble(_options.CompatibilityThreshold));

    /// <summary>
    /// Places every genome in the first species whose representative (a random member of that species
    /// in the previous generation) is within the compatibility threshold, founding a new species otherwise.
    /// </summary>
    private void Speciate()
    {
        foreach (var species in _species)
        {
            if (species.Members.Count > 0)
            {
                species.Representative = species.Members[_rng.Next(species.Members.Count)];
            }

            species.Members.Clear();
        }

        foreach (var genome in _population)
        {
            var home = _species.FirstOrDefault(s => IsCompatible(genome, s.Representative));
            if (home is null)
            {
                home = new Species(_nextSpeciesId++, genome, _generation);
                _species.Add(home);
            }

            home.Members.Add(genome);
        }

        _species.RemoveAll(s => s.Members.Count == 0);

        foreach (var species in _species)
        {
            UpdateSpeciesBest(species, Engine.TensorMaxValue(FitnessTensor(species.Members)));
        }
    }

    private void UpdateSpeciesBest(Species species, T best)
    {
        if (!species.HasBest || NumOps.GreaterThan(best, species.BestFitness))
        {
            species.BestFitness = best;
            species.HasBest = true;
            species.LastImproved = _generation;
        }
    }

    /// <summary>
    /// Offspring quota of every current species, from explicit fitness sharing. Species that have not
    /// improved in <see cref="NEATOptions.SpeciesStagnationLimit"/> generations get none; after
    /// <see cref="NEATOptions.PopulationStagnationLimit"/> generations without population-wide improvement
    /// only the top two species reproduce.
    /// </summary>
    private int[] AllocateOffspring()
    {
        var fitness = _species.Select(sp => FitnessTensor(sp.Members)).ToList();
        var eligible = _species
            .Select(sp => _generation - sp.LastImproved < _options.SpeciesStagnationLimit)
            .ToArray();
        if (_generation - _populationLastImproved >= _options.PopulationStagnationLimit && _species.Count > 2)
        {
            var means = Engine.TensorConcatenate(fitness
                .Select(f => Engine.ReduceMean(f, new[] { 0 }, true)).ToArray(), 0);
            var top = Engine.TensorArgsort(means, 0, true).ToArray().Take(2).ToArray();
            eligible = Enumerable.Range(0, _species.Count).Select(i => top.Contains(i)).ToArray();
        }

        return ShareOffspring(fitness, eligible, _populationSize);
    }

    /// <summary>
    /// Explicit fitness sharing (paper Section 3.3): a member's adjusted fitness is f_i / |S|, and each
    /// eligible species receives offspring in proportion to the sum of its adjusted fitness, which is its
    /// mean fitness. Fitness is shifted to be non-negative first. Rounding remainders go to the largest
    /// fractional shares; if no species is eligible, the one with the best mean keeps reproducing.
    /// </summary>
    internal static int[] ShareOffspring(IReadOnlyList<Tensor<T>> speciesFitness, bool[] eligible, int populationSize)
    {
        var engine = AiDotNetEngine.Current;
        int s = speciesFitness.Count;
        var means = engine.TensorConcatenate(speciesFitness
            .Select(f => engine.ReduceMean(f, new[] { 0 }, true)).ToArray(), 0);
        var all = engine.TensorConcatenate(speciesFitness.ToArray(), 0);
        var adjusted = engine.TensorSubtractScalar(means, NonPositivePart(engine.TensorMinValue(all)));

        eligible = (bool[])eligible.Clone();
        if (!eligible.Any(e => e))
        {
            // Never let the whole population go extinct: the best species keeps reproducing.
            eligible[engine.TensorArgMax(adjusted, 0).GetFlat(0)] = true;
        }

        T one = Ops.One;
        T zero = Ops.Zero;
        var mask = new Tensor<T>(eligible.Select(e => e ? one : zero).ToArray(), new[] { s });
        var weighted = engine.TensorMultiply(adjusted, mask);
        T total = engine.TensorSum(weighted);
        if (!Ops.GreaterThan(total, zero))
        {
            weighted = mask;
            total = engine.TensorSum(mask);
        }

        var share = engine.TensorMultiplyScalar(engine.TensorDivideScalar(weighted, total), Ops.FromDouble(populationSize));
        var floor = engine.TensorFloor(share);
        var counts = floor.Cast<int>().ToArray();
        int remainder = populationSize - counts.Sum();
        if (remainder > 0)
        {
            var fraction = engine.TensorWhere(mask, engine.TensorSubtract(share, floor), engine.TensorSubtractScalar(mask, one));
            var order = engine.TensorArgsort(fraction, 0, true).ToArray();
            int eligibleCount = eligible.Count(e => e);
            for (int r = 0; r < remainder; r++) counts[order[r % eligibleCount]]++;
        }

        return counts;
    }

    // min(x, 0): the amount by which fitness must be shifted up to be non-negative (as a subtrahend).
    private static T NonPositivePart(T value) => Ops.LessThan(value, Ops.Zero) ? value : Ops.Zero;

    /// <summary>
    /// Breeds the next generation species by species: the champion of a species with more than
    /// <see cref="NEATOptions.ChampionSpeciesSize"/> members is copied unchanged, the lowest performers
    /// are eliminated, and the rest of the species' offspring come from crossover (with an occasional
    /// interspecies mate) or from cloning, each followed by mutation.
    /// </summary>
    private List<Genome<T>> Reproduce(int[] offspring)
    {
        var next = new List<Genome<T>>(_populationSize);
        for (int s = 0; s < _species.Count; s++)
        {
            int quota = offspring[s];
            if (quota <= 0) continue;

            var species = _species[s];
            var order = Engine.TensorArgsort(FitnessTensor(species.Members), 0, true).ToArray();
            var ranked = order.Select(i => species.Members[i]).ToList();

            if (species.Members.Count > _options.ChampionSpeciesSize)
            {
                next.Add(ranked[0]);
                quota--;
            }

            int survivors = Math.Max(1, (int)Math.Ceiling(_options.SurvivalThreshold * ranked.Count));
            var parents = ranked.Take(survivors).ToList();

            for (int k = 0; k < quota; k++)
            {
                Genome<T> child;
                if (_rng.NextDouble() < _options.CrossoverRate)
                {
                    var mother = parents[_rng.Next(parents.Count)];
                    var father = _rng.NextDouble() < _options.InterspeciesMatingRate && _species.Count > 1
                        ? RandomOtherSpeciesMember(s)
                        : parents[_rng.Next(parents.Count)];
                    child = Crossover(mother, father);
                }
                else
                {
                    child = parents[_rng.Next(parents.Count)].Clone();
                }

                Mutate(child);
                next.Add(child);
            }
        }

        while (next.Count < _populationSize && _champion is not null)
        {
            var filler = _champion.Clone();
            Mutate(filler);
            next.Add(filler);
        }

        if (next.Count > _populationSize) next.RemoveRange(_populationSize, next.Count - _populationSize);
        return next;
    }

    private Genome<T> RandomOtherSpeciesMember(int speciesIndex)
    {
        int other = _rng.Next(_species.Count - 1);
        if (other >= speciesIndex) other++;
        var members = _species[other].Members;
        return members[_rng.Next(members.Count)];
    }

    /// <summary>1 when a is fitter, -1 when b is fitter, 0 when equal.</summary>
    private static int CompareFitness(Genome<T> a, Genome<T> b)
        => Ops.GreaterThan(a.Fitness, b.Fitness) ? 1 : Ops.GreaterThan(b.Fitness, a.Fitness) ? -1 : 0;

    /// <summary>
    /// Crossover by historical markings: matching genes are inherited from either parent at random;
    /// disjoint and excess genes come from the fitter parent, or from both when fitness is equal. A gene
    /// disabled in either parent is disabled in the child with probability
    /// <see cref="NEATOptions.DisableInheritedRate"/>.
    /// </summary>
    internal Genome<T> Crossover(Genome<T> parent1, Genome<T> parent2)
    {
        int comparison = CompareFitness(parent1, parent2);
        var fitter = comparison >= 0 ? parent1 : parent2;
        var other = comparison >= 0 ? parent2 : parent1;
        var otherGenes = ByInnovation(other);
        var child = new Genome<T>(Architecture.InputSize, Architecture.OutputSize);

        foreach (var gene in fitter.Connections)
        {
            if (otherGenes.TryGetValue(gene.Innovation, out var match))
            {
                var chosen = _rng.NextDouble() < 0.5 ? gene : match;
                child.AddConnection(chosen.FromNode, chosen.ToNode, chosen.Weight,
                    InheritEnabled(gene.IsEnabled && match.IsEnabled), gene.Innovation);
            }
            else
            {
                child.AddConnection(gene.FromNode, gene.ToNode, gene.Weight, InheritEnabled(gene.IsEnabled), gene.Innovation);
            }
        }

        if (comparison == 0)
        {
            var fitterInnovations = new HashSet<int>(fitter.Connections.Select(c => c.Innovation));
            foreach (var gene in other.Connections.Where(c => !fitterInnovations.Contains(c.Innovation)))
            {
                child.AddConnection(gene.FromNode, gene.ToNode, gene.Weight, InheritEnabled(gene.IsEnabled), gene.Innovation);
            }
        }

        return child;
    }

    private bool InheritEnabled(bool enabledInParents)
        => enabledInParents || _rng.NextDouble() >= _options.DisableInheritedRate;

    /// <summary>
    /// Mutation as in the paper: add-node, add-connection and weight mutation, each with its own
    /// probability.
    /// </summary>
    private void Mutate(Genome<T> genome)
    {
        if (_rng.NextDouble() < _options.AddNodeRate) AddNodeMutation(genome);
        if (_rng.NextDouble() < _options.AddConnectionRate) AddConnectionMutation(genome);
        if (_rng.NextDouble() < _options.WeightMutationRate) MutateWeights(genome);
    }

    /// <summary>
    /// Splits a random enabled connection: the old link is disabled, the link into the new node gets
    /// weight 1 and the link out of it the old weight. The same split in one generation reuses the same
    /// node id and innovation numbers.
    /// </summary>
    private void AddNodeMutation(Genome<T> genome)
    {
        var enabled = genome.Connections.Where(c => c.IsEnabled).ToList();
        if (enabled.Count == 0) return;

        var connection = enabled[_rng.Next(enabled.Count)];
        if (!_splitInnovations.TryGetValue(connection.Innovation, out var split))
        {
            split = (_nextNodeId++, _innovationNumber++, _innovationNumber++);
            _splitInnovations[connection.Innovation] = split;
        }

        if (genome.Connections.Any(c => c.FromNode == split.NodeId || c.ToNode == split.NodeId)) return;

        genome.DisableConnection(connection.Innovation);
        genome.AddConnection(connection.FromNode, split.NodeId, NumOps.One, true, split.InInnovation);
        genome.AddConnection(split.NodeId, connection.ToNode, connection.Weight, true, split.OutInnovation);
    }

    /// <summary>
    /// Links two previously unconnected nodes (source: any node; target: a hidden or output node) without
    /// creating a cycle. The same new link in one generation reuses one innovation number.
    /// </summary>
    private void AddConnectionMutation(Genome<T> genome)
    {
        int inputs = Architecture.InputSize;
        var hidden = genome.Connections.SelectMany(c => new[] { c.FromNode, c.ToNode })
            .Where(id => id > BiasNodeId).Distinct().ToList();
        var targets = Enumerable.Range(inputs, Architecture.OutputSize).Concat(hidden).ToArray();
        var sources = Enumerable.Range(0, BiasNodeId + 1).Concat(hidden).ToArray();
        var existing = new HashSet<(int, int)>(genome.Connections.Select(c => (c.FromNode, c.ToNode)));

        for (int attempt = 0; attempt < _options.AddConnectionTries; attempt++)
        {
            int from = sources[_rng.Next(sources.Length)];
            int to = targets[_rng.Next(targets.Length)];
            bool outputToOutput = from >= inputs && from < BiasNodeId && to < BiasNodeId;
            if (from == to || outputToOutput || existing.Contains((from, to)) || Reaches(genome, to, from)) continue;

            if (!_linkInnovations.TryGetValue((from, to), out int innovation))
            {
                innovation = _innovationNumber++;
                _linkInnovations[(from, to)] = innovation;
            }

            genome.AddConnection(from, to, RandomWeights(1)[0], true, innovation);
            return;
        }
    }

    /// <summary>Whether <paramref name="target"/> is reachable from <paramref name="start"/> along any connection.</summary>
    private static bool Reaches(Genome<T> genome, int start, int target)
    {
        var outgoing = genome.Connections.ToLookup(c => c.FromNode, c => c.ToNode);
        var visited = new HashSet<int>();
        var stack = new Stack<int>();
        stack.Push(start);
        while (stack.Count > 0)
        {
            int node = stack.Pop();
            if (node == target) return true;
            if (!visited.Add(node)) continue;
            foreach (int next in outgoing[node]) stack.Push(next);
        }

        return false;
    }

    /// <summary>
    /// Weight mutation over the whole genome at once: each weight is replaced by a new uniform value with
    /// probability <see cref="NEATOptions.WeightReplaceRate"/>, otherwise perturbed uniformly by up to
    /// <see cref="NEATOptions.WeightPerturbPower"/>, then clamped to the weight cap.
    /// </summary>
    private void MutateWeights(Genome<T> genome)
    {
        var connections = genome.Connections;
        int m = connections.Count;
        if (m == 0) return;

        var shape = new[] { m };
        T power = NumOps.FromDouble(_options.WeightPerturbPower);
        T cap = NumOps.FromDouble(_options.WeightCap);
        var weights = new Tensor<T>(connections.Select(c => c.Weight).ToArray(), shape);
        var draw = Engine.TensorRandomUniformRange(shape, NumOps.Zero, NumOps.One, _rng.Next());
        var perturbation = Engine.TensorRandomUniformRange(shape, NumOps.Negate(power), power, _rng.Next());
        var fresh = Engine.TensorRandomUniformRange(shape, NumOps.FromDouble(-1.0), NumOps.One, _rng.Next());
        var replace = Engine.TensorLessThan(draw, NumOps.FromDouble(_options.WeightReplaceRate));
        var mutated = Engine.TensorClamp(
            Engine.TensorWhere(replace, fresh, Engine.TensorAdd(weights, perturbation)),
            NumOps.Negate(cap), cap);

        var values = mutated.ToArray();
        for (int i = 0; i < m; i++) connections[i].Weight = values[i];
    }

    /// <summary>
    /// Predicts output values for input data using the best genome.
    /// </summary>
    /// <param name="input">A rank-1 [features] or rank-2 [batch, features] tensor.</param>
    /// <returns>The output activations, with the same rank as the input.</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var bestGenome = RequireBestGenome();

        if (input.Shape.Length < 1 || input.Shape.Length > 2)
        {
            throw new ArgumentException(
                $"NEAT.Predict expects rank-1 [features] or rank-2 [batch, features]; " +
                $"got rank {input.Shape.Length} (shape [{string.Join(",", input.Shape)}]).",
                nameof(input));
        }

        bool isBatch = input.Shape.Length == 2;
        var batch = isBatch ? input : input.Reshape(new[] { 1, input.Length });
        var outputs = Outputs(ActivateGenome(bestGenome, batch));
        return isBatch ? outputs : outputs.Reshape(new[] { Architecture.OutputSize });
    }

    /// <summary>The best genome, for paths that cannot proceed without one.</summary>
    private Genome<T> RequireBestGenome()
        => GetBestGenome()
            ?? throw new InvalidOperationException(
                "NEAT has no genomes yet. Evolve a population (Train) before predicting or "
                    + "reading genome metadata.");

    /// <summary>
    /// The champion (fittest genome evaluated under the current fitness function); before any
    /// evolution, the first genome of the population. Null only when the population is empty.
    /// </summary>
    private Genome<T>? GetBestGenome()
    {
        if (_champion is not null) return _champion;
        return _population.Count == 0 ? null : _population[0];
    }

    /// <summary>
    /// Activates a genome's network for a whole batch: returns the [batch, nodes] activation matrix,
    /// indexed by node id (inputs, outputs, bias = 1, hidden nodes).
    /// </summary>
    /// <remarks>
    /// Nodes are grouped into layers by their longest path from the inputs over enabled connections.
    /// Each layer is one matrix product with the layer's incoming weights followed by the steepened
    /// sigmoid, so every hidden activation is squashed before it feeds the next layer.
    /// </remarks>
    internal Tensor<T> ActivateGenome(Genome<T> genome, Tensor<T> inputs)
    {
        int batch = inputs.Shape[0];
        int features = inputs.Shape[1];
        int inputSize = Architecture.InputSize;
        int outputSize = Architecture.OutputSize;
        if (features < inputSize)
        {
            throw new ArgumentException(
                $"NEAT expects {inputSize} input features but the input has {features}.", nameof(inputs));
        }

        var enabled = genome.Connections.Where(c => c.IsEnabled).ToList();
        int nodes = Math.Max(BiasNodeId, enabled.Count == 0 ? 0 : enabled.Max(c => Math.Max(c.FromNode, c.ToNode))) + 1;
        var depth = NodeDepths(enabled, inputSize, outputSize);

        // Initial activations: inputs, zeroed outputs, the constant bias, zeroed hidden slots.
        var x = features == inputSize ? inputs : Engine.TensorSlice(inputs, new[] { 0, 0 }, new[] { batch, inputSize });
        var parts = new List<Tensor<T>> { x };
        if (outputSize > 0) parts.Add(new Tensor<T>(new[] { batch, outputSize }));
        var bias = new Tensor<T>(new[] { batch, 1 });
        Engine.TensorFill(bias, NumOps.One);
        parts.Add(bias);
        if (nodes > BiasNodeId + 1) parts.Add(new Tensor<T>(new[] { batch, nodes - BiasNodeId - 1 }));
        var activations = Engine.TensorConcatenate(parts.ToArray(), 1);

        T slope = NumOps.FromDouble(_options.SigmoidSlope);
        foreach (var layer in depth.Where(kv => kv.Value > 0).GroupBy(kv => kv.Value).OrderBy(g => g.Key))
        {
            var members = layer.Select(kv => kv.Key).OrderBy(id => id).ToArray();
            var position = members.Select((id, i) => (id, i)).ToDictionary(p => p.id, p => p.i);
            var incoming = enabled
                .Where(c => position.ContainsKey(c.ToNode) && depth.TryGetValue(c.FromNode, out int d) && d < layer.Key)
                .ToList();

            var weights = new Tensor<T>(new[] { nodes * members.Length });
            if (incoming.Count > 0)
            {
                var cells = new Tensor<int>(incoming.Select(c => c.FromNode * members.Length + position[c.ToNode]).ToArray(), new[] { incoming.Count });
                var values = new Tensor<T>(incoming.Select(c => c.Weight).ToArray(), new[] { incoming.Count });
                weights = Engine.TensorIndexAdd(weights, 0, cells, values);
            }

            var z = Engine.TensorMatMul(activations, weights.Reshape(new[] { nodes, members.Length }));
            var squashed = Engine.Sigmoid(Engine.TensorMultiplyScalar(z, slope));
            var placement = Engine.TensorOneHot<T>(new Tensor<int>(members, new[] { members.Length }), nodes);
            activations = Engine.TensorAdd(activations, Engine.TensorMatMul(squashed, placement));
        }

        return activations;
    }

    /// <summary>
    /// Longest-path depth of every node: inputs and bias 0, other nodes 1 + the deepest source. Output
    /// nodes are always present. A connection that would close a cycle is ignored.
    /// </summary>
    private Dictionary<int, int> NodeDepths(List<Connection<T>> enabled, int inputSize, int outputSize)
    {
        var sourcesOf = enabled.ToLookup(c => c.ToNode, c => c.FromNode);
        var depth = new Dictionary<int, int>();
        var onStack = new HashSet<int>();

        int Depth(int node)
        {
            if (node < inputSize || node == BiasNodeId) return 0;
            if (depth.TryGetValue(node, out int known)) return known;
            onStack.Add(node);
            int deepest = 0;
            foreach (int source in sourcesOf[node])
            {
                if (onStack.Contains(source)) continue;
                deepest = Math.Max(deepest, Depth(source));
            }

            onStack.Remove(node);
            depth[node] = deepest + 1;
            return deepest + 1;
        }

        for (int i = 0; i < inputSize; i++) depth[i] = 0;
        depth[BiasNodeId] = 0;
        foreach (int node in Enumerable.Range(inputSize, outputSize)
                     .Concat(enabled.SelectMany(c => new[] { c.FromNode, c.ToNode })).Distinct().ToList())
        {
            Depth(node);
        }

        return depth;
    }

    /// <summary>The output-node columns of an activation matrix, [batch, outputs].</summary>
    private Tensor<T> Outputs(Tensor<T> activations)
        => Engine.TensorSlice(activations, new[] { 0, Architecture.InputSize },
            new[] { activations.Shape[0], Architecture.OutputSize });

    /// <summary>
    /// Trains the NEAT system on supervised data: fitness is <c>1 / (loss + 0.01)</c> of the genome's
    /// predictions, and the population evolves for <see cref="NEATOptions.GenerationsPerTrain"/> generations.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input.Shape[0] != expectedOutput.Shape[0])
        {
            throw new ArgumentException("Input and expected output must have the same batch size");
        }

        var target = expectedOutput.ToVector();
        T epsilon = NumOps.FromDouble(0.01);

        T Loss(Genome<T> genome) => LossFunction.CalculateLoss(Outputs(ActivateGenome(genome, input)).ToVector(), target);
        T Fitness(Genome<T> genome) => NumOps.Divide(NumOps.One, NumOps.Add(Loss(genome), epsilon));

        EvolvePopulation(Fitness, _options.GenerationsPerTrain);

        var champion = RequireBestGenome();
        if (champion.Connections.Count > 0 && input.Shape[0] > 0)
        {
            LastLoss = Loss(champion);
        }

        return true;
    }

    /// <summary>
    /// Gets metadata about the NEAT model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var bestGenome = RequireBestGenome();

        double avgConnections = _population.Average(g => g.Connections.Count);
        int maxConnections = _population.Max(g => g.Connections.Count);
        var nodeCounts = _population.Select(g =>
            g.Connections.Any() ?
                g.Connections.Max(c => Math.Max(c.FromNode, c.ToNode)) + 1 :
                Architecture.InputSize + Architecture.OutputSize).ToList();

        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "PopulationSize", _populationSize },
                { "CrossoverRate", _options.CrossoverRate },
                { "WeightMutationRate", _options.WeightMutationRate },
                { "AddNodeRate", _options.AddNodeRate },
                { "AddConnectionRate", _options.AddConnectionRate },
                { "CompatibilityThreshold", _options.CompatibilityThreshold },
                { "SpeciesCount", _species.Count },
                { "Generation", _generation },
                { "InnovationNumber", _innovationNumber },
                { "AverageConnections", avgConnections },
                { "MaxConnections", maxConnections },
                { "AverageNodes", nodeCounts.Average() },
                { "MaxNodes", nodeCounts.Max() },
                { "BestGenomeFitness", Convert.ToDouble(bestGenome.Fitness) },
                { "BestGenomeConnections", bestGenome.Connections.Count },
                { "BestGenomeEnabledConnections", bestGenome.Connections.Count(c => c.IsEnabled) }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <summary>
    /// Gets named activations from the best genome's network when processing input.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var bestGenome = RequireBestGenome();
        var matrix = ActivateGenome(bestGenome, input.Reshape(new[] { 1, input.Length }));
        int nodes = matrix.Shape[1];
        var activations = matrix.Reshape(new[] { nodes });
        int inputSize = Architecture.InputSize;
        int outputSize = Architecture.OutputSize;

        var result = new Dictionary<string, Tensor<T>>
        {
            ["InputNodes"] = Engine.TensorSlice(activations, new[] { 0 }, new[] { inputSize }),
            ["OutputNodes"] = Engine.TensorSlice(activations, new[] { inputSize }, new[] { outputSize })
        };

        var hidden = bestGenome.Connections.Where(c => c.IsEnabled)
            .SelectMany(c => new[] { c.FromNode, c.ToNode })
            .Where(id => id > BiasNodeId && id < nodes)
            .Distinct().OrderBy(id => id).ToArray();
        if (hidden.Length > 0)
        {
            result["HiddenNodes"] = Engine.TensorIndexSelect(activations, new Tensor<int>(hidden, new[] { hidden.Length }), 0);
        }

        return result;
    }

    /// <summary>
    /// Checks if the NEAT model is ready to make predictions.
    /// </summary>
    /// <returns>True if the model has at least one genome with connections; otherwise, false.</returns>
    public bool IsReadyToPredict()
    {
        return _population != null &&
               _population.Count > 0 &&
               _population.Any(g => g.Connections.Count > 0);
    }
}
