namespace AiDotNet.Genetics;

/// <summary>
/// Implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II) for multi-objective optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
/// <remarks>
/// <para>
/// NSGA-II is an advanced genetic algorithm designed for problems with multiple competing objectives.
/// It extends the standard genetic algorithm by incorporating Pareto-based ranking and crowding distance
/// to maintain diversity in the population while optimizing for multiple objectives simultaneously.
/// </para>
/// <para><b>For Beginners:</b> Think of NSGA-II like a home-buying agent that helps you find the best house.
/// 
/// In regular optimization, you might just look for the cheapest house. But in real life, you care about:
/// - Price (lower is better)
/// - Location (closer to work is better)
/// - Size (larger is better)
/// - Quality (higher is better)
/// 
/// These goals conflict - a bigger house usually costs more, a better location often means smaller size, etc.
/// 
/// NSGA-II is like a smart agent that:
/// 1. Finds a variety of houses representing different trade-offs (the "Pareto front")
/// 2. Organizes houses into tiers based on how good the trade-offs are
/// 3. Makes sure you have diverse options within each tier (not just 10 similar houses)
/// 4. Gradually improves the selection over time by combining features from promising houses
/// 
/// Instead of one "best" solution, it provides a range of optimal trade-offs for you to choose from.
/// </para>
/// </remarks>
public class NSGAII<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    /// <summary>
    /// The storage for individual ranks based on non-dominated sorting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This dictionary stores the Pareto dominance rank of each individual in the population.
    /// Individuals with rank 0 are in the first Pareto front (non-dominated solutions),
    /// rank 1 are in the second front, and so on.
    /// </para>
    /// <para><b>For Beginners:</b> This is like assigning each house to a specific tier.
    /// 
    /// - Rank 0: "Gold tier" houses that aren't clearly worse than any other house in all aspects
    /// - Rank 1: "Silver tier" houses that are only dominated by gold tier houses
    /// - Rank 2: "Bronze tier" houses that are dominated by gold and silver tier houses
    /// 
    /// Lower rank (tier) houses are preferred during selection because they represent better trade-offs.
    /// </para>
    /// </remarks>
    private readonly Dictionary<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, int> _ranks = [];

    /// <summary>
    /// The storage for individual crowding distances within their Pareto fronts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This dictionary stores the crowding distance of each individual, which measures how
    /// isolated the individual is in its objective space. Higher values indicate more isolated
    /// individuals, which are prioritized to maintain diversity.
    /// </para>
    /// <para><b>For Beginners:</b> This is like measuring how unique each house is compared to similar ones.
    /// 
    /// - Houses with unusual combinations of features get higher scores
    /// - Houses that are very similar to many others get lower scores
    /// - When choosing between houses in the same tier, we prefer unique ones
    /// 
    /// This ensures you get a diverse selection of options rather than many similar alternatives.
    /// </para>
    /// </remarks>
    private readonly Dictionary<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, double> _crowdingDistances = [];

    /// <summary>
    /// The list of fitness calculators, one for each optimization objective.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list contains the fitness calculators for each objective being optimized.
    /// Each calculator evaluates individuals based on a different criterion, allowing
    /// the algorithm to handle multiple competing objectives.
    /// </para>
    /// <para><b>For Beginners:</b> These are like the different criteria for evaluating houses.
    /// 
    /// For example:
    /// - One calculator might score houses based on price
    /// - Another might score based on location
    /// - Another might score based on size
    /// - And so on for each factor you care about
    /// 
    /// Each model gets evaluated on all these criteria independently, allowing the algorithm
    /// to find solutions that represent different trade-offs between the objectives.
    /// </para>
    /// </remarks>
    private readonly List<IFitnessCalculator<T, TInput, TOutput>> _objectives = default!;

    /// <summary>
    /// Initializes a new instance of the NSGA-II algorithm with the specified model factory, objectives, and evaluator.
    /// </summary>
    /// <param name="modelFactory">Factory function that creates new model instances.</param>
    /// <param name="objectives">The list of fitness calculators for each objective (must contain at least two).</param>
    /// <param name="modelEvaluator">The evaluator used to assess model performance.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two objectives are provided.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new NSGA-II algorithm instance. It requires at least two
    /// different objectives to optimize, as multi-objective optimization doesn't make sense
    /// with only one objective. The first objective in the list is used as the primary
    /// fitness calculator for compatibility with the base class.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your house-hunting criteria.
    /// 
    /// When creating the algorithm:
    /// - You provide a way to create new house designs (modelFactory)
    /// - You specify your evaluation criteria (objectives) - at least two
    /// - You need a way to evaluate designs against your criteria (modelEvaluator)
    /// 
    /// The algorithm will use these components to evolve a set of house designs that
    /// offer different optimal trade-offs between your criteria.
    /// </para>
    /// </remarks>
    public NSGAII(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        List<IFitnessCalculator<T, TInput, TOutput>> objectives,
        IModelEvaluator<T, TInput, TOutput> modelEvaluator)
        : base(modelFactory, objectives[0], modelEvaluator)
    {
        if (objectives == null || objectives.Count < 2)
        {
            throw new ArgumentException("NSGA-II requires at least two objectives", nameof(objectives));
        }

        _objectives = objectives;
    }

    /// <summary>
    /// Creates the next generation of individuals using the NSGA-II algorithm.
    /// </summary>
    /// <param name="trainingInput">The input data used for training.</param>
    /// <param name="trainingOutput">The expected output data for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <returns>The new population for the next generation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core NSGA-II algorithm for creating the next generation:
    /// 1. Creates offspring through selection, crossover, and mutation
    /// 2. Combines parents and offspring into a larger pool
    /// 3. Performs non-dominated sorting to divide the pool into Pareto fronts
    /// 4. Selects the new population by taking complete fronts until the population size is reached
    /// 5. For the last front that only partially fits, uses crowding distance to select the most diverse individuals
    /// </para>
    /// <para><b>For Beginners:</b> This is like the process of improving your house options over time.
    /// 
    /// The algorithm:
    /// 1. Creates new house designs by combining features from promising existing designs
    /// 2. Puts all house designs (old and new) into one big pool
    /// 3. Sorts them into tiers based on how good their trade-offs are
    /// 4. Keeps the best tier completely, then the second-best tier, and so on
    /// 5. For the last tier that doesn't fit completely, keeps the most unique designs
    /// 
    /// This ensures each generation gets better while maintaining diverse options.
    /// </para>
    /// </remarks>
    protected override ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> CreateNextGeneration(
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default)
    {
        // Create offspring through selection, crossover, and mutation
        var offspring = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        while (offspring.Count < Population.Count)
        {
            // Tournament selection based on crowded comparison operator
            var parents = SelectParentsNSGAII(2);

            if (parents.Count < 2)
            {
                continue;
            }

            // Crossover
            var children = Crossover(parents[0], parents[1], GeneticParams.CrossoverRate);

            // Mutation
            foreach (var child in children)
            {
                var mutated = Mutate(child, GeneticParams.MutationRate);

                // Evaluate the child with all objectives
                EvaluateIndividualMultiObjective(
                    mutated, trainingInput, trainingOutput, validationInput, validationOutput);

                offspring.Add(mutated);

                if (offspring.Count >= Population.Count)
                {
                    break;
                }
            }
        }

        // Combine parents and offspring
        var combinedPopulation = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();
        combinedPopulation.AddRange(Population);
        combinedPopulation.AddRange(offspring);

        // Perform non-dominated sorting
        var fronts = NonDominatedSort(combinedPopulation);

        // Create the new population
        var newPopulation = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        // Add complete fronts as long as they fit
        int i = 0;
        while (newPopulation.Count + fronts[i].Count <= GeneticParams.PopulationSize && i < fronts.Count)
        {
            newPopulation.AddRange(fronts[i]);
            i++;
        }

        // If we need more individuals to fill the population
        if (newPopulation.Count < GeneticParams.PopulationSize && i < fronts.Count)
        {
            // Calculate crowding distance for the remaining front
            CalculateCrowdingDistance(fronts[i]);

            // Sort by crowding distance (descending) and take what we need
            var lastFrontSorted = fronts[i]
                .OrderByDescending(ind => _crowdingDistances.TryGetValue(ind, out double value) ? value : 0.0)
                .Take(GeneticParams.PopulationSize - newPopulation.Count)
                .ToList();

            newPopulation.AddRange(lastFrontSorted);
        }

        return newPopulation;
    }

    /// <summary>
    /// Selects parents for reproduction using the NSGA-II crowded tournament selection.
    /// </summary>
    /// <param name="count">The number of parents to select.</param>
    /// <returns>A list of selected individuals to be used as parents.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the crowded tournament selection procedure from NSGA-II.
    /// For each parent needed, it randomly selects two individuals and compares them
    /// based first on their non-dominated rank, and then on their crowding distance
    /// if they share the same rank.
    /// </para>
    /// <para><b>For Beginners:</b> This is like holding a series of mini-competitions to choose parents.
    /// 
    /// For each parent needed:
    /// 1. Two house designs are randomly picked from the population
    /// 2. If one is in a better tier (lower rank), it wins
    /// 3. If they're in the same tier, the more unique one (higher crowding distance) wins
    /// 4. The winner gets selected as a parent for creating new designs
    /// 
    /// This selection method balances quality (better tier) with diversity (uniqueness),
    /// ensuring both convergence to optimal solutions and exploration of different trade-offs.
    /// </para>
    /// </remarks>
    private List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> SelectParentsNSGAII(int count)
    {
        var selected = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        for (int i = 0; i < count; i++)
        {
            // Binary tournament selection
            int a = Random.Next(Population.Count);
            int b = Random.Next(Population.Count);

            var candidateA = Population[a];
            var candidateB = Population[b];

            // Get ranks
            int rankA = _ranks.TryGetValue(candidateA, out int valueA) ? valueA : int.MaxValue;
            int rankB = _ranks.TryGetValue(candidateB, out int valueB) ? valueB : int.MaxValue;

            if (rankA < rankB)
            {
                selected.Add(candidateA);
            }
            else if (rankB < rankA)
            {
                selected.Add(candidateB);
            }
            else
            {
                // Same rank, compare crowding distance
                double distanceA = _crowdingDistances.TryGetValue(candidateA, out double valueA2) ? valueA2 : 0.0;
                double distanceB = _crowdingDistances.TryGetValue(candidateB, out double valueB2) ? valueB2 : 0.0;

                if (distanceA > distanceB)
                {
                    selected.Add(candidateA);
                }
                else
                {
                    selected.Add(candidateB);
                }
            }
        }

        return selected;
    }

    /// <summary>
    /// Performs non-dominated sorting to divide the population into Pareto fronts.
    /// </summary>
    /// <param name="population">The population to sort.</param>
    /// <returns>A list of fronts, where each front is a list of non-dominated individuals.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the fast non-dominated sorting algorithm from NSGA-II.
    /// It divides the population into fronts where individuals in front 0 are not dominated by any other individual,
    /// individuals in front 1 are only dominated by individuals in front 0, and so on.
    /// </para>
    /// <para><b>For Beginners:</b> This is like organizing houses into quality tiers.
    /// 
    /// The process works like this:
    /// 1. For each house, count how many houses dominate it (are better in all aspects)
    /// 2. Houses not dominated by any others go in the "gold tier" (front 0)
    /// 3. Remove those houses and repeat - the next set goes to "silver tier" (front 1)
    /// 4. Continue until all houses are assigned to tiers
    /// 
    /// This creates a natural hierarchy of solutions based on the quality of their trade-offs,
    /// which is essential for multi-objective optimization.
    /// </para>
    /// </remarks>
    private List<List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>> NonDominatedSort(
        List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> population)
    {
        // Clear previous ranks
        _ranks.Clear();

        // List of fronts (each front is a list of individuals)
        var fronts = new List<List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>>();

        // For each individual, count how many individuals dominate it
        // and store which individuals it dominates
        var dominationCounts = new Dictionary<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, int>();
        var dominated = new Dictionary<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>,
            List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>>();

        foreach (var p in population)
        {
            dominationCounts[p] = 0;
            dominated[p] = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();
        }

        // Compare each individual with every other
        for (int i = 0; i < population.Count; i++)
        {
            var p = population[i];

            for (int j = i + 1; j < population.Count; j++)
            {
                var q = population[j];

                if (Dominates(p, q))
                {
                    dominated[p].Add(q);
                    dominationCounts[q]++;
                }
                else if (Dominates(q, p))
                {
                    dominated[q].Add(p);
                    dominationCounts[p]++;
                }
            }
        }

        // The first front consists of individuals with domination count 0
        var front0 = population.Where(p => dominationCounts[p] == 0).ToList();
        fronts.Add(front0);

        // Assign rank 0 to front 0
        foreach (var p in front0)
        {
            _ranks[p] = 0;
        }

        // Build the rest of the fronts
        int frontIndex = 0;
        while (fronts[frontIndex].Count > 0)
        {
            var nextFront = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

            foreach (var p in fronts[frontIndex])
            {
                foreach (var q in dominated[p])
                {
                    dominationCounts[q]--;

                    if (dominationCounts[q] == 0)
                    {
                        _ranks[q] = frontIndex + 1;
                        nextFront.Add(q);
                    }
                }
            }

            frontIndex++;
            if (nextFront.Count > 0)
            {
                fronts.Add(nextFront);
            }
        }

        return fronts;
    }

    /// <summary>
    /// Determines whether one individual dominates another in the Pareto sense.
    /// </summary>
    /// <param name="a">The first individual to compare.</param>
    /// <param name="b">The second individual to compare.</param>
    /// <returns>True if individual a dominates individual b, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if individual a Pareto-dominates individual b by comparing their
    /// objective values. Individual a dominates b if it is at least as good as b in all
    /// objectives and strictly better in at least one objective.
    /// </para>
    /// <para><b>For Beginners:</b> This is like comparing two houses to see if one is clearly better.
    /// 
    /// House A dominates House B if:
    /// - House A is at least as good as House B in ALL aspects (price, location, size, etc.)
    /// - AND House A is strictly better than House B in at least ONE aspect
    /// 
    /// For example, if House A costs the same as House B but is bigger and in a better location,
    /// then A dominates B. But if A is bigger and cheaper while B has a better location,
    /// neither dominates the other - they represent different trade-offs.
    /// </para>
    /// </remarks>
    private bool Dominates(ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> a,
                           ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> b)
    {
        bool atLeastOneBetter = false;

        // Compare objective values
        for (int i = 0; i < _objectives.Count; i++)
        {
            var calculator = _objectives[i];

            // Get fitness values
            T fitnessA = GetObjectiveFitness(a, i);
            T fitnessB = GetObjectiveFitness(b, i);

            // Compare based on whether higher is better
            bool isBetter;
            if (calculator.IsHigherScoreBetter)
            {
                isBetter = MathHelper.GetNumericOperations<T>().GreaterThan(fitnessA, fitnessB);
            }
            else
            {
                isBetter = MathHelper.GetNumericOperations<T>().LessThan(fitnessA, fitnessB);
            }

            bool isWorse;
            if (calculator.IsHigherScoreBetter)
            {
                isWorse = MathHelper.GetNumericOperations<T>().LessThan(fitnessA, fitnessB);
            }
            else
            {
                isWorse = MathHelper.GetNumericOperations<T>().GreaterThan(fitnessA, fitnessB);
            }

            // If a is worse than b in any objective, a does not dominate b
            if (isWorse)
            {
                return false;
            }

            // Keep track if a is better than b in at least one objective
            if (isBetter)
            {
                atLeastOneBetter = true;
            }
        }

        // a dominates b if it's not worse in any objective and better in at least one
        return atLeastOneBetter;
    }

    /// <summary>
    /// Calculates the crowding distance for individuals in a Pareto front.
    /// </summary>
    /// <param name="front">The Pareto front of individuals.</param>
    /// <remarks>
    /// <para>
    /// This method calculates the crowding distance for each individual in a Pareto front.
    /// The crowding distance is a measure of how close an individual is to its neighbors in the objective space.
    /// Boundary points (with extreme values in any objective) get an infinite distance.
    /// </para>
    /// <para><b>For Beginners:</b> This is like measuring how unique each house is within its tier.
    /// 
    /// For each tier of houses:
    /// 1. For each evaluation criterion (price, location, etc.):
    ///    - Sort houses based on that criterion
    ///    - Houses with extreme values (cheapest, most expensive, etc.) are considered unique
    ///    - For the rest, measure how different they are from their neighbors
    /// 2. Add up these differences across all criteria to get a "uniqueness score"
    /// 
    /// Houses with higher uniqueness scores offer trade-offs that aren't represented by many
    /// other options, making them valuable to preserve for diversity.
    /// </para>
    /// </remarks>
    private void CalculateCrowdingDistance(List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> front)
    {
        int n = front.Count;

        // Initialize distances
        foreach (var individual in front)
        {
            _crowdingDistances[individual] = 0;
        }

        // Calculate distance for each objective
        for (int objectiveIndex = 0; objectiveIndex < _objectives.Count; objectiveIndex++)
        {
            // Sort by objective value
            front.Sort((a, b) =>
            {
                T fitnessA = GetObjectiveFitness(a, objectiveIndex);
                T fitnessB = GetObjectiveFitness(b, objectiveIndex);

                var numOps = MathHelper.GetNumericOperations<T>();

                // Implement comparison using existing methods
                if (numOps.Equals(fitnessA, fitnessB))
                    return 0;
                else if (numOps.LessThan(fitnessA, fitnessB))
                    return -1;
                else
                    return 1;
            });

            // Boundary points get infinite distance
            _crowdingDistances[front[0]] = double.PositiveInfinity;
            _crowdingDistances[front[n - 1]] = double.PositiveInfinity;

            // Get min and max fitness for normalization
            T minFitness = GetObjectiveFitness(front[0], objectiveIndex);
            T maxFitness = GetObjectiveFitness(front[n - 1], objectiveIndex);

            var numOps = MathHelper.GetNumericOperations<T>();

            // Skip if all values are the same (avoid division by zero)
            if (numOps.Equals(minFitness, maxFitness))
            {
                continue;
            }

            // Calculate crowding distance for each point except boundaries
            for (int i = 1; i < n - 1; i++)
            {
                T prevFitness = GetObjectiveFitness(front[i - 1], objectiveIndex);
                T nextFitness = GetObjectiveFitness(front[i + 1], objectiveIndex);

                // Add normalized distance to crowding distance
                double fitnessRange = Convert.ToDouble(numOps.Subtract(maxFitness, minFitness));
                double distance = Convert.ToDouble(numOps.Subtract(nextFitness, prevFitness)) / fitnessRange;

                _crowdingDistances[front[i]] += distance;
            }
        }
    }

    /// <summary>
    /// Gets the fitness value of an individual for a specific objective.
    /// </summary>
    /// <param name="individual">The individual to evaluate.</param>
    /// <param name="objectiveIndex">The index of the objective.</param>
    /// <returns>The fitness value for the specified objective.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves the cached fitness value of an individual for a specific objective.
    /// The values are stored in the model's metadata during evaluation to avoid recalculating them.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking a house's score on a specific criterion.
    /// 
    /// For example:
    /// - objectiveIndex 0 might get the house's price score
    /// - objectiveIndex 1 might get the house's location score
    /// - objectiveIndex 2 might get the house's size score
    /// 
    /// The method looks up these previously calculated scores rather than re-evaluating
    /// the house each time, which makes the algorithm more efficient.
    /// </para>
    /// </remarks>
    private T GetObjectiveFitness(ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual, int objectiveIndex)
    {
        // Convert the individual to a model
        var model = IndividualToModel(individual);

        // Store objective values in the model's metadata for quick access
        var metadata = model.GetModelMetadata();

        if (metadata.AdditionalInfo.TryGetValue($"Objective_{objectiveIndex}", out object? objValue))
        {
            if (objValue is string fitnessStr)
            {
                return MathHelper.GetNumericOperations<T>().FromDouble(double.Parse(fitnessStr));
            }
        }

        // If not found, return default fitness (should not happen if evaluated properly)
        return individual.GetFitness();
    }

    /// <summary>
    /// Evaluates an individual against all objectives.
    /// </summary>
    /// <param name="individual">The individual to evaluate.</param>
    /// <param name="trainingInput">The input data used for training.</param>
    /// <param name="trainingOutput">The expected output data for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <remarks>
    /// <para>
    /// This method evaluates an individual against all objectives and stores the results
    /// in the model's metadata. It temporarily switches the fitness calculator for each objective
    /// and uses the base class's evaluation method.
    /// </para>
    /// <para><b>For Beginners:</b> This is like evaluating a house design on all your criteria.
    /// 
    /// For each house design:
    /// 1. Check how it scores on price
    /// 2. Check how it scores on location
    /// 3. Check how it scores on size
    /// 4. Check how it scores on quality
    /// 5. Store all these scores so they can be used for comparisons
    /// 
    /// The primary fitness (used for backward compatibility) is set to the first objective,
    /// but all objective values are stored and used in the NSGA-II algorithm.
    /// </para>
    /// </remarks>
    private void EvaluateIndividualMultiObjective(
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput,
        TOutput? validationOutput)
    {
        // Get the model for this individual
        var model = IndividualToModel(individual);

        // Evaluate the individual for each objective
        for (int i = 0; i < _objectives.Count; i++)
        {
            var calculator = _objectives[i];

            // Use the base class's evaluation method but with the specific calculator
            var tempCalculator = FitnessCalculator;
            FitnessCalculator = calculator;

            T fitness = EvaluateIndividual(individual, trainingInput, trainingOutput, validationInput, validationOutput);

            // Restore the original calculator
            FitnessCalculator = tempCalculator;

            // Store the objective value in the model's metadata
            var metadata = model.GetModelMetadata();
            metadata.AdditionalInfo[$"Objective_{i}"] = Convert.ToDouble(fitness);
        }

        // Set the primary fitness using the first objective
        individual.SetFitness(GetObjectiveFitness(individual, 0));
    }

    /// <summary>
    /// Gets metadata about the algorithm and its current state.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the algorithm.</returns>
    /// <remarks>
    /// <para>
    /// This method extends the base implementation to include NSGA-II-specific information
    /// in the metadata. It adds details about the algorithm type, description, and the number
    /// of objectives being optimized.
    /// </para>
    /// <para><b>For Beginners:</b> This is like getting a summary report about the house-hunting agent.
    /// 
    /// The metadata includes:
    /// - The type of algorithm being used (NSGA-II)
    /// - A description of what it does
    /// - How many different criteria it's considering (objectives)
    /// 
    /// This information is useful for understanding and documenting the optimization process,
    /// especially when comparing different algorithms or configurations.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetMetaData()
    {
        var metadata = base.GetMetaData();
        metadata.ModelType = ModelType.GeneticAlgorithmRegression;
        metadata.Description = "Model evolved using NSGA-II (Non-dominated Sorting Genetic Algorithm II)";
        metadata.AdditionalInfo["ObjectiveCount"] = _objectives.Count;

        return metadata;
    }
}