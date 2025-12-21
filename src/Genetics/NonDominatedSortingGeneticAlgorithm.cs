namespace AiDotNet.Genetics;

public class NSGAII<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    // Storage for non-dominated ranking and crowding distance
    private readonly Dictionary<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, int> _ranks = [];
    private readonly Dictionary<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, double> _crowdingDistances = [];
    private readonly List<IFitnessCalculator<T, TInput, TOutput>> _objectives;

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

    public override ModelMetadata<T> GetMetaData()
    {
        var metadata = base.GetMetaData();
        metadata.ModelType = ModelType.GeneticAlgorithmRegression;
        metadata.Description = "Model evolved using NSGA-II (Non-dominated Sorting Genetic Algorithm II)";
        metadata.AdditionalInfo["ObjectiveCount"] = _objectives.Count.ToString();

        return metadata;
    }
}
