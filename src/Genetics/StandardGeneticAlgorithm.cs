namespace AiDotNet.Genetics;

/// <summary>
/// Implements a standard genetic algorithm for evolving machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type produced by the model.</typeparam>
/// <remarks>
/// <para>
/// The StandardGeneticAlgorithm class provides a traditional implementation of a genetic algorithm
/// specialized for optimizing machine learning models. It handles the core evolutionary operations
/// including population initialization with various strategies, gene mutation, individual-to-model
/// conversion, and crossover. This class serves as the foundation for more specialized genetic
/// algorithm variants in the library.
/// </para>
/// <para><b>For Beginners:</b> Think of this algorithm like a plant breeding program.
/// 
/// Imagine you're trying to develop the perfect tomato plant:
/// - You start with a diverse collection of tomato plants (initial population)
/// - You evaluate each plant based on desired characteristics (fitness evaluation)
/// - You select the best plants to be parents (selection)
/// - You cross-pollinate these plants to create offspring (crossover)
/// - You introduce small random changes to some seeds (mutation)
/// - You grow the new generation and repeat the process
/// 
/// Over many generations, your tomato plants gradually develop the traits you're looking for.
/// In the same way, this algorithm evolves machine learning models to find optimal parameters
/// that solve your specific problem.
/// </para>
/// </remarks>
public class StandardGeneticAlgorithm<T, TInput, TOutput> :
    GeneticBase<T, TInput, TOutput>
{
    /// <summary>
    /// A factory function that creates new model instances.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores a function that can create new instances of the machine learning model
    /// being optimized. The factory is used whenever the algorithm needs to create a new model,
    /// such as during population initialization or when converting individuals to models.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as your seed supplier.
    /// 
    /// In our plant breeding analogy:
    /// - This is like having a reliable supplier who can provide you with fresh seeds whenever needed
    /// - Each time you call this function, you get a new, blank model (seed) ready to be configured
    /// - These seeds all have the same basic type, but each will grow differently based on the parameters (genetic material)
    /// 
    /// Having this factory ensures you can always create new models with the same structure
    /// but different parameters as the genetic algorithm explores the solution space.
    /// </para>
    /// </remarks>
    private readonly Func<IFullModel<T, TInput, TOutput>> _modelFactory = default!;

    /// <summary>
    /// Initializes a new instance of the StandardGeneticAlgorithm class.
    /// </summary>
    /// <param name="modelFactory">Factory function that creates new model instances.</param>
    /// <param name="fitnessCalculator">The calculator used to determine model fitness.</param>
    /// <param name="modelEvaluator">The evaluator used to assess model performance.</param>
    /// <exception cref="ArgumentNullException">Thrown when modelFactory is null.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new genetic algorithm with the specified components.
    /// It requires a model factory to create new model instances, a fitness calculator to
    /// evaluate solutions, and a model evaluator to test the models on actual data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your plant breeding program.
    /// 
    /// When starting your program, you need:
    /// - A way to get new seeds (the model factory)
    /// - A method to judge how good each plant is (the fitness calculator)
    /// - A system to grow and test each plant (the model evaluator)
    /// 
    /// With these three components in place, your breeding program is ready to begin
    /// the process of developing better plants over multiple generations.
    /// </para>
    /// </remarks>
    public StandardGeneticAlgorithm(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        IFitnessCalculator<T, TInput, TOutput> fitnessCalculator,
        IModelEvaluator<T, TInput, TOutput> modelEvaluator)
        : base(fitnessCalculator, modelEvaluator)
    {
        _modelFactory = modelFactory ?? throw new ArgumentNullException(nameof(modelFactory));
    }

    /// <summary>
    /// Mutates a gene by adding a small random perturbation to its value.
    /// </summary>
    /// <param name="gene">The gene to mutate.</param>
    /// <returns>A new gene with the mutated value.</returns>
    /// <remarks>
    /// <para>
    /// This method implements a simple mutation strategy that adds a small random value
    /// between -0.1 and 0.1 to the gene's current value. This creates a slight variation
    /// while staying relatively close to the original value.
    /// </para>
    /// <para><b>For Beginners:</b> This is like making a small random change to a plant's DNA.
    /// 
    /// Imagine:
    /// - Each gene controls one characteristic of your plant (e.g., fruit size)
    /// - When a mutation occurs, that characteristic changes slightly
    /// - If the fruit size was "medium" (value 0.5), it might become "medium-large" (value 0.57)
    /// - These small changes create the variation needed for evolution
    /// 
    /// Mutations are essential because they introduce new possibilities that
    /// weren't present in the original population.
    /// </para>
    /// </remarks>
    protected override ModelParameterGene<T> MutateGene(ModelParameterGene<T> gene)
    {
        double perturbation = (Random.NextDouble() * 2 - 1) * 0.1; // Small random change

        var clone = new ModelParameterGene<T>(
            gene.Index,
            NumOps.Add(gene.Value, NumOps.FromDouble(perturbation))
        );

        return clone;
    }

    /// <summary>
    /// Mutates a gene using a Gaussian (normal) distribution for the perturbation.
    /// </summary>
    /// <param name="gene">The gene to mutate.</param>
    /// <returns>A new gene with the mutated value.</returns>
    /// <remarks>
    /// <para>
    /// This method implements a Gaussian mutation strategy using the Box-Muller transform.
    /// It generates a random value from a normal distribution with standard deviation 0.1,
    /// creating smaller changes more frequently and larger changes occasionally.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a more natural mutation process.
    /// 
    /// In nature:
    /// - Most mutations cause very small changes
    /// - Occasionally, a mutation causes a moderate change
    /// - Rarely, a mutation causes a large change
    /// 
    /// The Gaussian distribution mimics this pattern, with:
    /// - 68% of mutations causing changes within ±0.1
    /// - 95% of mutations causing changes within ±0.2
    /// - 99.7% of mutations causing changes within ±0.3
    /// 
    /// This creates a more realistic pattern of variation in your population.
    /// </para>
    /// </remarks>
    protected override ModelParameterGene<T> MutateGeneGaussian(ModelParameterGene<T> gene)
    {
        // Box-Muller transform for Gaussian random number
        double u1 = 1.0 - Random.NextDouble();
        double u2 = 1.0 - Random.NextDouble();
        double stdDev = 0.1;
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        double perturbation = randStdNormal * stdDev;

        var clone = new ModelParameterGene<T>(
            gene.Index,
            NumOps.Add(gene.Value, NumOps.FromDouble(perturbation))
        );

        return clone;
    }

    /// <summary>
    /// Initializes a population of individuals with the specified size and initialization method.
    /// </summary>
    /// <param name="populationSize">The number of individuals to create.</param>
    /// <param name="initializationMethod">The method to use for initializing model parameters.</param>
    /// <returns>A collection of individuals forming the initial population.</returns>
    /// <remarks>
    /// <para>
    /// This method creates the initial population for the genetic algorithm. Each individual
    /// is created by instantiating a model, initializing its parameters using the specified
    /// initialization method, and then converting it to a ModelIndividual. Various initialization
    /// methods provide different strategies for setting up the initial parameter space exploration.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating your first generation of plants.
    /// 
    /// Depending on the initialization method chosen:
    /// - Random: You start with completely random seeds from diverse sources
    /// - Case-Based: You start with varieties similar to ones that worked well before
    /// - Heuristic: You use gardening wisdom to make educated guesses about promising varieties
    /// - Diverse: You deliberately choose very different types of seeds
    /// - Grid: You systematically sample across the full range of possible varieties
    /// - XavierUniform: You use a special technique that works particularly well for certain plant types
    /// 
    /// This initial diversity is crucial for giving the algorithm a good starting point to explore from.
    /// </para>
    /// </remarks>
    public override ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> InitializePopulation(
        int populationSize,
        InitializationMethod initializationMethod)
    {
        var population = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        for (int i = 0; i < populationSize; i++)
        {
            var model = _modelFactory();

            // Initialize parameters based on the initialization method
            var parameters = model.GetParameters();
            InitializeParameters(parameters, model, initializationMethod);
            model = model.WithParameters(parameters);

            // Convert model to a ModelIndividual
            var genes = CreateGenesFromParameters(parameters);
            var individual = new ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>(
                genes,
                g => IndividualGenesConvertToModel(g)
            );

            population.Add(individual);
        }

        return population;
    }

    /// <summary>
    /// Creates a collection of genes from a vector of model parameters.
    /// </summary>
    /// <param name="parameters">The vector of model parameters.</param>
    /// <returns>A collection of ModelParameterGene objects.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a vector of model parameters into a collection of ModelParameterGene objects.
    /// Each parameter value becomes a gene with an index corresponding to its position in the vector.
    /// This is used to convert a model's parameter representation to the genetic representation used by individuals.
    /// </para>
    /// <para><b>For Beginners:</b> This is like extracting the DNA from a plant.
    /// 
    /// For example:
    /// - You have a tomato plant with certain characteristics (parameters)
    /// - You need to extract its genetic material (genes) to use in your breeding program
    /// - Each characteristic (parameter) is encoded as a specific gene
    /// - The collection of genes represents the complete genetic makeup of the plant
    /// 
    /// This process allows the algorithm to work with the plant's genetic material
    /// rather than just its observable characteristics.
    /// </para>
    /// </remarks>
    private ICollection<ModelParameterGene<T>> CreateGenesFromParameters(Vector<T> parameters)
    {
        var genes = new List<ModelParameterGene<T>>();
        for (int i = 0; i < parameters.Length; i++)
        {
            genes.Add(new ModelParameterGene<T>(i, parameters[i]));
        }
        return genes;
    }

    /// <summary>
    /// Converts a collection of genes into a model by setting its parameters.
    /// </summary>
    /// <param name="genes">The collection of genes to convert.</param>
    /// <returns>A new model with parameters set from the genes.</returns>
    /// <remarks>
    /// <para>
    /// This method converts genetic information (genes) back into a model by creating a new model
    /// instance and setting its parameters based on the gene values. The genes are ordered by their
    /// index to ensure they map correctly to the corresponding parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is like growing a plant from its DNA.
    /// 
    /// Imagine:
    /// - You have the genetic code for a tomato plant (the genes)
    /// - You create a new seedling (new model instance)
    /// - You apply the genetic information to determine how the plant will grow
    /// - The result is a fully-grown plant with characteristics defined by that genetic code
    /// 
    /// This process allows the algorithm to evaluate how a specific set of genes
    /// would perform if expressed in an actual plant.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> IndividualGenesConvertToModel(ICollection<ModelParameterGene<T>> genes)
    {
        var model = _modelFactory();

        // Create a parameters vector from the genes
        var genesList = genes.OrderBy(g => g.Index).ToList();
        var parameters = new Vector<T>(genesList.Count);

        for (int i = 0; i < genesList.Count; i++)
        {
            parameters[i] = genesList[i].Value;
        }

        return model.WithParameters(parameters);
    }

    /// <summary>
    /// Initializes model parameters using the specified initialization method.
    /// </summary>
    /// <param name="parameters">The parameters vector to initialize.</param>
    /// <param name="model">The model associated with these parameters.</param>
    /// <param name="method">The initialization method to use.</param>
    /// <remarks>
    /// <para>
    /// This method implements various strategies for parameter initialization, including:
    /// - Random: Simple random values between -1 and 1
    /// - Case-Based: Values based on known good solutions with small variations
    /// - Heuristic: Values based on problem-specific knowledge
    /// - Diverse: Values from different distributions to ensure diversity
    /// - Grid: Systematic coverage of the parameter space
    /// - XavierUniform: Special initialization for neural networks
    /// Each strategy provides different trade-offs in terms of exploration and exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like choosing different ways to create your initial seed varieties.
    /// 
    /// The different methods are like:
    /// - Random: Picking seeds randomly from a large seed bank
    /// - Case-Based: Starting with seeds from plants that performed well in similar conditions
    /// - Heuristic: Using gardening knowledge to select promising varieties
    /// - Diverse: Deliberately choosing seeds with very different characteristics
    /// - Grid: Systematically selecting seeds that cover all possible combinations of traits
    /// - XavierUniform: Using a special technique developed specifically for certain types of plants
    /// 
    /// Each approach has its strengths, and the right choice depends on what you know about your problem.
    /// </para>
    /// </remarks>
    private void InitializeParameters(Vector<T> parameters, IFullModel<T, TInput, TOutput> model,
    InitializationMethod method)
    {
        switch (method)
        {
            case InitializationMethod.Random:
                // Simple random initialization between -1 and 1
                for (int i = 0; i < parameters.Length; i++)
                {
                    parameters[i] = NumOps.FromDouble(Random.NextDouble() * 2 - 1);
                }
                break;

            case InitializationMethod.CaseBased:
                // Initialize based on known good solutions from a case library
                var caseLibrary = GetCaseLibrary();
                if (caseLibrary.Count > 0)
                {
                    // Select a random case from the library
                    var selectedCase = caseLibrary[Random.Next(caseLibrary.Count)];

                    // Copy its parameters with slight variations
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        if (i < selectedCase.Length)
                        {
                            // Add small perturbation to the case parameters
                            double perturbation = (Random.NextDouble() * 0.2 - 0.1); // ±10% variation
                            parameters[i] = NumOps.Add(
                                selectedCase[i],
                                NumOps.FromDouble(perturbation * Convert.ToDouble(selectedCase[i]))
                            );
                        }
                        else
                        {
                            // Fill remaining parameters randomly
                            parameters[i] = NumOps.FromDouble(Random.NextDouble() * 2 - 1);
                        }
                    }
                }
                else
                {
                    // Fall back to random if no cases available
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        parameters[i] = NumOps.FromDouble(Random.NextDouble() * 2 - 1);
                    }
                }
                break;

            case InitializationMethod.Heuristic:
                // Initialize using problem-specific heuristics
                if (model is IFeatureAware featureAware)
                {
                    // Get active features
                    var activeFeatures = featureAware.GetActiveFeatureIndices().ToList();
                    int activeFeatureCount = activeFeatures.Count;

                    // Basic heuristic: smaller weights for more inputs
                    double scale = activeFeatureCount > 0 ? 1.0 / Math.Sqrt(activeFeatureCount) : 0.5;

                    for (int i = 0; i < parameters.Length; i++)
                    {
                        // Give non-zero weights to active features, smaller weights to others
                        bool isActiveParameter = i < activeFeatures.Count;
                        double paramScale = isActiveParameter ? scale : scale * 0.5;
                        parameters[i] = NumOps.FromDouble((Random.NextDouble() * 2 - 1) * paramScale);
                    }
                }
                else
                {
                    // Fall back to random if no feature information available
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        parameters[i] = NumOps.FromDouble(Random.NextDouble() * 2 - 1);
                    }
                }
                break;

            case InitializationMethod.Diverse:
                // Create a diverse set of individuals by using different distributions
                int distributionIndex = Random.Next(4);

                switch (distributionIndex)
                {
                    case 0:
                        // Uniform distribution in [-1, 1]
                        for (int i = 0; i < parameters.Length; i++)
                        {
                            parameters[i] = NumOps.FromDouble(Random.NextDouble() * 2 - 1);
                        }
                        break;

                    case 1:
                        // Normal distribution with mean 0, std 0.5
                        for (int i = 0; i < parameters.Length; i++)
                        {
                            double u1 = 1.0 - Random.NextDouble();
                            double u2 = 1.0 - Random.NextDouble();
                            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                            parameters[i] = NumOps.FromDouble(randStdNormal * 0.5);
                        }
                        break;

                    case 2:
                        // Small values [-0.1, 0.1]
                        for (int i = 0; i < parameters.Length; i++)
                        {
                            parameters[i] = NumOps.FromDouble((Random.NextDouble() * 0.2) - 0.1);
                        }
                        break;

                    case 3:
                        // Bimodal distribution (values near -0.5 or 0.5)
                        for (int i = 0; i < parameters.Length; i++)
                        {
                            double sign = Random.NextDouble() < 0.5 ? -1.0 : 1.0;
                            double value = 0.5 + (Random.NextDouble() * 0.2 - 0.1); // 0.4 to 0.6
                            parameters[i] = NumOps.FromDouble(sign * value);
                        }
                        break;
                }
                break;

            case InitializationMethod.Grid:
                // Systematically cover the parameter space
                int populationIndex = GetPopulationIndex();
                int totalParameters = parameters.Length;

                // Determine grid dimensions based on population size
                int gridPointsPerDimension = Math.Max(2, (int)Math.Pow(GeneticParams.PopulationSize, 1.0 / totalParameters));

                // Map population index to grid coordinates
                int[] coordinates = new int[totalParameters];
                int remainingIndex = populationIndex;

                for (int dim = 0; dim < totalParameters; dim++)
                {
                    coordinates[dim] = remainingIndex % gridPointsPerDimension;
                    remainingIndex /= gridPointsPerDimension;
                }

                // Set parameters based on grid coordinates
                for (int i = 0; i < totalParameters; i++)
                {
                    double normalizedValue = coordinates[i] / (double)(gridPointsPerDimension - 1); // 0.0 to 1.0
                    double value = normalizedValue * 2.0 - 1.0; // Map to [-1.0, 1.0]
                    parameters[i] = NumOps.FromDouble(value);

                    // Add small random noise to prevent exact duplication at grid points
                    double noise = (Random.NextDouble() * 0.02) - 0.01; // ±1% noise
                    parameters[i] = NumOps.Add(parameters[i], NumOps.FromDouble(noise));
                }
                break;

            // Add a custom implementation for Xavier initialization
            case InitializationMethod.XavierUniform:
                int inputDimension = EstimateInputDimension(model);
                int outputDimension = EstimateOutputDimension(model);

                double limit = Math.Sqrt(6.0 / (inputDimension + outputDimension));

                for (int i = 0; i < parameters.Length; i++)
                {
                    parameters[i] = NumOps.FromDouble(Random.NextDouble() * 2 * limit - limit);
                }
                break;

            default:
                // Default to random initialization
                for (int i = 0; i < parameters.Length; i++)
                {
                    parameters[i] = NumOps.FromDouble(Random.NextDouble() * 2 - 1);
                }
                break;
        }
    }

    /// <summary>
    /// Gets a library of known good parameter sets for case-based initialization.
    /// </summary>
    /// <returns>A list of parameter vectors representing known good solutions.</returns>
    /// <remarks>
    /// <para>
    /// This method should return a collection of parameter vectors that represent known good
    /// solutions for the problem being solved. These can be used as starting points in case-based
    /// initialization. In a real implementation, these could be loaded from a database or file.
    /// </para>
    /// <para><b>For Beginners:</b> This is like accessing your seed bank of known good varieties.
    /// 
    /// Imagine:
    /// - You've been breeding tomatoes for years
    /// - You have a collection of successful varieties from previous seasons
    /// - This method retrieves those successful varieties to use as starting points
    /// - Instead of starting from scratch, you build on what worked before
    /// 
    /// In this implementation, the seed bank is empty, but in a real-world scenario,
    /// it would contain parameter sets from previously successful models.
    /// </para>
    /// </remarks>
    private List<Vector<T>> GetCaseLibrary()
    {
        // This method should return a library of known good parameter sets
        // In a real implementation, these could be loaded from a database or file
        return new List<Vector<T>>();
    }

    /// <summary>
    /// Gets the current population index for grid-based initialization.
    /// </summary>
    /// <returns>An index used to position the current individual in the parameter grid.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a counter that is used during grid-based initialization to ensure
    /// that each individual is placed at a different point in the parameter grid. It cycles
    /// through indices from 0 to population size - 1.
    /// </para>
    /// <para><b>For Beginners:</b> This is like keeping track of which plot in your garden grid you're currently planting.
    /// 
    /// Imagine:
    /// - You've divided your garden into a grid of plots
    /// - You want to plant a different variety in each plot
    /// - This method tells you which plot number you're currently working on
    /// - It ensures you use every plot and don't plant the same plot twice
    /// 
    /// This systematic approach helps ensure even coverage of the parameter space
    /// during initialization.
    /// </para>
    /// </remarks>
    private int GetPopulationIndex()
    {
        // This method should return the index of the current individual in the population
        _populationIndex = (_populationIndex + 1) % GeneticParams.PopulationSize;
        return _populationIndex;
    }

    /// <summary>
    /// Estimates the input dimension of a model.
    /// </summary>
    /// <param name="model">The model to analyze.</param>
    /// <returns>An estimated number of input features for the model.</returns>
    /// <remarks>
    /// <para>
    /// This method attempts to determine the number of input features for a model by checking
    /// several sources of information: model metadata, active features (if the model implements
    /// IFeatureAware), or a conservative estimate based on parameter count. This information
    /// is useful for certain initialization methods like Xavier initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like figuring out how many nutrients a plant needs.
    /// 
    /// Imagine:
    /// - You need to know how many different nutrients a plant requires
    /// - You first check if there's a label on the plant
    /// - If not, you look at how many different types of soil the plant interacts with
    /// - If all else fails, you make an educated guess based on the plant's size
    /// 
    /// This information helps determine appropriate initial values for the model parameters,
    /// especially for initialization methods that depend on input dimensionality.
    /// </para>
    /// </remarks>
    private int EstimateInputDimension(IFullModel<T, TInput, TOutput> model)
    {
        // Try to estimate input dimension from model's metadata or type information
        var metadata = model.GetModelMetadata();
        var parameters = model.GetParameters();

        if (metadata.AdditionalInfo.TryGetValue("InputFeatures", out object? inputFeaturesObj))
        {
            // Handle object value properly
            if (inputFeaturesObj is int intValue)
            {
                return intValue;
            }
            else if (inputFeaturesObj is string strValue && int.TryParse(strValue, out int parsedValue))
            {
                return parsedValue;
            }
        }

        // If model implements IFeatureAware, count active features
        if (model is IFeatureAware featureAware)
        {
            int activeFeatureCount = featureAware.GetActiveFeatureIndices().Count();
            if (activeFeatureCount > 0)
            {
                return activeFeatureCount;
            }
        }

        // Default estimate based on parameter count
        return Math.Max(1, parameters.Length / 4);
    }

    /// <summary>
    /// Estimates the output dimension of a model.
    /// </summary>
    /// <param name="model">The model to analyze.</param>
    /// <returns>An estimated number of output features for the model.</returns>
    /// <remarks>
    /// <para>
    /// This method attempts to determine the number of output features for a model by checking
    /// the model's metadata. If no information is available, it defaults to 1, assuming a
    /// single output variable. This information is useful for certain initialization methods
    /// like Xavier initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like figuring out how many different fruits a plant produces.
    /// 
    /// Imagine:
    /// - You need to know how many different types of fruits this plant will yield
    /// - You first check if there's information on the plant label
    /// - If not, you assume it's a simple plant that produces just one type of fruit
    /// 
    /// This information, combined with the input dimension, helps determine appropriate
    /// initial values for the model parameters in initialization methods that depend on
    /// the input-output relationship.
    /// </para>
    /// </remarks>
    private int EstimateOutputDimension(IFullModel<T, TInput, TOutput> model)
    {
        // Try to estimate output dimension from model's metadata
        var metadata = model.GetModelMetadata();

        if (metadata.AdditionalInfo.TryGetValue("OutputFeatures", out object? outputFeaturesObj))
        {
            // Handle object value properly
            if (outputFeaturesObj is int intValue)
            {
                return intValue;
            }
            else if (outputFeaturesObj is string strValue && int.TryParse(strValue, out int parsedValue))
            {
                return parsedValue;
            }
        }

        // Default estimate
        return 1;
    }

    /// <summary>
    /// Field to track population index for grid initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field keeps track of the current individual's index during grid-based initialization.
    /// It is incremented by the GetPopulationIndex method to ensure systematic coverage of the parameter space.
    /// </para>
    /// </remarks>
    private int _populationIndex = -1;

    /// <summary>
    /// Creates a new individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to initialize the individual with.</param>
    /// <returns>A new ModelIndividual initialized with the specified genes.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new individual with the provided genes and a factory function that
    /// knows how to convert those genes back into a model. This is used during genetic operations
    /// like crossover, where new individuals need to be created from gene combinations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a new plant from a specific set of genes.
    /// 
    /// Imagine:
    /// - You've selected a specific combination of genes from your breeding program
    /// - You need to create a new plant that expresses those genes
    /// - This method creates that plant, ready to be grown and evaluated
    /// 
    /// This is essential for creating offspring during the evolutionary process,
    /// especially after combining genes from different parent plants.
    /// </para>
    /// </remarks>
    public override ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> CreateIndividual(
        ICollection<ModelParameterGene<T>> genes)
    {
        return new ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>(
            genes,
            g => IndividualGenesConvertToModel(g)
        );
    }

    /// <summary>
    /// Converts an individual to a model for evaluation.
    /// </summary>
    /// <param name="individual">The individual to convert.</param>
    /// <returns>A model with parameters set according to the individual's genes.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the genes from an individual and uses them to create a model with
    /// corresponding parameters. This allows the algorithm to evaluate how the genetic information
    /// performs when expressed in a model.
    /// </para>
    /// <para><b>For Beginners:</b> This is like growing a test plant from a seed in your breeding program.
    /// 
    /// When evaluating a plant variety:
    /// - You take the plant with its genetic makeup (the individual)
    /// - You grow a test plant that expresses those genes (the model)
    /// - You can then evaluate how well this test plant performs
    /// 
    /// This conversion is necessary to assess the quality of different genetic combinations
    /// and determine which ones should be selected for the next generation.
    /// </para>
    /// </remarks>
    public override IFullModel<T, TInput, TOutput> IndividualToModel(
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual)
    {
        var genes = individual.GetGenes();
        return IndividualGenesConvertToModel(genes);
    }

    /// <summary>
    /// Gets metadata about the algorithm and its current state.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the algorithm.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a metadata object containing information about the genetic algorithm,
    /// including its type, description, and additional information such as population size,
    /// number of generations evolved, and genetic operation rates.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a report about your breeding program.
    /// 
    /// The report includes:
    /// - What type of breeding program you're running
    /// - A description of your approach
    /// - How many plants you're working with (population size)
    /// - How many generations you've bred so far
    /// - The rates at which you perform crossbreeding and mutation
    /// 
    /// This information helps document your breeding program and understand
    /// how it's configured and what stage it has reached.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetMetaData()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GeneticAlgorithmRegression,
            Description = "Model evolved using a standard genetic algorithm",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "PopulationSize", GeneticParams.PopulationSize },
                { "Generations", CurrentStats?.Generation ?? 0 },
                { "CrossoverRate", GeneticParams.CrossoverRate },
                { "MutationRate", GeneticParams.MutationRate }
            }
        };
    }

    /// <summary>
    /// Serializes an individual to a byte array.
    /// </summary>
    /// <param name="individual">The individual to serialize.</param>
    /// <returns>A byte array containing the serialized individual.</returns>
    /// <remarks>
    /// <para>
    /// This method converts an individual to a model and then serializes both the model data
    /// and the individual's fitness score. This allows the genetic algorithm to save its
    /// population state for later restoration or analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a complete record of a plant variety.
    /// 
    /// Imagine:
    /// - You want to preserve information about a promising plant variety
    /// - You convert the plant to its genetic code and full characteristics (the model)
    /// - You record how well this plant performed (the fitness)
    /// - You store all this information in a seed bank for later use
    /// 
    /// This preservation allows you to store your progress and potentially resume
    /// your breeding program from the same point in the future.
    /// </para>
    /// </remarks>
    protected override byte[] SerializeIndividual(ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual)
    {
        // Convert individual to model and use its serialization
        var model = IndividualToModel(individual);

        // Also serialize the fitness
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] modelData = model.Serialize();
        writer.Write(modelData.Length);
        writer.Write(modelData);

        // Write the fitness
        writer.Write(Convert.ToDouble(individual.GetFitness()));

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes an individual from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized individual.</param>
    /// <returns>The deserialized individual.</returns>
    /// <remarks>
    /// <para>
    /// This method recreates an individual from serialized data. It deserializes the model,
    /// extracts genes from its parameters, creates a new individual with these genes,
    /// and restores the fitness score if available.
    /// </para>
    /// <para><b>For Beginners:</b> This is like regenerating a plant from stored information.
    /// 
    /// Imagine:
    /// - You have detailed records of a plant variety in your seed bank
    /// - You recreate the plant with all its genetic traits
    /// - You also restore the information about how well it performed
    /// - The result is a complete representation of the original plant
    /// 
    /// This allows you to continue working with plants from previous breeding cycles,
    /// even if you don't have the actual plants anymore.
    /// </para>
    /// </remarks>
    protected override ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> DeserializeIndividual(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        int modelDataLength = reader.ReadInt32();
        byte[] modelData = reader.ReadBytes(modelDataLength);

        var model = _modelFactory();
        model.Deserialize(modelData);

        // Get genes from model
        var parameters = model.GetParameters();
        var genes = CreateGenesFromParameters(parameters);

        var individual = new ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>(
            genes,
            g => IndividualGenesConvertToModel(g)
        );

        // Read fitness if available
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            double fitness = reader.ReadDouble();
            individual.SetFitness(NumOps.FromDouble(fitness));
        }

        return individual;
    }

    /// <summary>
    /// Serializes additional model-specific data.
    /// </summary>
    /// <returns>A byte array containing serialized model-specific data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes any additional data specific to this genetic algorithm implementation.
    /// In the case of the standard genetic algorithm, there is no additional data to serialize
    /// beyond what is already handled by the base class and individual serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like adding special notes to your breeding program records.
    /// 
    /// In this case:
    /// - The standard genetic algorithm doesn't have any special notes to add
    /// - More specialized algorithms might include additional information
    /// 
    /// This method ensures that any unique aspects of a specific genetic algorithm
    /// implementation can be preserved when saving its state.
    /// </para>
    /// </remarks>
    protected override byte[] SerializeModelData()
    {
        // No additional data to serialize
        return Array.Empty<byte>();
    }

    /// <summary>
    /// Deserializes additional model-specific data.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model-specific data.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes any additional data specific to this genetic algorithm implementation.
    /// In the case of the standard genetic algorithm, there is no additional data to deserialize
    /// beyond what is already handled by the base class and individual deserialization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like reading special notes from your breeding program records.
    /// 
    /// In this case:
    /// - The standard genetic algorithm doesn't have any special notes to read
    /// - More specialized algorithms might need to process additional information
    /// 
    /// This method ensures that any unique aspects of a specific genetic algorithm
    /// implementation can be restored when loading its state.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelData(byte[] data)
    {
        // Nothing to deserialize
    }

    /// <summary>
    /// Serializes the entire population to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized population.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the entire population of individuals into a byte array.
    /// It saves the number of individuals followed by each serialized individual's data.
    /// This allows the complete state of the population to be preserved.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a complete catalog of all plants in your breeding program.
    /// 
    /// Imagine:
    /// - You need to document every plant variety in your program
    /// - You record how many varieties you have
    /// - Then you create a detailed record of each plant variety
    /// - The result is a complete archive of your entire breeding program
    /// 
    /// This comprehensive preservation allows you to save the entire state of your
    /// evolutionary process for later restoration or analysis.
    /// </para>
    /// </remarks>
    protected override byte[] SerializePopulation()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(Population.Count);

        foreach (var individual in Population)
        {
            var individualData = SerializeIndividual(individual);
            writer.Write(individualData.Length);
            writer.Write(individualData);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a population from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized population.</param>
    /// <returns>The deserialized collection of individuals.</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs an entire population from serialized data. It reads the number
    /// of individuals, then deserializes each individual and adds it to the population collection.
    /// This allows the complete state of the population to be restored.
    /// </para>
    /// <para><b>For Beginners:</b> This is like recreating your entire collection of plant varieties from records.
    /// 
    /// Imagine:
    /// - You have a complete catalog of all your plant varieties
    /// - You first check how many varieties are documented
    /// - Then you recreate each plant variety from its detailed record
    /// - The result is a complete restoration of your entire breeding program
    /// 
    /// This comprehensive restoration allows you to resume your evolutionary process
    /// from exactly where you left off.
    /// </para>
    /// </remarks>
    protected override ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> DeserializePopulation(byte[] data)
    {
        var population = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        int count = reader.ReadInt32();

        for (int i = 0; i < count; i++)
        {
            int length = reader.ReadInt32();
            byte[] individualData = reader.ReadBytes(length);

            var individual = DeserializeIndividual(individualData);
            population.Add(individual);
        }

        return population;
    }
}