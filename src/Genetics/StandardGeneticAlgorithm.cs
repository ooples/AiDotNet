using AiDotNet.Extensions;

namespace AiDotNet.Genetics;

public class StandardGeneticAlgorithm<T, TInput, TOutput> :
    GeneticBase<T, TInput, TOutput>
{
    private readonly Func<IFullModel<T, TInput, TOutput>> _modelFactory;

    public StandardGeneticAlgorithm(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        IFitnessCalculator<T, TInput, TOutput> fitnessCalculator,
        IModelEvaluator<T, TInput, TOutput> modelEvaluator)
        : base(fitnessCalculator, modelEvaluator)
    {
        _modelFactory = modelFactory ?? throw new ArgumentNullException(nameof(modelFactory));
    }

    protected override ModelParameterGene<T> MutateGene(ModelParameterGene<T> gene)
    {
        double perturbation = (Random.NextDouble() * 2 - 1) * 0.1; // Small random change

        var clone = new ModelParameterGene<T>(
            gene.Index,
            NumOps.Add(gene.Value, NumOps.FromDouble(perturbation))
        );

        return clone;
    }

    protected override ModelParameterGene<T> MutateGeneGaussian(ModelParameterGene<T> gene)
    {
        double stdDev = 0.1;
        double perturbation = Random.NextGaussian() * stdDev;

        var clone = new ModelParameterGene<T>(
            gene.Index,
            NumOps.Add(gene.Value, NumOps.FromDouble(perturbation))
        );

        return clone;
    }

    public override ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> InitializePopulation(
        int populationSize,
        InitializationMethod initializationMethod)
    {
        var population = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        for (int i = 0; i < populationSize; i++)
        {
            var model = _modelFactory();

            // Get model parameters - if insufficient (untrained model), determine correct dimensions
            var parameters = model.GetParameters();

            // Check if model is untrained by comparing parameter count to expected count based on input
            // For untrained models with UseIntercept=true, GetParameters() returns just [intercept] (length 1)
            // For untrained models with UseIntercept=false, GetParameters() returns empty (length 0)
            // We need: numFeatures coefficients + optional intercept
            if (TrainingInputForInitialization is not null)
            {
                int inputDimensions = InputHelper<T, TInput>.GetInputSize(TrainingInputForInitialization);
                // For most regression models: numFeatures + 1 intercept (assuming UseIntercept=true as default)
                int expectedParameterCount = inputDimensions + 1;

                // If parameters don't match expected count, model is likely untrained - recreate with correct size
                if (parameters.Length != expectedParameterCount)
                {
                    parameters = new Vector<T>(expectedParameterCount);
                }
            }
            else if (parameters.Length == 0)
            {
                // Fallback: no training input available and parameters are empty
                // This shouldn't happen in normal flow but prevents crashes
                parameters = new Vector<T>(1);
            }

            // Initialize parameters based on the initialization method
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

    private ICollection<ModelParameterGene<T>> CreateGenesFromParameters(Vector<T> parameters)
    {
        var genes = new List<ModelParameterGene<T>>();
        for (int i = 0; i < parameters.Length; i++)
        {
            genes.Add(new ModelParameterGene<T>(i, parameters[i]));
        }
        return genes;
    }

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
                            parameters[i] = NumOps.FromDouble(Random.NextGaussian() * 0.5);
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

    // Helper methods
    private List<Vector<T>> GetCaseLibrary()
    {
        // This method should return a library of known good parameter sets
        // In a real implementation, these could be loaded from a database or file
        return new List<Vector<T>>();
    }

    private int GetPopulationIndex()
    {
        // This method should return the index of the current individual in the population
        _populationIndex = (_populationIndex + 1) % GeneticParams.PopulationSize;
        return _populationIndex;
    }

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

    // Field to track population index for grid initialization
    private int _populationIndex = -1;

    public override ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> CreateIndividual(
        ICollection<ModelParameterGene<T>> genes)
    {
        return new ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>(
            genes,
            g => IndividualGenesConvertToModel(g)
        );
    }

    public override IFullModel<T, TInput, TOutput> IndividualToModel(
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual)
    {
        var genes = individual.GetGenes();
        return IndividualGenesConvertToModel(genes);
    }

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
            individual.SetFitness(MathHelper.GetNumericOperations<T>().FromDouble(fitness));
        }

        return individual;
    }

    protected override byte[] SerializeModelData()
    {
        // No additional data to serialize
        return Array.Empty<byte>();
    }

    protected override void DeserializeModelData(byte[] data)
    {
        // Nothing to deserialize
    }

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
