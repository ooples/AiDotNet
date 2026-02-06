global using AiDotNet.Genetics;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Genetic Algorithm optimizer for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The Genetic Algorithm optimizer is an evolutionary optimization technique inspired by the process of natural selection.
/// It evolves a population of potential solutions over multiple generations to find an optimal or near-optimal solution.
/// </para>
/// <para><b>For Beginners:</b> Think of the Genetic Algorithm optimizer like breeding the best solutions:
/// 
/// - Start with a group of random solutions (like a group of different recipes)
/// - Test how good each solution is (like tasting each recipe)
/// - Choose the best solutions (like picking the tastiest recipes)
/// - Create new solutions by mixing the best ones (like combining ingredients from the best recipes)
/// - Sometimes make small random changes (like accidentally adding a new spice)
/// - Repeat this process many times to find the best solution (or the tastiest recipe!)
/// 
/// This approach is good at finding solutions for complex problems where traditional methods might struggle.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GeneticAlgorithmOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Genetic Algorithm.
    /// </summary>
    private GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> _geneticOptions;

    /// <summary>
    /// The current crossover rate, which determines how often solutions are combined.
    /// </summary>
    private T _currentCrossoverRate;

    /// <summary>
    /// The current mutation rate, which determines how often random changes are made to solutions.
    /// </summary>
    private T _currentMutationRate;

    /// <summary>
    /// The genetic algorithm instance used for optimization.
    /// </summary>
    private GeneticBase<T, TInput, TOutput> _geneticAlgorithm;

    /// <summary>
    /// Initializes a new instance of the GeneticAlgorithmOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the genetic algorithm with its initial settings.
    /// You can customize various aspects of how it works, or use default settings if you're unsure.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the genetic algorithm.</param>
    public GeneticAlgorithmOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>? options = null,
        GeneticBase<T, TInput, TOutput>? geneticAlgorithm = null,
        IFitnessCalculator<T, TInput, TOutput>? fitnessCalculator = null)
        : base(model, options ?? new())
    {
        _geneticOptions = options ?? new GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;

        // If no genetic algorithm is provided, create a default StandardGeneticAlgorithm
        if (geneticAlgorithm == null)
        {
            // Use the provided model as a template, falling back to SimpleRegression if not provided
            var templateModel = model;
            IFullModel<T, TInput, TOutput> modelFactory()
            {
                // Clone the template model to get a fresh instance with the same configuration
                if (templateModel != null)
                {
                    return templateModel.Clone();
                }
                return (IFullModel<T, TInput, TOutput>)new SimpleRegression<T>();
            }

            _geneticAlgorithm = new StandardGeneticAlgorithm<T, TInput, TOutput>(
                modelFactory,
                fitnessCalculator ?? new MeanSquaredErrorFitnessCalculator<T, TInput, TOutput>());
        }
        else
        {
            _geneticAlgorithm = geneticAlgorithm;
        }

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Updates the adaptive parameters used in the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm behaves based on its recent performance.
    /// It's like a chef adjusting their cooking technique based on how the last few dishes turned out.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
        AdaptiveParametersHelper<T, TInput, TOutput>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate,
            currentStepData, previousStepData, _geneticOptions);
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial rates for crossover (mixing solutions)
    /// and mutation (making small random changes). It's like setting the initial recipe and how often
    /// you'll try new ingredients.
    /// </para>
    /// </remarks>
    private new void InitializeAdaptiveParameters()
    {
        _currentCrossoverRate = NumOps.FromDouble(_geneticOptions.CrossoverRate);
        _currentMutationRate = NumOps.FromDouble(_geneticOptions.MutationRate);
    }

    /// <summary>
    /// Performs the main optimization process using the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the genetic algorithm. It:
    /// 1. Creates an initial group of random solutions
    /// 2. Evaluates how good each solution is
    /// 3. Selects the best solutions
    /// 4. Creates new solutions by mixing the best ones
    /// 5. Sometimes makes small random changes to solutions
    /// 6. Repeats this process for many generations
    /// 
    /// It's like running a cooking competition where each round you keep the best recipes,
    /// combine them to make new recipes, and occasionally add a surprise ingredient.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize genetic algorithm parameters
        var geneticParams = _geneticAlgorithm.GetGeneticParameters();
        geneticParams.PopulationSize = _geneticOptions.PopulationSize;
        geneticParams.MaxGenerations = Options.MaxIterations;
        geneticParams.CrossoverRate = Convert.ToDouble(_currentCrossoverRate);
        geneticParams.MutationRate = Convert.ToDouble(_currentMutationRate);

        _geneticAlgorithm.ConfigureGeneticParameters(geneticParams);

        // Let the genetic algorithm handle the evolutionary process
        var evolutionStats = _geneticAlgorithm.Evolve(
            Options.MaxIterations,
            inputData.XTrain,
            inputData.YTrain,
            inputData.XValidation,
            inputData.YValidation);

        // Convert the result to optimization result format
        var bestIndividual = _geneticAlgorithm.GetBestIndividual();
        var model = _geneticAlgorithm.IndividualToModel(bestIndividual);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = model,
            FitnessScore = bestIndividual.GetFitness(),
        };

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the options for the genetic algorithm optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the genetic algorithm
    /// while it's running. It's like adjusting the rules of your cooking competition mid-way through.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> geneticOptions)
        {
            _geneticOptions = geneticOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected GeneticAlgorithmOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for the genetic algorithm optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method returns the current settings of the genetic algorithm.
    /// It's like checking the current rules of your cooking competition.
    /// </para>
    /// </remarks>
    /// <returns>The current genetic algorithm optimizer options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _geneticOptions;
    }

    /// <summary>
    /// Serializes the genetic algorithm optimizer to a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves all the important information about the current state
    /// of the genetic algorithm into a format that can be easily stored or transmitted.
    /// It's like writing down all the details of your cooking competition so you can recreate it later.
    /// </para>
    /// </remarks>
    /// <returns>A byte array containing the serialized data of the optimizer.</returns>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GeneticAlgorithmOptimizerOptions
        string optionsJson = JsonConvert.SerializeObject(_geneticOptions);
        writer.Write(optionsJson);

        // Serialize the genetic algorithm itself
        byte[] geneticAlgorithmData = _geneticAlgorithm.Serialize();
        writer.Write(geneticAlgorithmData.Length);
        writer.Write(geneticAlgorithmData);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the genetic algorithm optimizer from a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method recreates the genetic algorithm optimizer from previously saved data.
    /// It's like using your written notes to set up your cooking competition exactly as it was before.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized data of the optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of the optimizer options fails.</exception>
    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize GeneticAlgorithmOptimizerOptions
        string optionsJson = reader.ReadString();
        _geneticOptions = JsonConvert.DeserializeObject<GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Deserialize the genetic algorithm if data is available
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            int geneticAlgorithmDataLength = reader.ReadInt32();
            byte[] geneticAlgorithmData = reader.ReadBytes(geneticAlgorithmDataLength);
            _geneticAlgorithm.Deserialize(geneticAlgorithmData);
        }

        InitializeAdaptiveParameters();
    }
}
