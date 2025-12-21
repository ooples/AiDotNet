using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements a normal optimization algorithm with adaptive parameters.
/// </summary>
/// <remarks>
/// <para>
/// The NormalOptimizer uses a combination of random search and adaptive parameter tuning to find optimal solutions.
/// It incorporates elements from genetic algorithms but operates on a single solution at a time.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're trying to find the highest peak in a mountain range, but you can't see very far.
/// This optimizer is like a hiker who starts at random spots, climbs to the nearest peak, and then jumps to another random spot.
/// The hiker learns from each climb and adjusts their strategy (like how far to jump or how carefully to look around) based on whether they're finding higher peaks or not.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NormalOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Options specific to the normal optimizer, including parameters inherited from genetic algorithms.
    /// </summary>
    private GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> _normalOptions;

    /// <summary>
    /// Initializes a new instance of the NormalOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the NormalOptimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing for your hike. You're deciding what equipment to bring, how long you'll hike for,
    /// and setting up rules for how you'll explore the mountain range.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The optimization options.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public NormalOptimizer(IFullModel<T, TInput, TOutput> model, GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _normalOptions = options ?? new();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Performs the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It generates random solutions,
    /// evaluates them, and keeps track of the best solution found. It also adapts its parameters
    /// based on the performance of each solution.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is your actual hike through the mountain range. You're repeatedly:
    /// 1. Picking a random spot to start from.
    /// 2. Climbing to the nearest peak and measuring its height.
    /// 3. Remembering the highest peak you've found so far.
    /// 4. Adjusting your strategy based on whether you're finding higher peaks or not.
    /// 5. Deciding whether to keep going or stop if you think you've found the highest peak.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = ModelHelper<T, TInput, TOutput>.CreateDefaultModel(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var currentSolution = InitializeRandomSolution(inputData.XTrain);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts various parameters of the optimization process based on the performance
    /// of the current solution compared to the previous one. It updates feature selection,
    /// mutation rate, exploration/exploitation balance, population size, and crossover rate.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your hiking strategy based on your recent experiences. If you're finding higher peaks,
    /// you might decide to look more closely in the areas you're in. If not, you might decide to take bigger jumps
    /// to new areas or look at different aspects of the landscape.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Adaptive feature selection
        UpdateFeatureSelectionParameters(currentStepData, previousStepData);

        // Adaptive mutation rate
        UpdateMutationRate(currentStepData, previousStepData);

        // Adaptive exploration vs exploitation balance
        UpdateExplorationExploitationBalance(currentStepData, previousStepData);

        // Adaptive population size
        UpdatePopulationSize(currentStepData, previousStepData);

        // Adaptive crossover rate
        UpdateCrossoverRate(currentStepData, previousStepData);
    }

    /// <summary>
    /// Updates the feature selection parameters based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the minimum and maximum number of features to consider in the model.
    /// If the current solution is better, it expands the range of features. Otherwise, it narrows the range.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding whether to pay attention to more or fewer landmarks during your hike.
    /// If you're finding better views, you might start looking at more things. If not, you might focus on fewer key features.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    private void UpdateFeatureSelectionParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        if (FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            Options.MinimumFeatures = Math.Max(1, Options.MinimumFeatures - 1);
            Options.MaximumFeatures = Math.Min(Options.MaximumFeatures + 1, _normalOptions.MaximumFeatures);
        }
        else
        {
            Options.MinimumFeatures = Math.Min(Options.MinimumFeatures + 1, _normalOptions.MaximumFeatures - 1);
            Options.MaximumFeatures = Math.Max(Options.MaximumFeatures - 1, Options.MinimumFeatures + 1);
        }
    }

    /// <summary>
    /// Updates the exploration vs exploitation balance based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the exploration rate. If the current solution is better, it decreases exploration
    /// (favoring exploitation). Otherwise, it increases exploration.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding whether to explore new areas or focus more on the areas you've already found to be good.
    /// If you're finding better peaks, you might explore less and focus more on the current area. If not, you might decide to look in new places.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    private void UpdateExplorationExploitationBalance(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        if (FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            Options.ExplorationRate *= 0.98; // Decrease exploration if improving
        }
        else
        {
            Options.ExplorationRate *= 1.02; // Increase exploration if not improving
        }
        Options.ExplorationRate = MathHelper.Clamp(Options.ExplorationRate, Options.MinExplorationRate, Options.MaxExplorationRate);
    }

    /// <summary>
    /// Updates the mutation rate based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the mutation rate. If the current solution is better, it decreases the mutation rate.
    /// Otherwise, it increases the mutation rate.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding how much to vary your path as you explore. If you're finding better peaks,
    /// you might make smaller, more careful adjustments. If not, you might make bigger, more random changes to your path.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    private void UpdateMutationRate(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        if (FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            _normalOptions.MutationRate *= 0.95; // Decrease mutation rate if improving
        }
        else
        {
            _normalOptions.MutationRate *= 1.05; // Increase mutation rate if not improving
        }

        _normalOptions.MutationRate = MathHelper.Clamp(_normalOptions.MutationRate, _normalOptions.MinMutationRate, _normalOptions.MaxMutationRate);
    }

    /// <summary>
    /// Updates the population size based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the population size. If the current solution is better, it decreases the population size.
    /// Otherwise, it increases the population size, always staying within the defined limits.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding how many different paths to try at once. If you're finding better peaks,
    /// you might focus on fewer paths. If not, you might try more paths to increase your chances of finding a good one.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    private void UpdatePopulationSize(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        if (FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            _normalOptions.PopulationSize = Math.Max(_normalOptions.MinPopulationSize, _normalOptions.PopulationSize - 1);
        }
        else
        {
            _normalOptions.PopulationSize = Math.Min(_normalOptions.PopulationSize + 1, _normalOptions.MaxPopulationSize);
        }
    }

    /// <summary>
    /// Updates the crossover rate based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the crossover rate. If the current solution is better, it increases the crossover rate.
    /// Otherwise, it decreases the crossover rate, always staying within the defined limits.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding how often to combine good paths you've found. If you're finding better peaks,
    /// you might try combining good paths more often. If not, you might try combining them less and explore more independently.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    private void UpdateCrossoverRate(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        if (FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            _normalOptions.CrossoverRate *= 1.02; // Increase crossover rate if improving
        }
        else
        {
            _normalOptions.CrossoverRate *= 0.98; // Decrease crossover rate if not improving
        }

        _normalOptions.CrossoverRate = MathHelper.Clamp(_normalOptions.CrossoverRate, _normalOptions.MinCrossoverRate, _normalOptions.MaxCrossoverRate);
    }

    /// <summary>
    /// Randomly selects a subset of features to use in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method chooses a random number of features between the minimum and maximum allowed,
    /// and then randomly selects specific features to include.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding which specific landmarks or terrain features you'll use to guide your climb.
    /// You're randomly picking a set of things to pay attention to, which will influence how you explore the area.
    /// </para>
    /// </remarks>
    /// <param name="totalFeatures">The total number of available features.</param>
    /// <returns>A list of indices representing the selected features.</returns>
    private List<int> RandomlySelectFeatures(int totalFeatures)
    {
        var selectedFeatures = new List<int>();
        int numFeatures = Random.Next(Options.MinimumFeatures, Math.Min(Options.MaximumFeatures, totalFeatures) + 1);

        while (selectedFeatures.Count < numFeatures)
        {
            int feature = Random.Next(totalFeatures);
            if (!selectedFeatures.Contains(feature))
            {
                selectedFeatures.Add(feature);
            }
        }

        return selectedFeatures;
    }

    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current set of options used by the NormalOptimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current hiking plan and equipment list.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _normalOptions;
    }

    /// <summary>
    /// Updates the optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the optimizer's options with a new set of options. It checks if the provided options
    /// are of the correct type before applying them.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like updating your hiking plan with new information or equipment. You're making sure the new plan
    /// is compatible with your current approach before adopting it.
    /// </para>
    /// </remarks>
    /// <param name="options">The new optimization algorithm options to apply.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the expected type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> geneticOptions)
        {
            _normalOptions = geneticOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NormalOptimizerOptions.");
        }
    }

    /// <summary>
    /// Serializes the current state of the optimizer into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its options, into a byte array
    /// that can be stored or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of your current hiking strategy and equipment setup,
    /// so you can recreate it exactly later or share it with others.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize NormalOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_normalOptions);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by the Serialize method) and uses it to
    /// restore the optimizer's state, including its options.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a saved snapshot of a hiking strategy to set up your approach exactly as it was before.
    /// You're recreating all the details of your previous setup from the saved information.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize NormalOptimizerOptions
            string optionsJson = reader.ReadString();
            _normalOptions = JsonConvert.DeserializeObject<GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}
