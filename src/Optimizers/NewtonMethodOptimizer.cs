using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements an optimization algorithm based on Newton's Method for finding the minimum of a function.
/// </summary>
/// <remarks>
/// <para>
/// Newton's Method uses both first and second-order derivatives (gradient and Hessian) to find the minimum
/// of a function more efficiently than methods that use only the gradient. This can lead to faster convergence
/// for smooth, well-behaved functions.
/// </para>
/// <para><b>For Beginners:</b>
/// Think of Newton's Method like navigating a mountainous landscape to find the lowest valley:
/// 
/// - Regular gradient descent just looks at the downhill direction (gradient) at each step
/// - Newton's Method looks at both the downhill direction AND the shape of the landscape (Hessian matrix)
/// - This extra information helps you take more intelligent steps and often reach the lowest point faster
/// - However, it requires more calculations at each step and can sometimes be unstable
/// 
/// It's like having both a compass (gradient) AND a detailed topographical map (Hessian) to help you
/// navigate, rather than just the compass alone.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class NewtonMethodOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Newton's Method optimizer.
    /// </summary>
    private NewtonMethodOptimizerOptions<T, TInput, TOutput> _newtonOptions = default!;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Initializes a new instance of the NewtonMethodOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Newton's Method optimizer with the provided model and options.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing all your tools and maps before starting your journey to find the lowest point in the valley.
    /// You're setting up how you'll make decisions and what information you'll use along the way.
    /// </para>
    /// </remarks>
    /// <param name="model">The machine learning model to optimize.</param>
    /// <param name="options">The Newton's Method-specific optimization options.</param>
    public NewtonMethodOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NewtonMethodOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _newtonOptions = options ?? new NewtonMethodOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Newton's Method optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the learning rate and resets the iteration count.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting your initial step size and resetting your step counter before you start your journey.
    /// The learning rate determines how big your steps will be, and the iteration count keeps track of how many steps you've taken.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_newtonOptions.InitialLearningRate);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the optimization process using Newton's Method algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It uses Newton's Method to update the solution iteratively,
    /// aiming to find the optimal set of parameters that minimize the loss function.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is your actual journey through the valley. At each step:
    /// 1. You look at the slope (gradient) and curvature (Hessian) of the valley around you.
    /// 2. Based on this information, you calculate the best direction to move.
    /// 3. You take a step in that direction.
    /// 4. You check if you've found a better spot than any you've seen before.
    /// 5. You decide whether to keep going or stop if you think you've found the lowest point.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            _iteration++;

            // Apply different optimization strategies based on optimization mode
            var modifiedSolution = CreateSolution(inputData.XTrain);

            if (Options.OptimizationMode == OptimizationMode.ParametersOnly ||
                Options.OptimizationMode == OptimizationMode.Both)
            {
                // Newton's Method parameter optimization
                if (Random.NextDouble() < _newtonOptions.ParameterAdjustmentProbability)
                {
                    var gradient = CalculateGradient(modifiedSolution, inputData.XTrain, inputData.YTrain);
                    var hessian = CalculateHessian(modifiedSolution, inputData);
                    var direction = CalculateDirection(gradient, hessian);
                    modifiedSolution = UpdateSolution(modifiedSolution, direction);
                }
            }

            var currentStepData = EvaluateSolution(modifiedSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            // Check for convergence
            if (previousStepData.FitnessScore != null &&
                NumOps.LessThan(
                    NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, previousStepData.FitnessScore)),
                    NumOps.FromDouble(_newtonOptions.Tolerance)))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the direction for the next step in Newton's Method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the direction by multiplying the inverse of the Hessian matrix with the gradient.
    /// If the Hessian is not invertible, it falls back to the negative gradient direction.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using your telescope and map to decide which way to go next. You're looking at both the steepness (gradient)
    /// and the shape (Hessian) of the valley to make the best guess about where the lowest point is. If your calculations get too
    /// complicated, you simply decide to go downhill (like in regular gradient descent).
    /// </para>
    /// </remarks>
    /// <param name="gradient">The gradient vector at the current point.</param>
    /// <param name="hessian">The Hessian matrix at the current point.</param>
    /// <returns>The direction vector for the next step.</returns>
    private Vector<T> CalculateDirection(Vector<T> gradient, Matrix<T> hessian)
    {
        try
        {
            var inverseHessian = hessian.Inverse();
            return inverseHessian.Multiply(gradient).Transform(x => NumOps.Negate(x));
        }
        catch (InvalidOperationException)
        {
            // If Hessian is not invertible, fall back to gradient descent
            return gradient.Transform(x => NumOps.Negate(x));
        }
    }

    /// <summary>
    /// Calculates the Hessian matrix for the current model and input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the Hessian matrix, which represents the second-order partial derivatives of the loss function.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Imagine creating a detailed topographic map of the valley around your current position. You're measuring how the slope changes
    /// in every direction, which gives you a complete picture of the valley's shape at your location.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The Hessian matrix at the current point.</returns>
    private Matrix<T> CalculateHessian(IFullModel<T, TInput, TOutput> model, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        int n = model.GetParameters().Length;
        var hessian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                hessian[i, j] = CalculateSecondPartialDerivative(model, inputData, i, j);
            }
        }

        return hessian;
    }

    /// <summary>
    /// Calculates the second partial derivative of the loss function with respect to two parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method uses finite differences to approximate the second partial derivative.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like measuring how the slope of the valley changes when you take tiny steps in two different directions.
    /// It helps you understand the curvature of the valley at your current position.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="inputData">The input data for optimization.</param>
    /// <param name="i">The index of the first parameter.</param>
    /// <param name="j">The index of the second parameter.</param>
    /// <returns>The approximated second partial derivative.</returns>
    private T CalculateSecondPartialDerivative(IFullModel<T, TInput, TOutput> model, OptimizationInputData<T, TInput, TOutput> inputData, int i, int j)
    {
        var parameters = model.GetParameters();
        var epsilon = NumOps.FromDouble(1e-5);
        var originalI = parameters[i];
        var originalJ = parameters[j];

        // f(x+h, y+h)
        parameters[i] = NumOps.Add(originalI, epsilon);
        parameters[j] = NumOps.Add(originalJ, epsilon);
        var modelPlusPlus = model.WithParameters(parameters);
        var fhh = CalculateLoss(modelPlusPlus, inputData);

        // f(x+h, y-h)
        parameters[j] = NumOps.Subtract(originalJ, epsilon);
        var modelPlusMinus = model.WithParameters(parameters);
        var fhm = CalculateLoss(modelPlusMinus, inputData);

        // f(x-h, y+h)
        parameters[i] = NumOps.Subtract(originalI, epsilon);
        parameters[j] = NumOps.Add(originalJ, epsilon);
        var modelMinusPlus = model.WithParameters(parameters);
        var fmh = CalculateLoss(modelMinusPlus, inputData);

        // f(x-h, y-h)
        parameters[j] = NumOps.Subtract(originalJ, epsilon);
        var modelMinusMinus = model.WithParameters(parameters);
        var fmm = CalculateLoss(modelMinusMinus, inputData);

        // Reset coefficients
        parameters[i] = originalI;
        parameters[j] = originalJ;
        model.WithParameters(parameters);

        // Calculate second partial derivative
        var numerator = NumOps.Subtract(NumOps.Add(fhh, fmm), NumOps.Add(fhm, fmh));
        var denominator = NumOps.Multiply(NumOps.FromDouble(4), NumOps.Multiply(epsilon, epsilon));

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Creates a potential solution based on the optimization mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a new model variant by either selecting features, adjusting parameters,
    /// or both, depending on the optimization mode.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a new version of the solution. Depending on what you're focusing on,
    /// you might change which features you use, how the model parameters are set,
    /// or both aspects at once.
    /// </para>
    /// </remarks>
    /// <param name="xTrain">Training data used to determine data dimensions.</param>
    /// <returns>A new potential solution (model variant).</returns>
    protected override IFullModel<T, TInput, TOutput> CreateSolution(TInput xTrain)
    {
        // Create a deep copy of the model to avoid modifying the original
        var solution = Model.DeepCopy();

        int numFeatures = InputHelper<T, TInput>.GetInputSize(xTrain);

        switch (Options.OptimizationMode)
        {
            case OptimizationMode.FeatureSelectionOnly:
                ApplyFeatureSelection(solution, numFeatures);
                break;

            case OptimizationMode.ParametersOnly:
                AdjustModelParameters(
                    solution,
                    _newtonOptions.ParameterAdjustmentScale,
                    _newtonOptions.SignFlipProbability);
                break;

            case OptimizationMode.Both:
            default:
                // With some probability, apply both or just one type of optimization
                if (Random.NextDouble() < _newtonOptions.FeatureSelectionProbability)
                {
                    ApplyFeatureSelection(solution, numFeatures);
                }

                if (Random.NextDouble() < _newtonOptions.ParameterAdjustmentProbability)
                {
                    AdjustModelParameters(
                        solution,
                        _newtonOptions.ParameterAdjustmentScale,
                        _newtonOptions.SignFlipProbability);
                }
                break;
        }

        return solution;
    }

    /// <summary>
    /// Updates the current solution using the calculated direction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the Newton's Method update rule to the current solution using the calculated direction and learning rate.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a step in the direction you've calculated. The size of your step is determined by the learning rate,
    /// and the direction is based on both the slope and the curvature of the valley at your current position.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution (model parameters).</param>
    /// <param name="direction">The direction to move in the parameter space.</param>
    /// <returns>A new model with updated parameters.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction)
    {
        var parameters = currentSolution.GetParameters();
        var newCoefficients = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            newCoefficients[i] = NumOps.Add(parameters[i], NumOps.Multiply(CurrentLearningRate, direction[i]));
        }

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters of the Newton's Method optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate and other parameters based on the improvement in fitness.
    /// It's used to fine-tune the algorithm's behavior as the optimization progresses.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your exploration strategy as you gather more information about the valley.
    /// If you're making good progress, you might adjust your settings to be more aggressive.
    /// If you're not improving, you might become more cautious and exploratory.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">The current optimization step data.</param>
    /// <param name="previousStepData">The previous optimization step data.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // Call the base implementation to update common parameters
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        bool isImproving = FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore);

        // Adaptive feature selection parameters
        if ((Options.OptimizationMode == OptimizationMode.FeatureSelectionOnly ||
             Options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateFeatureSelectionParameters(isImproving);
        }

        // Adaptive parameter adjustment settings
        if ((Options.OptimizationMode == OptimizationMode.ParametersOnly ||
             Options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateParameterAdjustmentSettings(isImproving);
        }
    }

    /// <summary>
    /// Updates the feature selection parameters based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    private void UpdateFeatureSelectionParameters(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, gradually expand the range of features to consider
            _newtonOptions.MinimumFeatures = Math.Max(1, _newtonOptions.MinimumFeatures - 1);
            _newtonOptions.MaximumFeatures = Math.Min(_newtonOptions.MaximumFeatures + 1, _newtonOptions.AbsoluteMaximumFeatures);

            // Slightly increase the probability of feature selection for future iterations
            _newtonOptions.FeatureSelectionProbability *= 1.02;
        }
        else
        {
            // If not improving, narrow the range to focus the search
            _newtonOptions.MinimumFeatures = Math.Min(_newtonOptions.MinimumFeatures + 1, _newtonOptions.AbsoluteMaximumFeatures - 1);
            _newtonOptions.MaximumFeatures = Math.Max(_newtonOptions.MaximumFeatures - 1, _newtonOptions.MinimumFeatures + 1);

            // Slightly decrease the probability of feature selection for future iterations
            _newtonOptions.FeatureSelectionProbability *= 0.98;
        }

        // Ensure probabilities stay within bounds
        _newtonOptions.FeatureSelectionProbability = MathHelper.Clamp(
            _newtonOptions.FeatureSelectionProbability,
            _newtonOptions.MinFeatureSelectionProbability,
            _newtonOptions.MaxFeatureSelectionProbability);
    }

    /// <summary>
    /// Updates the parameter adjustment settings based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    private void UpdateParameterAdjustmentSettings(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, make smaller adjustments to fine-tune
            _newtonOptions.ParameterAdjustmentScale *= 0.95;

            // Decrease the probability of sign flips when things are going well
            _newtonOptions.SignFlipProbability *= 0.9;

            // Increase the probability of parameter adjustments
            _newtonOptions.ParameterAdjustmentProbability *= 1.02;

            // Adjust learning rate
            if (_newtonOptions.UseAdaptiveLearningRate)
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_newtonOptions.LearningRateIncreaseFactor));
            }
        }
        else
        {
            // If not improving, make larger adjustments to explore more
            _newtonOptions.ParameterAdjustmentScale *= 1.05;

            // Increase the probability of sign flips to try more dramatic changes
            _newtonOptions.SignFlipProbability *= 1.1;

            // Slightly decrease the probability of parameter adjustments
            _newtonOptions.ParameterAdjustmentProbability *= 0.98;

            // Adjust learning rate
            if (_newtonOptions.UseAdaptiveLearningRate)
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_newtonOptions.LearningRateDecreaseFactor));
            }
        }

        // Ensure values stay within bounds
        _newtonOptions.ParameterAdjustmentScale = MathHelper.Clamp(
            _newtonOptions.ParameterAdjustmentScale,
            _newtonOptions.MinParameterAdjustmentScale,
            _newtonOptions.MaxParameterAdjustmentScale);

        _newtonOptions.SignFlipProbability = MathHelper.Clamp(
            _newtonOptions.SignFlipProbability,
            _newtonOptions.MinSignFlipProbability,
            _newtonOptions.MaxSignFlipProbability);

        _newtonOptions.ParameterAdjustmentProbability = MathHelper.Clamp(
            _newtonOptions.ParameterAdjustmentProbability,
            _newtonOptions.MinParameterAdjustmentProbability,
            _newtonOptions.MaxParameterAdjustmentProbability);

        if (_newtonOptions.UseAdaptiveLearningRate)
        {
            CurrentLearningRate = MathHelper.Clamp(
                CurrentLearningRate,
                NumOps.FromDouble(_newtonOptions.MinLearningRate),
                NumOps.FromDouble(_newtonOptions.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method ensures that only compatible option types are used with this optimizer.
    /// It updates the internal options if the provided options are of the correct type.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the rules for how you navigate the valley. It makes sure you're only using rules that work for
    /// Newton's Method of exploring the valley.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NewtonMethodOptimizerOptions<T, TInput, TOutput> newtonOptions)
        {
            _newtonOptions = newtonOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NewtonMethodOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current settings used by the Newton's Method optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current rulebook for exploring the valley. It tells you what settings and strategies
    /// you're currently using in your search for the lowest point.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _newtonOptions;
    }

    /// <summary>
    /// Serializes the current state of the optimizer into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the optimizer, including its base class state, options, and iteration count.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of your current position, all your tools, and your strategy for exploring the valley.
    /// You can use this snapshot later to continue your exploration from exactly where you left off.
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

            // Serialize optimization mode
            writer.Write((int)Options.OptimizationMode);

            // Serialize NewtonMethodOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_newtonOptions);
            writer.Write(optionsJson);

            // Serialize iteration count
            writer.Write(_iteration);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method restores the optimizer's state from a previously serialized byte array, including its base class state, options, and iteration count.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a snapshot you took earlier to set up your exploration exactly as it was at that point.
    /// You're restoring all your tools, your position in the valley, and your strategy to continue your search from where you left off.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer options cannot be deserialized.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize optimization mode
            Options.OptimizationMode = (OptimizationMode)reader.ReadInt32();

            // Deserialize NewtonMethodOptimizerOptions
            string optionsJson = reader.ReadString();
            _newtonOptions = JsonConvert.DeserializeObject<NewtonMethodOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize iteration count
            _iteration = reader.ReadInt32();
        }
    }
}