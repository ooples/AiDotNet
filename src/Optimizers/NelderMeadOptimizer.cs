using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Nelder-Mead optimization algorithm, also known as the downhill simplex method.
/// </summary>
/// <remarks>
/// <para>
/// The Nelder-Mead method is a heuristic search method that can optimize a problem with N variables.
/// It attempts to minimize a scalar-valued nonlinear function of n real variables using only function values,
/// without any derivative information.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're trying to find the lowest point in a hilly landscape. The Nelder-Mead method is like
/// having a group of explorers who work together, moving and reshaping their search pattern to find the lowest point.
/// They don't need to know which way is downhill; they just compare their positions and adjust accordingly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NelderMeadOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Nelder-Mead optimizer.
    /// </summary>
    private NelderMeadOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current iteration count.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// The reflection coefficient.
    /// </summary>
    private T _alpha;

    /// <summary>
    /// The contraction coefficient.
    /// </summary>
    private T _beta;

    /// <summary>
    /// The expansion coefficient.
    /// </summary>
    private T _gamma;

    /// <summary>
    /// The shrinkage coefficient.
    /// </summary>
    private T _delta;

    /// <summary>
    /// Initializes a new instance of the NelderMeadOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Nelder-Mead optimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your team of explorers before they start searching the landscape.
    /// You're giving them their initial instructions and tools.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The Nelder-Mead-specific optimization options.</param>
    public NelderMeadOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NelderMeadOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new NelderMeadOptimizerOptions<T, TInput, TOutput>();
        _alpha = NumOps.Zero;
        _beta = NumOps.Zero;
        _gamma = NumOps.Zero;
        _delta = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Nelder-Mead optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the reflection, contraction, expansion, and shrinkage coefficients.
    /// It also resets the iteration counter.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like giving your explorers their initial strategies for how to move around the landscape.
    /// You're setting up how far they should reflect, contract, expand, or shrink their search pattern.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _alpha = NumOps.FromDouble(_options.InitialAlpha);
        _beta = NumOps.FromDouble(_options.InitialBeta);
        _gamma = NumOps.FromDouble(_options.InitialGamma);
        _delta = NumOps.FromDouble(_options.InitialDelta);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the optimization process using the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It creates and manipulates a simplex
    /// (a geometric figure in N dimensions) to find the optimal solution.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual search process. Your team of explorers starts at different points,
    /// then repeatedly adjusts their positions based on which points are higher or lower.
    /// They reflect away from high points, expand towards promising areas, contract if they overshoot,
    /// and shrink their search area if they get stuck.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);
        var n = InputHelper<T, TInput>.GetInputSize(inputData.XTrain);
        var simplex = InitializeSimplex(inputData.XTrain, n);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;

            // Sort simplex points by fitness
            simplex = [.. simplex.OrderBy(p => EvaluateSolution(p, inputData).FitnessScore)];

            var centroid = CalculateCentroid(simplex, n);
            var reflected = Reflect(simplex[n], centroid);
            var reflectedStepData = EvaluateSolution(reflected, inputData);

            if (NumOps.GreaterThan(reflectedStepData.FitnessScore, EvaluateSolution(simplex[0], inputData).FitnessScore) &&
                NumOps.LessThan(reflectedStepData.FitnessScore, EvaluateSolution(simplex[n - 1], inputData).FitnessScore))
            {
                simplex[n] = reflected;
            }
            else if (NumOps.GreaterThan(reflectedStepData.FitnessScore, EvaluateSolution(simplex[0], inputData).FitnessScore))
            {
                var expanded = Expand(centroid, reflected);
                var expandedStepData = EvaluateSolution(expanded, inputData);

                simplex[n] = NumOps.GreaterThan(expandedStepData.FitnessScore, reflectedStepData.FitnessScore) ? expanded : reflected;
            }
            else
            {
                var contracted = Contract(simplex[n], centroid);
                var contractedStepData = EvaluateSolution(contracted, inputData);

                if (NumOps.GreaterThan(contractedStepData.FitnessScore, EvaluateSolution(simplex[n], inputData).FitnessScore))
                {
                    simplex[n] = contracted;
                }
                else
                {
                    Shrink(simplex);
                }
            }

            var currentStepData = EvaluateSolution(simplex[0], inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Initializes the simplex for the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates the initial set of points (simplex) for the optimization process.
    /// It generates n+1 random solutions, where n is the number of dimensions.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like choosing the starting positions for your team of explorers on the landscape.
    /// You're placing them at different random spots to begin their search.
    /// </para>
    /// </remarks>
    /// <param name="n">The number of dimensions (variables) in the optimization problem.</param>
    /// <returns>A list of initial solutions forming the simplex.</returns>
    private List<IFullModel<T, TInput, TOutput>> InitializeSimplex(TInput input, int n)
    {
        var simplex = new List<IFullModel<T, TInput, TOutput>>();
        for (int i = 0; i <= n; i++)
        {
            simplex.Add(InitializeRandomSolution(input));
        }

        return simplex;
    }

    /// <summary>
    /// Calculates the centroid of the simplex, excluding the worst point.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the average position of all points in the simplex except the worst one.
    /// The centroid is used as a reference point for reflection, expansion, and contraction operations.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Imagine your explorers (except the one at the highest point) all throwing ropes to each other and pulling until they meet at a central point.
    /// This central point is the centroid, which helps guide the next move.
    /// </para>
    /// </remarks>
    /// <param name="simplex">The current simplex (list of solution points).</param>
    /// <param name="n">The number of dimensions (variables) in the optimization problem.</param>
    /// <returns>A symbolic model representing the centroid of the simplex.</returns>
    private IFullModel<T, TInput, TOutput> CalculateCentroid(List<IFullModel<T, TInput, TOutput>> simplex, int n)
    {
        // === Vectorized Centroid Calculation using IEngine (Phase B: US-GPU-015) ===
        // centroid = sum(simplex points excluding worst) / (simplex.Count - 1)

        var templateModel = simplex[0];
        var templateParams = templateModel.GetParameters();
        var centroidCoefficients = new Vector<T>(templateParams.Length);

        // Sum all parameters vectorized (exclude worst point: last simplex entry)
        int pointCount = simplex.Count - 1;
        for (int i = 0; i < pointCount; i++)
        {
            centroidCoefficients = (Vector<T>)Engine.Add(centroidCoefficients, simplex[i].GetParameters());
        }

        // Vectorized division by number of points to get the average
        var pointCountScalar = NumOps.FromDouble(pointCount);
        centroidCoefficients = (Vector<T>)Engine.Divide(centroidCoefficients, pointCountScalar);

        // Create a new model with the centroid parameters using WithParameters
        return templateModel.WithParameters(centroidCoefficients);
    }

    /// <summary>
    /// Performs the reflection operation in the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method reflects the worst point through the centroid of the remaining points.
    /// It's used to move away from the worst solution in the simplex.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Imagine the worst explorer jumping over the center point, landing on the opposite side.
    /// This helps the team move away from bad areas.
    /// </para>
    /// </remarks>
    /// <param name="worst">The worst point in the simplex.</param>
    /// <param name="centroid">The centroid of the simplex (excluding the worst point).</param>
    /// <returns>A new point that is the reflection of the worst point through the centroid.</returns>
    private IFullModel<T, TInput, TOutput> Reflect(IFullModel<T, TInput, TOutput> worst, IFullModel<T, TInput, TOutput> centroid)
    {
        // Vectorized: result = centroid + alpha * (centroid - worst)
        var centroidParams = centroid.GetParameters();
        var worstParams = worst.GetParameters();
        var alphaVec = AiDotNetEngine.Current.Fill<T>(centroidParams.Length, _alpha);

        var diff = (Vector<T>)AiDotNetEngine.Current.Subtract(centroidParams, worstParams);
        var scaled = (Vector<T>)AiDotNetEngine.Current.Multiply(diff, alphaVec);
        var result = (Vector<T>)AiDotNetEngine.Current.Add(centroidParams, scaled);

        return centroid.WithParameters(result);
    }

    /// <summary>
    /// Performs the expansion operation in the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method expands the reflected point further away from the centroid.
    /// It's used when the reflected point is the best so far, to explore even further in that direction.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// If the reflected explorer found a good spot, this is like telling them to keep going even further in that direction.
    /// </para>
    /// </remarks>
    /// <param name="centroid">The centroid of the simplex.</param>
    /// <param name="reflected">The reflected point.</param>
    /// <returns>A new point that is an expansion of the reflected point away from the centroid.</returns>
    private IFullModel<T, TInput, TOutput> Expand(IFullModel<T, TInput, TOutput> centroid, IFullModel<T, TInput, TOutput> reflected)
    {
        return PerformVectorOperation(centroid, reflected, _gamma, (a, b, c) => NumOps.Add(a, NumOps.Multiply(c, NumOps.Subtract(b, a))));
    }

    /// <summary>
    /// Performs the contraction operation in the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method contracts the worst point towards the centroid.
    /// It's used when the reflected point isn't better than the second worst point.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// If the reflected explorer didn't find a good spot, this is like calling them back halfway towards the group.
    /// </para>
    /// </remarks>
    /// <param name="worst">The worst point in the simplex.</param>
    /// <param name="centroid">The centroid of the simplex.</param>
    /// <returns>A new point that is a contraction of the worst point towards the centroid.</returns>
    private IFullModel<T, TInput, TOutput> Contract(IFullModel<T, TInput, TOutput> worst, IFullModel<T, TInput, TOutput> centroid)
    {
        // Vectorized: result = centroid + beta * (worst - centroid)
        var centroidParams = centroid.GetParameters();
        var worstParams = worst.GetParameters();
        var betaVec = AiDotNetEngine.Current.Fill<T>(centroidParams.Length, _beta);

        var diff = (Vector<T>)AiDotNetEngine.Current.Subtract(worstParams, centroidParams);
        var scaled = (Vector<T>)AiDotNetEngine.Current.Multiply(diff, betaVec);
        var result = (Vector<T>)AiDotNetEngine.Current.Add(centroidParams, scaled);

        return centroid.WithParameters(result);
    }

    /// <summary>
    /// Performs the shrink operation in the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method shrinks all points (except the best) towards the best point.
    /// It's used when contraction fails to produce a better point.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// If none of the other moves worked well, this is like telling all explorers (except the best one) to move halfway towards the best explorer's position.
    /// </para>
    /// </remarks>
    /// <param name="simplex">The current simplex (list of solution points).</param>
    private void Shrink(List<IFullModel<T, TInput, TOutput>> simplex)
    {
        var best = simplex[0];
        var bestParams = best.GetParameters();
        var deltaVec = AiDotNetEngine.Current.Fill<T>(bestParams.Length, _delta);

        for (int i = 1; i < simplex.Count; i++)
        {
            // Vectorized: result = best + delta * (simplex[i] - best)
            var currentParams = simplex[i].GetParameters();
            var diff = (Vector<T>)AiDotNetEngine.Current.Subtract(currentParams, bestParams);
            var scaled = (Vector<T>)AiDotNetEngine.Current.Multiply(diff, deltaVec);
            var result = (Vector<T>)AiDotNetEngine.Current.Add(bestParams, scaled);

            simplex[i] = best.WithParameters(result);
        }
    }

    /// <summary>
    /// Performs a vector operation on two symbolic models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies a specified operation to each corresponding pair of coefficients in two symbolic models.
    /// It's used as a helper method for reflection, expansion, contraction, and shrinkage operations.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like having a rule for how to combine the positions of two explorers to find a new position.
    /// </para>
    /// </remarks>
    /// <param name="a">The first symbolic model.</param>
    /// <param name="b">The second symbolic model.</param>
    /// <param name="factor">A factor used in the operation.</param>
    /// <param name="operation">The operation to perform on each pair of coefficients.</param>
    /// <returns>A new symbolic model resulting from the vector operation.</returns>
    private IFullModel<T, TInput, TOutput> PerformVectorOperation(IFullModel<T, TInput, TOutput> a, IFullModel<T, TInput, TOutput> b, T factor, Func<T, T, T, T> operation)
    {
        // === Vectorized Vector Operation using IEngine (Phase B: US-GPU-015) ===
        // Most Nelder-Mead operations follow pattern: a + factor * (a - b) or a + factor * (b - a)
        // Vectorizing the common pattern while keeping backward compatibility

        var parametersA = a.GetParameters();
        var parametersB = b.GetParameters();

        // Check if this is the common pattern: a + factor * (a - b)
        // Validate across the whole vector to avoid degenerate mis-detections
        bool matchesPattern1 = parametersA.Length > 0;
        for (int i = 0; i < parametersA.Length && matchesPattern1; i++)
        {
            var testResult = operation(parametersA[i], parametersB[i], factor);
            var expectedPattern1 = NumOps.Add(
                parametersA[i],
                NumOps.Multiply(factor, NumOps.Subtract(parametersA[i], parametersB[i])));

            if (!NumOps.Equals(testResult, expectedPattern1))
            {
                matchesPattern1 = false;
            }
        }

        if (matchesPattern1)
        {
            // Vectorized: result = a + factor * (a - b)
            var diff = (Vector<T>)Engine.Subtract(parametersA, parametersB);
            var scaled = (Vector<T>)Engine.Multiply(diff, factor);
            var newCoefficients = (Vector<T>)Engine.Add(parametersA, scaled);
            return a.WithParameters(newCoefficients);
        }

        // Fall back to element-wise for custom operations
        var result = new Vector<T>(parametersA.Length);
        for (int i = 0; i < parametersA.Length; i++)
        {
            result[i] = operation(parametersA[i], parametersB[i], factor);
        }

        return a.WithParameters(result);
    }

    /// <summary>
    /// Updates the adaptive parameters of the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the reflection, expansion, contraction, and shrinkage coefficients based on the improvement in fitness.
    /// It's used to fine-tune the algorithm's behavior as the optimization progresses.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting how far the explorers move based on how well they're doing. If they're finding better spots, they might move more boldly.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">The current optimization step data.</param>
    /// <param name="previousStepData">The previous optimization step data.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveParameters)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            _alpha = NumOps.Add(_alpha, NumOps.Multiply(adaptationRate, improvement));
            _beta = NumOps.Add(_beta, NumOps.Multiply(adaptationRate, improvement));
            _gamma = NumOps.Add(_gamma, NumOps.Multiply(adaptationRate, improvement));
            _delta = NumOps.Add(_delta, NumOps.Multiply(adaptationRate, improvement));

            _alpha = MathHelper.Clamp(_alpha, NumOps.FromDouble(_options.MinAlpha), NumOps.FromDouble(_options.MaxAlpha));
            _beta = MathHelper.Clamp(_beta, NumOps.FromDouble(_options.MinBeta), NumOps.FromDouble(_options.MaxBeta));
            _gamma = MathHelper.Clamp(_gamma, NumOps.FromDouble(_options.MinGamma), NumOps.FromDouble(_options.MaxGamma));
            _delta = MathHelper.Clamp(_delta, NumOps.FromDouble(_options.MinDelta), NumOps.FromDouble(_options.MaxDelta));
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
    /// This is like changing the rules for how the explorers should search. It makes sure you're only using rules that work for this specific type of search (Nelder-Mead method).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NelderMeadOptimizerOptions<T, TInput, TOutput> nmOptions)
        {
            _options = nmOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NelderMeadOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the Nelder-Mead optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like asking to see the current set of rules the explorers are following in their search.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the Nelder-Mead optimizer to a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its options and parameters, into a byte array.
    /// This allows the optimizer's state to be saved or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of the entire search process, including where all the explorers are and what rules they're following, so you can save it or send it to someone else.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_iteration);
            writer.Write(Convert.ToDouble(_alpha));
            writer.Write(Convert.ToDouble(_beta));
            writer.Write(Convert.ToDouble(_gamma));
            writer.Write(Convert.ToDouble(_delta));

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the Nelder-Mead optimizer from a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method reconstructs the optimizer's state from a byte array, including its options and parameters.
    /// It's used to restore a previously saved or transmitted optimizer state.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a saved snapshot to set up the search process exactly as it was before, placing all the explorers back where they were and restoring the rules they were following.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer options cannot be deserialized.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<NelderMeadOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _alpha = NumOps.FromDouble(reader.ReadDouble());
            _beta = NumOps.FromDouble(reader.ReadDouble());
            _gamma = NumOps.FromDouble(reader.ReadDouble());
            _delta = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}
