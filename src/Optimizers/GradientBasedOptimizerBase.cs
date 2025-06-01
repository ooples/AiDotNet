namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a base class for gradient-based optimization algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Gradient-based optimizers use the gradient of the loss function to update the model parameters
/// in a direction that minimizes the loss. This base class provides common functionality for
/// various gradient-based optimization techniques.
/// </para>
/// <para><b>For Beginners:</b> Think of gradient-based optimization like finding the bottom of a valley:
/// 
/// - You start at a random point on a hilly landscape (your initial model parameters)
/// - You look around to see which way is steepest downhill (calculate the gradient)
/// - You take a step in that direction (update the parameters)
/// - You repeat this process until you reach the bottom of the valley (optimize the model)
/// 
/// This approach helps the model learn by gradually adjusting its parameters to minimize errors.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public abstract class GradientBasedOptimizerBase<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>, IGradientBasedOptimizer<T, TInput, TOutput>
{
    /// <summary>
    /// Options specific to gradient-based optimization algorithms.
    /// </summary>
    protected GradientBasedOptimizerOptions<T, TInput, TOutput> GradientOptions;

    /// <summary>
    /// The current learning rate used in the optimization process.
    /// </summary>
    private double _currentLearningRate;

    /// <summary>
    /// The current momentum factor used in the optimization process.
    /// </summary>
    private double _currentMomentum;

    /// <summary>
    /// The gradient from the previous optimization step, used for momentum calculations.
    /// </summary>
    protected Vector<T> _previousGradient;

    /// <summary>
    /// A cache for storing and retrieving gradients to improve performance.
    /// </summary>
    protected IGradientCache<T> GradientCache;

    /// <summary>
    /// A method used to compare the predicted values vs the actual values.
    /// </summary>
    protected ILossFunction<T> LossFunction;

    /// <summary>
    /// A method used to regularize the parameters so they don't get out of control.
    /// </summary>
    protected IRegularization<T, TInput, TOutput> Regularization;

    /// <summary>
    /// Initializes a new instance of the GradientBasedOptimizerBase class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the gradient-based optimizer with its initial settings.
    /// It's like preparing for your hike by choosing your starting point, deciding how big your steps
    /// will be, and how much you'll consider your previous direction when choosing your next step.
    /// </para>
    /// </remarks>
    /// <param name="model"> The model to be optimized.</param>
    /// <param name="options">Options for the gradient-based optimizer.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    protected GradientBasedOptimizerBase(IFullModel<T, TInput, TOutput> model,
        GradientBasedOptimizerOptions<T, TInput, TOutput> options) : 
        base(model, options)
    {
        GradientOptions = options;
        _currentLearningRate = GradientOptions.InitialLearningRate;
        _currentMomentum = GradientOptions.InitialMomentum;
        _previousGradient = Vector<T>.Empty();
        LossFunction = options.LossFunction;
        GradientCache = options.GradientCache;
        Regularization = options.Regularization;
    }

    /// <summary>
    /// Creates a regularization technique based on the provided options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up a way to prevent the model from becoming too complex.
    /// It's like adding rules to your hiking strategy to avoid taking unnecessarily complicated paths.
    /// </para>
    /// </remarks>
    /// <param name="options">The options specifying the regularization technique to use.</param>
    /// <returns>An instance of the specified regularization technique.</returns>
    protected IRegularization<T, TInput, TOutput> CreateRegularization(GradientDescentOptimizerOptions<T, TInput, TOutput> options)
    {
        return RegularizationFactory.CreateRegularization<T, TInput, TOutput>(options.RegularizationOptions);
    }

    /// <summary>
    /// Calculates the gradient for the given model and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how steep the hill is and in which direction.
    /// It helps determine which way the optimizer should step to improve the model.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated gradient.</returns>
    /// <summary>
    /// Calculates the gradient for the given solution and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how steep the hill is and in which direction.
    /// It helps determine which way the optimizer should step to improve the model.
    /// This implementation uses the loss function's derivative for efficient gradient calculation.
    /// </para>
    /// </remarks>
    /// <param name="solution">The current solution.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated gradient.</returns>
    protected virtual Vector<T> CalculateGradient(
        IFullModel<T, TInput, TOutput> solution,
        TInput X,
        TOutput y)
    {
        string cacheKey = GenerateGradientCacheKey(solution, X, y);
        var cachedGradient = GradientCache.GetCachedGradient(cacheKey);
        if (cachedGradient != null)
        {
            return cachedGradient.Parameters;
        }

        try
        {
            // Get predictions and parameters
            TOutput predictions = solution.Predict(X);
            Vector<T> parameters = solution.GetParameters();
            int paramCount = parameters.Length;

            // Calculate prediction-level gradient (one per sample)
            Vector<T> predictionGradient;
            if (predictions is Tensor<T> tensorPredictions && y is Tensor<T> tensorY)
            {
                predictionGradient = LossFunction.CalculateDerivative(tensorPredictions.ToVector(), tensorY.ToVector());
            }
            else if (predictions is Vector<T> vectorPredictions && y is Vector<T> vectorY)
            {
                predictionGradient = LossFunction.CalculateDerivative(vectorPredictions, vectorY);
            }
            else
            {
                throw new ArgumentException("Unsupported prediction or target type");
            }

            // Transform prediction-level gradient to parameter-level gradient
            Vector<T> parameterGradient;

            // If the model implements the gradient transformation interface
            if (solution is IGradientTransformable<T, TInput, TOutput> gradientModel)
            {
                // Let the model handle the transformation
                parameterGradient = gradientModel.TransformGradient(X, predictionGradient);
            }
            // For models that don't implement it, use a fallback approach
            else
            {
                // Create a parameter gradient with same length as parameters
                parameterGradient = new Vector<T>(paramCount);

                // Basic approach: calculate weighted average of prediction gradients
                for (int i = 0; i < paramCount; i++)
                {
                    T weightedSum = NumOps.Zero;

                    for (int j = 0; j < predictionGradient.Length; j++)
                    {
                        // Simple weighting - specific models should implement IGradientTransformable
                        T weight = NumOps.Divide(NumOps.One, NumOps.FromDouble(predictionGradient.Length));
                        weightedSum = NumOps.Add(weightedSum, NumOps.Multiply(predictionGradient[j], weight));
                    }

                    parameterGradient[i] = weightedSum;
                }
            }

            // Apply regularization to the parameter-level gradient
            Vector<T> regularizationGradient = Regularization.Regularize(parameters);

            // Ensure regularization gradient has the correct size
            if (regularizationGradient.Length != parameterGradient.Length)
            {
                var resizedRegGrad = new Vector<T>(parameterGradient.Length);
                for (int i = 0; i < Math.Min(regularizationGradient.Length, parameterGradient.Length); i++)
                {
                    resizedRegGrad[i] = regularizationGradient[i];
                }

                regularizationGradient = resizedRegGrad;
            }

            // Add regularization
            parameterGradient = parameterGradient.Add(regularizationGradient);

            // Scale by batch size
            int batchSize = InputHelper<T, TInput>.GetBatchSize(X);
            if (batchSize > 0) // Avoid division by zero
            {
                parameterGradient = parameterGradient.Divide(NumOps.FromDouble(batchSize));
            }

            // Cache and return
            var gradientModelResult = new GradientModel<T>(parameterGradient);
            GradientCache.CacheGradient(cacheKey, gradientModelResult);

            return parameterGradient;
        }
        catch (Exception)
        {
            // Return a zero gradient with appropriate size
            int paramCount = 10; // Default size
            try { paramCount = solution.GetParameters().Length; } catch { /* Use default size */ }

            var zeroGradient = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                zeroGradient[i] = NumOps.Zero;
            }

            return zeroGradient;
        }
    }

    /// <summary>
    /// Performs a line search to find an appropriate step size.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="direction">The search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The step size to use.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines how big of a step to take in the chosen direction.
    /// It tries to find a step size that sufficiently decreases the function value while not being too small.
    /// </para>
    /// </remarks>
    protected T LineSearch(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var alpha = NumOps.FromDouble(1.0);
        var c1 = NumOps.FromDouble(1e-4);
        var c2 = NumOps.FromDouble(0.9);
        var xTrain = inputData.XTrain;
        var yTrain = inputData.YTrain;

        var initialValue = CalculateLoss(currentSolution, inputData);
        var initialSlope = gradient.DotProduct(direction);

        while (true)
        {
            var newCoefficients = currentSolution.GetParameters().Add(direction.Multiply(alpha));
            var newSolution = currentSolution.WithParameters(newCoefficients);
            var newValue = CalculateLoss(newSolution, inputData);

            if (NumOps.LessThanOrEquals(newValue, NumOps.Add(initialValue, NumOps.Multiply(NumOps.Multiply(c1, alpha), initialSlope))))
            {
                var newGradient = CalculateGradient(newSolution, xTrain, yTrain);
                var newSlope = newGradient.DotProduct(direction);

                if (NumOps.GreaterThanOrEquals(NumOps.Abs(newSlope), NumOps.Multiply(c2, NumOps.Abs(initialSlope))))
                {
                    return alpha;
                }
            }

            alpha = NumOps.Multiply(alpha, NumOps.FromDouble(0.5));

            if (NumOps.LessThan(alpha, NumOps.FromDouble(1e-10)))
            {
                return NumOps.FromDouble(1e-10);
            }
        }
    }


    /// <summary>
    /// Calculates the gradient for a given solution using a batch of training data.
    /// </summary>
    /// <param name="solution">The current solution (model).</param>
    /// <param name="xTrain">The training input data.</param>
    /// <param name="yTrain">The training target data.</param>
    /// <param name="batchIndices">The indices to use for the current batch.</param>
    /// <returns>A vector representing the gradient of the loss function with respect to the model parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The gradient tells us which direction to adjust our model's
    /// parameters to improve performance. It's like a compass showing the way to a better solution.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> CalculateGradient(
        IFullModel<T, TInput, TOutput> solution, 
        TInput xTrain, 
        TOutput yTrain, 
        int[] batchIndices)
    {
        // Extract batch data using your InputHelper
        var xBatch = InputHelper<T, TInput>.GetBatch(xTrain, batchIndices);
        var yBatch = InputHelper<T, TOutput>.GetBatch(yTrain, batchIndices);
    
        // Get the current parameters
        var parameters = solution.GetParameters();
        var gradient = new Vector<T>(parameters.Length);
    
        // Initialize gradient vector with zeros
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = NumOps.Zero;
        }
    
        // For each sample in the batch
        for (int i = 0; i < batchIndices.Length; i++)
        {
            // Get the current input and output for this sample
            var input = InputHelper<T, TInput>.GetItem(xBatch, i);
            var target = InputHelper<T, TOutput>.GetItem(yBatch, i);
        
            // Predict the output
            var prediction = solution.Predict(input);
        
            // Calculate the error
            var error = LossFunction.CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(prediction), ConversionsHelper.ConvertToVector<T, TOutput>(target));
        
            // Update gradient based on the error
            for (int j = 0; j < parameters.Length; j++)
            {
                var featureValue = InputHelper<T, TInput>.GetFeatureValue(input, j);
                var contribution = NumOps.Multiply(error, featureValue);
                gradient[j] = NumOps.Add(gradient[j], contribution);
            }
        }
    
        // Average the gradient
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = NumOps.Divide(gradient[i], NumOps.FromDouble(batchIndices.Length));
        }
    
        return gradient;
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>A new solution with updated parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method moves the model's parameters in the direction
    /// indicated by the gradient, hopefully improving the model's performance.
    /// </para>
    /// </remarks>
    protected virtual IFullModel<T, TInput, TOutput> UpdateSolution(
        IFullModel<T, TInput, TOutput> currentSolution, 
        Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();
        var newParameters = UpdateParameters(parameters, gradient);

        return currentSolution.WithParameters(newParameters);
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for each gradient calculation.
    /// It's like labeling each spot on the hill so you can remember what the gradient was there.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>A string key for caching the gradient.</returns>
    protected virtual string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        return $"{model.GetType().Name}_{InputHelper<T, TInput>.GetBatchSize(X)}_{InputHelper<T, TInput>.GetInputSize(X)}_{GradientOptions.GetType().Name}";
    }

    /// <summary>
    /// Resets the optimizer to its initial state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method clears all the remembered information and starts fresh.
    /// It's like wiping your map clean and starting your hike from the beginning.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();
        GradientCache.ClearCache();
    }

    /// <summary>
    /// Applies momentum to the gradient calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method considers the direction you were moving in previously
    /// when deciding which way to go next. It's like considering your momentum when hiking -
    /// you might keep going in roughly the same direction rather than abruptly changing course.
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The gradient adjusted for momentum.</returns>
    protected virtual Vector<T> ApplyMomentum(Vector<T> gradient)
    {
        if (_previousGradient == null)
        {
            _previousGradient = gradient;
            return gradient;
        }

        var momentumGradient = _previousGradient.Add(gradient.Multiply(NumOps.FromDouble(_currentMomentum)));
        _previousGradient = momentumGradient;
        return momentumGradient;
    }

    /// <summary>
    /// Updates the parameters of the model based on the calculated gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters to improve its performance.
    /// It's like taking steps in the direction that will lead to better results, based on what we've learned
    /// from the data.</para>
    /// </remarks>
    /// <param name="layers">The layers of the neural network containing the parameters to update.</param>
    public virtual void UpdateParameters(List<ILayer<T>> layers)
    {
        foreach (var layer in layers)
        {
            if (layer.SupportsTraining)
            {
                Vector<T> parameters = layer.GetParameters();
                Vector<T> gradients = layer.GetParameterGradients();

                // Apply simple gradient descent update
                Vector<T> newParameters = new Vector<T>(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    T update = NumOps.Multiply(gradients[i], NumOps.FromDouble(_currentLearningRate));
                    newParameters[i] = NumOps.Subtract(parameters[i], update);
                }

                layer.SetParameters(newParameters);
                layer.ClearGradients();
            }
        }
    }

    /// <summary>
    /// Updates a matrix of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters to improve its performance.
    /// It's like taking a step in the direction you've determined will lead you downhill.
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated parameters.</returns>
    public virtual Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    /// <summary>
    /// Updates a tensor of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters stored in tensor format to improve its performance.
    /// It's like taking a step in the direction you've determined will lead you downhill, but for more complex
    /// multi-dimensional data structures. Tensor<double>s are useful for representing parameters in deep neural networks
    /// where data has multiple dimensions (like images with width, height, and channels).
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current tensor parameters.</param>
    /// <param name="gradient">The calculated gradient tensor.</param>
    /// <returns>The updated tensor parameters.</returns>
    public virtual Tensor<T> UpdateParameters(Tensor<T> parameters, Tensor<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);

        // Scale the gradient by the learning rate
        var scaledGradient = gradient.Multiply(learningRate);

        // Subtract the scaled gradient from the parameters
        return parameters.Subtract(scaledGradient);
    }

    /// <summary>
    /// Updates a vector of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is similar to UpdateMatrix, but for when the parameters
    /// are in a vector format instead of a matrix. It's another way of taking a step to improve the model.
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated parameters.</returns>
    public virtual Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    /// <summary>
    /// Updates the options for the gradient-based optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer
    /// while it's running. It's like adjusting your hiking strategy mid-journey based on the terrain you encounter.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is GradientBasedOptimizerOptions<T, TInput, TOutput> gradientOptions)
        {
            GradientOptions = gradientOptions;
        }
    }
}