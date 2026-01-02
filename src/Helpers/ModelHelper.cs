namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for model-related operations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This helper class contains methods for creating and working with different
/// types of machine learning models. It makes it easier to initialize models, handle different data types,
/// and perform common operations needed when working with models.</para>
/// </remarks>
public static class ModelHelper<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the thread-safe random number generator for creating randomized models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is used to generate random values when creating models.
    /// Uses the centralized RandomHelper for thread safety and consistent randomness.</para>
    /// </remarks>
    private static Random _random => RandomHelper.ThreadSafeRandom;

    /// <summary>
    /// Numeric operations provider for type T.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides mathematical operations for the specific
    /// number type being used (like float or double). It allows the helper to work with 
    /// different numeric types without changing the code.</para>
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates default empty model data for initialization purposes.
    /// </summary>
    /// <returns>A tuple containing default empty X, Y, and Predictions data structures.</returns>
    /// <exception cref="InvalidOperationException">Thrown when TInput and TOutput are unsupported types.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates empty placeholders for your data and predictions.
    /// Think of it like creating empty containers that you'll fill later with your actual data.
    /// This is useful when you need to set up a model structure before you have the real data.</para>
    /// </remarks>
    public static (TInput X, TOutput Y, TOutput Predictions) CreateDefaultModelData()
    {
        if (typeof(TInput) == typeof(Matrix<T>) && typeof(TOutput) == typeof(Vector<T>))
        {
            var x = (TInput)(object)Matrix<T>.Empty();
            var y = (TOutput)(object)Vector<T>.Empty();
            var predictions = (TOutput)(object)Vector<T>.Empty();

            return (x, y, predictions);
        }
        else if (typeof(TInput) == typeof(Tensor<T>) && typeof(TOutput) == typeof(Tensor<T>))
        {
            var x = (TInput)(object)Tensor<T>.Empty();
            var y = (TOutput)(object)Tensor<T>.Empty();
            var predictions = (TOutput)(object)Tensor<T>.Empty();

            return (x, y, predictions);
        }
        else if (typeof(TInput) == typeof(Vector<T>) && typeof(TOutput) == typeof(Vector<T>))
        {
            // Note: The intermediate cast to object is required for generic type conversion in C#.
            // We can't directly cast Vector<T> to TInput even when we know they're the same type at runtime.
            var x = (TInput)(object)Vector<T>.Empty();
            var y = (TOutput)(object)Vector<T>.Empty();
            var predictions = (TOutput)(object)Vector<T>.Empty();

            return (x, y, predictions);
        }
        else if (typeof(TInput) == typeof(Matrix<T>) && typeof(TOutput) == typeof(Tensor<T>))
        {
            // Support for meta-learning algorithms like ProtoNets that use Matrix input and Tensor output
            var x = (TInput)(object)Matrix<T>.Empty();
            var y = (TOutput)(object)Tensor<T>.Empty();
            var predictions = (TOutput)(object)Tensor<T>.Empty();

            return (x, y, predictions);
        }
        else
        {
            throw new InvalidOperationException("Unsupported types for TInput and TOutput");
        }
    }

    /// <summary>
    /// Creates a default implementation of IFullModel based on the input and output types.
    /// </summary>
    /// <returns>A default implementation of IFullModel.</returns>
    /// <remarks>
    /// <para>
    /// This method creates an appropriate default model based on the input and output types.
    /// It supports creating VectorModel for linear models with matrix input and vector output,
    /// and NeuralNetworkModel for models with tensor input and output.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps create a starting point model based on your data types.
    /// 
    /// - For simple linear models (like regression), it creates a VectorModel.
    /// - For more complex models (like neural networks), it creates a NeuralNetworkModel.
    /// - If your data types don't match these patterns, it will throw an error to let you know.
    /// 
    /// This is useful when you need a basic model to start with, which you can then customize or optimize.
    /// </para>
    /// </remarks>
    public static IFullModel<T, TInput, TOutput> CreateDefaultModel()
    {
        // Create the appropriate model type based on the generic parameters
        if (typeof(TInput) == typeof(Matrix<T>) && typeof(TOutput) == typeof(Vector<T>))
        {
            // For linear models (matrix input, vector output)
            return (IFullModel<T, TInput, TOutput>)new VectorModel<T>(Vector<T>.Empty());
        }
        else if (typeof(TInput) == typeof(Tensor<T>) && typeof(TOutput) == typeof(Tensor<T>))
        {
            // For neural network models (tensor input and output)
            // Use OneDimensional input type with minimal configuration for a default placeholder model
            return (IFullModel<T, TInput, TOutput>)(object)new NeuralNetwork<T>(
                new NeuralNetworkArchitecture<T>(
                    InputType.OneDimensional,
                    NeuralNetworkTaskType.Regression,
                    NetworkComplexity.Simple,
                    inputSize: 1,
                    outputSize: 1));
        }
        else
        {
            // For other combinations, provide a clear error message
            throw new InvalidOperationException(
                $"Unsupported combination of input type {typeof(TInput).Name} and output type {typeof(TOutput).Name}. " +
                "Currently supported combinations are: " +
                $"(Matrix<{typeof(T).Name}>, Vector<{typeof(T).Name}>) for linear models and " +
                $"(Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>) for neural network models.");
        }
    }

    /// <summary>
    /// Gets column vectors from the input data based on the specified indices.
    /// </summary>
    /// <param name="input">The input data from which to extract columns.</param>
    /// <param name="indices">The indices of the columns to extract.</param>
    /// <returns>A collection of column vectors from the input data.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts specific columns from the input data based on the provided indices.
    /// It supports different input types like Matrix<T> and Tensor<T>.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps you select specific features from your dataset.
    /// 
    /// Think of your data as a spreadsheet:
    /// - Each column represents a different feature (like age, income, or temperature)
    /// - This method lets you pick only the columns you're interested in
    /// - It works with different types of data structures your model might use
    /// 
    /// For example, if you only want to use features 0, 2, and 5 from your dataset,
    /// you would pass [0, 2, 5] as the indices.
    /// </para>
    /// </remarks>
    public static List<Vector<T>> GetColumnVectors(TInput input, int[] indices)
    {
        if (input is Matrix<T> matrix)
        {
            return [.. indices.Select(i => matrix.GetColumn(i))];
        }
        else if (input is Tensor<T> tensor)
        {
            if (tensor.Shape.Length < 2)
            {
                throw new ArgumentException("Tensor must have at least 2 dimensions to extract columns");
            }

            var result = new List<Vector<T>>();
            foreach (int index in indices)
            {
                if (index < 0 || index >= tensor.Shape[1])
                {
                    throw new ArgumentOutOfRangeException(nameof(indices),
                        $"Column index {index} is out of range for tensor with shape {string.Join("×", tensor.Shape)}");
                }

                // Create a vector from the column
                Vector<T> column = new Vector<T>(tensor.Shape[0]);
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    column[i] = tensor[i, index];
                }

                result.Add(column);
            }

            return result;
        }
        else
        {
            throw new InvalidOperationException($"Unsupported input type: {input?.GetType().Name ?? "null"}");
        }
    }

    /// <summary>
    /// Creates a random model that emphasizes specific features.
    /// </summary>
    /// <param name="activeFeatures">The indices of features to emphasize in the model.</param>
    /// <param name="totalFeatures">The total number of available features.</param>
    /// <param name="useExpressionTrees">Whether to use expression trees instead of vector models.</param>
    /// <param name="maxExpressionTreeDepth">The maximum depth for expression trees if used.</param>
    /// <returns>A randomly initialized model appropriate for the input/output types.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a model that focuses on the specified active features while ignoring or 
    /// de-emphasizing others. It automatically selects an appropriate model type based on the 
    /// input and output generic parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a model that focuses on specific features.
    /// 
    /// For example, if you have data with features like age, income, education, etc.:
    /// - You can specify that the model should focus on just [age, education]
    /// - The model will be configured to use those features more significantly
    /// - Other features will be given minimal weight or ignored completely
    /// 
    /// This approach helps build more focused models that can perform better by
    /// concentrating on the most relevant features.
    /// </para>
    /// </remarks>
    public static IFullModel<T, TInput, TOutput> CreateRandomModelWithFeatures(
        int[] activeFeatures,
        int totalFeatures,
        bool useExpressionTrees = false,
        int maxExpressionTreeDepth = 3)
    {
        // Create a model based on the input/output types with the selected features
        if (typeof(TInput) == typeof(Matrix<T>) && typeof(TOutput) == typeof(Vector<T>))
        {
            if (useExpressionTrees)
            {
                // Create an expression tree model
                return CreateRandomExpressionTreeWithFeatures(activeFeatures, maxExpressionTreeDepth);
            }
            else
            {
                // Create a vector model
                return CreateRandomVectorModelWithFeatures(activeFeatures, totalFeatures);
            }
        }
        else if (typeof(TInput) == typeof(Tensor<T>) && typeof(TOutput) == typeof(Tensor<T>))
        {
            // Create a neural network model
            return CreateRandomNeuralNetworkWithFeatures(activeFeatures, totalFeatures);
        }

        // For other combinations, provide a clear error message
        throw new InvalidOperationException(
            $"Unsupported combination of input type {typeof(TInput).Name} and output type {typeof(TOutput).Name}. " +
            "Currently supported combinations are: " +
            $"(Matrix<{typeof(T).Name}>, Vector<{typeof(T).Name}>) for linear models and " +
            $"(Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>) for neural network models.");
    }

    /// <summary>
    /// Creates a random vector model that emphasizes specific features.
    /// </summary>
    /// <param name="activeFeatures">The indices of features to emphasize.</param>
    /// <param name="totalFeatures">The total number of available features.</param>
    /// <returns>A vector model with random weights that emphasize the active features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a linear model that gives importance to specific features.
    /// 
    /// The model will have:
    /// - Meaningful weights for the selected features (like age, income, etc.)
    /// - Very small weights for other features, essentially ignoring them
    /// 
    /// This way, predictions will primarily depend on the selected features.
    /// </para>
    /// </remarks>
    private static IFullModel<T, TInput, TOutput> CreateRandomVectorModelWithFeatures(
        int[] activeFeatures,
        int totalFeatures)
    {
        // Create a vector with the total number of features
        Vector<T> coefficients = new Vector<T>(totalFeatures);
        // Initialize all features with very small values (effectively disabling them)
        for (int i = 0; i < totalFeatures; i++)
        {
            coefficients[i] = _numOps.FromDouble(_random.NextDouble() * 0.0001); // Near-zero initialization
        }
        // Set meaningful values for active features
        foreach (int index in activeFeatures)
        {
            // Initialize active features with substantial random values
            coefficients[index] = _numOps.FromDouble((_random.NextDouble() * 2 - 1) * 0.5); // Values between -0.5 and 0.5
        }

        // Create the vector model
        return (IFullModel<T, TInput, TOutput>)(object)new VectorModel<T>(coefficients);
    }

    /// <summary>
    /// Creates a random expression tree that uses only the specified feature indices.
    /// </summary>
    /// <param name="activeFeatures">The feature indices that can be used in the tree.</param>
    /// <param name="maxDepth">The maximum depth of the expression tree.</param>
    /// <returns>A randomly generated expression tree.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a mathematical formula that only uses specific inputs.
    /// 
    /// For example, if your features are [age, height, weight, income] and you specify active
    /// features as [0, 3] (age and income), the resulting formula might be something like:
    /// (age * 2.5) + (income / 1000)
    /// 
    /// The tree structure enables complex mathematical relationships between the chosen features.
    /// </para>
    /// </remarks>
    private static ExpressionTree<T, TInput, TOutput> CreateRandomExpressionTreeWithFeatures(
        int[] activeFeatures,
        int maxDepth)
    {
        // Helper function to build a random tree
        ExpressionTree<T, TInput, TOutput> BuildTree(int depth)
        {
            // At max depth or with some probability, create a leaf node
            if (depth >= maxDepth || _random.NextDouble() < 0.3)
            {
                // Either a constant or a variable
                if (_random.NextDouble() < 0.5)
                {
                    // Create a constant with random value
                    return new ExpressionTree<T, TInput, TOutput>(
                        ExpressionNodeType.Constant,
                        _numOps.FromDouble(_random.NextDouble() * 2 - 1)); // Value between -1 and 1
                }
                else if (activeFeatures.Length > 0)
                {
                    // Create a variable (feature) node, but only use active features
                    int featureIndex = activeFeatures[_random.Next(activeFeatures.Length)];
                    return new ExpressionTree<T, TInput, TOutput>(
                        ExpressionNodeType.Variable,
                        _numOps.FromDouble(featureIndex));
                }
                else
                {
                    // If no active features, just create a constant
                    return new ExpressionTree<T, TInput, TOutput>(
                        ExpressionNodeType.Constant,
                        _numOps.FromDouble(_random.NextDouble() * 2 - 1));
                }
            }

            // Create an operation node
            ExpressionNodeType nodeType = (ExpressionNodeType)_random.Next(2, 6); // Add, Subtract, Multiply, or Divide

            var left = BuildTree(depth + 1);
            var right = BuildTree(depth + 1);

            return new ExpressionTree<T, TInput, TOutput>(nodeType, default, left, right);
        }

        // Build the random tree starting at depth 0
        return BuildTree(0);
    }

    /// <summary>
    /// Creates a random neural network model that emphasizes specific features.
    /// </summary>
    /// <param name="activeFeatures">The indices of features to emphasize.</param>
    /// <param name="totalFeatures">The total number of available features.</param>
    /// <returns>A neural network model configured to work with the active features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a neural network that focuses on specific inputs.
    /// 
    /// Neural networks are more complex models that can learn non-linear relationships.
    /// This method creates a network that's structured to emphasize the selected features.
    /// </para>
    /// </remarks>
    private static IFullModel<T, TInput, TOutput> CreateRandomNeuralNetworkWithFeatures(
        int[] activeFeatures,
        int totalFeatures)
    {
        // Create a neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: totalFeatures,
            outputSize: 1  // Assuming regression task with single output
        );

        // Create the neural network model
        var neuralModel = new NeuralNetwork<T>(architecture);

        return (IFullModel<T, TInput, TOutput>)(object)neuralModel;
    }

    /// <summary>
    /// Creates a random expression tree with a specified maximum depth.
    /// </summary>
    /// <param name="maxDepth">The maximum depth of the expression tree.</param>
    /// <returns>A randomly generated expression tree.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a random mathematical formula with operations
    /// like addition, subtraction, multiplication, and division.
    /// 
    /// The maxDepth parameter controls how complex the formula can be. Higher
    /// values create more complex formulas with more nested operations.
    /// </para>
    /// </remarks>
    private static ExpressionTree<T, TInput, TOutput> CreateRandomExpressionTree(int maxDepth)
    {
        if (maxDepth == 0 || _random.NextDouble() < 0.3) // 30% chance of generating a leaf node
        {
            return new ExpressionTree<T, TInput, TOutput>(ExpressionNodeType.Constant, _numOps.FromDouble(_random.NextDouble()));
        }

        ExpressionNodeType nodeType = (ExpressionNodeType)_random.Next(0, 4); // Randomly choose between Add, Subtract, Multiply, Divide
        var left = CreateRandomExpressionTree(maxDepth - 1);
        var right = CreateRandomExpressionTree(maxDepth - 1);

        return new ExpressionTree<T, TInput, TOutput>(nodeType, default, left, right);
    }
}
