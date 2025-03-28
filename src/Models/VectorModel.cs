namespace AiDotNet.Models;

/// <summary>
/// Represents a linear model that uses a vector of coefficients to make predictions.
/// </summary>
/// <remarks>
/// <para>
/// This class implements a simple linear model where predictions are made by computing the dot product of the input 
/// features and a vector of coefficients. It provides methods for training the model using linear regression, 
/// evaluating predictions, and genetic algorithm operations like mutation and crossover. This model is useful for 
/// linear regression problems and can serve as a building block for more complex models.
/// </para>
/// <para><b>For Beginners:</b> This is a simple linear model that makes predictions by multiplying each input by a weight and adding them up.
/// 
/// When using this model:
/// - Each input feature has a corresponding coefficient (weight)
/// - Predictions are made by multiplying each input by its coefficient and summing the results
/// - The model can be trained using linear regression on example data
/// - It supports genetic algorithm operations for optimization
/// 
/// For example, if predicting house prices, the model might learn that:
/// price = 50,000 × bedrooms + 100 × square_feet + 20,000 × bathrooms
/// 
/// This is one of the simplest and most interpretable machine learning models,
/// making it a good starting point for many problems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class VectorModel<T> : ISymbolicModel<T>
{
    /// <summary>
    /// Gets the vector of coefficients used by the model.
    /// </summary>
    /// <value>A Vector&lt;T&gt; containing the model's coefficients.</value>
    /// <remarks>
    /// <para>
    /// This property contains the vector of coefficients that the model uses to make predictions. Each coefficient 
    /// corresponds to a feature in the input data and represents the weight or importance of that feature in the model's 
    /// predictions. The coefficients are learned during training and are used to compute the dot product with input 
    /// features when making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the weights that the model applies to each input feature.
    /// 
    /// The coefficients:
    /// - Are the numbers that multiply each input feature
    /// - Determine how much each feature contributes to the prediction
    /// - Are learned during training to minimize prediction error
    /// 
    /// For example, in a house price prediction model:
    /// - A coefficient of 50,000 for bedrooms means each bedroom adds $50,000 to the predicted price
    /// - A coefficient of 100 for square feet means each square foot adds $100 to the predicted price
    /// 
    /// These coefficients make the model interpretable - you can see exactly how each feature affects the prediction.
    /// </para>
    /// </remarks>
    public Vector<T> Coefficients { get; }
    
    /// <summary>
    /// The numeric operations provider used for mathematical operations on type T.
    /// </summary>
    /// <remarks>
    /// This static field provides access to basic mathematical operations for the generic type T,
    /// allowing the class to perform calculations regardless of the specific numeric type.
    /// </remarks>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the VectorModel class with the specified coefficients.
    /// </summary>
    /// <param name="coefficients">The vector of coefficients for the model.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new VectorModel instance with the specified coefficients. The coefficients vector 
    /// determines the number of features the model expects and how it weights each feature when making predictions. 
    /// This constructor is useful when creating a model with predetermined coefficients or when creating a new model 
    /// as part of genetic algorithm operations.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new linear model with the specified weights.
    /// 
    /// When creating a VectorModel:
    /// - You provide a vector of coefficients (weights)
    /// - The length of this vector determines how many input features the model expects
    /// - The values determine how each feature affects the prediction
    /// 
    /// This constructor is used when:
    /// - Creating a model with specific, known coefficients
    /// - Creating a model as part of a genetic algorithm
    /// - Copying or modifying an existing model
    /// 
    /// For example: new VectorModel<double>(new Vector<double>([2.5, -1.3, 0.7]))
    /// creates a model that expects 3 features with the specified weights.
    /// </para>
    /// </remarks>
    public VectorModel(Vector<T> coefficients)
    {
        Coefficients = coefficients;
    }

    /// <summary>
    /// Gets the number of features used by the model.
    /// </summary>
    /// <value>An integer representing the number of input features.</value>
    /// <remarks>
    /// <para>
    /// This property returns the number of features that the model uses, which is determined by the length of the 
    /// coefficients vector. Each coefficient corresponds to a feature in the input data, so the number of coefficients 
    /// equals the number of features the model expects when making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many input variables the model uses.
    /// 
    /// The feature count:
    /// - Is equal to the length of the coefficients vector
    /// - Tells you how many input values the model expects
    /// - Must match the number of features in your input data
    /// 
    /// For example, if FeatureCount is 3, the model expects three input values
    /// for each prediction (like bedrooms, bathrooms, and square footage).
    /// 
    /// This property is useful when:
    /// - Checking if your input data has the right number of features
    /// - Understanding the complexity of the model
    /// - Preparing data for prediction
    /// </para>
    /// </remarks>
    public int FeatureCount => Coefficients.Length;

    /// <summary>
    /// Gets the complexity of the model.
    /// </summary>
    /// <value>An integer representing the model's complexity.</value>
    /// <exception cref="NotImplementedException">This property is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// This property is intended to return a measure of the model's complexity, which could be used for model selection 
    /// or regularization. However, it is not implemented in this class and will throw a NotImplementedException if accessed.
    /// </para>
    /// <para><b>For Beginners:</b> This property is supposed to tell you how complex the model is, but it's not implemented yet.
    /// 
    /// The complexity:
    /// - Would typically measure how intricate or sophisticated the model is
    /// - Is not implemented in this class (throws an exception if used)
    /// - For a linear model, could simply be the number of non-zero coefficients
    /// 
    /// This property would be useful for:
    /// - Comparing models based on their complexity
    /// - Implementing regularization to prevent overfitting
    /// - Selecting simpler models when performance is similar
    /// 
    /// But since it's not implemented, you should avoid using it with this class.
    /// </para>
    /// </remarks>
    public int Complexity => throw new NotImplementedException();

    /// <summary>
    /// Determines whether a specific feature is used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature has a non-zero coefficient, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method determines whether a specific feature is used by the model by checking if its corresponding coefficient 
    /// is non-zero. A zero coefficient means that the feature has no impact on the model's predictions, effectively 
    /// removing it from the model. This method is useful for feature selection and for understanding which features 
    /// the model considers important.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a particular input variable actually affects the model's predictions.
    /// 
    /// The IsFeatureUsed method:
    /// - Checks if the coefficient for a specific feature is non-zero
    /// - Returns true if the feature affects predictions, false if it doesn't
    /// - Helps identify which features are actually important to the model
    /// 
    /// For example, if the coefficient for square footage is 0, then
    /// IsFeatureUsed(2) would return false (assuming square footage is the third feature).
    /// 
    /// This method is useful when:
    /// - Analyzing which features matter to the model
    /// - Simplifying the model by removing unused features
    /// - Understanding the model's behavior
    /// </para>
    /// </remarks>
    public bool IsFeatureUsed(int featureIndex)
    {
        return !NumOps.Equals(Coefficients[featureIndex], NumOps.Zero);
    }

    /// <summary>
    /// Evaluates the model for a given input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>The model's prediction for the input.</returns>
    /// <exception cref="ArgumentException">Thrown when the input vector length doesn't match the coefficients length.</exception>
    /// <remarks>
    /// <para>
    /// This method evaluates the model for a given input vector by computing the dot product of the input and the 
    /// coefficients. This is the core prediction function of the linear model, where each feature value is multiplied 
    /// by its corresponding coefficient and the results are summed to produce the final prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the model's prediction for a single input.
    /// 
    /// The Evaluate method:
    /// - Takes a vector of input values
    /// - Multiplies each input by its corresponding coefficient
    /// - Adds up all these products to get the final prediction
    /// - Throws an error if the input has the wrong number of features
    /// 
    /// This is the core of how a linear model works - it's just a weighted sum:
    /// prediction = (input1 × coefficient1) + (input2 × coefficient2) + ...
    /// 
    /// For example, with coefficients [50000, 100, 20000] and input [3, 1500, 2],
    /// the prediction would be: 3×50000 + 1500×100 + 2×20000 = 350,000
    /// </para>
    /// </remarks>
    public T Evaluate(Vector<T> input)
    {
        if (input.Length != Coefficients.Length)
        {
            throw new ArgumentException("Input vector length must match coefficients length.");
        }

        T result = NumOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(Coefficients[i], input[i]));
        }

        return result;
    }

    /// <summary>
    /// Creates a mutated version of the model.
    /// </summary>
    /// <param name="mutationRate">The probability of each coefficient being mutated.</param>
    /// <returns>A new VectorModel with mutated coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a mutated version of the model by randomly modifying some of its coefficients. Each coefficient 
    /// has a probability equal to the mutation rate of being mutated. When a coefficient is selected for mutation, a small 
    /// random value is added to it. This method is useful in genetic algorithms for exploring the solution space and 
    /// potentially finding better models.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a slightly modified version of the model by randomly changing some coefficients.
    /// 
    /// The Mutate method:
    /// - Creates a new model with slightly different coefficients
    /// - Uses the mutationRate to determine how many coefficients to change
    /// - Adds small random values to the selected coefficients
    /// 
    /// For example, with a mutationRate of 0.1:
    /// - Each coefficient has a 10% chance of being modified
    /// - On average, 1 in 10 coefficients will change
    /// - The changes are small (±0.05 times the original value)
    /// 
    /// This method is used in genetic algorithms to:
    /// - Explore new possible solutions
    /// - Avoid getting stuck in local optima
    /// - Gradually improve models through random variation
    /// </para>
    /// </remarks>
    public ISymbolicModel<T> Mutate(double mutationRate)
    {
        Vector<T> mutatedCoefficients = new Vector<T>(Coefficients.Length);
        Random random = new Random();

        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (random.NextDouble() < mutationRate)
            {
                // Mutate the coefficient by adding a small random value
                T mutation = NumOps.FromDouble((random.NextDouble() - 0.5) * 0.1);
                mutatedCoefficients[i] = NumOps.Add(Coefficients[i], mutation);
            }
            else
            {
                mutatedCoefficients[i] = Coefficients[i];
            }
        }

        return new VectorModel<T>(mutatedCoefficients);
    }

    /// <summary>
    /// Creates a new model by crossing over this model with another model.
    /// </summary>
    /// <param name="other">The other model to crossover with.</param>
    /// <param name="crossoverRate">The probability of performing crossover at each position.</param>
    /// <returns>A new VectorModel resulting from the crossover.</returns>
    /// <exception cref="ArgumentException">Thrown when the other model is not a VectorModel or has a different number of features.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a new model by combining the coefficients of this model with those of another model. For each 
    /// coefficient position, there is a probability equal to the crossover rate of performing crossover, which in this 
    /// implementation means taking the average of the two parent coefficients. Otherwise, one of the parent coefficients 
    /// is randomly selected. This method is useful in genetic algorithms for combining the characteristics of two 
    /// potentially good solutions to create a new solution that might be even better.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a new model by combining features from this model and another model.
    /// 
    /// The Crossover method:
    /// - Creates a new model with coefficients derived from two parent models
    /// - Uses the crossoverRate to determine how to combine coefficients
    /// - For each position, either averages the two coefficients or picks one
    /// 
    /// For example, with parent coefficients [1, 2, 3] and [4, 5, 6]:
    /// - With crossover, a position might become (1+4)/2 = 2.5
    /// - Without crossover, a position might randomly take either 1 or 4
    /// 
    /// This method is used in genetic algorithms to:
    /// - Combine good features from different solutions
    /// - Create diversity in the population
    /// - Explore new combinations of successful traits
    /// </para>
    /// </remarks>
    public ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate)
    {
        if (!(other is VectorModel<T> otherVector))
        {
            throw new ArgumentException("Crossover can only be performed with another VectorModel.");
        }

        if (Coefficients.Length != otherVector.Coefficients.Length)
        {
            throw new ArgumentException("Vector lengths must match for crossover.");
        }

        Vector<T> childCoefficients = new Vector<T>(Coefficients.Length);
        Random random = new Random();

        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (random.NextDouble() < crossoverRate)
            {
                // Perform crossover by taking the average of the two coefficients
                childCoefficients[i] = NumOps.Divide(
                    NumOps.Add(Coefficients[i], otherVector.Coefficients[i]),
                    NumOps.FromDouble(2.0)
                );
            }
            else
            {
                // Randomly choose from either parent
                childCoefficients[i] = random.NextDouble() < 0.5 ? Coefficients[i] : otherVector.Coefficients[i];
            }
        }

        return new VectorModel<T>(childCoefficients);
    }

    /// <summary>
    /// Creates a copy of the model.
    /// </summary>
    /// <returns>A new VectorModel with the same coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the model by creating a new VectorModel with a new coefficients vector that has 
    /// the same values as the original. This ensures that modifications to the copy do not affect the original model. This 
    /// method is useful when you need to create a duplicate of a model for experimentation or as part of genetic algorithm 
    /// operations.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of the model.
    /// 
    /// The Copy method:
    /// - Creates a new model with the same coefficients as this one
    /// - Ensures the new model is completely independent of the original
    /// - Creates a "deep copy" where all data is duplicated, not just references
    /// 
    /// This method is useful when:
    /// - You need to create a duplicate of a model for experimentation
    /// - You want to ensure changes to one model don't affect another
    /// - You're implementing algorithms that require model copies
    /// 
    /// For example, you might copy a model before mutating it to preserve the original.
    /// </para>
    /// </remarks>
    public ISymbolicModel<T> Copy()
    {
        Vector<T> clonedCoefficients = new(Coefficients.Length);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            clonedCoefficients[i] = Coefficients[i];
        }
        return new VectorModel<T>(clonedCoefficients);
    }

    /// <summary>
    /// Fits the model to the provided training data.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <remarks>
    /// <para>
    /// This method fits the model to the provided training data by calling the Train method. It is provided as an alternative 
    /// name for the Train method to maintain consistency with the ISymbolicModel interface. In this implementation, Fit and 
    /// Train are identical.
    /// </para>
    /// <para><b>For Beginners:</b> This method trains the model on data (same as the Train method).
    /// 
    /// The Fit method:
    /// - Adjusts the model's coefficients to best predict the target values
    /// - Is just another name for the Train method in this class
    /// - Is provided for consistency with the ISymbolicModel interface
    /// 
    /// This method is useful when:
    /// - You want to train the model on your data
    /// - You're using code that expects a method named "Fit"
    /// 
    /// For example, some machine learning libraries use "fit" as the standard term
    /// for training a model, so this provides a familiar interface.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        // For VectorModel, Fit is the same as Train
        Train(X, y);
    }

    /// <summary>
    /// Trains the model on the provided data using linear regression.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <exception cref="ArgumentException">Thrown when the dimensions of X and y don't match or when X has the wrong number of columns.</exception>
    /// <remarks>
    /// <para>
    /// This method trains the model on the provided data using linear regression with the normal equation: 
    /// (X^T * X)^-1 * X^T * y. This approach finds the coefficients that minimize the sum of squared errors between the 
    /// model's predictions and the actual target values. The method updates the model's coefficients in place. Note that 
    /// this implementation requires that X^T * X is invertible, which may not be the case if there are linearly dependent 
    /// features or if there are more features than data points.
    /// </para>
    /// <para><b>For Beginners:</b> This method learns the best coefficients from your training data.
    /// 
    /// The Train method:
    /// - Takes a matrix of input features (X) and a vector of target values (y)
    /// - Uses linear regression to find the best coefficients
    /// - Updates the model's coefficients to minimize prediction error
    /// 
    /// It uses the "normal equation" approach to linear regression:
    /// - Calculates (X^T * X)^-1 * X^T * y to find optimal coefficients
    /// - This minimizes the sum of squared errors between predictions and actual values
    /// 
    /// For example, given house features (size, bedrooms, etc.) and prices,
    /// it would find the coefficients that best predict price from features.
    /// 
    /// Note: This approach requires that X^T * X is invertible, which may not be
    /// the case if features are linearly dependent or if there are more features
    /// than data points.
    /// </para>
    /// </remarks>
    public void Train(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in X must match the length of y.");
        }

        if (X.Columns != FeatureCount)
        {
            throw new ArgumentException($"Number of columns in X ({X.Columns}) must match the FeatureCount ({FeatureCount}).");
        }

        // Implement a simple linear regression using the normal equation
        // (X^T * X)^-1 * X^T * y
        Matrix<T> XTranspose = X.Transpose();
        Matrix<T> XTX = XTranspose * X;
        Matrix<T> XTXInverse = XTX.Inverse();
        Matrix<T> XTY = XTranspose * Matrix<T>.FromVector(y);
        Vector<T> newCoefficients = (XTXInverse * XTY).GetColumn(0);

        // Update the coefficients
        for (int i = 0; i < FeatureCount; i++)
        {
            Coefficients[i] = newCoefficients[i];
        }
    }

    /// <summary>
    /// Predicts the outputs for multiple input rows in a matrix.
    /// </summary>
    /// <param name="input">The input matrix.</param>
    /// <returns>A vector of predictions, one for each row in the input matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when the input matrix has the wrong number of columns.</exception>
    /// <remarks>
    /// <para>
    /// This method predicts the outputs for multiple input rows in a matrix by evaluating the model for each row. It returns 
    /// a vector of predictions, one for each row in the input matrix. This is more efficient than calling Evaluate separately 
    /// for each input row, especially for large numbers of predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes predictions for multiple inputs at once.
    /// 
    /// The Predict method:
    /// - Takes a matrix where each row is a set of input features
    /// - Returns a vector with one prediction for each row
    /// - Calls Evaluate for each row internally
    /// 
    /// This is more efficient than calling Evaluate separately for each input,
    /// especially when making many predictions.
    /// 
    /// For example, if predicting house prices, you could pass a matrix with
    /// 100 houses (rows) and get back a vector with 100 price predictions.
    /// 
    /// This method throws an error if the input has the wrong number of features.
    /// </para>
    /// </remarks>
    public Vector<T> Predict(Matrix<T> input)
    {
        if (input.Columns != FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {input.Columns} columns, but the model expects {FeatureCount} features.");
        }

        Vector<T> predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = Evaluate(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, feature count, complexity, and additional 
    /// information about the coefficients. The metadata includes the model type (Vector), the number of features, the 
    /// complexity (which for a vector model is the number of features), a description, and additional information such as 
    /// the norm of the coefficients, the number of non-zero coefficients, and the mean, maximum, and minimum coefficient 
    /// values. This metadata is useful for model selection, analysis, and visualization.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns detailed information about the model.
    /// 
    /// The GetModelMetadata method:
    /// - Creates a ModelMetadata object with information about the model
    /// - Includes basic properties like model type, feature count, and complexity
    /// - Adds additional statistics about the coefficients
    /// 
    /// The additional information includes:
    /// - CoefficientNorm: A measure of the magnitude of all coefficients
    /// - NonZeroCoefficients: How many features are actually used (have non-zero weights)
    /// - MeanCoefficient: The average coefficient value
    /// - MaxCoefficient: The largest coefficient value
    /// - MinCoefficient: The smallest coefficient value
    /// 
    /// This information is useful for:
    /// - Analyzing the model's characteristics
    /// - Comparing different models
    /// - Visualizing or reporting on the model
    /// </para>
    /// </remarks>
    public ModelMetadata<T> GetModelMetadata()
    {
        T norm = Coefficients.Norm();
        norm ??= NumOps.Zero;

        return new ModelMetadata<T>
        {
            ModelType = ModelType.Vector,
            FeatureCount = FeatureCount,
            Complexity = FeatureCount, // For a vector model, complexity is the number of features
            Description = $"Vector model with {FeatureCount} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "CoefficientNorm", norm! },
                { "NonZeroCoefficients", Coefficients.Count(c => !NumOps.Equals(c, NumOps.Zero)) },
                { "MeanCoefficient", Coefficients.Mean()! },
                { "MaxCoefficient", Coefficients.Max()! },
                { "MinCoefficient", Coefficients.Min()! }
            }
        };
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model to a byte array by writing the number of coefficients and then each coefficient 
    /// value. The serialization format is simple: first an integer indicating the number of coefficients, followed by each 
    /// coefficient as a double. This allows the model to be stored or transmitted and later reconstructed using the 
    /// Deserialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the model to a byte array that can be saved or transmitted.
    /// 
    /// The Serialize method:
    /// - Converts the model to a compact binary format
    /// - Writes the number of coefficients and each coefficient value
    /// - Returns a byte array that can be stored or transmitted
    /// 
    /// The serialization format is:
    /// 1. An integer with the number of coefficients
    /// 2. Each coefficient value as a double
    /// 
    /// This method is useful when:
    /// - Saving models to files or databases
    /// - Sending models over a network
    /// - Persisting models between application runs
    /// 
    /// The resulting byte array can be converted back to a model using Deserialize.
    /// </para>
    /// </remarks>
    public byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Write the number of coefficients
        writer.Write(Coefficients.Length);

        // Write each coefficient
        for (int i = 0; i < Coefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(Coefficients[i]));
        }

        return ms.ToArray();
    }

        /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the model from a byte array by reading the number of coefficients and then each coefficient 
    /// value. It expects the same format as produced by the Serialize method: first an integer indicating the number of 
    /// coefficients, followed by each coefficient as a double. This allows a model that was previously serialized to be 
    /// reconstructed.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs a model from a byte array created by Serialize.
    /// 
    /// The Deserialize method:
    /// - Takes a byte array containing a serialized model
    /// - Reads the number of coefficients and each coefficient value
    /// - Updates the model's coefficients with the deserialized values
    /// 
    /// It expects the same format created by Serialize:
    /// 1. An integer with the number of coefficients
    /// 2. Each coefficient value as a double
    /// 
    /// This method is useful when:
    /// - Loading models from files or databases
    /// - Receiving models over a network
    /// - Restoring models from persistent storage
    /// 
    /// Note that this method updates the existing model's coefficients rather than
    /// creating a new model, which is different from most other methods in this class.
    /// </para>
    /// </remarks>
    public void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Read the number of coefficients
        int length = reader.ReadInt32();

        // Create a new Vector<T> to hold the deserialized coefficients
        Vector<T> newCoefficients = new Vector<T>(length);

        // Read each coefficient
        for (int i = 0; i < length; i++)
        {
            newCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Update the Coefficients property
        for (int i = 0; i < length; i++)
        {
            Coefficients[i] = newCoefficients[i];
        }
    }

    /// <summary>
    /// Creates a new model with updated coefficients.
    /// </summary>
    /// <param name="newCoefficients">The new coefficients to use.</param>
    /// <returns>A new VectorModel with the updated coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the new coefficients vector has a different length than the current feature count.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a new model with updated coefficients. It checks that the new coefficients vector has the same 
    /// length as the current feature count and then creates a new VectorModel with the new coefficients. This method is 
    /// useful for creating variations of a model with different coefficient values without modifying the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a new model with different weights but the same structure.
    /// 
    /// The UpdateCoefficients method:
    /// - Takes a new vector of coefficients (weights)
    /// - Checks that it has the right length
    /// - Creates and returns a new model with these coefficients
    /// 
    /// Unlike some other methods, this doesn't modify the current model but
    /// creates a completely new one.
    /// 
    /// This method is useful when:
    /// - You want to manually set specific coefficient values
    /// - You're implementing optimization algorithms that test different coefficients
    /// - You need to create a modified version without changing the original
    /// 
    /// For example, you might use this to create a simplified version of a model
    /// by zeroing out small coefficients, or to test how changing certain weights
    /// affects predictions.
    /// </para>
    /// </remarks>
    public ISymbolicModel<T> UpdateCoefficients(Vector<T> newCoefficients)
    {
        if (newCoefficients.Length != this.FeatureCount)
        {
            throw new ArgumentException($"The number of new coefficients ({newCoefficients.Length}) must match the current feature count ({this.FeatureCount}).");
        }

        // Create a new VectorModel with the updated coefficients
        return new VectorModel<T>(newCoefficients);
    }
}