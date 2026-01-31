using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;

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
/// price = 50,000 * bedrooms + 100 * square_feet + 20,000 * bathrooms
///
/// This is one of the simplest and most interpretable machine learning models,
/// making it a good starting point for many problems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class VectorModel<T> : IFullModel<T, Matrix<T>, Vector<T>>, IInterpretableModel<T>
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
    /// <para>
    /// This field provides access to basic mathematical operations for the generic type T,
    /// allowing the class to perform calculations regardless of the specific numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a way to do math with different number types.
    /// 
    /// Since the model can work with different types of numbers (float, double, etc.),
    /// we need a way to perform math operations like addition and multiplication
    /// without knowing exactly what number type we're using. This helper provides
    /// those operations in a consistent way regardless of the number type.
    /// </para>
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Cached feature importance to avoid recreating on every GetModelMetadata() call.
    /// </summary>
    private Dictionary<string, T>? _cachedFeatureImportance;

    /// <summary>
    /// The default loss function used by this model for gradient computation.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Initializes a new instance of the VectorModel class with the specified coefficients.
    /// </summary>
    /// <param name="coefficients">The vector of coefficients for the model.</param>
    /// <param name="lossFunction">Optional loss function to use for training. If null, uses Mean Squared Error (MSE) for regression.</param>
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
    public VectorModel(Vector<T> coefficients, ILossFunction<T>? lossFunction = null)
    {
        Coefficients = coefficients ?? throw new ArgumentNullException(nameof(coefficients));
        _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Gets the default loss function used by this model for gradient computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For VectorModel (linear regression), the default loss function is Mean Squared Error (MSE),
    /// which is the standard loss function for regression problems.
    /// </para>
    /// </remarks>
    public ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

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
    /// <remarks>
    /// <para>
    /// This property returns the number of non-zero coefficients in the model, which provides a simple
    /// measure of the model's complexity. Lower complexity can indicate a more generalizable model that's
    /// less likely to overfit.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you how complex the model is.
    /// 
    /// The complexity:
    /// - Is measured by counting how many non-zero coefficients the model uses
    /// - Lower complexity often means better generalization to new data
    /// - Higher complexity might mean the model is overfitting to training data
    /// 
    /// For example, if a model has coefficients [0.5, 0, 0, 2.0, 0.3], its complexity would be 3,
    /// since only three coefficients are non-zero and actually contribute to predictions.
    /// 
    /// This is useful for comparing different models and for regularization approaches
    /// that aim to reduce model complexity.
    /// </para>
    /// </remarks>
    public int Complexity => Coefficients.Count(c => !_numOps.Equals(c, _numOps.Zero));

    /// <summary>
    /// Gets the number of trainable parameters in the model.
    /// </summary>
    /// <value>An integer representing the number of parameters.</value>
    /// <remarks>
    /// <para>
    /// For a VectorModel, the number of parameters equals the number of coefficients,
    /// as each coefficient is a trainable parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many weights the model has to learn.
    ///
    /// For a linear model:
    /// - Each coefficient is a parameter that can be trained
    /// - More parameters generally means more complexity
    /// - The parameter count equals the number of features
    /// </para>
    /// </remarks>
    public int ParameterCount => Coefficients.Length;

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
        if (featureIndex < 0 || featureIndex >= FeatureCount)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                $"Feature index must be between 0 and {FeatureCount - 1}");
        }

        return !_numOps.Equals(Coefficients[featureIndex], _numOps.Zero);
    }

    /// <summary>
    /// Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters.
    /// </summary>
    /// <param name="input">The input data matrix.</param>
    /// <param name="target">The target/expected output vector.</param>
    /// <param name="lossFunction">The loss function to use. If null, uses the model's default loss function.</param>
    /// <returns>A vector containing gradients with respect to all model parameters (coefficients).</returns>
    /// <exception cref="ArgumentNullException">If input or target is null.</exception>
    /// <exception cref="ArgumentException">If input and target dimensions don't match.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of the loss function with respect to the model's coefficients.
    /// For a linear model: y_pred = coefficients · x
    /// The gradient is computed as: ∂L/∂coefficients = (1/n) * X^T * ∂L/∂y_pred
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This calculates how to adjust each coefficient to reduce the prediction error,
    /// but it doesn't actually change the coefficients. This is useful for:
    /// - Distributed training: average gradients from multiple machines before updating
    /// - Custom optimization: use advanced optimizers like Adam or RMSprop
    /// - Analysis: understand which coefficients need the most adjustment
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (target == null)
            throw new ArgumentNullException(nameof(target));
        if (input.Rows != target.Length)
            throw new ArgumentException($"Input rows ({input.Rows}) must match target length ({target.Length})");
        if (input.Columns != Coefficients.Length)
            throw new ArgumentException($"Input columns ({input.Columns}) must match coefficient count ({Coefficients.Length})");

        var loss = lossFunction ?? DefaultLossFunction;

        // Forward pass: compute predictions
        var predictions = PredictInternal(input);

        // Compute loss gradient w.r.t. predictions: ∂L/∂y_pred
        var predictionGradient = loss.CalculateDerivative(predictions, target);

        // Compute gradient w.r.t. coefficients: ∂L/∂coefficients = (1/n) * X^T * ∂L/∂y_pred
        var gradients = new Vector<T>(Coefficients.Length);
        for (int j = 0; j < Coefficients.Length; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < input.Rows; i++)
            {
                // gradient[j] += input[i,j] * predictionGradient[i]
                sum = _numOps.Add(sum, _numOps.Multiply(input[i, j], predictionGradient[i]));
            }
            // Average over all samples
            gradients[j] = _numOps.Divide(sum, _numOps.FromDouble(input.Rows));
        }

        return gradients;
    }

    /// <summary>
    /// Applies pre-computed gradients to update the model parameters (coefficients).
    /// </summary>
    /// <param name="gradients">The gradient vector to apply.</param>
    /// <param name="learningRate">The learning rate for the update.</param>
    /// <exception cref="ArgumentNullException">If gradients is null.</exception>
    /// <exception cref="ArgumentException">If gradient vector length doesn't match coefficient count.</exception>
    /// <remarks>
    /// <para>
    /// Updates coefficients using: coefficients = coefficients - learningRate * gradients
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After computing gradients (seeing which direction to adjust each coefficient),
    /// this method actually adjusts them. The learning rate controls how big of an adjustment to make.
    ///
    /// In distributed training, this applies the averaged gradients from multiple machines
    /// to ensure all machines keep their models synchronized.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));
        if (gradients.Length != Coefficients.Length)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match coefficient count ({Coefficients.Length})",
                nameof(gradients));
        }

        // Apply gradient descent: coefficients = coefficients - learningRate * gradients
        for (int i = 0; i < Coefficients.Length; i++)
        {
            T update = _numOps.Multiply(learningRate, gradients[i]);
            Coefficients[i] = _numOps.Subtract(Coefficients[i], update);
        }

        // Invalidate cached feature importance since coefficients changed
        _cachedFeatureImportance = null;
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
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Length != Coefficients.Length)
        {
            throw new ArgumentException($"Input vector length ({input.Length}) must match coefficients length ({Coefficients.Length}).", nameof(input));
        }

        T result = _numOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            result = _numOps.Add(result, _numOps.Multiply(Coefficients[i], input[i]));
        }

        return result;
    }

    /// <summary>
    /// Trains the model on the provided data using linear regression.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <exception cref="ArgumentNullException">Thrown when X or y is null.</exception>
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
    private void TrainInternal(Matrix<T> X, Vector<T> y)
    {
        if (X == null)
        {
            throw new ArgumentNullException(nameof(X));
        }

        if (y == null)
        {
            throw new ArgumentNullException(nameof(y));
        }

        if (X.Rows != y.Length)
        {
            throw new ArgumentException($"Number of rows in X ({X.Rows}) must match the length of y ({y.Length}).");
        }

        if (X.Columns != FeatureCount)
        {
            throw new ArgumentException($"Number of columns in X ({X.Columns}) must match the FeatureCount ({FeatureCount}).");
        }

        try
        {
            // Implement a simple linear regression using the normal equation
            // (X^T * X)^-1 * X^T * y
            Matrix<T> XTranspose = X.Transpose();
            Matrix<T> XTX = XTranspose * X;

            // Check if XTX is singular (not invertible)
            if (!XTX.IsInvertible())
            {
                throw new InvalidOperationException("The matrix X^T * X is not invertible. " +
                    "This can happen when features are linearly dependent or when there are more features than data points.");
            }

            Matrix<T> XTXInverse = XTX.Inverse();
            Matrix<T> XTY = XTranspose * Matrix<T>.FromVector(y);
            Vector<T> newCoefficients = (XTXInverse * XTY).GetColumn(0);

            // Update the coefficients
            for (int i = 0; i < FeatureCount; i++)
            {
                Coefficients[i] = newCoefficients[i];
            }

            // Invalidate cached feature importance
            _cachedFeatureImportance = null;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Failed to train the model using linear regression. Consider adding regularization or using a different training method.", ex);
        }
    }

    /// <summary>
    /// Predicts the outputs for multiple input rows in a matrix.
    /// </summary>
    /// <param name="input">The input matrix.</param>
    /// <returns>A vector of predictions, one for each row in the input matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
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
    private Vector<T> PredictInternal(Matrix<T> input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Columns != FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {input.Columns} columns, but the model expects {FeatureCount} features.", nameof(input));
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
        norm ??= _numOps.Zero;

        int nonZeroCount = Coefficients.Count(c => !_numOps.Equals(c, _numOps.Zero));

        return new ModelMetadata<T>
        {
            FeatureCount = FeatureCount,
            Complexity = nonZeroCount,
            Description = $"Vector model with {FeatureCount} features ({nonZeroCount} active)",
            FeatureImportance = GetFeatureImportance(),
            AdditionalInfo = new Dictionary<string, object>
            {
                { "CoefficientNorm", norm! },
                { "NonZeroCoefficients", nonZeroCount },
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

        // Write a version number for forward compatibility
        writer.Write(1); // Version 1

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
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    /// <exception cref="ArgumentException">Thrown when data is empty or invalid.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the serialized coefficients count doesn't match the model's coefficients count.</exception>
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
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data.Length == 0)
        {
            throw new ArgumentException("Serialized data cannot be empty.", nameof(data));
        }

        try
        {
            using MemoryStream ms = new MemoryStream(data);
            using BinaryReader reader = new BinaryReader(ms);

            // Read version number
            int version = reader.ReadInt32();

            // Read the number of coefficients
            int length = reader.ReadInt32();

            // Validate coefficient count
            if (length != Coefficients.Length)
            {
                throw new InvalidOperationException($"Serialized coefficients count ({length}) doesn't match model's coefficients count ({Coefficients.Length}).");
            }

            // Read each coefficient
            for (int i = 0; i < length; i++)
            {
                Coefficients[i] = _numOps.FromDouble(reader.ReadDouble());
            }

            // Invalidate cached feature importance
            _cachedFeatureImportance = null;
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is ArgumentException || ex is InvalidOperationException))
        {
            throw new ArgumentException("Failed to deserialize the model. The data may be corrupted or in an invalid format.", nameof(data), ex);
        }
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model will be saved.</param>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the model to a file by serializing it to a byte array and writing it to disk.
    /// The model can later be loaded using the LoadModel method.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to a file so you can use it later.
    ///
    /// To use this method:
    /// - Provide a file path where the model should be saved
    /// - The model will be serialized and written to disk
    /// - You can load it later with LoadModel
    ///
    /// For example: model.SaveModel("my_model.bin");
    /// </para>
    /// </remarks>
    public void SaveModel(string filePath)
    {
        if (filePath == null)
        {
            throw new ArgumentNullException(nameof(filePath));
        }

        try
        {
            File.WriteAllBytes(filePath, Serialize());
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new InvalidOperationException($"Access denied when saving model to '{filePath}'. Check file permissions.", ex);
        }
        catch (DirectoryNotFoundException ex)
        {
            throw new InvalidOperationException($"Directory not found when saving model to '{filePath}'. Ensure the directory exists.", ex);
        }
        catch (IOException ex)
        {
            throw new InvalidOperationException($"IO error occurred while saving model to '{filePath}'. The file may be in use or the disk may be full.", ex);
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is InvalidOperationException))
        {
            throw new InvalidOperationException($"Unexpected error saving model to '{filePath}'.", ex);
        }
    }

    /// <summary>
    /// Loads a model from a file into the current instance.
    /// </summary>
    /// <param name="filePath">The path of the file containing the serialized model.</param>
    /// <exception cref="ArgumentNullException">Thrown when filePath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <remarks>
    /// <para>
    /// This method loads a model from a file by reading the byte array and deserializing it.
    /// The model must have been saved using the SaveModel method.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a saved model from a file.
    ///
    /// To use this method:
    /// - Provide the file path where the model was saved
    /// - The model will be loaded and replace the current model's coefficients
    /// - The file must have been created with SaveModel
    ///
    /// For example: model.LoadModel("my_model.bin");
    /// </para>
    /// </remarks>
    public void LoadModel(string filePath)
    {
        if (filePath == null)
        {
            throw new ArgumentNullException(nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        }

        try
        {
            byte[] data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new InvalidOperationException($"Access denied when loading model from '{filePath}'. Check file permissions.", ex);
        }
        catch (IOException ex)
        {
            throw new InvalidOperationException($"IO error occurred while loading model from '{filePath}'. The file may be in use or corrupted.", ex);
        }
        catch (ArgumentException ex)
        {
            throw new InvalidOperationException($"Invalid model file format at '{filePath}'. The file may be corrupted or not a valid VectorModel.", ex);
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is FileNotFoundException || ex is InvalidOperationException))
        {
            throw new InvalidOperationException($"Unexpected error loading model from '{filePath}'.", ex);
        }
    }

    /// <summary>
    /// Trains the model on the provided generic input and expected output.
    /// </summary>
    /// <param name="input">The input data, which must be a Matrix of type T.</param>
    /// <param name="expectedOutput">The expected output, which must be a Vector of type T.</param>
    /// <exception cref="ArgumentNullException">Thrown when input or expectedOutput is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the input or output types are incompatible with the model.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the IModel interface's Train method by delegating to the internal Train method.
    /// It validates the input and output types before proceeding with training.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model using your data.
    /// 
    /// When using this method:
    /// - You provide a matrix of input features and a vector of expected outputs
    /// - The model learns to predict the outputs from the inputs
    /// - It finds the weights (coefficients) that work best for your data
    /// 
    /// This is the main method you'll use to train the model on your dataset.
    /// </para>
    /// </remarks>
    public void Train(Matrix<T> input, Vector<T> expectedOutput)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (expectedOutput == null)
        {
            throw new ArgumentNullException(nameof(expectedOutput));
        }

        TrainInternal(input, expectedOutput);
    }

    /// <summary>
    /// Predicts outputs for the provided input.
    /// </summary>
    /// <param name="input">The input data to make predictions for.</param>
    /// <returns>A vector of predictions.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
    /// <exception cref="ArgumentException">Thrown when input has the wrong number of columns.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the IModel interface's Predict method by delegating to the internal Predict method.
    /// It accepts a matrix where each row is a separate data point and returns a vector of predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes predictions using your input data.
    /// 
    /// When using this method:
    /// - You provide a matrix where each row is a separate data point
    /// - Each row has the same features you trained the model with
    /// - The model returns one prediction for each row
    /// 
    /// This is the main method you'll use to make predictions once your model is trained.
    /// </para>
    /// </remarks>
    public Vector<T> Predict(Matrix<T> input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        return PredictInternal(input);
    }

    /// <summary>
    /// Gets all trainable parameters of the model as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (the coefficients).</returns>
    /// <remarks>
    /// <para>
    /// This method returns the coefficients of the model as a vector, which represents all trainable parameters of the model.
    /// For a vector model, the parameters are simply the coefficients vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you access to all the weights the model uses.
    /// 
    /// For a linear model:
    /// - The parameters are simply the coefficients (weights)
    /// - This method returns a copy of those coefficients
    /// 
    /// This is useful for:
    /// - Saving the model for later use
    /// - Analyzing the learned weights
    /// - Transferring weights to another model
    /// - Implementing optimization algorithms
    /// </para>
    /// </remarks>
    public Vector<T> GetParameters()
    {
        // Create a copy of the coefficients vector to avoid external modification
        Vector<T> parameters = new Vector<T>(Coefficients.Length);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            parameters[i] = Coefficients[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets the parameters of the model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when parameters has a different length than the model's coefficients.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the coefficients of the model from the provided parameters vector.
    /// For a vector model, the parameters are simply the coefficients vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the weights the model uses.
    /// 
    /// For a linear model:
    /// - The parameters are simply the coefficients (weights)
    /// - This method updates those coefficients
    /// 
    /// This is useful for:
    /// - Loading a saved model
    /// - Updating weights during optimization
    /// - Implementing learning algorithms
    /// </para>
    /// </remarks>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        if (parameters.Length != Coefficients.Length)
        {
            throw new ArgumentException($"Parameters length ({parameters.Length}) must match coefficients length ({Coefficients.Length}).", nameof(parameters));
        }

        // Update coefficients from parameters
        for (int i = 0; i < parameters.Length; i++)
        {
            Coefficients[i] = parameters[i];
        }

        // Invalidate cached feature importance
        _cachedFeatureImportance = null;
    }

    /// <summary>
    /// Updates the model with new parameter values.
    /// </summary>
    /// <param name="parameters">The new parameter values to use.</param>
    /// <returns>A new model with the updated parameters.</returns>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when parameters has a different length than the model's coefficients.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a new model with the provided parameter values. For a vector model, the parameters are
    /// simply the coefficients, so this creates a new model with new coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a new model with different weights.
    /// 
    /// The WithParameters method:
    /// - Takes a vector of new weights (parameters)
    /// - Creates a new model using these weights
    /// - Returns the new model without modifying the original
    /// 
    /// This is useful for:
    /// - Testing different sets of weights
    /// - Implementing optimization algorithms
    /// - Creating variations of a model
    /// 
    /// For example, you might try different weight values to see which ones
    /// give the best predictions or modify weights to simplify the model.
    /// </para>
    /// </remarks>
    public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        // Create a new model with the provided parameters
        // Allow different sizes to support genetic algorithm optimization which may resize models
        return new VectorModel<T>(parameters);
    }

    /// <summary>
    /// Gets the indices of all features used by this model.
    /// </summary>
    /// <returns>A collection of feature indices for features with non-zero coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the indices of all features that have non-zero coefficients. These are the features
    /// that actually contribute to the model's predictions. Features with zero coefficients have no effect on
    /// the output.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you which input features actually matter to the model.
    /// 
    /// It returns:
    /// - The positions (indices) of all features with non-zero weights
    /// - Only the features that actually affect the prediction
    /// 
    /// For example, if your model has coefficients [0.5, 0, 0, 2.0, 0.3],
    /// this method would return [0, 3, 4], since only those positions have non-zero values.
    /// 
    /// This is useful for:
    /// - Understanding which features the model considers important
    /// - Feature selection (identifying which features to keep)
    /// - Simplifying the model by focusing only on important features
    /// </para>
    /// </remarks>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (!_numOps.Equals(Coefficients[i], _numOps.Zero))
            {
                yield return i;
            }
        }
    }

    /// <summary>
    /// Creates a deep copy of this model.
    /// </summary>
    /// <returns>A new instance with the same coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the model by creating a new VectorModel with a new coefficients vector that has 
    /// the same values as the original. This ensures that modifications to the copy do not affect the original model. This 
    /// method is useful when you need to create a duplicate of a model for experimentation or as part of genetic algorithm 
    /// operations.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of the model.
    /// 
    /// The DeepCopy method:
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
    public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        // Create a new coefficients vector with the same values
        Vector<T> clonedCoefficients = new Vector<T>(Coefficients.Length);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            clonedCoefficients[i] = Coefficients[i];
        }

        // Create a new model with the cloned coefficients
        return new VectorModel<T>(clonedCoefficients);
    }

    /// <summary>
    /// Creates a shallow copy of this model.
    /// </summary>
    /// <returns>A new instance with the same coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a shallow copy of the model. For VectorModel, this is equivalent to DeepCopy because
    /// the only state that needs to be copied is the Coefficients vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a duplicate of the model.
    /// 
    /// For the VectorModel:
    /// - Clone and DeepCopy do the same thing
    /// - Both create a new model with a copy of the coefficients
    /// 
    /// The Clone method is provided for compatibility with the IFullModel interface.
    /// </para>
    /// </remarks>
    public IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
    }

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
            throw new ArgumentNullException(nameof(featureIndices));

        var indices = featureIndices.ToList();

        // Validate indices
        foreach (var index in indices)
        {
            if (index < 0 || index >= FeatureCount)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} is out of range. Must be between 0 and {FeatureCount - 1}.");
            }
        }

        // Since Coefficients is read-only, we need to modify the underlying data
        // Set non-active features to zero
        for (int i = 0; i < FeatureCount; i++)
        {
            if (!indices.Contains(i))
            {
                // Zero out the coefficient for inactive features
                Coefficients[i] = _numOps.Zero;
            }
        }

        // Invalidate cached feature importance
        _cachedFeatureImportance = null;
    }

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    /// <returns>A dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the feature importance scores for the model. For a VectorModel,
    /// the importance is based on the absolute value of each coefficient, as larger absolute
    /// values have a greater impact on predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method shows which features matter most to the model.
    ///
    /// For a linear model:
    /// - Features with larger coefficients (positive or negative) are more important
    /// - The importance is the absolute value of each coefficient
    /// - Features are named "Feature_0", "Feature_1", etc.
    ///
    /// This is useful for:
    /// - Understanding which features drive predictions
    /// - Feature selection (identifying which features to keep)
    /// - Model interpretation and explanation
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFeatureImportance()
    {
        if (_cachedFeatureImportance != null)
        {
            return new Dictionary<string, T>(_cachedFeatureImportance);
        }

        var importance = new Dictionary<string, T>();
        for (int i = 0; i < Coefficients.Length; i++)
        {
            importance.Add($"Feature_{i}", _numOps.Abs(Coefficients[i]));
        }
        _cachedFeatureImportance = importance;
        return new Dictionary<string, T>(importance);
    }

    #region IInterpretableModel Implementation

    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
    protected Vector<int>? _sensitiveFeatures;
    protected readonly List<FairnessMetric> _fairnessMetrics = new();
    protected IFullModel<T, Matrix<T>, Vector<T>>? _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync<T>(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Matrix<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync<T>(this, _enabledMethods, ConversionsHelper.ConvertToTensor<T>(input));
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Matrix<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods, ConversionsHelper.ConvertToTensor<T>(inputs));
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Matrix<T> input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(this, _enabledMethods, ConversionsHelper.ConvertToTensor<T>(input), numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        await Task.CompletedTask; // Satisfy async method signature

        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices));
        }

        if (featureIndices.Length == 0)
        {
            throw new ArgumentException("Feature indices cannot be empty.", nameof(featureIndices));
        }

        if (gridResolution <= 0)
        {
            throw new ArgumentException("Grid resolution must be greater than zero.", nameof(gridResolution));
        }

        // Validate feature indices
        for (int i = 0; i < featureIndices.Length; i++)
        {
            if (featureIndices[i] < 0 || featureIndices[i] >= FeatureCount)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {featureIndices[i]} is out of range. Must be between 0 and {FeatureCount - 1}.");
            }
        }

        // For a linear model (VectorModel), partial dependence is simply the coefficient
        // times the grid values for each feature, since the model is additive.
        // PD(x_j) = coefficient_j * x_j (for a linear model with no intercept)

        var gridValues = new Dictionary<int, Vector<T>>();
        var pdpMatrix = new List<Vector<T>>();

        // For VectorModel, we compute partial dependence for one feature at a time
        // since the model is linear and additive
        if (featureIndices.Length == 1)
        {
            int featureIdx = featureIndices[0];

            // Generate grid values for the feature
            // Using a range from -1 to 1 as a default since we don't have training data
            T minVal = _numOps.FromDouble(-1.0);
            T maxVal = _numOps.FromDouble(1.0);

            Vector<T> gridVals = new Vector<T>(gridResolution);
            for (int i = 0; i < gridResolution; i++)
            {
                T ratio = _numOps.FromDouble((double)i / (gridResolution - 1));
                T range = _numOps.Subtract(maxVal, minVal);
                gridVals[i] = _numOps.Add(minVal, _numOps.Multiply(range, ratio));
            }

            gridValues[featureIdx] = gridVals;

            // For a linear model, partial dependence is simply coefficient * feature_value
            Vector<T> pdValues = new Vector<T>(gridResolution);
            for (int i = 0; i < gridResolution; i++)
            {
                pdValues[i] = _numOps.Multiply(Coefficients[featureIdx], gridVals[i]);
            }

            pdpMatrix.Add(pdValues);
        }
        else
        {
            // For multiple features, compute the interaction effect
            // For a linear model without interaction terms, this is just the sum
            // of individual feature effects

            // Generate grid for each feature
            var grids = new List<Vector<T>>();
            foreach (int featureIdx in featureIndices)
            {
                T minVal = _numOps.FromDouble(-1.0);
                T maxVal = _numOps.FromDouble(1.0);

                Vector<T> gridVals = new Vector<T>(gridResolution);
                for (int i = 0; i < gridResolution; i++)
                {
                    T ratio = _numOps.FromDouble((double)i / (gridResolution - 1));
                    T range = _numOps.Subtract(maxVal, minVal);
                    gridVals[i] = _numOps.Add(minVal, _numOps.Multiply(range, ratio));
                }

                gridValues[featureIdx] = gridVals;
                grids.Add(gridVals);
            }

            // Compute partial dependence for each combination
            // For a 2D case with gridResolution=20, we'd have 20x20=400 points
            int totalPoints = 1;
            for (int i = 0; i < featureIndices.Length; i++)
            {
                totalPoints *= gridResolution;
            }

            // Generate all combinations and compute PD values
            Vector<T> pdValues = new Vector<T>(totalPoints);
            int pointIdx = 0;

            // Generate combinations recursively
            var currentPoint = new T[featureIndices.Length];
            ComputePDCombinations(featureIndices, grids, 0, currentPoint, pdValues, ref pointIdx);

            pdpMatrix.Add(pdValues);
        }

        // Create the result matrix
        Matrix<T> pdMatrix;
        if (pdpMatrix.Count == 1)
        {
            // Single row for 1D or single vector for multi-feature
            pdMatrix = new Matrix<T>(1, pdpMatrix[0].Length);
            for (int j = 0; j < pdpMatrix[0].Length; j++)
            {
                pdMatrix[0, j] = pdpMatrix[0][j];
            }
        }
        else
        {
            pdMatrix = new Matrix<T>(pdpMatrix.Count, pdpMatrix[0].Length);
            for (int i = 0; i < pdpMatrix.Count; i++)
            {
                for (int j = 0; j < pdpMatrix[i].Length; j++)
                {
                    pdMatrix[i, j] = pdpMatrix[i][j];
                }
            }
        }

        return new PartialDependenceData<T>
        {
            FeatureIndices = featureIndices,
            GridValues = gridValues,
            PartialDependenceValues = pdMatrix,
            GridResolution = gridResolution,
            IceCurves = new List<Matrix<T>>()
        };
    }

    /// <summary>
    /// Helper method to recursively compute partial dependence combinations.
    /// </summary>
    private void ComputePDCombinations(Vector<int> featureIndices, List<Vector<T>> grids,
        int depth, T[] currentPoint, Vector<T> pdValues, ref int pointIdx)
    {
        if (depth == featureIndices.Length)
        {
            // Compute PD value for this combination
            T pdValue = _numOps.Zero;
            for (int i = 0; i < featureIndices.Length; i++)
            {
                int featureIdx = featureIndices[i];
                pdValue = _numOps.Add(pdValue, _numOps.Multiply(Coefficients[featureIdx], currentPoint[i]));
            }
            pdValues[pointIdx++] = pdValue;
            return;
        }

        // Recurse through grid values for this feature
        Vector<T> currentGrid = grids[depth];
        for (int i = 0; i < currentGrid.Length; i++)
        {
            currentPoint[depth] = currentGrid[i];
            ComputePDCombinations(featureIndices, grids, depth + 1, currentPoint, pdValues, ref pointIdx);
        }
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Matrix<T> input, Vector<T> desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(this, _enabledMethods, ConversionsHelper.ConvertToTensor<T>(input), ConversionsHelper.ConvertToTensor<T>(desiredOutput), maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync<T>(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(Matrix<T> input, Vector<T> prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync<T>(this, ConversionsHelper.ConvertToTensor<T>(input), ConversionsHelper.ConvertToTensor<T>(prediction));
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Matrix<T> inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Matrix<T> input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(this, _enabledMethods, ConversionsHelper.ConvertToTensor<T>(input), threshold);
    }

    // IInterpretableModel<T> interface implementations with Tensor<T> parameters

    /// <summary>
    /// Gets the local feature importance for a specific input (IInterpretableModel implementation).
    /// </summary>
    public virtual Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
    {
        return InterpretableModelHelper.GetLocalFeatureImportanceAsync<T>(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs (IInterpretableModel implementation).
    /// </summary>
    public virtual Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
    {
        return InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods, inputs);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input (IInterpretableModel implementation).
    /// </summary>
    public virtual Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
    {
        return InterpretableModelHelper.GetLimeExplanationAsync<T>(this, _enabledMethods, input, numFeatures);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output (IInterpretableModel implementation).
    /// </summary>
    public virtual Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
    {
        return InterpretableModelHelper.GetCounterfactualAsync<T>(this, _enabledMethods, input, desiredOutput, maxChanges);
    }

    /// <summary>
    /// Generates a text explanation for a prediction (IInterpretableModel implementation).
    /// </summary>
    public virtual Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
    {
        return InterpretableModelHelper.GenerateTextExplanationAsync<T>(this, input, prediction);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs (IInterpretableModel implementation).
    /// </summary>
    public virtual Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
    {
        return InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input (IInterpretableModel implementation).
    /// </summary>
    public virtual Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
    {
        return InterpretableModelHelper.GetAnchorExplanationAsync(this, _enabledMethods, input, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis (IInterpretableModel implementation).
    /// </summary>
    public virtual void SetBaseModel<TInput, TOutput>(IFullModel<T, TInput, TOutput> model)
    {
        _baseModel = model as IFullModel<T, Matrix<T>, Vector<T>> ?? throw new ArgumentException("Base model must be compatible with Matrix<T> input and Vector<T> output.", nameof(model));
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IFullModel<T, Matrix<T>, Vector<T>> model)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public virtual void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

    #endregion

    /// <summary>
    /// Saves the model's current state (parameters and configuration) to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes all the information needed to recreate the model's current state,
    /// including the model's coefficients. It uses the existing Serialize method and writes
    /// the data to the provided stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a snapshot of your trained linear model.
    ///
    /// When you call SaveState:
    /// - All the learned coefficients (weights) are written to the stream
    /// - The model's configuration is preserved
    ///
    /// This is particularly useful for:
    /// - Checkpointing during long training sessions
    /// - Knowledge distillation (saving teacher/student models)
    /// - Resuming interrupted training
    /// - Creating model ensembles
    ///
    /// You can later use LoadState to restore the model to this exact state.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        try
        {
            var data = this.Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to save model state to stream: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unexpected error while saving model state: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the model's state (parameters and configuration) from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes model state that was previously saved with SaveState,
    /// restoring all coefficients and configuration to recreate the saved model.
    /// It uses the existing Deserialize method after reading data from the stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your linear model.
    ///
    /// When you call LoadState:
    /// - All the coefficients are read from the stream
    /// - The model is configured to match the saved state
    /// - The model becomes identical to when SaveState was called
    ///
    /// After loading, the model can:
    /// - Make predictions using the restored coefficients
    /// - Continue training from where it left off
    /// - Be used as a teacher model in knowledge distillation
    ///
    /// This is essential for:
    /// - Resuming interrupted training sessions
    /// - Loading the best checkpoint after training
    /// - Deploying trained models to production
    /// - Knowledge distillation workflows
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data.</exception>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        try
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            if (data.Length == 0)
                throw new InvalidOperationException("Stream contains no data.");

            this.Deserialize(data);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read model state from stream: {ex.Message}", ex);
        }
        catch (InvalidOperationException)
        {
            // Re-throw InvalidOperationException from Deserialize
            throw;
        }
        catch (ArgumentException)
        {
            // Re-throw ArgumentException from Deserialize
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to deserialize model state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
        }
    }

    #region IJitCompilable Implementation

    /// <summary>
    /// Gets a value indicating whether this model supports JIT compilation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// VectorModel supports JIT compilation by converting its linear regression computation
    /// (matrix-vector multiplication) to a computation graph. This enables 5-10x faster inference.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation makes predictions much faster.
    ///
    /// Linear regression is simple: output = input @ coefficients
    /// With JIT, this computation is compiled to optimized native code for maximum speed.
    ///
    /// Especially beneficial for:
    /// - Processing large datasets
    /// - Real-time prediction systems
    /// - Production deployments
    /// </para>
    /// </remarks>
    public bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the linear regression model as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the linear regression computation into a computation graph:
    /// output = input @ coefficients
    ///
    /// The graph represents a simple matrix-vector multiplication that the JIT compiler
    /// can optimize and compile to native code.
    /// </para>
    /// <para><b>For Beginners:</b> This converts your linear model into a form the JIT compiler can optimize.
    ///
    /// The conversion:
    /// 1. Converts Matrix/Vector to Tensor (JIT works with Tensors)
    /// 2. Creates computation nodes for input and coefficients
    /// 3. Builds a graph: output = MatMul(input, coefficients)
    /// 4. Returns the output node
    ///
    /// Once converted, the JIT compiler can:
    /// - Optimize the computation
    /// - Generate fast native code
    /// - Provide 5-10x faster predictions
    /// </para>
    /// </remarks>
    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Convert coefficients Vector to Tensor
        // Shape: (features,) -> (features, 1) for matrix multiplication
        var coeffTensor = VectorToTensor(Coefficients);
        var coeffNode = new ComputationNode<T>(coeffTensor);

        // Create an input node representing a single batch
        // Expected shape: (batch_size, features)
        var inputShape = new int[] { 1, FeatureCount }; // Batch size 1, FeatureCount features
        var inputTensor = new Tensor<T>(inputShape);
        var inputNode = new ComputationNode<T>(inputTensor);
        inputNodes.Add(inputNode);

        // Linear regression: output = input @ coefficients
        // This is a matrix-vector multiplication
        var outputNode = TensorOperations<T>.MatrixMultiply(inputNode, coeffNode);

        return outputNode;
    }

    /// <summary>
    /// Converts a Vector to a Tensor for use in computation graphs.
    /// </summary>
    private Tensor<T> VectorToTensor(Vector<T> vector)
    {
        // Convert Vector to 2D Tensor: (length,) -> (length, 1)
        var shape = new int[] { vector.Length, 1 };
        var data = new T[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            data[i] = vector[i];
        }
        return new Tensor<T>(shape, new Vector<T>(data));
    }

    #endregion
}
