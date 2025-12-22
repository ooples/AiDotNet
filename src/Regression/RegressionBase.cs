global using AiDotNet.Factories;
using AiDotNet.Autodiff;
using Newtonsoft.Json;

namespace AiDotNet.Regression;

/// <summary>
/// Provides a base implementation for regression algorithms that model the relationship
/// between a dependent variable and one or more independent variables.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for regression models, including
/// prediction, serialization/deserialization, and solving linear systems. Specific regression
/// algorithms should inherit from this class and implement the Train method.
/// </para>
/// <para>
/// The class supports various options like regularization to prevent overfitting and
/// different decomposition methods for solving linear systems.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Regression is a statistical method for modeling the relationship between variables.
/// This base class provides the foundation for different regression techniques, handling
/// common operations like making predictions and saving/loading models. Think of it as
/// a template that specific regression algorithms can customize while reusing the shared
/// functionality.
/// </para>
/// </remarks>
public abstract class RegressionBase<T> : IRegression<T>
{
    /// <summary>
    /// Gets the numeric operations for the specified type T.
    /// </summary>
    /// <value>
    /// An object that provides mathematical operations for type T.
    /// </value>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the execution engine (CPU or GPU) for performing
    /// vectorized operations. The engine is determined by the global AiDotNetEngine configuration
    /// and allows automatic fallback from GPU to CPU when GPU is not available.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives access to either CPU or GPU processing for faster computations.
    /// The system automatically chooses the best available option and falls back to CPU
    /// if GPU acceleration is not available.
    /// </para>
    /// </remarks>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the regression options.
    /// </summary>
    /// <value>
    /// Configuration options for the regression model.
    /// </value>
    protected RegressionOptions<T> Options { get; private set; }

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    /// <value>
    /// An object that implements regularization for the regression model.
    /// </value>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

    /// <summary>
    /// Gets the default loss function for this regression model.
    /// </summary>
    /// <value>
    /// The loss function used for gradient computation.
    /// </value>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets or sets the coefficients (weights) of the regression model.
    /// </summary>
    /// <value>
    /// A vector of coefficients, one for each feature.
    /// </value>
    public Vector<T> Coefficients { get; protected set; }

    /// <summary>
    /// Gets or sets the intercept (bias) term of the regression model.
    /// </summary>
    /// <value>
    /// The intercept value.
    /// </value>
    public T Intercept { get; protected set; }

    /// <summary>
    /// Gets a value indicating whether the model includes an intercept term.
    /// </summary>
    /// <value>
    /// True if the model includes an intercept; otherwise, false.
    /// </value>
    public bool HasIntercept => Options.UseIntercept;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    /// <value>
    /// An array of feature names. If not set, feature indices will be used as names.
    /// </value>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets the expected number of parameters (coefficients plus intercept if used).
    /// </summary>
    /// <value>
    /// The total number of parameters, which equals the number of coefficients plus 1 if an intercept is used, or just the number of coefficients otherwise.
    /// </value>
    protected int ExpectedParameterCount => Coefficients.Length + (Options.UseIntercept ? 1 : 0);

    /// <summary>
    /// Initializes a new instance of the RegressionBase class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <param name="lossFunction">Loss function for gradient computation. If null, defaults to Mean Squared Error.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data. The loss function determines how
    /// prediction errors are measured during training.
    /// </para>
    /// </remarks>
    protected RegressionBase(RegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null, ILossFunction<T>? lossFunction = null)
    {
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new RegressionOptions<T>();
        Coefficients = new Vector<T>(0);
        Intercept = NumOps.Zero;
        _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Trains the regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to train the regression model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training is the process where the model learns from your data. Different regression algorithms
    /// implement this method differently, but they all aim to find the best coefficients (weights)
    /// that minimize the prediction error on the training data.
    /// </para>
    /// </remarks>
    public abstract void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates predictions by multiplying the input features by the model coefficients
    /// and adding the intercept if one is used.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After training, this method is used to make predictions on new data. It applies the learned
    /// coefficients to the input features and adds the intercept (if used) to produce the final prediction.
    /// For linear regression, this is simply the dot product of the features and coefficients plus the intercept.
    /// </para>
    /// </remarks>
    public virtual Vector<T> Predict(Matrix<T> input)
    {
        var predictions = input.Multiply(Coefficients);

        if (Options.UseIntercept)
        {
            predictions = predictions.Add(Intercept);
        }

        return predictions;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, feature count, complexity,
    /// description, and additional information like coefficient norm and feature importances.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Model metadata provides information about the model itself, rather than the predictions it makes.
    /// This includes details about the model's structure (like how many features it uses) and characteristics
    /// (like which features are most important). This information can be useful for understanding and
    /// comparing different models.
    /// </para>
    /// </remarks>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            FeatureCount = Coefficients.Length,
            Complexity = Coefficients.Length,
            Description = $"{GetModelType()} model with {Coefficients.Length} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HasIntercept", HasIntercept },
                { "CoefficientNorm", Coefficients.Norm()! },
                { "FeatureImportances", CalculateFeatureImportances().ToArray() }
            }
        };
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to specify the model type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method simply returns an identifier that indicates what type of regression model this is
    /// (e.g., linear regression, ridge regression). It's used internally by the library to keep track
    /// of different types of models.
    /// </para>
    /// </remarks>
    protected abstract ModelType GetModelType();

    /// <summary>
    /// Calculates the importance of each feature in the model.
    /// </summary>
    /// <returns>A vector of feature importances.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates feature importances based on the absolute values of the coefficients.
    /// Derived classes may override this method to provide more sophisticated feature importance calculations.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Feature importance tells you which input variables have the most influence on the predictions.
    /// In basic regression models, this is calculated from the absolute values of the coefficients -
    /// larger coefficients (ignoring sign) indicate more important features.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> CalculateFeatureImportances()
    {
        return Coefficients.Transform(NumOps.Abs);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including coefficients, intercept, and regularization options,
    /// to a JSON format and then converts it to a byte array.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "Coefficients", Coefficients.ToArray() },
            { "Intercept", Intercept ?? NumOps.Zero! },
            { "RegularizationOptions", Regularization.GetOptions() }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including coefficients,
    /// intercept, and regularization options.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var coefficientsToken = modelDataObj["Coefficients"];
        var interceptToken = modelDataObj["Intercept"];
        if (coefficientsToken == null || interceptToken == null)
        {
            throw new InvalidOperationException("Deserialization failed: Missing required regression parameters.");
        }

        var coefficientsAsDoubles = coefficientsToken.ToObject<double[]>() ?? Array.Empty<double>();
        var coefficients = new Vector<T>(coefficientsAsDoubles.Length);
        for (int i = 0; i < coefficientsAsDoubles.Length; i++)
        {
            coefficients[i] = NumOps.FromDouble(coefficientsAsDoubles[i]);
        }

        Coefficients = coefficients;
        Intercept = NumOps.FromDouble(interceptToken.ToObject<double>());

        var regularizationOptionsToken = modelDataObj["RegularizationOptions"];
        if (regularizationOptionsToken == null)
        {
            throw new InvalidOperationException("Deserialization failed: Missing regularization options.");
        }

        var regularizationOptionsJson = JsonConvert.SerializeObject(regularizationOptionsToken);
        var regularizationOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regularizationOptionsJson)
            ?? throw new InvalidOperationException("Deserialization failed: Unable to deserialize regularization options.");

        Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regularizationOptions);
    }

    /// <summary>
    /// Solves a linear system of equations using the specified decomposition method.
    /// </summary>
    /// <param name="a">The coefficient matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector.</returns>
    /// <remarks>
    /// <para>
    /// This method solves the linear system Ax = b using either the specified decomposition method
    /// or the normal equation as a fallback.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Many regression problems involve solving a system of linear equations. This method provides
    /// a way to solve such systems using various mathematical techniques. The choice of technique
    /// can affect the accuracy and efficiency of the solution, especially for large or ill-conditioned
    /// systems.
    /// </para>
    /// </remarks>
    protected Vector<T> SolveSystem(Matrix<T> a, Vector<T> b)
    {
        var decomposition = Options.DecompositionMethod;

        if (decomposition != null)
        {
            return decomposition.Solve(b);
        }
        else
        {
            // Prefer a direct solve for square systems; only use the normal equation as a fallback
            // for non-square (over/under-determined) systems.
            if (a.Rows == a.Columns)
            {
                try
                {
                    // Fast/stable for symmetric positive definite systems.
                    var cholesky = new CholeskyDecomposition<T>(a);
                    return cholesky.Solve(b);
                }
                catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
                {
                    try
                    {
                        // General-purpose fallback for square systems.
                        var lu = new LuDecomposition<T>(a);
                        return lu.Solve(b);
                    }
                    catch (Exception ex2) when (ex2 is ArgumentException or InvalidOperationException)
                    {
                        // Most robust fallback (handles singular/ill-conditioned matrices).
                        var svd = new SvdDecomposition<T>(a);
                        return svd.Solve(b);
                    }
                }
            }

            // Use normal equation if specifically selected or as a fallback
            return SolveNormalEquation(a, b);
        }
    }

    /// <summary>
    /// Solves a linear system using the normal equation.
    /// </summary>
    /// <param name="a">The coefficient matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector.</returns>
    /// <remarks>
    /// <para>
    /// This method solves the normal equation (A^T A)x = A^T b using LU decomposition.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// The normal equation is a way to solve linear regression problems by transforming them into
    /// a form that can be solved directly. It's computationally efficient for small to medium-sized
    /// problems but can be numerically unstable for ill-conditioned matrices. This method is used
    /// as a fallback when no specific decomposition method is specified.
    /// </para>
    /// </remarks>
    private Vector<T> SolveNormalEquation(Matrix<T> a, Vector<T> b)
    {
        var aTa = a.Transpose().Multiply(a);
        var aTb = a.Transpose().Multiply(b);

        // Use LU decomposition for solving the normal equation
        var normalDecomposition = new NormalDecomposition<T>(aTa);
        return normalDecomposition.Solve(aTb);
    }

    /// <summary>
    /// Gets all model parameters (coefficients and intercept) as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a vector containing all model parameters (coefficients followed by intercept)
    /// for use with optimization algorithms or model comparison.
    /// </para>
    /// <para><b>For Beginners:</b> This method packages all the model's parameters into a single collection.
    ///
    /// Think of the parameters as the "recipe" for your model's predictions:
    /// - The coefficients represent how much each feature contributes to the prediction
    /// - The intercept is the baseline prediction when all features are zero
    ///
    /// Getting all parameters at once allows tools to optimize the model or compare different models.
    /// For example, an optimization algorithm might try different combinations of parameters to find
    /// the ones that give the most accurate predictions.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        // Create a new vector with enough space for coefficients + intercept (if used)
        int paramCount = Coefficients.Length + (Options.UseIntercept ? 1 : 0);
        Vector<T> parameters = new Vector<T>(paramCount);

        // Copy coefficients to the parameters vector
        for (int i = 0; i < Coefficients.Length; i++)
        {
            parameters[i] = Coefficients[i];
        }

        // Add the intercept as the last element (if used)
        if (Options.UseIntercept)
        {
            parameters[Coefficients.Length] = Intercept;
        }

        return parameters;
    }

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters (coefficients and intercept).</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has an incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a new model with the same options but different parameter values.
    /// The parameters vector should contain coefficients followed by the intercept (if the model uses one).
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a new model using a specific set of parameters.
    ///
    /// It's like creating a new recipe based on an existing one, but with different ingredient amounts.
    /// You provide all the parameters (coefficients and intercept) in a single collection, and the method:
    /// - Creates a new model
    /// - Sets its parameters to the values you provided
    /// - Returns this new model ready to use for predictions
    ///
    /// This is useful for:
    /// - Testing how different parameter values affect predictions
    /// - Using optimization algorithms that try different parameter sets
    /// - Creating ensemble models that combine multiple parameter variations
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        if (parameters.Length != ExpectedParameterCount)
        {
            throw new ArgumentException($"Expected {ExpectedParameterCount} parameters, but got {parameters.Length}", nameof(parameters));
        }

        // Create a new instance of the model
        var newModel = (RegressionBase<T>)Clone();

        // Extract coefficients
        Vector<T> newCoefficients = new Vector<T>(Coefficients.Length);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            newCoefficients[i] = parameters[i];
        }

        // Set the coefficients in the new model
        newModel.Coefficients = newCoefficients;

        // Set the intercept if used
        if (Options.UseIntercept)
        {
            newModel.Intercept = parameters[Coefficients.Length];
        }
        else
        {
            newModel.Intercept = NumOps.Zero;
        }

        return newModel;
    }

    /// <summary>
    /// Gets the indices of features that are actively used in the model.
    /// </summary>
    /// <returns>An enumerable collection of indices for features with non-zero coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies which features are actually contributing to the model's predictions by
    /// returning the indices of all features with non-zero coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you which input features actually matter in the model.
    ///
    /// Not all features necessarily contribute to predictions. Some might have coefficients of zero,
    /// meaning they're effectively ignored by the model. This method returns the positions (indices) of
    /// features that do have an effect on predictions.
    ///
    /// For example, if your model has 10 features but only features at positions 2, 5, and 7
    /// have non-zero coefficients, this method would return [2, 5, 7].
    ///
    /// This is useful for:
    /// - Feature selection (identifying which features are most important)
    /// - Model simplification (removing unused features)
    /// - Understanding which inputs actually affect the prediction
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        for (int i = 0; i < Coefficients.Length; i++)
        {
            // If the coefficient is not zero (using a threshold for floating-point comparison)
            if (!NumOps.Equals(Coefficients[i], NumOps.Zero))
            {
                yield return i;
            }
        }
    }

    /// <summary>
    /// Determines whether a specific feature is used in the model.
    /// </summary>
    /// <param name="featureIndex">The zero-based index of the feature to check.</param>
    /// <returns>True if the feature has a non-zero coefficient; otherwise, false.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the feature index is outside the valid range.</exception>
    /// <remarks>
    /// <para>
    /// This method checks whether a specific feature is actively contributing to the model's predictions
    /// by verifying if its corresponding coefficient is non-zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a specific input feature affects the model's predictions.
    ///
    /// You provide the position (index) of a feature, and the method tells you whether that feature
    /// is actually used in making predictions. A feature is considered "used" if its coefficient
    /// is not zero.
    ///
    /// For example, if feature #3 has a coefficient of 0, this method would return false because
    /// that feature doesn't affect the model's output.
    ///
    /// This is useful when you want to check a specific feature's importance rather than
    /// getting all important features at once.
    /// </para>
    /// </remarks>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= Coefficients.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                $"Feature index must be between 0 and {Coefficients.Length - 1}");
        }

        return !NumOps.Equals(Coefficients[featureIndex], NumOps.Zero);
    }

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters (coefficients and intercept).</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has an incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the model's parameters in-place. The parameters vector should contain
    /// coefficients followed by the intercept (if the model uses one).
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the model's parameters directly.
    ///
    /// Unlike WithParameters() which creates a new model, this method modifies the current model.
    /// The parameters include the coefficients (how much each feature affects the prediction) and
    /// the intercept (the baseline value).
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ExpectedParameterCount)
        {
            throw new ArgumentException($"Expected {ExpectedParameterCount} parameters, but got {parameters.Length}", nameof(parameters));
        }

        // Extract and set coefficients
        for (int i = 0; i < Coefficients.Length; i++)
        {
            Coefficients[i] = parameters[i];
        }

        // Set the intercept if used
        if (Options.UseIntercept)
        {
            Intercept = parameters[Coefficients.Length];
        }
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to activate.</param>
    /// <remarks>
    /// <para>
    /// This method sets the coefficients for the specified features to their current values
    /// and sets all other coefficients to zero, effectively activating only the specified features.
    /// </para>
    /// <para><b>For Beginners:</b> This method selectively activates only certain features.
    ///
    /// You provide a list of feature positions (indices), and the method will:
    /// - Keep the coefficients for those features
    /// - Set all other feature coefficients to zero
    ///
    /// This is useful for feature selection, where you want to use only a subset of available features.
    /// </para>
    /// </remarks>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Create a set for fast lookup
        var activeSet = new HashSet<int>(featureIndices);

        // Set coefficients to zero for inactive features
        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (!activeSet.Contains(i))
            {
                Coefficients[i] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Gets the feature importance scores as a dictionary.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    /// <remarks>
    /// <para>
    /// This method returns feature importance scores based on the absolute values of coefficients.
    /// If feature names are not available, it uses indices as names (e.g., "Feature_0", "Feature_1").
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you which features are most important.
    ///
    /// It returns a dictionary where:
    /// - Keys are feature names (or "Feature_0", "Feature_1", etc. if names aren't set)
    /// - Values are importance scores (higher means more important)
    ///
    /// In regression models, importance is typically based on the absolute value of coefficients.
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var importances = CalculateFeatureImportances();
        var result = new Dictionary<string, T>();

        for (int i = 0; i < importances.Length; i++)
        {
            string featureName = FeatureNames != null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[featureName] = importances[i];
        }

        return result;
    }

    /// <summary>
    /// Creates a deep copy of the regression model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and options.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the regression model with the same parameters
    /// and configuration options as the current instance.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact independent copy of your model.
    ///
    /// The copy has the same:
    /// - Coefficients (weights for each feature)
    /// - Intercept (base prediction value)
    /// - Configuration options (like regularization settings)
    ///
    /// But it's completely separate from the original model - changes to one won't affect the other.
    ///
    /// This is useful when you want to:
    /// - Experiment with modifying a model without affecting the original
    /// - Create multiple similar models to use in different contexts
    /// - Save a "checkpoint" of your model before making changes
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        // The most reliable way to create a deep copy is through serialization/deserialization
        byte[] serialized = Serialize();

        // Create a new instance of the same type as this network
        var copy = CreateNewInstance();

        // Load the serialized data into the new instance
        copy.Deserialize(serialized);

        return copy;
    }

    /// <summary>
    /// Creates a new instance of the same type as this neural network.
    /// </summary>
    /// <returns>A new instance of the same neural network type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a blank version of the same type of neural network.
    ///
    /// It's used internally by methods like DeepCopy and Clone to create the right type of
    /// network before copying the data into it.
    /// </para>
    /// </remarks>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance();

    /// <summary>
    /// Creates a clone of the regression model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and options.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the regression model with the same parameters and configuration
    /// options as the current instance. Derived classes should override this method to provide proper cloning
    /// behavior specific to their implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact independent copy of your model.
    ///
    /// Cloning a model means creating a new model that's exactly the same as the original,
    /// including all its learned parameters and settings. However, the clone is independent -
    /// changes to one model won't affect the other.
    ///
    /// Think of it like photocopying a document - the copy has all the same information,
    /// but you can mark up the copy without changing the original.
    ///
    /// Note: Specific regression algorithms will customize this method to ensure all their
    /// unique properties are properly copied.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        // By default, Clone behaves the same as DeepCopy
        return DeepCopy();
    }

    public virtual int ParameterCount
    {
        get { return ExpectedParameterCount; }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For regression models, the default loss function is Mean Squared Error (MSE), which measures
    /// the average squared difference between predicted and actual values. This can be customized
    /// by passing a different loss function to the constructor.
    /// </para>
    /// <para><b>For Beginners:</b> This property specifies how the model measures prediction errors.
    ///
    /// Mean Squared Error (MSE) is the standard loss function for regression because it:
    /// - Penalizes large errors more than small errors (due to squaring)
    /// - Provides smooth gradients for optimization
    /// - Has a clear mathematical interpretation (average squared distance from truth)
    ///
    /// You can customize this by passing your own loss function when creating the model.
    /// The loss function is used during gradient computation to determine how to adjust parameters
    /// to improve predictions.
    /// </para>
    /// </remarks>
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// This method computes gradients for regression models using numerical differentiation.
    /// For linear regression models, the gradient of the loss with respect to coefficients is:
    /// ∂L/∂w = (1/n) * X^T * (predictions - targets)
    /// where X is the input matrix, predictions are model outputs, and targets are the desired outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how to adjust the model's parameters to reduce errors.
    ///
    /// Gradients tell us:
    /// - Which direction to change each parameter (positive or negative)
    /// - How much to change each parameter (magnitude)
    ///
    /// For regression, we compute gradients by:
    /// 1. Making a prediction with current parameters
    /// 2. Computing the error using the loss function
    /// 3. Calculating how much each parameter contributed to the error
    ///
    /// These gradients are then used by ApplyGradients() to update the parameters and improve predictions.
    /// </para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Note: Linear regression uses closed-form least-squares solution (MSE-based).
        // The lossFunction parameter is ignored because the gradient computation is specific
        // to the MSE objective function: ∇L = (1/n) * X^T * (predictions - target)
        // For custom loss functions, use NonLinearRegressionBase instead.

        // Make predictions
        var predictions = Predict(input);

        // Compute prediction errors (MSE gradient component)
        var errors = predictions.Subtract(target);

        // Compute gradients using MSE formula: (1/n) * X^T * errors
        var n = NumOps.FromDouble(input.Rows);
        var gradCoefficients = input.Transpose().Multiply(errors).Divide(n);

        // Build full gradient vector (coefficients + intercept)
        var gradients = new Vector<T>(ExpectedParameterCount);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            gradients[i] = gradCoefficients[i];
        }

        // Gradient for intercept is mean of errors
        if (Options.UseIntercept)
        {
            T interceptGrad = NumOps.Zero;
            for (int i = 0; i < errors.Length; i++)
            {
                interceptGrad = NumOps.Add(interceptGrad, errors[i]);
            }
            gradients[Coefficients.Length] = NumOps.Divide(interceptGrad, n);
        }

        return gradients;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// This method updates the model's parameters (coefficients and intercept) using the computed gradients.
    /// The update rule is: parameter_new = parameter_old - learningRate * gradient
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters to improve predictions.
    ///
    /// Think of it like adjusting a recipe:
    /// - The gradients tell you which ingredients to increase or decrease
    /// - The learning rate controls how big the adjustments are
    /// - Small learning rates = slow, careful adjustments
    /// - Large learning rates = fast, aggressive adjustments (but risk overshooting)
    ///
    /// The method:
    /// 1. Takes the gradients (directions to improve)
    /// 2. Scales them by the learning rate (controls step size)
    /// 3. Subtracts them from current parameters (gradient descent moves opposite to gradient)
    ///
    /// After calling this method, the model should make better predictions (lower loss).
    /// </para>
    /// </remarks>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients.Length != ExpectedParameterCount)
        {
            throw new ArgumentException($"Expected {ExpectedParameterCount} gradients, but got {gradients.Length}", nameof(gradients));
        }

        // Get current parameters
        var currentParams = GetParameters();

        // Apply gradient descent: params = params - learningRate * gradients
        var newParams = new Vector<T>(currentParams.Length);
        for (int i = 0; i < currentParams.Length; i++)
        {
            newParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        // Use SetParameters to update all model state
        SetParameters(newParams);
    }

    /// <summary>
    /// Saves the regression model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method saves the complete state of the regression model, including coefficients, intercept,
    /// and all configuration options, to a file.
    /// </para>
    /// <para><b>For Beginners:</b> This saves your trained model to a file so you can use it later.
    ///
    /// Think of it like saving a recipe:
    /// - It captures all the model's learned parameters (coefficients and intercept)
    /// - It saves the configuration settings used to train the model
    /// - You can load it later to make predictions without retraining
    ///
    /// This is useful for:
    /// - Deploying models to production
    /// - Sharing models with others
    /// - Avoiding the need to retrain on the same data
    /// </para>
    /// </remarks>
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Loads a regression model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    /// <remarks>
    /// <para>
    /// This method loads the complete state of the regression model from a file, including coefficients,
    /// intercept, and all configuration options.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously trained model from a file.
    ///
    /// It's like loading a saved recipe:
    /// - It restores all the model's learned parameters
    /// - It restores the configuration settings
    /// - The model is immediately ready to make predictions
    ///
    /// This allows you to:
    /// - Reuse models without retraining
    /// - Share models with others
    /// - Deploy models to production environments
    /// </para>
    /// </remarks>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);
        Deserialize(serializedData);
    }

    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    public virtual void SaveState(Stream stream)
    {
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    public virtual void LoadState(Stream stream)
    {
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        byte[] serializedData = memoryStream.ToArray();
        Deserialize(serializedData);
    }

    #region IJitCompilable Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Regression models support JIT compilation for accelerated inference.
    /// The computation graph represents the linear regression formula:
    /// output = input @ coefficients + intercept (if HasIntercept)
    /// </para>
    /// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation optimizes the model for faster predictions.
    ///
    /// Instead of performing matrix operations step-by-step at runtime, JIT compilation:
    /// - Analyzes the model's structure ahead of time
    /// - Generates optimized native code
    /// - Results in 5-10x faster predictions
    ///
    /// This is especially beneficial for:
    /// - Real-time prediction systems
    /// - High-throughput applications
    /// - Batch processing of many predictions
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation => true;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Exports the regression model as a computation graph for JIT compilation.
    /// The graph represents: output = input @ coefficients + intercept
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the regression model into a computation graph.
    ///
    /// A computation graph is like a recipe that describes:
    /// 1. Take input features (a matrix)
    /// 2. Multiply by learned coefficients
    /// 3. Add intercept (if the model uses one)
    /// 4. Return predictions
    ///
    /// The JIT compiler uses this graph to:
    /// - Optimize the operations
    /// - Combine steps where possible
    /// - Generate fast native code
    ///
    /// For linear regression: y = X * w + b
    /// - X: input features
    /// - w: coefficients (weights)
    /// - b: intercept (bias)
    /// </para>
    /// </remarks>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        // Validation: Ensure model is trained
        if (Coefficients == null || Coefficients.Length == 0)
        {
            throw new InvalidOperationException("Cannot export computation graph: Model has not been trained yet.");
        }

        // Create input node (placeholder for input features)
        // Shape: [batch_size, feature_count]
        var inputShape = new int[] { 1, Coefficients.Length };
        var inputTensor = new Tensor<T>(inputShape);
        var inputNode = new ComputationNode<T>(inputTensor);
        inputNodes.Add(inputNode);

        // Convert coefficients Vector<T> to Tensor<T>
        // Shape: [feature_count, 1] for matrix multiplication
        var coeffShape = new int[] { Coefficients.Length, 1 };
        var coeffData = new T[Coefficients.Length];
        for (int i = 0; i < Coefficients.Length; i++)
        {
            coeffData[i] = Coefficients[i];
        }
        var coeffTensor = new Tensor<T>(coeffShape, new Vector<T>(coeffData));
        var coeffNode = new ComputationNode<T>(coeffTensor);

        // MatMul: input @ coefficients
        // Result shape: [batch_size, 1]
        var outputNode = TensorOperations<T>.MatrixMultiply(inputNode, coeffNode);

        // Add intercept if used
        if (HasIntercept)
        {
            // Convert scalar intercept to Tensor<T>
            // Shape: [1, 1] (scalar broadcasted)
            var interceptShape = new int[] { 1, 1 };
            var interceptData = new T[] { Intercept };
            var interceptTensor = new Tensor<T>(interceptShape, new Vector<T>(interceptData));
            var interceptNode = new ComputationNode<T>(interceptTensor);

            // Add: (input @ coefficients) + intercept
            outputNode = TensorOperations<T>.Add(outputNode, interceptNode);
        }

        return outputNode;
    }

    #endregion
}
