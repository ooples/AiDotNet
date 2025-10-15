global using System.Threading.Tasks;
using AiDotNet.Factories;

using AiDotNet.Interpretability;

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
public abstract class RegressionModelBase<T> : IRegressionModel<T>
{
    /// <summary>
    /// Gets the numeric operations for the specified type T.
    /// </summary>
    /// <value>
    /// An object that provides mathematical operations for type T.
    /// </value>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores feature indices that have been explicitly set as active through
    /// the SetActiveFeatureIndices method, overriding the automatic determination based
    /// on non-zero coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks which input features have been manually
    /// selected as important for the regression model, regardless of their coefficient values.
    /// 
    /// When set, these manually selected features take precedence over the automatic
    /// feature detection based on non-zero coefficients.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

    /// <summary>
    /// Gets the regression options.
    /// </summary>
    /// <value>
    /// Configuration options for the regression model.
    /// </value>
    protected RegressionOptions<T> Options { get; private set; }

    /// <summary>
    /// Gets the actual decomposition method that was successfully used in the most recent solve operation.
    /// </summary>
    /// <remarks>
    /// This property may return null if no decomposition has been performed yet or if the
    /// decomposition method wasn't retained after solving.
    /// </remarks>
    protected IMatrixDecomposition<T>? LastUsedDecomposition { get; private set; }

    /// <summary>
    /// Gets the name of the decomposition method that was successfully used in the most recent solve operation.
    /// </summary>
    protected string LastUsedDecompositionName { get; private set; } = "None";

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    /// <value>
    /// An object that implements regularization for the regression model.
    /// </value>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

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
    /// Gets the number of features that this model was trained on.
    /// </summary>
    public int FeatureCount { get; protected set; }

    /// <summary>
    /// Gets a value indicating whether the model includes an intercept term.
    /// </summary>
    /// <value>
    /// True if the model includes an intercept; otherwise, false.
    /// </value>
    public bool HasIntercept => Options.UseIntercept;

    /// <summary>
    /// Initializes a new instance of the RegressionModelBase class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    protected RegressionModelBase(RegressionOptions<T> options, IRegularization<T, Matrix<T>, Vector<T>> regularization)
    {
        Regularization = regularization;
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options;
        Coefficients = new Vector<T>(0);
        Intercept = NumOps.Zero;
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
    /// Predicts output values based on input features using the trained model coefficients.
    /// </summary>
    /// <param name="input">The input matrix containing only the selected features.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentException">Thrown when the input matrix doesn't match the expected feature count.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method applies the trained model to make predictions.
    /// It multiplies each feature by its respective coefficient and sums the results.
    /// If an intercept is used, it adds that constant value to each prediction.
    /// </remarks>
    public virtual Vector<T> Predict(Matrix<T> input)
    {
        // Validate that the model has been trained
        if (Coefficients.Length == 0 && !Options.UseIntercept)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // Get active feature indices
        var activeIndices = GetActiveFeatureIndices().ToArray();

        // Case 1: Input has exactly the number of features the model expects
        if (input.Columns == FeatureCount)
        {
            // Standard prediction
            var predictions = new Vector<T>(input.Rows);
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = NumOps.Zero;
            }

            for (int col = 0; col < input.Columns; col++)
            {
                for (int row = 0; row < input.Rows; row++)
                {
                    var contribution = NumOps.Multiply(input[row, col], Coefficients[col]);
                    predictions[row] = NumOps.Add(predictions[row], contribution);
                }
            }

            // Add intercept if used
            if (Options.UseIntercept)
            {
                for (int row = 0; row < predictions.Length; row++)
                {
                    predictions[row] = NumOps.Add(predictions[row], Intercept);
                }
            }

            return predictions;
        }
        // Case 2: Input has more columns than the model was trained on
        else if (activeIndices.Length > 0 && input.Columns > FeatureCount)
        {
            // Check if all active indices are within range of input columns
            foreach (int index in activeIndices)
            {
                if (index >= input.Columns)
                {
                    throw new ArgumentException(
                        $"Active feature index {index} is out of range for input with {input.Columns} columns.");
                }
            }

            // Create a prediction vector initialized with zeros
            var predictions = new Vector<T>(input.Rows);
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = NumOps.Zero;
            }

            // Use only the active features for prediction
            for (int i = 0; i < activeIndices.Length; i++)
            {
                int featureIndex = activeIndices[i];

                for (int row = 0; row < input.Rows; row++)
                {
                    var contribution = NumOps.Multiply(input[row, featureIndex], Coefficients[i]);
                    predictions[row] = NumOps.Add(predictions[row], contribution);
                }
            }

            // Add intercept if used
            if (Options.UseIntercept)
            {
                for (int row = 0; row < predictions.Length; row++)
                {
                    predictions[row] = NumOps.Add(predictions[row], Intercept);
                }
            }

            return predictions;
        }
        else
        {
            throw new ArgumentException(
                $"Input matrix has {input.Columns} columns but model expects {FeatureCount} features. " +
                "Make sure to apply appropriate feature selection during prediction.");
        }
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
                { "FeatureImportances", CalculateFeatureImportances().ToArray() },
                { "DecompositionUsed", LastUsedDecompositionName }
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
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including coefficients, intercept, and options,
    /// to a binary format for storage or transmission.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's state into a format that can be saved to disk
    /// or transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize common options
        writer.Write(Options.UseIntercept);

        // First get the active feature indices
        var activeIndices = GetActiveFeatureIndices().ToArray();

        // Write the number of active features
        writer.Write(activeIndices.Length);

        // Write the active feature indices
        foreach (var index in activeIndices)
        {
            writer.Write(index);
        }

        // Serialize coefficients (should match the number of active features)
        writer.Write(Coefficients.Length);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(Coefficients[i]));
        }

        // Serialize intercept
        writer.Write(Convert.ToDouble(Intercept));

        // Serialize regularization options
        var regularizationOptions = Regularization.GetOptions();
        writer.Write((int)regularizationOptions.Type);

        // Serialize decomposition method used
        writer.Write(LastUsedDecompositionName);

        // Serialize explicitly set active features flag
        bool hasExplicitActiveFeatures = _explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0;
        writer.Write(hasExplicitActiveFeatures);

        if (hasExplicitActiveFeatures)
        {
            writer.Write(_explicitlySetActiveFeatures?.Count ?? 0);
            foreach (var index in _explicitlySetActiveFeatures ?? [])
            {
                writer.Write(index);
            }
        }

        // Let derived classes serialize their specific data
        SerializeCore(writer);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the serialized data is corrupted or incompatible.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including coefficients,
    /// intercept, and options.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data), "Serialized data cannot be null.");
        }

        try
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);

            // Deserialize common options
            Options.UseIntercept = reader.ReadBoolean();

            // Read the number of active features
            int activeFeatureCount = reader.ReadInt32();

            // Read the active feature indices
            int[] activeIndices = new int[activeFeatureCount];
            for (int i = 0; i < activeFeatureCount; i++)
            {
                activeIndices[i] = reader.ReadInt32();
            }

            // Set FeatureCount from the active feature count
            FeatureCount = activeFeatureCount;

            // Deserialize coefficients
            int coefficientCount = reader.ReadInt32();
            Coefficients = new Vector<T>(coefficientCount);
            for (int i = 0; i < coefficientCount; i++)
            {
                Coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            // Verify that coefficient count matches active feature count
            if (coefficientCount != activeFeatureCount)
            {
                throw new InvalidOperationException(
                    $"Serialized model has {coefficientCount} coefficients but {activeFeatureCount} active features.");
            }

            // Deserialize intercept
            Intercept = NumOps.FromDouble(reader.ReadDouble());

            // Deserialize regularization options
            var regType = (RegularizationType)reader.ReadInt32();

            // Recreate regularization with the deserialized options
            var regOptions = new RegularizationOptions
            {
                Type = regType
            };
            Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regOptions);

            LastUsedDecompositionName = reader.ReadString();

            // Deserialize explicitly set active features
            bool hasExplicitActiveFeatures = reader.ReadBoolean();

            if (hasExplicitActiveFeatures)
            {
                _explicitlySetActiveFeatures = new HashSet<int>();
                int count = reader.ReadInt32();

                for (int i = 0; i < count; i++)
                {
                    int featureIndex = reader.ReadInt32();
                    _explicitlySetActiveFeatures.Add(featureIndex);
                }
            }
            else
            {
                _explicitlySetActiveFeatures = null;
            }

            // Let derived classes deserialize their specific data
            DeserializeCore(reader);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Failed to deserialize model data. The data may be corrupted or incompatible with this model version.", ex);
        }
    }

    /// <summary>
    /// Serializes model-specific data to the binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method should be overridden by derived classes to serialize their specific data.
    /// The base implementation does nothing.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method allows specific regression model types to save additional data
    /// that isn't part of the common regression model structure. Different regression
    /// algorithms might need to store different types of information.
    /// </para>
    /// </remarks>
    protected virtual void SerializeCore(BinaryWriter writer)
    {
        // Base implementation does nothing
        // Derived classes should override this to serialize their specific data
    }

    /// <summary>
    /// Deserializes model-specific data from the binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method should be overridden by derived classes to deserialize their specific data.
    /// The base implementation does nothing.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method allows specific regression model types to load additional data
    /// that isn't part of the common regression model structure. It should read
    /// exactly what was written by the corresponding SerializeCore method.
    /// </para>
    /// </remarks>
    protected virtual void DeserializeCore(BinaryReader reader)
    {
        // Base implementation does nothing
        // Derived classes should override this to deserialize their specific data
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
        // Reset the last used decomposition tracker
        LastUsedDecompositionName = "None";
        LastUsedDecomposition = null;

        // Check if regularization should be applied
        var regOptions = Regularization.GetOptions();
        bool shouldApplyRegularization =
            regOptions.Type != RegularizationType.None &&
            regOptions.Strength > 0.0;

        // Apply regularization if needed
        if (shouldApplyRegularization)
        {
            // Apply regularization to input features
            a = Regularization.Regularize(a);
        }

        // Use specified decomposition method if provided
        if (Options.DecompositionMethod != null)
        {
            try
            {
                var solution = Options.DecompositionMethod.Solve(b);
                LastUsedDecompositionName = Options.DecompositionMethod.GetType().Name;
                LastUsedDecomposition = Options.DecompositionMethod;
                return solution;
            }
            catch (Exception ex)
            {
                // If fallbacks are not allowed, rethrow the exception
                if (!Options.AllowDecompositionFallbacks)
                {
                    throw new InvalidOperationException(
                        $"The specified decomposition method ({Options.DecompositionMethod.GetType().Name}) failed and fallbacks are disabled.", ex);
                }
                // Otherwise continue to fallback methods
            }
        }

        // Attempt different solution strategies if allowed
        return SolveWithFallbackStrategies(a, b);
    }

    /// <summary>
    /// Attempts to solve a linear system using multiple decomposition strategies.
    /// </summary>
    /// <param name="a">The coefficient matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector.</returns>
    private Vector<T> SolveWithFallbackStrategies(Matrix<T> a, Vector<T> b)
    {
        var aTa = a.Transpose().Multiply(a);
        var aTb = a.Transpose().Multiply(b);

        // Determine epsilon for regularization
        var regOptions = Regularization.GetOptions();
        double epsilonBase = regOptions.Strength > 0 ? regOptions.Strength : 1e-8;
        var epsilon = NumOps.FromDouble(epsilonBase * 1e-2);
        string lastError = string.Empty;

        // Define a list of solution strategies to try in order
        var strategies = new List<(string Name, Func<(Vector<T> Solution, IMatrixDecomposition<T> Decomposition)> SolveMethod)>
        {
            // 1. First try Normal decomposition (your original approach)
            ("Normal", () => {
                var decomp = new NormalDecomposition<T>(aTa);
                return (decomp.Solve(aTb), decomp);
            }),
        
            // 2. Try Cholesky decomposition (also fast)
            ("Cholesky", () => {
                var decomp = new CholeskyDecomposition<T>(aTa);
                return (decomp.Solve(aTb), decomp);
            }),
        
            // 3. Try Cholesky with regularization
            ("Regularized Cholesky", () => {
                var regularizedMatrix = aTa.Clone();
                for (int i = 0; i < regularizedMatrix.Rows; i++)
                {
                    regularizedMatrix[i, i] = NumOps.Add(regularizedMatrix[i, i], epsilon);
                }
                var decomp = new CholeskyDecomposition<T>(regularizedMatrix);
                return (decomp.Solve(aTb), decomp);
            }),
        
            // 4. Try QR decomposition
            ("QR", () => {
                var decomp = new QrDecomposition<T>(a);
                return (decomp.Solve(b), decomp);
            }),
        
            // 5. Try LU decomposition
            ("LU", () => {
                var decomp = new LuDecomposition<T>(aTa);
                return (decomp.Solve(aTb), decomp);
            }),
        
            // 6. Last resort: SVD (most robust but slowest)
            ("SVD", () => {
                var decomp = new SvdDecomposition<T>(a);
                return (decomp.Solve(b), decomp);
            })
        };

        // Try each strategy in sequence until one succeeds
        foreach (var (name, method) in strategies)
        {
            try
            {
                var (solution, decomposition) = method();

                // Record which decomposition was used
                LastUsedDecompositionName = name;
                LastUsedDecomposition = decomposition;

                return solution;
            }
            catch (Exception ex)
            {
                // Remember the last error but continue to next strategy
                lastError = $"{name} decomposition failed: {ex.Message}";
            }
        }

        // If we get here, all methods failed
        throw new InvalidOperationException(
            "Unable to solve the linear system. All decomposition methods failed. Last error: " + lastError);
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
        // Get active feature indices
        var activeIndices = GetActiveFeatureIndices().ToArray();

        // Ensure that coefficient count matches active feature count
        if (Coefficients.Length != activeIndices.Length)
        {
            throw new InvalidOperationException(
                $"Model has {Coefficients.Length} coefficients but {activeIndices.Length} active features.");
        }

        // Create a new vector with enough space for active coefficients + intercept (if used)
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
        // Get active feature indices
        var activeIndices = GetActiveFeatureIndices().ToArray();

        // Calculate expected parameter count
        int expectedParamCount = activeIndices.Length + (Options.UseIntercept ? 1 : 0);

        if (parameters.Length != expectedParamCount)
        {
            throw new ArgumentException(
                $"Expected {expectedParamCount} parameters ({activeIndices.Length} active features + {(Options.UseIntercept ? 1 : 0)} intercept), but got {parameters.Length}");
        }

        // Create a new instance of the model
        var newModel = (RegressionModelBase<T>)Clone();

        // Set the coefficients in the new model
        newModel.Coefficients = new Vector<T>(activeIndices.Length);
        for (int i = 0; i < activeIndices.Length; i++)
        {
            newModel.Coefficients[i] = parameters[i];
        }

        // Set the FeatureCount to match active feature count
        newModel.FeatureCount = activeIndices.Length;

        // Set the intercept if used
        if (Options.UseIntercept)
        {
            newModel.Intercept = parameters[activeIndices.Length];
        }
        else
        {
            newModel.Intercept = NumOps.Zero;
        }

        // Copy the active feature set to the new model
        if (_explicitlySetActiveFeatures != null)
        {
            newModel.SetActiveFeatureIndices(_explicitlySetActiveFeatures);
        }
        else
        {
            newModel.SetActiveFeatureIndices(activeIndices);
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
        // If we have explicitly set active features, return those
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0)
        {
            // Order the features and yield each one individually
            foreach (int featureIndex in _explicitlySetActiveFeatures.OrderBy(i => i))
            {
                yield return featureIndex;
            }

            // Exit early - we only want to return the explicitly set features
            yield break;
        }

        // Otherwise, continue with the existing implementation
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

        // If feature index is explicitly set as active, return true immediately
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Contains(featureIndex))
        {
            return true;
        }

        // If explicitly set active features exist but don't include this index, it's not used
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0)
        {
            return false;
        }

        return !NumOps.Equals(Coefficients[featureIndex], NumOps.Zero);
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

    /// <summary>
    /// Sets which features should be considered active in the model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is outside the valid range.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which features should be considered active in the
    /// regression model, overriding the automatic determination based on non-zero coefficients.
    /// Any features not included in the provided collection will be considered inactive,
    /// regardless of their coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you manually tell the model which input features
    /// are important, regardless of what the model learned during training.
    /// 
    /// For example, if you have 10 features but want to focus on only features 2, 5, and 7,
    /// you can use this method to specify exactly those features. After setting these features:
    /// - Only these specific features will be reported as active by GetActiveFeatureIndices()
    /// - Only these features will return true when checked with IsFeatureUsed()
    /// - This selection will persist when the model is saved and loaded
    /// 
    /// This can be useful for:
    /// - Feature selection experiments (testing different feature subsets)
    /// - Simplifying model interpretation
    /// - Ensuring consistency across different models
    /// - Highlighting specific features you know are important from domain expertise
    /// </para>
    /// </remarks>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices), "Feature indices cannot be null.");
        }

        // Initialize the hash set if it doesn't exist
        _explicitlySetActiveFeatures ??= [];

        // Clear existing explicitly set features
        _explicitlySetActiveFeatures.Clear();

        // Add the new feature indices - with special handling when coefficients aren't initialized
        if (Coefficients.Length == 0)
        {
            // When coefficients aren't initialized yet, we can only validate that indices are non-negative
            foreach (var index in featureIndices)
            {
                if (index < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(featureIndices),
                        $"Feature index {index} cannot be negative");
                }

                _explicitlySetActiveFeatures.Add(index);
            }
        }
        else
        {
            // Normal case - coefficients are initialized
            foreach (var index in featureIndices)
            {
                if (index < 0 || index >= Coefficients.Length)
                {
                    throw new ArgumentOutOfRangeException(nameof(featureIndices),
                        $"Feature index {index} must be between 0 and {Coefficients.Length - 1}");
                }

                _explicitlySetActiveFeatures.Add(index);
            }
        }
    }

    /// <summary>
    /// Sets the parameters of the model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method sets the model coefficients and intercept from a parameter vector.
    /// The parameters should be in the same format as returned by GetParameters.
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        // Calculate expected parameter count
        int expectedParamCount = Coefficients.Length + (Options.UseIntercept ? 1 : 0);
        
        if (parameters.Length != expectedParamCount)
        {
            throw new ArgumentException($"Expected {expectedParamCount} parameters, got {parameters.Length}");
        }

        // Set coefficients
        for (int i = 0; i < Coefficients.Length; i++)
        {
            Coefficients[i] = parameters[i];
        }

        // Set intercept if used
        if (Options.UseIntercept)
        {
            Intercept = parameters[Coefficients.Length];
        }
    }

    #region IInterpretableModel Implementation

    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
    protected Vector<int> _sensitiveFeatures;
    protected readonly List<FairnessMetric> _fairnessMetrics = new();
    protected IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Matrix<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Matrix<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Matrix<T> input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Matrix<T> input, Vector<T> desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(Matrix<T> input, Vector<T> prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
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
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> model)
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
}