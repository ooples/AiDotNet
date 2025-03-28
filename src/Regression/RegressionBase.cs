global using AiDotNet.Factories;

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
    protected IRegularization<T> Regularization { get; private set; }

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
    /// Initializes a new instance of the RegressionBase class with the specified options and regularization.
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
    protected RegressionBase(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        Regularization = regularization ?? new NoRegularization<T>();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new RegressionOptions<T>();
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
        var modelDataDict = JsonConvert.DeserializeObject<Dictionary<string, object>>(modelDataString);

        if (modelDataDict == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        Coefficients = new Vector<T>((T[])modelDataDict["Coefficients"]);
        Intercept = (T)modelDataDict["Intercept"];

        var regularizationOptionsJson = JsonConvert.SerializeObject(modelDataDict["RegularizationOptions"]);
        var regularizationOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regularizationOptionsJson) 
            ?? throw new InvalidOperationException("Deserialization failed: Unable to deserialize regularization options.");
    
        Regularization = RegularizationFactory.CreateRegularization<T>(regularizationOptions);
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
}