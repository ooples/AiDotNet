namespace AiDotNet.Regression;

/// <summary>
/// Represents a robust regression model that is resistant to outliers in the data.
/// </summary>
/// <remarks>
/// <para>
/// Robust regression provides an alternative to traditional regression methods when data contains 
/// outliers or influential observations. By using weight functions, it reduces the influence of outliers 
/// on the final model. This implementation uses an iterative reweighted least squares approach to 
/// estimate coefficients that are less affected by extreme values.
/// </para>
/// <para><b>For Beginners:</b> Traditional regression models can be heavily influenced by outliers 
/// (unusual data points that don't follow the general pattern). 
/// 
/// Think of robust regression like a smart voting system:
/// - It identifies which data points are "suspicious" (potential outliers)
/// - It gives these points less influence (lower weight) in determining the final model
/// - It focuses more on the reliable data points to find the true pattern
/// 
/// For example, if most houses in a neighborhood cost $200,000-300,000, but one special mansion costs 
/// $2 million, robust regression would recognize this as an outlier and reduce its influence when 
/// predicting house prices based on size or features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RobustRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Gets the configuration options used by this robust regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the robust regression algorithm, including the type of weight 
    /// function used, convergence criteria, and other parameters that affect how outliers are handled.
    /// </para>
    /// <para><b>For Beginners:</b> These settings determine how the model behaves:
    /// 
    /// - They control how the model identifies outliers
    /// - They set how many attempts (iterations) the model makes to improve its accuracy
    /// - They determine when the model decides it's "good enough" and stops improving
    /// </para>
    /// </remarks>
    private readonly RobustRegressionOptions<T> _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="RobustRegression{T}"/> class with the specified options 
    /// and regularization method.
    /// </summary>
    /// <param name="options">The configuration options for the robust regression. If null, default options are used.</param>
    /// <param name="regularization">The regularization method to apply. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new robust regression model with the specified configuration options and 
    /// regularization method. If options are not provided, default values are used. Regularization helps prevent 
    /// overfitting by adding penalties for model complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up a new robust regression model with your chosen settings.
    /// 
    /// Think of it like configuring a new smartphone:
    /// - You can use the default settings (by not specifying options)
    /// - Or you can customize how it works (by providing specific options)
    /// - You can also add regularization, which helps prevent the model from memorizing the data
    ///   instead of learning patterns (similar to adding parental controls)
    /// </para>
    /// </remarks>
    public RobustRegression(RobustRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new RobustRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the robust regression model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input matrix where rows represent observations and columns represent features.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <remarks>
    /// <para>
    /// This method implements the iterative reweighted least squares algorithm for robust regression. It starts with 
    /// an initial estimate using standard regression, then iteratively recalculates weights based on residuals and 
    /// performs weighted regression until convergence or the maximum number of iterations is reached.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions based on your data.
    /// 
    /// The training process works like this:
    /// 
    /// 1. Start with a regular regression (like finding the best straight line through your data points)
    /// 2. Calculate how far each point is from this line (called "residuals")
    /// 3. Give lower weights to points that are far from the line (potential outliers)
    /// 4. Create a new line that pays more attention to points with higher weights
    /// 5. Repeat steps 2-4 until the line stops changing significantly
    /// 
    /// This way, the final model is less influenced by outliers and better represents the true pattern in your data.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;
        // Note: Robust regression handles regularization through the weight function,
        // not through data transformation

        // Initial regression estimate
        IRegression<T> initialRegression = _options.InitialRegression ?? new MultipleRegression<T>();
        initialRegression.Train(x, y);
        Vector<T> parameters = initialRegression.GetParameters();

        // Extract coefficients and intercept from parameters
        if (Options.UseIntercept)
        {
            // If using intercept, the last element is the intercept
            Coefficients = new Vector<T>(parameters.Length - 1);
            for (int i = 0; i < parameters.Length - 1; i++)
            {
                Coefficients[i] = parameters[i];
            }
            Intercept = parameters[parameters.Length - 1];
        }
        else
        {
            // If not using intercept, all parameters are coefficients
            Coefficients = parameters;
            Intercept = NumOps.Zero;
        }

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            Vector<T> residuals = y.Subtract(Predict(x));
            Vector<T> weights = WeightFunctionHelper<T>.CalculateWeights(residuals, _options.WeightFunction, _options.TuningConstant);
            // Weighted least squares
            Matrix<T> weightedX = x.PointwiseMultiply(weights);
            Vector<T> weightedY = y.PointwiseMultiply(weights);
            var wls = new MultipleRegression<T>();
            wls.Train(weightedX, weightedY);
            Vector<T> newCoefficients = wls.Coefficients;
            T newIntercept = wls.Intercept;
            // Check for convergence
            if (IsConverged(Coefficients, newCoefficients, Intercept, newIntercept))
            {
                break;
            }
            Coefficients = newCoefficients;
            Intercept = newIntercept;
        }
    }

    /// <summary>
    /// Determines whether the model has converged based on changes in coefficients and intercept.
    /// </summary>
    /// <param name="oldCoefficients">The coefficients from the previous iteration.</param>
    /// <param name="newCoefficients">The coefficients from the current iteration.</param>
    /// <param name="oldIntercept">The intercept from the previous iteration.</param>
    /// <param name="newIntercept">The intercept from the current iteration.</param>
    /// <returns>
    /// <c>true</c> if the changes in coefficients and intercept are within the specified tolerance; otherwise, <c>false</c>.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method checks whether the iterative robust regression algorithm has converged by comparing the 
    /// coefficients and intercept between consecutive iterations. If the changes are smaller than the specified 
    /// tolerance, the algorithm is considered to have converged and iteration stops.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides when the model has finished improving.
    /// 
    /// Think of it like baking a cake:
    /// - You check periodically to see if it's done
    /// - Instead of using a toothpick, the model checks if its values are still changing significantly
    /// - If the changes become very small (less than the tolerance), it's considered "done"
    /// - This prevents the model from running forever and saves computation time
    /// 
    /// For example, if our model's slope changed from 2.5001 to 2.5002, and our tolerance is 0.001,
    /// we would consider it converged because the change (0.0001) is smaller than our tolerance.
    /// </para>
    /// </remarks>
    private bool IsConverged(Vector<T> oldCoefficients, Vector<T> newCoefficients, T oldIntercept, T newIntercept)
    {
        T tolerance = NumOps.FromDouble(_options.Tolerance);

        for (int i = 0; i < oldCoefficients.Length; i++)
        {
            if (NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(oldCoefficients[i], newCoefficients[i])), tolerance))
            {
                return false;
            }
        }
        return NumOps.LessThanOrEquals(NumOps.Abs(NumOps.Subtract(oldIntercept, newIntercept)), tolerance);
    }

    /// <summary>
    /// Gets the type of this regression model.
    /// </summary>
    /// <returns>The model type enum value representing robust regression.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the enum value that identifies this model as a robust regression model. This is used 
    /// for model identification in serialization/deserialization and for logging purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply tells the system what kind of model this is.
    /// 
    /// It's like a name tag for the model that says "I am a robust regression model."
    /// This is useful when:
    /// - Saving the model to a file
    /// - Loading a model from a file
    /// - Logging information about the model
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.RobustRegression;

    /// <summary>
    /// Serializes the robust regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the model, including its coefficients, intercept, and configuration options, into a 
    /// byte array. This enables the model to be saved to a file, stored in a database, or transmitted over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to computer memory so you can use it later.
    /// 
    /// Think of it like taking a snapshot of the model:
    /// - It captures all the important values and settings
    /// - It converts them into a format that can be easily stored
    /// - The resulting byte array can be saved to a file or database
    /// 
    /// This is useful when you want to:
    /// - Train the model once and use it many times
    /// - Share the model with others
    /// - Use the model in a different application
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);
        // Serialize RobustRegression specific options
        writer.Write(_options.TuningConstant);
        writer.Write(_options.MaxIterations);
        writer.Write(_options.Tolerance);
        writer.Write((int)_options.WeightFunction);
        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the robust regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model from a byte array created by the Serialize method. It restores 
    /// the model's coefficients, intercept, and configuration options, allowing a previously saved model 
    /// to be loaded and used for predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a saved model from computer memory.
    /// 
    /// Think of it like restoring a model from a snapshot:
    /// - It takes the byte array created by the Serialize method
    /// - It reconstructs all the important values and settings
    /// - The model is then ready to use for making predictions
    /// 
    /// This allows you to:
    /// - Use a previously trained model without retraining it
    /// - Load models that others have shared with you
    /// - Use the same model across different applications
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);
        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);
        // Deserialize RobustRegression specific options
        _options.TuningConstant = reader.ReadDouble();
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
        _options.WeightFunction = (WeightFunction)reader.ReadInt32();
    }

    /// <summary>
    /// Gets the model parameters (coefficients and intercept) as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method packages all the model's parameters into a single vector.
    /// 
    /// The returned vector combines:
    /// - All the coefficients (the weights for each feature)
    /// - The intercept (baseline value when all features are zero)
    /// 
    /// This is useful for optimization algorithms that need to work with all parameters at once.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Create a new vector with enough space for coefficients + intercept
        Vector<T> parameters = new Vector<T>(Coefficients.Length + 1);

        // Copy coefficients to the parameters vector
        for (int i = 0; i < Coefficients.Length; i++)
        {
            parameters[i] = Coefficients[i];
        }

        // Add the intercept as the last element
        parameters[Coefficients.Length] = Intercept;

        return parameters;
    }

    /// <summary>
    /// Creates a new model instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters (coefficients and intercept).</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new model using a provided set of parameters.
    /// 
    /// It takes a parameter vector (which combines coefficients and intercept) and:
    /// - Extracts the coefficients (all values except the last)
    /// - Extracts the intercept (the last value)
    /// - Creates a new model with these values
    /// 
    /// This allows you to try different parameter sets without changing the original model.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        if (parameters.Length != Coefficients.Length + 1)
        {
            throw new ArgumentException($"Expected {Coefficients.Length + 1} parameters, but got {parameters.Length}");
        }

        // Create a new instance of the model
        var newModel = (RobustRegression<T>)this.Clone();

        // Extract coefficients (all elements except the last)
        Vector<T> newCoefficients = new Vector<T>(parameters.Length - 1);
        for (int i = 0; i < parameters.Length - 1; i++)
        {
            newCoefficients[i] = parameters[i];
        }

        // Extract intercept (last element)
        T newIntercept = parameters[parameters.Length - 1];

        // Set the parameters in the new model
        newModel.Coefficients = newCoefficients;
        newModel.Intercept = newIntercept;

        return newModel;
    }

    /// <summary>
    /// Creates a new instance of the robust regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the robust regression model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the robust regression model with the same configuration
    /// options and regularization method as the current instance, but without copying the trained
    /// coefficients or intercept.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters.
    /// 
    /// Think of it like getting a blank template with the same settings, 
    /// but without any of the values that were learned from training data. The new model has the same:
    /// - Weight function (how outliers are handled)
    /// - Tuning constant (how sensitive the model is to outliers)
    /// - Maximum iterations (how many times it will try to improve)
    /// - Tolerance (when it decides it's "good enough")
    /// - Regularization settings (how it prevents overfitting)
    /// 
    /// But it doesn't have any of the coefficients or intercept values that were learned from data.
    /// 
    /// This is mainly used internally when doing things like cross-validation or 
    /// creating multiple similar models with different training data.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        return new RobustRegression<T>(_options, Regularization);
    }
}
