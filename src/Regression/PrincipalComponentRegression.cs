namespace AiDotNet.Regression;

/// <summary>
/// Implements Principal Component Regression (PCR), a technique that combines principal component analysis (PCA) 
/// with linear regression to handle multicollinearity in the predictor variables.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Principal Component Regression works by first performing principal component analysis (PCA) on the predictor 
/// variables to reduce their dimensionality, then using these principal components as predictors in a linear 
/// regression model. This approach is particularly useful when dealing with multicollinearity (high correlation 
/// among predictor variables) or when the number of predictors is large relative to the number of observations.
/// </para>
/// <para>
/// The algorithm first centers and scales the data, performs PCA to extract principal components, selects a 
/// subset of these components based on either a fixed number or explained variance ratio, and then performs 
/// linear regression using the selected components.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of PCR as a two-step process: first, it finds the most important patterns in your input data 
/// (principal components), then it uses these patterns instead of the original variables to build a regression model. 
/// This can help when your original variables are highly related to each other (multicollinear), which can cause 
/// problems in standard regression.
/// </para>
/// </remarks>
public class PrincipalComponentRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the principal component regression model.
    /// </summary>
    /// <value>
    /// Contains settings like the number of components to use or the explained variance ratio threshold.
    /// </value>
    private readonly PrincipalComponentRegressionOptions<T> _options;

    /// <summary>
    /// The principal components extracted from the training data.
    /// </summary>
    /// <value>
    /// A matrix where each column represents a principal component.
    /// </value>
    private Matrix<T> _components;

    /// <summary>
    /// The mean of each predictor variable used for centering.
    /// </summary>
    /// <value>
    /// A vector containing the mean value of each predictor variable.
    /// </value>
    private Vector<T> _xMean;

    /// <summary>
    /// The mean of the target variable used for centering.
    /// </summary>
    /// <value>
    /// A vector containing the mean value of the target variable.
    /// </value>
    private Vector<T> _yMean;

    /// <summary>
    /// The standard deviation of each predictor variable used for scaling.
    /// </summary>
    /// <value>
    /// A vector containing the standard deviation of each predictor variable.
    /// </value>
    private Vector<T> _xStd;

    /// <summary>
    /// The standard deviation of the target variable used for scaling.
    /// </summary>
    /// <value>
    /// The standard deviation value of the target variable.
    /// </value>
    private T _yStd;

    /// <summary>
    /// Initializes a new instance of the PrincipalComponentRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the PCR model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the PCR model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public PrincipalComponentRegression(PrincipalComponentRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new PrincipalComponentRegressionOptions<T>();
        _components = new Matrix<T>(0, 0);
        _xMean = Vector<T>.Empty();
        _yMean = Vector<T>.Empty();
        _xStd = Vector<T>.Empty();
        _yStd = NumOps.Zero;
    }

    /// <summary>
    /// Trains the principal component regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Validates the input data
    /// 2. Centers and scales the data
    /// 3. Performs principal component analysis (PCA) on the predictor variables
    /// 4. Selects the appropriate number of principal components
    /// 5. Projects the data onto the selected principal components
    /// 6. Performs linear regression on the projected data
    /// 7. Transforms the coefficients back to the original space
    /// 8. Applies regularization to the coefficients
    /// 9. Adjusts for scaling and calculates the intercept
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Training is the process where the model learns from your data. The PCR algorithm first centers and scales
    /// your data (makes all variables have similar ranges), then finds the most important patterns (principal components)
    /// in your input features. It selects a subset of these patterns based on your settings, and uses them to build
    /// a regression model. Finally, it converts the model back to work with your original variables.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);

        // Center and scale the data
        (Matrix<T> xScaled, Vector<T> yScaled, _xMean, _xStd, _yStd) = RegressionHelper<T>.CenterAndScale(x, y);

        // Perform PCA
        (Matrix<T> components, Vector<T> explainedVariance) = PerformPCA(xScaled);

        // Select number of components
        int numComponents = SelectNumberOfComponents(explainedVariance);
        _components = components.Submatrix(0, 0, components.Rows, numComponents);

        // Project data onto principal components
        Matrix<T> xProjected = xScaled.Multiply(_components);

        // Perform linear regression on projected data
        Vector<T> coefficients = SolveSystem(xProjected, yScaled);

        // Transform coefficients back to original space
        Coefficients = _components.Multiply(coefficients);

        // Apply regularization to coefficients
        Coefficients = Regularization.Regularize(Coefficients);

        // Adjust for scaling
        for (int i = 0; i < Coefficients.Length; i++)
        {
            Coefficients[i] = NumOps.Divide(NumOps.Multiply(Coefficients[i], _yStd), _xStd[i]);
        }

        // Calculate intercept
        Intercept = NumOps.Subtract(_yMean[0], Coefficients.DotProduct(_xMean));
    }

    /// <summary>
    /// Validates the input data before training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in x doesn't match the length of y.</exception>
    /// <remarks>
    /// <para>
    /// This method checks that the input data is valid before proceeding with training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method makes sure your data is in the correct format before training begins.
    /// It checks that you have the same number of target values as you have examples in your input data.
    /// If not, it will raise an error to let you know there's a mismatch.
    /// </para>
    /// </remarks>
    private void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in x must match the length of y.");
        }
    }

    /// <summary>
    /// Performs Principal Component Analysis (PCA) on the input data.
    /// </summary>
    /// <param name="x">The centered and scaled input features matrix.</param>
    /// <returns>A tuple containing the principal components matrix and the explained variance vector.</returns>
    /// <remarks>
    /// <para>
    /// This method performs Singular Value Decomposition (SVD) on the input matrix to extract the principal components
    /// and calculate the explained variance for each component.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> PCA is a technique that finds the most important patterns (principal components) in your data.
    /// These components are directions in the data space where the data varies the most. The method uses
    /// a mathematical technique called Singular Value Decomposition (SVD) to find these components and
    /// calculate how much of the total variation each component explains.
    /// </para>
    /// </remarks>
    private (Matrix<T>, Vector<T>) PerformPCA(Matrix<T> x)
    {
        // Perform SVD
        var svd = new SvdDecomposition<T>(x);
        (Matrix<T> u, Vector<T> s, Matrix<T> vt) = (svd.U, svd.S, svd.Vt);

        // Components are the right singular vectors (rows of vt)
        Matrix<T> components = vt.Transpose();

        // Calculate explained variance
        Vector<T> explainedVariance = s.Transform(val => NumOps.Multiply(val, val));
        T totalVariance = explainedVariance.Sum();
        explainedVariance = explainedVariance.Transform(val => NumOps.Divide(val, totalVariance));

        return (components, explainedVariance);
    }

    /// <summary>
    /// Selects the appropriate number of principal components to use in the regression model.
    /// </summary>
    /// <param name="explainedVariance">The explained variance vector for each principal component.</param>
    /// <returns>The number of principal components to use.</returns>
    /// <remarks>
    /// <para>
    /// This method selects the number of components based on either a fixed number specified in the options
    /// or a threshold for the cumulative explained variance ratio.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method decides how many principal components to keep for the regression model. It can either use
    /// a fixed number that you specify, or it can select enough components to explain a certain percentage of
    /// the total variation in your data (e.g., keep enough components to explain 95% of the variance).
    /// </para>
    /// </remarks>
    private int SelectNumberOfComponents(Vector<T> explainedVariance)
    {
        if (_options.NumComponents > 0)
        {
            return Math.Min(_options.NumComponents, explainedVariance.Length);
        }

        T cumulativeVariance = NumOps.Zero;
        for (int i = 0; i < explainedVariance.Length; i++)
        {
            cumulativeVariance = NumOps.Add(cumulativeVariance, explainedVariance[i]);
            if (NumOps.GreaterThanOrEquals(cumulativeVariance, NumOps.FromDouble(_options.ExplainedVarianceRatio)))
            {
                return i + 1;
            }
        }

        return explainedVariance.Length;
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method scales the input data using the means and standard deviations from the training data,
    /// applies the regression coefficients, and adjusts the predictions back to the original scale.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After training, this method is used to make predictions on new data. It first scales your input data
    /// the same way the training data was scaled, then applies the learned model to calculate the predicted values.
    /// Finally, it transforms the predictions back to the original scale of your target variable.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Scale the input
        Matrix<T> scaledInput = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < input.Columns; j++)
            {
                scaledInput[i, j] = NumOps.Divide(NumOps.Subtract(input[i, j], _xMean[j]), _xStd[j]);
            }
        }

        // Make predictions
        Vector<T> predictions = scaledInput.Multiply(Coefficients);
        for (int i = 0; i < predictions.Length; i++)
        {
            predictions[i] = NumOps.Add(NumOps.Multiply(predictions[i], _yStd), _yMean[0]);
        }

        return predictions;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, coefficients, principal components,
    /// number of components used, and feature importance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Model metadata provides information about the model itself, rather than the predictions it makes.
    /// This includes details about how the model is configured (like how many components it uses) and
    /// information about the importance of different features. This can help you understand which input
    /// variables are most influential in making predictions.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            AdditionalInfo = new Dictionary<string, object>
        {
            { "Coefficients", Coefficients },
            { "Components", _components },
            { "NumComponents", _components.Columns },
            { "FeatureImportance", CalculateFeatureImportances() }
        }
        };
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for principal component regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method simply returns an identifier that indicates this is a principal component regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.PrincipalComponentRegression;

    /// <summary>
    /// Calculates the importance of each feature in the model.
    /// </summary>
    /// <returns>A vector containing the importance score for each feature.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates feature importances based on the absolute values of the regression coefficients.
    /// Larger absolute values indicate more important features.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Feature importance tells you which input variables have the most influence on the predictions.
    /// In PCR, this is calculated based on the magnitude (absolute value) of each coefficient in the model.
    /// Features with larger coefficient magnitudes have a stronger effect on the predictions and are considered
    /// more important.
    /// </para>
    /// </remarks>
    protected override Vector<T> CalculateFeatureImportances()
    {
        // Feature importances are based on the magnitude of the coefficients
        return Coefficients.Transform(NumOps.Abs);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including base class data and PCR-specific data
    /// such as options, principal components, means, and standard deviations.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Write base class data
        base.Serialize();

        // Write PCR-specific data
        writer.Write(_options.NumComponents);
        writer.Write(_options.ExplainedVarianceRatio);
        SerializationHelper<T>.SerializeMatrix(writer, _components);
        SerializationHelper<T>.SerializeVector(writer, _xMean);
        SerializationHelper<T>.SerializeVector(writer, _yMean);
        SerializationHelper<T>.SerializeVector(writer, _xStd);
        SerializationHelper<T>.WriteValue(writer, _yStd);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including base class data
    /// and PCR-specific data such as options, principal components, means, and standard deviations.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using MemoryStream ms = new MemoryStream(modelData);
        using BinaryReader reader = new BinaryReader(ms);

        // Read base class data
        base.Deserialize(modelData);

        // Read PCR-specific data
        _options.NumComponents = reader.ReadInt32();
        _options.ExplainedVarianceRatio = reader.ReadDouble();
        _components = SerializationHelper<T>.DeserializeMatrix(reader);
        _xMean = SerializationHelper<T>.DeserializeVector(reader);
        _yMean = SerializationHelper<T>.DeserializeVector(reader);
        _xStd = SerializationHelper<T>.DeserializeVector(reader);
        _yStd = SerializationHelper<T>.ReadValue(reader);
    }

    /// <summary>
    /// Creates a new instance of the Principal Component Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Principal Component Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Principal Component Regression model, including its options,
    /// principal components, coefficients, intercept, and preprocessing parameters (means and standard deviations).
    /// The new instance is completely independent of the original, allowing modifications without affecting the original model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect copy of your regression model:
    /// - It duplicates all the configuration settings (like how many components to use)
    /// - It copies the learned principal components (the patterns found in your data)
    /// - It preserves the coefficients and intercept (the actual formula for making predictions)
    /// - It maintains all the scaling information (means and standard deviations) needed to process new data
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        var newModel = new PrincipalComponentRegression<T>(_options, Regularization);

        // Copy coefficients and intercept from base class
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }
        newModel.Intercept = Intercept;

        // Copy principal components matrix
        if (_components != null)
        {
            newModel._components = _components.Clone();
        }

        // Copy means and standard deviations used for scaling
        if (_xMean != null)
        {
            newModel._xMean = _xMean.Clone();
        }

        if (_yMean != null)
        {
            newModel._yMean = _yMean.Clone();
        }

        if (_xStd != null)
        {
            newModel._xStd = _xStd.Clone();
        }

        newModel._yStd = _yStd;

        return newModel;
    }
}
