namespace AiDotNet.Regression;

/// <summary>
/// Implements Partial Least Squares Regression (PLS), a technique that combines features from principal 
/// component analysis and multiple linear regression to handle situations with many correlated predictors.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Partial Least Squares Regression is particularly useful when dealing with many predictor variables 
/// that may be highly correlated. It works by finding a linear combination of the predictors (components) 
/// that maximizes the covariance between the predictors and the response variable.
/// </para>
/// <para>
/// Unlike Principal Component Regression which only considers the variance in the predictor variables, 
/// PLS regression considers both the variance in the predictors and their relationship with the response variable.
/// This often leads to models with better predictive power, especially when the predictors are highly correlated.
/// </para>
/// <para>
/// For Beginners:
/// Think of PLS regression as a way to find the most important patterns in your input data that are also 
/// strongly related to what you're trying to predict. It's like finding the key ingredients in a recipe 
/// that most influence the taste, rather than just the most abundant ingredients.
/// </para>
/// </remarks>
public class PartialLeastSquaresRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the partial least squares regression model.
    /// </summary>
    /// <value>
    /// Contains settings like the number of components to extract.
    /// </value>
    private readonly PartialLeastSquaresRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The loadings matrix (P) that represents how the original variables load onto the components.
    /// </summary>
    /// <value>
    /// A matrix where each column represents the loadings for a component.
    /// </value>
    private Matrix<T> _loadings;

    /// <summary>
    /// The scores matrix (T) that represents the projection of the original data onto the components.
    /// </summary>
    /// <value>
    /// A matrix where each column represents the scores for a component.
    /// </value>
    private Matrix<T> _scores;

    /// <summary>
    /// The weights matrix (W) used to transform the original variables into components.
    /// </summary>
    /// <value>
    /// A matrix where each column represents the weights for a component.
    /// </value>
    private Matrix<T> _weights;

    /// <summary>
    /// The mean of the target variable used for centering.
    /// </summary>
    /// <value>
    /// The mean value of the target variable.
    /// </value>
    private T _yMean;

    /// <summary>
    /// The means of the predictor variables used for centering.
    /// </summary>
    /// <value>
    /// A vector containing the mean value of each predictor variable.
    /// </value>
    private Vector<T> _xMean;

    /// <summary>
    /// The standard deviation of the target variable used for scaling.
    /// </summary>
    /// <value>
    /// The standard deviation value of the target variable.
    /// </value>
    private T _yStd;

    /// <summary>
    /// The standard deviations of the predictor variables used for scaling.
    /// </summary>
    /// <value>
    /// A vector containing the standard deviation of each predictor variable.
    /// </value>
    private Vector<T> _xStd;

    /// <summary>
    /// Initializes a new instance of the PartialLeastSquaresRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the PLS regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the PLS regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public PartialLeastSquaresRegression(PartialLeastSquaresRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new PartialLeastSquaresRegressionOptions<T>();
        _loadings = new Matrix<T>(0, 0);
        _scores = new Matrix<T>(0, 0);
        _weights = new Matrix<T>(0, 0);
        _yMean = NumOps.Zero;
        _xMean = new Vector<T>(0);
        _yStd = NumOps.Zero;
        _xStd = new Vector<T>(0);
    }

    /// <summary>
    /// Trains the partial least squares regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Validates the input data
    /// 2. Centers and scales the data
    /// 3. Extracts the specified number of components using the NIPALS algorithm
    /// 4. Calculates the regression coefficients
    /// 5. Adjusts the coefficients for the scaling
    /// 6. Calculates the intercept
    /// 7. Applies regularization to the model matrices
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training is the process where the model learns from your data. The PLS algorithm first centers and scales
    /// your data (makes all variables have similar ranges), then finds the most important patterns (components)
    /// that explain both the variation in your input features and their relationship with the target variable.
    /// These components are then used to build a regression model that can predict the target variable from
    /// new input features.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);

        // Center and scale the data
        (Matrix<T> xScaled, Vector<T> yScaled, _xMean, _xStd, _yMean, _yStd) = RegressionHelper<T>.CenterAndScale(x, y);

        int numComponents = Math.Min(_options.NumComponents, x.Columns);
        _loadings = new Matrix<T>(x.Columns, numComponents);
        _scores = new Matrix<T>(x.Rows, numComponents);
        _weights = new Matrix<T>(x.Columns, numComponents);

        Matrix<T> xResidual = xScaled.Clone();
        Vector<T> yResidual = yScaled.Clone();

        for (int i = 0; i < numComponents; i++)
        {
            Vector<T> w = xResidual.Transpose().Multiply(yResidual);
            w = w.Normalize();

            Vector<T> t = xResidual.Multiply(w);
            T tt = t.DotProduct(t);

            Vector<T> p = xResidual.Transpose().Multiply(t).Divide(tt);
            T q = NumOps.Divide(yResidual.DotProduct(t), tt);

            xResidual = xResidual.Subtract(t.OuterProduct(p));
            yResidual = yResidual.Subtract(t.Multiply(q));

            _loadings.SetColumn(i, p);
            _scores.SetColumn(i, t);
            _weights.SetColumn(i, w);
        }

        // Calculate regression coefficients
        Matrix<T> W = _weights;
        Matrix<T> P = _loadings;
        Matrix<T> invPtW = P.Transpose().Multiply(W).Inverse();
        Coefficients = W.Multiply(invPtW).Multiply(_scores.Transpose()).Multiply(yScaled);

        // Apply regularization to coefficients
        Coefficients = Regularization.Regularize(Coefficients);

        // Adjust for scaling
        for (int i = 0; i < Coefficients.Length; i++)
        {
            Coefficients[i] = NumOps.Divide(NumOps.Multiply(Coefficients[i], _yStd), _xStd[i]);
        }

        // Calculate intercept
        Intercept = NumOps.Subtract(_yMean, Coefficients.DotProduct(_xMean));

        // Apply regularization to the model matrices
        _loadings = Regularization.Regularize(_loadings);
        _scores = Regularization.Regularize(_scores);
        _weights = Regularization.Regularize(_weights);
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
    /// For Beginners:
    /// This method makes sure your data is in the correct format before training begins.
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
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method scales the input data using the means and standard deviations from the training data,
    /// applies the regression coefficients, and adds the intercept to produce predictions.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, this method is used to make predictions on new data. It first scales your input data
    /// the same way the training data was scaled, then applies the learned model to calculate the predicted values.
    /// This is the main purpose of building a regression model - to predict values for new examples.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        int rows = input.Rows;
        int cols = input.Columns;

        // Scale the input
        Matrix<T> scaledInput = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                scaledInput[i, j] = NumOps.Divide(NumOps.Subtract(input[i, j], _xMean[j]), _xStd[j]);
            }
        }

        // Make predictions
        Vector<T> predictions = scaledInput.Multiply(Coefficients);
        for (int i = 0; i < predictions.Length; i++)
        {
            predictions[i] = NumOps.Add(predictions[i], Intercept);
        }

        return predictions;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, coefficients, loadings, scores, weights,
    /// number of components, and feature importance.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Model metadata provides information about the model itself, rather than the predictions it makes.
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
            { "Loadings", _loadings },
            { "Scores", _scores },
            { "Weights", _weights },
            { "NumComponents", _options.NumComponents },
            { "FeatureImportance", CalculateFeatureImportances() }
        }
        };
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for partial least squares regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates this is a partial least squares regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.PartialLeastSquaresRegression;

    /// <summary>
    /// Calculates the importance of each feature in the model.
    /// </summary>
    /// <returns>A vector containing the importance score for each feature.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the Variable Importance in Projection (VIP) scores, which measure the contribution
    /// of each variable to the model based on the variance explained by each PLS component and the weights
    /// of each variable in those components.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Feature importance tells you which input variables have the most influence on the predictions.
    /// In PLS regression, this is calculated using a measure called VIP (Variable Importance in Projection),
    /// which considers both how much each component explains the variation in the data and how much each
    /// variable contributes to those components. Higher values indicate more important variables.
    /// </para>
    /// </remarks>
    protected override Vector<T> CalculateFeatureImportances()
    {
        // VIP (Variable Importance in Projection) scores
        Vector<T> vip = new Vector<T>(Coefficients.Length);

        // Calculate ssY (sum of squares of Y)
        T ssY = NumOps.Zero;
        Matrix<T> scoresTransposeMultiplyScores = _scores.Transpose().Multiply(_scores);
        for (int i = 0; i < scoresTransposeMultiplyScores.Rows; i++)
        {
            ssY = NumOps.Add(ssY, scoresTransposeMultiplyScores[i, i]);
        }

        for (int j = 0; j < Coefficients.Length; j++)
        {
            T score = NumOps.Zero;
            for (int a = 0; a < _options.NumComponents; a++)
            {
                T w = _weights[j, a];
                T t = _scores.GetColumn(a).DotProduct(_scores.GetColumn(a));
                score = NumOps.Add(score, NumOps.Multiply(NumOps.Multiply(w, w), t));
            }
            vip[j] = NumOps.Multiply(NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(Coefficients.Length), NumOps.Divide(score, ssY))), Coefficients[j]);
        }

        return vip;
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including base class data and PLS-specific data
    /// such as loadings, scores, weights, means, and standard deviations.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Serialization converts the model's internal state into a format that can be saved to disk or
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

        // Write PLS-specific data
        writer.Write(_options.NumComponents);
        SerializationHelper<T>.SerializeMatrix(writer, _loadings);
        SerializationHelper<T>.SerializeMatrix(writer, _scores);
        SerializationHelper<T>.SerializeMatrix(writer, _weights);
        SerializationHelper<T>.WriteValue(writer, _yMean);
        SerializationHelper<T>.SerializeVector(writer, _xMean);
        SerializationHelper<T>.WriteValue(writer, _yStd);
        SerializationHelper<T>.SerializeVector(writer, _xStd);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including base class data
    /// and PLS-specific data such as loadings, scores, weights, means, and standard deviations.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
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

        // Read PLS-specific data
        _options.NumComponents = reader.ReadInt32();
        _loadings = SerializationHelper<T>.DeserializeMatrix(reader);
        _scores = SerializationHelper<T>.DeserializeMatrix(reader);
        _weights = SerializationHelper<T>.DeserializeMatrix(reader);
        _yMean = SerializationHelper<T>.ReadValue(reader);
        _xMean = SerializationHelper<T>.DeserializeVector(reader);
        _yStd = SerializationHelper<T>.ReadValue(reader);
        _xStd = SerializationHelper<T>.DeserializeVector(reader);
    }

    /// <summary>
    /// Creates a new instance of the Partial Least Squares Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Partial Least Squares Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Partial Least Squares Regression model, including its options,
    /// coefficients, intercept, loadings, scores, weights, and data scaling parameters. The new instance is completely 
    /// independent of the original, allowing modifications without affecting the original model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate:
    /// - It copies all the configuration settings (like the number of components)
    /// - It preserves the coefficients and intercept that define your regression model
    /// - It duplicates all the internal matrices (loadings, scores, weights) that capture the patterns in your data
    /// - It maintains the scaling information (means and standard deviations) needed to process new data
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
        var newModel = new PartialLeastSquaresRegression<T>(_options, Regularization);

        // Copy coefficients and intercept from base class
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }
        newModel.Intercept = Intercept;

        // Copy PLS-specific components
        if (_loadings != null)
        {
            newModel._loadings = _loadings.Clone();
        }

        if (_scores != null)
        {
            newModel._scores = _scores.Clone();
        }

        if (_weights != null)
        {
            newModel._weights = _weights.Clone();
        }

        // Copy means and standard deviations used for scaling
        newModel._yMean = _yMean;

        if (_xMean != null)
        {
            newModel._xMean = _xMean.Clone();
        }

        newModel._yStd = _yStd;

        if (_xStd != null)
        {
            newModel._xStd = _xStd.Clone();
        }

        return newModel;
    }
}
