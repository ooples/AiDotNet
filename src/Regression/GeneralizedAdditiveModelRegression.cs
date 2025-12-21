namespace AiDotNet.Regression;

/// <summary>
/// Implements a Generalized Additive Model (GAM) for regression, which models the target as a sum of smooth functions
/// of individual features, allowing for flexible nonlinear relationships while maintaining interpretability.
/// </summary>
/// <remarks>
/// <para>
/// Generalized Additive Models extend linear regression by allowing nonlinear relationships between features and the target
/// variable, while maintaining additivity. Each feature is transformed using basis functions (typically splines), and
/// the model is expressed as a sum of these transformations. This approach balances flexibility and interpretability,
/// as the effect of each feature can be visualized independently.
/// </para>
/// <para><b>For Beginners:</b> A Generalized Additive Model is like a more flexible version of linear regression.
/// 
/// Instead of assuming that each feature has a straight-line relationship with the target (like y = mx + b),
/// GAMs allow each feature to have its own curved relationship with the target. The model then adds up
/// all these individual curves to make a prediction.
/// 
/// Think of it this way:
/// - Linear regression: House price = a × Size + b × Age + c × Location + ...
/// - GAM: House price = f1(Size) + f2(Age) + f3(Location) + ...
///   Where f1, f2, f3 are curves rather than straight lines
/// 
/// The benefit is that you can:
/// - Capture more complex patterns in your data
/// - Still understand how each feature individually affects the prediction
/// - Visualize the shape of the relationship for each feature
/// 
/// GAMs are a good middle ground between simple linear models and complex "black box" models
/// like neural networks.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GeneralizedAdditiveModel<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Generalized Additive Model.
    /// </summary>
    private readonly GeneralizedAdditiveModelOptions<T> _options;

    /// <summary>
    /// Matrix of basis functions applied to the input features.
    /// </summary>
    private Matrix<T> _basisFunctions;

    /// <summary>
    /// Vector of model coefficients for the basis functions.
    /// </summary>
    private Vector<T> _coefficients;

    /// <summary>
    /// Initializes a new instance of the <see cref="GeneralizedAdditiveModel{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the Generalized Additive Model.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Generalized Additive Model with the specified options and regularization
    /// strategy. If no options are provided, default values are used. If no regularization is specified, no regularization
    /// is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new GAM model.
    /// 
    /// When creating a GAM, you can specify:
    /// - Options: Controls how many splines to use and their shape
    /// - Regularization: Helps prevent the model from becoming too wiggly or overfitting
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a GAM with default settings
    /// var gam = new GeneralizedAdditiveModel&lt;double&gt;();
    /// 
    /// // Create a GAM with custom options
    /// var options = new GeneralizedAdditiveModelOptions&lt;double&gt; { 
    ///     NumSplines = 10,
    ///     Degree = 3
    /// };
    /// var customGam = new GeneralizedAdditiveModel&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public GeneralizedAdditiveModel(
        GeneralizedAdditiveModelOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new GeneralizedAdditiveModelOptions<T>();
        _basisFunctions = new Matrix<T>(0, 0);
        _coefficients = new Vector<T>(0);
    }

    /// <summary>
    /// Trains the Generalized Additive Model using the provided input features and target values.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method builds the Generalized Additive Model by creating basis functions (splines) for each feature,
    /// and then fitting coefficients to these basis functions to minimize the prediction error.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model how to make predictions using your data.
    /// 
    /// During training:
    /// 1. The model transforms each feature into a set of basis functions (mathematical curves)
    /// 2. For each feature, it creates multiple basis functions to capture different aspects of the relationship
    /// 3. It then finds the best coefficients (weights) for each basis function
    /// 4. These coefficients determine how much each curve contributes to the final prediction
    /// 
    /// After training, the model can predict values for new data by applying these same transformations
    /// and combining them according to the learned coefficients.
    /// 
    /// Example:
    /// ```csharp
    /// // Train the model
    /// gam.Train(features, targets);
    /// ```
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        _basisFunctions = CreateBasisFunctions(x);
        FitModel(y);
    }

    /// <summary>
    /// Validates that the input data dimensions are compatible.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in x doesn't match the length of y.</exception>
    private void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in x must match the length of y.");
        }
    }

    /// <summary>
    /// Creates basis functions for each feature in the input data.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <returns>A matrix of basis functions.</returns>
    private Matrix<T> CreateBasisFunctions(Matrix<T> x)
    {
        int numFeatures = x.Columns;
        int numBasisFunctions = _options.NumSplines * numFeatures;
        Matrix<T> basisFunctions = new Matrix<T>(x.Rows, numBasisFunctions);

        for (int i = 0; i < numFeatures; i++)
        {
            Vector<T> feature = x.GetColumn(i);
            Vector<T> knots = CreateKnots(feature);

            for (int j = 0; j < _options.NumSplines; j++)
            {
                Vector<T> spline = CreateSpline(feature, knots[j], _options.Degree);
                basisFunctions.SetColumn(i * _options.NumSplines + j, spline);
            }
        }

        return basisFunctions;
    }

    /// <summary>
    /// Creates knot points for spline basis functions based on the feature values.
    /// </summary>
    /// <param name="feature">The feature vector.</param>
    /// <returns>A vector of knot points.</returns>
    private Vector<T> CreateKnots(Vector<T> feature)
    {
        Vector<T> sortedFeature = new Vector<T>(feature.OrderBy(v => v));
        int step = sortedFeature.Length / (_options.NumSplines + 1);
        return new Vector<T>([.. Enumerable.Range(1, _options.NumSplines).Select(i => sortedFeature[i * step])]);
    }

    /// <summary>
    /// Creates a spline basis function for a feature using the specified knot and degree.
    /// </summary>
    /// <param name="feature">The feature vector.</param>
    /// <param name="knot">The knot point for the spline.</param>
    /// <param name="degree">The degree of the spline.</param>
    /// <returns>A vector of spline function values.</returns>
    private Vector<T> CreateSpline(Vector<T> feature, T knot, int degree)
    {
        return new Vector<T>(feature.Select(x => SplineFunction(x, knot, degree)));
    }

    /// <summary>
    /// Computes the spline function value for a given input, knot, and degree.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="knot">The knot point.</param>
    /// <param name="degree">The degree of the spline.</param>
    /// <returns>The spline function value.</returns>
    private T SplineFunction(T x, T knot, int degree)
    {
        T diff = NumOps.Subtract(x, knot);
        return NumOps.GreaterThan(diff, NumOps.Zero)
            ? NumOps.Power(diff, NumOps.FromDouble(degree))
            : NumOps.Zero;
    }

    /// <summary>
    /// Fits the model coefficients using the basis functions and target values.
    /// </summary>
    /// <param name="y">The target vector.</param>
    private void FitModel(Vector<T> y)
    {
        Matrix<T> penaltyMatrix = CreatePenaltyMatrix();
        Matrix<T> xTx = _basisFunctions.Transpose().Multiply(_basisFunctions);
        Matrix<T> regularizedXTX = Regularization.Regularize(xTx);
        Vector<T> xTy = _basisFunctions.Transpose().Multiply(y);

        _coefficients = SolveSystem(regularizedXTX, xTy);
        _coefficients = Regularization.Regularize(_coefficients);
    }

    /// <summary>
    /// Creates a penalty matrix for regularization.
    /// </summary>
    /// <returns>The penalty matrix.</returns>
    private Matrix<T> CreatePenaltyMatrix()
    {
        int size = _basisFunctions.Columns;
        Matrix<T> penaltyMatrix = Matrix<T>.CreateIdentity(size);
        return penaltyMatrix;
    }

    /// <summary>
    /// Predicts target values for the provided input features using the trained Generalized Additive Model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data by transforming the input features into basis functions
    /// and applying the learned coefficients to compute the predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// The prediction process:
    /// 1. Each input feature is transformed into the same basis functions used during training
    /// 2. These basis functions are multiplied by the coefficients learned during training
    /// 3. The results are summed to produce the final prediction
    /// 
    /// This is similar to how linear regression makes predictions, but with transformed features
    /// that allow for non-linear relationships.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = gam.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Matrix<T> inputBasisFunctions = CreateBasisFunctions(input);
        return inputBasisFunctions.Multiply(_coefficients);
    }

    /// <summary>
    /// Gets metadata about the Generalized Additive Model and its configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, coefficients, feature importances,
    /// and configuration options. This information can be useful for model management, comparison, visualization,
    /// and documentation purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about your GAM model.
    /// 
    /// The metadata includes:
    /// - The type of model (Generalized Additive Model)
    /// - The coefficients for each basis function
    /// - How important each feature is for predictions
    /// - Configuration settings like the number of splines and their degree
    /// 
    /// This information is helpful when:
    /// - Comparing different models
    /// - Visualizing how each feature affects predictions
    /// - Documenting your model's configuration
    /// - Troubleshooting model performance
    /// 
    /// Example:
    /// ```csharp
    /// var metadata = gam.GetModelMetadata();
    /// Console.WriteLine($"Model type: {metadata.ModelType}");
    /// Console.WriteLine($"Number of splines: {metadata.AdditionalInfo["NumSplines"]}");
    /// ```
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Coefficients", _coefficients },
                { "FeatureImportance", CalculateFeatureImportances() },
                { "NumSplines", _options.NumSplines },
                { "Degree", _options.Degree }
            }
        };
    }

    /// <summary>
    /// Gets the model type of the Generalized Additive Model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>
    protected override ModelType GetModelType() => ModelType.GeneralizedAdditiveModelRegression;

    /// <summary>
    /// Calculates the importance of each feature in the model based on the magnitude of its coefficients.
    /// </summary>
    /// <returns>A vector of feature importance scores.</returns>
    protected override Vector<T> CalculateFeatureImportances()
    {
        int numFeatures = _basisFunctions.Columns / _options.NumSplines;
        Vector<T> importances = new Vector<T>(numFeatures);

        for (int i = 0; i < numFeatures; i++)
        {
            T importance = NumOps.Zero;
            for (int j = 0; j < _options.NumSplines; j++)
            {
                importance = NumOps.Add(importance, NumOps.Abs(_coefficients[i * _options.NumSplines + j]));
            }
            importances[i] = importance;
        }

        return importances;
    }

    /// <summary>
    /// Serializes the Generalized Additive Model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the Generalized Additive Model into a byte array that can be stored in a file, database,
    /// or transmitted over a network. The serialized data includes the model's configuration options, basis functions,
    /// and learned coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained model as a sequence of bytes.
    /// 
    /// Serialization allows you to:
    /// - Save your model to a file
    /// - Store your model in a database
    /// - Send your model over a network
    /// - Keep your model for later use without having to retrain it
    /// 
    /// The serialized data includes:
    /// - All the model's settings (like number of splines and their degree)
    /// - The basis functions used to transform features
    /// - The coefficients learned during training
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = gam.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("gam.model", modelData);
    /// ```
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Write base class data
        base.Serialize();

        // Write GAM-specific data
        writer.Write(_options.NumSplines);
        writer.Write(_options.Degree);

        // Write _basisFunctions
        writer.Write(_basisFunctions.Rows);
        writer.Write(_basisFunctions.Columns);
        for (int i = 0; i < _basisFunctions.Rows; i++)
        {
            for (int j = 0; j < _basisFunctions.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_basisFunctions[i, j]));
            }
        }

        // Write _coefficients
        writer.Write(_coefficients.Length);
        for (int i = 0; i < _coefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_coefficients[i]));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized Generalized Additive Model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a Generalized Additive Model from a byte array that was previously created using the
    /// Serialize method. It restores the model's configuration options, basis functions, and learned coefficients,
    /// allowing the model to be used for predictions without retraining.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize a model:
    /// - All settings are restored
    /// - The basis functions are reconstructed
    /// - The learned coefficients are recovered
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("gam.model");
    /// 
    /// // Deserialize the model
    /// var gam = new GeneralizedAdditiveModel&lt;double&gt;();
    /// gam.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = gam.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using MemoryStream ms = new MemoryStream(modelData);
        using BinaryReader reader = new BinaryReader(ms);

        // Read base class data
        base.Deserialize(modelData);

        // Read GAM-specific data
        _options.NumSplines = reader.ReadInt32();
        _options.Degree = reader.ReadInt32();

        // Read _basisFunctions
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _basisFunctions = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _basisFunctions[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Read _coefficients
        int length = reader.ReadInt32();
        _coefficients = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            _coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Creates a new instance of the GeneralizedAdditiveModel with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new GeneralizedAdditiveModel instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the GeneralizedAdditiveModel with the same configuration options
    /// and regularization settings as the current instance. This is useful for model cloning, ensemble methods, or
    /// cross-validation scenarios where multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GeneralizedAdditiveModel<T>(_options, Regularization);
    }
}
