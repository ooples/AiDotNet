namespace AiDotNet.Regression;

/// <summary>
/// Implements orthogonal regression (also known as total least squares), which minimizes the perpendicular 
/// distance from data points to the fitted line or hyperplane.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Unlike ordinary least squares regression which minimizes vertical distances, orthogonal regression 
/// minimizes the perpendicular (orthogonal) distance from each data point to the regression line or hyperplane.
/// This approach is more appropriate when both dependent and independent variables contain measurement errors.
/// </para>
/// <para>
/// The algorithm works by centering the data, optionally scaling the variables, and then finding the 
/// solution using matrix decomposition methods such as SVD (Singular Value Decomposition).
/// </para>
/// <para>
/// For Beginners:
/// Orthogonal regression is useful when you're not sure which variable is dependent and which is independent,
/// or when both variables have measurement errors. Think of it as finding the line that's as close as possible
/// to all points when measuring distance perpendicular to the line, rather than just vertically.
/// </para>
/// </remarks>
public class OrthogonalRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the orthogonal regression model.
    /// </summary>
    /// <value>
    /// Contains settings like tolerance, maximum iterations, and whether to scale variables.
    /// </value>
    private readonly OrthogonalRegressionOptions<T> _options;

    /// <summary>
    /// Initializes a new instance of the OrthogonalRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the orthogonal regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the orthogonal regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public OrthogonalRegression(OrthogonalRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new OrthogonalRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the orthogonal regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Applies regularization to the input matrix
    /// 2. Centers the data by subtracting the mean from each feature and the target
    /// 3. Optionally scales the variables to have unit variance
    /// 4. Computes the augmented matrix [X y]
    /// 5. Uses matrix decomposition to find the solution
    /// 6. Rescales the solution and normalizes it
    /// 7. Extracts the coefficients and intercept
    /// 8. Applies regularization to the coefficients
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training is the process where the model learns from your data. The algorithm first centers the data
    /// (subtracts the average from each variable) and optionally scales it (makes all variables have similar ranges).
    /// Then it uses advanced matrix math to find the best-fitting line or plane that minimizes the perpendicular
    /// distance from all points. The result is a set of coefficients that define this line or plane.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Apply regularization to the input matrix
        x = Regularization.Regularize(x);

        // Center the data
        Vector<T> meanX = new(p);
        for (int j = 0; j < p; j++)
        {
            meanX[j] = x.GetColumn(j).Mean();
        }
        T meanY = y.Mean();

        Matrix<T> centeredX = new(n, p);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                centeredX[i, j] = NumOps.Subtract(x[i, j], meanX[j]);
            }
        }
        Vector<T> centeredY = y.Subtract(meanY);

        // Scale the variables if option is set
        Vector<T> scaleX = Vector<T>.CreateDefault(p, NumOps.One);
        if (_options.ScaleVariables)
        {
            for (int j = 0; j < p; j++)
            {
                T columnVariance = centeredX.GetColumn(j).Variance();
                scaleX[j] = NumOps.Sqrt(columnVariance);
                if (!NumOps.Equals(scaleX[j], NumOps.Zero))
                {
                    for (int i = 0; i < n; i++)
                    {
                        centeredX[i, j] = NumOps.Divide(centeredX[i, j], scaleX[j]);
                    }
                }
            }
        }

        // Compute the augmented matrix [X y]
        Matrix<T> augmentedMatrix = new(n, p + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                augmentedMatrix[i, j] = centeredX[i, j];
            }
            augmentedMatrix[i, p] = centeredY[i];
        }

        // Use the decomposition method from options
        IMatrixDecomposition<T> decomposition = Options.DecompositionMethod ?? new SvdDecomposition<T>(augmentedMatrix);

        // Get the solution using MatrixSolutionHelper
        Vector<T> solution = MatrixSolutionHelper.SolveLinearSystem(augmentedMatrix, augmentedMatrix.GetColumn(p), MatrixDecompositionFactory.GetDecompositionType(Options.DecompositionMethod));

        // Rescale the solution
        for (int j = 0; j < p; j++)
        {
            solution[j] = NumOps.Divide(solution[j], scaleX[j]);
        }

        // Normalize the solution
        T norm = NumOps.Sqrt(solution.DotProduct(solution));
        solution = solution.Divide(norm);

        // Extract coefficients and intercept
        Coefficients = solution.GetSubVector(0, p);

        // Apply regularization to the coefficients
        Coefficients = Regularization.Regularize(Coefficients);

        Intercept = NumOps.Subtract(meanY, Coefficients.DotProduct(meanX));
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for orthogonal regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates this is an orthogonal regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.OrthogonalRegression;

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes both the base class data and the orthogonal regression specific options,
    /// including tolerance, maximum iterations, and whether to scale variables.
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
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize OrthogonalRegression specific options
        writer.Write(_options.Tolerance);
        writer.Write(_options.MaxIterations);
        writer.Write(_options.ScaleVariables);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes both the base class data and the orthogonal regression specific options,
    /// reconstructing the model's state from the serialized data.
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
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize OrthogonalRegression specific options
        _options.Tolerance = reader.ReadDouble();
        _options.MaxIterations = reader.ReadInt32();
        _options.ScaleVariables = reader.ReadBoolean();
    }

    /// <summary>
    /// Creates a new instance of the Orthogonal Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Orthogonal Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Orthogonal Regression model, including its options,
    /// coefficients, intercept, and regularization settings. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate:
    /// - It copies all the configuration settings (like tolerance and whether to scale variables)
    /// - It preserves the coefficients (the weights for each feature)
    /// - It maintains the intercept (the starting point of your regression line or plane)
    /// - It includes the same regularization settings to prevent overfitting
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before making changes to the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        var newModel = new OrthogonalRegression<T>(_options, Regularization);

        // Copy coefficients if they exist
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept
        newModel.Intercept = Intercept;

        return newModel;
    }
}
