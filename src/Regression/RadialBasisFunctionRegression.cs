namespace AiDotNet.Regression;

/// <summary>
/// Implements Radial Basis Function (RBF) Regression, a technique that uses radial basis functions
/// as the basis for approximating complex nonlinear relationships between inputs and outputs.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Radial Basis Function Regression works by transforming the input space using a set of radial basis functions,
/// each centered at a different point. These functions produce a response that depends on the distance from the
/// input to the center point. The model then combines these responses linearly to make predictions.
/// </para>
/// <para>
/// The algorithm first selects a set of centers (typically using k-means clustering), computes the RBF features
/// for each input point, and then solves a linear regression problem to find the optimal weights.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of RBF regression as placing a set of "bell curves" at strategic locations in your input space.
/// Each curve gives a strong response when an input is close to its center and a weak response when it's far away.
/// The model predicts by combining these responses with learned weights. This approach is particularly good at
/// modeling complex, non-linear relationships in data.
/// </para>
/// </remarks>
public class RadialBasisFunctionRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the radial basis function regression model.
    /// </summary>
    /// <value>
    /// Contains settings like the number of centers, gamma parameter, and random seed.
    /// </value>
    private readonly RadialBasisFunctionOptions _options;

    /// <summary>
    /// The centers of the radial basis functions.
    /// </summary>
    /// <value>
    /// A matrix where each row represents a center point in the input space.
    /// </value>
    private Matrix<T> _centers;

    /// <summary>
    /// The weights used to combine the radial basis function outputs.
    /// </summary>
    /// <value>
    /// A vector of weights, including a bias term.
    /// </value>
    private Vector<T> _weights;

    /// <summary>
    /// Initializes a new instance of the RadialBasisFunctionRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the RBF regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the RBF regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public RadialBasisFunctionRegression(RadialBasisFunctionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new RadialBasisFunctionOptions();
        _centers = Matrix<T>.Empty();
        _weights = Vector<T>.Empty();
    }

    /// <summary>
    /// Optimizes the model parameters based on the training data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method implements the core of the RBF regression algorithm. The steps are:
    /// 1. Select centers using k-means clustering
    /// 2. Compute RBF features for each input point
    /// 3. Apply regularization to the RBF features
    /// 4. Solve a linear regression problem to find the optimal weights
    /// 5. Apply regularization to the weights
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the main training method where the model learns from your data. It first finds good locations
    /// for the "bell curves" (centers) using a clustering algorithm, then calculates how each input point
    /// responds to these centers. Finally, it solves a linear equation to find the best weights for combining
    /// these responses to predict the target values.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Select centers
        _centers = SelectCenters(x);

        // Compute RBF features
        Matrix<T> rbfFeatures = ComputeRBFFeatures(x);

        // Solve for weights using linear regression
        // Note: Regularization is applied within SolveLinearRegression via ridge penalty
        _weights = SolveLinearRegression(rbfFeatures, y);
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms the input data using the RBF features and then applies the learned weights
    /// to make predictions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After training, this method is used to make predictions on new data. It first transforms each input
    /// example using the radial basis functions (calculating how close it is to each center), then combines
    /// these transformed values using the learned weights to produce the final prediction.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Matrix<T> rbfFeatures = ComputeRBFFeatures(input);
        // RBF features are computed directly - no transformation needed
        return rbfFeatures.Multiply(_weights);
    }

    /// <summary>
    /// Predicts the value for a single input vector.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms a single input vector using the RBF features and then applies the learned weights
    /// to make a prediction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the core prediction function for a single example. It calculates how the input responds to each
    /// radial basis function (center), then combines these responses using the learned weights to produce
    /// the final prediction.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        Vector<T> rbfFeatures = ComputeRBFFeaturesSingle(input);
        // RBF features are computed directly - no transformation needed
        return rbfFeatures.DotProduct(_weights);
    }

    /// <summary>
    /// Selects centers for the radial basis functions using k-means clustering.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <returns>A matrix where each row represents a center point.</returns>
    /// <remarks>
    /// <para>
    /// This method implements k-means clustering to select centers for the radial basis functions.
    /// The steps are:
    /// 1. Initialize centers randomly
    /// 2. Iterate until convergence or maximum iterations:
    ///    a. Assign each point to the nearest center
    ///    b. Recompute centers as the mean of assigned points
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method finds good locations for the "bell curves" (centers) by grouping similar data points together
    /// and placing a center at the middle of each group. It uses an algorithm called k-means clustering,
    /// which iteratively assigns points to the nearest center and then updates the centers based on these assignments.
    /// </para>
    /// </remarks>
    private Matrix<T> SelectCenters(Matrix<T> x)
    {
        int numCenters = Math.Min(_options.NumberOfCenters, x.Rows);
        var random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();

        // Initialize centers randomly
        var centers = new Matrix<T>(numCenters, x.Columns);
        var selectedIndices = new HashSet<int>();
        while (selectedIndices.Count < numCenters)
        {
            int index = random.Next(x.Rows);
            if (selectedIndices.Add(index))
            {
                centers.SetRow(selectedIndices.Count - 1, x.GetRow(index));
            }
        }

        // Perform K-means clustering
        const int maxIterations = 100;
        var assignments = new int[x.Rows];
        var newCenters = new Matrix<T>(numCenters, x.Columns);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            bool changed = false;

            // Assign points to nearest center
            for (int i = 0; i < x.Rows; i++)
            {
                int nearestCenter = 0;
                T minDistance = EuclideanDistance(x.GetRow(i), centers.GetRow(0));

                for (int j = 1; j < numCenters; j++)
                {
                    T distance = EuclideanDistance(x.GetRow(i), centers.GetRow(j));
                    if (NumOps.LessThan(distance, minDistance))
                    {
                        minDistance = distance;
                        nearestCenter = j;
                    }
                }

                if (assignments[i] != nearestCenter)
                {
                    assignments[i] = nearestCenter;
                    changed = true;
                }
            }

            if (!changed)
            {
                break; // Convergence reached
            }

            // Compute new centers
            var counts = new int[numCenters];
            for (int i = 0; i < numCenters; i++)
            {
                newCenters.SetRow(i, new Vector<T>(x.Columns));
            }

            for (int i = 0; i < x.Rows; i++)
            {
                int assignment = assignments[i];
                newCenters.SetRow(assignment, newCenters.GetRow(assignment).Add(x.GetRow(i)));
                counts[assignment]++;
            }

            for (int i = 0; i < numCenters; i++)
            {
                if (counts[i] > 0)
                {
                    newCenters.SetRow(i, newCenters.GetRow(i).Divide(NumOps.FromDouble(counts[i])));
                }
                else
                {
                    // If a center has no assigned points, reinitialize it randomly
                    int randomIndex = random.Next(x.Rows);
                    newCenters.SetRow(i, x.GetRow(randomIndex));
                }
            }

            centers = newCenters;
        }

        return centers;
    }

    /// <summary>
    /// Computes the RBF features for a matrix of input points.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <returns>A matrix of RBF features, including a bias term.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the RBF features for each input point by calculating the response of each
    /// radial basis function to the input.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method transforms your input data by calculating how close each point is to each center,
    /// then applying the radial basis function (a bell curve) to these distances. The result is a new
    /// representation of your data that captures non-linear relationships.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeRBFFeatures(Matrix<T> x)
    {
        var rbfFeatures = new Matrix<T>(x.Rows, _centers.Rows + 1);

        for (int i = 0; i < x.Rows; i++)
        {
            var row = ComputeRBFFeaturesSingle(x.GetRow(i));
            for (int j = 0; j < row.Length; j++)
            {
                rbfFeatures[i, j] = row[j];
            }
        }

        return rbfFeatures;
    }

    /// <summary>
    /// Computes the RBF features for a single input vector.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>A vector of RBF features, including a bias term.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the RBF features for a single input point by calculating the response of each
    /// radial basis function to the input.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method transforms a single input example by calculating how close it is to each center,
    /// then applying the radial basis function (a bell curve) to these distances. The first element
    /// is always 1, which serves as a bias term (intercept) in the model.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeRBFFeaturesSingle(Vector<T> x)
    {
        var features = new Vector<T>(_centers.Rows + 1)
        {
            [0] = NumOps.One // Bias term
        };

        for (int i = 0; i < _centers.Rows; i++)
        {
            T distance = EuclideanDistance(x, _centers.GetRow(i));
            features[i + 1] = RbfKernel(distance);
        }

        return features;
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The Euclidean distance between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the Euclidean distance (straight-line distance) between two points in the input space.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Euclidean distance is the straight-line distance between two points in space, calculated using the
    /// Pythagorean theorem. This is used to determine how close an input point is to each center.
    /// </para>
    /// </remarks>
    private T EuclideanDistance(Vector<T> x1, Vector<T> x2)
    {
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = NumOps.Subtract(x1[i], x2[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sumSquared);
    }

    /// <summary>
    /// Applies the radial basis function kernel to a distance value.
    /// </summary>
    /// <param name="distance">The distance value.</param>
    /// <returns>The result of applying the RBF kernel to the distance.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the Gaussian radial basis function kernel, which is defined as exp(-gamma * distance^2).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This function creates the "bell curve" shape of the radial basis function. It takes a distance value
    /// and returns a value between 0 and 1, where 1 means the input is exactly at the center (distance = 0)
    /// and values close to 0 mean the input is far from the center. The gamma parameter controls how quickly
    /// the function drops off with distance.
    /// </para>
    /// </remarks>
    private T RbfKernel(T distance)
    {
        T gamma = NumOps.FromDouble(_options.Gamma);
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, NumOps.Multiply(distance, distance))));
    }

    /// <summary>
    /// Solves a linear regression problem to find the optimal weights using ridge regularization.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>The optimal weights vector.</returns>
    /// <remarks>
    /// <para>
    /// This method solves the linear regression problem using the normal equations approach with ridge
    /// regularization (Tikhonov regularization). The regularization term (lambda * I) is added to X^T * X
    /// to ensure numerical stability and prevent overfitting. This computes: w = (X^T * X + lambda * I)^-1 * X^T * y.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After transforming the input data using radial basis functions, this method finds the best weights
    /// to combine these transformed features to predict the target values. It uses a mathematical technique
    /// called the "normal equations" with a small regularization penalty to find stable weights that
    /// minimize prediction error while avoiding numerical issues.
    /// </para>
    /// </remarks>
    private Vector<T> SolveLinearRegression(Matrix<T> x, Vector<T> y)
    {
        // Use pseudo-inverse with ridge regularization to solve for weights
        // Ridge regularization (Tikhonov regularization) adds a small penalty term (λI)
        // to prevent numerical instability and improve generalization
        Matrix<T> xTranspose = x.Transpose();
        Matrix<T> xTx = xTranspose.Multiply(x);

        // Add ridge regularization: (X^T X + λI)^-1 X^T y
        // Using a small lambda value (1e-8) for numerical stability
        T lambda = NumOps.FromDouble(1e-8);
        Matrix<T> identity = Matrix<T>.CreateIdentity(xTx.Rows);
        Matrix<T> xTxRegularized = xTx.Add(identity.Multiply(lambda));

        Matrix<T> xTxInverse = xTxRegularized.Inverse();
        Matrix<T> xTxInverseXT = xTxInverse.Multiply(xTranspose);
        return xTxInverseXT.Multiply(y);
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for radial basis function regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method simply returns an identifier that indicates this is a radial basis function regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.RadialBasisFunctionRegression;
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including base class data, options, centers, and weights.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
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

        // Serialize RBF specific data
        writer.Write(_options.NumberOfCenters);
        writer.Write(_options.Gamma);
        writer.Write(_options.Seed ?? -1);

        // Serialize centers
        writer.Write(_centers.Rows);
        writer.Write(_centers.Columns);
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_centers[i, j]));
            }
        }

        // Serialize weights
        writer.Write(_weights.Length);
        for (int i = 0; i < _weights.Length; i++)
        {
            writer.Write(Convert.ToDouble(_weights[i]));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including base class data,
    /// options, centers, and weights.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
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

        // Deserialize RBF specific data
        _options.NumberOfCenters = reader.ReadInt32();
        _options.Gamma = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;

        // Deserialize centers
        int centerRows = reader.ReadInt32();
        int centerColumns = reader.ReadInt32();
        _centers = new Matrix<T>(centerRows, centerColumns);
        for (int i = 0; i < centerRows; i++)
        {
            for (int j = 0; j < centerColumns; j++)
            {
                _centers[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize weights
        int weightsLength = reader.ReadInt32();
        _weights = new Vector<T>(weightsLength);
        for (int i = 0; i < weightsLength; i++)
        {
            _weights[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Creates a new instance of the radial basis function regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the RBF regression model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the radial basis function regression model with the same 
    /// configuration options as the current instance, but without copying the trained parameters (centers and weights).
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters. It's like getting a blank template with the same settings.
    /// 
    /// Think of it like getting a fresh copy of a form with all the same fields and settings,
    /// but without any of the data filled in. The new model has the same:
    /// - Number of centers
    /// - Gamma parameter (controls how quickly the influence of each center drops off)
    /// - Regularization settings
    /// - Other configuration options
    /// 
    /// But it doesn't have the learned centers or weights from training.
    /// 
    /// This is mainly used internally by the framework when performing operations like
    /// cross-validation or creating ensembles of similar models.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options and regularization
        return new RadialBasisFunctionRegression<T>(_options, Regularization);
    }
}
