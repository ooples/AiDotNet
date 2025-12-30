namespace AiDotNet.Regression;

/// <summary>
/// Implements spline regression, which models nonlinear relationships by fitting piecewise polynomial functions.
/// This advanced regression technique offers more flexibility than simple linear regression by allowing the model
/// to change its behavior across different regions of the data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Spline regression uses basis functions centered at specific points called knots. The model combines:
/// - A constant term
/// - Polynomial terms of the input features (up to a specified degree)
/// - Spline terms that activate beyond each knot
///
/// This creates a piecewise function that can smoothly adapt to local patterns in the data.
/// </para>
/// <para><b>For Beginners:</b> Spline regression is like drawing a curve through your data that can bend and adjust
/// at specific points (called knots).
/// 
/// Think of it like this:
/// - Instead of forcing a single straight line through all your data
/// - The model places connection points (knots) where the curve can change direction
/// - These knots let the model adapt to different patterns in different regions of your data
/// 
/// For example, if modeling how temperature affects plant growth:
/// - Below freezing: plants don't grow at all (flat line)
/// - From freezing to optimal: growth increases rapidly (steep curve)
/// - Above optimal: growth slows again (flatter curve)
/// 
/// A spline regression can capture these changing relationships much better than a simple line.
/// </para>
/// </remarks>
public class SplineRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the spline regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control key aspects of the spline regression algorithm, including the number
    /// of knots to use and the polynomial degree of the spline functions.
    /// </para>
    /// <para><b>For Beginners:</b> These settings determine how flexible your model will be:
    /// 
    /// - The number of knots controls how many times your curve can change its behavior
    /// - The degree controls how smoothly the curve bends (higher degree = smoother curves)
    /// 
    /// Like adjusting the sensitivity settings on a drawing tool, these options help
    /// you balance between a model that's too rigid (underfitting) and one that's too
    /// wiggly (overfitting).
    /// </para>
    /// </remarks>
    private readonly SplineRegressionOptions _options;

    /// <summary>
    /// The collection of knot points for each feature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each vector in this list contains the knot positions for a specific input feature. Knots are
    /// the values where the spline functions change their behavior, allowing the model to adapt
    /// to different regions in the data.
    /// </para>
    /// <para><b>For Beginners:</b> Knots are like the special points where your curve can change direction.
    /// 
    /// Think of knots as the joints in a flexible pipe:
    /// - The pipe can bend at each joint
    /// - Between joints, the pipe follows a smooth curve
    /// - More joints = more places where the pipe can change direction
    /// 
    /// These knots help the model capture changing patterns across your data range.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _knots;

    /// <summary>
    /// The coefficients for the spline model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These coefficients determine the weight of each basis function in the final model. The vector includes
    /// coefficients for the constant term, polynomial terms, and spline terms.
    /// </para>
    /// <para><b>For Beginners:</b> These are the numbers that define your model's behavior.
    /// 
    /// Think of coefficients like recipe ingredients:
    /// - Each ingredient (basis function) contributes to the final dish (prediction)
    /// - The coefficients tell you how much of each ingredient to use
    /// - The model finds the perfect "recipe" during training
    /// 
    /// The coefficients are what the model learns when you train it on your data.
    /// </para>
    /// </remarks>
    private Vector<T> _coefficients;

    /// <summary>
    /// Creates a new spline regression model.
    /// </summary>
    /// <param name="options">
    /// Optional configuration settings for the spline regression model. These settings control aspects like:
    /// - The number of knots to use for each feature
    /// - The polynomial degree of the spline functions
    /// - The matrix decomposition method used for solving the system
    /// If not provided, default options will be used.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. Regularization is a technique that helps
    /// the model perform better on new, unseen data by preventing it from fitting the training data too closely.
    /// If not provided, no regularization will be applied.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new spline regression model with the specified configuration options and
    /// regularization method. If options are not provided, default values are used. The constructor
    /// initializes the knots list and coefficients vector to empty collections, which will be populated
    /// during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up a new spline regression model before training.
    /// 
    /// Think of it like preparing to build a flexible curve:
    /// - You decide how many bend points (knots) you want
    /// - You choose how smooth the curve should be (degree)
    /// - You can add safeguards to prevent the curve from becoming too wiggly (regularization)
    /// 
    /// After setting up the model with these options, you'll need to train it on your data
    /// to find the actual curve that best fits your points.
    /// </para>
    /// </remarks>
    public SplineRegression(SplineRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
    : base(options, regularization)
    {
        _options = options ?? new SplineRegressionOptions();
        _knots = new List<Vector<T>>();
        _coefficients = new Vector<T>(0);
    }

    /// <summary>
    /// Optimizes the spline regression model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input feature matrix, where rows represent observations and columns represent features.</param>
    /// <param name="y">The target values vector containing the actual output values to predict.</param>
    /// <remarks>
    /// <para>
    /// This method implements the core optimization for spline regression. It:
    /// 1. Generates appropriate knots for each feature based on data distribution
    /// 2. Creates basis functions for the model (constant, polynomial, and spline terms)
    /// 3. Applies regularization to the basis functions if configured
    /// 4. Solves the linear system to find optimal coefficients
    /// 5. Applies regularization to the coefficients if configured
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best curve to fit your data points.
    /// 
    /// The process works like this:
    /// 
    /// 1. The model places knots (bend points) at strategic positions in your data
    /// 2. It creates a set of special functions that can form curves with bends at these knots
    /// 3. It finds the perfect combination of these functions to match your data
    /// 4. If regularization is enabled, it ensures the curve stays reasonably smooth
    /// 
    /// After optimization, the model has learned the specific curve shape that best represents
    /// the pattern in your data.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Generate knots for each feature
        _knots = new List<Vector<T>>();
        for (int i = 0; i < x.Columns; i++)
        {
            _knots.Add(GenerateKnots(x.GetColumn(i)));
        }

        // Generate basis functions
        var basisFunctions = GenerateBasisFunctions(x);

        // Solve for coefficients with optional ridge regularization
        var xTx = basisFunctions.Transpose().Multiply(basisFunctions);

        // Add ridge regularization to the diagonal if strength is specified
        var regularizationStrength = Regularization?.GetOptions().Strength ?? 0.0;
        if (regularizationStrength > 0)
        {
            T regTerm = NumOps.FromDouble(regularizationStrength);
            for (int i = 0; i < xTx.Rows; i++)
            {
                xTx[i, i] = NumOps.Add(xTx[i, i], regTerm);
            }
        }

        var xTy = basisFunctions.Transpose().Multiply(y);
        _coefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _options.DecompositionType);
    }

    /// <summary>
    /// Generates the basis functions matrix for the input data.
    /// </summary>
    /// <param name="x">The input feature matrix.</param>
    /// <returns>A matrix of basis function values for each observation.</returns>
    /// <remarks>
    /// <para>
    /// This method constructs the basis functions used in the spline regression model. It creates:
    /// 1. A constant term (intercept)
    /// 2. Polynomial terms for each feature up to the specified degree
    /// 3. Spline terms for each knot and feature
    /// 
    /// The resulting matrix contains the values of each basis function evaluated at each data point.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the building blocks for your flexible curve.
    /// 
    /// Think of basis functions like a set of different shaped pieces:
    /// - A flat piece (the constant term)
    /// - Basic curve pieces of different steepness (polynomial terms)
    /// - Special pieces that activate after passing each knot point
    /// 
    /// By combining these pieces with different weights (coefficients), the model
    /// can create a wide variety of curve shapes to match your data pattern.
    /// </para>
    /// </remarks>
    private Matrix<T> GenerateBasisFunctions(Matrix<T> x)
    {
        int totalBasis = 1; // Constant term
        for (int i = 0; i < x.Columns; i++)
        {
            totalBasis += _options.Degree + _knots[i].Length;
        }

        var basis = new Matrix<T>(x.Rows, totalBasis);

        // Constant term
        for (int i = 0; i < x.Rows; i++)
            basis[i, 0] = NumOps.One;

        int columnIndex = 1;

        for (int feature = 0; feature < x.Columns; feature++)
        {
            var featureVector = x.GetColumn(feature);

            // Linear and higher-order terms
            for (int degree = 1; degree <= _options.Degree; degree++)
            {
                for (int i = 0; i < x.Rows; i++)
                    basis[i, columnIndex] = NumOps.Power(featureVector[i], NumOps.FromDouble(degree));
                columnIndex++;
            }

            // Knot terms
            for (int k = 0; k < _knots[feature].Length; k++)
            {
                for (int i = 0; i < x.Rows; i++)
                {
                    var diff = NumOps.Subtract(featureVector[i], _knots[feature][k]);
                    basis[i, columnIndex] = NumOps.GreaterThan(diff, NumOps.Zero)
                        ? NumOps.Power(diff, NumOps.FromDouble(_options.Degree))
                        : NumOps.Zero;
                }
                columnIndex++;
            }
        }

        return basis;
    }

    /// <summary>
    /// Predicts target values for a matrix of input features.
    /// </summary>
    /// <param name="input">The input feature matrix for which to make predictions.</param>
    /// <returns>A vector of predicted values, one for each row in the input matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method makes predictions for multiple input samples. It:
    /// 1. Generates the basis functions for the input data
    /// 2. Multiplies these basis functions by the model coefficients to get predictions
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions for new data.
    /// 
    /// The prediction process works like this:
    /// 1. The model transforms your input data into the special basis functions
    /// 2. It applies the learned coefficients to these functions
    /// 3. It combines the results to produce the final predictions
    /// 
    /// It's like using your recipe (the model) to prepare new dishes (predictions)
    /// using new ingredients (input data).
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var basisFunctions = GenerateBasisFunctions(input);
        return basisFunctions.Multiply(_coefficients);
    }

    /// <summary>
    /// Predicts a target value for a single input feature vector.
    /// </summary>
    /// <param name="input">The input feature vector for which to make a prediction.</param>
    /// <returns>The predicted value for the input vector.</returns>
    /// <remarks>
    /// <para>
    /// This method makes a prediction for a single input sample by:
    /// 1. Converting the input vector to a matrix with one row
    /// 2. Generating the basis functions for this matrix
    /// 3. Multiplying by the coefficients to get a prediction
    /// 4. Extracting the single prediction value
    /// </para>
    /// <para><b>For Beginners:</b> This method predicts the output for a single data point.
    /// 
    /// It works just like the batch prediction method, but for just one input:
    /// 1. Transform the single input into basis functions
    /// 2. Apply the learned coefficients
    /// 3. Return the resulting prediction
    /// 
    /// For example, given a house's features, it would return a single predicted price.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        var basisFunctions = GenerateBasisFunctions(new Matrix<T>([input]));
        return basisFunctions.Multiply(_coefficients)[0];
    }

    /// <summary>
    /// Generates knots for a single feature vector.
    /// </summary>
    /// <param name="x">The feature vector for which to generate knots.</param>
    /// <returns>A vector of knot positions for the feature.</returns>
    /// <remarks>
    /// <para>
    /// This method determines appropriate knot positions for a feature by:
    /// 1. Sorting the feature values in ascending order
    /// 2. Selecting values at regular percentiles of the sorted data
    /// 
    /// This approach ensures that knots are placed where they can effectively capture
    /// the distribution of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides where to place the bend points in your curve.
    /// 
    /// Think of it like this:
    /// - It first arranges all your data points from smallest to largest
    /// - It then places knots at evenly spaced positions throughout this range
    /// - This ensures the knots cover the full range of your data distribution
    /// 
    /// For example, if your data ranges from 0 to 100, and you want 3 knots,
    /// they might be placed at approximately 25, 50, and 75.
    /// </para>
    /// </remarks>
    private Vector<T> GenerateKnots(Vector<T> x)
    {
        int numKnots = _options.NumberOfKnots;
        var sortedX = x.OrderBy(v => Convert.ToDouble(v)).ToArray();
        var knotIndices = Enumerable.Range(1, numKnots)
            .Select(i => (int)Math.Round((double)(i * (sortedX.Length - 1)) / (numKnots + 1)))
            .ToArray();

        return new Vector<T>([.. knotIndices.Select(i => sortedX[i])]);
    }

    /// <summary>
    /// Returns the type identifier for this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for spline regression.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the enum value that identifies this model as a spline regression model. This is used 
    /// for model identification in serialization/deserialization and for logging purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply tells the system what kind of model this is.
    /// 
    /// It's like a name tag for the model that says "I am a spline regression model."
    /// This is useful when:
    /// - Saving the model to a file
    /// - Loading a model from a file
    /// - Logging information about the model
    /// 
    /// You generally won't need to call this method directly in your code.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.SplineRegression;

    /// <summary>
    /// Serializes the spline regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the model, including its coefficients, knots, and configuration options, into a 
    /// byte array. This enables the model to be saved to a file, stored in a database, or transmitted over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to computer memory so you can use it later.
    /// 
    /// Think of it like taking a snapshot of the model:
    /// - It captures all the important values, knots, and coefficients
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

        // Serialize SplineRegression specific data
        writer.Write(_options.NumberOfKnots);
        writer.Write(_options.Degree);

        // Serialize knots
        writer.Write(_knots.Count);
        foreach (var knotVector in _knots)
        {
            writer.Write(knotVector.Length);
            for (int i = 0; i < knotVector.Length; i++)
                writer.Write(Convert.ToDouble(knotVector[i]));
        }

        // Serialize coefficients
        writer.Write(_coefficients.Length);
        for (int i = 0; i < _coefficients.Length; i++)
            writer.Write(Convert.ToDouble(_coefficients[i]));

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the spline regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model from a byte array created by the Serialize method. It restores 
    /// the model's coefficients, knots, and configuration options, allowing a previously saved model 
    /// to be loaded and used for predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a saved model from computer memory.
    /// 
    /// Think of it like opening a saved document:
    /// - It takes the byte array created by the Serialize method
    /// - It rebuilds all the knots, coefficients, and settings
    /// - The model is then ready to use for making predictions
    /// 
    /// This allows you to:
    /// - Use a previously trained model without having to train it again
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

        // Deserialize SplineRegression specific data
        _options.NumberOfKnots = reader.ReadInt32();
        _options.Degree = reader.ReadInt32();

        // Deserialize knots
        int knotsCount = reader.ReadInt32();
        _knots = new List<Vector<T>>();
        for (int j = 0; j < knotsCount; j++)
        {
            int knotsLength = reader.ReadInt32();
            var knotVector = new Vector<T>(knotsLength);
            for (int i = 0; i < knotsLength; i++)
                knotVector[i] = NumOps.FromDouble(reader.ReadDouble());
            _knots.Add(knotVector);
        }

        // Deserialize coefficients
        int coefficientsLength = reader.ReadInt32();
        _coefficients = new Vector<T>(coefficientsLength);
        for (int i = 0; i < coefficientsLength; i++)
            _coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
    }

    /// <summary>
    /// Creates a new instance of the Spline Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Spline Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Spline Regression model, including its options,
    /// knots, coefficients, and regularization settings. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate of your flexible curve:
    /// - It copies all the configuration settings (like number of knots and degree)
    /// - It preserves the locations of all bend points (knots)
    /// - It duplicates all the coefficients that define the curve's shape
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        var newModel = new SplineRegression<T>(_options, Regularization);

        // Deep copy the knots list
        newModel._knots = new List<Vector<T>>();
        foreach (var knotVector in _knots)
        {
            newModel._knots.Add(knotVector.Clone());
        }

        // Deep copy the coefficients
        if (_coefficients != null)
        {
            newModel._coefficients = _coefficients.Clone();
        }

        return newModel;
    }
}
