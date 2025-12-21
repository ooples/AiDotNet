namespace AiDotNet.Regression;

/// <summary>
/// Implements polynomial regression, which extends linear regression by fitting a polynomial equation to the data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// Polynomial regression is useful when the relationship between variables is not linear.
/// It works by creating new features that are powers of the original features (x, x², x³, etc.),
/// then applying linear regression techniques to these expanded features.
/// 
/// <b>For Beginners:</b> While linear regression fits a straight line to your data,
/// polynomial regression can fit curves, allowing it to capture more complex patterns.
/// </remarks>
public class PolynomialRegression<T> : RegressionBase<T>
{
    private readonly PolynomialRegressionOptions<T> _polyOptions;

    /// <summary>
    /// Creates a new instance of the polynomial regression model.
    /// </summary>
    /// <param name="options">
    /// Configuration options for the polynomial regression model, including the degree of the polynomial.
    /// The degree determines how complex the curve can be (e.g., degree 2 allows for parabolas).
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. Regularization adds a penalty for large 
    /// coefficient values, which helps create a more generalizable model.
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> The degree of the polynomial determines how "curvy" your model can be.
    /// A higher degree (like 3 or 4) allows for more complex curves but may lead to overfitting
    /// if you don't have enough data.
    /// </remarks>
    public PolynomialRegression(PolynomialRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _polyOptions = options ?? new PolynomialRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the polynomial regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">
    /// The input feature matrix where each row represents a data point and each column represents a feature.
    /// </param>
    /// <param name="y">
    /// The target values vector where each element corresponds to a row in the input matrix.
    /// </param>
    /// <remarks>
    /// This method:
    /// 1. Transforms the original features into polynomial features (x, x², x³, etc.)
    /// 2. Adds a constant column for the intercept if specified in the options
    /// 3. Solves the least squares equation to find the optimal coefficients
    /// 
    /// <b>For Beginners:</b> Training means finding the best curve that fits your data points.
    /// The algorithm creates additional features by raising your original features to different powers,
    /// then finds the best combination of these features to predict your target values.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        var polyX = CreatePolynomialFeatures(x);

        if (Options.UseIntercept)
            polyX = polyX.AddConstantColumn(NumOps.One);

        var xTx = polyX.Transpose().Multiply(polyX);
        var regularizedXTx = xTx.Add(Regularization.Regularize(xTx));
        var xTy = polyX.Transpose().Multiply(y);

        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>([.. solution.Skip(1)]);
        }
        else
        {
            Coefficients = new Vector<T>(solution);
        }
    }

    /// <summary>
    /// Creates polynomial features from the original input features.
    /// </summary>
    /// <param name="x">The original input feature matrix.</param>
    /// <returns>A new matrix with polynomial features up to the specified degree.</returns>
    /// <remarks>
    /// This method transforms each feature x into multiple features: x, x², x³, etc.,
    /// up to the degree specified in the options.
    /// </remarks>
    private Matrix<T> CreatePolynomialFeatures(Matrix<T> x)
    {
        var rows = x.Rows;
        var cols = x.Columns * _polyOptions.Degree;
        var polyX = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                for (int d = 0; d < _polyOptions.Degree; d++)
                {
                    polyX[i, j * _polyOptions.Degree + d] = NumOps.Power(x[i, j], NumOps.FromDouble(d + 1));
                }
            }
        }

        return polyX;
    }

    /// <summary>
    /// Makes predictions using the trained polynomial regression model.
    /// </summary>
    /// <param name="input">
    /// The input feature matrix where each row represents a data point to predict and each column represents a feature.
    /// </param>
    /// <returns>
    /// A vector of predicted values, one for each row in the input matrix.
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> Once the model is trained, this method uses the discovered polynomial equation
    /// to predict outcomes for new data points. It first transforms the input features into polynomial features,
    /// then applies the learned coefficients to make predictions.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var polyInput = CreatePolynomialFeatures(input);
        return base.Predict(polyInput);
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type identifier for polynomial regression.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.PolynomialRegression;
    }

    /// <summary>
    /// Creates a new instance of the Polynomial Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Polynomial Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// This method creates a deep copy of the current Polynomial Regression model, including its coefficients,
    /// intercept, and configuration options. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original model.
    /// 
    /// <b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate recipe:
    /// - It copies all the configuration settings (like the polynomial degree)
    /// - It preserves the coefficients (the weights for each polynomial term)
    /// - It maintains the intercept (the starting point of your curve)
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        var newModel = new PolynomialRegression<T>(_polyOptions, Regularization);

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
