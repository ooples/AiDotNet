global using AiDotNet.Extensions;

namespace AiDotNet.Regression;

/// <summary>
/// Implements weighted regression, a variation of linear regression where each data point has a different 
/// level of importance (weight) in determining the model parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// Weighted regression is useful when:
/// - Some data points are more reliable than others
/// - You want to give more importance to certain observations
/// - You're dealing with heteroscedastic data (data with varying levels of noise)
/// 
/// The weights determine how much each data point influences the final model.
/// Higher weights mean more influence, lower weights mean less influence.
/// </remarks>
public class WeightedRegression<T> : RegressionBase<T>
{
    private readonly Vector<T> _weights;
    private readonly int _order;

    /// <summary>
    /// Creates a new instance of the weighted regression model.
    /// </summary>
    /// <param name="options">
    /// Configuration options for the weighted regression model, including the weights for each data point
    /// and the polynomial order. The weights determine how much influence each data point has on the model.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. Regularization adds a penalty for large 
    /// coefficient values, which helps create a more generalizable model.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when the options parameter is null or doesn't contain weights.
    /// </exception>
    /// <remarks>
    /// For beginners: Think of weights as a way to tell the model "pay more attention to these points 
    /// and less attention to those points" when finding the best fit line.
    /// </remarks>
    public WeightedRegression(WeightedRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _weights = options?.Weights ?? throw new ArgumentNullException(nameof(options), "Weights must be provided for weighted regression.");
        _order = options.Order;
    }

    /// <summary>
    /// Trains the weighted regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">
    /// The input feature matrix where each row represents a data point and each column represents a feature.
    /// </param>
    /// <param name="y">
    /// The target values vector where each element corresponds to a row in the input matrix.
    /// </param>
    /// <remarks>
    /// This method:
    /// 1. Expands the features to the specified polynomial order
    /// 2. Adds a constant column for the intercept if specified in the options
    /// 3. Applies the weights to give different importance to each data point
    /// 4. Solves the weighted least squares equation to find the optimal coefficients
    /// 
    /// For beginners: Training means finding the best line (or curve) that fits your data points,
    /// while taking into account how important each point is (based on its weight).
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        var expandedX = ExpandFeatures(x);

        if (Options.UseIntercept)
            expandedX = expandedX.AddConstantColumn(NumOps.One);

        var weightMatrix = Matrix<T>.CreateDiagonal(_weights);
        var xTWx = expandedX.Transpose().Multiply(weightMatrix).Multiply(expandedX);
        var regularizedXTWx = xTWx.Add(Regularization.RegularizeMatrix(xTWx));
        var xTWy = expandedX.Transpose().Multiply(weightMatrix).Multiply(y);

        var solution = SolveSystem(regularizedXTWx, xTWy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = solution.Slice(1, solution.Length - 1);
        }
        else
        {
            Coefficients = solution;
        }
    }

    /// <summary>
    /// Makes predictions using the trained weighted regression model.
    /// </summary>
    /// <param name="input">
    /// The input feature matrix where each row represents a data point to predict and each column represents a feature.
    /// </param>
    /// <returns>
    /// A vector of predicted values, one for each row in the input matrix.
    /// </returns>
    /// <remarks>
    /// For beginners: Once the model is trained, this method uses the discovered pattern (equation)
    /// to predict outcomes for new data points.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var expandedInput = ExpandFeatures(input);
        return base.Predict(expandedInput);
    }

    /// <summary>
    /// Expands the input features to include polynomial terms up to the specified order.
    /// </summary>
    /// <param name="x">The original input feature matrix.</param>
    /// <returns>
    /// A new matrix with expanded polynomial features.
    /// </returns>
    private Matrix<T> ExpandFeatures(Matrix<T> x)
    {
        var expandedX = new Matrix<T>(x.Rows, x.Columns * _order);

        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                for (int k = 0; k < _order; k++)
                {
                    expandedX[i, j * _order + k] = NumOps.Power(x[i, j], NumOps.FromDouble(k + 1));
                }
            }
        }

        return expandedX;
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type identifier for weighted regression.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.WeightedRegression;
    }
}