global using AiDotNet.Regression;

namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Gaussian Process interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Gaussian Process interpolation is a probabilistic approach to interpolation that not only
/// provides estimates for unknown points but also quantifies the uncertainty in those estimates.
/// 
/// <b>For Beginners:</b> This class helps you estimate values between known data points using a technique
/// that's especially good when your data might contain noise or uncertainty. Think of it like
/// drawing a smooth line through your points, but also showing a "confidence band" around that
/// line to indicate how certain the estimates are. It's particularly useful when you have limited
/// data or when you want to know how reliable your estimates are.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class GaussianProcessInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points (independent variable).
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the data points (dependent variable).
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The Gaussian Process Regression model used for interpolation.
    /// </summary>
    private readonly GaussianProcessRegression<T> _gpr;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new Gaussian Process interpolation from the given data points.
    /// </summary>
    /// <remarks>
    /// The constructor initializes and trains the Gaussian Process model with the provided data points.
    /// 
    /// <b>For Beginners:</b> When you create a new GaussianProcessInterpolation object with your data points,
    /// it automatically sets up and trains a statistical model that can make predictions at any point.
    /// The model learns patterns from your data and can use these patterns to make estimates at new points.
    /// The training process involves finding the best parameters that explain your data.
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <exception cref="ArgumentException">Thrown when the input vectors have different lengths.</exception>
    public GaussianProcessInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        // Configure the Gaussian Process Regression options
        var options = new GaussianProcessRegressionOptions
        {
            // Automatically find the best parameters for the model
            OptimizeHyperparameters = true,

            // Maximum number of iterations for parameter optimization
            MaxIterations = 100,

            // Convergence threshold for optimization
            Tolerance = 1e-6,

            // Small amount of noise to improve numerical stability
            NoiseLevel = 1e-8
        };

        // Create the Gaussian Process Regression model with the specified options
        _gpr = new GaussianProcessRegression<T>(options);

        // Train the GPR model with our data points
        // First, convert the x vector to a matrix format required by the GPR
        Matrix<T> xMatrix = new Matrix<T>(_x.Length, 1);
        for (int i = 0; i < _x.Length; i++)
        {
            xMatrix[i, 0] = _x[i];
        }

        // Train the model with the prepared data
        _gpr.Train(xMatrix, _y);
    }

    /// <summary>
    /// Calculates the interpolated y-value for a given x-value using Gaussian Process interpolation.
    /// </summary>
    /// <remarks>
    /// This method uses the trained Gaussian Process model to predict the y-value at the given x-value.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use after creating the interpolation.
    /// Give it any x-value within (or even slightly outside) your data range, and it will return
    /// the estimated y-value at that point. The estimate is based on patterns learned from your
    /// original data points. Unlike simpler methods, Gaussian Processes consider the overall pattern
    /// of your data rather than just connecting points with curves.
    /// </remarks>
    /// <param name="x">The x-value at which to interpolate.</param>
    /// <returns>The interpolated y-value at the given x-value.</returns>
    public T Interpolate(T x)
    {
        // Create a vector containing the single x-value
        Vector<T> xVector = new Vector<T>([x]);

        // Convert the x-value to a matrix format required by the GPR
        Matrix<T> xMatrix = new Matrix<T>(1, 1);
        xMatrix[0, 0] = x;

        // Use the trained model to predict the y-value
        Vector<T> prediction = _gpr.Predict(xMatrix);

        // Return the predicted value
        return prediction[0];
    }
}
