using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Global Surrogate Model explainer that approximates a complex model with an interpretable one.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A global surrogate model is a simple, interpretable model (like
/// linear regression or a decision tree) that tries to mimic a complex "black box" model.
///
/// How it works:
/// 1. Use the complex model to make predictions on a dataset
/// 2. Train a simple model to predict what the complex model predicts
/// 3. Analyze the simple model to understand the complex one
///
/// Think of it like having a translator explain a complex foreign language document.
/// The translator (surrogate model) isn't perfect, but they can explain the main ideas
/// in a way you understand.
///
/// The R² score tells you how well the surrogate approximates the black box:
/// - R² close to 1.0 = surrogate explains the black box well
/// - R² far from 1.0 = surrogate is too simple to capture the black box behavior
/// </para>
/// </remarks>
public class GlobalSurrogateExplainer<T> : IGlobalExplainer<T, SurrogateExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _blackBoxPredictFunction;
    private readonly string[]? _featureNames;

    private Vector<T>? _surrogateCoefficients;
    private T _surrogateIntercept;
    private T _fidelity; // R² of surrogate vs black box

    /// <inheritdoc/>
    public string MethodName => "GlobalSurrogate";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => false;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures { get; }

    /// <summary>
    /// Gets whether the surrogate model has been fitted.
    /// </summary>
    public bool IsFitted => _surrogateCoefficients is not null;

    /// <summary>
    /// Gets the fidelity (R²) of the surrogate model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fidelity measures how well the simple surrogate model
    /// approximates the complex black box model. A fidelity close to 1.0 means
    /// the surrogate explains the black box well; you can trust its explanations.
    /// Low fidelity means the black box is too complex for a linear surrogate.
    /// </para>
    /// </remarks>
    public T Fidelity => IsFitted ? _fidelity : throw new InvalidOperationException("Surrogate not fitted.");

    /// <summary>
    /// Initializes a new Global Surrogate explainer.
    /// </summary>
    /// <param name="blackBoxPredictFunction">A function that returns the black box model's predictions.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="featureNames">Optional names for features.</param>
    public GlobalSurrogateExplainer(
        Func<Matrix<T>, Vector<T>> blackBoxPredictFunction,
        int numFeatures,
        string[]? featureNames = null)
    {
        _blackBoxPredictFunction = blackBoxPredictFunction ?? throw new ArgumentNullException(nameof(blackBoxPredictFunction));

        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (featureNames != null && featureNames.Length != numFeatures)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match numFeatures ({numFeatures}).", nameof(featureNames));

        NumFeatures = numFeatures;
        _featureNames = featureNames;
        _surrogateIntercept = NumOps.Zero;
        _fidelity = NumOps.Zero;
    }

    /// <summary>
    /// Creates a Global Surrogate explainer from a model.
    /// </summary>
    public static GlobalSurrogateExplainer<T> FromModel<TInput, TOutput>(
        IFullModel<T, TInput, TOutput> blackBoxModel,
        int numFeatures,
        string[]? featureNames = null)
    {
        Func<Matrix<T>, Vector<T>> predictFunc = data =>
        {
            var input = ConvertToModelInput<TInput>(data);
            var output = blackBoxModel.Predict(input);
            return ConvertFromModelOutput<TOutput>(output);
        };

        return new GlobalSurrogateExplainer<T>(predictFunc, numFeatures, featureNames);
    }

    /// <summary>
    /// Fits the surrogate model to approximate the black box on the given data.
    /// </summary>
    /// <param name="X">The feature matrix to train on.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method trains the simple surrogate model.
    /// Use representative data that covers the input space your model operates on.
    /// The more diverse the data, the better the surrogate will approximate
    /// the black box's global behavior.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X)
    {
        if (X.Columns != NumFeatures)
            throw new ArgumentException($"Data has {X.Columns} features but expected {NumFeatures}.");

        // Get black box predictions
        var blackBoxPredictions = _blackBoxPredictFunction(X);

        // Fit linear surrogate: y = X * β + intercept
        FitLinearSurrogate(X, blackBoxPredictions);
    }

    /// <inheritdoc/>
    public SurrogateExplanation<T> ExplainGlobal(Matrix<T> data)
    {
        Fit(data);

        return new SurrogateExplanation<T>(
            coefficients: _surrogateCoefficients!,
            intercept: _surrogateIntercept,
            fidelity: _fidelity,
            featureNames: _featureNames);
    }

    /// <summary>
    /// Uses the surrogate model to make predictions.
    /// </summary>
    /// <param name="X">Input features.</param>
    /// <returns>Surrogate model predictions.</returns>
    public Vector<T> PredictSurrogate(Matrix<T> X)
    {
        if (!IsFitted)
            throw new InvalidOperationException("Surrogate model has not been fitted. Call Fit() first.");

        if (X.Columns != NumFeatures)
            throw new ArgumentException($"Data has {X.Columns} features but expected {NumFeatures}.");

        var predictions = new T[X.Rows];
        for (int i = 0; i < X.Rows; i++)
        {
            var pred = _surrogateIntercept;
            for (int j = 0; j < NumFeatures; j++)
            {
                pred = NumOps.Add(pred, NumOps.Multiply(_surrogateCoefficients![j], X[i, j]));
            }
            predictions[i] = pred;
        }

        return new Vector<T>(predictions);
    }

    /// <summary>
    /// Compares the surrogate predictions with the black box predictions.
    /// </summary>
    /// <param name="X">Input features.</param>
    /// <returns>A tuple containing (surrogate predictions, black box predictions, R² fidelity).</returns>
    public (Vector<T> SurrogatePredictions, Vector<T> BlackBoxPredictions, T Fidelity) Compare(Matrix<T> X)
    {
        var surrogatePreds = PredictSurrogate(X);
        var blackBoxPreds = _blackBoxPredictFunction(X);
        var fidelity = ComputeR2(blackBoxPreds, surrogatePreds);

        return (surrogatePreds, blackBoxPreds, fidelity);
    }

    /// <summary>
    /// Fits a linear regression model to approximate the black box.
    /// </summary>
    private void FitLinearSurrogate(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        int p = X.Columns;

        // Center the data
        var xMeans = new double[p];
        double yMean = 0;

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                xMeans[j] += NumOps.ToDouble(X[i, j]);
            }
            xMeans[j] /= n;
        }

        for (int i = 0; i < n; i++)
        {
            yMean += NumOps.ToDouble(y[i]);
        }
        yMean /= n;

        // Compute X'X and X'y
        var XtX = new double[p, p];
        var Xty = new double[p];

        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = 0; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    double xij1 = NumOps.ToDouble(X[i, j1]) - xMeans[j1];
                    double xij2 = NumOps.ToDouble(X[i, j2]) - xMeans[j2];
                    sum += xij1 * xij2;
                }
                XtX[j1, j2] = sum;
            }

            double sumY = 0;
            for (int i = 0; i < n; i++)
            {
                double xij = NumOps.ToDouble(X[i, j1]) - xMeans[j1];
                double yi = NumOps.ToDouble(y[i]) - yMean;
                sumY += xij * yi;
            }
            Xty[j1] = sumY;
        }

        // Add regularization
        for (int j = 0; j < p; j++)
        {
            XtX[j, j] += 1e-6;
        }

        // Solve for coefficients
        var beta = SolveLinearSystem(XtX, Xty);

        // Compute intercept
        double intercept = yMean;
        for (int j = 0; j < p; j++)
        {
            intercept -= beta[j] * xMeans[j];
        }

        // Store results
        var coefficients = new T[p];
        for (int j = 0; j < p; j++)
        {
            coefficients[j] = NumOps.FromDouble(beta[j]);
        }
        _surrogateCoefficients = new Vector<T>(coefficients);
        _surrogateIntercept = NumOps.FromDouble(intercept);

        // Compute fidelity (R²)
        var predictions = PredictSurrogate(X);
        _fidelity = ComputeR2(y, predictions);
    }

    private double[] SolveLinearSystem(double[,] A, double[] b)
    {
        int n = b.Length;
        var augmented = new double[n, n + 1];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int k = 0; k < n; k++)
        {
            int maxRow = k;
            for (int i = k + 1; i < n; i++)
            {
                if (Math.Abs(augmented[i, k]) > Math.Abs(augmented[maxRow, k]))
                {
                    maxRow = i;
                }
            }

            for (int j = k; j <= n; j++)
            {
                (augmented[k, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[k, j]);
            }

            for (int i = k + 1; i < n; i++)
            {
                if (Math.Abs(augmented[k, k]) > 1e-10)
                {
                    double factor = augmented[i, k] / augmented[k, k];
                    for (int j = k; j <= n; j++)
                    {
                        augmented[i, j] -= factor * augmented[k, j];
                    }
                }
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                x[i] -= augmented[i, j] * x[j];
            }
            if (Math.Abs(augmented[i, i]) > 1e-10)
            {
                x[i] /= augmented[i, i];
            }
            else
            {
                // Near-singular matrix: throw exception to prevent silent failures
                throw new InvalidOperationException(
                    $"Near-singular matrix detected at index {i}. " +
                    "The surrogate model coefficients cannot be reliably computed. " +
                    "This may indicate multicollinearity in the features or insufficient data variation.");
            }
        }

        return x;
    }

    private T ComputeR2(Vector<T> actual, Vector<T> predicted)
    {
        int n = actual.Length;
        if (n == 0) return NumOps.Zero;

        double mean = 0;
        for (int i = 0; i < n; i++)
            mean += NumOps.ToDouble(actual[i]);
        mean /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double a = NumOps.ToDouble(actual[i]);
            double p = NumOps.ToDouble(predicted[i]);
            ssRes += (a - p) * (a - p);
            ssTot += (a - mean) * (a - mean);
        }

        if (ssTot < 1e-10) return NumOps.Zero;
        return NumOps.FromDouble(1 - ssRes / ssTot);
    }

    private static TInput ConvertToModelInput<TInput>(Matrix<T> data)
    {
        object result;
        if (typeof(TInput) == typeof(Matrix<T>))
            result = data;
        else if (typeof(TInput) == typeof(Tensor<T>))
            result = Tensor<T>.FromRowMatrix(data);
        else if (typeof(TInput) == typeof(Vector<T>) && data.Rows == 1)
            result = data.GetRow(0);
        else
            throw new NotSupportedException($"Cannot convert Matrix<T> to {typeof(TInput).Name}");

        return (TInput)result;
    }

    private static Vector<T> ConvertFromModelOutput<TOutput>(TOutput output)
    {
        if (output is Vector<T> vector)
            return vector;
        if (output is Matrix<T> matrix)
            return matrix.GetColumn(0);
        if (output is Tensor<T> tensor)
            return tensor.ToVector();
        if (output is T scalar)
            return new Vector<T>([scalar]);

        throw new NotSupportedException($"Cannot convert {typeof(TOutput).Name} to Vector<T>");
    }
}

/// <summary>
/// Represents a global surrogate model explanation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SurrogateExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the coefficients of the linear surrogate model.
    /// </summary>
    public Vector<T> Coefficients { get; }

    /// <summary>
    /// Gets the intercept of the linear surrogate model.
    /// </summary>
    public T Intercept { get; }

    /// <summary>
    /// Gets the fidelity (R²) indicating how well the surrogate approximates the black box.
    /// </summary>
    public T Fidelity { get; }

    /// <summary>
    /// Gets the feature names, if available.
    /// </summary>
    public string[]? FeatureNames { get; }

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures => Coefficients.Length;

    /// <summary>
    /// Initializes a new surrogate explanation.
    /// </summary>
    public SurrogateExplanation(
        Vector<T> coefficients,
        T intercept,
        T fidelity,
        string[]? featureNames = null)
    {
        Coefficients = coefficients ?? throw new ArgumentNullException(nameof(coefficients));
        Intercept = intercept;
        Fidelity = fidelity;
        FeatureNames = featureNames;
    }

    /// <summary>
    /// Gets features sorted by absolute coefficient (most important first).
    /// </summary>
    public IEnumerable<(int Index, string? Name, T Weight)> GetSortedFeatures()
    {
        return Enumerable.Range(0, NumFeatures)
            .Select(i => (
                Index: i,
                Name: FeatureNames?[i],
                Weight: Coefficients[i]))
            .OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Weight)));
    }

    /// <summary>
    /// Gets the top N most important features according to the surrogate model.
    /// </summary>
    public IEnumerable<(int Index, string? Name, T Weight)> GetTopFeatures(int n)
    {
        return GetSortedFeatures().Take(n);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopFeatures(10).ToList();
        var lines = new List<string>
        {
            $"Global Surrogate Explanation (Linear Model):",
            $"  Fidelity (R²): {Fidelity}",
            $"  Intercept: {Intercept}",
            $"  Feature Weights:"
        };

        foreach (var (index, name, weight) in top)
        {
            var featureLabel = name ?? $"Feature {index}";
            var direction = NumOps.ToDouble(weight) >= 0 ? "+" : "";
            lines.Add($"    {featureLabel}: {direction}{weight}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
