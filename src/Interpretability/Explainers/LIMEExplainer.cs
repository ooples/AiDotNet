using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic LIME (Local Interpretable Model-agnostic Explanations) explainer.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LIME explains individual predictions by building a simple,
/// interpretable model (like linear regression) that approximates the complex model
/// locally around the prediction you want to understand.
///
/// How it works:
/// 1. Generate perturbed samples near your instance (slightly modified versions)
/// 2. Get the complex model's predictions for these samples
/// 3. Fit a simple linear model to these nearby predictions
/// 4. The linear model's coefficients show which features matter most
///
/// Think of it like zooming in on a curvy road - if you zoom in enough,
/// even a curved road looks straight. LIME zooms in on your prediction
/// and fits a "straight line" (linear model) to explain it.
/// </para>
/// </remarks>
public class LIMEExplainer<T> : ILocalExplainer<T, LIMEExplanationResult<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly int _nSamples;
    private readonly double _kernelWidth;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;
    private readonly double[]? _featureStdDevs;

    /// <inheritdoc/>
    public string MethodName => "LIME";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Gets the number of features being explained.
    /// </summary>
    public int NumFeatures { get; }

    /// <summary>
    /// Initializes a new LIME explainer.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="nSamples">Number of perturbed samples to generate (default: 500).</param>
    /// <param name="kernelWidth">Width of the exponential kernel for weighting samples (default: 0.75).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="featureStdDevs">Standard deviations for perturbing features (default: 1.0 for all).</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>nSamples</b>: More samples = more accurate but slower. Start with 500.
    /// - <b>kernelWidth</b>: Controls how "local" the explanation is. Smaller = more local.
    /// - <b>featureStdDevs</b>: How much to perturb each feature. Use feature scales from your data.
    /// </para>
    /// </remarks>
    public LIMEExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        int numFeatures,
        int nSamples = 500,
        double kernelWidth = 0.75,
        string[]? featureNames = null,
        double[]? featureStdDevs = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));

        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (nSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1.", nameof(nSamples));
        if (kernelWidth <= 0)
            throw new ArgumentException("Kernel width must be positive.", nameof(kernelWidth));
        if (featureNames != null && featureNames.Length != numFeatures)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match numFeatures ({numFeatures}).", nameof(featureNames));
        if (featureStdDevs != null && featureStdDevs.Length != numFeatures)
            throw new ArgumentException($"featureStdDevs length ({featureStdDevs.Length}) must match numFeatures ({numFeatures}).", nameof(featureStdDevs));

        NumFeatures = numFeatures;
        _nSamples = nSamples;
        _kernelWidth = kernelWidth;
        _featureNames = featureNames;
        _featureStdDevs = featureStdDevs;
        _randomState = randomState;
    }

    /// <summary>
    /// Creates a LIME explainer from a model.
    /// </summary>
    public static LIMEExplainer<T> FromModel<TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        int numFeatures,
        int nSamples = 500,
        double kernelWidth = 0.75,
        string[]? featureNames = null,
        double[]? featureStdDevs = null,
        int? randomState = null)
    {
        Func<Matrix<T>, Vector<T>> predictFunc = data =>
        {
            var input = ConvertToModelInput<TInput>(data);
            var output = model.Predict(input);
            return ConvertFromModelOutput<TOutput>(output);
        };

        return new LIMEExplainer<T>(predictFunc, numFeatures, nSamples, kernelWidth,
            featureNames, featureStdDevs, randomState);
    }

    /// <inheritdoc/>
    public LIMEExplanationResult<T> Explain(Vector<T> instance)
    {
        if (instance.Length != NumFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but explainer expects {NumFeatures}.");

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Generate perturbed samples
        var (perturbedData, perturbedBinary) = GeneratePerturbedSamples(instance, rand);

        // Get predictions for perturbed samples
        var predictions = _predictFunction(perturbedData);

        // Compute distances and weights
        var weights = ComputeWeights(perturbedBinary, instance);

        // Fit weighted linear model
        var (coefficients, intercept, r2) = FitWeightedLinearModel(perturbedBinary, predictions, weights);

        // Get original prediction
        var originalPrediction = PredictSingle(instance);

        return new LIMEExplanationResult<T>(
            coefficients: coefficients,
            intercept: intercept,
            prediction: originalPrediction,
            localR2: r2,
            featureNames: _featureNames);
    }

    /// <inheritdoc/>
    public LIMEExplanationResult<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new LIMEExplanationResult<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Generates perturbed samples around the instance.
    /// </summary>
    private (Matrix<T> Data, Matrix<T> Binary) GeneratePerturbedSamples(Vector<T> instance, Random rand)
    {
        var data = new T[_nSamples, NumFeatures];
        var binary = new T[_nSamples, NumFeatures];

        // First sample is the original instance
        for (int j = 0; j < NumFeatures; j++)
        {
            data[0, j] = instance[j];
            binary[0, j] = NumOps.One;
        }

        // Generate perturbed samples
        for (int i = 1; i < _nSamples; i++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                // Randomly decide whether to keep or perturb this feature
                bool keep = rand.NextDouble() > 0.5;
                binary[i, j] = keep ? NumOps.One : NumOps.Zero;

                if (keep)
                {
                    data[i, j] = instance[j];
                }
                else
                {
                    // Perturb with Gaussian noise
                    double stdDev = _featureStdDevs?[j] ?? 1.0;
                    double originalValue = NumOps.ToDouble(instance[j]);
                    double perturbation = SampleGaussian(rand) * stdDev;
                    data[i, j] = NumOps.FromDouble(originalValue + perturbation);
                }
            }
        }

        return (new Matrix<T>(data), new Matrix<T>(binary));
    }

    /// <summary>
    /// Computes sample weights based on distance from original instance.
    /// </summary>
    private Vector<T> ComputeWeights(Matrix<T> binaryData, Vector<T> instance)
    {
        var weights = new T[binaryData.Rows];

        for (int i = 0; i < binaryData.Rows; i++)
        {
            // Compute Euclidean distance in binary space
            double distance = 0;
            for (int j = 0; j < NumFeatures; j++)
            {
                double diff = 1.0 - NumOps.ToDouble(binaryData[i, j]);
                distance += diff * diff;
            }
            distance = Math.Sqrt(distance);

            // Exponential kernel
            double weight = Math.Exp(-distance * distance / (_kernelWidth * _kernelWidth));
            weights[i] = NumOps.FromDouble(weight);
        }

        return new Vector<T>(weights);
    }

    /// <summary>
    /// Fits a weighted linear regression model.
    /// </summary>
    private (Vector<T> Coefficients, T Intercept, T R2) FitWeightedLinearModel(
        Matrix<T> X,
        Vector<T> y,
        Vector<T> weights)
    {
        int n = X.Rows;
        int p = X.Columns;

        // Add intercept column
        var XWithIntercept = new double[n, p + 1];
        for (int i = 0; i < n; i++)
        {
            XWithIntercept[i, 0] = 1.0; // Intercept
            for (int j = 0; j < p; j++)
            {
                XWithIntercept[i, j + 1] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Convert to double arrays
        var yDouble = new double[n];
        var wDouble = new double[n];
        for (int i = 0; i < n; i++)
        {
            yDouble[i] = NumOps.ToDouble(y[i]);
            wDouble[i] = NumOps.ToDouble(weights[i]);
        }

        // Weighted least squares: (X'WX)^-1 X'Wy
        var XtWX = new double[p + 1, p + 1];
        var XtWy = new double[p + 1];

        for (int j1 = 0; j1 <= p; j1++)
        {
            for (int j2 = 0; j2 <= p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += XWithIntercept[i, j1] * wDouble[i] * XWithIntercept[i, j2];
                }
                XtWX[j1, j2] = sum;
            }

            double sumY = 0;
            for (int i = 0; i < n; i++)
            {
                sumY += XWithIntercept[i, j1] * wDouble[i] * yDouble[i];
            }
            XtWy[j1] = sumY;
        }

        // Add regularization
        for (int j = 0; j <= p; j++)
        {
            XtWX[j, j] += 1e-6;
        }

        // Solve
        var beta = SolveLinearSystem(XtWX, XtWy);

        // Extract intercept and coefficients
        T intercept = NumOps.FromDouble(beta[0]);
        var coefficients = new T[p];
        for (int j = 0; j < p; j++)
        {
            coefficients[j] = NumOps.FromDouble(beta[j + 1]);
        }

        // Compute R²
        double ssRes = 0, ssTot = 0;
        double yMean = yDouble.Average();
        for (int i = 0; i < n; i++)
        {
            double yPred = beta[0];
            for (int j = 0; j < p; j++)
            {
                yPred += beta[j + 1] * NumOps.ToDouble(X[i, j]);
            }
            double residual = yDouble[i] - yPred;
            ssRes += wDouble[i] * residual * residual;
            ssTot += wDouble[i] * (yDouble[i] - yMean) * (yDouble[i] - yMean);
        }

        T r2 = ssTot > 1e-10 ? NumOps.FromDouble(1 - ssRes / ssTot) : NumOps.Zero;

        return (new Vector<T>(coefficients), intercept, r2);
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
                // Near-singular matrix: set to NaN to indicate unreliable result
                x[i] = double.NaN;
            }
        }

        return x;
    }

    private double SampleGaussian(Random rand)
    {
        // Guard against u1 == 0 which would cause Log(0) = -Infinity and result in NaN
        double u1;
        do { u1 = rand.NextDouble(); } while (u1 == 0);
        double u2 = rand.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private T PredictSingle(Vector<T> instance)
    {
        var matrix = new Matrix<T>(1, instance.Length);
        for (int j = 0; j < instance.Length; j++)
        {
            matrix[0, j] = instance[j];
        }
        var predictions = _predictFunction(matrix);
        return predictions[0];
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
/// Represents a LIME explanation result.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LIMEExplanationResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the coefficients of the local linear model (feature weights).
    /// </summary>
    public Vector<T> Coefficients { get; }

    /// <summary>
    /// Gets the intercept of the local linear model.
    /// </summary>
    public T Intercept { get; }

    /// <summary>
    /// Gets the original prediction being explained.
    /// </summary>
    public T Prediction { get; }

    /// <summary>
    /// Gets the R² score of the local linear model (how well it fits locally).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> R² close to 1.0 means the simple linear model explains
    /// the complex model well in this local region. Low R² means the model behavior
    /// is harder to approximate with a linear model locally.
    /// </para>
    /// </remarks>
    public T LocalR2 { get; }

    /// <summary>
    /// Gets the feature names, if available.
    /// </summary>
    public string[]? FeatureNames { get; }

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures => Coefficients.Length;

    /// <summary>
    /// Initializes a new LIME explanation result.
    /// </summary>
    public LIMEExplanationResult(
        Vector<T> coefficients,
        T intercept,
        T prediction,
        T localR2,
        string[]? featureNames = null)
    {
        Coefficients = coefficients ?? throw new ArgumentNullException(nameof(coefficients));
        Intercept = intercept;
        Prediction = prediction;
        LocalR2 = localR2;
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
    /// Gets the top N most important features.
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
        var top = GetTopFeatures(5).ToList();
        var lines = new List<string>
        {
            $"LIME Explanation:",
            $"  Prediction: {Prediction}",
            $"  Local R²: {LocalR2}",
            $"  Intercept: {Intercept}",
            $"  Top contributing features:"
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
