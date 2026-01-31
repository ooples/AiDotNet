using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic SHAP (SHapley Additive exPlanations) explainer using Kernel SHAP algorithm.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SHAP values come from game theory and answer the question:
/// "How much did each feature contribute to this specific prediction?"
///
/// Imagine you're splitting a restaurant bill fairly among friends based on what each person ordered.
/// SHAP does something similar - it fairly distributes the "credit" for a prediction among all input features.
///
/// Key properties of SHAP values:
/// - They sum up to the difference between the prediction and the average prediction
/// - Positive values mean the feature pushed the prediction higher
/// - Negative values mean the feature pushed the prediction lower
/// - The magnitude shows how important that feature was
///
/// This implementation uses Kernel SHAP, which works with ANY model by treating it as a black box.
/// </para>
/// </remarks>
public class SHAPExplainer<T> : ILocalExplainer<T, SHAPExplanation<T>>, IGlobalExplainer<T, GlobalSHAPExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly Matrix<T> _backgroundData;
    private readonly int _nSamples;
    private readonly int? _randomState;
    private readonly T _baselineValue;
    private readonly string[]? _featureNames;

    /// <inheritdoc/>
    public string MethodName => "KernelSHAP";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <summary>
    /// Gets the baseline (expected) prediction value computed from background data.
    /// </summary>
    public T BaselineValue => _baselineValue;

    /// <summary>
    /// Gets the number of features being explained.
    /// </summary>
    public int NumFeatures => _backgroundData.Columns;

    /// <summary>
    /// Initializes a new SHAP explainer with a prediction function and background data.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="backgroundData">Representative data used to compute expected values.</param>
    /// <param name="nSamples">Number of samples for Kernel SHAP approximation (default: 100).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The background data represents "typical" inputs to your model.
    /// SHAP compares each prediction to what the model would predict on average for the background data.
    /// Using training data or a sample of it usually works well.
    ///
    /// More samples (nSamples) gives more accurate SHAP values but takes longer to compute.
    /// Start with 100 and increase if you need more precision.
    /// </para>
    /// </remarks>
    public SHAPExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Matrix<T> backgroundData,
        int nSamples = 100,
        string[]? featureNames = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _backgroundData = backgroundData ?? throw new ArgumentNullException(nameof(backgroundData));

        if (backgroundData.Rows == 0)
            throw new ArgumentException("Background data must have at least one row.", nameof(backgroundData));
        if (nSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1.", nameof(nSamples));

        _nSamples = nSamples;
        _featureNames = featureNames;
        _randomState = randomState;

        // Compute baseline value (expected prediction on background data)
        var backgroundPredictions = _predictFunction(_backgroundData);
        _baselineValue = ComputeMean(backgroundPredictions);
    }

    /// <summary>
    /// Initializes a new SHAP explainer with a model that implements prediction.
    /// </summary>
    /// <typeparam name="TInput">The model's input type.</typeparam>
    /// <typeparam name="TOutput">The model's output type.</typeparam>
    /// <param name="model">The model to explain.</param>
    /// <param name="backgroundData">Representative data used to compute expected values.</param>
    /// <param name="nSamples">Number of samples for Kernel SHAP approximation.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    public static SHAPExplainer<T> FromModel<TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        Matrix<T> backgroundData,
        int nSamples = 100,
        string[]? featureNames = null,
        int? randomState = null)
    {
        Func<Matrix<T>, Vector<T>> predictFunc = data =>
        {
            var input = ConvertToModelInput<TInput>(data);
            var output = model.Predict(input);
            return ConvertFromModelOutput<TOutput>(output);
        };

        return new SHAPExplainer<T>(predictFunc, backgroundData, nSamples, featureNames, randomState);
    }

    /// <inheritdoc/>
    public SHAPExplanation<T> Explain(Vector<T> instance)
    {
        if (instance.Length != NumFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but model expects {NumFeatures}.");

        var shapValues = ComputeKernelSHAP(instance);
        var prediction = PredictSingle(instance);

        return new SHAPExplanation<T>(
            shapValues: shapValues,
            baselineValue: _baselineValue,
            prediction: prediction,
            featureNames: _featureNames);
    }

    /// <inheritdoc/>
    public SHAPExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new SHAPExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            var instance = instances.GetRow(i);
            explanations[i] = Explain(instance);
        }
        return explanations;
    }

    /// <inheritdoc/>
    public GlobalSHAPExplanation<T> ExplainGlobal(Matrix<T> data)
    {
        var localExplanations = ExplainBatch(data);
        return new GlobalSHAPExplanation<T>(localExplanations, _featureNames);
    }

    /// <summary>
    /// Computes SHAP values using the Kernel SHAP algorithm.
    /// </summary>
    private Vector<T> ComputeKernelSHAP(Vector<T> instance)
    {
        int numFeatures = instance.Length;
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Generate coalition samples
        var coalitions = new List<bool[]>();
        var weights = new List<double>();
        var predictions = new List<T>();

        // Always include empty and full coalitions
        coalitions.Add(new bool[numFeatures]); // All false
        weights.Add(double.MaxValue / 2); // High weight for empty coalition

        var fullCoalition = new bool[numFeatures];
        for (int i = 0; i < numFeatures; i++)
            fullCoalition[i] = true;
        coalitions.Add(fullCoalition);
        weights.Add(double.MaxValue / 2); // High weight for full coalition

        // Sample random coalitions
        for (int s = 0; s < _nSamples; s++)
        {
            var coalition = new bool[numFeatures];
            int coalitionSize = 0;

            for (int j = 0; j < numFeatures; j++)
            {
                coalition[j] = rand.NextDouble() > 0.5;
                if (coalition[j]) coalitionSize++;
            }

            // Kernel SHAP weight: M / (k * (M - k) * C(M, k))
            // Simplified: weight inversely proportional to distance from 0 and M
            double weight = 1.0;
            if (coalitionSize > 0 && coalitionSize < numFeatures)
            {
                weight = (numFeatures - 1.0) / (coalitionSize * (numFeatures - coalitionSize));
            }

            coalitions.Add(coalition);
            weights.Add(weight);
        }

        // Compute predictions for each coalition
        foreach (var coalition in coalitions)
        {
            var maskedPrediction = ComputeCoalitionPrediction(instance, coalition);
            predictions.Add(maskedPrediction);
        }

        // Solve weighted least squares to get SHAP values
        var shapValues = SolveWeightedLeastSquares(coalitions, predictions, weights, numFeatures);

        return shapValues;
    }

    /// <summary>
    /// Computes the expected prediction for a coalition by marginalizing over background data.
    /// </summary>
    private T ComputeCoalitionPrediction(Vector<T> instance, bool[] coalition)
    {
        int nBackground = Math.Min(_backgroundData.Rows, 10); // Limit for efficiency
        var maskedInputs = new T[nBackground, NumFeatures];

        for (int b = 0; b < nBackground; b++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                // If feature is in coalition, use instance value; otherwise use background value
                maskedInputs[b, j] = coalition[j] ? instance[j] : _backgroundData[b, j];
            }
        }

        var maskedMatrix = new Matrix<T>(maskedInputs);
        var predictions = _predictFunction(maskedMatrix);
        return ComputeMean(predictions);
    }

    /// <summary>
    /// Solves weighted least squares to compute SHAP values from coalition predictions.
    /// </summary>
    private Vector<T> SolveWeightedLeastSquares(
        List<bool[]> coalitions,
        List<T> predictions,
        List<double> weights,
        int numFeatures)
    {
        int n = coalitions.Count;

        // Build design matrix X (coalition indicators) and target vector y
        var X = new double[n, numFeatures];
        var y = new double[n];
        var w = new double[n];

        double maxWeight = weights.Max();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                X[i, j] = coalitions[i][j] ? 1.0 : 0.0;
            }
            y[i] = NumOps.ToDouble(predictions[i]) - NumOps.ToDouble(_baselineValue);
            w[i] = Math.Min(weights[i], maxWeight) / maxWeight; // Normalize weights
        }

        // Weighted least squares: (X'WX)^-1 X'Wy
        var XtWX = new double[numFeatures, numFeatures];
        var XtWy = new double[numFeatures];

        for (int j1 = 0; j1 < numFeatures; j1++)
        {
            for (int j2 = 0; j2 < numFeatures; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += X[i, j1] * w[i] * X[i, j2];
                }
                XtWX[j1, j2] = sum;
            }

            double sumY = 0;
            for (int i = 0; i < n; i++)
            {
                sumY += X[i, j1] * w[i] * y[i];
            }
            XtWy[j1] = sumY;
        }

        // Add regularization for numerical stability
        for (int j = 0; j < numFeatures; j++)
        {
            XtWX[j, j] += 1e-6;
        }

        // Solve using Gaussian elimination
        var shapValues = SolveLinearSystem(XtWX, XtWy);

        var result = new T[numFeatures];
        for (int j = 0; j < numFeatures; j++)
        {
            result[j] = NumOps.FromDouble(shapValues[j]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Solves a linear system Ax = b using Gaussian elimination with partial pivoting.
    /// </summary>
    private double[] SolveLinearSystem(double[,] A, double[] b)
    {
        int n = b.Length;
        var augmented = new double[n, n + 1];

        // Build augmented matrix
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
            // Find pivot
            int maxRow = k;
            for (int i = k + 1; i < n; i++)
            {
                if (Math.Abs(augmented[i, k]) > Math.Abs(augmented[maxRow, k]))
                {
                    maxRow = i;
                }
            }

            // Swap rows
            for (int j = k; j <= n; j++)
            {
                (augmented[k, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[k, j]);
            }

            // Eliminate column
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
        }

        return x;
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

    private T ComputeMean(Vector<T> values)
    {
        if (values.Length == 0)
            return NumOps.Zero;

        var sum = NumOps.Zero;
        for (int i = 0; i < values.Length; i++)
        {
            sum = NumOps.Add(sum, values[i]);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(values.Length));
    }

    private static TInput ConvertToModelInput<TInput>(Matrix<T> data)
    {
        if (typeof(TInput) == typeof(Matrix<T>))
            return (TInput)(object)data;
        if (typeof(TInput) == typeof(Tensor<T>))
            return (TInput)(object)Tensor<T>.FromRowMatrix(data);
        if (typeof(TInput) == typeof(Vector<T>) && data.Rows == 1)
            return (TInput)(object)data.GetRow(0);

        throw new NotSupportedException($"Cannot convert Matrix<T> to {typeof(TInput).Name}");
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
