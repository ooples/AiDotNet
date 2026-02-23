using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Classification.Ordinal;

/// <summary>
/// Ordinal Ridge Regression using the Immediate-Threshold approach with L2 regularization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is a variant of ordinal classification that treats the problem
/// as a regression task with specially-designed thresholds. It uses ridge (L2) regularization
/// to prevent overfitting, which adds a penalty for large coefficient values.</para>
///
/// <para><b>How it works:</b> Instead of modeling cumulative probabilities like ordinal logistic
/// regression, this approach:
/// <list type="number">
/// <item>Treats the ordinal labels as numeric targets</item>
/// <item>Learns a linear function f(X) = β·X</item>
/// <item>Uses thresholds to convert predictions back to ordinal classes</item>
/// </list>
/// </para>
///
/// <para><b>Immediate-Threshold method:</b> After training the regression model, thresholds are
/// placed at the midpoints between consecutive class means in the training data. This gives
/// natural boundaries between classes.</para>
///
/// <para><b>Ridge regularization:</b> Adds λ·||β||² to the loss function, which:
/// <list type="bullet">
/// <item>Prevents coefficients from becoming too large</item>
/// <item>Reduces overfitting on noisy data</item>
/// <item>Provides a closed-form solution (no iterative optimization needed)</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When you want a fast, closed-form solution</item>
/// <item>When the ordinal levels are roughly equally spaced</item>
/// <item>When you need regularization to handle multicollinearity</item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Frank, E. &amp; Hall, M. (2001). "A Simple Approach to Ordinal Classification"</item>
/// <item>Chu, W. &amp; Keerthi, S.S. (2007). "Support Vector Ordinal Regression"</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OrdinalRidgeRegression<T> : OrdinalClassifierBase<T>
{
    /// <summary>
    /// The learned coefficient vector (β).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These coefficients determine how each feature affects the
    /// predicted value. Positive coefficients push predictions toward higher classes.</para>
    /// </remarks>
    private Vector<T>? _coefficients;

    /// <summary>
    /// The bias (intercept) term.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The bias is a constant added to all predictions. It shifts
    /// the entire prediction line up or down without changing its slope.</para>
    /// </remarks>
    private T _bias;

    /// <summary>
    /// Ridge regularization parameter (λ).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values make the model simpler by shrinking coefficients
    /// toward zero. This prevents overfitting but may underfit if too high.</para>
    /// </remarks>
    private readonly double _alpha;

    /// <summary>
    /// Whether to fit an intercept term.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The intercept allows the prediction line to not pass through
    /// the origin. Almost always set to true unless your data is already centered.</para>
    /// </remarks>
    private readonly bool _fitIntercept;

    /// <summary>
    /// Initializes a new instance of OrdinalRidgeRegression.
    /// </summary>
    /// <param name="alpha">Ridge regularization strength. Higher = more regularization. Default is 1.0.</param>
    /// <param name="fitIntercept">Whether to fit an intercept term. Default is true.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// <list type="bullet">
    /// <item><b>alpha:</b> Controls the amount of regularization. Start with 1.0 and increase if
    /// overfitting, decrease if underfitting. Common values range from 0.01 to 100.</item>
    /// <item><b>fitIntercept:</b> Leave as true unless you've manually centered your data.</item>
    /// </list>
    /// </para>
    /// </remarks>
    public OrdinalRidgeRegression(double alpha = 1.0, bool fitIntercept = true)
        : base()
    {
        _alpha = alpha;
        _fitIntercept = fitIntercept;
        _bias = NumOps.Zero;
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.OrdinalRidgeRegression.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This identifier helps the system know what type of model
    /// this is, which is useful for serialization and model management.</para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.OrdinalRidgeRegression;

    /// <summary>
    /// Trains the ordinal ridge regression model.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Ordinal class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training involves two steps:
    /// <list type="number">
    /// <item><b>Ridge regression:</b> Solve (X^T X + λI)^(-1) X^T y for coefficients</item>
    /// <item><b>Threshold learning:</b> Find midpoints between class means for decision boundaries</item>
    /// </list>
    ///
    /// The closed-form solution means this is very fast compared to iterative methods.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples must match number of labels.");
        }

        NumFeatures = x.Columns;
        ExtractOrderedClasses(y);

        int n = x.Rows;
        int p = NumFeatures;
        int K = NumClasses;

        // Center data if fitting intercept
        var featureMeans = new double[p];
        double targetMean = 0;

        if (_fitIntercept)
        {
            // Compute feature means
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += NumOps.ToDouble(x[i, j]);
                }
                featureMeans[j] = sum / n;
            }

            // Compute target mean
            for (int i = 0; i < n; i++)
            {
                targetMean += NumOps.ToDouble(y[i]);
            }
            targetMean /= n;
        }

        // Build X^T X + λI matrix
        var XtX = new double[p, p];
        var XtY = new double[p];

        for (int i = 0; i < n; i++)
        {
            // Center features if fitting intercept
            var xi = new double[p];
            for (int j = 0; j < p; j++)
            {
                xi[j] = NumOps.ToDouble(x[i, j]) - (_fitIntercept ? featureMeans[j] : 0);
            }

            double yi = NumOps.ToDouble(y[i]) - (_fitIntercept ? targetMean : 0);

            // Accumulate X^T X
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    XtX[j, k] += xi[j] * xi[k];
                }
                XtY[j] += xi[j] * yi;
            }
        }

        // Add ridge regularization to diagonal
        for (int j = 0; j < p; j++)
        {
            XtX[j, j] += _alpha;
        }

        // Solve the system using Cholesky decomposition (or simple inversion for small p)
        _coefficients = new Vector<T>(p);

        // Simple Gaussian elimination for solving the linear system
        var augmented = new double[p, p + 1];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                augmented[i, j] = XtX[i, j];
            }
            augmented[i, p] = XtY[i];
        }

        // Forward elimination
        for (int i = 0; i < p; i++)
        {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < p; k++)
            {
                if (Math.Abs(augmented[k, i]) > Math.Abs(augmented[maxRow, i]))
                {
                    maxRow = k;
                }
            }

            // Swap rows
            for (int k = i; k <= p; k++)
            {
                (augmented[i, k], augmented[maxRow, k]) = (augmented[maxRow, k], augmented[i, k]);
            }

            // Eliminate column
            if (Math.Abs(augmented[i, i]) > 1e-10)
            {
                for (int k = i + 1; k < p; k++)
                {
                    double factor = augmented[k, i] / augmented[i, i];
                    for (int j = i; j <= p; j++)
                    {
                        augmented[k, j] -= factor * augmented[i, j];
                    }
                }
            }
        }

        // Back substitution
        var solution = new double[p];
        for (int i = p - 1; i >= 0; i--)
        {
            double sum = augmented[i, p];
            for (int j = i + 1; j < p; j++)
            {
                sum -= augmented[i, j] * solution[j];
            }
            solution[i] = Math.Abs(augmented[i, i]) > 1e-10 ? sum / augmented[i, i] : 0;
        }

        for (int i = 0; i < p; i++)
        {
            _coefficients[i] = NumOps.FromDouble(solution[i]);
        }

        // Compute intercept
        if (_fitIntercept)
        {
            double interceptVal = targetMean;
            for (int j = 0; j < p; j++)
            {
                interceptVal -= solution[j] * featureMeans[j];
            }
            _bias = NumOps.FromDouble(interceptVal);
        }
        else
        {
            _bias = NumOps.Zero;
        }

        // Learn thresholds using immediate-threshold method
        // Compute class means in prediction space
        var classSums = new double[K];
        var classCounts = new int[K];

        for (int i = 0; i < n; i++)
        {
            // Compute prediction for this sample
            double pred = NumOps.ToDouble(_bias);
            for (int j = 0; j < p; j++)
            {
                pred += NumOps.ToDouble(_coefficients[j]) * NumOps.ToDouble(x[i, j]);
            }

            int classIdx = GetClassIndex(y[i]);
            classSums[classIdx] += pred;
            classCounts[classIdx]++;
        }

        var classMeans = new double[K];
        for (int k = 0; k < K; k++)
        {
            classMeans[k] = classCounts[k] > 0 ? classSums[k] / classCounts[k] : 0;
        }

        // Set thresholds at midpoints between class means
        _thresholds = new Vector<T>(K - 1);
        for (int k = 0; k < K - 1; k++)
        {
            double threshold = (classMeans[k] + classMeans[k + 1]) / 2.0;
            _thresholds[k] = NumOps.FromDouble(threshold);
        }
    }

    /// <summary>
    /// Predicts ordinal class labels.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prediction works by:
    /// <list type="number">
    /// <item>Computing the linear prediction: f(X) = β·X + bias</item>
    /// <item>Comparing to thresholds to find which class interval the prediction falls into</item>
    /// <item>Returning the corresponding class label</item>
    /// </list>
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_coefficients is null || _thresholds is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Compute linear prediction
            double pred = NumOps.ToDouble(_bias);
            for (int j = 0; j < NumFeatures; j++)
            {
                pred += NumOps.ToDouble(_coefficients[j]) * NumOps.ToDouble(input[i, j]);
            }

            // Find class based on thresholds
            int classIdx = 0;
            for (int k = 0; k < NumClasses - 1; k++)
            {
                if (pred > NumOps.ToDouble(_thresholds[k]))
                {
                    classIdx = k + 1;
                }
            }

            predictions[i] = ClassLabels[classIdx];
        }

        return predictions;
    }

    /// <summary>
    /// Predicts cumulative probabilities P(Y ≤ k).
    /// </summary>
    /// <param name="features">Feature matrix.</param>
    /// <returns>Cumulative probability matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Since this is a regression-based approach, we convert the
    /// continuous predictions to probabilities using a sigmoid function. The probability
    /// that Y ≤ k is modeled as σ(threshold_k - prediction), where σ is the sigmoid function.</para>
    /// </remarks>
    public override Matrix<T> PredictCumulativeProbabilities(Matrix<T> features)
    {
        if (_coefficients is null || _thresholds is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        int n = features.Rows;
        int K = NumClasses;
        var cumProbs = new Matrix<T>(n, K - 1);

        for (int i = 0; i < n; i++)
        {
            // Compute linear prediction
            double pred = NumOps.ToDouble(_bias);
            for (int j = 0; j < NumFeatures; j++)
            {
                pred += NumOps.ToDouble(_coefficients[j]) * NumOps.ToDouble(features[i, j]);
            }

            // Convert to cumulative probabilities using sigmoid
            for (int k = 0; k < K - 1; k++)
            {
                double z = NumOps.ToDouble(_thresholds[k]) - pred;
                // Use a scaled sigmoid for smoother probabilities
                double prob = 1.0 / (1.0 + Math.Exp(-z));
                cumProbs[i, k] = NumOps.FromDouble(prob);
            }
        }

        return cumProbs;
    }

    /// <summary>
    /// Gets the model parameters (coefficients, bias, and thresholds).
    /// </summary>
    /// <returns>Vector containing all model parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This packs all learned parameters into a single vector:
    /// first the coefficients, then the bias, then the thresholds.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_coefficients is null || _thresholds is null)
        {
            return new Vector<T>(0);
        }

        int totalLen = _coefficients.Length + 1 + _thresholds.Length;
        var parameters = new Vector<T>(totalLen);

        for (int i = 0; i < _coefficients.Length; i++)
        {
            parameters[i] = _coefficients[i];
        }

        parameters[_coefficients.Length] = _bias;

        for (int i = 0; i < _thresholds.Length; i++)
        {
            parameters[_coefficients.Length + 1 + i] = _thresholds[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Vector containing all model parameters.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This unpacks parameters from a single vector and sets
    /// the internal state. The vector format is: [coefficients..., bias, thresholds...].</para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length == 0)
        {
            return;
        }

        // If we have NumFeatures and NumClasses set, use them
        if (NumFeatures > 0 && NumClasses > 0)
        {
            int numCoef = NumFeatures;
            int numThresh = NumClasses - 1;
            int expectedLen = numCoef + 1 + numThresh;

            if (parameters.Length == expectedLen)
            {
                _coefficients = new Vector<T>(numCoef);
                for (int i = 0; i < numCoef; i++)
                {
                    _coefficients[i] = parameters[i];
                }

                _bias = parameters[numCoef];

                _thresholds = new Vector<T>(numThresh);
                for (int i = 0; i < numThresh; i++)
                {
                    _thresholds[i] = parameters[numCoef + 1 + i];
                }

                return;
            }
        }

        // Fallback: infer structure from parameter length
        // Assume: coefficients = (len - 1) / 2, thresholds = (len - 1) / 2
        int numThresholds = (parameters.Length - 1) / 2;
        int numCoefs = parameters.Length - 1 - numThresholds;

        _coefficients = new Vector<T>(numCoefs);
        for (int i = 0; i < numCoefs; i++)
        {
            _coefficients[i] = parameters[i];
        }

        _bias = parameters[numCoefs];

        _thresholds = new Vector<T>(numThresholds);
        for (int i = 0; i < numThresholds; i++)
        {
            _thresholds[i] = parameters[numCoefs + 1 + i];
        }

        NumFeatures = numCoefs;
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">Parameters for the new instance.</param>
    /// <returns>New model instance with the specified parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a copy of this model with different parameter values.
    /// Useful for optimization or ensemble methods.</para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var model = new OrdinalRidgeRegression<T>(_alpha, _fitIntercept);
        model.NumFeatures = NumFeatures;
        model.NumClasses = NumClasses;
        if (ClassLabels is not null)
        {
            model.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                model.ClassLabels[i] = ClassLabels[i];
            }
        }
        model.SetParameters(parameters);
        return model;
    }

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    /// <returns>New instance with same hyperparameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates an untrained copy with the same settings.</para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OrdinalRidgeRegression<T>(_alpha, _fitIntercept);
    }

    /// <summary>
    /// Computes gradients for the model parameters.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="target">Target labels.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <returns>Gradient vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes how much changing each parameter would affect the
    /// prediction error. For ridge regression, the gradient of the mean squared error with
    /// regularization is: ∂L/∂β = (2/n) * X^T(Xβ - y) + 2λβ</para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        int n = input.Rows;
        int p = NumFeatures;

        // Compute predictions
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
        {
            double pred = NumOps.ToDouble(_bias);
            for (int j = 0; j < p; j++)
            {
                pred += NumOps.ToDouble(_coefficients[j]) * NumOps.ToDouble(input[i, j]);
            }
            residuals[i] = pred - NumOps.ToDouble(target[i]);
        }

        // Compute gradients for coefficients: (2/n) * X^T * residuals + 2*alpha*beta
        var gradCoef = new double[p];
        for (int j = 0; j < p; j++)
        {
            double grad = 0;
            for (int i = 0; i < n; i++)
            {
                grad += NumOps.ToDouble(input[i, j]) * residuals[i];
            }
            gradCoef[j] = (2.0 / n) * grad + 2.0 * _alpha * NumOps.ToDouble(_coefficients[j]);
        }

        // Gradient for bias: (2/n) * sum(residuals)
        double gradBias = 0;
        for (int i = 0; i < n; i++)
        {
            gradBias += residuals[i];
        }
        gradBias = (2.0 / n) * gradBias;

        // Pack into vector (thresholds have zero gradient as they're derived from data)
        int K = NumClasses;
        var gradients = new Vector<T>(p + 1 + (K - 1));

        for (int j = 0; j < p; j++)
        {
            gradients[j] = NumOps.FromDouble(gradCoef[j]);
        }
        gradients[p] = NumOps.FromDouble(gradBias);

        // Thresholds derived from data, so gradient is zero
        for (int k = 0; k < K - 1; k++)
        {
            gradients[p + 1 + k] = NumOps.Zero;
        }

        return gradients;
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <param name="learningRate">Learning rate for the update.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Updates parameters by moving in the opposite direction of
    /// the gradients. Note that thresholds are not updated via gradient descent in this
    /// implementation - they're derived from the data.</para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        double lr = NumOps.ToDouble(learningRate);

        // Update coefficients
        for (int j = 0; j < _coefficients.Length; j++)
        {
            double newVal = NumOps.ToDouble(_coefficients[j]) - lr * NumOps.ToDouble(gradients[j]);
            _coefficients[j] = NumOps.FromDouble(newVal);
        }

        // Update bias
        double newBias = NumOps.ToDouble(_bias) - lr * NumOps.ToDouble(gradients[_coefficients.Length]);
        _bias = NumOps.FromDouble(newBias);

        // Note: Thresholds are not updated via gradient descent
    }

    /// <summary>
    /// Gets feature importance based on coefficient magnitude.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features with larger absolute coefficient values have more
    /// impact on predictions. Values are normalized to sum to 1.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        if (_coefficients is null)
        {
            return base.GetFeatureImportance();
        }

        var importance = new Dictionary<string, T>();

        // Sum of absolute coefficients for normalization
        double total = 0;
        for (int i = 0; i < _coefficients.Length; i++)
        {
            total += Math.Abs(NumOps.ToDouble(_coefficients[i]));
        }

        if (total == 0) total = 1;

        for (int i = 0; i < _coefficients.Length; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            double imp = Math.Abs(NumOps.ToDouble(_coefficients[i])) / total;
            importance[name] = NumOps.FromDouble(imp);
        }

        return importance;
    }
}
