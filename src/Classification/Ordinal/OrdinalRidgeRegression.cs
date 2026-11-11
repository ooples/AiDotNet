using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
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
/// <example>
/// <code>
/// // Create ordinal ridge regression with L2 regularization
///
/// // Prepare training data: 6 samples with 2 features, ordinal labels
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 0.5;
/// features[1, 0] = 1.5; features[1, 1] = 1.0;
/// features[2, 0] = 2.0; features[2, 1] = 1.5;
/// features[3, 0] = 2.5; features[3, 1] = 2.0;
/// features[4, 0] = 3.0; features[4, 1] = 2.5;
/// features[5, 0] = 3.5; features[5, 1] = 3.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1, 2, 2 });
///
/// // Train with closed-form ridge solution and immediate-threshold method
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new OrdinalRidgeRegression&lt;double&gt;())
///     .Build(features, labels);
///
/// // Predict ordinal class using learned thresholds
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 2.2; newSample[0, 1] = 1.8;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Ridge Regression: Biased Estimation for Nonorthogonal Problems", "https://doi.org/10.1080/00401706.1970.10488634")]
public partial class OrdinalRidgeRegression<T> : OrdinalClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>, IGradientComputable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The ridge coefficients followed by the ordinal thresholds. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }
    /// <summary>
    /// The learned coefficient vector (β).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These coefficients determine how each feature affects the
    /// predicted value. Positive coefficients push predictions toward higher classes.</para>
    /// </remarks>
    [AiDotNet.Attributes.FittedParameter]
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
    private double _alpha;

    /// <summary>
    /// Whether to fit an intercept term.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The intercept allows the prediction line to not pass through
    /// the origin. Almost always set to true unless your data is already centered.</para>
    /// </remarks>
    private bool _fitIntercept;

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

        var engine = Engine;
        var features = Tensor<T>.FromMatrix(x);
        if (x.Columns != p)
        {
            features = features.Slice(1, 0, p);
        }

        // Center data if fitting intercept
        var featureMeans = new Tensor<T>(new[] { p });
        T targetMean = NumOps.Zero;
        var targets = y.AsTensor();
        if (_fitIntercept && n > 0)
        {
            featureMeans = engine.ReduceMean(features, new[] { 0 }, false);
            targetMean = engine.TensorMean(targets);
            features = engine.TensorSubtract(features, featureMeans.Reshape(1, p));
            targets = engine.TensorSubtractScalar(targets, targetMean);
        }

        // Build X^T X + lambda*I and X^T y, then solve the ridge normal equations
        var xT = engine.TensorTranspose(features);
        var xtx = engine.TensorAdd(engine.TensorMatMul(xT, features),
            engine.TensorMultiplyScalar(engine.TensorEye<T>(p), NumOps.FromDouble(_alpha)));
        var xty = engine.TensorMatMul(xT, targets.Reshape(n, 1));
        _coefficients = p == 0 ? new Vector<T>(0) : SolveSymmetric(xtx, xty).Reshape(p).ToVector();

        // Compute intercept
        _bias = _fitIntercept
            ? NumOps.Subtract(targetMean, p == 0 ? NumOps.Zero : engine.DotProduct(_coefficients, featureMeans.ToVector()))
            : NumOps.Zero;

        // Learn thresholds using immediate-threshold method
        // Compute class means in prediction space
        var predictions = LinearPredictions(x, _coefficients, _bias);
        var membership = engine.TensorOneHot<T>(GetOrdinalClassIndices(y), K);
        var classSums = engine.TensorMatMul(engine.TensorTranspose(membership), predictions.Reshape(n, 1)).Reshape(K);
        var classCounts = engine.TensorClampMin(engine.ReduceSum(membership, new[] { 0 }, false), NumOps.One);
        var classMeans = engine.TensorDivide(classSums, classCounts);

        // Set thresholds at midpoints between class means
        _thresholds = K <= 1
            ? new Vector<T>(0)
            : engine.TensorMultiplyScalar(engine.TensorAdd(classMeans.Slice(0, 0, K - 1), classMeans.Slice(0, 1, K)),
                NumOps.FromDouble(0.5)).ToVector();
    }

    /// <summary>
    /// Solves a x = b with LU and partial pivoting, falling back to the pseudo-inverse for singular systems.
    /// </summary>
    private Tensor<T> SolveSymmetric(Tensor<T> a, Tensor<T> b)
    {
        var (solution, info) = Linalg.SolveEx(a, b);
        bool singular = info.Length > 0 && info.ToArray()[0] != 0;
        return singular ? Engine.TensorMatMul(Linalg.Pinv(a), b) : solution;
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

        int n = input.Rows;
        int k = NumClasses;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Class = 1 + the last threshold the prediction exceeds (0 when it exceeds none):
        // score 0.5 for class 0 and (j + 1) for class j + 1 when pred > threshold_j, so argmax picks it
        var engine = Engine;
        var pred = LinearPredictions(input, _coefficients, _bias).Reshape(n, 1);
        var baseline = new Tensor<T>(new[] { n, 1 });
        engine.TensorFill(baseline, NumOps.FromDouble(0.5));
        if (k <= 1)
        {
            return ClassIndicesToLabels(new Tensor<int>(new[] { n }));
        }

        var exceeds = engine.TensorGreaterThan(pred, _thresholds.AsTensor().Reshape(1, k - 1));
        var ranks = engine.TensorArange<T>(1, k).Reshape(1, k - 1);
        var scores = engine.TensorConcatenate(new[] { baseline, engine.TensorMultiply(exceeds, ranks) }, 1);
        return ClassIndicesToLabels(engine.TensorArgMax(scores, 1));
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
        if (n == 0 || K <= 1)
        {
            return new Matrix<T>(n, Math.Max(0, K - 1));
        }

        // Convert to cumulative probabilities using sigmoid(threshold_k - pred)
        var engine = Engine;
        var pred = LinearPredictions(features, _coefficients, _bias).Reshape(n, 1);
        var z = engine.TensorSubtract(_thresholds.AsTensor().Reshape(1, K - 1), pred);
        return engine.TensorSigmoid(z).ToMatrix();
    }

    /// <summary>
    /// Gets the model parameters (coefficients, bias, and thresholds).
    /// </summary>
    /// <returns>Vector containing all model parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This packs all learned parameters into a single vector:
    /// first the coefficients, then the bias, then the thresholds.</para>
    /// </remarks>
    private Vector<T> PackParameters()
    {
        if (_coefficients is null || _thresholds is null)
        {
            return new Vector<T>(0);
        }

        return Vector<T>.Concatenate(Vector<T>.Concatenate(_coefficients, FilledVector(1, _bias)), _thresholds);
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Vector containing all model parameters.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This unpacks parameters from a single vector and sets
    /// the internal state. The vector format is: [coefficients..., bias, thresholds...].</para>
    /// </remarks>
    private void UnpackParameters(Vector<T> parameters)
    {
        if (parameters.Length == 0)
        {
            return;
        }

        // If we have NumFeatures and NumClasses set, use them
        int numCoefs;
        int numThresholds;
        if (NumFeatures > 0 && NumClasses > 0 && parameters.Length == NumFeatures + 1 + (NumClasses - 1))
        {
            numCoefs = NumFeatures;
            numThresholds = NumClasses - 1;
        }
        else
        {
            // Fallback: infer structure from parameter length
            // Assume: coefficients = (len - 1) / 2, thresholds = (len - 1) / 2
            numThresholds = (parameters.Length - 1) / 2;
            numCoefs = parameters.Length - 1 - numThresholds;
            NumFeatures = numCoefs;
        }

        _coefficients = parameters.GetSubVector(0, numCoefs);
        _bias = parameters.GetSubVector(numCoefs, 1).ToArray()[0];
        _thresholds = parameters.GetSubVector(numCoefs + 1, numThresholds);
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
            model.ClassLabels = ClassLabels.Clone();
        }
        model.SetParameters(parameters);
        return model;
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
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var engine = Engine;
        int n = input.Rows;
        int p = NumFeatures;
        int K = NumClasses;
        var coefficients = FitToLength(_coefficients, p);
        T twoOverN = NumOps.FromDouble(2.0 / n);

        // Residuals of the linear predictions
        var residuals = engine.TensorSubtract(LinearPredictions(input, coefficients, _bias), target.AsTensor());

        // Gradients for coefficients: (2/n) * X^T * residuals + 2*alpha*beta
        var x = Tensor<T>.FromMatrix(input);
        if (input.Columns != p)
        {
            x = x.Slice(1, 0, p);
        }

        var xtr = n == 0
            ? new Tensor<T>(new[] { p })
            : engine.TensorMatMul(engine.TensorTranspose(x), residuals.Reshape(n, 1)).Reshape(p);
        var gradCoef = engine.TensorAdd(engine.TensorMultiplyScalar(xtr, twoOverN),
            engine.TensorMultiplyScalar(coefficients.AsTensor(), NumOps.FromDouble(2.0 * _alpha)));

        // Gradient for bias: (2/n) * sum(residuals)
        T gradBias = NumOps.Multiply(twoOverN, n == 0 ? NumOps.Zero : engine.TensorSum(residuals));

        // Pack into vector (thresholds have zero gradient as they're derived from data)
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(gradCoef.ToVector(), FilledVector(1, gradBias)),
            new Vector<T>(Math.Max(0, K - 1)));
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
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // Update coefficients and bias: w = w - lr * g
        var engine = Engine;
        int numCoefficients = _coefficients.Length;
        _coefficients = engine.Subtract(_coefficients, engine.Multiply(gradients.GetSubVector(0, numCoefficients), learningRate));
        T biasGradient = gradients.GetSubVector(numCoefficients, 1).ToArray()[0];
        _bias = NumOps.Subtract(_bias, NumOps.Multiply(learningRate, biasGradient));

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

        return NormalizedCoefficientImportance(_coefficients);
    }
}
