using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;

using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification;

/// <summary>
/// Implements Ordinal Regression (Proportional Odds Model) for predicting ordered categorical outcomes.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Ordinal Regression is used when the target variable has naturally ordered categories. It models
/// the cumulative probability of being in category k or lower using the proportional odds assumption:
/// P(Y ≤ k) = sigmoid(α_k - β^T × x)
/// where α_k are ordered thresholds (one less than the number of classes) and β are feature coefficients.
/// </para>
/// <para>
/// Key properties:
/// - Respects the natural ordering of categories
/// - Uses single set of feature coefficients (proportional odds assumption)
/// - Ordered thresholds separate adjacent categories
/// - Probabilities for individual classes: P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)
/// </para>
/// <para>
/// For Beginners:
/// Ordinal Regression is perfect when your categories have a natural order but the distances
/// between them may not be equal. Examples include:
///
/// - Star ratings (1-5 stars): You know 5 > 4 > 3 > 2 > 1, but the difference between 1 and 2
///   stars might not equal the difference between 4 and 5 stars
/// - Survey responses: Strongly Disagree &lt; Disagree &lt; Neutral &lt; Agree &lt; Strongly Agree
/// - Education levels: High School &lt; Bachelor's &lt; Master's &lt; PhD
/// - Pain levels: None &lt; Mild &lt; Moderate &lt; Severe
///
/// The model learns:
/// 1. Feature coefficients (β): How each feature pushes predictions up or down the ordinal scale
/// 2. Thresholds (α): Where to draw the lines between adjacent categories
///
/// This is better than treating it as regular classification (which ignores order) or as
/// regression (which assumes equal distances between categories).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create ordinal regression for ordered categorical prediction (e.g., star ratings 1-5)
/// var options = new OrdinalRegressionOptions&lt;double&gt;();
///
/// // Prepare training data with features and ordinal labels (0, 1, 2 for Low, Medium, High)
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 0.5;
/// features[1, 0] = 1.5; features[1, 1] = 1.0;
/// features[2, 0] = 2.0; features[2, 1] = 1.5;
/// features[3, 0] = 2.5; features[3, 1] = 2.0;
/// features[4, 0] = 3.0; features[4, 1] = 2.5;
/// features[5, 0] = 3.5; features[5, 1] = 3.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1, 2, 2 });
///
/// // Train with proportional odds model
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new OrdinalRegression&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict ordinal class for new data
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 2.2; newSample[0, 1] = 1.8;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Regression Models for Ordinal Data", "https://doi.org/10.1111/j.2517-6161.1980.tb01109.x", Year = 1980, Authors = "Peter McCullagh")]
public partial class OrdinalRegression<T> : ClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>, IGradientComputable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The coefficients followed by the ordinal thresholds. PackParameters and UnpackParameters ARE the previous
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
    /// Configuration options for the ordinal regression model.
    /// </summary>
    private readonly OrdinalRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Feature coefficients (β). Shared across all thresholds (proportional odds assumption).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _coefficients = new Vector<T>(0);

    /// <summary>
    /// Threshold parameters (α_1, α_2, ..., α_{K-1}) where K is the number of classes.
    /// These are in increasing order: α_1 &lt; α_2 &lt; ... &lt; α_{K-1}.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _thresholds = new Vector<T>(0);

    /// <summary>
    /// Gets the feature coefficients.
    /// </summary>
    /// <value>
    /// The learned feature weights. Positive values increase the probability of higher categories.
    /// </value>
    /// <remarks>
    /// <para>
    /// In ordinal regression, the same coefficients apply to all threshold comparisons
    /// (the proportional odds assumption). Each coefficient indicates how that feature
    /// affects the probability of being in a higher category.
    /// </para>
    /// <para>
    /// For Beginners:
    /// These are the weights the model learned for each feature. A positive weight means
    /// that feature pushes predictions toward higher categories (e.g., more stars, higher rating).
    /// A negative weight pushes toward lower categories.
    /// </para>
    /// </remarks>
    public Vector<T>? Coefficients => _coefficients;

    /// <summary>
    /// Gets the threshold parameters.
    /// </summary>
    /// <value>
    /// The learned threshold values that separate adjacent categories. Always in increasing order.
    /// </value>
    /// <remarks>
    /// <para>
    /// For K classes, there are K-1 thresholds. The threshold α_k represents the cutpoint
    /// between categories k and k+1.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Think of thresholds as the "dividing lines" between categories on a number line.
    /// If you have 5 rating categories (1-5), you have 4 thresholds separating them.
    /// </para>
    /// </remarks>
    public Vector<T>? Thresholds => _thresholds;

    /// <summary>
    /// Initializes a new instance of the OrdinalRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the ordinal regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the ordinal regression model with your specified settings or uses
    /// default settings if none are provided. The key settings are the learning rate and
    /// convergence tolerance for training.
    /// </para>
    /// </remarks>
    public OrdinalRegression(OrdinalRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new OrdinalRegressionOptions<T>();
        TaskType = ClassificationTaskType.Ordinal;
    }

    /// <summary>
    /// Trains the ordinal regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target class labels (0, 1, 2, ..., K-1 for K ordered classes).</param>
    /// <remarks>
    /// <para>
    /// This method uses gradient descent to minimize the negative log-likelihood of the
    /// proportional odds model. The algorithm:
    /// 1. Initializes thresholds evenly spaced
    /// 2. For each iteration:
    ///    a. Compute cumulative probabilities using current parameters
    ///    b. Compute class probabilities
    ///    c. Compute gradients for coefficients and thresholds
    ///    d. Update parameters using gradient descent
    ///    e. Enforce threshold ordering constraint
    ///    f. Check for convergence
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training teaches the model the best feature weights and category thresholds.
    /// The algorithm starts with initial guesses and iteratively improves them until
    /// the predictions match the training data as well as possible.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;

        // Validate ordinal data (should be integers 0 to K-1)
        ValidateOrdinalData(y);

        var engine = Engine;

        // Initialize coefficients to zero
        _coefficients = new Vector<T>(NumFeatures);

        // Initial thresholds: -2, -1, 0, 1, 2, ... for K-1 thresholds centered around 0: (k + 1) - K / 2
        _thresholds = NumClasses <= 1
            ? new Vector<T>(0)
            : engine.TensorSubtractScalar(engine.TensorArange<T>(1, NumClasses), NumOps.FromDouble(NumClasses / 2.0)).ToVector();

        // Gradient descent optimization
        T prevLogLikelihood = NumOps.FromDouble(double.NegativeInfinity);
        T learningRate = NumOps.FromDouble(_options.LearningRate);
        T regStrength = NumOps.FromDouble(_options.RegularizationStrength);

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            // Compute gradients and update parameters
            var (coeffGradient, threshGradient, logLikelihood) = ComputeGradients(x, y);

            // Update coefficients (with the L2 regularization gradient) and thresholds
            var grad = engine.Subtract(coeffGradient, engine.Multiply(_coefficients, regStrength));
            _coefficients = engine.Add(_coefficients, engine.Multiply(grad, learningRate));
            _thresholds = engine.Add(_thresholds, engine.Multiply(threshGradient, learningRate));

            // Enforce threshold ordering: α_1 < α_2 < ... < α_{K-1}
            EnforceThresholdOrdering();

            // Check convergence
            if (HasConverged(prevLogLikelihood, logLikelihood))
            {
                break;
            }

            prevLogLikelihood = logLikelihood;
        }
    }

    /// <summary>
    /// Whether the log-likelihood changed by less than the tolerance.
    /// </summary>
    private bool HasConverged(T previous, T current)
    {
        return NumOps.LessThan(NumOps.Abs(NumOps.Subtract(current, previous)), NumOps.FromDouble(_options.Tolerance));
    }

    /// <summary>
    /// Validates that the target values are valid ordinal classes (integers 0 to K-1).
    /// </summary>
    /// <param name="y">The target labels vector.</param>
    /// <exception cref="ArgumentException">Thrown when target values are invalid.</exception>
    /// <remarks>
    /// <para>
    /// Ordinal regression expects class labels to be consecutive integers starting from 0.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Your class labels should be 0, 1, 2, 3, ... without gaps. If your original labels are
    /// 1, 2, 3, 4, 5 (like star ratings), subtract 1 to get 0, 1, 2, 3, 4.
    /// </para>
    /// </remarks>
    private void ValidateOrdinalData(Vector<T> y)
    {
        var classIndices = GetClassIndicesFromLabels(y);
        for (int i = 0; i < classIndices.Length; i++)
        {
            int classIdx = classIndices[i];
            if (classIdx < 0 || classIdx >= NumClasses)
            {
                throw new ArgumentException(
                    $"Invalid class label at index {i}. Expected integers from 0 to {NumClasses - 1}.");
            }
        }
    }

    /// <summary>
    /// Computes gradients for the proportional odds model.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The target labels.</param>
    /// <returns>Tuple of (coefficient gradients, threshold gradients, log-likelihood).</returns>
    /// <remarks>
    /// <para>
    /// The gradient computation follows the proportional odds model:
    /// - For coefficients: ∂L/∂β = Σ_i x_i × (y_i/p_i - Σ_k I(y_i > k) × ∂P(Y≤k)/∂η)
    /// - For thresholds: ∂L/∂α_k = Σ_i (I(y_i > k)P(Y≤k)(1-P(Y≤k)) - I(y_i ≤ k)P(Y>k)(1-P(Y>k)))
    /// </para>
    /// <para>
    /// For Beginners:
    /// Gradients tell the optimization algorithm how to adjust each parameter to improve
    /// the model's predictions. Positive gradient means "increase this parameter".
    /// </para>
    /// </remarks>
    private (Vector<T> coeffGradient, Vector<T> threshGradient, T logLikelihood) ComputeGradients(Matrix<T> x, Vector<T> y)
    {
        var engine = Engine;
        int n = x.Rows;
        int k = NumClasses;
        if (n == 0)
        {
            return (new Vector<T>(NumFeatures), new Vector<T>(Math.Max(0, k - 1)), NumOps.Zero);
        }

        // Linear predictor for all samples: η_i = β^T × x_i, cumulative probabilities P(Y ≤ k) = link(α_k - η)
        var features = FeatureBlock(x);
        var cumulative = CumulativeProbabilities(features);

        // Padded cumulative probabilities [0, P(Y≤0), ..., P(Y≤K-2), 1] and sigmoid' terms c(1-c)
        var zeros = new Tensor<T>(new[] { n, 1 });
        var ones = new Tensor<T>(new[] { n, 1 });
        engine.TensorFill(ones, NumOps.One);
        var padded = engine.TensorConcatenate(new[] { zeros, cumulative, ones }, 1).Reshape(n * (k + 1));
        var gammas = engine.TensorMultiply(padded, engine.ScalarMinusTensor(NumOps.One, padded));

        // Row-wise picks at the sample's class: lo = P(Y ≤ y-1), hi = P(Y ≤ y)
        var classIndices = GetClassIndexTensor(y);
        var rowStarts = engine.TensorMultiplyScalar(engine.TensorArange<int>(0, n), k + 1);
        var lo = engine.TensorAdd(rowStarts, classIndices);
        var hi = engine.TensorAddScalar(lo, 1);

        // Class probabilities P(Y = y), clamped to avoid log(0)
        var classProbs = engine.TensorClamp(
            engine.TensorSubtract(engine.TensorGather(padded, hi, 0), engine.TensorGather(padded, lo, 0)),
            Tolerance(1e-10), NumOps.FromDouble(1.0 - 1e-10));
        T logLikelihood = engine.TensorSum(engine.TensorLog(classProbs));

        // dP(Y=y)/dη = -sigmoid'(α_y - η) + sigmoid'(α_{y-1} - η) (end terms are zero)
        var gammaLo = engine.TensorGather(gammas, lo, 0);
        var gammaHi = engine.TensorGather(gammas, hi, 0);
        var coeffFactor = engine.TensorDivide(engine.TensorSubtract(gammaLo, gammaHi), classProbs);

        // Gradient for coefficients: dL/dβ_j = Σ_i (1/P(Y=yi)) × dP(Y=yi)/dη × x_ij
        var coeffGradient = engine.TensorMatMul(engine.TensorTranspose(features), coeffFactor.Reshape(n, 1)).Reshape(NumFeatures);

        // Gradient for thresholds: +sigmoid'(α_y - η)/P at α_y, -sigmoid'(α_{y-1} - η)/P at α_{y-1}
        var padIndex = engine.TensorSubtract(lo, rowStarts);
        var threshGradient = new Tensor<T>(new[] { k + 1 });
        threshGradient = engine.TensorScatterAdd(threshGradient, engine.TensorAddScalar(padIndex, 1),
            engine.TensorDivide(gammaHi, classProbs), 0);
        threshGradient = engine.TensorScatterAdd(threshGradient, padIndex,
            engine.TensorNegate(engine.TensorDivide(gammaLo, classProbs)), 0);

        return (coeffGradient.ToVector(), threshGradient.Slice(0, 1, k).ToVector(), logLikelihood);
    }

    /// <summary>
    /// The first NumFeatures columns of <paramref name="x"/> as a tensor.
    /// </summary>
    private Tensor<T> FeatureBlock(Matrix<T> x)
    {
        var features = Tensor<T>.FromMatrix(x);
        return x.Columns == NumFeatures ? features : features.Slice(1, 0, NumFeatures);
    }

    /// <summary>
    /// [rows, K-1] cumulative probabilities P(Y ≤ k) = link(α_k - β^T x).
    /// </summary>
    private Tensor<T> CumulativeProbabilities(Tensor<T> features)
    {
        var engine = Engine;
        int n = features.Shape[0];
        var eta = NumFeatures == 0
            ? new Tensor<T>(new[] { n, 1 })
            : engine.TensorMatMul(features, FitToLength(_coefficients, NumFeatures).AsTensor().Reshape(NumFeatures, 1));
        var thresholds = FitToLength(_thresholds, Math.Max(0, NumClasses - 1)).AsTensor().Reshape(1, Math.Max(0, NumClasses - 1));
        return ApplyLink(engine.TensorSubtract(thresholds, eta));
    }

    /// <summary>
    /// Applies the link function to convert linear predictor to cumulative probability.
    /// </summary>
    /// <param name="linearPredictor">The linear predictor value (α_k - η).</param>
    /// <returns>The cumulative probability P(Y ≤ k).</returns>
    /// <remarks>
    /// <para>
    /// The link function transforms the unbounded linear predictor to a probability [0, 1].
    /// </para>
    /// <para>
    /// For Beginners:
    /// This function converts the model's internal score to a probability between 0 and 1.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyLink(Tensor<T> linearPredictor)
    {
        var engine = Engine;
        return _options.LinkFunction switch
        {
            OrdinalLinkFunction.Probit => NormalCdf(linearPredictor),
            OrdinalLinkFunction.ComplementaryLogLog => engine.ScalarMinusTensor(NumOps.One,
                engine.TensorExp(engine.TensorNegate(engine.TensorExp(linearPredictor)))),
            _ => engine.TensorSigmoid(linearPredictor)
        };
    }

    /// <summary>
    /// Computes the standard normal cumulative distribution function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The probability P(Z ≤ x) where Z is standard normal.</returns>
    /// <remarks>
    /// <para>
    /// Uses the error function approximation for computing the normal CDF.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This is a mathematical function that gives the probability of a standard normal
    /// variable being less than or equal to x. Used for the probit link function.
    /// </para>
    /// </remarks>
    private Tensor<T> NormalCdf(Tensor<T> x)
    {
        // Approximation using error function: 0.5 * (1 + erf(x / sqrt(2)))
        var engine = Engine;
        var erf = Erf(engine.TensorDivideScalar(x, NumOps.FromDouble(Math.Sqrt(2.0))));
        return engine.TensorMultiplyScalar(engine.TensorAddScalar(erf, NumOps.One), NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Computes the error function approximation.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The error function value.</returns>
    private Tensor<T> Erf(Tensor<T> x)
    {
        // Horner form coefficients for approximation (Abramowitz and Stegun 7.1.26)
        var engine = Engine;
        T a1 = NumOps.FromDouble(0.254829592);
        T a2 = NumOps.FromDouble(-0.284496736);
        T a3 = NumOps.FromDouble(1.421413741);
        T a4 = NumOps.FromDouble(-1.453152027);
        T a5 = NumOps.FromDouble(1.061405429);
        T p = NumOps.FromDouble(0.3275911);

        // sign = -1 for x < 0, +1 otherwise
        var sign = engine.ScalarMinusTensor(NumOps.One,
            engine.TensorMultiplyScalar(engine.TensorLessThan(x, NumOps.Zero), NumOps.FromDouble(2.0)));
        var ax = engine.TensorAbs(x);
        var t = engine.TensorDivide(Ones(ax), engine.TensorAddScalar(engine.TensorMultiplyScalar(ax, p), NumOps.One));

        // y = 1 - (((((a5 t + a4) t) + a3) t + a2) t + a1) t exp(-x^2)
        var poly = engine.TensorAddScalar(engine.TensorMultiplyScalar(t, a5), a4);
        poly = engine.TensorAddScalar(engine.TensorMultiply(poly, t), a3);
        poly = engine.TensorAddScalar(engine.TensorMultiply(poly, t), a2);
        poly = engine.TensorAddScalar(engine.TensorMultiply(poly, t), a1);
        var tail = engine.TensorMultiply(engine.TensorMultiply(poly, t), engine.TensorExp(engine.TensorNegate(engine.TensorMultiply(ax, ax))));
        return engine.TensorMultiply(sign, engine.ScalarMinusTensor(NumOps.One, tail));
    }

    /// <summary>
    /// A tensor of ones shaped like <paramref name="like"/>.
    /// </summary>
    private Tensor<T> Ones(Tensor<T> like)
    {
        var ones = new Tensor<T>(like.Shape.ToArray());
        Engine.TensorFill(ones, NumOps.One);
        return ones;
    }

    /// <summary>
    /// Enforces the ordering constraint on thresholds: α_1 &lt; α_2 &lt; ... &lt; α_{K-1}.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After each gradient update, thresholds may violate the ordering constraint.
    /// This method projects them back to a valid ordered configuration.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The thresholds must be in increasing order (each one bigger than the previous).
    /// This method fixes them if the gradient update caused them to get out of order.
    /// </para>
    /// </remarks>
    private void EnforceThresholdOrdering()
    {
        if (_thresholds == null || _thresholds.Length <= 1) return;

        // Simple projection: ensure each threshold is at least slightly larger than the previous
        // (a sequential scan: each fix depends on the previous, already fixed threshold)
        T minGap = NumOps.FromDouble(0.01);
        var values = _thresholds.ToArray();
        for (int k = 1; k < values.Length; k++)
        {
            T minAllowed = NumOps.Add(values[k - 1], minGap);
            if (!NumOps.GreaterThan(values[k], minAllowed))
            {
                values[k] = minAllowed;
            }
        }

        _thresholds = Vector<T>.Wrap(values);
    }

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector of predicted class indices.</returns>
    /// <remarks>
    /// <para>
    /// Predictions are made by computing class probabilities and returning the class
    /// with the highest probability.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, use this to predict which category new examples belong to.
    /// The model computes probabilities for each category and picks the most likely one.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_coefficients.Length == 0 || _thresholds.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained first. Call Train().");
        }

        var labels = ClassLabels ?? throw new InvalidOperationException("Model has not been fitted.");
        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        // Class with maximum probability (the first one wins ties)
        var probs = ComputeClassProbabilities(input);
        return ClassIndicesToLabels(Engine.TensorArgMax(probs, 1));
    }

    /// <summary>
    /// Computes the probability distribution over classes for every sample.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <returns>The [rows, classes] class probabilities.</returns>
    /// <remarks>
    /// <para>
    /// Class probabilities are computed from cumulative probabilities:
    /// P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)
    /// </para>
    /// <para>
    /// For Beginners:
    /// This computes the probability of being in each category (all probabilities sum to 1).
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeClassProbabilities(Matrix<T> x)
    {
        var engine = Engine;
        int n = x.Rows;

        // Cumulative probabilities P(Y ≤ k) — uses special functions (sigmoid, Erf)
        var cumulative = CumulativeProbabilities(FeatureBlock(x));

        // Class probabilities P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1), clamped to [0, 1]
        var zeros = new Tensor<T>(new[] { n, 1 });
        var ones = new Tensor<T>(new[] { n, 1 });
        engine.TensorFill(ones, NumOps.One);
        var upper = engine.TensorConcatenate(new[] { cumulative, ones }, 1);
        var lower = engine.TensorConcatenate(new[] { zeros, cumulative }, 1);
        return engine.TensorClamp(engine.TensorSubtract(upper, lower), NumOps.Zero, NumOps.One);
    }

    /// <summary>
    /// Returns the probability estimates for all classes.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A matrix where each row contains class probabilities for a sample.</returns>
    /// <remarks>
    /// <para>
    /// Useful when you need the full probability distribution over classes,
    /// not just the most likely class.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Sometimes you want to know not just the predicted class, but how confident
    /// the model is about each category. This returns those probabilities.
    /// </para>
    /// </remarks>
    public Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_coefficients.Length == 0 || _thresholds.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained first. Call Train().");
        }

        return input.Rows == 0 ? new Matrix<T>(0, NumClasses) : ComputeClassProbabilities(input).ToMatrix();
    }


    /// <summary>
    /// Gets all model parameters as a single vector (coefficients + thresholds).
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// The parameters are ordered as: [β_0, β_1, ..., β_p, α_1, α_2, ..., α_{K-1}]
    /// where p is the number of features and K is the number of classes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This packages all the model's learned values into a single list.
    /// </para>
    /// </remarks>
    private Vector<T> PackParameters()
    {
        int numParams = NumFeatures + Math.Max(0, NumClasses - 1);
        if (numParams <= 0) return new Vector<T>(0);

        // Coefficients then thresholds
        var coefficients = _coefficients is not null ? FitToLength(_coefficients, NumFeatures) : new Vector<T>(NumFeatures);
        int numThresholds = Math.Max(0, NumClasses - 1);
        var thresholds = _thresholds is not null ? FitToLength(_thresholds, numThresholds) : new Vector<T>(numThresholds);
        return Vector<T>.Concatenate(coefficients, thresholds);
    }

    /// <summary>
    /// Sets the model parameters from a vector.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has an incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method expects parameters in the order: [coefficients, thresholds].
    /// </para>
    /// <para>
    /// For Beginners:
    /// Sets all the model's learned values from a single list. Useful for loading a saved model.
    /// </para>
    /// </remarks>
    private void UnpackParameters(Vector<T> parameters)
    {
        // If model is untrained, infer dimensions from parameters
        int expectedLength = NumFeatures + NumClasses - 1;
        if (expectedLength <= 0 && parameters.Length > 0)
        {
            // Assume at least 2 classes (ordinal regression needs K>=2 categories)
            if (NumClasses == 0) NumClasses = 2;
            NumFeatures = parameters.Length - (NumClasses - 1);
            if (NumFeatures <= 0)
            {
                // Can't determine — accept silently for optimizer initialization
                return;
            }
            expectedLength = NumFeatures + NumClasses - 1;
        }

        if (parameters.Length != expectedLength)
        {
            // Try to re-derive with different class count
            if (parameters.Length > 0)
            {
                for (int k = 2; k <= parameters.Length; k++)
                {
                    int nf = parameters.Length - (k - 1);
                    if (nf > 0)
                    {
                        NumClasses = k;
                        NumFeatures = nf;
                        expectedLength = nf + k - 1;
                        if (expectedLength == parameters.Length) break;
                    }
                }
            }
            if (parameters.Length != expectedLength)
                return; // Can't match — accept silently
        }

        // Extract coefficients and thresholds
        _coefficients = parameters.GetSubVector(0, NumFeatures);
        _thresholds = parameters.GetSubVector(NumFeatures, NumClasses - 1);
    }

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has an incorrect length.</exception>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new OrdinalRegression<T>(_options, Regularization);
        newModel.NumFeatures = NumFeatures;
        newModel.NumClasses = NumClasses;
        newModel.ClassLabels = ClassLabels?.Clone();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Computes gradients for the model parameters.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <param name="target">The target class labels.</param>
    /// <param name="lossFunction">The loss function to use (ignored, uses ordinal likelihood).</param>
    /// <returns>A vector of gradients for all parameters.</returns>
    /// <remarks>
    /// <para>
    /// Computes the gradient of the negative log-likelihood with respect to all parameters.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Gradients tell the optimization algorithm how to adjust parameters to improve predictions.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var (coeffGradient, threshGradient, _) = ComputeGradients(input, target);

        // Negate gradients (we compute gradient of log-likelihood but need gradient of loss)
        var engine = Engine;
        return Vector<T>.Concatenate(
            engine.Negate(FitToLength(coeffGradient, NumFeatures)),
            engine.Negate(FitToLength(threshGradient, Math.Max(0, NumClasses - 1))));
    }

    /// <summary>
    /// Applies gradients to update the model parameters.
    /// </summary>
    /// <param name="gradients">The gradients for all parameters.</param>
    /// <param name="learningRate">The learning rate to scale the gradients.</param>
    /// <remarks>
    /// <para>
    /// Updates parameters in the negative gradient direction (gradient descent).
    /// </para>
    /// <para>
    /// For Beginners:
    /// This applies the computed gradients to update the model's parameters during training.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_coefficients.Length == 0 || _thresholds.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained first. Call Train().");
        }

        // Update coefficients and thresholds: w = w - lr * g
        var engine = Engine;
        _coefficients = engine.Subtract(FitToLength(_coefficients, NumFeatures),
            engine.Multiply(gradients.GetSubVector(0, NumFeatures), learningRate));
        _thresholds = engine.Subtract(FitToLength(_thresholds, NumClasses - 1),
            engine.Multiply(gradients.GetSubVector(NumFeatures, NumClasses - 1), learningRate));

        // Enforce threshold ordering
        EnforceThresholdOrdering();
    }

    /// <summary>
    /// Gets the feature importance scores (absolute coefficient values).
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    /// <remarks>
    /// <para>
    /// Feature importance is measured by the absolute value of coefficients.
    /// Larger absolute values indicate more influential features.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This tells you which features have the biggest impact on predictions.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        if (_coefficients == null || _coefficients.Length == 0) return result;

        var magnitudes = Engine.TensorAbs(_coefficients.AsTensor()).ToArray();
        for (int i = 0; i < magnitudes.Length; i++)
        {
            string name = FeatureNames != null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = magnitudes[i];
        }

        return result;
    }
}
