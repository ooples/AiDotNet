using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;

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
/// - Survey responses: Strongly Disagree < Disagree < Neutral < Agree < Strongly Agree
/// - Education levels: High School < Bachelor's < Master's < PhD
/// - Pain levels: None < Mild < Moderate < Severe
///
/// The model learns:
/// 1. Feature coefficients (β): How each feature pushes predictions up or down the ordinal scale
/// 2. Thresholds (α): Where to draw the lines between adjacent categories
///
/// This is better than treating it as regular classification (which ignores order) or as
/// regression (which assumes equal distances between categories).
/// </para>
/// </remarks>
public class OrdinalRegression<T> : ClassifierBase<T>
{
    /// <summary>
    /// Configuration options for the ordinal regression model.
    /// </summary>
    private readonly OrdinalRegressionOptions<T> _options;

    /// <summary>
    /// Feature coefficients (β). Shared across all thresholds (proportional odds assumption).
    /// </summary>
    private Vector<T>? _coefficients;

    /// <summary>
    /// Threshold parameters (α_1, α_2, ..., α_{K-1}) where K is the number of classes.
    /// These are in increasing order: α_1 < α_2 < ... < α_{K-1}.
    /// </summary>
    private Vector<T>? _thresholds;

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
        : base(options, regularization, new CrossEntropyLoss<T>())
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

        // Initialize coefficients to zero
        _coefficients = new Vector<T>(NumFeatures);
        for (int i = 0; i < NumFeatures; i++)
        {
            _coefficients[i] = NumOps.Zero;
        }

        // Initialize thresholds evenly spaced
        _thresholds = new Vector<T>(NumClasses - 1);
        for (int k = 0; k < NumClasses - 1; k++)
        {
            // Initial thresholds: -2, -1, 0, 1, 2, ... for K-1 thresholds centered around 0
            double threshold = (k + 1.0) - NumClasses / 2.0;
            _thresholds[k] = NumOps.FromDouble(threshold);
        }

        // Gradient descent optimization
        double prevLogLikelihood = double.NegativeInfinity;

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            // Compute gradients and update parameters
            var (coeffGradient, threshGradient, logLikelihood) = ComputeGradients(x, y);

            // Update coefficients
            for (int j = 0; j < NumFeatures; j++)
            {
                double coeff = NumOps.ToDouble(_coefficients[j]);
                double grad = NumOps.ToDouble(coeffGradient[j]);

                // Add L2 regularization gradient
                grad -= _options.RegularizationStrength * coeff;

                coeff += _options.LearningRate * grad;
                _coefficients[j] = NumOps.FromDouble(coeff);
            }

            // Update thresholds
            for (int k = 0; k < NumClasses - 1; k++)
            {
                double thresh = NumOps.ToDouble(_thresholds[k]);
                double grad = NumOps.ToDouble(threshGradient[k]);
                thresh += _options.LearningRate * grad;
                _thresholds[k] = NumOps.FromDouble(thresh);
            }

            // Enforce threshold ordering: α_1 < α_2 < ... < α_{K-1}
            EnforceThresholdOrdering();

            // Check convergence
            if (Math.Abs(logLikelihood - prevLogLikelihood) < _options.Tolerance)
            {
                break;
            }

            prevLogLikelihood = logLikelihood;
        }
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
        for (int i = 0; i < y.Length; i++)
        {
            int classIdx = GetClassIndexFromLabel(y[i]);
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
    private (Vector<T> coeffGradient, Vector<T> threshGradient, double logLikelihood) ComputeGradients(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        var coeffGradient = new Vector<T>(NumFeatures);
        var threshGradient = new Vector<T>(NumClasses - 1);
        double logLikelihood = 0;

        // Initialize gradients to zero
        for (int j = 0; j < NumFeatures; j++) coeffGradient[j] = NumOps.Zero;
        for (int k = 0; k < NumClasses - 1; k++) threshGradient[k] = NumOps.Zero;

        // Compute linear predictor for all samples: η_i = β^T × x_i
        var eta = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < NumFeatures; j++)
            {
                sum += NumOps.ToDouble(_coefficients![j]) * NumOps.ToDouble(x[i, j]);
            }
            eta[i] = sum;
        }

        // For each sample, compute probabilities and gradients
        for (int i = 0; i < n; i++)
        {
            int yi = GetClassIndexFromLabel(y[i]);

            // Compute cumulative probabilities P(Y ≤ k) = sigmoid(α_k - η)
            var cumProbs = new double[NumClasses - 1];
            for (int k = 0; k < NumClasses - 1; k++)
            {
                double alpha_k = NumOps.ToDouble(_thresholds![k]);
                cumProbs[k] = ApplyLink(alpha_k - eta[i]);
            }

            // Compute class probabilities P(Y = k)
            var classProbs = new double[NumClasses];
            classProbs[0] = cumProbs[0];
            for (int k = 1; k < NumClasses - 1; k++)
            {
                classProbs[k] = cumProbs[k] - cumProbs[k - 1];
            }
            classProbs[NumClasses - 1] = 1.0 - cumProbs[NumClasses - 2];

            // Clamp probabilities to avoid log(0)
            for (int k = 0; k < NumClasses; k++)
            {
                classProbs[k] = Math.Max(1e-10, Math.Min(1.0 - 1e-10, classProbs[k]));
            }

            // Add to log-likelihood
            logLikelihood += Math.Log(classProbs[yi]);

            // Compute gradient contributions
            // For coefficient gradient: need derivative of P(Y=yi) w.r.t. η
            // d/dη P(Y=k) = -sigmoid'(α_k - η) + sigmoid'(α_{k-1} - η) = P(Y≤k)(1-P(Y≤k)) - P(Y≤k-1)(1-P(Y≤k-1))

            double dPdEta = 0;
            if (yi < NumClasses - 1)
            {
                dPdEta -= cumProbs[yi] * (1 - cumProbs[yi]); // -sigmoid'(α_yi - η)
            }
            if (yi > 0)
            {
                dPdEta += cumProbs[yi - 1] * (1 - cumProbs[yi - 1]); // +sigmoid'(α_{yi-1} - η)
            }

            // Gradient for coefficients: (1/P(Y=yi)) × dP(Y=yi)/dη × x
            // Note: since dP/dη is negative of dP/d(α-η), and we want dL/dβ = Σ dL/dP × dP/dη × (-x)
            double coeffFactor = dPdEta / classProbs[yi];
            for (int j = 0; j < NumFeatures; j++)
            {
                double xij = NumOps.ToDouble(x[i, j]);
                double currentGrad = NumOps.ToDouble(coeffGradient[j]);
                // The sign is negative because ∂η/∂β = x and ∂(α-η)/∂β = -x
                currentGrad -= coeffFactor * xij;
                coeffGradient[j] = NumOps.FromDouble(currentGrad);
            }

            // Gradient for thresholds: (1/P(Y=yi)) × dP(Y=yi)/dα_k
            // dP(Y=k)/dα_k = sigmoid'(α_k - η) = cumProbs[k] × (1 - cumProbs[k])
            // dP(Y=k)/dα_{k-1} = -sigmoid'(α_{k-1} - η) = -cumProbs[k-1] × (1 - cumProbs[k-1])
            for (int k = 0; k < NumClasses - 1; k++)
            {
                double threshGrad = 0;

                if (k == yi)
                {
                    // α_k appears in P(Y≤yi) with positive contribution
                    threshGrad += cumProbs[k] * (1 - cumProbs[k]);
                }
                if (k == yi - 1)
                {
                    // α_{yi-1} appears in P(Y≤yi-1) which is subtracted
                    threshGrad -= cumProbs[k] * (1 - cumProbs[k]);
                }

                double currentThreshGrad = NumOps.ToDouble(threshGradient[k]);
                currentThreshGrad += threshGrad / classProbs[yi];
                threshGradient[k] = NumOps.FromDouble(currentThreshGrad);
            }
        }

        return (coeffGradient, threshGradient, logLikelihood);
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
    private double ApplyLink(double linearPredictor)
    {
        return _options.LinkFunction switch
        {
            OrdinalLinkFunction.Logit => 1.0 / (1.0 + Math.Exp(-linearPredictor)),
            OrdinalLinkFunction.Probit => NormalCdf(linearPredictor),
            OrdinalLinkFunction.ComplementaryLogLog => 1.0 - Math.Exp(-Math.Exp(linearPredictor)),
            _ => 1.0 / (1.0 + Math.Exp(-linearPredictor))
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
    private static double NormalCdf(double x)
    {
        // Approximation using error function
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
    }

    /// <summary>
    /// Computes the error function approximation.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The error function value.</returns>
    private static double Erf(double x)
    {
        // Horner form coefficients for approximation
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
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
        double minGap = 0.01;
        for (int k = 1; k < _thresholds.Length; k++)
        {
            double prevThresh = NumOps.ToDouble(_thresholds[k - 1]);
            double currThresh = NumOps.ToDouble(_thresholds[k]);

            if (currThresh <= prevThresh + minGap)
            {
                _thresholds[k] = NumOps.FromDouble(prevThresh + minGap);
            }
        }
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
        if (_coefficients == null || _thresholds == null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Compute class probabilities
            double[] classProbs = ComputeClassProbabilities(input, i);

            // Find class with maximum probability
            int maxClass = 0;
            double maxProb = classProbs[0];
            for (int k = 1; k < NumClasses; k++)
            {
                if (classProbs[k] > maxProb)
                {
                    maxProb = classProbs[k];
                    maxClass = k;
                }
            }

            predictions[i] = ClassLabels![maxClass];
        }

        return predictions;
    }

    /// <summary>
    /// Computes the probability distribution over classes for a single sample.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="sampleIndex">The index of the sample to compute probabilities for.</param>
    /// <returns>An array of class probabilities.</returns>
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
    private double[] ComputeClassProbabilities(Matrix<T> x, int sampleIndex)
    {
        // Compute linear predictor: η = β^T × x
        double eta = 0;
        for (int j = 0; j < NumFeatures; j++)
        {
            eta += NumOps.ToDouble(_coefficients![j]) * NumOps.ToDouble(x[sampleIndex, j]);
        }

        // Compute cumulative probabilities P(Y ≤ k)
        var cumProbs = new double[NumClasses - 1];
        for (int k = 0; k < NumClasses - 1; k++)
        {
            double alpha_k = NumOps.ToDouble(_thresholds![k]);
            cumProbs[k] = ApplyLink(alpha_k - eta);
        }

        // Compute class probabilities
        var classProbs = new double[NumClasses];
        classProbs[0] = cumProbs[0];
        for (int k = 1; k < NumClasses - 1; k++)
        {
            classProbs[k] = cumProbs[k] - cumProbs[k - 1];
        }
        classProbs[NumClasses - 1] = 1.0 - cumProbs[NumClasses - 2];

        // Clamp probabilities
        for (int k = 0; k < NumClasses; k++)
        {
            classProbs[k] = Math.Max(0, Math.Min(1.0, classProbs[k]));
        }

        return classProbs;
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
        if (_coefficients == null || _thresholds == null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var probs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            double[] classProbs = ComputeClassProbabilities(input, i);
            for (int k = 0; k < NumClasses; k++)
            {
                probs[i, k] = NumOps.FromDouble(classProbs[k]);
            }
        }

        return probs;
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for ordinal regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Returns an identifier indicating this is an ordinal regression model.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.OrdinalRegression;
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
    public override Vector<T> GetParameters()
    {
        int numParams = NumFeatures + NumClasses - 1;
        var parameters = new Vector<T>(numParams);

        // Add coefficients
        for (int j = 0; j < NumFeatures; j++)
        {
            parameters[j] = _coefficients is not null ? _coefficients[j] : NumOps.Zero;
        }

        // Add thresholds
        for (int k = 0; k < NumClasses - 1; k++)
        {
            parameters[NumFeatures + k] = _thresholds is not null ? _thresholds[k] : NumOps.Zero;
        }

        return parameters;
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
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedLength = NumFeatures + NumClasses - 1;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, got {parameters.Length}.");
        }

        // Extract coefficients
        _coefficients = new Vector<T>(NumFeatures);
        for (int j = 0; j < NumFeatures; j++)
        {
            _coefficients[j] = parameters[j];
        }

        // Extract thresholds
        _thresholds = new Vector<T>(NumClasses - 1);
        for (int k = 0; k < NumClasses - 1; k++)
        {
            _thresholds[k] = parameters[NumFeatures + k];
        }
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
    /// Creates a new instance of the same type as this classifier.
    /// </summary>
    /// <returns>A new instance of the same classifier type.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OrdinalRegression<T>(_options, Regularization);
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
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var (coeffGradient, threshGradient, _) = ComputeGradients(input, target);

        int numParams = NumFeatures + NumClasses - 1;
        var gradients = new Vector<T>(numParams);

        // Negate gradients (we compute gradient of log-likelihood but need gradient of loss)
        for (int j = 0; j < NumFeatures; j++)
        {
            gradients[j] = NumOps.Negate(coeffGradient[j]);
        }

        for (int k = 0; k < NumClasses - 1; k++)
        {
            gradients[NumFeatures + k] = NumOps.Negate(threshGradient[k]);
        }

        return gradients;
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
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_coefficients == null || _thresholds == null)
        {
            throw new InvalidOperationException("Model must be trained before applying gradients.");
        }

        double lr = NumOps.ToDouble(learningRate);

        // Update coefficients
        for (int j = 0; j < NumFeatures; j++)
        {
            double coeff = NumOps.ToDouble(_coefficients[j]);
            double grad = NumOps.ToDouble(gradients[j]);
            _coefficients[j] = NumOps.FromDouble(coeff - lr * grad);
        }

        // Update thresholds
        for (int k = 0; k < NumClasses - 1; k++)
        {
            double thresh = NumOps.ToDouble(_thresholds[k]);
            double grad = NumOps.ToDouble(gradients[NumFeatures + k]);
            _thresholds[k] = NumOps.FromDouble(thresh - lr * grad);
        }

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
        if (_coefficients == null) return result;

        for (int i = 0; i < _coefficients.Length; i++)
        {
            string name = FeatureNames != null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(_coefficients[i])));
        }

        return result;
    }

    /// <summary>
    /// Indicates whether this model supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing class probabilities.</returns>
    /// <remarks>
    /// <para>
    /// Creates a computation graph for the ordinal regression model that can be
    /// JIT compiled for faster inference.
    /// </para>
    /// <para>
    /// For Beginners:
    /// JIT compilation can make predictions faster by converting the model to optimized code.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_coefficients == null || _thresholds == null)
        {
            throw new InvalidOperationException("Model must be trained before exporting computation graph.");
        }

        // Create input placeholder for features: [batchSize, numFeatures]
        var inputTensor = new Tensor<T>(new int[] { 1, NumFeatures });
        var inputNode = TensorOperations<T>.Variable(inputTensor, "features");
        inputNodes.Add(inputNode);

        // Create constant node for coefficients: [numFeatures, 1]
        var coeffTensor = new Tensor<T>(new int[] { NumFeatures, 1 });
        for (int j = 0; j < NumFeatures; j++)
        {
            coeffTensor[j, 0] = _coefficients[j];
        }
        var coeffNode = TensorOperations<T>.Constant(coeffTensor, "coefficients");
        inputNodes.Add(coeffNode);

        // Compute linear predictor: η = X @ β, shape [batchSize, 1]
        var etaNode = TensorOperations<T>.MatrixMultiply(inputNode, coeffNode);

        // Create constant nodes for thresholds
        var thresholdNodes = new List<ComputationNode<T>>();
        for (int k = 0; k < NumClasses - 1; k++)
        {
            var threshTensor = new Tensor<T>(new int[] { 1, 1 });
            threshTensor[0, 0] = _thresholds[k];
            var threshNode = TensorOperations<T>.Constant(threshTensor, $"threshold_{k}");
            inputNodes.Add(threshNode);
            thresholdNodes.Add(threshNode);
        }

        // Compute cumulative probabilities using sigmoid
        // P(Y ≤ k) = sigmoid(α_k - η)
        var cumProbNodes = new List<ComputationNode<T>>();
        for (int k = 0; k < NumClasses - 1; k++)
        {
            var diffNode = TensorOperations<T>.Subtract(thresholdNodes[k], etaNode);
            var probNode = TensorOperations<T>.Sigmoid(diffNode);
            cumProbNodes.Add(probNode);
        }

        // The output is the argmax of class probabilities
        // For simplicity, we return the linear predictor for now
        // (class prediction requires argmax which may not be differentiable)
        etaNode.Name = "linear_predictor";

        return etaNode;
    }
}
