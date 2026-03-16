using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Ordinal;

/// <summary>
/// Ordinal Logistic Regression (Proportional Odds Model).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Ordinal logistic regression is the standard approach for ordinal data.
/// It's also known as the "proportional odds model" or "cumulative link model".</para>
///
/// <para><b>How it works:</b> Instead of predicting the class directly, it models the cumulative
/// probability of being in class k or lower:
/// <code>
/// log(P(Y ≤ k) / P(Y &gt; k)) = θₖ - β·X
/// </code>
/// Where:
/// <list type="bullet">
/// <item><b>θₖ</b> = threshold (cutpoint) for class k</item>
/// <item><b>β</b> = coefficient vector (same for all classes - "proportional odds")</item>
/// <item><b>X</b> = feature vector</item>
/// </list>
/// </para>
///
/// <para><b>Key assumption - Proportional Odds:</b> The effect of each feature is the same
/// regardless of which threshold we're considering. This means moving from 1→2 stars has
/// the same relationship with features as moving from 4→5 stars.</para>
///
/// <para><b>Example:</b> Rating prediction for a restaurant review:
/// <code>
/// // Train on review features (sentiment, length, etc.) and ratings (1-5)
/// var model = new OrdinalLogisticRegression&lt;double&gt;();
/// model.Train(features, ratings);
///
/// // Predict most likely rating
/// var predicted = model.Predict(newFeatures);
///
/// // Get probability distribution over ratings
/// var probs = model.PredictProbabilities(newFeatures);
/// </code>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>McCullagh, P. (1980). "Regression Models for Ordinal Data"</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Regression Models for Ordinal Data", "https://doi.org/10.1111/j.2517-6161.1980.tb01109.x", Year = 1980, Authors = "Peter McCullagh")]
public class OrdinalLogisticRegression<T> : OrdinalClassifierBase<T>
{
    /// <summary>
    /// The learned coefficient vector (β).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These coefficients tell us how each feature affects the
    /// ordinal outcome. A positive coefficient means higher values of that feature push
    /// predictions toward higher classes (e.g., more stars).</para>
    /// </remarks>
    private Vector<T>? _coefficients;

    /// <summary>
    /// The gradient-based optimizer for parameter updates.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Matrix<T>, Vector<T>> _paramOptimizer;

    /// <summary>
    /// Maximum iterations for optimization.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// L2 regularization strength.
    /// </summary>
    private readonly double _regularizationStrength;

    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Legacy learning rate field — kept for backward compatibility with serialization/clone.
    /// The optimizer handles learning rate internally.
    /// </summary>
    private readonly double _learningRate;

    /// <summary>
    /// Initializes a new instance of OrdinalLogisticRegression.
    /// </summary>
    /// <param name="optimizer">Gradient-based optimizer for training. Defaults to AdamOptimizer with industry-standard settings.</param>
    /// <param name="maxIterations">Maximum training iterations. Default is 1000.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-6.</param>
    /// <param name="regularization">L2 regularization strength. Default is 0.0.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public OrdinalLogisticRegression(
        IGradientBasedOptimizer<T, Matrix<T>, Vector<T>>? optimizer = null,
        int maxIterations = 1000,
        double tolerance = 1e-6,
        double regularization = 0.0,
        int? seed = null)
        : base()
    {
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _regularizationStrength = regularization;
        _learningRate = 0.01; // Legacy field for serialization
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        // Default to Adam optimizer — industry standard with adaptive learning rates
        // that don't require manual tuning. This replaces the hand-rolled gradient descent
        // that used a fixed learning rate of 0.01 which was too small for convergence.
        _paramOptimizer = optimizer ?? new AdamOptimizer<T, Matrix<T>, Vector<T>>(null);
    }

    /// <summary>
    /// Legacy constructor for backward compatibility with existing code and serialization.
    /// The learningRate is passed to the Adam optimizer as its initial learning rate.
    /// </summary>
    public OrdinalLogisticRegression(
        double learningRate,
        int maxIterations = 1000,
        double tolerance = 1e-6,
        double regularization = 0.0,
        int? seed = null)
        : base()
    {
        _learningRate = learningRate;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _regularizationStrength = regularization;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        // Use Adam optimizer with the user-specified learning rate
        var adamOptions = new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            InitialLearningRate = learningRate
        };
        _paramOptimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(null, adamOptions);
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.OrdinalLogisticRegression.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns an identifier for this model type,
    /// used internally for serialization and model management.</para>
    /// </remarks>

    /// <summary>
    /// Trains the ordinal logistic regression model.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Ordinal class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training finds:
    /// <list type="number">
    /// <item>Coefficients (β) that show how features affect the ordinal outcome</item>
    /// <item>Thresholds (θ) that separate the ordered classes</item>
    /// </list>
    ///
    /// The training uses gradient descent to minimize the negative log-likelihood:
    /// minimize -Σᵢ log P(Yᵢ = yᵢ | Xᵢ)
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

        int K = NumClasses;  // Number of classes
        int P = NumFeatures; // Number of features

        // Initialize coefficients and thresholds
        _coefficients = new Vector<T>(P);
        _thresholds = new Vector<T>(K - 1);

        // Initialize thresholds to evenly spaced values
        for (int k = 0; k < K - 1; k++)
        {
            _thresholds[k] = NumOps.FromDouble(-2.0 + 4.0 * (k + 1) / K);
        }

        // Initialize coefficients to small random values
        for (int p = 0; p < P; p++)
        {
            _coefficients[p] = NumOps.FromDouble(0.01 * (_random.NextDouble() - 0.5));
        }

        // Convert labels to class indices
        var yIndices = new int[y.Length];
        for (int i = 0; i < y.Length; i++)
        {
            yIndices[i] = GetClassIndex(y[i]);
        }

        // Optimization using the configured gradient-based optimizer (Adam by default).
        // The optimizer handles adaptive learning rates, momentum, and step size internally.
        double prevLoss = double.MaxValue;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Compute gradients
            var gradCoef = new double[P];
            var gradThresh = new double[K - 1];
            double loss = 0;

            for (int i = 0; i < x.Rows; i++)
            {
                int yi = yIndices[i];

                // Compute linear predictor
                double eta = 0;
                for (int p = 0; p < P; p++)
                {
                    eta += NumOps.ToDouble(_coefficients[p]) * NumOps.ToDouble(x[i, p]);
                }

                // Compute cumulative probabilities
                var cumProbs = new double[K - 1];
                for (int k = 0; k < K - 1; k++)
                {
                    double z = NumOps.ToDouble(_thresholds[k]) - eta;
                    cumProbs[k] = 1.0 / (1.0 + Math.Exp(-z));
                }

                // Compute class probabilities
                var probs = new double[K];
                probs[0] = cumProbs[0];
                for (int k = 1; k < K - 1; k++)
                {
                    probs[k] = cumProbs[k] - cumProbs[k - 1];
                }
                probs[K - 1] = 1.0 - cumProbs[K - 2];

                // Ensure non-zero probabilities
                for (int k = 0; k < K; k++)
                {
                    probs[k] = Math.Max(probs[k], 1e-15);
                }

                // Accumulate loss
                loss -= Math.Log(probs[yi]);

                // Compute gradients for coefficients w.r.t. negative log-likelihood.
                // The sign is negative because ∂η/∂β = x but ∂(α-η)/∂β = -x,
                // so the chain rule introduces a sign flip: ∂NLL/∂β = -∂NLL/∂η * x.
                for (int p = 0; p < P; p++)
                {
                    double xi = NumOps.ToDouble(x[i, p]);

                    if (yi == 0)
                    {
                        // First class: dP(Y=0)/dη = -σ'(α₀-η) = -(cumProbs[0])(1-cumProbs[0])
                        // dNLL/dβ = -(dP/dη / P) * x = (σ'(α₀-η) / P) * x
                        gradCoef[p] -= xi * (cumProbs[0] - 1);
                    }
                    else if (yi == K - 1)
                    {
                        // Last class: dP(Y=K-1)/dη = σ'(α_{K-2}-η)
                        // dNLL/dβ = -(dP/dη / P) * x
                        gradCoef[p] -= xi * cumProbs[K - 2];
                    }
                    else
                    {
                        // Middle classes
                        double term = cumProbs[yi] * (1 - cumProbs[yi]) - cumProbs[yi - 1] * (1 - cumProbs[yi - 1]);
                        gradCoef[p] -= xi * term / probs[yi];
                    }
                }

                // Compute gradients for thresholds
                for (int k = 0; k < K - 1; k++)
                {
                    double gamma_k = cumProbs[k] * (1 - cumProbs[k]);

                    if (yi == k)
                    {
                        gradThresh[k] += gamma_k / probs[yi];
                    }
                    else if (yi == k + 1)
                    {
                        gradThresh[k] += -gamma_k / probs[yi];
                    }
                }
            }

            // Add L2 regularization to coefficients gradient
            for (int p = 0; p < P; p++)
            {
                gradCoef[p] += _regularizationStrength * NumOps.ToDouble(_coefficients[p]);
                loss += 0.5 * _regularizationStrength * NumOps.ToDouble(_coefficients[p]) * NumOps.ToDouble(_coefficients[p]);
            }

            // Pack coefficients + thresholds into a single parameter vector for the optimizer
            double invN = 1.0 / x.Rows;
            var paramVec = new Vector<T>(P + K - 1);
            var gradVec = new Vector<T>(P + K - 1);

            for (int p = 0; p < P; p++)
            {
                paramVec[p] = _coefficients[p];
                gradVec[p] = NumOps.FromDouble(gradCoef[p] * invN);
            }
            for (int k = 0; k < K - 1; k++)
            {
                paramVec[P + k] = _thresholds[k];
                gradVec[P + k] = NumOps.FromDouble(gradThresh[k] * invN);
            }

            // Let the optimizer (Adam by default) handle the update with adaptive learning rates
            var updatedParams = _paramOptimizer.UpdateParameters(paramVec, gradVec);

            // Unpack updated parameters back into coefficients and thresholds
            for (int p = 0; p < P; p++)
            {
                _coefficients[p] = updatedParams[p];
            }
            for (int k = 0; k < K - 1; k++)
            {
                _thresholds[k] = updatedParams[P + k];
            }

            // Ensure thresholds are monotonically increasing
            for (int k = 1; k < K - 1; k++)
            {
                if (NumOps.LessThanOrEquals(_thresholds[k], _thresholds[k - 1]))
                {
                    _thresholds[k] = NumOps.FromDouble(NumOps.ToDouble(_thresholds[k - 1]) + 0.001);
                }
            }

            // Check convergence
            loss /= x.Rows;
            if (Math.Abs(prevLoss - loss) < _tolerance)
            {
                break;
            }
            prevLoss = loss;
        }
    }

    /// <summary>
    /// Predicts ordinal class labels.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method predicts the most likely class for each sample
    /// by finding the class with the highest probability from the probability distribution.</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            int bestClass = 0;
            T bestProb = probs[i, 0];

            for (int k = 1; k < NumClasses; k++)
            {
                if (NumOps.Compare(probs[i, k], bestProb) > 0)
                {
                    bestProb = probs[i, k];
                    bestClass = k;
                }
            }

            predictions[i] = ClassLabels is not null ? ClassLabels[bestClass] : NumOps.FromDouble(bestClass);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts cumulative probabilities P(Y ≤ k).
    /// </summary>
    /// <param name="features">Feature matrix.</param>
    /// <returns>Cumulative probability matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each sample and each threshold, this computes the
    /// probability of being in that class or lower. Uses the logistic function to convert
    /// the linear predictor (threshold - β·X) into a probability.</para>
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
            // Compute linear predictor
            double eta = 0;
            for (int p = 0; p < NumFeatures; p++)
            {
                eta += NumOps.ToDouble(_coefficients[p]) * NumOps.ToDouble(features[i, p]);
            }

            // Compute cumulative probabilities
            for (int k = 0; k < K - 1; k++)
            {
                double z = NumOps.ToDouble(_thresholds[k]) - eta;
                double prob = 1.0 / (1.0 + Math.Exp(-z));
                cumProbs[i, k] = NumOps.FromDouble(prob);
            }
        }

        return cumProbs;
    }

    /// <summary>
    /// Gets the model parameters (coefficients and thresholds).
    /// </summary>
    /// <returns>Vector containing all model parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This packs all learned parameters into a single vector
    /// for serialization or optimization purposes. The coefficients come first, followed
    /// by the thresholds.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_coefficients is null || _thresholds is null)
        {
            return new Vector<T>(0);
        }

        int totalLen = _coefficients.Length + _thresholds.Length;
        var parameters = new Vector<T>(totalLen);

        for (int i = 0; i < _coefficients.Length; i++)
        {
            parameters[i] = _coefficients[i];
        }

        for (int i = 0; i < _thresholds.Length; i++)
        {
            parameters[_coefficients.Length + i] = _thresholds[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Vector containing all model parameters.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This unpacks parameters from a single vector and sets
    /// the internal coefficients and thresholds. Used when loading a saved model.</para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length == 0)
        {
            return;
        }

        // If we have NumFeatures set, use it to determine split
        if (NumFeatures > 0 && NumClasses > 0)
        {
            int numCoef = NumFeatures;
            int numThresh = NumClasses - 1;

            if (parameters.Length == numCoef + numThresh)
            {
                _coefficients = new Vector<T>(numCoef);
                _thresholds = new Vector<T>(numThresh);

                for (int i = 0; i < numCoef; i++)
                {
                    _coefficients[i] = parameters[i];
                }

                for (int i = 0; i < numThresh; i++)
                {
                    _thresholds[i] = parameters[numCoef + i];
                }

                return;
            }
        }

        // Fallback: assume roughly equal split between coefficients and thresholds
        int numThresholds = (parameters.Length + 1) / 2;
        int numCoefs = parameters.Length - numThresholds;

        _coefficients = new Vector<T>(numCoefs);
        _thresholds = new Vector<T>(numThresholds);

        for (int i = 0; i < numCoefs; i++)
        {
            _coefficients[i] = parameters[i];
        }

        for (int i = 0; i < numThresholds; i++)
        {
            _thresholds[i] = parameters[numCoefs + i];
        }

        NumFeatures = numCoefs;
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">Parameters for the new instance.</param>
    /// <returns>New model instance with the specified parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh copy of the model with specific parameter
    /// values. Useful for optimization algorithms that explore different parameter settings.</para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var model = new OrdinalLogisticRegression<T>(_learningRate, _maxIterations, _tolerance, _regularizationStrength);
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
    /// <para><b>For Beginners:</b> Creates a fresh, untrained copy of the model with
    /// the same configuration settings (learning rate, iterations, etc.).</para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OrdinalLogisticRegression<T>(_learningRate, _maxIterations, _tolerance, _regularizationStrength);
    }

    /// <summary>
    /// Computes gradients for the model parameters.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="target">Target labels.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <returns>Gradient vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients indicate how changing each parameter would
    /// affect the prediction error. The optimizer uses these to improve the model.
    /// This computes the derivative of the negative log-likelihood with respect to
    /// each parameter.</para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        int K = NumClasses;
        int P = NumFeatures;
        var gradCoef = new double[P];
        var gradThresh = new double[K - 1];

        for (int i = 0; i < input.Rows; i++)
        {
            int yi = GetClassIndex(target[i]);

            // Compute linear predictor
            double eta = 0;
            for (int p = 0; p < P; p++)
            {
                eta += NumOps.ToDouble(_coefficients[p]) * NumOps.ToDouble(input[i, p]);
            }

            // Compute cumulative probabilities
            var cumProbs = new double[K - 1];
            for (int k = 0; k < K - 1; k++)
            {
                double z = NumOps.ToDouble(_thresholds[k]) - eta;
                cumProbs[k] = 1.0 / (1.0 + Math.Exp(-z));
            }

            // Compute class probabilities
            var probs = new double[K];
            probs[0] = cumProbs[0];
            for (int k = 1; k < K - 1; k++)
            {
                probs[k] = cumProbs[k] - cumProbs[k - 1];
            }
            probs[K - 1] = 1.0 - cumProbs[K - 2];

            // Ensure non-zero probabilities
            for (int k = 0; k < K; k++)
            {
                probs[k] = Math.Max(probs[k], 1e-15);
            }

            // Compute gradients for coefficients
            for (int p = 0; p < P; p++)
            {
                double xi = NumOps.ToDouble(input[i, p]);

                if (yi == 0)
                {
                    gradCoef[p] += xi * (cumProbs[0] - 1);
                }
                else if (yi == K - 1)
                {
                    gradCoef[p] += xi * cumProbs[K - 2];
                }
                else
                {
                    double term = cumProbs[yi] * (1 - cumProbs[yi]) - cumProbs[yi - 1] * (1 - cumProbs[yi - 1]);
                    gradCoef[p] += xi * term / probs[yi];
                }
            }

            // Compute gradients for thresholds
            for (int k = 0; k < K - 1; k++)
            {
                double gamma_k = cumProbs[k] * (1 - cumProbs[k]);

                if (yi == k)
                {
                    gradThresh[k] += gamma_k / probs[yi];
                }
                else if (yi == k + 1)
                {
                    gradThresh[k] += -gamma_k / probs[yi];
                }
            }
        }

        // Normalize by batch size
        int n = input.Rows;
        var gradients = new Vector<T>(P + K - 1);

        for (int p = 0; p < P; p++)
        {
            gradients[p] = NumOps.FromDouble(gradCoef[p] / n);
        }

        for (int k = 0; k < K - 1; k++)
        {
            gradients[P + k] = NumOps.FromDouble(gradThresh[k] / n);
        }

        return gradients;
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <param name="learningRate">Learning rate for the update.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This updates the model parameters by moving them in the
    /// opposite direction of the gradients (to reduce error). The learning rate controls
    /// how big the step is.</para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        double lr = NumOps.ToDouble(learningRate);

        // Update coefficients
        for (int p = 0; p < _coefficients.Length; p++)
        {
            double newVal = NumOps.ToDouble(_coefficients[p]) - lr * NumOps.ToDouble(gradients[p]);
            _coefficients[p] = NumOps.FromDouble(newVal);
        }

        // Update thresholds
        for (int k = 0; k < _thresholds.Length; k++)
        {
            double newVal = NumOps.ToDouble(_thresholds[k]) - lr * NumOps.ToDouble(gradients[_coefficients.Length + k]);
            _thresholds[k] = NumOps.FromDouble(newVal);
        }

        // Ensure thresholds are monotonically increasing
        for (int k = 1; k < _thresholds.Length; k++)
        {
            if (NumOps.LessThanOrEquals(_thresholds[k], _thresholds[k - 1]))
            {
                _thresholds[k] = NumOps.FromDouble(NumOps.ToDouble(_thresholds[k - 1]) + 0.001);
            }
        }
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() }
        };

        if (_coefficients is not null)
        {
            var coefArray = new double[_coefficients.Length];
            for (int i = 0; i < _coefficients.Length; i++)
                coefArray[i] = NumOps.ToDouble(_coefficients[i]);
            modelData["Coefficients"] = coefArray;
        }

        if (_thresholds is not null)
        {
            var threshArray = new double[_thresholds.Length];
            for (int i = 0; i < _thresholds.Length; i++)
                threshArray[i] = NumOps.ToDouble(_thresholds[i]);
            modelData["Thresholds"] = threshArray;
        }

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<JObject>(modelDataString);

        if (modelDataObj == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        NumClasses = modelDataObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(modelDataObj["TaskType"]?.ToObject<int>() ?? 0);

        var classLabelsToken = modelDataObj["ClassLabels"];
        if (classLabelsToken is not null)
        {
            var classLabelsAsDoubles = classLabelsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (classLabelsAsDoubles.Length > 0)
            {
                ClassLabels = new Vector<T>(classLabelsAsDoubles.Length);
                for (int i = 0; i < classLabelsAsDoubles.Length; i++)
                    ClassLabels[i] = NumOps.FromDouble(classLabelsAsDoubles[i]);
            }
        }

        var coefToken = modelDataObj["Coefficients"];
        if (coefToken is not null)
        {
            var coefArray = coefToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (coefArray.Length > 0)
            {
                _coefficients = new Vector<T>(coefArray.Length);
                for (int i = 0; i < coefArray.Length; i++)
                    _coefficients[i] = NumOps.FromDouble(coefArray[i]);
            }
        }

        var threshToken = modelDataObj["Thresholds"];
        if (threshToken is not null)
        {
            var threshArray = threshToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (threshArray.Length > 0)
            {
                _thresholds = new Vector<T>(threshArray.Length);
                for (int i = 0; i < threshArray.Length; i++)
                    _thresholds[i] = NumOps.FromDouble(threshArray[i]);
            }
        }
    }

    /// <summary>
    /// Gets feature importance based on coefficient magnitude.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feature importance tells you which features have the
    /// biggest impact on predictions. Features with larger absolute coefficient values
    /// are more important. Values are normalized to sum to 1.</para>
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
