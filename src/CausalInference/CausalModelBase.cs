using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Newtonsoft.Json;

namespace AiDotNet.CausalInference;

/// <summary>
/// Abstract base class for causal inference models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This base class provides common functionality for causal inference models including
/// propensity score estimation, treatment effect calculation, and overlap checking.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class contains shared code that all causal inference models need,
/// so each specific model (like PropensityScoreMatching) doesn't have to reimplement it.
///
/// Key shared functionality:
/// - Estimating propensity scores (probability of treatment)
/// - Calculating treatment effects
/// - Checking overlap assumptions
/// - Managing fitted model state
/// </para>
/// </remarks>
public abstract class CausalModelBase<T> : ICausalModel<T>
{
    /// <summary>
    /// Numeric operations helper for generic math.
    /// </summary>
    protected readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The default loss function for gradient computation.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Indicates whether the model has been fitted.
    /// </summary>
    protected bool IsFitted;

    /// <summary>
    /// Gets the number of features the model was trained on.
    /// </summary>
    protected int NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets whether the model is trained.
    /// </summary>
    public bool IsTrained => IsFitted;

    /// <summary>
    /// Gets the default loss function.
    /// </summary>
    public ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    public virtual int ParameterCount => NumFeatures;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the CausalModelBase class.
    /// </summary>
    protected CausalModelBase()
    {
        _defaultLossFunction = new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Ensures the model has been fitted before making predictions.
    /// </summary>
    protected void EnsureFitted()
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                "Model must be fitted before making predictions. Call Fit first.");
        }
    }

    #region ICausalModel Implementation

    /// <summary>
    /// Estimates the Average Treatment Effect (ATE) from the data.
    /// </summary>
    public abstract (T estimate, T standardError) EstimateATE(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome);

    /// <summary>
    /// Estimates the Average Treatment Effect on the Treated (ATT).
    /// </summary>
    public abstract (T estimate, T standardError) EstimateATT(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome);

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE) for each individual.
    /// </summary>
    public abstract Vector<T> EstimateCATEPerIndividual(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome);

    /// <summary>
    /// Predicts the treatment effect for new individuals.
    /// </summary>
    public abstract Vector<T> PredictTreatmentEffect(Matrix<T> x);

    /// <summary>
    /// Estimates propensity scores for each individual.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Propensity scores estimate how likely each person was to
    /// receive treatment based on their characteristics. This uses logistic regression
    /// by default, but subclasses can override with other methods.
    /// </para>
    /// </remarks>
    public virtual Vector<T> EstimatePropensityScores(Matrix<T> x)
    {
        EnsureFitted();
        return EstimatePropensityScoresCore(x);
    }

    /// <summary>
    /// Core propensity score estimation to be implemented by derived classes.
    /// </summary>
    protected abstract Vector<T> EstimatePropensityScoresCore(Matrix<T> x);

    /// <summary>
    /// Checks the overlap/positivity assumption.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Overlap means that both treated and control groups have
    /// individuals at all levels of propensity scores. Without overlap, we can't
    /// make valid comparisons.
    /// </para>
    /// </remarks>
    public virtual (T treatmentMin, T treatmentMax, T controlMin, T controlMax) CheckOverlap(
        Matrix<T> x, Vector<int> treatment)
    {
        var propensityScores = EstimatePropensityScores(x);

        T treatmentMin = NumOps.MaxValue;
        T treatmentMax = NumOps.MinValue;
        T controlMin = NumOps.MaxValue;
        T controlMax = NumOps.MinValue;

        for (int i = 0; i < treatment.Length; i++)
        {
            T score = propensityScores[i];

            if (treatment[i] == 1)
            {
                if (NumOps.Compare(score, treatmentMin) < 0)
                    treatmentMin = score;
                if (NumOps.Compare(score, treatmentMax) > 0)
                    treatmentMax = score;
            }
            else
            {
                if (NumOps.Compare(score, controlMin) < 0)
                    controlMin = score;
                if (NumOps.Compare(score, controlMax) > 0)
                    controlMax = score;
            }
        }

        return (treatmentMin, treatmentMax, controlMin, controlMax);
    }

    #endregion

    #region Data Validation

    /// <summary>
    /// Validates causal inference data inputs.
    /// </summary>
    protected void ValidateCausalData(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (x.Rows != treatment.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match number of treatments ({treatment.Length}).");
        }

        if (x.Rows != outcome.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match number of outcomes ({outcome.Length}).");
        }

        // Check treatment is binary
        for (int i = 0; i < treatment.Length; i++)
        {
            if (treatment[i] != 0 && treatment[i] != 1)
            {
                throw new ArgumentException(
                    $"Treatment indicators must be 0 or 1. Found {treatment[i]} at index {i}.");
            }
        }

        // Check that both treatment groups have data
        int numTreated = 0;
        for (int i = 0; i < treatment.Length; i++)
        {
            numTreated += treatment[i];
        }

        if (numTreated == 0)
        {
            throw new ArgumentException("No treated individuals in the data.");
        }

        if (numTreated == treatment.Length)
        {
            throw new ArgumentException("No control individuals in the data.");
        }
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public virtual ModelType GetModelType() => ModelType.None;

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            FeatureCount = NumFeatures,
            Complexity = NumFeatures,
            Description = $"{GetModelType()} causal inference model with {NumFeatures} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "IsFitted", IsFitted }
            }
        };
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    public virtual byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumFeatures", NumFeatures },
            { "IsFitted", IsFitted }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    public virtual void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata?.ModelData is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        IsFitted = modelDataObj["IsFitted"]?.ToObject<bool>() ?? false;
    }

    /// <summary>
    /// Standard model training - fits the causal model.
    /// </summary>
    public virtual void Train(Matrix<T> x, Vector<T> y)
    {
        // Default: treat y as outcome with no treatment effect
        // Subclasses should override with proper causal training
        NumFeatures = x.Columns;
        IsFitted = true;
    }

    /// <summary>
    /// Standard prediction - returns predicted outcomes.
    /// </summary>
    public abstract Vector<T> Predict(Matrix<T> input);

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    public abstract Vector<T> GetParameters();

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    public abstract IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters);

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    public abstract void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Creates a new instance of the same type.
    /// </summary>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance();

    /// <summary>
    /// Gets the indices of features that are actively used in the model.
    /// </summary>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        for (int i = 0; i < NumFeatures; i++)
        {
            yield return i;
        }
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Default: no-op
    }

    /// <summary>
    /// Determines whether a specific feature is used in the model.
    /// </summary>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex));
        }
        return true;
    }

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.One;
        }
        return result;
    }

    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        byte[] serialized = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(serialized);
        return copy;
    }

    /// <summary>
    /// Creates a clone of the model.
    /// </summary>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
    }

    /// <summary>
    /// Computes gradients for the given input and target.
    /// </summary>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Causal models typically don't use standard gradient computation
        return new Vector<T>(ParameterCount);
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Default: no-op
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);
        Deserialize(serializedData);
    }

    /// <summary>
    /// Saves the model's state to a stream.
    /// </summary>
    public virtual void SaveState(Stream stream)
    {
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    public virtual void LoadState(Stream stream)
    {
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        byte[] serializedData = memoryStream.ToArray();
        Deserialize(serializedData);
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported for this causal model.");
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Calculates the standard error using bootstrap resampling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bootstrap is a way to estimate uncertainty by repeatedly
    /// resampling the data with replacement and recalculating the estimate.
    /// </para>
    /// </remarks>
    protected T CalculateBootstrapStandardError(
        Func<Matrix<T>, Vector<int>, Vector<T>, T> estimator,
        Matrix<T> x,
        Vector<int> treatment,
        Vector<T> outcome,
        int numBootstraps = 100)
    {
        var random = Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var estimates = new List<double>();
        int n = x.Rows;

        for (int b = 0; b < numBootstraps; b++)
        {
            // Bootstrap sample
            var indices = new int[n];
            for (int i = 0; i < n; i++)
            {
                indices[i] = random.Next(n);
            }

            // Create bootstrapped data
            var xBoot = new Matrix<T>(n, x.Columns);
            var treatmentBoot = new Vector<int>(n);
            var outcomeBoot = new Vector<T>(n);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                {
                    xBoot[i, j] = x[indices[i], j];
                }
                treatmentBoot[i] = treatment[indices[i]];
                outcomeBoot[i] = outcome[indices[i]];
            }

            T estimate = estimator(xBoot, treatmentBoot, outcomeBoot);
            estimates.Add(NumOps.ToDouble(estimate));
        }

        // Calculate standard deviation of estimates
        double mean = estimates.Average();
        double sumSqDiff = estimates.Sum(e => (e - mean) * (e - mean));
        double variance = sumSqDiff / (numBootstraps - 1);
        double se = Math.Sqrt(variance);

        return NumOps.FromDouble(se);
    }

    /// <summary>
    /// Fits a simple logistic regression for propensity score estimation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Logistic regression predicts the probability of treatment
    /// based on covariates. The coefficients tell us how each feature affects the
    /// likelihood of receiving treatment.
    /// </para>
    /// </remarks>
    protected Vector<T> FitLogisticRegression(
        Matrix<T> x,
        Vector<int> treatment,
        int maxIterations = 100,
        double learningRate = 0.1)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Initialize coefficients to zero
        var coefficients = new Vector<T>(p + 1); // +1 for intercept

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Compute predicted probabilities
            var probs = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                double logit = NumOps.ToDouble(coefficients[0]); // intercept
                for (int j = 0; j < p; j++)
                {
                    logit += NumOps.ToDouble(coefficients[j + 1]) * NumOps.ToDouble(x[i, j]);
                }
                probs[i] = NumOps.FromDouble(Sigmoid(logit));
            }

            // Compute gradient
            var gradient = new Vector<T>(p + 1);
            for (int i = 0; i < n; i++)
            {
                double error = treatment[i] - NumOps.ToDouble(probs[i]);

                gradient[0] = NumOps.Add(gradient[0], NumOps.FromDouble(error)); // intercept
                for (int j = 0; j < p; j++)
                {
                    gradient[j + 1] = NumOps.Add(gradient[j + 1],
                        NumOps.FromDouble(error * NumOps.ToDouble(x[i, j])));
                }
            }

            // Update coefficients
            for (int j = 0; j <= p; j++)
            {
                coefficients[j] = NumOps.Add(coefficients[j],
                    NumOps.FromDouble(learningRate * NumOps.ToDouble(gradient[j]) / n));
            }
        }

        return coefficients;
    }

    /// <summary>
    /// Predicts propensity scores using fitted logistic regression coefficients.
    /// </summary>
    protected Vector<T> PredictPropensityWithCoefficients(Matrix<T> x, Vector<T> coefficients)
    {
        int n = x.Rows;
        int p = x.Columns;
        var probs = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            double logit = NumOps.ToDouble(coefficients[0]); // intercept
            for (int j = 0; j < p; j++)
            {
                logit += NumOps.ToDouble(coefficients[j + 1]) * NumOps.ToDouble(x[i, j]);
            }
            probs[i] = NumOps.FromDouble(Sigmoid(logit));
        }

        return probs;
    }

    /// <summary>
    /// Sigmoid function for logistic regression.
    /// </summary>
    private static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        else
        {
            double exp = Math.Exp(x);
            return exp / (1.0 + exp);
        }
    }

    #endregion
}
