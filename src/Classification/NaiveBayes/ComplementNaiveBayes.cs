using System.Text;
using AiDotNet.Classification;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.NaiveBayes;

/// <summary>
/// Complement Naive Bayes classifier designed for imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Complement Naive Bayes (CNB) addresses some of the drawbacks of the standard
/// Multinomial Naive Bayes, particularly on imbalanced datasets. Instead of
/// computing P(feature|class), it computes P(feature|NOT class).
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of it this way: instead of asking "how likely is this word in spam?",
/// CNB asks "how unlikely is this word in NOT-spam?"
///
/// This helps when:
/// - One class has many more examples than others
/// - Features are not uniformly distributed across classes
/// - Standard Multinomial NB is biased toward the majority class
///
/// Example: In text classification with 95% non-spam and 5% spam,
/// standard NB might always predict non-spam. CNB corrects this.
///
/// CNB is particularly effective for:
/// - Text classification with imbalanced classes
/// - Sentiment analysis
/// - Topic categorization
/// </para>
/// </remarks>
public class ComplementNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Complement feature log-probabilities: log P(feature|NOT class).
    /// </summary>
    private Matrix<T>? _complementLogProbs;

    /// <summary>
    /// Whether to normalize feature weights.
    /// </summary>
    private readonly bool _normalize;

    /// <summary>
    /// Initializes a new instance of the ComplementNaiveBayes class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    /// <param name="normalize">Whether to normalize feature weights (default true).</param>
    public ComplementNaiveBayes(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        bool normalize = true)
        : base(options, regularization)
    {
        _normalize = normalize;
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.ComplementNaiveBayes;

    /// <summary>
    /// Computes class-specific parameters for Complement Naive Bayes.
    /// </summary>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("ClassLabels must be set before computing class parameters.");
        }

        // Initialize complement log probabilities
        _complementLogProbs = new Matrix<T>(NumClasses, NumFeatures);

        T alpha = NumOps.FromDouble(Options.Alpha);

        // Compute complement parameters for each class
        for (int classIndex = 0; classIndex < NumClasses; classIndex++)
        {
            T classLabel = ClassLabels[classIndex];

            // For CNB, we compute feature counts for all OTHER classes
            var complementCounts = new Vector<T>(NumFeatures);
            T complementTotal = NumOps.Zero;

            for (int i = 0; i < x.Rows; i++)
            {
                // Include this sample if it's NOT the current class
                if (NumOps.Compare(y[i], classLabel) != 0)
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        complementCounts[j] = NumOps.Add(complementCounts[j], x[i, j]);
                        complementTotal = NumOps.Add(complementTotal, x[i, j]);
                    }
                }
            }

            // Compute log probabilities with smoothing
            // log P(feature|NOT class) = log((count + alpha) / (total + alpha * n_features))
            T denominator = NumOps.Add(complementTotal, NumOps.Multiply(alpha, NumOps.FromDouble(NumFeatures)));

            for (int j = 0; j < NumFeatures; j++)
            {
                T numerator = NumOps.Add(complementCounts[j], alpha);
                _complementLogProbs[classIndex, j] = NumOps.Log(NumOps.Divide(numerator, denominator));
            }

            // If normalizing, divide by sum of absolute values
            if (_normalize)
            {
                T sumAbs = NumOps.Zero;
                for (int j = 0; j < NumFeatures; j++)
                {
                    sumAbs = NumOps.Add(sumAbs, NumOps.Abs(_complementLogProbs[classIndex, j]));
                }

                if (NumOps.Compare(sumAbs, NumOps.Zero) > 0)
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        _complementLogProbs[classIndex, j] = NumOps.Divide(_complementLogProbs[classIndex, j], sumAbs);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes the log-likelihood for a sample given a class.
    /// </summary>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        if (_complementLogProbs is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // For CNB, we compute NEGATIVE sum of (feature * complement_weight)
        // because we want to minimize the complement probability
        T logLikelihood = NumOps.Zero;

        for (int j = 0; j < NumFeatures; j++)
        {
            // Negate because we're using complement weights
            T contribution = NumOps.Multiply(sample[j], _complementLogProbs[classIndex, j]);
            logLikelihood = NumOps.Subtract(logLikelihood, contribution);
        }

        return logLikelihood;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new ComplementNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors
        }, null, _normalize);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new ComplementNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors
        }, null, _normalize);

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (LogPriors is not null)
        {
            clone.LogPriors = new Vector<T>(LogPriors.Length);
            for (int i = 0; i < LogPriors.Length; i++)
            {
                clone.LogPriors[i] = LogPriors[i];
            }
        }

        if (_complementLogProbs is not null)
        {
            clone._complementLogProbs = new Matrix<T>(_complementLogProbs.Rows, _complementLogProbs.Columns);
            for (int i = 0; i < _complementLogProbs.Rows; i++)
            {
                for (int j = 0; j < _complementLogProbs.Columns; j++)
                {
                    clone._complementLogProbs[i, j] = _complementLogProbs[i, j];
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Normalize"] = _normalize;
        return metadata;
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "RegularizationOptions", Regularization.GetOptions() },
            { "Normalize", _normalize }
        };

        if (ClassCounts is not null)
            modelData["ClassCounts"] = ClassCounts;

        if (LogPriors is not null)
        {
            var logPriorsArray = new double[LogPriors.Length];
            for (int i = 0; i < LogPriors.Length; i++)
                logPriorsArray[i] = NumOps.ToDouble(LogPriors[i]);
            modelData["LogPriors"] = logPriorsArray;
        }

        SerializeMatrix(modelData, "ComplementLogProbs", _complementLogProbs);

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

        var classCountsToken = modelDataObj["ClassCounts"];
        if (classCountsToken is not null)
            ClassCounts = classCountsToken.ToObject<int[]>();

        var logPriorsToken = modelDataObj["LogPriors"];
        if (logPriorsToken is not null)
        {
            var logPriorsArray = logPriorsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (logPriorsArray.Length > 0)
            {
                LogPriors = new Vector<T>(logPriorsArray.Length);
                for (int i = 0; i < logPriorsArray.Length; i++)
                    LogPriors[i] = NumOps.FromDouble(logPriorsArray[i]);
            }
        }

        _complementLogProbs = DeserializeMatrix(modelDataObj, "ComplementLogProbs");
    }

    private void SerializeMatrix(Dictionary<string, object> data, string name, Matrix<T>? matrix)
    {
        if (matrix is null) return;
        var arr = new double[matrix.Rows * matrix.Columns];
        int idx = 0;
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                arr[idx++] = NumOps.ToDouble(matrix[i, j]);
        data[name] = arr;
        data[$"{name}Rows"] = matrix.Rows;
        data[$"{name}Cols"] = matrix.Columns;
    }

    private Matrix<T>? DeserializeMatrix(JObject obj, string name)
    {
        var token = obj[name];
        if (token is null) return null;
        var arr = token.ToObject<double[]>() ?? Array.Empty<double>();
        int rows = obj[$"{name}Rows"]?.ToObject<int>() ?? 0;
        int cols = obj[$"{name}Cols"]?.ToObject<int>() ?? 0;
        if (rows <= 0 || cols <= 0) return null;
        var matrix = new Matrix<T>(rows, cols);
        int idx = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = NumOps.FromDouble(arr[idx++]);
        return matrix;
    }
}
