using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
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
/// <example>
/// <code>
/// // Create Complement NB for imbalanced text classification
/// var options = new NaiveBayesOptions&lt;double&gt;();
/// var classifier = new ComplementNaiveBayes&lt;double&gt;(options);
///
/// // Prepare word count features (term-frequency vectors)
/// var features = new Matrix&lt;double&gt;(4, 3);
/// features[0, 0] = 3; features[0, 1] = 0; features[0, 2] = 1; // Class 0
/// features[1, 0] = 2; features[1, 1] = 0; features[1, 2] = 2; // Class 0
/// features[2, 0] = 0; features[2, 1] = 3; features[2, 2] = 1; // Class 1
/// features[3, 0] = 0; features[3, 1] = 2; features[3, 2] = 2; // Class 1
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1 });
///
/// // Train using complement class statistics to handle imbalance
/// classifier.Train(features, labels);
///
/// // Predict class using complement likelihood ratio
/// var newSample = new Matrix&lt;double&gt;(1, 3);
/// newSample[0, 0] = 3; newSample[0, 1] = 0; newSample[0, 2] = 1;
/// var prediction = classifier.Predict(newSample);
/// Console.WriteLine($"Predicted class: {prediction[0]}");
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Tackling the Poor Assumptions of Naive Bayes Text Classifiers", "https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf", Year = 2003, Authors = "Jason D. Rennie, Lawrence Shih, Jaime Teevan, David R. Karger")]
public class ComplementNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Complement feature log-probabilities: log P(feature|NOT class).
    /// </summary>
    private Matrix<T>? _complementLogProbs;

    /// <summary>
    /// Whether to normalize feature weights.
    /// </summary>
    private bool _normalize;

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

    /// <summary>
    /// Computes class-specific parameters for Complement Naive Bayes.
    /// </summary>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("ClassLabels must be set before computing class parameters.");
        }

        // Complement Naive Bayes requires non-negative feature values (counts/frequencies).
        // Negative values cause Log(negative) = NaN, silently corrupting the model.
        for (int i = 0; i < x.Rows; i++)
        {
            for (int f = 0; f < x.Columns; f++)
            {
                if (NumOps.LessThan(x[i, f], NumOps.Zero))
                {
                    throw new ArgumentException(
                        $"ComplementNaiveBayes requires non-negative feature values, " +
                        $"but found {NumOps.ToDouble(x[i, f]):F4} at row {i}, column {f}. " +
                        "Use GaussianNaiveBayes for continuous features that may be negative.",
                        nameof(x));
                }
            }
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

        // Restore _normalize
        var normalizeToken = modelDataObj["Normalize"];
        if (normalizeToken is not null)
            _normalize = normalizeToken.ToObject<bool>();

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
        if (arr.Length < rows * cols)
        {
            throw new InvalidOperationException(
                $"Deserialization failed: {name} array length {arr.Length} is less than expected {rows}x{cols}={rows * cols}.");
        }
        var matrix = new Matrix<T>(rows, cols);
        int idx = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = NumOps.FromDouble(arr[idx++]);
        return matrix;
    }
}
