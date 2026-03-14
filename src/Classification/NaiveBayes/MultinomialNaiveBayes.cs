using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.NaiveBayes;

/// <summary>
/// Multinomial Naive Bayes classifier for discrete count data.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Multinomial Naive Bayes is suitable for classification with discrete features
/// representing counts or frequencies (e.g., word counts in text classification).
/// It models each feature as a multinomial distribution.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This classifier works best with count data - things you can count, like:
/// - Number of times each word appears in a document (text classification)
/// - Number of each type of event
/// - Frequency of features in categorical data
///
/// It's the go-to algorithm for spam detection and sentiment analysis!
///
/// During training, it learns how often each feature occurs in each class.
/// During prediction, it calculates which class is most likely given the
/// observed feature counts.
///
/// Example use cases:
/// - Spam detection (word counts in emails)
/// - Topic classification of documents
/// - Sentiment analysis (positive/negative word counts)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Multinomial Naive Bayes for text classification with word counts
/// var options = new NaiveBayesOptions&lt;double&gt;();
/// var classifier = new MultinomialNaiveBayes&lt;double&gt;(options);
///
/// // Prepare word count features (bag-of-words representation)
/// var features = Matrix&lt;double&gt;.Build.Dense(4, 3, new double[] {
///     5, 1, 0,  3, 2, 0,  // Class 0: high word1 frequency
///     0, 1, 4,  0, 2, 3 });  // Class 1: high word3 frequency
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1 });
///
/// // Train by learning word frequency distributions per class
/// classifier.Train(features, labels);
///
/// // Predict class based on word count likelihoods
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 3, new double[] { 4, 1, 0 });
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
public class MultinomialNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Log of feature probabilities for each class.
    /// Shape: [NumClasses, NumFeatures]
    /// Contains log P(feature | class) with Laplace smoothing.
    /// </summary>
    private Matrix<T>? _logFeatureProbs;

    /// <summary>
    /// Initializes a new instance of the MultinomialNaiveBayes class.
    /// </summary>
    /// <param name="options">Configuration options for the Naive Bayes classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public MultinomialNaiveBayes(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    /// <returns>ModelType.MultinomialNaiveBayes</returns>

    /// <summary>
    /// Computes the log feature probabilities for each class using Laplace smoothing.
    /// </summary>
    /// <param name="x">The input features matrix (should contain non-negative counts).</param>
    /// <param name="y">The target class labels vector.</param>
    /// <remarks>
    /// <para>
    /// For multinomial Naive Bayes, the probability of feature f given class c is:
    /// P(f|c) = (count(f,c) + alpha) / (sum of all feature counts for c + alpha * n_features)
    ///
    /// where alpha is the smoothing parameter (Laplace smoothing when alpha=1).
    /// </para>
    /// </remarks>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        _logFeatureProbs = new Matrix<T>(NumClasses, NumFeatures);

        T alpha = NumOps.FromDouble(Options.Alpha);
        T alphaTimesNumFeatures = NumOps.FromDouble(Options.Alpha * NumFeatures);

        // For each class, compute the feature probabilities
        for (int c = 0; c < NumClasses; c++)
        {
            // Sum of all feature values for this class
            var featureSums = new Vector<T>(NumFeatures);
            T totalSum = NumOps.Zero;

            // Accumulate feature sums for samples in this class
            for (int i = 0; i < y.Length; i++)
            {
                if (GetClassIndex(y[i]) == c)
                {
                    for (int f = 0; f < NumFeatures; f++)
                    {
                        featureSums[f] = NumOps.Add(featureSums[f], x[i, f]);
                        totalSum = NumOps.Add(totalSum, x[i, f]);
                    }
                }
            }

            // Compute log probabilities with Laplace smoothing
            // P(f|c) = (count(f,c) + alpha) / (total_count(c) + alpha * n_features)
            T denominator = NumOps.Add(totalSum, alphaTimesNumFeatures);

            for (int f = 0; f < NumFeatures; f++)
            {
                T numerator = NumOps.Add(featureSums[f], alpha);
                T prob = NumOps.Divide(numerator, denominator);
                _logFeatureProbs[c, f] = NumOps.Log(prob);
            }
        }
    }

    /// <summary>
    /// Computes the log-likelihood of a sample given a class using multinomial distribution.
    /// </summary>
    /// <param name="sample">The feature vector for a single sample (should be non-negative counts).</param>
    /// <param name="classIndex">The class index.</param>
    /// <returns>The log-likelihood log P(sample|class).</returns>
    /// <remarks>
    /// <para>
    /// For multinomial Naive Bayes, the log-likelihood is:
    /// log P(x|c) = sum over all features of: x_f * log P(f|c)
    ///
    /// where x_f is the count for feature f in the sample.
    /// </para>
    /// </remarks>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        if (_logFeatureProbs == null)
        {
            throw new InvalidOperationException("Model must be trained before computing log-likelihood.");
        }

        T logLikelihood = NumOps.Zero;

        for (int f = 0; f < NumFeatures; f++)
        {
            // Contribution: count * log(probability)
            T contribution = NumOps.Multiply(sample[f], _logFeatureProbs[classIndex, f]);
            logLikelihood = NumOps.Add(logLikelihood, contribution);
        }

        return logLikelihood;
    }

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    /// <returns>A new MultinomialNaiveBayes instance.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new MultinomialNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors,
            ClassPriors = Options.ClassPriors,
            MinVariance = Options.MinVariance
        });
    }

    /// <summary>
    /// Creates a deep clone of this model.
    /// </summary>
    /// <returns>A cloned MultinomialNaiveBayes instance.</returns>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new MultinomialNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors,
            ClassPriors = Options.ClassPriors?.ToArray(),
            MinVariance = Options.MinVariance
        });

        // Copy trained state
        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels != null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (LogPriors != null)
        {
            clone.LogPriors = new Vector<T>(LogPriors.Length);
            for (int i = 0; i < LogPriors.Length; i++)
            {
                clone.LogPriors[i] = LogPriors[i];
            }
        }

        if (ClassCounts != null)
        {
            clone.ClassCounts = ClassCounts.ToArray();
        }

        if (_logFeatureProbs != null)
        {
            clone._logFeatureProbs = new Matrix<T>(_logFeatureProbs.Rows, _logFeatureProbs.Columns);
            for (int i = 0; i < _logFeatureProbs.Rows; i++)
            {
                for (int j = 0; j < _logFeatureProbs.Columns; j++)
                {
                    clone._logFeatureProbs[i, j] = _logFeatureProbs[i, j];
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["DistributionType"] = "Multinomial";
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
            { "ClassCounts", ClassCounts ?? Array.Empty<int>() }
        };

        if (LogPriors is not null)
        {
            var logPriorsArray = new double[LogPriors.Length];
            for (int i = 0; i < LogPriors.Length; i++)
                logPriorsArray[i] = NumOps.ToDouble(LogPriors[i]);
            modelData["LogPriors"] = logPriorsArray;
        }

        if (_logFeatureProbs is not null)
        {
            var featureProbsArray = new double[_logFeatureProbs.Rows * _logFeatureProbs.Columns];
            int idx = 0;
            for (int i = 0; i < _logFeatureProbs.Rows; i++)
                for (int j = 0; j < _logFeatureProbs.Columns; j++)
                    featureProbsArray[idx++] = NumOps.ToDouble(_logFeatureProbs[i, j]);
            modelData["LogFeatureProbs"] = featureProbsArray;
            modelData["LogFeatureProbsRows"] = _logFeatureProbs.Rows;
            modelData["LogFeatureProbsCols"] = _logFeatureProbs.Columns;
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

        var featureProbsToken = modelDataObj["LogFeatureProbs"];
        if (featureProbsToken is not null)
        {
            var featureProbsArray = featureProbsToken.ToObject<double[]>() ?? Array.Empty<double>();
            int rows = modelDataObj["LogFeatureProbsRows"]?.ToObject<int>() ?? 0;
            int cols = modelDataObj["LogFeatureProbsCols"]?.ToObject<int>() ?? 0;
            if (rows > 0 && cols > 0)
            {
                if (featureProbsArray.Length < rows * cols)
                {
                    throw new InvalidOperationException(
                        $"Deserialization failed: LogFeatureProbs array length {featureProbsArray.Length} is less than expected {rows}x{cols}={rows * cols}.");
                }
                _logFeatureProbs = new Matrix<T>(rows, cols);
                int idx = 0;
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        _logFeatureProbs[i, j] = NumOps.FromDouble(featureProbsArray[idx++]);
            }
        }
    }
}
