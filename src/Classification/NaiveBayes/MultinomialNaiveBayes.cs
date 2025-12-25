using AiDotNet.Models.Options;

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
    protected override ModelType GetModelType() => ModelType.MultinomialNaiveBayes;

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
}
