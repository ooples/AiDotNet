using AiDotNet.Models.Options;

namespace AiDotNet.Classification.NaiveBayes;

/// <summary>
/// Bernoulli Naive Bayes classifier for binary/boolean features.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Bernoulli Naive Bayes is suitable for classification with binary features
/// (features that are either 0 or 1, present or absent). It models each feature
/// as a Bernoulli distribution.
/// </para>
/// <para>
/// Unlike Multinomial Naive Bayes, Bernoulli NB explicitly models the absence of
/// features, making it suitable for problems where "not having" a feature is
/// informative.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This classifier works best with yes/no, true/false, or present/absent data:
/// - Does the document contain this word? (yes/no)
/// - Is this feature present? (0 or 1)
/// - Does the user have this attribute?
///
/// The key difference from Multinomial NB is that Bernoulli NB cares about
/// absence - it penalizes when a feature that's usually present for a class
/// is absent in the sample.
///
/// Example use cases:
/// - Document classification with binary word presence (not counts)
/// - Spam detection with binary features
/// - Any classification with boolean attributes
/// </para>
/// </remarks>
public class BernoulliNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Log of feature probabilities for presence (P(f=1|c)) for each class.
    /// Shape: [NumClasses, NumFeatures]
    /// </summary>
    private Matrix<T>? _logFeatureProbsPresent;

    /// <summary>
    /// Log of feature probabilities for absence (P(f=0|c) = 1 - P(f=1|c)) for each class.
    /// Shape: [NumClasses, NumFeatures]
    /// </summary>
    private Matrix<T>? _logFeatureProbsAbsent;

    /// <summary>
    /// Binarization threshold for converting continuous features to binary.
    /// </summary>
    private readonly T _binarizeThreshold;

    /// <summary>
    /// Initializes a new instance of the BernoulliNaiveBayes class.
    /// </summary>
    /// <param name="options">Configuration options for the Naive Bayes classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    /// <param name="binarizeThreshold">Threshold for binarizing features. Default is 0.0.</param>
    public BernoulliNaiveBayes(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        double binarizeThreshold = 0.0)
        : base(options, regularization)
    {
        _binarizeThreshold = NumOps.FromDouble(binarizeThreshold);
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    /// <returns>ModelType.BernoulliNaiveBayes</returns>
    protected override ModelType GetModelType() => ModelType.BernoulliNaiveBayes;

    /// <summary>
    /// Computes the log feature probabilities for presence and absence for each class.
    /// </summary>
    /// <param name="x">The input features matrix (will be binarized internally).</param>
    /// <param name="y">The target class labels vector.</param>
    /// <remarks>
    /// <para>
    /// For Bernoulli Naive Bayes, the probability of feature f being present given class c is:
    /// P(f=1|c) = (count of samples with f=1 in class c + alpha) / (count of samples in class c + 2*alpha)
    ///
    /// where alpha is the smoothing parameter.
    /// </para>
    /// </remarks>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        _logFeatureProbsPresent = new Matrix<T>(NumClasses, NumFeatures);
        _logFeatureProbsAbsent = new Matrix<T>(NumClasses, NumFeatures);

        T alpha = NumOps.FromDouble(Options.Alpha);
        T twoAlpha = NumOps.FromDouble(2.0 * Options.Alpha);

        // For each class, compute the feature probabilities
        for (int c = 0; c < NumClasses; c++)
        {
            // Count of samples in this class (already computed in base class)
            int classCount = ClassCounts![c];
            T classSampleCount = NumOps.FromDouble(classCount);

            // Count features present (value > threshold) for each feature
            var featurePresenceCounts = new int[NumFeatures];

            for (int i = 0; i < y.Length; i++)
            {
                if (GetClassIndex(y[i]) == c)
                {
                    for (int f = 0; f < NumFeatures; f++)
                    {
                        // Binarize: count as present if > threshold
                        if (NumOps.Compare(x[i, f], _binarizeThreshold) > 0)
                        {
                            featurePresenceCounts[f]++;
                        }
                    }
                }
            }

            // Compute log probabilities with Laplace smoothing
            // P(f=1|c) = (count + alpha) / (class_count + 2*alpha)
            T denominator = NumOps.Add(classSampleCount, twoAlpha);

            for (int f = 0; f < NumFeatures; f++)
            {
                T numerator = NumOps.Add(NumOps.FromDouble(featurePresenceCounts[f]), alpha);
                T probPresent = NumOps.Divide(numerator, denominator);
                T probAbsent = NumOps.Subtract(NumOps.One, probPresent);

                _logFeatureProbsPresent[c, f] = NumOps.Log(probPresent);
                _logFeatureProbsAbsent[c, f] = NumOps.Log(probAbsent);
            }
        }
    }

    /// <summary>
    /// Computes the log-likelihood of a sample given a class using Bernoulli distribution.
    /// </summary>
    /// <param name="sample">The feature vector for a single sample.</param>
    /// <param name="classIndex">The class index.</param>
    /// <returns>The log-likelihood log P(sample|class).</returns>
    /// <remarks>
    /// <para>
    /// For Bernoulli Naive Bayes, the log-likelihood accounts for both presence and absence:
    /// log P(x|c) = sum over all features of:
    ///   if feature is present: log P(f=1|c)
    ///   if feature is absent: log P(f=0|c)
    /// </para>
    /// </remarks>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        if (_logFeatureProbsPresent == null || _logFeatureProbsAbsent == null)
        {
            throw new InvalidOperationException("Model must be trained before computing log-likelihood.");
        }

        T logLikelihood = NumOps.Zero;

        for (int f = 0; f < NumFeatures; f++)
        {
            // Check if feature is present (binarize)
            bool isPresent = NumOps.Compare(sample[f], _binarizeThreshold) > 0;

            if (isPresent)
            {
                logLikelihood = NumOps.Add(logLikelihood, _logFeatureProbsPresent[classIndex, f]);
            }
            else
            {
                logLikelihood = NumOps.Add(logLikelihood, _logFeatureProbsAbsent[classIndex, f]);
            }
        }

        return logLikelihood;
    }

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    /// <returns>A new BernoulliNaiveBayes instance.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new BernoulliNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors,
            ClassPriors = Options.ClassPriors,
            MinVariance = Options.MinVariance
        }, binarizeThreshold: NumOps.ToDouble(_binarizeThreshold));
    }

    /// <summary>
    /// Creates a deep clone of this model.
    /// </summary>
    /// <returns>A cloned BernoulliNaiveBayes instance.</returns>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new BernoulliNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors,
            ClassPriors = Options.ClassPriors?.ToArray(),
            MinVariance = Options.MinVariance
        }, binarizeThreshold: NumOps.ToDouble(_binarizeThreshold));

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

        if (_logFeatureProbsPresent != null)
        {
            clone._logFeatureProbsPresent = new Matrix<T>(_logFeatureProbsPresent.Rows, _logFeatureProbsPresent.Columns);
            for (int i = 0; i < _logFeatureProbsPresent.Rows; i++)
            {
                for (int j = 0; j < _logFeatureProbsPresent.Columns; j++)
                {
                    clone._logFeatureProbsPresent[i, j] = _logFeatureProbsPresent[i, j];
                }
            }
        }

        if (_logFeatureProbsAbsent != null)
        {
            clone._logFeatureProbsAbsent = new Matrix<T>(_logFeatureProbsAbsent.Rows, _logFeatureProbsAbsent.Columns);
            for (int i = 0; i < _logFeatureProbsAbsent.Rows; i++)
            {
                for (int j = 0; j < _logFeatureProbsAbsent.Columns; j++)
                {
                    clone._logFeatureProbsAbsent[i, j] = _logFeatureProbsAbsent[i, j];
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["DistributionType"] = "Bernoulli";
        metadata.AdditionalInfo["BinarizeThreshold"] = NumOps.ToDouble(_binarizeThreshold);
        return metadata;
    }
}
