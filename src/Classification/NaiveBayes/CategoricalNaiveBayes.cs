using AiDotNet.Classification;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.NaiveBayes;

/// <summary>
/// Categorical Naive Bayes classifier for categorical/discrete features.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Categorical Naive Bayes handles features that take on discrete categorical values.
/// Unlike Multinomial NB (which works with counts) or Bernoulli NB (which works with
/// binary features), Categorical NB handles multi-valued categorical features directly.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Use this when your features are categories, like:
/// - Color: Red, Blue, Green
/// - Size: Small, Medium, Large
/// - Weather: Sunny, Rainy, Cloudy
///
/// The classifier computes: P(category_value | class) for each feature.
///
/// For example, predicting if someone will buy an umbrella:
/// - P(Rainy | Will Buy) might be high
/// - P(Sunny | Will Buy) might be low
///
/// Features should be encoded as integers (0, 1, 2, ...) representing categories.
///
/// Use Categorical NB when:
/// - Features are truly categorical (not ordinal)
/// - Features can have more than 2 values (otherwise use Bernoulli)
/// - Features are not counts (otherwise use Multinomial)
/// </para>
/// </remarks>
public class CategoricalNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Number of categories per feature.
    /// </summary>
    private int[]? _numCategories;

    /// <summary>
    /// Category log-probabilities: log P(category|class).
    /// Structure: [class_index][feature_index][category_index]
    /// </summary>
    private Matrix<T>[]? _categoryLogProbs;

    /// <summary>
    /// Initializes a new instance of the CategoricalNaiveBayes class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public CategoricalNaiveBayes(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.CategoricalNaiveBayes;

    /// <summary>
    /// Computes category probabilities for all classes.
    /// </summary>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("ClassLabels must be set before computing class parameters.");
        }

        // Determine number of categories per feature
        _numCategories = new int[NumFeatures];
        for (int j = 0; j < NumFeatures; j++)
        {
            int maxCategory = 0;
            for (int i = 0; i < x.Rows; i++)
            {
                int category = (int)NumOps.ToDouble(x[i, j]);
                if (category > maxCategory)
                {
                    maxCategory = category;
                }
            }
            _numCategories[j] = maxCategory + 1; // Categories are 0-indexed
        }

        // Initialize category log probabilities
        _categoryLogProbs = new Matrix<T>[NumClasses];
        int maxCats = 0;
        for (int j = 0; j < NumFeatures; j++)
        {
            if (_numCategories[j] > maxCats)
            {
                maxCats = _numCategories[j];
            }
        }
        for (int c = 0; c < NumClasses; c++)
        {
            _categoryLogProbs[c] = new Matrix<T>(NumFeatures, maxCats);
        }

        T alpha = NumOps.FromDouble(Options.Alpha);

        // Compute parameters for each class
        for (int classIndex = 0; classIndex < NumClasses; classIndex++)
        {
            T classLabel = ClassLabels[classIndex];

            // Count samples in this class
            int classCount = 0;
            for (int i = 0; i < y.Length; i++)
            {
                if (NumOps.Compare(y[i], classLabel) == 0)
                {
                    classCount++;
                }
            }

            T classCountT = NumOps.FromDouble(classCount);

            // For each feature, count category occurrences
            for (int j = 0; j < NumFeatures; j++)
            {
                int numCats = _numCategories[j];
                var categoryCounts = new int[numCats];

                for (int i = 0; i < x.Rows; i++)
                {
                    if (NumOps.Compare(y[i], classLabel) == 0)
                    {
                        int category = (int)NumOps.ToDouble(x[i, j]);
                        if (category >= 0 && category < numCats)
                        {
                            categoryCounts[category]++;
                        }
                    }
                }

                // Compute log probability for each category with Laplace smoothing
                // P(category | class) = (count + alpha) / (class_count + alpha * num_categories)
                T denominator = NumOps.Add(classCountT, NumOps.Multiply(alpha, NumOps.FromDouble(numCats)));

                for (int k = 0; k < numCats; k++)
                {
                    T numerator = NumOps.Add(NumOps.FromDouble(categoryCounts[k]), alpha);
                    _categoryLogProbs[classIndex][j, k] = NumOps.Log(NumOps.Divide(numerator, denominator));
                }
            }
        }
    }

    /// <summary>
    /// Computes the log-likelihood for a sample given a class.
    /// </summary>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        if (_numCategories is null || _categoryLogProbs is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        T logLikelihood = NumOps.Zero;

        for (int j = 0; j < NumFeatures; j++)
        {
            int category = (int)NumOps.ToDouble(sample[j]);
            if (category >= 0 && category < _numCategories[j])
            {
                logLikelihood = NumOps.Add(logLikelihood, _categoryLogProbs[classIndex][j, category]);
            }
            // Unknown categories contribute 0 (log(1) = 0)
        }

        return logLikelihood;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new CategoricalNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new CategoricalNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors
        });

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

        if (_numCategories is not null)
        {
            clone._numCategories = new int[_numCategories.Length];
            Array.Copy(_numCategories, clone._numCategories, _numCategories.Length);
        }

        if (_categoryLogProbs is not null)
        {
            clone._categoryLogProbs = new Matrix<T>[_categoryLogProbs.Length];
            for (int c = 0; c < _categoryLogProbs.Length; c++)
            {
                var src = _categoryLogProbs[c];
                clone._categoryLogProbs[c] = new Matrix<T>(src.Rows, src.Columns);
                for (int i = 0; i < src.Rows; i++)
                {
                    for (int j = 0; j < src.Columns; j++)
                    {
                        clone._categoryLogProbs[c][i, j] = src[i, j];
                    }
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        if (_numCategories is not null)
        {
            metadata.AdditionalInfo["TotalCategories"] = _numCategories.Sum();
            metadata.AdditionalInfo["MaxCategoriesPerFeature"] = _numCategories.Max();
        }
        return metadata;
    }
}
