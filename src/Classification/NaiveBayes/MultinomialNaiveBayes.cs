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
///
/// // Prepare word count features (bag-of-words representation)
/// var features = new Matrix&lt;double&gt;(4, 3);
/// features[0, 0] = 5; features[0, 1] = 1; features[0, 2] = 0; // Class 0: high word1
/// features[1, 0] = 3; features[1, 1] = 2; features[1, 2] = 0; // Class 0
/// features[2, 0] = 0; features[2, 1] = 1; features[2, 2] = 4; // Class 1: high word3
/// features[3, 0] = 0; features[3, 1] = 2; features[3, 2] = 3; // Class 1
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1 });
///
/// // Train by learning word frequency distributions per class
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new MultinomialNaiveBayes&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class based on word count likelihoods
/// var newSample = new Matrix&lt;double&gt;(1, 3);
/// newSample[0, 0] = 4; newSample[0, 1] = 1; newSample[0, 2] = 0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("A Comparison of Event Models for Naive Bayes Text Classification", "https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf")]
public partial class MultinomialNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Log of feature probabilities for each class.
    /// Shape: [NumClasses, NumFeatures]
    /// Contains log P(feature | class) with Laplace smoothing.
    /// </summary>
    [Buffer]
    private Matrix<T>? _logFeatureProbs;
    private Vector<T>? _featureMinShift;

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
        // Multinomial Naive Bayes requires non-negative feature values (counts/frequencies).
        // Negative values cause Log(negative) = NaN, silently corrupting the model.
        // Shift features to non-negative if any negatives are present.
        // Per scikit-learn: MultinomialNB handles this by shifting each feature by its minimum.
        var engine = Engine;
        if (x.Rows == 0)
        {
            throw new ArgumentException("Training data must contain at least one sample.", nameof(x));
        }

        // shift_f = max(0, -min_i x[i, f]); adding a zero shift leaves the features unchanged
        var features = Tensor<T>.FromMatrix(x);
        var shift = engine.TensorClampMin(engine.ReduceMax(engine.TensorNegate(features), new[] { 0 }, false), NumOps.Zero);
        _featureMinShift = shift.ToVector();
        features = engine.TensorAdd(features, shift.Reshape(1, x.Columns));

        T alpha = NumOps.FromDouble(Options.Alpha);
        T alphaTimesNumFeatures = NumOps.FromDouble(Options.Alpha * NumFeatures);

        // Sum of all feature values per class via the one-hot membership matrix
        var featureSums = new Tensor<T>(new[] { NumClasses, NumFeatures });
        if (y.Length > 0 && NumClasses > 0)
        {
            featureSums = engine.TensorMatMul(engine.TensorTranspose(ClassMembership(y)), features);
        }

        var totalSums = engine.ReduceSum(featureSums, new[] { 1 }, true);

        // Compute log probabilities with Laplace smoothing
        // P(f|c) = (count(f,c) + alpha) / (total_count(c) + alpha * n_features)
        var probs = engine.TensorDivide(engine.TensorAddScalar(featureSums, alpha),
            engine.TensorAddScalar(totalSums, alphaTimesNumFeatures));
        _logFeatureProbs = engine.TensorLog(probs).ToMatrix();
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
        return SingleSampleLogLikelihood(sample, classIndex);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ComputeLogLikelihoods(Matrix<T> input)
    {
        if (_logFeatureProbs == null)
        {
            throw new InvalidOperationException("Model must be trained before computing log-likelihood.");
        }

        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Tensor<T>(new[] { input.Rows, NumClasses });
        }

        // Apply the same shift used during training (clamped at zero), then sum x_f * log P(f|c)
        var engine = Engine;
        var features = Tensor<T>.FromMatrix(input);
        if (input.Columns != NumFeatures)
        {
            features = features.Slice(1, 0, NumFeatures);
        }

        if (_featureMinShift is not null)
        {
            var shift = FitToLength(_featureMinShift, NumFeatures).AsTensor().Reshape(1, NumFeatures);
            features = engine.TensorAdd(features, shift);
        }

        features = engine.TensorClampMin(features, NumOps.Zero);
        return engine.TensorMatMul(features, engine.TensorTranspose(Tensor<T>.FromMatrix(_logFeatureProbs)));
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["DistributionType"] = "Multinomial";
        return metadata;
    }
}
