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
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ComplementNaiveBayes&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class using complement likelihood ratio
/// var newSample = new Matrix&lt;double&gt;(1, 3);
/// newSample[0, 0] = 3; newSample[0, 1] = 0; newSample[0, 2] = 1;
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
[ResearchPaper("Tackling the Poor Assumptions of Naive Bayes Text Classifiers", "https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf", Year = 2003, Authors = "Jason D. Rennie, Lawrence Shih, Jaime Teevan, David R. Karger")]
public partial class ComplementNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Complement feature log-probabilities: log P(feature|NOT class).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _complementLogProbs;
    private Vector<T>? _featureMinShift;

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

        if (x.Rows == 0)
        {
            throw new ArgumentException("Training data must contain at least one sample.", nameof(x));
        }

        // Shift features to non-negative if any negatives are present: shift_f = max(0, -min_i x[i, f])
        var engine = Engine;
        var features = Tensor<T>.FromMatrix(x);
        var shift = engine.TensorClampMin(engine.ReduceMax(engine.TensorNegate(features), new[] { 0 }, false), NumOps.Zero);
        _featureMinShift = shift.ToVector();

        // CNB expects non-negative counts per Rennie et al. (2003)
        features = engine.TensorClampMin(engine.TensorAdd(features, shift.Reshape(1, x.Columns)), NumOps.Zero);
        if (x.Columns != NumFeatures)
        {
            features = features.Slice(1, 0, NumFeatures);
        }

        T alpha = NumOps.FromDouble(Options.Alpha);

        // For CNB, feature counts over all OTHER classes = all counts - this class's counts
        var totalCounts = engine.ReduceSum(features, new[] { 0 }, true);
        var classCounts = engine.TensorMatMul(engine.TensorTranspose(ClassMembership(y)), features);
        var complementCounts = engine.TensorSubtract(totalCounts, classCounts);
        var complementTotals = engine.ReduceSum(complementCounts, new[] { 1 }, true);

        // Compute log probabilities with smoothing
        // log P(feature|NOT class) = log((count + alpha) / (total + alpha * n_features))
        var denominators = engine.TensorAddScalar(complementTotals, NumOps.Multiply(alpha, NumOps.FromDouble(NumFeatures)));
        var logProbs = engine.TensorLog(engine.TensorDivide(engine.TensorAddScalar(complementCounts, alpha), denominators));

        // If normalizing, divide by sum of absolute values (rows whose sum is zero are left as is)
        if (_normalize)
        {
            var sumAbs = engine.ReduceSum(engine.TensorAbs(logProbs), new[] { 1 }, true);
            var positive = engine.TensorGreaterThan(sumAbs, NumOps.Zero);
            var ones = engine.TensorAddScalar(engine.TensorMultiplyScalar(sumAbs, NumOps.Zero), NumOps.One);
            logProbs = engine.TensorDivide(logProbs, engine.TensorWhere(positive, sumAbs, ones));
        }

        _complementLogProbs = logProbs.ToMatrix();
    }

    /// <summary>
    /// Computes the log-likelihood for a sample given a class.
    /// </summary>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        return SingleSampleLogLikelihood(sample, classIndex);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ComputeLogLikelihoods(Matrix<T> input)
    {
        if (_complementLogProbs is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Tensor<T>(new[] { input.Rows, NumClasses });
        }

        // For CNB, we compute NEGATIVE sum of (feature * complement_weight)
        // because we want to minimize the complement probability.
        var engine = Engine;
        var features = Tensor<T>.FromMatrix(input);
        if (input.Columns != NumFeatures)
        {
            features = features.Slice(1, 0, NumFeatures);
        }

        // Apply the same shift used during training, clamped at zero
        if (_featureMinShift is not null)
        {
            features = engine.TensorAdd(features, FitToLength(_featureMinShift, NumFeatures).AsTensor().Reshape(1, NumFeatures));
        }

        features = engine.TensorClampMin(features, NumOps.Zero);
        var weighted = engine.TensorMatMul(features, engine.TensorTranspose(Tensor<T>.FromMatrix(_complementLogProbs)));
        return engine.TensorNegate(weighted);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Normalize"] = _normalize;
        return metadata;
    }

    private void SerializeMatrix(Dictionary<string, object> data, string name, Matrix<T>? matrix)
    {
        SerializeMatrixValues(data, name, matrix);
    }

    private Matrix<T>? DeserializeMatrix(JObject obj, string name)
    {
        return DeserializeMatrixValues(obj, name);
    }
}
