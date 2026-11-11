using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.NaiveBayes;

/// <summary>
/// Gaussian Naive Bayes classifier for continuous features.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Gaussian Naive Bayes assumes that the continuous features follow a Gaussian (normal)
/// distribution within each class. It estimates the mean and variance of each feature
/// for each class during training, then uses these to compute the probability density
/// during prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This classifier works well with continuous data (like measurements: height, weight, temperature).
/// It assumes each feature follows a bell-shaped curve (normal distribution) for each class.
///
/// During training, it learns:
/// - The average value of each feature for each class
/// - How spread out (variance) each feature is for each class
///
/// During prediction, it calculates how likely a new data point is under each class's
/// distribution and picks the most likely class.
///
/// Example use cases:
/// - Classifying iris flowers based on petal/sepal measurements
/// - Medical diagnosis based on patient vitals
/// - Weather prediction based on sensor readings
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Gaussian Naive Bayes for continuous feature classification
/// var options = new NaiveBayesOptions&lt;double&gt;();
///
/// // Prepare continuous measurement data: 6 samples with 2 features
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train by estimating per-class Gaussian parameters (mean, variance)
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new GaussianNaiveBayes&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict using Gaussian probability density function
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
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
public partial class GaussianNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Mean values for each feature in each class.
    /// Shape: [NumClasses, NumFeatures]
    /// </summary>
    [Buffer]
    private Matrix<T>? _means;

    /// <summary>
    /// Variance values for each feature in each class.
    /// Shape: [NumClasses, NumFeatures]
    /// </summary>
    [Buffer]
    private Matrix<T>? _variances;

    /// <summary>
    /// Precomputed log(2 * pi) for efficiency in log-likelihood calculation.
    /// </summary>
    private readonly T _log2Pi;

    /// <summary>
    /// Initializes a new instance of the GaussianNaiveBayes class.
    /// </summary>
    /// <param name="options">Configuration options for the Naive Bayes classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public GaussianNaiveBayes(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _log2Pi = NumOps.FromDouble(Math.Log(2.0 * Math.PI));
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    /// <returns>ModelType.GaussianNaiveBayes</returns>

    /// <summary>
    /// Computes the mean and variance of each feature for each class.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels vector.</param>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        var engine = Engine;
        T minVar = NumOps.FromDouble(Options.MinVariance);
        if (NumClasses == 0 || NumFeatures == 0 || y.Length == 0)
        {
            // Zero mean and minimum variance for classes with no samples
            _means = new Matrix<T>(NumClasses, NumFeatures);
            var empty = new Tensor<T>(new[] { NumClasses, NumFeatures });
            engine.TensorFill(empty, minVar);
            _variances = empty.ToMatrix();
            return;
        }

        // Per-class sample counts and feature sums via the one-hot membership matrix
        var membership = ClassMembership(y);
        var membershipT = engine.TensorTranspose(membership);
        var features = Tensor<T>.FromMatrix(x);
        var counts = engine.TensorClampMin(engine.ReduceSum(membership, new[] { 0 }, false), NumOps.One)
            .Reshape(NumClasses, 1);

        // Mean of each feature (zero for classes with no samples)
        var means = engine.TensorDivide(engine.TensorMatMul(membershipT, features), counts);

        // Variance of each feature around its class mean (using n, not n-1, for sklearn compatibility)
        var sampleMeans = engine.TensorGather(means, GetClassIndexTensor(y), 0);
        var centered = engine.TensorSubtract(features, sampleMeans);
        var variances = engine.TensorDivide(
            engine.TensorMatMul(membershipT, engine.TensorMultiply(centered, centered)), counts);

        // Apply minimum variance to prevent division by zero (classes with no samples get exactly the minimum)
        _means = means.ToMatrix();
        _variances = engine.TensorClampMin(variances, minVar).ToMatrix();
    }

    /// <summary>
    /// Computes the log-likelihood of a sample given a class using Gaussian distribution.
    /// </summary>
    /// <param name="sample">The feature vector for a single sample.</param>
    /// <param name="classIndex">The class index.</param>
    /// <returns>The log-likelihood log P(sample|class).</returns>
    /// <remarks>
    /// <para>
    /// The Gaussian log-likelihood for a feature x given class c is:
    /// log P(x|c) = -0.5 * [log(2*pi) + log(variance) + (x - mean)^2 / variance]
    /// </para>
    /// <para>
    /// The total log-likelihood is the sum over all features (assuming independence).
    /// </para>
    /// </remarks>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        return SingleSampleLogLikelihood(sample, classIndex);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ComputeLogLikelihoods(Matrix<T> input)
    {
        if (_means == null || _variances == null)
        {
            throw new InvalidOperationException("Model must be trained before computing log-likelihood.");
        }

        var engine = Engine;
        int n = input.Rows;
        if (n == 0 || NumClasses == 0)
        {
            return new Tensor<T>(new[] { n, NumClasses });
        }

        // log P(x|c) per feature: -0.5 * (log(2*pi) + log(variance) + (x-mean)^2/variance), summed over features
        var means = Tensor<T>.FromMatrix(_means).Reshape(1, NumClasses, NumFeatures);
        var variances = Tensor<T>.FromMatrix(_variances).Reshape(1, NumClasses, NumFeatures);
        var logNormalizer = engine.TensorAddScalar(engine.TensorLog(variances), _log2Pi);
        var features = Tensor<T>.FromMatrix(input);
        if (input.Columns != NumFeatures)
        {
            features = features.Slice(1, 0, NumFeatures);
        }

        T negHalf = NumOps.Negate(NumOps.FromDouble(0.5));
        int batch = Math.Max(1, (1 << 22) / Math.Max(1, NumClasses * NumFeatures));
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < n; start += batch)
        {
            int rows = Math.Min(batch, n - start);
            var block = features.Slice(0, start, start + rows).Reshape(rows, 1, NumFeatures);
            var diff = engine.TensorSubtract(block, means);
            var terms = engine.TensorAdd(engine.TensorDivide(engine.TensorMultiply(diff, diff), variances), logNormalizer);
            blocks.Add(engine.TensorMultiplyScalar(engine.ReduceSum(terms, new[] { 2 }, false), negHalf));
        }

        return blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["MinVariance"] = Options.MinVariance;
        return metadata;
    }
}
