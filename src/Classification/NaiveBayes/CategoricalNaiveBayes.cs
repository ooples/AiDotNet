using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
/// <example>
/// <code>
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for categoricalnaivebayes classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new CategoricalNaiveBayes&lt;double&gt;(
///         new NaiveBayesOptions&lt;double&gt;()));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("A Comparison of Event Models for Naive Bayes Text Classification", "https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf")]
public partial class CategoricalNaiveBayes<T> : NaiveBayesBase<T>
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

    /// <summary>
    /// Computes category probabilities for all classes.
    /// </summary>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("ClassLabels must be set before computing class parameters.");
        }

        var engine = Engine;
        var features = Tensor<T>.FromMatrix(x);
        if (x.Columns != NumFeatures)
        {
            features = features.Slice(1, 0, NumFeatures);
        }

        // Determine number of categories per feature: floor(max value) + 1 (categories are 0-indexed, at least one)
        int largestCategory = 0;
        if (x.Rows > 0 && NumFeatures > 0)
        {
            largestCategory = Math.Max(0, (int)NumOps.ToDouble(engine.TensorMaxValue(features)));
            var columnMax = engine.ReduceMax(features, new[] { 0 }, false);
            _numCategories = engine.TensorAddScalar(
                engine.TensorClampMin(CategoryIndices(columnMax, largestCategory), 0), 1).ToArray();
        }
        else
        {
            _numCategories = Enumerable.Repeat(1, NumFeatures).ToArray();
        }

        int maxCats = NumFeatures == 0 ? 0 : _numCategories.Max();
        int width = maxCats + 1; // extra slot collects invalid (negative) categories

        // Count category occurrences per class and feature with one scatter-add
        T alpha = NumOps.FromDouble(Options.Alpha);
        var counts = new Tensor<T>(new[] { NumClasses * NumFeatures * width });
        if (x.Rows > 0 && NumFeatures > 0 && NumClasses > 0)
        {
            var choices = CategoryChoices(features, largestCategory, maxCats);
            var classOffsets = engine.TensorMultiplyScalar(GetClassIndexTensor(y), NumFeatures * width).Reshape(x.Rows, 1);
            var featureOffsets = engine.TensorMultiplyScalar(engine.TensorArange<int>(0, NumFeatures), width).Reshape(1, NumFeatures);
            var codes = engine.TensorAdd(engine.TensorAdd(choices, classOffsets), featureOffsets).Reshape(x.Rows * NumFeatures);
            var ones = new Tensor<T>(new[] { x.Rows * NumFeatures });
            engine.TensorFill(ones, NumOps.One);
            counts = engine.TensorScatterAdd(counts, codes, ones, 0);
        }

        // P(category | class) = (count + alpha) / (class_count + alpha * num_categories); unused slots stay 0
        var categoryCounts = counts.Reshape(NumClasses, NumFeatures, width).Slice(2, 0, maxCats);
        var classCounts = ClassIndicesToValues(new Tensor<int>(ClassCounts ?? new int[NumClasses], new[] { NumClasses }), y.Length + 1);
        var numCats = ClassIndicesToValues(new Tensor<int>(_numCategories, new[] { NumFeatures }), maxCats + 1);
        var denominators = engine.TensorAdd(
            classCounts.Length == 0 ? new Tensor<T>(new[] { 0, 1, 1 }) : classCounts.AsTensor().Reshape(NumClasses, 1, 1),
            numCats.Length == 0
                ? new Tensor<T>(new[] { 1, 0, 1 })
                : engine.TensorMultiplyScalar(numCats.AsTensor(), alpha).Reshape(1, NumFeatures, 1));
        var logProbs = engine.TensorLog(engine.TensorDivide(engine.TensorAddScalar(categoryCounts, alpha), denominators));
        var zeros = new Tensor<T>(new[] { NumClasses, NumFeatures, maxCats });
        var used = engine.TensorAdd(zeros, UsedCategorySlots(maxCats).Reshape(1, NumFeatures, maxCats));
        var table = engine.TensorWhere(used, logProbs, zeros);

        // Category log probabilities per class: [NumFeatures, maxCats]
        _categoryLogProbs = new Matrix<T>[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            _categoryLogProbs[c] = table.Slice(0, c, c + 1).Reshape(NumFeatures, maxCats).ToMatrix();
        }
    }

    /// <summary>
    /// floor(value) for values in [0, limit), -1 below zero and limit at or above limit.
    /// </summary>
    private Tensor<int> CategoryIndices(Tensor<T> values, int limit)
    {
        var boundaries = Engine.TensorArange<T>(0, limit + 1);
        return Engine.TensorSubtractScalar(Engine.TensorSearchSorted(boundaries, values, true), 1);
    }

    /// <summary>
    /// The table slot of every value: its category, or <paramref name="invalidSlot"/> for categories that are
    /// negative or not below <paramref name="invalidSlot"/>.
    /// </summary>
    private Tensor<int> CategoryChoices(Tensor<T> values, int limit, int invalidSlot)
    {
        var engine = Engine;
        var categories = engine.TensorClampMax(CategoryIndices(values, Math.Max(limit, invalidSlot)), invalidSlot);
        var negative = engine.TensorLessThan(categories, 0);
        return engine.TensorAdd(categories, engine.TensorMultiplyScalar(negative, invalidSlot + 1));
    }

    /// <summary>
    /// [NumFeatures, maxCats] 1 where category k exists for feature j (k &lt; num_categories[j]), otherwise 0.
    /// </summary>
    private Tensor<T> UsedCategorySlots(int maxCats)
    {
        var flags = new int[NumFeatures * maxCats];
        if (_numCategories is not null)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                for (int k = 0; k < _numCategories[j] && k < maxCats; k++)
                {
                    flags[j * maxCats + k] = 1;
                }
            }
        }

        return Engine.TensorGather(Engine.TensorArange<T>(0, 2), new Tensor<int>(flags, new[] { flags.Length }), 0)
            .Reshape(NumFeatures, maxCats);
    }

    /// <summary>
    /// Computes the log-likelihood for a sample given a class.
    /// </summary>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        return SingleSampleLogLikelihood(sample, classIndex);
    }

    /// <inheritdoc/>
    private protected override Tensor<T> ComputeLogLikelihoods(Matrix<T> input)
    {
        if (_numCategories is null || _categoryLogProbs is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0 || NumClasses == 0 || NumFeatures == 0)
        {
            return new Tensor<T>(new[] { input.Rows, NumClasses });
        }

        // Sum the log probability of each feature's category; unknown categories contribute 0 (log(1) = 0)
        var engine = Engine;
        var features = Tensor<T>.FromMatrix(input);
        if (input.Columns != NumFeatures)
        {
            features = features.Slice(1, 0, NumFeatures);
        }

        int maxCats = _categoryLogProbs[0].Columns;
        var perClass = new Tensor<T>[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            perClass[c] = Tensor<T>.FromMatrix(_categoryLogProbs[c]);
        }

        // Slot maxCats is zero and absorbs negative or too-large categories
        var table = engine.TensorConcatenate(new[]
        {
            engine.TensorStack(perClass, 0),
            new Tensor<T>(new[] { NumClasses, NumFeatures, 1 })
        }, 2);
        return SumSelectedLogProbabilities(CategoryChoices(features, maxCats, maxCats), table, maxCats + 1);
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

    private void SerializeMatrix(Dictionary<string, object> data, string name, Matrix<T>? matrix)
    {
        SerializeMatrixValues(data, name, matrix);
    }

    private Matrix<T>? DeserializeMatrix(JObject obj, string name)
    {
        return DeserializeMatrixValues(obj, name);
    }
}
