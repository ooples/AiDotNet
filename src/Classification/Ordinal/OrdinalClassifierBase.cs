using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Classification.Ordinal;

/// <summary>
/// Base class for ordinal classification models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Ordinal classification (also called ordinal regression) is used when
/// your target variable has a natural order. Examples include:
/// <list type="bullet">
/// <item>Star ratings (1, 2, 3, 4, 5 stars)</item>
/// <item>Education levels (High School, Bachelor's, Master's, PhD)</item>
/// <item>Disease severity (Mild, Moderate, Severe)</item>
/// <item>Likert scale responses (Strongly Disagree to Strongly Agree)</item>
/// </list>
/// </para>
///
/// <para>Unlike regular classification where all misclassifications are equally bad,
/// in ordinal classification being "close" to the true class is better than being far away.
/// Predicting 4 stars when the truth is 5 stars is a smaller error than predicting 1 star.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract partial class OrdinalClassifierBase<T> : ClassifierBase<T>, IOrdinalClassifier<T>
{
    /// <summary>
    /// The learned thresholds that separate ordinal classes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In ordinal classification, thresholds are like dividing lines
    /// on a number line. If you have 5 classes (1-5), you need 4 thresholds to separate them:
    /// threshold 1 separates class 1 from 2, threshold 2 separates class 2 from 3, etc.</para>
    /// </remarks>
    protected Vector<T>? _thresholds;

    /// <summary>
    /// Gets the ordered class labels.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the possible class values in their natural order.
    /// For a 5-star rating system, this would be [1, 2, 3, 4, 5].</para>
    /// </remarks>
    public Vector<T>? OrderedClasses => ClassLabels;

    /// <summary>
    /// Initializes a new instance of OrdinalClassifierBase.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <param name="lossFunction">Loss function for gradient computation.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the ordinal classifier with
    /// optional settings. If you don't provide settings, sensible defaults are used.</para>
    /// </remarks>
    protected OrdinalClassifierBase(
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        ILossFunction<T>? lossFunction = null)
        : base(options, regularization, lossFunction)
    {
        // Set task type to ordinal
        TaskType = ClassificationTaskType.Ordinal;
    }

    /// <summary>
    /// Predicts cumulative probabilities P(Y ≤ k) for each class threshold.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <returns>Cumulative probability matrix [n_samples, n_classes-1].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cumulative probabilities answer questions like:
    /// "What's the probability that the rating is 3 or less?"
    ///
    /// For 5 classes (1-5), this returns 4 columns:
    /// - Column 0: P(Y ≤ 1)
    /// - Column 1: P(Y ≤ 2)
    /// - Column 2: P(Y ≤ 3)
    /// - Column 3: P(Y ≤ 4)
    ///
    /// Note: P(Y ≤ 5) = 1.0 always, so it's not included.
    ///
    /// From cumulative probabilities, you can compute class probabilities:
    /// P(Y = 3) = P(Y ≤ 3) - P(Y ≤ 2)
    /// </para>
    /// </remarks>
    public abstract Matrix<T> PredictCumulativeProbabilities(Matrix<T> features);

    /// <summary>
    /// Predicts class probabilities for each ordinal class.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <returns>Class probability matrix [n_samples, n_classes].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method returns the probability of each class
    /// for each sample. The probabilities for each sample sum to 1.0.
    ///
    /// For example, for a 5-star rating prediction:
    /// - Column 0: P(Y = 1 star)
    /// - Column 1: P(Y = 2 stars)
    /// - Column 2: P(Y = 3 stars)
    /// - Column 3: P(Y = 4 stars)
    /// - Column 4: P(Y = 5 stars)
    /// </para>
    /// </remarks>
    public virtual Matrix<T> PredictProbabilities(Matrix<T> features)
    {
        var cumProbs = PredictCumulativeProbabilities(features);
        int n = features.Rows;
        int k = NumClasses;
        if (n == 0 || k == 0)
        {
            return new Matrix<T>(n, k);
        }

        // P(Y = j) = P(Y <= j) - P(Y <= j-1), with P(Y <= -1) = 0 and P(Y <= k-1) = 1
        var engine = Engine;
        var cumulative = Tensor<T>.FromMatrix(cumProbs);
        if (cumProbs.Columns != k - 1)
        {
            cumulative = cumulative.Slice(1, 0, k - 1);
        }

        var ones = new Tensor<T>(new[] { n, 1 });
        engine.TensorFill(ones, NumOps.One);
        var zeros = new Tensor<T>(new[] { n, 1 });
        var upper = engine.TensorConcatenate(new[] { cumulative, ones }, 1);
        var lower = engine.TensorConcatenate(new[] { zeros, cumulative }, 1);
        return engine.TensorSubtract(upper, lower).ToMatrix();
    }

    /// <summary>
    /// Extracts ordered class labels from the training labels.
    /// </summary>
    /// <param name="labels">The training labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds all unique class values in your
    /// training data and sorts them in ascending order. This establishes the natural
    /// ordering for the ordinal classification.</para>
    /// </remarks>
    protected void ExtractOrderedClasses(Vector<T> labels)
    {
        // Unique labels in ascending order
        ClassLabels = ExtractClassLabels(labels);
        NumClasses = ClassLabels.Length;
    }

    /// <summary>
    /// Converts a class label to its ordinal index.
    /// </summary>
    /// <param name="label">The class label to convert.</param>
    /// <returns>The ordinal index (0, 1, 2, ...) corresponding to the label.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This maps a class value (like 3 stars) to its
    /// position in the ordered class list (index 2 for classes [1,2,3,4,5]).
    /// If the exact label isn't found, it returns the closest class.</para>
    /// </remarks>
    protected int GetClassIndex(T label)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        return GetOrdinalClassIndices(Vector<T>.Wrap(new[] { label })).ToArray()[0];
    }

    /// <summary>
    /// The class index of every label: the matching class, or otherwise the closest class (the first one on ties).
    /// </summary>
    /// <param name="labels">The labels to map.</param>
    /// <returns>One class index per label.</returns>
    private protected Tensor<int> GetOrdinalClassIndices(Vector<T> labels)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (labels.Length == 0 || ClassLabels.Length == 0)
        {
            return new Tensor<int>(new[] { labels.Length });
        }

        // An exact match has distance 0, so the closest class is the matching one when it exists
        var engine = Engine;
        var distances = engine.TensorAbs(engine.TensorSubtract(
            labels.AsTensor().Reshape(labels.Length, 1), ClassLabels.AsTensor().Reshape(1, ClassLabels.Length)));
        return engine.TensorArgMin(distances, 1);
    }

    /// <summary>
    /// Computes the sigmoid function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The sigmoid of x, which is 1 / (1 + exp(-x)).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The sigmoid function "squashes" any real number
    /// into the range (0, 1). This makes it perfect for representing probabilities.
    /// Very negative inputs give values close to 0, very positive inputs give values
    /// close to 1, and 0 gives exactly 0.5.</para>
    /// </remarks>
    protected T Sigmoid(T x)
    {
        if (NumOps.GreaterThan(x, NumOps.FromDouble(20))) return NumOps.One;
        if (NumOps.LessThan(x, NumOps.FromDouble(-20))) return NumOps.Zero;
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(x))));
    }

    /// <summary>
    /// Element-wise <see cref="Sigmoid(T)"/>: 1 above 20, 0 below -20, otherwise 1 / (1 + exp(-x)).
    /// </summary>
    /// <param name="x">The inputs.</param>
    /// <returns>The sigmoid values.</returns>
    private protected Tensor<T> SigmoidSaturated(Tensor<T> x)
    {
        var engine = Engine;
        var ones = new Tensor<T>(x.Shape.ToArray());
        engine.TensorFill(ones, NumOps.One);
        var zeros = new Tensor<T>(x.Shape.ToArray());
        var sigmoid = engine.TensorSigmoid(x);
        var high = engine.TensorGreaterThan(x, NumOps.FromDouble(20));
        var low = engine.TensorLessThan(x, NumOps.FromDouble(-20));
        return engine.TensorWhere(high, ones, engine.TensorWhere(low, zeros, sigmoid));
    }

    /// <summary>
    /// Infers the task type for ordinal classification.
    /// </summary>
    /// <param name="y">The target labels.</param>
    /// <returns>Always returns ClassificationTaskType.Ordinal.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Since this is an ordinal classifier, we always
    /// return the Ordinal task type regardless of the number of classes.</para>
    /// </remarks>
    /// <summary>
    /// Feature importance as |coefficient| divided by the sum of absolute coefficients.
    /// </summary>
    /// <param name="coefficients">The coefficients.</param>
    /// <returns>Importance per feature name.</returns>
    private protected Dictionary<string, T> NormalizedCoefficientImportance(Vector<T> coefficients)
    {
        var importance = new Dictionary<string, T>();
        if (coefficients.Length == 0)
        {
            return importance;
        }

        // |coef| normalized by the sum of absolute coefficients (1 when they are all zero)
        var engine = Engine;
        var magnitudes = engine.TensorAbs(coefficients.AsTensor());
        T total = engine.TensorSum(magnitudes);
        if (NumOps.Equals(total, NumOps.Zero)) total = NumOps.One;
        var values = engine.TensorDivideScalar(magnitudes, total).ToArray();

        for (int i = 0; i < values.Length; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            importance[name] = values[i];
        }

        return importance;
    }

    /// <summary>
    /// bias + x * coefficients for every row (only the first coefficients.Length columns are used).
    /// </summary>
    /// <param name="features">The samples.</param>
    /// <param name="coefficients">The coefficients.</param>
    /// <param name="bias">The intercept.</param>
    /// <returns>The [rows] linear predictions.</returns>
    private protected Tensor<T> LinearPredictions(Matrix<T> features, Vector<T> coefficients, T bias)
    {
        var engine = Engine;
        int n = features.Rows;
        int p = coefficients.Length;
        if (n == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        var x = Tensor<T>.FromMatrix(features);
        if (features.Columns != p)
        {
            x = x.Slice(1, 0, p);
        }

        var products = p == 0
            ? new Tensor<T>(new[] { n })
            : engine.TensorMatMul(x, coefficients.AsTensor().Reshape(p, 1)).Reshape(n);
        return engine.TensorAddScalar(products, bias);
    }

    protected override ClassificationTaskType InferTaskType(Vector<T> y)
    {
        return ClassificationTaskType.Ordinal;
    }
}
