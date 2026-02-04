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
public abstract class OrdinalClassifierBase<T> : ClassifierBase<T>, IOrdinalClassifier<T>
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
        var probs = new Matrix<T>(n, k);

        for (int i = 0; i < n; i++)
        {
            // P(Y = 0) = P(Y ≤ 0)
            probs[i, 0] = cumProbs[i, 0];

            // P(Y = j) = P(Y ≤ j) - P(Y ≤ j-1) for middle classes
            for (int j = 1; j < k - 1; j++)
            {
                probs[i, j] = NumOps.Subtract(cumProbs[i, j], cumProbs[i, j - 1]);
            }

            // P(Y = k-1) = 1 - P(Y ≤ k-2)
            probs[i, k - 1] = NumOps.Subtract(NumOps.One, cumProbs[i, k - 2]);
        }

        return probs;
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
        var uniqueLabels = new SortedSet<double>();
        for (int i = 0; i < labels.Length; i++)
        {
            uniqueLabels.Add(NumOps.ToDouble(labels[i]));
        }

        ClassLabels = new Vector<T>(uniqueLabels.Count);
        int idx = 0;
        foreach (double label in uniqueLabels)
        {
            ClassLabels[idx++] = NumOps.FromDouble(label);
        }

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

        for (int i = 0; i < ClassLabels.Length; i++)
        {
            if (NumOps.Compare(ClassLabels[i], label) == 0)
            {
                return i;
            }
        }

        // If not found, return the closest class
        double labelVal = NumOps.ToDouble(label);
        int closestIdx = 0;
        double minDist = double.MaxValue;

        for (int i = 0; i < ClassLabels.Length; i++)
        {
            double dist = Math.Abs(NumOps.ToDouble(ClassLabels[i]) - labelVal);
            if (dist < minDist)
            {
                minDist = dist;
                closestIdx = i;
            }
        }

        return closestIdx;
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
        double val = NumOps.ToDouble(x);
        if (val > 20) return NumOps.One;
        if (val < -20) return NumOps.Zero;
        return NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
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
    protected override ClassificationTaskType InferTaskType(Vector<T> y)
    {
        return ClassificationTaskType.Ordinal;
    }
}
