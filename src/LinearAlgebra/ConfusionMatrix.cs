namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a confusion matrix for evaluating the performance of a classification model.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// A confusion matrix is a table that summarizes the prediction results of a classification model.
/// It shows the counts of true positives, true negatives, false positives, and false negatives.
/// </para>
/// <para>
/// <b>For Beginners:</b> A confusion matrix helps you understand how well your AI model is performing
/// when classifying data into categories. It shows four important numbers:
/// <list type="bullet">
///   <item>
///     <term>True Positives</term>
///     <description>When your model correctly predicted "Yes" (e.g., correctly identified a cat as a cat)</description>
///   </item>
///   <item>
///     <term>True Negatives</term>
///     <description>When your model correctly predicted "No" (e.g., correctly identified a non-cat as not a cat)</description>
///   </item>
///   <item>
///     <term>False Positives</term>
///     <description>When your model incorrectly predicted "Yes" (e.g., identified a dog as a cat) - also called a "Type I error"</description>
///   </item>
///   <item>
///     <term>False Negatives</term>
///     <description>When your model incorrectly predicted "No" (e.g., identified a cat as not a cat) - also called a "Type II error"</description>
///   </item>
/// </list>
/// These numbers help calculate various metrics that tell you how accurate your model is.
/// </para>
/// </remarks>
public class ConfusionMatrix<T> : MatrixBase<T>
{
    /// <summary>
    /// Gets the number of true positive predictions (correctly predicted positive cases).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> True positives are when your model correctly identified something as positive.
    /// For example, if your model is detecting spam emails, a true positive is when it correctly
    /// identifies a spam email as spam.
    /// </remarks>
    public T TruePositives => this[0, 0];

    /// <summary>
    /// Gets the number of true negative predictions (correctly predicted negative cases).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> True negatives are when your model correctly identified something as negative.
    /// For example, if your model is detecting spam emails, a true negative is when it correctly
    /// identifies a legitimate email as not spam.
    /// </remarks>
    public T TrueNegatives => this[1, 1];

    /// <summary>
    /// Gets the number of false positive predictions (incorrectly predicted positive cases).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> False positives are when your model incorrectly identified something as positive
    /// when it was actually negative. This is also called a "Type I error". For example, if your model
    /// is detecting spam emails, a false positive is when it incorrectly marks a legitimate email as spam.
    /// </remarks>
    public T FalsePositives => this[1, 0];

    /// <summary>
    /// Gets the number of false negative predictions (incorrectly predicted negative cases).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> False negatives are when your model incorrectly identified something as negative
    /// when it was actually positive. This is also called a "Type II error". For example, if your model
    /// is detecting spam emails, a false negative is when it incorrectly lets a spam email into your inbox.
    /// </remarks>
    public T FalseNegatives => this[0, 1];

    /// <summary>
    /// Initializes a new instance of the <see cref="ConfusionMatrix{T}"/> class with the specified values.
    /// </summary>
    /// <param name="truePositives">The number of true positive predictions.</param>
    /// <param name="trueNegatives">The number of true negative predictions.</param>
    /// <param name="falsePositives">The number of false positive predictions.</param>
    /// <param name="falseNegatives">The number of false negative predictions.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor creates a new confusion matrix with the four basic counts
    /// that describe how well your model performed. The matrix is always 2x2 in size, representing
    /// the four possible outcomes of a binary classification.
    /// </remarks>
    public ConfusionMatrix(T truePositives, T trueNegatives, T falsePositives, T falseNegatives)
        : base(2, 2)
    {
        this[0, 0] = truePositives;
        this[1, 1] = trueNegatives;
        this[1, 0] = falsePositives;
        this[0, 1] = falseNegatives;
    }

    /// <summary>
    /// Creates a new instance of a matrix with the specified dimensions.
    /// </summary>
    /// <param name="rows">The number of rows in the new matrix.</param>
    /// <param name="cols">The number of columns in the new matrix.</param>
    /// <returns>A new matrix instance with the specified dimensions.</returns>
    /// <remarks>
    /// This is an internal method used for matrix operations that require creating new matrices.
    /// </remarks>
    protected override MatrixBase<T> CreateInstance(int rows, int cols)
    {
        return new Matrix<T>(rows, cols);
    }

    /// <summary>
    /// Gets the accuracy of the classification model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Accuracy is calculated as (True Positives + True Negatives) / (Total Predictions).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Accuracy tells you what percentage of all predictions your model got right.
    /// It's calculated by adding up all the correct predictions (both true positives and true negatives)
    /// and dividing by the total number of predictions made. A higher accuracy means your model is performing better.
    /// However, accuracy alone can be misleading if your data is imbalanced (e.g., if most of your data belongs to one class).
    /// </para>
    /// </remarks>
    public T Accuracy
    {
        get
        {
            T numerator = _numOps.Add(TruePositives, TrueNegatives);
            T denominator = _numOps.Add(_numOps.Add(_numOps.Add(TruePositives, TrueNegatives), FalsePositives), FalseNegatives);

            return _numOps.Divide(numerator, denominator);
        }
    }

    /// <summary>
    /// Gets the precision of the classification model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Precision is calculated as True Positives / (True Positives + False Positives).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Precision answers the question: "Of all the items my model predicted as positive,
    /// what percentage was actually positive?" It's a measure of how trustworthy the positive predictions are.
    /// High precision means that when your model says something is positive, it's usually correct.
    /// This is important in cases where false positives are costly (e.g., in medical diagnoses where a false
    /// positive might lead to unnecessary treatment).
    /// </para>
    /// </remarks>
    public T Precision
    {
        get
        {
            T denominator = _numOps.Add(TruePositives, FalsePositives);
            return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(TruePositives, denominator);
        }
    }

    /// <summary>
    /// Gets the recall (sensitivity) of the classification model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Recall is calculated as True Positives / (True Positives + False Negatives).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Recall (also called sensitivity) answers the question: "Of all the items that were
    /// actually positive, what percentage did my model correctly identify as positive?" It measures how good
    /// your model is at finding all the positive cases. High recall means your model rarely misses positive cases.
    /// This is important in situations where missing a positive case is costly (e.g., in cancer detection where
    /// missing a cancer diagnosis could be life-threatening).
    /// </para>
    /// </remarks>
    public T Recall
    {
        get
        {
            T denominator = _numOps.Add(TruePositives, FalseNegatives);
            return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(TruePositives, denominator);
        }
    }

    /// <summary>
    /// Gets the F1 score of the classification model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// F1 score is calculated as 2 * (Precision * Recall) / (Precision + Recall).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The F1 score is a balance between precision and recall. It's useful when you need
    /// to find a middle ground between these two metrics. A high F1 score means that your model has both good
    /// precision and good recall. This is particularly useful when your data is imbalanced (when one class has
    /// many more examples than another). The F1 score ranges from 0 (worst) to 1 (best).
    /// </para>
    /// </remarks>
    public T F1Score
    {
        get
        {
            T precision = Precision;
            T recall = Recall;
            T numerator = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(precision, recall));
            T denominator = _numOps.Add(precision, recall);

            return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(numerator, denominator);
        }
    }

    /// <summary>
    /// Gets the specificity of the classification model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specificity is calculated as True Negatives / (True Negatives + False Positives).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Specificity answers the question: "Of all the items that were actually negative,
    /// what percentage did my model correctly identify as negative?" It measures how good your model is at
    /// avoiding false alarms. High specificity means your model rarely misclassifies negative cases as positive.
    /// This is important in situations where false positives are costly (e.g., in spam detection where marking
    /// legitimate emails as spam would be problematic).
    /// </para>
    /// </remarks>
    public T Specificity
    {
        get
        {
            T denominator = _numOps.Add(TrueNegatives, FalsePositives);
            return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(TrueNegatives, denominator);
        }
    }
}