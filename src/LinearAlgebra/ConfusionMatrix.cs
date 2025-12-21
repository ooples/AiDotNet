namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a confusion matrix for evaluating the performance of a classification model.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// A confusion matrix is a table that summarizes the prediction results of a classification model.
/// It supports both binary classification (2x2 matrix) and multi-class classification (NxN matrix).
/// </para>
/// <para>
/// <b>For Beginners:</b> A confusion matrix helps you understand how well your AI model is performing
/// when classifying data into categories.
/// </para>
/// <para>
/// <b>Binary Classification:</b> For 2-class problems, it shows four important numbers:
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
/// </para>
/// <para>
/// <b>Multi-Class Classification:</b> For 3+ class problems, the matrix is NxN where N is the number of classes.
/// Rows represent predicted classes, columns represent actual classes. Cell [i,j] contains the count of samples
/// predicted as class i but actually belonging to class j.
/// </para>
/// </remarks>
public class ConfusionMatrix<T> : MatrixBase<T>
{
    /// <summary>
    /// Gets the number of classes represented in the confusion matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For multi-class classification, this property returns the dimension <c>N</c> of the NxN confusion matrix,
    /// where each class is represented by a row and a column.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many categories your model is trying to predict.
    /// For example, if you are classifying images of digits (0-9), <c>ClassCount</c> will be 10.
    /// </para>
    /// </remarks>
    public int ClassCount => Rows;
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
    /// Initializes a new instance of the <see cref="ConfusionMatrix{T}"/> class with the specified dimension.
    /// </summary>
    /// <param name="dimension">The number of classes (creates a dimension x dimension matrix).</param>
    /// <exception cref="ArgumentException">Thrown when dimension is less than 2.</exception>
    /// <remarks>
    /// <para>
    /// Creates a zero-initialized NxN confusion matrix for multi-class classification.
    /// Use the <see cref="Increment"/> method to populate the matrix as predictions are made.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates an empty confusion matrix for problems with multiple classes.
    /// For example, if you're classifying images into 10 categories (0-9 digits), you would use dimension=10.
    /// The matrix starts with all zeros, and you increment cells as your model makes predictions.
    /// </para>
    /// </remarks>
    public ConfusionMatrix(int dimension)
        : base(dimension, dimension)
    {
        if (dimension < 2)
        {
            throw new ArgumentException("Dimension must be at least 2", nameof(dimension));
        }
    }

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

    /// <summary>
    /// Increments the count for a specific prediction-actual class pair.
    /// </summary>
    /// <param name="predictedClass">The predicted class index (0-based).</param>
    /// <param name="actualClass">The actual class index (0-based).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class indices are out of range.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is used to build up your confusion matrix as you evaluate predictions.
    /// Each time your model makes a prediction, you call this method with what it predicted and what the true answer was.
    /// For example, if your model predicted class 2 but the actual class was 1, you call Increment(2, 1).
    /// </para>
    /// </remarks>
    public void Increment(int predictedClass, int actualClass)
    {
        if (predictedClass < 0 || predictedClass >= Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(predictedClass),
                $"Predicted class {predictedClass} is out of range [0, {Rows - 1}]");
        }

        if (actualClass < 0 || actualClass >= Columns)
        {
            throw new ArgumentOutOfRangeException(nameof(actualClass),
                $"Actual class {actualClass} is out of range [0, {Columns - 1}]");
        }

        this[predictedClass, actualClass] = _numOps.Add(this[predictedClass, actualClass], _numOps.One);
    }

    /// <summary>
    /// Gets the true positives for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index (0-based).</param>
    /// <returns>The number of true positives for the specified class.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// True positives for class i are the diagonal element [i,i] - cases correctly predicted as class i.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many times your model correctly identified this specific class.
    /// For example, if you're classifying digits 0-9, GetTruePositives(5) tells you how many times your model
    /// correctly identified the digit 5 as 5.
    /// </para>
    /// </remarks>
    public T GetTruePositives(int classIndex)
    {
        if (classIndex < 0 || classIndex >= Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex),
                $"Class index {classIndex} is out of range [0, {Rows - 1}]");
        }

        return this[classIndex, classIndex];
    }

    /// <summary>
    /// Gets the false positives for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index (0-based).</param>
    /// <returns>The number of false positives for the specified class.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// False positives for class i are the sum of row i excluding the diagonal element [i,i].
    /// These are cases incorrectly predicted as class i when they were actually other classes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many times your model incorrectly identified something as this class
    /// when it was actually a different class. For example, GetFalsePositives(5) tells you how many times your model
    /// predicted digit 5 when it was actually a different digit.
    /// </para>
    /// </remarks>
    public T GetFalsePositives(int classIndex)
    {
        if (classIndex < 0 || classIndex >= Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex),
                $"Class index {classIndex} is out of range [0, {Rows - 1}]");
        }

        T sum = _numOps.Zero;
        for (int j = 0; j < Columns; j++)
        {
            if (j != classIndex)
            {
                sum = _numOps.Add(sum, this[classIndex, j]);
            }
        }

        return sum;
    }

    /// <summary>
    /// Gets the false negatives for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index (0-based).</param>
    /// <returns>The number of false negatives for the specified class.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// False negatives for class i are the sum of column i excluding the diagonal element [i,i].
    /// These are cases that were actually class i but were incorrectly predicted as other classes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many times your model missed this class by predicting something else.
    /// For example, GetFalseNegatives(5) tells you how many times the digit was actually 5, but your model predicted
    /// it as some other digit.
    /// </para>
    /// </remarks>
    public T GetFalseNegatives(int classIndex)
    {
        if (classIndex < 0 || classIndex >= Columns)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex),
                $"Class index {classIndex} is out of range [0, {Columns - 1}]");
        }

        T sum = _numOps.Zero;
        for (int i = 0; i < Rows; i++)
        {
            if (i != classIndex)
            {
                sum = _numOps.Add(sum, this[i, classIndex]);
            }
        }

        return sum;
    }

    /// <summary>
    /// Gets the true negatives for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index (0-based).</param>
    /// <returns>The number of true negatives for the specified class.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// True negatives for class i are all predictions where neither predicted nor actual was class i.
    /// This is the sum of all cells excluding row i and column i.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many times your model correctly identified that something was NOT
    /// this class. For example, GetTrueNegatives(5) tells you how many times your model correctly determined that
    /// a digit was NOT 5 (it was some other digit, and the model predicted correctly).
    /// </para>
    /// </remarks>
    public T GetTrueNegatives(int classIndex)
    {
        if (classIndex < 0 || classIndex >= Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex),
                $"Class index {classIndex} is out of range [0, {Rows - 1}]");
        }

        T sum = _numOps.Zero;
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                if (i != classIndex && j != classIndex)
                {
                    sum = _numOps.Add(sum, this[i, j]);
                }
            }
        }

        return sum;
    }

    /// <summary>
    /// Gets the overall accuracy across all classes.
    /// </summary>
    /// <returns>The accuracy as a value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Accuracy is calculated as the sum of diagonal elements (correct predictions) divided by the sum of all elements (total predictions).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you what percentage of all predictions your model got right, across all classes.
    /// An accuracy of 0.85 means your model was correct 85% of the time.
    /// </para>
    /// </remarks>
    public T GetAccuracy()
    {
        T correctPredictions = _numOps.Zero;
        T totalPredictions = _numOps.Zero;

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                totalPredictions = _numOps.Add(totalPredictions, this[i, j]);
                if (i == j)
                {
                    correctPredictions = _numOps.Add(correctPredictions, this[i, j]);
                }
            }
        }

        return _numOps.Equals(totalPredictions, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(correctPredictions, totalPredictions);
    }

    /// <summary>
    /// Gets the precision for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index (0-based).</param>
    /// <returns>The precision for the specified class.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// Precision for class i is calculated as TP / (TP + FP).
    /// It answers: "Of all predictions for this class, what percentage was correct?"
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how trustworthy your model's predictions are for this specific class.
    /// High precision means when your model says something is class i, it's usually right.
    /// </para>
    /// </remarks>
    public T GetPrecision(int classIndex)
    {
        T tp = GetTruePositives(classIndex);
        T fp = GetFalsePositives(classIndex);
        T denominator = _numOps.Add(tp, fp);

        return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(tp, denominator);
    }

    /// <summary>
    /// Gets the recall for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index (0-based).</param>
    /// <returns>The recall for the specified class.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// Recall for class i is calculated as TP / (TP + FN).
    /// It answers: "Of all actual cases of this class, what percentage did we correctly identify?"
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how good your model is at finding all instances of this class.
    /// High recall means your model rarely misses cases of this class.
    /// </para>
    /// </remarks>
    public T GetRecall(int classIndex)
    {
        T tp = GetTruePositives(classIndex);
        T fn = GetFalseNegatives(classIndex);
        T denominator = _numOps.Add(tp, fn);

        return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(tp, denominator);
    }

    /// <summary>
    /// Gets the F1 score for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index (0-based).</param>
    /// <returns>The F1 score for the specified class.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when class index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// F1 score for class i is the harmonic mean of precision and recall: 2 * (Precision * Recall) / (Precision + Recall).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a balanced measure that considers both precision and recall for this class.
    /// It's useful when you want a single metric that captures both how accurate and how complete your predictions are.
    /// </para>
    /// </remarks>
    public T GetF1Score(int classIndex)
    {
        T precision = GetPrecision(classIndex);
        T recall = GetRecall(classIndex);
        T numerator = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(precision, recall));
        T denominator = _numOps.Add(precision, recall);

        return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Gets the macro-averaged precision across all classes.
    /// </summary>
    /// <returns>The macro-averaged precision.</returns>
    /// <remarks>
    /// <para>
    /// Macro-average treats all classes equally by computing the metric independently for each class
    /// and then taking the average. This gives equal weight to each class regardless of support.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the average precision across all classes, treating each class equally.
    /// It's useful when you care about performance on all classes equally, even rare ones.
    /// </para>
    /// </remarks>
    public T GetMacroPrecision()
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < ClassCount; i++)
        {
            sum = _numOps.Add(sum, GetPrecision(i));
        }

        return _numOps.Divide(sum, _numOps.FromDouble(ClassCount));
    }

    /// <summary>
    /// Gets the macro-averaged recall across all classes.
    /// </summary>
    /// <returns>The macro-averaged recall.</returns>
    /// <remarks>
    /// <para>
    /// Macro-average treats all classes equally by computing the metric independently for each class
    /// and then taking the average. This gives equal weight to each class regardless of support.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the average recall across all classes, treating each class equally.
    /// It tells you how good your model is at finding instances across all classes on average.
    /// </para>
    /// </remarks>
    public T GetMacroRecall()
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < ClassCount; i++)
        {
            sum = _numOps.Add(sum, GetRecall(i));
        }

        return _numOps.Divide(sum, _numOps.FromDouble(ClassCount));
    }

    /// <summary>
    /// Gets the macro-averaged F1 score across all classes.
    /// </summary>
    /// <returns>The macro-averaged F1 score.</returns>
    /// <remarks>
    /// <para>
    /// Macro-average treats all classes equally by computing the metric independently for each class
    /// and then taking the average. This gives equal weight to each class regardless of support.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the average F1 score across all classes, treating each class equally.
    /// It provides a balanced view of model performance across all classes.
    /// </para>
    /// </remarks>
    public T GetMacroF1Score()
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < ClassCount; i++)
        {
            sum = _numOps.Add(sum, GetF1Score(i));
        }

        return _numOps.Divide(sum, _numOps.FromDouble(ClassCount));
    }

    /// <summary>
    /// Gets the micro-averaged precision across all classes.
    /// </summary>
    /// <returns>The micro-averaged precision.</returns>
    /// <remarks>
    /// <para>
    /// Micro-average aggregates the contributions of all classes to compute the average metric.
    /// For precision, this equals accuracy in multi-class classification.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This calculates precision by considering all predictions together, giving more
    /// weight to classes with more samples. For multi-class problems, this equals overall accuracy.
    /// </para>
    /// </remarks>
    public T GetMicroPrecision()
    {
        return GetAccuracy();
    }

    /// <summary>
    /// Gets the micro-averaged recall across all classes.
    /// </summary>
    /// <returns>The micro-averaged recall.</returns>
    /// <remarks>
    /// <para>
    /// Micro-average aggregates the contributions of all classes to compute the average metric.
    /// For recall, this equals accuracy in multi-class classification.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This calculates recall by considering all predictions together, giving more
    /// weight to classes with more samples. For multi-class problems, this equals overall accuracy.
    /// </para>
    /// </remarks>
    public T GetMicroRecall()
    {
        return GetAccuracy();
    }

    /// <summary>
    /// Gets the micro-averaged F1 score across all classes.
    /// </summary>
    /// <returns>The micro-averaged F1 score.</returns>
    /// <remarks>
    /// <para>
    /// Micro-average aggregates the contributions of all classes to compute the average metric.
    /// For F1 score, this equals accuracy in multi-class classification.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This calculates F1 score by considering all predictions together, giving more
    /// weight to classes with more samples. For multi-class problems, this equals overall accuracy.
    /// </para>
    /// </remarks>
    public T GetMicroF1Score()
    {
        return GetAccuracy();
    }

    /// <summary>
    /// Gets the weighted-averaged precision across all classes.
    /// </summary>
    /// <returns>The weighted-averaged precision.</returns>
    /// <remarks>
    /// <para>
    /// Weighted-average computes the metric for each class and averages them, weighted by the number of
    /// true instances for each class (support). This accounts for class imbalance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like macro-average, but gives more importance to classes that appear
    /// more frequently in your data. It's useful when you have imbalanced classes but still want to see
    /// per-class performance weighted by how common each class is.
    /// </para>
    /// </remarks>
    public T GetWeightedPrecision()
    {
        T weightedSum = _numOps.Zero;
        T totalSupport = _numOps.Zero;

        for (int i = 0; i < ClassCount; i++)
        {
            T support = _numOps.Zero;
            for (int j = 0; j < Rows; j++)
            {
                support = _numOps.Add(support, this[j, i]);
            }

            T precision = GetPrecision(i);
            weightedSum = _numOps.Add(weightedSum, _numOps.Multiply(precision, support));
            totalSupport = _numOps.Add(totalSupport, support);
        }

        return _numOps.Equals(totalSupport, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(weightedSum, totalSupport);
    }

    /// <summary>
    /// Gets the weighted-averaged recall across all classes.
    /// </summary>
    /// <returns>The weighted-averaged recall.</returns>
    /// <remarks>
    /// <para>
    /// Weighted-average computes the metric for each class and averages them, weighted by the number of
    /// true instances for each class (support). This accounts for class imbalance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like macro-average, but gives more importance to classes that appear
    /// more frequently in your data. For recall, this often equals overall accuracy.
    /// </para>
    /// </remarks>
    public T GetWeightedRecall()
    {
        T weightedSum = _numOps.Zero;
        T totalSupport = _numOps.Zero;

        for (int i = 0; i < ClassCount; i++)
        {
            T support = _numOps.Zero;
            for (int j = 0; j < Rows; j++)
            {
                support = _numOps.Add(support, this[j, i]);
            }

            T recall = GetRecall(i);
            weightedSum = _numOps.Add(weightedSum, _numOps.Multiply(recall, support));
            totalSupport = _numOps.Add(totalSupport, support);
        }

        return _numOps.Equals(totalSupport, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(weightedSum, totalSupport);
    }

    /// <summary>
    /// Gets the weighted-averaged F1 score across all classes.
    /// </summary>
    /// <returns>The weighted-averaged F1 score.</returns>
    /// <remarks>
    /// <para>
    /// Weighted-average computes the metric for each class and averages them, weighted by the number of
    /// true instances for each class (support). This accounts for class imbalance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like macro-average, but gives more importance to classes that appear
    /// more frequently in your data. It provides a balanced measure that accounts for class imbalance.
    /// </para>
    /// </remarks>
    public T GetWeightedF1Score()
    {
        T weightedSum = _numOps.Zero;
        T totalSupport = _numOps.Zero;

        for (int i = 0; i < ClassCount; i++)
        {
            T support = _numOps.Zero;
            for (int j = 0; j < Rows; j++)
            {
                support = _numOps.Add(support, this[j, i]);
            }

            T f1 = GetF1Score(i);
            weightedSum = _numOps.Add(weightedSum, _numOps.Multiply(f1, support));
            totalSupport = _numOps.Add(totalSupport, support);
        }

        return _numOps.Equals(totalSupport, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(weightedSum, totalSupport);
    }
}
