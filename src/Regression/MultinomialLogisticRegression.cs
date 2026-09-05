using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Regression;

/// <summary>
/// Represents a multinomial logistic regression model for multi-class classification problems.
/// </summary>
/// <remarks>
/// <para>
/// Multinomial logistic regression extends binary logistic regression to handle multiple classes. It models the probabilities
/// of different possible outcomes using the softmax function. For each class, the model learns a set of coefficients that
/// determine how each feature affects the probability of that class. During prediction, it assigns the input to the class
/// with the highest probability.
/// </para>
/// <para><b>For Beginners:</b> Multinomial logistic regression is a method for classifying data into multiple categories.
/// 
/// Think of it like a voting system where:
/// - Each feature (input variable) gets to "vote" for different categories
/// - The importance of each feature's vote is learned from training data
/// - For any new data point, we count the weighted votes for each category
/// - The category with the most votes wins and becomes the prediction
/// 
/// For example, when classifying emails into categories like "work," "personal," or "spam,"
/// certain words might strongly suggest one category over others. The model learns which
/// features (words) are most helpful for distinguishing between the different categories.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a multinomial logistic regression for multi-class classification
/// var options = new MultinomialLogisticRegressionOptions&lt;double&gt;();
/// var model = new MultinomialLogisticRegression&lt;double&gt;(options);
///
/// // Prepare training data: 9 samples with 2 features, 3 classes (0, 1, 2)
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 1 }, { 1, 2 }, { 2, 1 }, { 4, 4 }, { 5, 4 }, { 4, 5 }, { 8, 8 }, { 9, 8 }, { 8, 9 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 });
///
/// // Train the multi-class model with softmax
/// model.Train(features, labels);
///
/// // Predict class probabilities for a new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 5, 5 } });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Applied Logistic Regression", "https://doi.org/10.1002/0471722146")]
public partial class MultinomialLogisticRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// The configuration options for the multinomial logistic regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the multinomial logistic regression algorithm during training, including
    /// parameters such as the maximum number of iterations, convergence tolerance, and the matrix decomposition method
    /// used for solving the linear system.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings that control how the model learns.
    /// 
    /// Key settings include:
    /// - How many attempts (iterations) the model makes to improve itself
    /// - How precise the model needs to be before it stops training
    /// - What mathematical method to use for calculations
    /// 
    /// These settings affect how quickly the model trains and how accurate it becomes.
    /// Think of them as the "knobs" you can adjust to fine-tune the learning process.
    /// </para>
    /// </remarks>
    private readonly MultinomialLogisticRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The coefficients matrix, where each row corresponds to a class and each column to a feature (plus intercept).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The coefficients matrix contains the learned weights for the multinomial logistic regression model. Each row
    /// corresponds to a different class, and each column corresponds to a different feature (with an additional column
    /// for the intercept term). These coefficients determine how each feature influences the probability of each class.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "knowledge" the model learns from the training data.
    /// 
    /// The coefficients:
    /// - Show how important each feature is for predicting each class
    /// - Positive values mean the feature increases the chance of that class
    /// - Negative values mean the feature decreases the chance of that class
    /// - Larger absolute values (further from zero) indicate stronger influences
    /// 
    /// For example, in email classification, the word "meeting" might have a high coefficient for
    /// "work" emails and a low coefficient for "spam" emails.
    /// </para>
    /// </remarks>
    [AiDotNet.Attributes.FittedParameter(
        Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private Matrix<T>? _coefficients;

    /// <summary>
    /// The number of distinct classes in the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of distinct classes found in the training data. It determines the number of rows
    /// in the coefficients matrix, as each class has its own set of coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of different categories the model can predict.
    /// 
    /// For example:
    /// - In email classification, it might be 3 (work, personal, spam)
    /// - In product categorization, it might be dozens or hundreds of categories
    /// - In sentiment analysis, it might be 3 (positive, neutral, negative)
    /// 
    /// The model learns a separate set of weights for each of these categories.
    /// </para>
    /// </remarks>
    private int _numClasses;

    /// <summary>
    /// The class labels the model was trained on, ascending. <see cref="Predict"/> returns values drawn
    /// from this list, so predictions come back in whatever labels were supplied for training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If you trained with categories numbered 2, 5 and 9, the model works with 0, 1
    /// and 2 internally but this remembers your numbering, so predictions are reported as 2, 5 and 9 again.
    /// </remarks>
    public IReadOnlyList<T> ClassLabels => _classLabels;

    private List<T> _classLabels = new List<T>();

    /// <summary>
    /// Multinomial logistic is a classification model — no optimizer parameter injection.
    /// </summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultinomialLogisticRegression{T}"/> class with optional custom options and regularization.
    /// </summary>
    /// <param name="options">Custom options for the multinomial logistic regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new multinomial logistic regression model with the specified options and regularization.
    /// If no options are provided, default values are used. Regularization helps prevent overfitting by penalizing
    /// large coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new multinomial logistic regression model with your chosen settings.
    /// 
    /// When creating the model:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps prevent the model from memorizing the training data
    /// 
    /// Regularization is like adding a penalty for complexity, encouraging the model to keep things
    /// simple unless there's strong evidence for complexity. This typically helps the model
    /// perform better on new, unseen data.
    /// </para>
    /// </remarks>
    public MultinomialLogisticRegression(MultinomialLogisticRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new MultinomialLogisticRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the multinomial logistic regression model using the provided features and target values.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the class labels (as integers) for each sample.</param>
    /// <remarks>
    /// <para>
    /// This method trains the multinomial logistic regression model using Newton's method to find the maximum likelihood
    /// estimates of the coefficients. It iteratively computes the probabilities, gradient, and Hessian matrix, and updates
    /// the coefficients until convergence or until the maximum number of iterations is reached. Regularization is applied
    /// if specified.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// 1. The model starts with initial guesses for the coefficients
    /// 2. It calculates how likely each class is for each training example
    /// 3. It calculates how to change the coefficients to improve the predictions
    /// 4. It updates the coefficients based on a mathematically optimal approach (Newton's method)
    /// 5. It repeats steps 2-4 until the changes become very small or a maximum number of iterations is reached
    /// 
    /// This process finds the best coefficients for distinguishing between the different classes
    /// based on the features in your training data.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);

        // Two substitutions used to happen here, both silent:
        //
        //   1. A continuous target set `_useOLS = true` and fitted ordinary least squares, so the
        //      object presented a linear model's coefficients as the classifier's.
        //   2. An integer target with many distinct labels was QUANTIZED into at most 5 equal-width
        //      bins. The model then reported classes the caller had never supplied, and Predict
        //      returned bin indices that did not correspond to any label in the training data.
        //
        // Both are replaced by validation. Class labels need not be consecutive or zero-based --
        // {2, 5, 9} trains fine and encodes to {0, 1, 2} -- and a genuinely continuous target throws
        // with a message naming what was observed and which model to use instead.
        var (encodedY, classes) = ValidationHelper<T>.EncodeMulticlassTarget(
            y, nameof(MultinomialLogisticRegression<T>));
        _classLabels = classes;
        y = encodedY;

        _numClasses = classes.Count;

        int numFeatures = x.Columns;
        _coefficients = new Matrix<T>(_numClasses, numFeatures + 1);

        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Matrix<T> probabilities = ComputeProbabilities(xWithIntercept);
            Matrix<T> gradient = ComputeGradient(xWithIntercept, y, probabilities);
            Matrix<T> hessian = ComputeHessian(xWithIntercept, probabilities);

            if (Regularization != null)
            {
                gradient = gradient.Add(Regularization.Regularize(gradient));
                hessian = hessian.Add(Regularization.Regularize(hessian));
            }

            Vector<T> flattenedGradient = gradient.Flatten();
            Vector<T> update = MatrixSolutionHelper.SolveLinearSystem(hessian, flattenedGradient, _options.DecompositionType);

            // Reshape update into _coefficients shape: _numClasses × (numFeatures+1)
            Matrix<T> updateMatrix = new Matrix<T>(_coefficients.Rows, _coefficients.Columns);
            for (int i = 0; i < Math.Min(update.Length, _coefficients.Rows * _coefficients.Columns); i++)
            {
                updateMatrix[i / _coefficients.Columns, i % _coefficients.Columns] = update[i];
            }

            _coefficients = _coefficients.Subtract(updateMatrix);

            if (HasConverged(updateMatrix))
            {
                break;
            }
        }

        // `_coefficients` is [class, feature] with the intercept in the last column, so a COLUMN is one
        // feature across all classes and a ROW is one class across all features. GetColumn(0) therefore
        // returned "feature 0's weight for each class" -- a vector as long as the number of CLASSES, being
        // published as the per-feature coefficient vector. With four classes over three features it
        // reported a coefficient for feature 3, and GetActiveFeatureIndices duly returned an index outside
        // the input feature space.
        //
        // A multiclass softmax has no single coefficient per feature: it has one per (class, feature), and
        // they are identified only up to a shift shared across classes. What the single-vector contract is
        // used for here is feature ACTIVITY, so each entry is the mean absolute weight that feature
        // carries across the classes -- zero exactly when no class uses the feature. Per-class values
        // remain available through the model's own coefficient matrix.
        int featureCount = _coefficients.Columns - 1;
        var perFeature = new Vector<T>(featureCount);
        for (int j = 0; j < featureCount; j++)
        {
            T total = NumOps.Zero;
            for (int c = 0; c < _coefficients.Rows; c++)
            {
                total = NumOps.Add(total, NumOps.Abs(_coefficients[c, j]));
            }

            perFeature[j] = NumOps.Divide(total, NumOps.FromDouble(_coefficients.Rows));
        }

        Coefficients = perFeature;
        Intercept = _coefficients[0, _coefficients.Columns - 1];
        TrainingFeatureCount = featureCount;
    }

    /// <summary>
    /// Computes the probabilities of each class for each sample using the softmax function.
    /// </summary>
    /// <param name="x">The feature matrix with intercept term.</param>
    /// <returns>A matrix of probabilities, where each row corresponds to a sample and each column to a class.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the coefficients have not been initialized.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the probabilities of each class for each sample using the softmax function. It first calculates
    /// the raw scores by multiplying the features with the coefficients, then applies the softmax function to convert these
    /// scores into probabilities that sum to 1 for each sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how likely each class is for each data point.
    /// 
    /// The probability calculation:
    /// 1. Multiplies each feature by its corresponding coefficient for each class (weighted voting)
    /// 2. Sums these values to get a "score" for each class
    /// 3. Applies the softmax function, which converts these scores into probabilities
    /// 4. The probabilities for all classes for a single sample add up to 100%
    /// 
    /// The softmax function ensures that increasing the score for one class increases its probability
    /// while decreasing the probability of other classes proportionally.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeProbabilities(Matrix<T> x)
    {
        if (_coefficients == null)
            throw new InvalidOperationException("Coefficients have not been initialized.");

        Matrix<T> scores = x.Multiply(_coefficients.Transpose());
        Vector<T> maxScores = scores.RowWiseMax();
        Matrix<T> expScores = scores.Transform((s, i, j) => NumOps.Exp(NumOps.Subtract(s, maxScores[i])));
        Vector<T> sumExpScores = expScores.RowWiseSum();

        // Normalize: divide each row element by the row sum (broadcast division)
        var result = new Matrix<T>(expScores.Rows, expScores.Columns);
        for (int i = 0; i < expScores.Rows; i++)
        {
            for (int j = 0; j < expScores.Columns; j++)
            {
                result[i, j] = NumOps.Divide(expScores[i, j], sumExpScores[i]);
            }
        }
        return result;
    }

    /// <summary>
    /// Computes the gradient of the log-likelihood with respect to the coefficients.
    /// </summary>
    /// <param name="x">The feature matrix with intercept term.</param>
    /// <param name="y">The target vector containing the class labels.</param>
    /// <param name="probabilities">The matrix of class probabilities for each sample.</param>
    /// <returns>The gradient matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of the log-likelihood with respect to the coefficients. The gradient indicates
    /// how the log-likelihood would change with small changes in the coefficients, providing the direction for updating
    /// the coefficients to increase the likelihood.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how to change the coefficients to improve predictions.
    /// 
    /// The gradient:
    /// - Shows the direction and amount to change each coefficient to make better predictions
    /// - Compares the predicted probabilities with the actual classes
    /// - Larger gradient values indicate coefficients that need more adjustment
    /// 
    /// It's like getting feedback on which knobs need the most adjustment to improve
    /// the model's performance.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeGradient(Matrix<T> x, Vector<T> y, Matrix<T> probabilities)
    {
        Matrix<T> yOneHot = CreateOneHotEncoding(y);
        return x.Transpose().Multiply(yOneHot.Subtract(probabilities));
    }

    /// <summary>
    /// Computes the Hessian matrix of the log-likelihood with respect to the coefficients.
    /// </summary>
    /// <param name="x">The feature matrix with intercept term.</param>
    /// <param name="probabilities">The matrix of class probabilities for each sample.</param>
    /// <returns>The Hessian matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Hessian matrix of the log-likelihood with respect to the coefficients. The Hessian
    /// contains the second derivatives of the log-likelihood, providing information about the curvature of the
    /// log-likelihood surface. This is used in Newton's method to determine not just the direction but also the
    /// optimal step size for updating the coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the model is to changes in each coefficient.
    /// 
    /// The Hessian:
    /// - Measures how quickly the gradient changes as the coefficients change
    /// - Helps determine the optimal step size for updating each coefficient
    /// - Accounts for interactions between different coefficients
    /// 
    /// It's like having a map of the terrain that helps you take steps of the right size
    /// in each direction, rather than always taking fixed-size steps.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeHessian(Matrix<T> x, Matrix<T> probabilities)
    {
        int n = x.Rows;
        int p = x.Columns;
        Matrix<T> hessian = new(p * _numClasses, p * _numClasses);

        for (int i = 0; i < n; i++)
        {
            Vector<T> xi = x.GetRow(i);
            Vector<T> probs = probabilities.GetRow(i);
            Matrix<T> diagP = Matrix<T>.CreateDiagonal(probs);
            Matrix<T> ppt = probs.OuterProduct(probs);
            Matrix<T> h = diagP.Subtract(ppt);
            Matrix<T> xxt = xi.OuterProduct(xi);
            Matrix<T> block = xxt.KroneckerProduct(h);
            hessian = hessian.Add(block);
        }

        return hessian.Negate();
    }

    /// <summary>
    /// Creates a one-hot encoding of the class labels.
    /// </summary>
    /// <param name="y">The target vector containing the class labels as integers.</param>
    /// <returns>A matrix where each row is a one-hot encoded vector for the corresponding class label.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a one-hot encoding of the class labels, which is a binary matrix representation where each row
    /// corresponds to a sample and each column corresponds to a class. For each sample, the element corresponding to its
    /// class is set to 1, and all other elements are set to 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts class labels into a special matrix format.
    /// 
    /// One-hot encoding:
    /// - Represents categorical data as a matrix of 0s and 1s
    /// - Each row represents one data point
    /// - Each column represents one possible class
    /// - A 1 in position (i,j) means the i-th data point belongs to the j-th class
    /// - All other positions contain 0s
    /// 
    /// For example, if there are 3 classes (0, 1, 2), the label "1" would be encoded as [0, 1, 0],
    /// meaning "not class 0, yes class 1, not class 2".
    /// </para>
    /// </remarks>
    private Matrix<T> CreateOneHotEncoding(Vector<T> y)
    {
        Matrix<T> oneHot = new Matrix<T>(y.Length, _numClasses);
        for (int i = 0; i < y.Length; i++)
        {
            int classIndex = Convert.ToInt32(NumOps.ToInt32(y[i]));
            oneHot[i, classIndex] = NumOps.One;
        }

        return oneHot;
    }

    /// <summary>
    /// Determines if the training has converged based on the magnitude of the coefficient updates.
    /// </summary>
    /// <param name="update">The matrix of coefficient updates from the current iteration.</param>
    /// <returns>True if the maximum absolute update is less than the tolerance, indicating convergence; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the training has converged by comparing the maximum absolute value of the coefficient updates
    /// to the specified tolerance. If the maximum update is smaller than the tolerance, the algorithm is considered to have
    /// converged, meaning that further iterations would not significantly improve the model.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the model has finished learning.
    /// 
    /// Convergence means:
    /// - The model is no longer making significant improvements
    /// - The coefficient updates are very small (below a threshold)
    /// - Further training is unlikely to yield better results
    /// 
    /// It's like knowing when to stop studying for a test - at some point, additional effort
    /// yields diminishing returns, and your time is better spent elsewhere.
    /// </para>
    /// </remarks>
    private bool HasConverged(Matrix<T> update)
    {
        T maxChange = update.Max(NumOps.Abs);
        return NumOps.LessThan(maxChange, NumOps.FromDouble(_options.Tolerance));
    }

    /// <summary>
    /// Predicts the class labels for new data points using the trained multinomial logistic regression model.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A vector containing the predicted class labels (as integers).</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the class labels for new data points by computing the probabilities for each class and
    /// selecting the class with the highest probability for each sample. The class labels are returned as integers.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model makes predictions on new data.
    /// 
    /// The prediction process:
    /// 1. Calculate the probability of each class for each data point
    /// 2. For each data point, select the class with the highest probability
    /// 3. Return these predicted classes as the results
    /// 
    /// It's like a voting system where each feature casts a weighted vote for each class,
    /// and the class with the most votes wins.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        Matrix<T> probabilities = ComputeProbabilities(xWithIntercept);

        // Argmax gives the internal class index; map it back to the label the caller trained with,
        // so predictions are comparable with the y that was passed to Train. Returning the raw index
        // would silently relabel the caller's classes whenever they were not already 0..K-1.
        Vector<T> indices = probabilities.RowWiseArgmax();
        if (_classLabels.Count == 0)
        {
            return indices;
        }

        var labelled = new Vector<T>(indices.Length);
        for (int i = 0; i < indices.Length; i++)
        {
            int index = (int)Math.Round(NumOps.ToDouble(indices[i]));
            index = Math.Max(0, Math.Min(_classLabels.Count - 1, index));
            labelled[i] = _classLabels[index];
        }

        return labelled;
    }

    /// <summary>
    /// Predicts the probabilities of each class for new data points.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A matrix where each row corresponds to a sample and each column to the probability of a class.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the probabilities of each class for new data points using the trained model. The resulting
    /// matrix contains the probability of each class for each sample. These probabilities sum to 1 across the classes
    /// for each sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides the likelihood of each class for each data point.
    /// 
    /// Rather than just giving the final prediction, it provides:
    /// - The probability of each possible class
    /// - A measure of the model's confidence in each prediction
    /// - Values between 0 (impossible) and 1 (certain), with all classes summing to 1
    /// 
    /// This is useful when you need to know not just the predicted class but also
    /// how confident the model is in that prediction. For example, you might treat a
    /// prediction with 95% confidence differently than one with 51% confidence.
    /// </para>
    /// </remarks>
    public Matrix<T> PredictProbabilities(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        return ComputeProbabilities(xWithIntercept);
    }
}
