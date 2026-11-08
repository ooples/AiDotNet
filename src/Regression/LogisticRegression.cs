using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Optimizers;

namespace AiDotNet.Regression;

/// <summary>
/// Represents a logistic regression model for binary classification problems.
/// </summary>
/// <remarks>
/// <para>
/// Logistic regression is a statistical method used for binary classification tasks, where the goal is to predict
/// one of two possible outcomes (such as yes/no, true/false, 0/1). Unlike linear regression, which predicts continuous values,
/// logistic regression outputs probabilities between 0 and 1, which can be interpreted as the likelihood of belonging to the
/// positive class.
/// </para>
/// <para><b>For Beginners:</b> Logistic regression is like a decision-maker that predicts whether something belongs to
/// one category or another.
/// 
/// Think of it like determining whether an email is spam or not:
/// - The model looks at different "features" of the email (like certain words or sender information)
/// - It calculates how much each feature suggests "spam" or "not spam"
/// - It combines all this information to make a final prediction between 0 and 1
/// - Values closer to 1 mean "more likely spam", values closer to 0 mean "more likely not spam"
/// 
/// For example, words like "free" or "offer" might increase the spam probability, while emails from your contacts
/// might decrease it. Logistic regression finds the right balance of these factors to make accurate predictions.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new LogisticRegression&lt;double&gt;())
///     .Build(features, targets);
/// var prediction = result.Predict(new Matrix&lt;double&gt;(new double[,] { { 3.0, 3.0 } }));
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Applied Logistic Regression", "https://doi.org/10.1002/0471722146")]
public partial class LogisticRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// The configuration options for the logistic regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the logistic regression algorithm during training, including
    /// parameters such as the maximum number of iterations and convergence tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings that control how the model learns.
    /// 
    /// They include:
    /// - How many attempts (iterations) the model makes to improve itself
    /// - How precise the model needs to be before it stops training
    /// - How quickly the model adjusts its predictions (learning rate)
    /// 
    /// Think of these like the knobs on a machine that you can adjust to get better results.
    /// </para>
    /// </remarks>
    private readonly LogisticRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="LogisticRegression{T}"/> class with optional custom options
    /// and regularization.
    /// </summary>
    /// <param name="options">Custom options for the logistic regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new logistic regression model with the specified options and regularization.
    /// If no options are provided, default values are used. Regularization helps prevent overfitting by penalizing
    /// large coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new logistic regression model with your chosen settings.
    /// 
    /// When creating a logistic regression model:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps the model generalize better to new data
    /// 
    /// Regularization is like adding training wheels to prevent the model from memorizing the training data too closely,
    /// which would make it perform poorly on new, unseen data.
    /// </para>
    /// </remarks>
    public LogisticRegression(LogisticRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new LogisticRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the logistic regression model using the provided features and target values.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the binary labels (0 or 1) for each sample.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in X does not match the length of y.</exception>
    /// <remarks>
    /// <para>
    /// This method trains the logistic regression model using gradient ascent to maximize the likelihood of the observed data.
    /// The algorithm iteratively updates the coefficients and intercept based on the prediction errors until convergence
    /// or until the maximum number of iterations is reached. Regularization is applied if specified.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// - The model starts with initial guesses for how important each feature is
    /// - It makes predictions based on these guesses
    /// - It compares its predictions with the actual answers
    /// - It adjusts its guesses to reduce errors
    /// - This process repeats until the model stops improving significantly
    /// 
    /// For example, with email classification, the model might learn that the word "meeting" is a strong indicator
    /// of a legitimate email, while "click here to claim" suggests spam.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Logistic regression is a classification model — no optimizer parameter injection.
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
    /// The two class labels the model was trained on, ascending. <see cref="Predict"/> returns the
    /// probability of <c>ClassLabels[1]</c>, the higher of the two.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If you trained on labels like -1 and +1, this remembers them, so you can tell
    /// which of your own two categories the returned probability refers to. It is always the second one.
    /// </remarks>
    public IReadOnlyList<T> ClassLabels => _classLabels;

    private List<T> _classLabels = new List<T>();

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
            throw new ArgumentException("The number of rows in X must match the length of y.");
        int n = x.Rows;
        int p = x.Columns;

        // A target that is not two classes is a caller error, not a cue to fit something else.
        // This previously set `_useOLS = true` and fitted ordinary least squares, so the object
        // reported Coefficients from a linear model while presenting as a logistic one -- and
        // Predict skipped the sigmoid, returning unbounded values where a probability was promised.
        // EncodeBinaryTarget accepts any two labels ({0,1}, {-1,1}, {1,2}, ...) and throws with a
        // diagnostic naming the observed target otherwise.
        var (encodedY, classes) = ValidationHelper<T>.EncodeBinaryTarget(y, nameof(LogisticRegression<T>));
        _classLabels = classes;
        y = encodedY;

        // Fit on standardized features, then transform the coefficients back.
        //
        // Batch gradient ascent moves the intercept by at most LearningRate per iteration, so on raw
        // features it cannot reach the intercept the fit requires. With features spanning [0,10] and a
        // balanced split, that intercept is around -60; at the default learning rate of 0.01 over 1000
        // iterations the intercept can travel about 5, and the coefficients end up absorbing the shortfall
        // by turning NEGATIVE -- so the fitted model ran the response backwards in every feature. Centering
        // and scaling puts the required intercept near zero and every coefficient on a comparable scale,
        // which is exactly why standardization is the standing advice for gradient-based logistic
        // regression. The back-transform at the end keeps Coefficients and Intercept meaningful on the
        // caller's own scale, so nothing outside this method has to know.
        var (featureMean, featureStd) = ComputeStandardization(x, p);
        var (standardized, targets) = StandardizedDoubles(x, y, featureMean, featureStd);

        // One objective for every solver: the mean log loss plus the regularization penalty, over [coefficients,
        // intercept]. This was a hand-written gradient ascent of 1,000 fixed steps, 160x slower than scikit-learn's
        // L-BFGS on 20,000 rows to the same accuracy, and it added the penalty's gradient with the ascent's sign, so
        // regularization pushed the coefficients AWAY from zero during the fit and a single shrink at the end stood
        // in for it. As a term of the minimized objective the penalty now acts throughout.
        var solver = _options.Solver ?? LBFGSOptimizer<T, Matrix<T>, Vector<T>>.CreateForFunction();
        var solution = solver.Minimize(
            new Vector<T>(p + 1),
            weights => LogLossAndGradient(weights, standardized, targets, p),
            _options.MaxIterations,
            NumOps.FromDouble(_options.Tolerance));

        Coefficients = new Vector<T>(p);
        for (int j = 0; j < p; j++)
            Coefficients[j] = solution[j];
        Intercept = solution[p];

        // Back to the caller's scale: a standardized fit b*(x - mu)/sigma + a is the same linear
        // predictor as (b/sigma) * x + (a - sum(b * mu / sigma)).
        var rescaled = new Vector<T>(p);
        T interceptShift = NumOps.Zero;
        for (int j = 0; j < p; j++)
        {
            rescaled[j] = NumOps.Divide(Coefficients[j], featureStd[j]);
            interceptShift = NumOps.Add(interceptShift, NumOps.Multiply(rescaled[j], featureMean[j]));
        }

        Coefficients = rescaled;
        Intercept = NumOps.Subtract(Intercept, interceptShift);
        TrainingFeatureCount = p;
    }

    /// <summary>
    /// Creates a fixed-step gradient descent solver at this model's <see cref="LogisticRegressionOptions{T}.LearningRate"/>,
    /// for <see cref="LogisticRegressionOptions{T}.Solver"/>.
    /// </summary>
    /// <returns>A gradient descent optimizer configured for function minimization.</returns>
    /// <remarks>
    /// <para>
    /// This is how the model was fitted before L-BFGS became the default. It reaches the same minimum given enough
    /// iterations, but typically needs many more of them.
    /// </para>
    /// <para><b>For Beginners:</b> Only use this if you specifically want plain gradient descent:
    /// <code>
    /// var options = new LogisticRegressionOptions&lt;double&gt; { LearningRate = 0.05, MaxIterations = 5000 };
    /// var model = new LogisticRegression&lt;double&gt;(options);
    /// options.Solver = model.CreateGradientDescentSolver();
    /// </code>
    /// </para>
    /// </remarks>
    public IFunctionOptimizer<T> CreateGradientDescentSolver()
        => GradientDescentOptimizer<T, Matrix<T>, Vector<T>>.CreateForFunction(
            new GradientDescentOptimizerOptions<T, Matrix<T>, Vector<T>> { InitialLearningRate = _options.LearningRate });

    /// <summary>Rows per block of the objective's parallel pass.</summary>
    private const int LossBlockRows = 2048;

    /// <summary>
    /// The objective every solver minimizes, and its gradient, at <paramref name="weights"/> = [coefficients, intercept].
    /// </summary>
    /// <remarks>
    /// The mean log loss uses <c>softplus(z) - y z</c>, which stays finite for any margin <c>z</c>. The penalty's value is
    /// recovered from the regularizer's gradient by integrating along the ray from zero, <c>integral over t in [0, 1] of
    /// beta . g(t beta)</c>, with the midpoint rule <c>beta . g(beta / 2)</c>: exact for L1, L2 and elastic net, whose
    /// penalty gradients are constant or linear along that ray for t &gt; 0. A rule that samples t = 0 is not exact for L1,
    /// whose gradient there is the subgradient 0 rather than the constant it takes on the rest of the ray. The objective and gradient must agree for a line search to work, and
    /// <see cref="IRegularization{T, TInput, TOutput}"/> exposes only the gradient. The intercept is not penalized.
    /// </remarks>
    internal (T Objective, Vector<T> Gradient) LogLossAndGradient(
        Vector<T> weights, double[] x, double[] y, int p)
    {
        int n = y.Length;
        var coefficients = new Vector<T>(p);
        var beta = new double[p];
        for (int j = 0; j < p; j++)
        {
            coefficients[j] = weights[j];
            beta[j] = NumOps.ToDouble(weights[j]);
        }

        double intercept = NumOps.ToDouble(weights[p]);

        // One pass over the rows: the margin, its loss term and residual, and the residual's share of the gradient.
        // Fixed blocks of rows run in parallel and their sums are added in block order, so the result does not
        // depend on how many threads ran it. Each block keeps [gradient sums..., loss, residual total].
        int blocks = (n + LossBlockRows - 1) / LossBlockRows;
        var partials = new double[blocks][];
        System.Threading.Tasks.Parallel.For(0, blocks, block =>
        {
            var sums = new double[p + 2];
            int end = Math.Min(n, (block + 1) * LossBlockRows);
            for (int i = block * LossBlockRows; i < end; i++)
            {
                int offset = i * p;
                double z = intercept;
                for (int j = 0; j < p; j++) z += x[offset + j] * beta[j];

                // One exponential serves both terms: with e = exp(-|z|),
                // softplus(z) = max(z, 0) + log(1 + e), and sigmoid(z) = 1 / (1 + e) for z >= 0, e / (1 + e) below.
                double e = Math.Exp(-Math.Abs(z));
                double onePlusE = 1.0 + e;
                sums[p] += (z > 0 ? z : 0) + Math.Log(onePlusE) - y[i] * z;

                double residual = (z >= 0 ? 1.0 : e) / onePlusE - y[i];
                sums[p + 1] += residual;
                for (int j = 0; j < p; j++) sums[j] += residual * x[offset + j];
            }

            partials[block] = sums;
        });

        var gradientSums = new double[p];
        double loss = 0;
        double residualTotal = 0;
        foreach (var sums in partials)
        {
            for (int j = 0; j < p; j++) gradientSums[j] += sums[j];
            loss += sums[p];
            residualTotal += sums[p + 1];
        }

        var coefficientGradient = new Vector<T>(p);
        for (int j = 0; j < p; j++) coefficientGradient[j] = NumOps.FromDouble(gradientSums[j] / n);
        T objective = NumOps.FromDouble(loss / n);
        T residualSum = NumOps.FromDouble(residualTotal);

        if (Regularization != null)
        {
            var zero = new Vector<T>(p);
            var atEnd = Regularization.Regularize(zero, coefficients);
            var atMiddle = Regularization.Regularize(zero, coefficients.Multiply(NumOps.FromDouble(0.5)));
            objective = NumOps.Add(objective, coefficients.DotProduct(atMiddle));
            coefficientGradient = coefficientGradient.Add(atEnd);
        }

        var gradient = new Vector<T>(p + 1);
        for (int j = 0; j < p; j++)
            gradient[j] = coefficientGradient[j];
        gradient[p] = NumOps.Divide(residualSum, NumOps.FromDouble(n));
        return (objective, gradient);
    }

    /// <summary>
    /// Computes the per-feature mean and standard deviation used to standardize the design matrix.
    /// </summary>
    /// <param name="x">The design matrix.</param>
    /// <param name="featureCount">Number of columns in <paramref name="x"/>.</param>
    /// <returns>The per-feature mean and standard deviation; a constant column gets a deviation of 1.</returns>
    private (Vector<T> Mean, Vector<T> Std) ComputeStandardization(Matrix<T> x, int featureCount)
    {
        var mean = new Vector<T>(featureCount);
        var std = new Vector<T>(featureCount);
        int n = x.Rows;

        // Row by row, each column's sums accumulating in row order (the order a column-by-column pass would take).
        var sums = new double[featureCount];
        for (int i = 0; i < n; i++)
        {
            var row = x.GetRowReadOnlySpan(i);
            for (int j = 0; j < featureCount; j++) sums[j] += NumOps.ToDouble(row[j]);
        }

        var means = new double[featureCount];
        for (int j = 0; j < featureCount; j++) means[j] = sums[j] / n;

        var squares = new double[featureCount];
        for (int i = 0; i < n; i++)
        {
            var row = x.GetRowReadOnlySpan(i);
            for (int j = 0; j < featureCount; j++)
            {
                double d = NumOps.ToDouble(row[j]) - means[j];
                squares[j] += d * d;
            }
        }

        for (int j = 0; j < featureCount; j++)
        {
            double s = Math.Sqrt(squares[j] / n);
            if (s < 1e-10) s = 1.0;

            mean[j] = NumOps.FromDouble(means[j]);
            std[j] = NumOps.FromDouble(s);
        }

        return (mean, std);
    }

    /// <summary>
    /// The design matrix with each column centered and scaled, row-major in double precision, and the 0/1 targets:
    /// the data every evaluation of the objective reads.
    /// </summary>
    private (double[] X, double[] Y) StandardizedDoubles(Matrix<T> x, Vector<T> y, Vector<T> mean, Vector<T> std)
    {
        int p = x.Columns;
        var meanValues = new double[p];
        var stdValues = new double[p];
        for (int j = 0; j < p; j++)
        {
            meanValues[j] = NumOps.ToDouble(mean[j]);
            stdValues[j] = NumOps.ToDouble(std[j]);
        }

        var result = new double[x.Rows * p];
        for (int i = 0; i < x.Rows; i++)
        {
            var source = x.GetRowReadOnlySpan(i);
            for (int j = 0; j < p; j++)
                result[i * p + j] = (NumOps.ToDouble(source[j]) - meanValues[j]) / stdValues[j];
        }

        var targets = new double[y.Length];
        for (int i = 0; i < y.Length; i++) targets[i] = NumOps.ToDouble(y[i]);
        return (result, targets);
    }

    /// <summary>
    /// Makes predictions for new data points using the trained logistic regression model.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <returns>A vector of predicted probabilities for the positive class.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the predicted probabilities for each sample in the input feature matrix.
    /// It computes the raw scores by multiplying the features by the learned coefficients and adding the intercept,
    /// then transforms these scores into probabilities using the sigmoid function.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model makes predictions on new data.
    /// 
    /// During prediction:
    /// - The model calculates a score for each example using the learned weights
    /// - The score is converted to a probability between 0 and 1 using the sigmoid function
    /// - Values closer to 1 indicate higher confidence in the positive class
    /// 
    /// For instance, if you've trained the model to detect fraudulent transactions, 
    /// a probability of 0.92 would suggest the transaction is likely fraudulent,
    /// while 0.03 would suggest it's probably legitimate.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> x)
    {
        Vector<T> scores = x.Multiply(Coefficients).Add(Intercept);
        return scores.Transform(Sigmoid);
    }

    /// <summary>
    /// Applies the sigmoid (logistic) function to transform a raw score into a probability.
    /// </summary>
    /// <param name="x">The raw score to transform.</param>
    /// <returns>A probability value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// The sigmoid function transforms any real number into a value between 0 and 1, which can be interpreted as a probability.
    /// It is defined as f(x) = 1 / (1 + e^(-x)), where e is the base of the natural logarithm.
    /// </para>
    /// <para><b>For Beginners:</b> The sigmoid function converts any number into a probability between 0 and 1.
    /// 
    /// The sigmoid function:
    /// - Takes any number as input (from negative infinity to positive infinity)
    /// - Always outputs a number between 0 and 1
    /// - Has an S-shaped curve that flattens at the extremes
    /// 
    /// For example:
    /// - Large positive numbers (like +5) become close to 1
    /// - Large negative numbers (like -5) become close to 0
    /// - Zero becomes exactly 0.5
    /// 
    /// This is perfect for binary classification because it converts raw scores into probabilities.
    /// </para>
    /// </remarks>
    private T Sigmoid(T x)
    {
        T expNegX = NumOps.Exp(NumOps.Negate(x));
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }
}
