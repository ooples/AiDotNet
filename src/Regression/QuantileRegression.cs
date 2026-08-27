using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Quantile Regression, a technique that estimates the conditional quantiles of a response variable
/// distribution in the linear model, providing a more complete view of the relationship between variables.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Unlike ordinary least squares regression which estimates the conditional mean of the response variable,
/// quantile regression estimates the conditional median or other quantiles of the response variable.
/// This makes it robust to outliers and useful for modeling heterogeneous conditional distributions.
/// </para>
/// <para>
/// The algorithm solves the Koenker-Bassett linear-program formulation of quantile loss exactly.
/// </para>
/// <para>
/// <b>For Beginners:</b> While standard regression tells you about the average relationship between variables, quantile regression
/// lets you explore different parts of the data distribution. For example, median regression (quantile=0.5)
/// tells you about the middle of the distribution, while quantile=0.9 tells you about the upper end.
/// This is useful when you suspect that the relationship between variables might be different for different
/// ranges of the outcome.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a quantile regression model (e.g., median regression with tau=0.5)
/// var options = new QuantileRegressionOptions&lt;double&gt;();
/// var model = new QuantileRegression&lt;double&gt;(options);
///
/// // Prepare training data: 5 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(5, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10 });
/// var targets = new Vector&lt;double&gt;(new double[] { 2.5, 5.3, 8.1, 10.9, 13.7 });
///
/// // Train to estimate conditional quantile of the response
/// model.Train(features, targets);
///
/// // Predict the specified quantile for a new sample
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 11, 12 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Regression Quantiles", "https://doi.org/10.2307/1913643", Year = 1978, Authors = "Roger Koenker, Gilbert Bassett Jr.")]
public partial class QuantileRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the quantile regression model.
    /// </summary>
    /// <value>
    /// Contains the quantile, exact-solver settings, and a dense-memory safety budget.
    /// </value>
    private readonly QuantileRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the QuantileRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the quantile regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the quantile regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public QuantileRegression(QuantileRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new QuantileRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the quantile regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method builds and solves the exact linear-program formulation of quantile regression.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Training is the process where the model learns from your data. The algorithm starts with initial guesses
    /// for the coefficients and then iteratively improves them. At each step, it calculates how far off its
    /// predictions are, but unlike standard regression, it penalizes over-predictions and under-predictions
    /// differently based on the quantile you specified. It then adjusts the coefficients to reduce these errors.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;
        TrainingFeatureCount = p;

        if (n == 0)
        {
            throw new ArgumentException("Quantile regression requires at least one observation.", nameof(x));
        }

        // Quantile regression is solved EXACTLY as a linear program (Koenker and Bassett, 1978 —
        // the paper that introduced the method). Splitting each residual into its positive and
        // negative parts, u_i - v_i = y_i - (b0 + x_i . b) with u_i, v_i >= 0, turns the pinball
        // loss into something linear:
        //
        //     minimize  sum_i [ tau * u_i + (1 - tau) * v_i ]
        //     subject to  b0 + x_i . b + u_i - v_i = y_i     for every i
        //                 u_i >= 0,  v_i >= 0,  b0 and b free
        //
        // At the optimum exactly one of u_i, v_i is non-zero for each observation, so the objective
        // really is the asymmetric absolute loss: over-predictions are charged (1 - tau) and
        // under-predictions tau.
        //
        // Two things were wrong before. First, the method returned the ORDINARY LEAST SQUARES fit
        // whenever there was at least one feature — the quantile-specific code below that early
        // return was unreachable — so it estimated the conditional MEAN regardless of the requested
        // quantile, which is the one thing quantile regression exists not to do. Second, the
        // unreachable code ran gradient descent on the pinball loss, which is not differentiable at
        // zero, exactly where the optimum sits.
        // Koenker and Bassett's DUAL formulation, not the primal. The primal has one equality row
        // per observation and two slack variables per observation, so a 200-row fit builds a
        // 200-by-600 tableau and takes thousands of pivots — which at this library's generic
        // arithmetic (every add and multiply is a virtual call) runs for minutes. The dual has one
        // row per REGRESSION PARAMETER, typically two or three, and one bounded variable per
        // observation:
        //
        //     maximize  yᵀa   subject to  Zᵀa = 0,  τ−1 ≤ a ≤ τ,   Z = [1 | X]
        //
        // which is a 3-by-200 tableau for the same fit. This is the formulation Koenker's own
        // software solves, and it is why quantile regression is practical at all at scale.
        int regressionParameterCount = p + (Options.UseIntercept ? 1 : 0);
        long denseEntries = checked((long)regressionParameterCount * n);
        if (denseEntries > _options.MaximumDenseLinearProgramEntries)
        {
            throw new InvalidOperationException(
                $"Exact quantile regression requires {denseEntries:N0} dense matrix entries for {n:N0} rows and {p:N0} features, " +
                $"which exceeds the configured budget of {_options.MaximumDenseLinearProgramEntries:N0}. " +
                "Use fewer rows, raise MaximumDenseLinearProgramEntries when sufficient memory is available, " +
                "or choose a large-scale quantile estimator.");
        }

        double quantile = _options.Quantile;

        // Z = [1 | X], the design matrix including the intercept column when one is fitted.
        var design = new Matrix<T>(n, regressionParameterCount);
        int coefficientColumn = Options.UseIntercept ? 1 : 0;
        for (int i = 0; i < n; i++)
        {
            if (Options.UseIntercept) design[i, 0] = NumOps.One;
            for (int j = 0; j < p; j++) design[i, coefficientColumn + j] = x[i, j];
        }

        // The solver minimizes, so the dual's maximization of yᵀa is posed as minimizing −yᵀa.
        var objective = new Vector<T>(n);
        for (int i = 0; i < n; i++) objective[i] = NumOps.Negate(y[i]);

        var equalityMatrix = new Matrix<T>(regressionParameterCount, n);
        for (int r = 0; r < regressionParameterCount; r++)
        {
            for (int i = 0; i < n; i++) equalityMatrix[r, i] = design[i, r];
        }

        var equalityBounds = new Vector<T>(regressionParameterCount);

        var lowerBounds = new Vector<T>(n);
        var upperBounds = new Vector<T>(n);
        T lowerBound = NumOps.FromDouble(quantile - 1.0);
        T upperBound = NumOps.FromDouble(quantile);
        for (int i = 0; i < n; i++)
        {
            lowerBounds[i] = lowerBound;
            upperBounds[i] = upperBound;
        }

        var program = new AiDotNet.Solvers.LinearProgramming.LinearProgram<T>(
            objective,
            equalityMatrix: equalityMatrix,
            equalityBounds: equalityBounds,
            lowerBounds: lowerBounds,
            upperBounds: upperBounds);

        var solver = new AiDotNet.Solvers.LinearProgramming.SimplexSolver<T>(
            new SimplexSolverOptions(_options.SolverOptions));

        var solution = solver.Solve(program);

        if (solution.Status != AiDotNet.Solvers.LinearProgramming.LinearProgramStatus.Optimal || solution.Solution is null)
        {
            throw new InvalidOperationException(
                $"The quantile regression linear program did not solve (status {solution.Status}). " +
                "This usually means the design matrix contains non-finite values.");
        }

        var fitted = RecoverCoefficients(
            design, y, solution.Solution, regressionParameterCount, quantile);

        Intercept = Options.UseIntercept ? fitted[0] : NumOps.Zero;
        var coefficients = new Vector<T>(p);
        for (int j = 0; j < p; j++) coefficients[j] = fitted[coefficientColumn + j];

        // Regularization is applied to the fitted coefficients, matching every other regression in
        // the library; the intercept is deliberately left unpenalized.
        Coefficients = Regularization.Regularize(coefficients);
    }

    /// <summary>
    /// Recovers the regression coefficients from the dual solution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A quantile regression fit passes exactly through <c>k</c> of the observations, where <c>k</c>
    /// is the number of estimated parameters — that is the defining geometric property of the
    /// solution, and it is what makes the fit robust to outliers in the response. Complementary
    /// slackness says those interpolated points are precisely the ones whose dual variable sits
    /// strictly between its bounds: a dual variable pinned at <c>τ</c> or <c>τ−1</c> marks an
    /// observation the fit passes above or below, while one in between marks an observation the fit
    /// passes through.
    /// </para>
    /// <para>
    /// So the coefficients follow from solving the small system <c>Z_h·β = y_h</c> over just those
    /// rows. Recovering them this way rather than from the dual's own multipliers keeps the result
    /// independent of the solver's sign convention for equality duals, and gives a fit that
    /// satisfies the interpolation property exactly rather than to a tolerance.
    /// </para>
    /// <para>
    /// Degeneracy — ties in the data, or a design in which more or fewer than <c>k</c> points end up
    /// interior — is handled by least squares over whichever rows are interior, which reduces to the
    /// exact interpolation when there are exactly <c>k</c> of them.
    /// </para>
    /// </remarks>
    private Vector<T> RecoverCoefficients(
        Matrix<T> design,
        Vector<T> responses,
        Vector<T> dualSolution,
        int parameterCount,
        double quantile)
    {
        int n = design.Rows;

        // A dual variable is "interior" when it is strictly inside [τ−1, τ]. The tolerance is
        // relative to the interval's width so it means the same thing at any quantile.
        double interiorTolerance = 1e-7;
        var interior = new List<int>(parameterCount);
        for (int i = 0; i < n; i++)
        {
            double value = NumOps.ToDouble(dualSolution[i]);
            if (value > quantile - 1.0 + interiorTolerance && value < quantile - interiorTolerance)
            {
                interior.Add(i);
            }
        }

        // Too few interior points to pin the fit down: fall back to every row, which is the
        // least-absolute-deviations fit's least-squares shadow rather than a wrong answer.
        var rows = interior.Count >= parameterCount ? interior : BuildAllRows(n);

        // Normal equations over the selected rows. With exactly parameterCount rows this reproduces
        // the interpolation Z_h β = y_h exactly; with more it is the least-squares compromise.
        var normal = new Matrix<T>(parameterCount, parameterCount);
        var rightHandSide = new Vector<T>(parameterCount);

        for (int r = 0; r < parameterCount; r++)
        {
            for (int c = 0; c < parameterCount; c++)
            {
                T accumulator = NumOps.Zero;
                foreach (int row in rows)
                {
                    accumulator = NumOps.Add(
                        accumulator, NumOps.Multiply(design[row, r], design[row, c]));
                }

                normal[r, c] = accumulator;
            }

            T target = NumOps.Zero;
            foreach (int row in rows)
            {
                target = NumOps.Add(target, NumOps.Multiply(design[row, r], responses[row]));
            }

            rightHandSide[r] = target;
        }

        // A ridge term keeps a rank-deficient selection solvable; it is far below the scale of the
        // data and vanishes from a well-determined fit.
        T ridge = NumOps.FromDouble(1e-12);
        for (int r = 0; r < parameterCount; r++)
        {
            normal[r, r] = NumOps.Add(normal[r, r], ridge);
        }

        try
        {
            return new AiDotNet.DecompositionMethods.MatrixDecomposition.LuDecomposition<T>(normal)
                .Solve(rightHandSide);
        }
        catch (Exception)
        {
            return new Vector<T>(parameterCount);
        }
    }

    private static List<int> BuildAllRows(int count)
    {
        var rows = new List<int>(count);
        for (int i = 0; i < count; i++) rows.Add(i);
        return rows;
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the specified quantile of the conditional distribution for each input example.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After training, this method is used to make predictions on new data. For each example in your input data,
    /// it calculates the predicted value at the quantile you specified. For instance, if you set quantile=0.5,
    /// it predicts the median value; if you set quantile=0.9, it predicts the value below which 90% of the
    /// observations would fall.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = Predict(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Predicts the value for a single input vector.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted value at the specified quantile.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the dot product of the input vector and the coefficients, then adds the intercept.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the core prediction function that calculates the predicted value for a single example.
    /// It multiplies each feature value by its corresponding coefficient, sums these products, and adds
    /// the intercept term to get the final prediction.
    /// </para>
    /// </remarks>
    private T Predict(Vector<T> input)
    {
        return NumOps.Add(Coefficients.DotProduct(input), Intercept);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type and the quantile being estimated.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Model metadata provides information about the model itself, rather than the predictions it makes.
    /// For quantile regression, this includes which quantile the model is estimating (e.g., median, 90th percentile).
    /// This information can be useful for understanding and comparing different models.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Quantile"] = _options.Quantile;

        return metadata;
    }
}
