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
public class QuantileRegression<T> : RegressionBase<T>
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
        int regressionParameterCount = p + (Options.UseIntercept ? 1 : 0);
        int variableCount = regressionParameterCount + 2 * n;
        long denseEntries = checked((long)n * variableCount);
        if (denseEntries > _options.MaximumDenseLinearProgramEntries)
        {
            throw new InvalidOperationException(
                $"Exact quantile regression requires {denseEntries:N0} dense matrix entries for {n:N0} rows and {p:N0} features, " +
                $"which exceeds the configured budget of {_options.MaximumDenseLinearProgramEntries:N0}. " +
                "Use fewer rows, raise MaximumDenseLinearProgramEntries when sufficient memory is available, " +
                "or choose a large-scale quantile estimator.");
        }
        int interceptColumn = 0;
        int coefficientColumn = Options.UseIntercept ? 1 : 0;
        int positiveResidualColumn = regressionParameterCount;
        int negativeResidualColumn = regressionParameterCount + n;

        double quantile = _options.Quantile;
        var objective = new Vector<T>(variableCount);
        for (int i = 0; i < n; i++)
        {
            objective[positiveResidualColumn + i] = NumOps.FromDouble(quantile);
            objective[negativeResidualColumn + i] = NumOps.FromDouble(1.0 - quantile);
        }

        var equalityMatrix = new Matrix<T>(n, variableCount);
        var equalityBounds = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            if (Options.UseIntercept)
            {
                equalityMatrix[i, interceptColumn] = NumOps.One;
            }
            for (int j = 0; j < p; j++) equalityMatrix[i, coefficientColumn + j] = x[i, j];
            equalityMatrix[i, positiveResidualColumn + i] = NumOps.One;
            equalityMatrix[i, negativeResidualColumn + i] = NumOps.Negate(NumOps.One);
            equalityBounds[i] = y[i];
        }

        // The intercept and slopes are unrestricted in sign; the residual parts are non-negative.
        var lowerBounds = new Vector<T>(variableCount);
        var upperBounds = new Vector<T>(variableCount);
        var negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);
        var positiveInfinity = NumOps.FromDouble(double.PositiveInfinity);
        for (int c = 0; c < variableCount; c++)
        {
            lowerBounds[c] = c < regressionParameterCount ? negativeInfinity : NumOps.Zero;
            upperBounds[c] = positiveInfinity;
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

        Intercept = Options.UseIntercept ? solution.Solution[interceptColumn] : NumOps.Zero;
        var coefficients = new Vector<T>(p);
        for (int j = 0; j < p; j++) coefficients[j] = solution.Solution[coefficientColumn + j];

        // Regularization is applied to the fitted coefficients, matching every other regression in
        // the library; the intercept is deliberately left unpenalized.
        Coefficients = Regularization.Regularize(coefficients);
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

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes both the base class data and the quantile regression specific options,
    /// including the quantile, solver configuration, and memory safety budget.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize QuantileRegression specific data
        writer.Write(_options.Quantile);
        writer.Write(_options.SolverOptions.MaxIterations);
        writer.Write(_options.SolverOptions.Tolerance);
        writer.Write(_options.SolverOptions.DegeneratePivotsBeforeBlandsRule);
        writer.Write(_options.MaximumDenseLinearProgramEntries);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes both the base class data and the quantile regression specific options,
    /// reconstructing the model's state from the serialized data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize QuantileRegression specific data
        _options.Quantile = reader.ReadDouble();
        _options.SolverOptions.MaxIterations = reader.ReadInt32();
        _options.SolverOptions.Tolerance = reader.ReadDouble();
        _options.SolverOptions.DegeneratePivotsBeforeBlandsRule = reader.ReadInt32();
        _options.MaximumDenseLinearProgramEntries = reader.ReadInt64();
    }

    /// <summary>
    /// Creates a new instance of the quantile regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the quantile regression model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the quantile regression model with the same configuration
    /// options and regularization method as the current instance, but without copying the trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters.
    /// 
    /// Think of it like getting a blank notepad with the same paper quality and size, 
    /// but without any writing on it yet. The new model has the same:
    /// - Quantile setting (which part of the distribution you're estimating)
    /// - Learning rate (how quickly the model adjusts during training)
    /// - Maximum iterations (how long the model will train)
    /// - Regularization settings (safeguards against overfitting)
    /// 
    /// But it doesn't have any of the coefficient values that were learned from data.
    /// 
    /// This is mainly used internally when doing things like cross-validation or 
    /// creating ensembles of similar models with different training data.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        return new QuantileRegression<T>(new QuantileRegressionOptions<T>(_options), Regularization);
    }
}
