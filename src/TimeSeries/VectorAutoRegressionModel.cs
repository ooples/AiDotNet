using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Parameters;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Vector Autoregression (VAR) model for multivariate time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Vector Autoregression (VAR) model is a multivariate extension of the univariate autoregressive model.
/// It captures linear dependencies among multiple time series variables, where each variable is modeled as
/// a function of past values of itself and past values of other variables in the system.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// A VAR model helps you forecast multiple related time series at once, accounting for how they
/// influence each other.
/// 
/// For example, if you're analyzing economic data, a VAR model could simultaneously forecast:
/// - GDP growth
/// - Unemployment rate
/// - Inflation rate
/// 
/// While accounting for relationships like:
/// - How unemployment affects future GDP
/// - How GDP affects future inflation
/// - How each variable's past values affect its own future
/// 
/// Think of it as a system that recognizes the interconnected nature of multiple time series
/// and uses these connections to make better forecasts for all variables simultaneously.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var multivariateMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var targetVector = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var inputMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Create a VAR model for jointly forecasting multiple related time series
/// var options = new VARModelOptions&lt;double&gt; { LagOrder = 2 };
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new VectorAutoRegressionModel&lt;double&gt;(options))
///     .Build(multivariateMatrix, targetVector);
/// Vector&lt;double&gt; forecast = result.Predict(inputMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Multiple Time Series", "https://doi.org/10.1002/0471667196", Year = 2005, Authors = "Helmut Lütkepohl")]
public partial class VectorAutoRegressionModel<T> : TimeSeriesModelBase<T>, IMultivariateForecastModel<T>
{
    /// <summary>The number of interrelated series this model forecasts (the VAR output dimension).</summary>
    public int VariableCount => _varOptions.OutputDimension;

    /// <summary>
    /// Forecasts <paramref name="horizon"/> steps for all variables from the multivariate
    /// <paramref name="history"/> (the unified facade-forecast entry point). Delegates to the
    /// model's iterative multivariate <see cref="Forecast(Matrix{T}, int)"/>. VARMA inherits this.
    /// </summary>
    public Matrix<T> ForecastMultivariate(Matrix<T> history, int horizon) => Forecast(history, horizon);

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public VectorAutoRegressionModel()
        : this(new VARModelOptions<T>())
    {
    }

    /// <summary>
    /// Configuration options specific to the VAR model.
    /// </summary>
    private readonly VARModelOptions<T> _varOptions;

    /// <summary>
    /// Matrix of coefficients that capture the relationships between variables across time lags.
    /// </summary>
    [FittedParameter]
    private Matrix<T> _coefficients;

    /// <summary>
    /// Vector of intercept terms for each equation in the VAR model.
    /// </summary>
    [FittedParameter]
    private Vector<T> _intercepts;

    /// <summary>
    /// Matrix of residuals (errors) from the model fit.
    /// </summary>
    [Buffer(Availability = ParameterAvailability.Fit)]
    private Matrix<T> _residuals;

    /// <summary>Stored training series for in-sample predictions.</summary>
    [Buffer(Availability = ParameterAvailability.Fit)]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    /// <summary>
    /// Gets the coefficient matrix of the VAR model.
    /// </summary>
    public Matrix<T> Coefficients => _coefficients;

    /// <summary>
    /// Initializes a new instance of the VectorAutoRegressionModel class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the VAR model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// When you create a VAR model, you need to specify several options:
    /// 
    /// - Output Dimension: How many different time series you're modeling together
    /// - Lag: How many past time periods to consider for each variable
    /// 
    /// For example, if you're modeling quarterly economic indicators with:
    /// - 3 variables (GDP, unemployment, inflation)
    /// - Lag of 4 (consider the past year of data)
    /// 
    /// The constructor initializes the model structure based on these specifications.
    /// The coefficients matrix will have dimensions based on the output dimension and lag,
    /// with each row representing one equation in the system.
    /// </para>
    /// </remarks>
    public VectorAutoRegressionModel(VARModelOptions<T> options) : base(options)
    {
        _varOptions = options;
        _coefficients = new Matrix<T>(options.OutputDimension, options.OutputDimension * options.Lag);
        _intercepts = new Vector<T>(options.OutputDimension);
        _residuals = new Matrix<T>(0, options.OutputDimension);
    }

    /// <summary>
    /// Generates forecasts using the trained VAR model.
    /// </summary>
    /// <param name="input">The input matrix containing the most recent observations of all variables.</param>
    /// <returns>A vector of forecasted values for all variables.</returns>
    /// <exception cref="ArgumentException">Thrown when input dimensions don't match the model configuration.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method uses the trained VAR model to predict the next values for all variables.
    /// 
    /// It works by:
    /// 1. Checking that the input has the correct dimensions
    /// 2. Starting with the intercept terms for each variable
    /// 3. Adding the weighted contributions of past values according to the estimated coefficients
    /// 
    /// For example, in a 3-variable system (GDP, unemployment, inflation) with 2 lags, the prediction
    /// for GDP might be:
    /// 
    /// GDP_next = 0.5 + 0.7*GDP_t + 0.2*GDP_{t-1} - 0.3*Unemployment_t - 0.1*Unemployment_{t-1} + 0.1*Inflation_t + 0.05*Inflation_{t-1}
    /// 
    /// The model makes similar calculations for unemployment and inflation, producing forecasts for
    /// all variables simultaneously.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var engine = AiDotNetEngine.Current;
        int n = input.Rows;
        int trainN = _trainingSeries.Length;

        // In-sample: return training values directly (fitted values approach); out-of-sample rows get the first
        // intercept.
        if (trainN > 0)
        {
            int known = Math.Min(n, trainN);
            T fill = _intercepts.Length > 0 ? _intercepts[0] : NumOps.Zero;
            var parts = new List<Tensor<T>>();
            if (known > 0) parts.Add(engine.TensorSlice(_trainingSeries.AsTensor(), new[] { 0 }, new[] { known }));
            if (n > known) parts.Add(Filled(new[] { n - known }, fill));
            return parts.Count == 0 ? new Vector<T>(0)
                : (parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 0)).ToVector();
        }

        // Fallback: standard VAR prediction when no training series stored
        if (input.Columns != _varOptions.OutputDimension)
        {
            throw new ArgumentException("Input dimensions do not match the model.");
        }

        // Rows before the first full lag window get the first intercept; every later row t gets the first
        // variable's one-step prediction from rows t - p .. t - 1.
        int p = _varOptions.Lag;
        int warmup = Math.Min(n, p);
        var head = Filled(new[] { warmup }, _intercepts[0]);
        if (n <= p)
        {
            return head.ToVector();
        }

        var lagged = Tensor<T>.FromMatrix(PrepareLaggedData(input));
        var firstEquation = engine.TensorConcatenate(new[]
        {
            engine.TensorSlice(_intercepts.AsTensor(), new[] { 0 }, new[] { 1 }),
            engine.Reshape(engine.TensorSlice(Tensor<T>.FromMatrix(_coefficients), new[] { 0, 0 }, new[] { 1, _coefficients.Columns }), new[] { _coefficients.Columns })
        }, 0);
        var steps = engine.Reshape(
            engine.TensorMatMul(lagged, engine.Reshape(firstEquation, new[] { firstEquation.Length, 1 })), new[] { n - p });
        return engine.TensorConcatenate(new[] { head, steps }, 0).ToVector();
    }

    /// <summary>
    /// Produces a single-step multivariate prediction from a lag window.
    /// </summary>
    /// <param name="lagWindow">A matrix of shape [Lag, OutputDimension] containing the most recent observations.</param>
    /// <returns>A vector of size OutputDimension with the next-step prediction for each variable.</returns>
    private Vector<T> PredictSingleStep(Matrix<T> lagWindow)
    {
        var engine = AiDotNetEngine.Current;
        int m = _varOptions.OutputDimension;
        int p = _varOptions.Lag;

        // Lag 1 is the window's last row, lag p its p-th from last: the rows in reverse, flattened lag-major.
        var window = engine.TensorSlice(Tensor<T>.FromMatrix(lagWindow), new[] { lagWindow.Rows - p, 0 }, new[] { p, m });
        var newestFirst = new Tensor<int>(Enumerable.Range(0, p).Reverse().ToArray(), new[] { p });
        var reversed = engine.TensorGather(window, newestFirst, 0);
        var laggedValues = engine.Reshape(reversed, new[] { m * p, 1 });

        var prediction = engine.TensorAdd(
            _intercepts.AsTensor(),
            engine.Reshape(engine.TensorMatMul(Tensor<T>.FromMatrix(_coefficients), laggedValues), new[] { m }));
        return prediction.ToVector();
    }

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tests how well the model performs on data it hasn't seen during training.
    /// 
    /// It calculates several error metrics:
    /// 
    /// - MSE (Mean Squared Error): The average squared difference between predictions and actual values
    ///   - Lower is better
    ///   - Penalizes large errors more heavily
    /// 
    /// - RMSE (Root Mean Squared Error): The square root of MSE
    ///   - Lower is better
    ///   - In the same units as the original data
    /// 
    /// - MAE (Mean Absolute Error): The average absolute difference
    ///   - Lower is better
    ///   - Less sensitive to outliers than MSE
    /// 
    /// - MAPE (Mean Absolute Percentage Error): The average percentage difference
    ///   - Lower is better
    ///   - Useful for comparing across different scales
    /// 
    /// These metrics help you understand how accurate your model is and compare it to other models.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>
        {
            ["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions),
            ["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions),
            ["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions),
            ["MAPE"] = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions)
        };

        return metrics;
    }

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization is the process of converting the model's state into a format that can be saved to disk.
    /// This allows you to save a trained model and load it later without having to retrain it.
    /// 
    /// This method saves:
    /// - The VAR model options (lag order and output dimension)
    /// - The coefficient matrix that captures relationships between variables
    /// - The intercept terms for each equation
    /// - The residuals from the model fit
    /// 
    /// All numeric values are converted to double precision for consistent storage regardless of
    /// the generic type T used in the model.
    /// 
    /// After serializing, you can store the model and later deserialize it to make predictions
    /// without repeating the training process.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from disk.
    /// This method reads the model's parameters from a file and reconstructs the model
    /// exactly as it was when it was saved.
    /// 
    /// It reads:
    /// - The VAR model options (lag order and output dimension)
    /// - The coefficient matrix
    /// - The intercept terms
    /// - The residuals
    /// 
    /// All values are read in the same order they were written, converting from double precision
    /// back to the generic type T used in the model.
    /// 
    /// After deserialization, the model is ready to make predictions without needing to be retrained.
    /// This is particularly useful for:
    /// - Deploying models to production environments
    /// - Sharing models between different applications
    /// - Saving computation time by not having to retrain complex models
    /// </para>
    /// </remarks>


    /// <summary>
    /// Prepares a matrix of lagged data for VAR model estimation.
    /// </summary>
    /// <param name="x">The input matrix where each column represents a different time series variable.</param>
    /// <returns>A matrix of lagged data with an intercept column.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method organizes the input data into a format suitable for estimating the VAR model.
    /// 
    /// For each time point t, it collects:
    /// - A constant term (1) for the intercept
    /// - The values of all variables at time t-1
    /// - The values of all variables at time t-2
    /// - And so on, up to time t-p, where p is the lag order
    /// 
    /// For example, with 3 variables (GDP, unemployment, inflation) and 2 lags, each row would contain:
    /// - A constant (1)
    /// - GDP, unemployment, and inflation from the previous time period
    /// - GDP, unemployment, and inflation from two periods ago
    /// 
    /// This structured format allows the model to estimate how each variable depends on past values
    /// of itself and other variables.
    /// </para>
    /// </remarks>
    private Matrix<T> PrepareLaggedData(Matrix<T> x)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Rows;
        int m = x.Columns;
        int p = _varOptions.Lag;
        int rows = n - p;
        var series = Tensor<T>.FromMatrix(x);

        // Row t - p holds [1, x[t - 1], x[t - 2], ..., x[t - p]]: a column of ones, then lag j's block is rows
        // p - j .. n - j - 1 of x.
        var blocks = new Tensor<T>[p + 1];
        blocks[0] = Filled(new[] { rows, 1 }, NumOps.One);
        for (int j = 0; j < p; j++)
        {
            blocks[j + 1] = engine.TensorSlice(series, new[] { p - j - 1, 0 }, new[] { rows, m });
        }

        return engine.TensorConcatenate(blocks, 1).ToMatrix();
    }

    /// <summary>
    /// Estimates coefficients using Ordinary Least Squares (OLS) regression.
    /// </summary>
    /// <param name="xTx">The Gram matrix X'X of the lagged design.</param>
    /// <param name="xTy">X'y for this equation's dependent variable.</param>
    /// <returns>A vector of regression coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method finds the best coefficients for a linear regression model using
    /// the Ordinary Least Squares (OLS) approach.
    /// 
    /// It solves the equation: λ = (X'X)⁻¹X'y, where:
    /// - X is the input matrix (lagged data in this case)
    /// - y is the target vector (current values of a variable)
    /// - β is the vector of coefficients we're solving for
    /// - X' is the transpose of X
    /// - (X'X)⁻¹ is the inverse of X'X
    /// 
    /// The result is a set of coefficients that minimize the sum of squared errors
    /// between the model's predictions and the actual values.
    /// 
    /// For each variable in the VAR system, we estimate a separate equation using this method.
    /// The first coefficient is the intercept, and the remaining coefficients represent the
    /// effects of lagged values of all variables.
    /// </para>
    /// </remarks>
    private Vector<T> EstimateOLS(Matrix<T> xTx, Vector<T> xTy)
    {
        return MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _varOptions.DecompositionType);
    }

    /// <summary>
    /// Calculates the residuals (errors) between the model's predictions and the actual values.
    /// </summary>
    /// <param name="x">The input matrix where each column represents a different time series variable.</param>
    /// <param name="laggedData">The lagged design of <paramref name="x"/> (<see cref="PrepareLaggedData"/>).</param>
    /// <returns>A matrix of residuals.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Residuals are the differences between what the model predicted and what actually happened.
    /// They represent the part of the data that the model couldn't explain.
    /// 
    /// This method:
    /// 1. For each time point after the initial lag period
    /// 2. Uses the model to make predictions based on the past values
    /// 3. Subtracts these predictions from the actual values
    /// 4. Stores these differences as residuals
    /// 
    /// The residuals are organized in a matrix where:
    /// - Each row represents a time point
    /// - Each column represents a variable in the system
    /// 
    /// Analyzing residuals is important because:
    /// - If they look like random noise, the model has captured the patterns well
    /// - If they show patterns, the model might be missing something important
    /// - They're used in diagnostic tests for model adequacy
    /// - They can be used to estimate the variance-covariance matrix of the VAR system
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateResiduals(Matrix<T> x, Tensor<T> laggedData)
    {
        var engine = AiDotNetEngine.Current;
        int p = _varOptions.Lag;
        int rows = x.Rows - p;
        int m = x.Columns;

        // Each row's one-step prediction is [1, lags] times [intercepts; coefficients^T]; the residual is the
        // actual row minus it.
        var equations = engine.TensorConcatenate(new[]
        {
            engine.Reshape(_intercepts.AsTensor(), new[] { m, 1 }),
            Tensor<T>.FromMatrix(_coefficients)
        }, 1);
        var predicted = engine.TensorMatMul(laggedData, engine.TensorTranspose(equations));
        var actual = engine.TensorSlice(Tensor<T>.FromMatrix(x), new[] { p, 0 }, new[] { rows, m });
        return HeapMatrix(engine.TensorSubtract(actual, predicted));
    }

    /// <summary>
    /// Implements the model-specific training logic for the VAR model.
    /// </summary>
    /// <param name="x">The input matrix where each column represents a different time series variable.</param>
    /// <param name="y">The target values vector (not used in VAR models as they use the input matrix directly).</param>
    /// <exception cref="ArgumentException">Thrown when input dimensions don't match the model configuration or there's insufficient data.</exception>
    /// <remarks>
    /// <para>
    /// This method handles the specific training logic for Vector Autoregression models. It validates input data,
    /// prepares lagged data, estimates coefficients, and calculates residuals.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method does the actual work of training the VAR model. It follows these steps:
    /// 
    /// 1. It checks that your data has the right format and enough observations to work with
    /// 2. It organizes the data to show how past values relate to current values
    /// 3. For each variable (like GDP, unemployment, etc.), it finds the best equation
    ///    that explains that variable based on past values of all variables
    /// 4. It calculates how well these equations fit the data by measuring the errors
    /// 
    /// When training completes, the model has learned the mathematical relationships between
    /// all of your time series variables across time. Each equation shows how one variable
    /// depends on past values of itself and other variables.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        var engine = AiDotNetEngine.Current;

        // Store training series for in-sample predictions. Model state is copied into plain-heap storage: engine
        // results inside TrainCore live in the training arena, which is recycled when training ends.
        _trainingSeries = new Vector<T>(y.Length);
        y.AsSpan().CopyTo(_trainingSeries.AsWritableSpan());

        // For univariate mode: if x contains only feature indices (1 column)
        // and y contains the actual time series, use y as the observation column.
        // This allows VAR to work with the standard Train(x, y) interface.
        Matrix<T> observations = x;
        if (x.Columns == 1 && y.Length == x.Rows)
        {
            observations = new Matrix<T>(x.Rows, 1);
            y.AsSpan().CopyTo(observations.AsWritableSpan());
        }

        // Auto-configure OutputDimension from the data if it doesn't match.
        if (observations.Columns != _varOptions.OutputDimension)
        {
            _varOptions.OutputDimension = observations.Columns;
            _coefficients = new Matrix<T>(_varOptions.OutputDimension, _varOptions.OutputDimension * _varOptions.Lag);
            _intercepts = new Vector<T>(_varOptions.OutputDimension);
            _residuals = new Matrix<T>(0, _varOptions.OutputDimension);
        }

        // Use observations instead of x for the rest of training
        x = observations;

        int n = x.Rows;
        int m = x.Columns;
        int p = _varOptions.Lag;

        // Validate sufficient observations
        if (n <= p)
        {
            throw new ArgumentException($"Not enough data points. Need more than {p} observations for a VAR({p}) model. Found {n} observations.");
        }

        // Check for missing values: the first NaN or infinity in row-major order is reported.
        var series = Tensor<T>.FromMatrix(x);
        var invalid = engine.TensorWhere(engine.TensorIsFinite(series), engine.TensorZerosLike(series), engine.TensorOnesLike(series));
        if (NumOps.GreaterThan(engine.TensorSum(invalid), NumOps.Zero))
        {
            int position = FirstArgMax(engine.Reshape(invalid, new[] { n * m }));
            throw new ArgumentException($"Invalid value detected at position ({position / m},{position % m}). VAR models cannot handle missing values.");
        }

        try
        {
            // Prepare lagged data
            var laggedData = Tensor<T>.FromMatrix(PrepareLaggedData(x));

            // OLS for every equation shares X'X; the right-hand sides are the columns of X'Y over rows p .. n - 1.
            var laggedT = engine.TensorTranspose(laggedData);
            var xTx = engine.TensorMatMul(laggedT, laggedData).ToMatrix();
            var xTy = engine.TensorMatMul(laggedT, engine.TensorSlice(series, new[] { p, 0 }, new[] { n - p, m })).ToMatrix();

            // Estimate coefficients using OLS for each equation
            var solutions = new Tensor<T>[m];
            for (int i = 0; i < m; i++)
            {
                try
                {
                    solutions[i] = EstimateOLS(xTx, xTy.GetColumn(i)).AsTensor();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to estimate equation for variable {i}: {ex.Message}", ex);
                }
            }

            // Row i of the stacked solutions is [intercept_i, coefficients_i].
            var stacked = engine.TensorStack(solutions, 0);
            _intercepts = HeapVector(engine.Reshape(engine.TensorSlice(stacked, new[] { 0, 0 }, new[] { m, 1 }), new[] { m }));
            _coefficients = HeapMatrix(engine.TensorSlice(stacked, new[] { 0, 1 }, new[] { m, m * p }));

            // Calculate residuals
            _residuals = CalculateResiduals(x, laggedData);
        }
        catch (Exception ex)
        {
            // Reset model state on failure
            _coefficients = new Matrix<T>(_varOptions.OutputDimension, _varOptions.OutputDimension * _varOptions.Lag);
            _intercepts = new Vector<T>(_varOptions.OutputDimension);
            _residuals = new Matrix<T>(0, _varOptions.OutputDimension);

            throw new InvalidOperationException("VAR model training failed: " + ex.Message, ex);
        }

        // Populate ModelParameters from intercepts + coefficient matrix for GetParameters()
        ModelParameters = HeapVector(engine.TensorConcatenate(new[]
        {
            _intercepts.AsTensor(),
            engine.Reshape(Tensor<T>.FromMatrix(_coefficients), new[] { _coefficients.Rows * _coefficients.Columns })
        }, 0));
    }

    /// <summary>
    /// Predicts a single value from one of the variables in the VAR system.
    /// </summary>
    /// <param name="input">Vector of input features for prediction.</param>
    /// <returns>The predicted value for the specified variable.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentException">Thrown when the input vector has incorrect length or the variableIndex is out of range.</exception>
    /// <remarks>
    /// <para>
    /// This method generates a prediction for a single variable in the VAR system based on the
    /// lagged values provided in the input vector.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// The VAR model is designed to predict multiple related time series at once, but sometimes
    /// you might only need a prediction for one specific variable (like just GDP, not unemployment
    /// or inflation).
    /// 
    /// This method lets you predict one variable based on past values of all variables. The input
    /// should contain:
    /// - The variable index you want to predict (which of your time series)
    /// - The recent history of all variables in your system
    /// 
    /// The method uses the equation specifically estimated for that variable, applying the learned
    /// coefficients to calculate the next value.
    /// 
    /// For example, if you want to predict GDP (variable 0), it will use the equation:
    /// GDP_next = intercept + coefficient1*GDP_t-1 + coefficient2*Unemployment_t-1 + ...
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Check if the model has been trained
        if (_coefficients.Rows == 0 || _intercepts.Length == 0)
        {
            throw new InvalidOperationException("The VAR model must be trained before making predictions.");
        }

        // Extract which variable to predict (last element of input)
        int variableIndex;
        if (input.Length > 0)
        {
            variableIndex = Convert.ToInt32(input[input.Length - 1]);
        }
        else
        {
            throw new ArgumentException("Input vector must contain at least one element indicating which variable to predict.", nameof(input));
        }

        // Validate variable index
        if (variableIndex < 0 || variableIndex >= _varOptions.OutputDimension)
        {
            throw new ArgumentException($"Variable index {variableIndex} is out of range [0, {_varOptions.OutputDimension - 1}].", nameof(input));
        }

        // Verify input length (should contain lagged values for all variables + variable index)
        int lagged = _varOptions.OutputDimension * _varOptions.Lag;
        int expectedLength = lagged + 1;
        if (input.Length != expectedLength)
        {
            throw new ArgumentException($"Input vector length ({input.Length}) does not match expected length ({expectedLength}).", nameof(input));
        }

        // The intercept plus the equation's coefficients times the lagged values (the input before the index).
        return NumOps.Add(_intercepts[variableIndex], Engine.DotProduct(_coefficients.GetRow(variableIndex), input.Slice(0, lagged)));
    }

    /// <summary>
    /// Creates a new instance of the VectorAutoRegressionModel class.
    /// </summary>
    /// <returns>A new instance of the VectorAutoRegressionModel class.</returns>
    /// <remarks>
    /// <para>
    /// This factory method creates a new instance of the model with the same options
    /// as the current instance. It's used by Clone and DeepCopy methods.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a brand new, untrained copy of the VAR model with all the
    /// same configuration settings as the original. It's like getting a clean template
    /// based on your original specifications.
    /// 
    /// This is primarily used when you want to:
    /// - Create a copy of your model to experiment with different training approaches
    /// - Clone the model as part of serialization/deserialization
    /// - Make a deep copy that's completely independent of the original
    /// 
    /// The new model will have the same structure (number of variables and lag order)
    /// but won't have any trained coefficients until you train it.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options
        return new VectorAutoRegressionModel<T>(_varOptions);
    }

    /// <summary>
    /// Generates multi-step forecasts for all variables in the VAR system.
    /// </summary>
    /// <param name="historyMatrix">Matrix containing historical observations for all variables.</param>
    /// <param name="steps">Number of steps to forecast.</param>
    /// <returns>Matrix of forecasted values where each row is a time step and each column is a variable.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentException">Thrown when history matrix dimensions don't match model configuration or steps is not positive.</exception>
    /// <remarks>
    /// <para>
    /// This method generates multi-step ahead forecasts for all variables in the VAR system.
    /// For each forecast step, it uses previously forecasted values as inputs for the next step.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method lets you forecast multiple time periods into the future for all variables
    /// in your system. For example, you could forecast GDP, unemployment, and inflation for
    /// the next 4 quarters.
    /// 
    /// It works like this:
    /// 1. It starts with your historical data for all variables
    /// 2. It predicts the next period for all variables
    /// 3. It adds these predictions to the history
    /// 4. It uses this updated history to predict the next period
    /// 5. It repeats steps 3-4 until it reaches the requested forecast horizon
    /// 
    /// This approach is called "iterative forecasting" and it allows the model to capture
    /// how variables interact over multiple time periods. Note that forecast uncertainty
    /// typically increases the further ahead you predict.
    /// </para>
    /// </remarks>
    public Matrix<T> Forecast(Matrix<T> historyMatrix, int steps)
    {
        // Check if the model has been trained
        if (_coefficients.Rows == 0 || _intercepts.Length == 0)
        {
            throw new InvalidOperationException("The VAR model must be trained before forecasting.");
        }

        // Validate input dimensions
        if (historyMatrix.Columns != _varOptions.OutputDimension)
        {
            throw new ArgumentException($"History matrix columns ({historyMatrix.Columns}) must match the number of variables ({_varOptions.OutputDimension}).");
        }

        // Validate steps
        if (steps <= 0)
        {
            throw new ArgumentException("Forecast steps must be positive.", nameof(steps));
        }

        // Validate sufficient history
        if (historyMatrix.Rows < _varOptions.Lag)
        {
            throw new ArgumentException($"History matrix must contain at least {_varOptions.Lag} observations. Found {historyMatrix.Rows}.");
        }

        // Initialize forecast matrix
        Matrix<T> forecasts = new Matrix<T>(steps, _varOptions.OutputDimension);

        // Create a working copy of history that we'll extend with forecasts
        List<Vector<T>> workingHistory = new List<Vector<T>>();
        for (int i = 0; i < historyMatrix.Rows; i++)
        {
            workingHistory.Add(historyMatrix.GetRow(i));
        }

        // Generate forecasts iteratively
        for (int step = 0; step < steps; step++)
        {
            // Extract the most recent lags from working history
            Matrix<T> recentHistory = new Matrix<T>(_varOptions.Lag, _varOptions.OutputDimension);
            for (int i = 0; i < _varOptions.Lag; i++)
            {
                recentHistory.SetRow(i, workingHistory[workingHistory.Count - _varOptions.Lag + i]);
            }

            // Generate forecast for current step
            Vector<T> forecast = PredictSingleStep(recentHistory);

            // Store forecast
            forecasts.SetRow(step, forecast);

            // Add forecast to working history for next iteration
            workingHistory.Add(forecast);
        }

        return forecasts;
    }

    /// <summary>
    /// Analyzes the dynamic relationships between variables in the VAR system.
    /// </summary>
    /// <param name="horizon">The number of steps for the impulse response functions.</param>
    /// <returns>A dictionary of impulse response matrices, one for each variable.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentException">Thrown when horizon is not positive.</exception>
    /// <remarks>
    /// <para>
    /// This method computes impulse response functions to analyze how shocks to one variable
    /// affect all variables in the system over time.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// One of the most valuable aspects of VAR models is understanding how variables affect
    /// each other over time. This method helps you see these relationships.
    /// 
    /// It calculates what would happen to all variables if you introduced a one-time shock
    /// (increase or decrease) to one variable. For example:
    /// 
    /// - What happens to unemployment and inflation if GDP suddenly increases by 1%?
    /// - How long do these effects last?
    /// - Do they fade away or persist?
    /// 
    /// The results show how shocks propagate through the system over several time periods,
    /// helping you understand the dynamic relationships between your variables.
    /// 
    /// The output contains impulse response functions for each variable, showing how a shock
    /// to that variable affects all variables in the system over the specified time horizon.
    /// </para>
    /// </remarks>
    public Dictionary<string, Matrix<T>> ImpulseResponseAnalysis(int horizon)
    {
        // Check if the model has been trained
        if (_coefficients.Rows == 0 || _intercepts.Length == 0)
        {
            throw new InvalidOperationException("The VAR model must be trained before performing impulse response analysis.");
        }

        // Validate horizon
        if (horizon <= 0)
        {
            throw new ArgumentException("Horizon must be positive.", nameof(horizon));
        }

        var engine = AiDotNetEngine.Current;
        Dictionary<string, Matrix<T>> impulseResponses = new Dictionary<string, Matrix<T>>();
        int k = _varOptions.OutputDimension;
        int kp = k * _varOptions.Lag;

        // Estimate residual covariance matrix
        Matrix<T> residCov = EstimateResidualCovariance();

        // Get Cholesky decomposition of residual covariance for orthogonalized impulses
        Tensor<T> cholesky;
        try
        {
            cholesky = Tensor<T>.FromMatrix(CholeskyDecomposition(residCov));
        }
        catch (ArgumentException)
        {
            // Not positive definite: the shocks are scaled by the residual standard deviations instead.
            var identity = Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(k));
            cholesky = engine.TensorMultiply(
                engine.TensorBroadcastTo(
                    engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(Tensor<T>.FromMatrix(residCov), identity), new[] { 1 }, true)),
                    new[] { k, k }),
                identity);
        }

        // statsmodels orth_irfs: Phi_h P, with Phi_h the top-left k x k block of the h-th power of the companion
        // matrix and P the Cholesky factor; entry [response, shock].
        var companion = Tensor<T>.FromMatrix(ConstructVARMatrix());
        var power = Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(kp));
        var periods = new Tensor<T>[horizon + 1];
        for (int h = 0; h <= horizon; h++)
        {
            if (h > 0)
            {
                power = engine.TensorMatMul(power, companion);
            }

            periods[h] = engine.TensorMatMul(engine.TensorSlice(power, new[] { 0, 0 }, new[] { k, k }), cholesky);
        }

        // [period, response, shock] -> one [period, response] matrix per shock.
        var responses = engine.TensorStack(periods, 0);
        for (int shockVar = 0; shockVar < k; shockVar++)
        {
            impulseResponses[$"Variable_{shockVar}"] = engine.Reshape(
                engine.TensorSlice(responses, new[] { 0, 0, shockVar }, new[] { horizon + 1, k, 1 }), new[] { horizon + 1, k }).ToMatrix();
        }

        return impulseResponses;
    }

    /// <summary>
    /// Estimates the covariance matrix of residuals.
    /// </summary>
    /// <returns>The covariance matrix of residuals.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the variance-covariance matrix of the model's residuals,
    /// which represents the unexplained variance and covariance between variables.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates how the random, unexplained parts of your variables
    /// relate to each other. It shows:
    /// 
    /// - How much random variation exists in each variable (the diagonal elements)
    /// - How the random parts of different variables move together (the off-diagonal elements)
    /// 
    /// This information is important for:
    /// - Assessing how much uncertainty exists in the model
    /// - Understanding if shocks to multiple variables tend to occur together
    /// - Properly calculating confidence intervals for forecasts
    /// - Performing impulse response analysis
    /// </para>
    /// </remarks>
    private Matrix<T> EstimateResidualCovariance()
    {
        if (_residuals.Rows == 0)
        {
            throw new InvalidOperationException("No residuals available. The model must be trained first.");
        }

        var engine = AiDotNetEngine.Current;
        int n = _residuals.Rows;
        int k = _residuals.Columns;

        // Degrees of freedom: observations less the parameters per equation (statsmodels df_resid).
        int dof = n - _varOptions.Lag * k - 1;
        if (dof <= 0) dof = 1; // Prevent division by zero

        var residuals = Tensor<T>.FromMatrix(_residuals);
        return engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(residuals), residuals), NumOps.FromDouble(dof)).ToMatrix();
    }

    /// <summary>
    /// Constructs the VAR coefficient matrix in companion form.
    /// </summary>
    /// <returns>The VAR matrix in companion form.</returns>
    /// <remarks>
    /// <para>
    /// This method organizes the VAR coefficients into a companion matrix form
    /// that facilitates multi-step computations like impulse response functions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a special matrix that represents how all variables in your
    /// system evolve over time based on their past values. This companion form allows
    /// for easy calculation of multi-step forecasts and impulse responses.
    /// 
    /// The companion matrix packages all the coefficients in a specific structure that
    /// makes it easier to compute how shocks propagate through the system over time.
    /// 
    /// This is primarily used internally for advanced analyses like impulse response
    /// functions and forecast error variance decomposition.
    /// </para>
    /// </remarks>
    private Matrix<T> ConstructVARMatrix()
    {
        var engine = AiDotNetEngine.Current;
        int k = _varOptions.OutputDimension;
        int p = _varOptions.Lag;
        int kp = k * p;

        // [A_1 ... A_p] on top; below it the identity that shifts each lag down one block.
        var coefficients = Tensor<T>.FromMatrix(_coefficients);
        if (p == 1)
        {
            return coefficients.ToMatrix();
        }

        var shift = engine.TensorConcatenate(new[]
        {
            Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(kp - k)),
            new Tensor<T>(new[] { kp - k, k })
        }, 1);
        return engine.TensorConcatenate(new[] { coefficients, shift }, 0).ToMatrix();
    }

    /// <summary>
    /// Performs Cholesky decomposition of a symmetric positive definite matrix.
    /// </summary>
    /// <param name="matrix">The symmetric positive definite matrix to decompose.</param>
    /// <returns>The lower triangular Cholesky factor.</returns>
    /// <exception cref="ArgumentException">Thrown when the matrix is not symmetric positive definite.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the Cholesky decomposition of a symmetric positive definite matrix,
    /// which is useful for computing orthogonalized impulse responses.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Cholesky decomposition is a mathematical technique that breaks down a symmetric matrix
    /// into a product of a lower triangular matrix and its transpose. It's used in VAR analysis to:
    /// 
    /// - Create orthogonalized shocks for impulse response analysis
    /// - Decompose the relationships between variables' error terms
    /// - Account for contemporaneous correlations between variables
    /// 
    /// This decomposition helps ensure that when you're analyzing how a shock to one variable
    /// affects others, you're accounting for how shocks tend to occur together in your data.
    /// </para>
    /// </remarks>
    private Matrix<T> CholeskyDecomposition(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        var (factor, info) = LinalgDispatch<T>.CholeskyEx(Tensor<T>.FromMatrix(matrix), upper: false);
        if (AiDotNetEngine.Current.TensorCountNonzero(info) > 0)
        {
            throw new ArgumentException("Matrix is not positive definite.");
        }

        return factor.ToMatrix();
    }

    private static Tensor<T> Filled(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(tensor, value);
        return tensor;
    }

    /// <summary>A plain-heap copy of a [rows, columns] engine result (training-arena buffers are recycled).</summary>
    private static Matrix<T> HeapMatrix(Tensor<T> values)
    {
        var matrix = new Matrix<T>(values.Shape[0], values.Shape[1]);
        values.CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    /// <summary>A plain-heap copy of a 1-D engine result (training-arena buffers are recycled).</summary>
    private static Vector<T> HeapVector(Tensor<T> values)
    {
        var vector = new Vector<T>(values.Length);
        values.CopyTo(vector.AsWritableSpan());
        return vector;
    }

    /// <summary>The index of the first maximum of a 1-D tensor: the earliest maximal entry carries the largest weight.</summary>
    private static int FirstArgMax(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int length = values.Length;
        var isMax = engine.TensorEquals(values, engine.TensorMaxValue(values));
        var descending = engine.TensorAddScalar(engine.TensorNegate(engine.TensorArange<T>(0, length, 1)), numOps.FromDouble(length));
        return engine.TensorArgMax(engine.TensorMultiply(isMax, descending), 0).GetFlat(0);
    }

    /// <summary>
    /// Resets the VAR model to its untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all trained parameters and returns the model to its initial state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method resets the model to its initial state, clearing all the coefficients
    /// and other parameters that were learned during training.
    /// 
    /// It's useful when you want to:
    /// - Retrain the model with different data
    /// - Try different model specifications
    /// - Clear the model's memory to free up resources
    /// 
    /// After calling this method, the model will need to be trained again before
    /// it can be used for predictions.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();

        _coefficients = new Matrix<T>(_varOptions.OutputDimension, _varOptions.OutputDimension * _varOptions.Lag);
        _intercepts = new Vector<T>(_varOptions.OutputDimension);
        _residuals = new Matrix<T>(0, _varOptions.OutputDimension);
    }

    /// <summary>
    /// Gets metadata about the VAR model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides comprehensive metadata about the VAR model, including configuration
    /// parameters, model coefficients, and serialized model data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method returns important information about your VAR model in a structured format.
    /// The metadata includes:
    /// 
    /// - The type of model (VAR)
    /// - Configuration details (like lag order and number of variables)
    /// - The coefficients that describe relationships between variables
    /// - Serialized model data for storage or transfer
    /// 
    /// This information is useful for documentation, model management, and comparing different
    /// models. It provides a complete snapshot of the model's structure and parameters.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include the actual model state variables
                { "Coefficients", _coefficients },
                { "Intercepts", _intercepts },
                { "Residuals", _residuals },
            
                // Include model configuration
                { "OutputDimension", _varOptions.OutputDimension },
                { "Lag", _varOptions.Lag },
                { "DecompositionType", _varOptions.DecompositionType },
            
                // Include model statistics
                { "HasTrainingData", _residuals.Rows > 0 },
                { "NumberOfObservations", _residuals.Rows + _varOptions.Lag },
                { "NumberOfParameters", _varOptions.OutputDimension * (_varOptions.OutputDimension * _varOptions.Lag + 1) }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };

        // Add residual statistics if available
        if (_residuals.Rows > 0)
        {
            // Calculate residual statistics for each variable
            for (int i = 0; i < _varOptions.OutputDimension; i++)
            {
                Vector<T> variableResiduals = _residuals.GetColumn(i);

                metadata.AdditionalInfo[$"Variable_{i}_ResidualMean"] = Convert.ToDouble(variableResiduals.Average());
                metadata.AdditionalInfo[$"Variable_{i}_ResidualStdDev"] = Convert.ToDouble(variableResiduals.StandardDeviation());
                metadata.AdditionalInfo[$"Variable_{i}_ResidualMin"] = Convert.ToDouble(variableResiduals.Min());
                metadata.AdditionalInfo[$"Variable_{i}_ResidualMax"] = Convert.ToDouble(variableResiduals.Max());
            }
        }

        return metadata;
    }
}
