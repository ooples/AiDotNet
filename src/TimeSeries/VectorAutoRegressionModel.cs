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
public class VectorAutoRegressionModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options specific to the VAR model.
    /// </summary>
    private readonly VARModelOptions<T> _varOptions;
    
    /// <summary>
    /// Matrix of coefficients that capture the relationships between variables across time lags.
    /// </summary>
    private Matrix<T> _coefficients;
    
    /// <summary>
    /// Vector of intercept terms for each equation in the VAR model.
    /// </summary>
    private Vector<T> _intercepts;
    
    /// <summary>
    /// Matrix of residuals (errors) from the model fit.
    /// </summary>
    private Matrix<T> _residuals;
    
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
    /// Trains the VAR model using the provided input data.
    /// </summary>
    /// <param name="x">The input matrix where each column represents a different time series variable.</param>
    /// <param name="y">The target values vector (not used in VAR models as they use the input matrix directly).</param>
    /// <exception cref="ArgumentException">Thrown when input dimensions don't match the model configuration or there's insufficient data.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training a VAR model involves finding the best coefficients to explain how each variable
    /// depends on past values of itself and other variables.
    /// 
    /// The training process follows these steps:
    /// 
    /// 1. Validate that the input data has the correct dimensions and sufficient observations
    /// 2. Prepare the lagged data matrix (organizing past values for use in estimation)
    /// 3. For each variable, estimate an equation using Ordinary Least Squares (OLS) regression
    /// 4. Calculate the residuals (errors) between the model's predictions and actual values
    /// 
    /// Each equation in the VAR model has the form:
    /// y_t = c + A₁y_{t-1} + A₂y_{t-2} + ... + A_py_{t-p} + e_t
    /// 
    /// Where:
    /// - y_t is the vector of variables at time t
    /// - c is the vector of intercepts
    /// - A₁, A₂, etc. are matrices of coefficients
    /// - p is the lag order
    /// - e_t is the vector of error terms
    /// 
    /// The y parameter is typically not used in VAR models since all variables are contained in the x matrix.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Columns != _varOptions.OutputDimension)
        {
            throw new ArgumentException("The number of columns in x must match the OutputDimension.");
        }

        int n = x.Rows;
        int m = x.Columns;

        if (n <= _varOptions.Lag)
        {
            throw new ArgumentException($"Not enough data points. Need more than {_varOptions.Lag} observations.");
        }

        // Prepare lagged data
        Matrix<T> laggedData = PrepareLaggedData(x);

        // Estimate coefficients using OLS for each equation
        for (int i = 0; i < m; i++)
        {
            Vector<T> yi = x.GetColumn(i).Slice(_varOptions.Lag, n - _varOptions.Lag);
            Vector<T> coeffs = EstimateOLS(laggedData, yi);
            _intercepts[i] = coeffs[0];
            for (int j = 0; j < m * _varOptions.Lag; j++)
            {
                _coefficients[i, j] = coeffs[j + 1];
            }
        }

        // Calculate residuals
        _residuals = CalculateResiduals(x);
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
        if (input.Columns != _varOptions.OutputDimension)
        {
            throw new ArgumentException("Input dimensions do not match the model.");
        }

        Vector<T> prediction = new Vector<T>(_varOptions.OutputDimension);
        Vector<T> laggedValues = input.GetRow(input.Rows - 1);

        for (int i = 0; i < _varOptions.OutputDimension; i++)
        {
            prediction[i] = _intercepts[i];
            for (int j = 0; j < _varOptions.OutputDimension * _varOptions.Lag; j++)
            {
                prediction[i] = NumOps.Add(prediction[i], NumOps.Multiply(_coefficients[i, j], laggedValues[j]));
            }
        }

        return prediction;
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
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize VARModelOptions
        writer.Write(_varOptions.Lag);
        writer.Write(_varOptions.OutputDimension);

        // Serialize _coefficients
        writer.Write(_coefficients.Rows);
        writer.Write(_coefficients.Columns);
        for (int i = 0; i < _coefficients.Rows; i++)
            for (int j = 0; j < _coefficients.Columns; j++)
                writer.Write(Convert.ToDouble(_coefficients[i, j]));

        // Serialize _intercepts
        writer.Write(_intercepts.Length);
        for (int i = 0; i < _intercepts.Length; i++)
            writer.Write(Convert.ToDouble(_intercepts[i]));

        // Serialize _residuals
        writer.Write(_residuals.Rows);
        writer.Write(_residuals.Columns);
        for (int i = 0; i < _residuals.Rows; i++)
            for (int j = 0; j < _residuals.Columns; j++)
                writer.Write(Convert.ToDouble(_residuals[i, j]));
    }

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
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize VARModelOptions
        _varOptions.Lag = reader.ReadInt32();
        _varOptions.OutputDimension = reader.ReadInt32();

        // Deserialize _coefficients
        int coeffRows = reader.ReadInt32();
        int coeffCols = reader.ReadInt32();
        _coefficients = new Matrix<T>(coeffRows, coeffCols);
        for (int i = 0; i < coeffRows; i++)
            for (int j = 0; j < coeffCols; j++)
                _coefficients[i, j] = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize _intercepts
        int interceptsLength = reader.ReadInt32();
        _intercepts = new Vector<T>(interceptsLength);
        for (int i = 0; i < interceptsLength; i++)
            _intercepts[i] = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize _residuals
        int residualsRows = reader.ReadInt32();
        int residualsCols = reader.ReadInt32();
        _residuals = new Matrix<T>(residualsRows, residualsCols);
        for (int i = 0; i < residualsRows; i++)
            for (int j = 0; j < residualsCols; j++)
                                _residuals[i, j] = NumOps.FromDouble(reader.ReadDouble());
    }

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
        int n = x.Rows;
        int m = x.Columns;
        Matrix<T> laggedData = new Matrix<T>(n - _varOptions.Lag, m * _varOptions.Lag + 1);

        for (int i = _varOptions.Lag; i < n; i++)
        {
            laggedData[i - _varOptions.Lag, 0] = NumOps.One;
            for (int j = 0; j < _varOptions.Lag; j++)
            {
                for (int k = 0; k < m; k++)
                {
                    laggedData[i - _varOptions.Lag, j * m + k + 1] = x[i - j - 1, k];
                }
            }
        }

        return laggedData;
    }

    /// <summary>
    /// Estimates coefficients using Ordinary Least Squares (OLS) regression.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A vector of regression coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method finds the best coefficients for a linear regression model using
    /// the Ordinary Least Squares (OLS) approach.
    /// 
    /// It solves the equation: β = (X'X)⁻¹X'y, where:
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
    private Vector<T> EstimateOLS(Matrix<T> x, Vector<T> y)
    {
        Matrix<T> xTx = x.Transpose().Multiply(x);
        Vector<T> xTy = x.Transpose().Multiply(y);

        return MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _varOptions.DecompositionType);
    }

    /// <summary>
    /// Calculates the residuals (errors) between the model's predictions and the actual values.
    /// </summary>
    /// <param name="x">The input matrix where each column represents a different time series variable.</param>
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
    private Matrix<T> CalculateResiduals(Matrix<T> x)
    {
        int n = x.Rows;
        int m = x.Columns;
        Matrix<T> residuals = new Matrix<T>(n - _varOptions.Lag, m);

        for (int i = _varOptions.Lag; i < n; i++)
        {
            Vector<T> predicted = Predict(x.Slice(i - _varOptions.Lag, _varOptions.Lag));
            Vector<T> actual = x.GetRow(i);
            residuals.SetRow(i - _varOptions.Lag, actual.Subtract(predicted));
        }

        return residuals;
    }
}