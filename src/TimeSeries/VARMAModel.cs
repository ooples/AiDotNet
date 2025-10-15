namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Vector Autoregressive Moving Average (VARMA) model for multivariate time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The VARMA model extends the Vector Autoregressive (VAR) model by incorporating Moving Average (MA) terms,
/// allowing it to capture more complex dynamics in multivariate time series data. It models the relationships
/// between multiple time series variables and their past values, as well as past error terms.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// A VARMA model helps you forecast multiple related time series at once, accounting for:
/// 
/// - How each variable depends on its own past values
/// - How each variable depends on other variables' past values
/// - How past prediction errors affect current values
/// 
/// For example, if you're analyzing economic data, a VARMA model could simultaneously forecast:
/// - GDP growth
/// - Unemployment rate
/// - Inflation rate
/// 
/// While accounting for how these variables affect each other and incorporating information
/// from past prediction errors to improve accuracy.
/// 
/// Think of it as a sophisticated forecasting system that recognizes both the interconnections
/// between different variables and learns from its own mistakes.
/// </para>
/// </remarks>
public class VARMAModel<T> : VectorAutoRegressionModel<T>
{
    /// <summary>
    /// Configuration options specific to the VARMA model.
    /// </summary>
    private readonly VARMAModelOptions<T> _varmaOptions = default!;
    
    /// <summary>
    /// Matrix of Moving Average (MA) coefficients that capture the dependency on past error terms.
    /// </summary>
    private Matrix<T> _maCoefficients = default!;
    
    /// <summary>
    /// Matrix of residuals (errors) from the model fit.
    /// </summary>
    private Matrix<T> _residuals = default!;

    /// <summary>
    /// Initializes a new instance of the VARMAModel class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the VARMA model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// When you create a VARMA model, you need to specify several options:
    /// 
    /// - Output Dimension: How many different time series you're modeling together
    /// - AR Lag: How many past time periods to consider for the autoregressive part
    /// - MA Lag: How many past time periods to consider for the moving average part
    /// 
    /// For example, if you're modeling quarterly economic indicators with:
    /// - 3 variables (GDP, unemployment, inflation)
    /// - AR Lag of 4 (consider the past year of data)
    /// - MA Lag of 2 (consider prediction errors from the past two quarters)
    /// 
    /// The constructor initializes the model structure based on these specifications.
    /// The MA coefficients matrix will have dimensions based on the output dimension and MA lag.
    /// </para>
    /// </remarks>
    public VARMAModel(VARMAModelOptions<T> options) : base(options)
    {
        _varmaOptions = options;
        _maCoefficients = new Matrix<T>(options.OutputDimension, options.OutputDimension * options.MaLag);
        _residuals = Matrix<T>.Empty();
    }

    /// <summary>
    /// Generates forecasts using the trained VARMA model.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector of forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// When making predictions with a VARMA model, two components are combined:
    /// 
    /// 1. The AR (autoregressive) prediction:
    ///    - This comes from the VAR part of the model
    ///    - It's based on past values of all variables
    /// 
    /// 2. The MA (moving average) prediction:
    ///    - This is based on past prediction errors
    ///    - It helps correct for systematic errors the AR part might make
    /// 
    /// The final prediction is the sum of these two components.
    /// 
    /// For example, if the AR part predicts economic growth of 2.5%, but the model has
    /// consistently underestimated growth by about 0.3% recently, the MA part might
    /// add that 0.3% correction, resulting in a final prediction of 2.8% growth.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> arPrediction = base.Predict(input);
        Vector<T> maPrediction = PredictMA();

        return arPrediction.Add(maPrediction);
    }

    /// <summary>
    /// Estimates the Moving Average (MA) coefficients using the residuals from the VAR model.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when there are not enough observations to estimate MA coefficients.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method finds the best values for the MA coefficients, which determine how
    /// past prediction errors affect current predictions.
    /// 
    /// It works by:
    /// 1. Checking if there are enough observations to estimate the coefficients
    /// 2. Preparing a matrix of lagged residuals (past prediction errors)
    /// 3. Using Ordinary Least Squares (OLS) regression to find the best coefficients
    ///    for each output variable
    /// 
    /// The MA coefficients help the model learn from its mistakes. For example, if the model
    /// consistently underestimates a variable after certain patterns in the data, the MA
    /// coefficients will capture this pattern and adjust future predictions accordingly.
    /// </para>
    /// </remarks>
    private void EstimateMACoefficients()
    {
        int n = _residuals.Rows;
        int m = _residuals.Columns;

        if (n <= _varmaOptions.MaLag)
        {
            throw new InvalidOperationException($"Not enough residuals. Need more than {_varmaOptions.MaLag} observations.");
        }

        // Prepare lagged residuals
        Matrix<T> laggedResiduals = PrepareLaggedResiduals();

        // Estimate MA coefficients using OLS for each equation
        for (int i = 0; i < m; i++)
        {
            Vector<T> yi = _residuals.GetColumn(i).Slice(_varmaOptions.MaLag, n - _varmaOptions.MaLag);
            Vector<T> coeffs = SolveOLS(laggedResiduals, yi);
            for (int j = 0; j < m * _varmaOptions.MaLag; j++)
            {
                _maCoefficients[i, j] = coeffs[j];
            }
        }
    }

    /// <summary>
    /// Prepares a matrix of lagged residuals for estimating MA coefficients.
    /// </summary>
    /// <returns>A matrix of lagged residuals.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method organizes past prediction errors (residuals) into a format that can be
    /// used to estimate the MA coefficients.
    /// 
    /// For each time point, it collects the prediction errors from previous time points
    /// for all variables being modeled. The result is a matrix where:
    /// - Each row represents a time point
    /// - Each column represents a specific variable's error from a specific past time point
    /// 
    /// For example, with 3 variables and an MA lag of 2, each row would contain 6 values:
    /// - The previous errors for variables 1, 2, and 3
    /// - The errors from two periods ago for variables 1, 2, and 3
    /// 
    /// This structured format allows the model to determine how each past error affects
    /// each current variable.
    /// </para>
    /// </remarks>
    private Matrix<T> PrepareLaggedResiduals()
    {
        int n = _residuals.Rows;
        int m = _residuals.Columns;
        Matrix<T> laggedResiduals = new Matrix<T>(n - _varmaOptions.MaLag, m * _varmaOptions.MaLag);

        for (int i = _varmaOptions.MaLag; i < n; i++)
        {
            for (int j = 0; j < _varmaOptions.MaLag; j++)
            {
                for (int k = 0; k < m; k++)
                {
                    laggedResiduals[i - _varmaOptions.MaLag, j * m + k] = _residuals[i - j - 1, k];
                }
            }
        }

        return laggedResiduals;
    }

    /// <summary>
    /// Calculates the Moving Average (MA) component of the prediction.
    /// </summary>
    /// <returns>A vector containing the MA component of the prediction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates how past prediction errors contribute to the current forecast.
    /// 
    /// It works by:
    /// 1. Starting with a vector of zeros (one for each output variable)
    /// 2. For each output variable, calculating the weighted sum of past residuals
    ///    using the MA coefficients as weights
    /// 
    /// The result is the MA component of the prediction, which is then added to the
    /// AR component to get the final forecast.
    /// 
    /// This adjustment based on past errors helps the model correct for systematic
    /// biases in its predictions, making it more accurate over time.
    /// </para>
    /// </remarks>
    private Vector<T> PredictMA()
    {
        Vector<T> maPrediction = new Vector<T>(_varmaOptions.OutputDimension);
        Vector<T> lastResiduals = _residuals.GetRow(_residuals.Rows - 1);

        for (int i = 0; i < _varmaOptions.OutputDimension; i++)
        {
            for (int j = 0; j < _varmaOptions.OutputDimension * _varmaOptions.MaLag; j++)
            {
                maPrediction[i] = NumOps.Add(maPrediction[i], NumOps.Multiply(_maCoefficients[i, j], lastResiduals[j]));
            }
        }

        return maPrediction;
    }

    /// <summary>
    /// Calculates the residuals (errors) between the VAR model's predictions and the actual values.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A matrix of residuals.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Residuals are the differences between what the model predicted and what actually happened.
    /// They represent the part of the data that the VAR model couldn't explain.
    /// 
    /// This method:
    /// 1. Uses the VAR part of the model to make predictions
    /// 2. Subtracts these predictions from the actual values
    /// 3. Organizes these differences into a matrix
    /// 
    /// These residuals are important because:
    /// - They're used to estimate the MA coefficients
    /// - They help the model learn from its mistakes
    /// - They're used directly in future predictions via the MA component
    /// 
    /// Analyzing residuals can also help diagnose model problems, like missing variables
    /// or nonlinear relationships that the model isn't capturing.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateResiduals(Matrix<T> x, Vector<T> y)
    {
        Vector<T> predictions = base.Predict(x);
        return Matrix<T>.FromColumns(y.Subtract(predictions));
    }

    /// <summary>
    /// Solves a linear regression problem using Ordinary Least Squares (OLS).
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
    /// - X is the input matrix (lagged residuals in this case)
    /// - y is the target vector (current residuals)
    /// - β is the vector of coefficients we're solving for
    /// - X' is the transpose of X
    /// - (X'X)⁻¹ is the inverse of X'X
    /// 
    /// The result is a set of coefficients that minimize the sum of squared errors
    /// between the model's predictions and the actual values.
    /// 
    /// This is used to estimate the MA coefficients, which determine how past prediction
    /// errors affect current predictions.
    /// </para>
    /// </remarks>
    private Vector<T> SolveOLS(Matrix<T> x, Vector<T> y)
    {
        return MatrixSolutionHelper.SolveLinearSystem(x.Transpose().Multiply(x), x.Transpose().Multiply(y), _varmaOptions.DecompositionType);
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
    /// This method:
    /// 1. First calls the base class's serialization method to save the VAR part of the model
        /// 2. Then saves the VARMA-specific options (like the MA lag)
    /// 3. Finally saves the MA coefficients matrix
    /// 
    /// The MA coefficients are saved row by row, column by column, as double values.
    /// This ensures that when the model is loaded later, it will make exactly the same
    /// predictions as it did before being saved.
    /// 
    /// Serialization is particularly useful for:
    /// - Deploying models to production environments
    /// - Sharing models between different applications
    /// - Saving computation time by not having to retrain complex models
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        base.SerializeCore(writer);

        // Serialize VARMAModelOptions
        writer.Write(_varmaOptions.MaLag);

        // Serialize _maCoefficients
        writer.Write(_maCoefficients.Rows);
        writer.Write(_maCoefficients.Columns);
        for (int i = 0; i < _maCoefficients.Rows; i++)
            for (int j = 0; j < _maCoefficients.Columns; j++)
                writer.Write(Convert.ToDouble(_maCoefficients[i, j]));
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
    /// This method:
    /// 1. First calls the base class's deserialization method to load the VAR part of the model
    /// 2. Then loads the VARMA-specific options (like the MA lag)
    /// 3. Finally loads the MA coefficients matrix
    /// 
    /// The MA coefficients are read in the same order they were written: row by row, column by column.
    /// 
    /// After deserialization, the model is ready to make predictions without needing to be retrained.
    /// This is particularly useful when:
    /// - You have a complex model that took a long time to train
    /// - You want to ensure consistent predictions across different runs or applications
    /// - You're deploying a model to a production environment where training isn't feasible
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        base.DeserializeCore(reader);

        // Deserialize VARMAModelOptions
        _varmaOptions.MaLag = reader.ReadInt32();

        // Deserialize _maCoefficients
        int maCoeffRows = reader.ReadInt32();
        int maCoeffCols = reader.ReadInt32();
        _maCoefficients = new Matrix<T>(maCoeffRows, maCoeffCols);
        for (int i = 0; i < maCoeffRows; i++)
            for (int j = 0; j < maCoeffCols; j++)
                _maCoefficients[i, j] = NumOps.FromDouble(reader.ReadDouble());
    }
}
    