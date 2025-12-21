namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Vector Autoregressive (VAR) models, which model the linear interdependencies
/// among multiple time series.
/// </summary>
/// <remarks>
/// <para>
/// Vector Autoregressive (VAR) models are a generalization of univariate autoregressive models to multivariate 
/// time series. In a VAR model, each variable is modeled as a linear function of past values of itself and past 
/// values of all other variables in the system. This approach captures the dynamic relationships and feedback 
/// effects among multiple interrelated time series. VAR models are widely used in economics, finance, and other 
/// fields for forecasting, structural analysis, and policy analysis. They provide a flexible framework for 
/// analyzing the joint dynamics of multiple variables without imposing strong a priori restrictions on the 
/// relationships. This class provides configuration options for controlling the structure and estimation of 
/// VAR models.
/// </para>
/// <para><b>For Beginners:</b> VAR models help you understand and forecast multiple related time series simultaneously.
/// 
/// When dealing with multiple time series that affect each other:
/// - Simple models treat each series independently
/// - But in reality, variables often influence each other over time
/// 
/// VAR models solve this by:
/// - Modeling all variables together as a system
/// - Allowing each variable to depend on past values of all variables
/// - Capturing the relationships and feedback effects between variables
/// 
/// This approach offers several benefits:
/// - Better forecasts by incorporating relationships between variables
/// - Understanding how shocks to one variable affect other variables
/// - Analyzing the dynamic interactions in a system
/// 
/// For example, in economics, VAR models might show how changes in interest rates
/// affect GDP, inflation, and unemployment over several quarters.
/// 
/// This class lets you configure how the VAR model is structured and estimated.
/// </para>
/// </remarks>
public class VARModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the lag order for the VAR model.
    /// </summary>
    /// <value>A positive integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the lag order for the VAR model, often denoted as p. The lag order determines how 
    /// many past time periods are included in the model for each variable. A VAR model with lag order p means that 
    /// each variable is regressed on p lags of itself and p lags of each other variable in the system. A higher 
    /// order allows the model to capture more complex dynamic relationships but increases the number of parameters 
    /// to estimate and the risk of overfitting. The default value of 1 provides a simple first-order VAR model 
    /// suitable for many applications, capturing immediate dependencies between variables. The optimal lag order 
    /// depends on the temporal dependencies in the data and can be determined using information criteria such as 
    /// AIC or BIC, or through statistical tests.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many past time periods influence the current values.
    /// 
    /// The lag order:
    /// - Determines how many previous time periods affect the current values
    /// - Helps the model capture patterns and relationships that extend over time
    /// - Higher values can model more complex temporal dependencies
    /// 
    /// The default value of 1 means:
    /// - The model considers only the immediate previous values
    /// - This creates a simple VAR(1) model suitable for many applications
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2 or 4): Can capture more complex relationships over longer periods
    /// - Lower values (e.g., 1): Simpler model focusing on immediate effects
    /// 
    /// When to adjust this value:
    /// - Increase it when variables show dependencies beyond the immediate previous period
    /// - Keep at 1 for a simple model or when working with limited data
    /// - Consider using information criteria (AIC, BIC) to select the optimal lag
    /// 
    /// For example, in quarterly economic data where effects might take several quarters to manifest,
    /// you might increase this to 4 to capture a full year of lagged effects.
    /// </para>
    /// </remarks>
    public int Lag { get; set; } = 1;

    /// <summary>
    /// Gets or sets the dimension of the output vector.
    /// </summary>
    /// <value>A positive integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of variables (time series) in the VAR model. Each variable is modeled 
    /// as a function of past values of itself and past values of all other variables. A higher dimension allows 
    /// the model to capture relationships among more variables but increases the number of parameters to estimate 
    /// and the data requirements. The default value of 1 corresponds to a univariate autoregressive model, which 
    /// is a special case of VAR. For true multivariate analysis, this value should be set to the number of 
    /// interrelated time series being modeled. The appropriate value depends on the specific application and the 
    /// number of time series that are believed to have meaningful dynamic relationships.
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies how many different time series you're modeling together.
    /// 
    /// The output dimension:
    /// - Defines the number of variables in your VAR system
    /// - Each variable will be modeled as depending on past values of all variables
    /// - Affects the complexity and data requirements of the model
    /// 
    /// The default value of 1 means:
    /// - The model handles just one time series
    /// - This actually makes it a simple autoregressive (AR) model, not a true VAR
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 3 or 5): Model multiple related variables as a system
    /// - Value of 1: Model a single time series (not utilizing VAR's multivariate capabilities)
    /// 
    /// When to adjust this value:
    /// - Set to the exact number of time series you want to model together
    /// - For true VAR analysis, this should be at least 2
    /// - Consider data availability - each additional variable increases data requirements
    /// 
    /// For example, if you're modeling GDP, inflation, and unemployment together,
    /// you would set this to 3 to create a three-variable VAR system.
    /// </para>
    /// </remarks>
    public int OutputDimension { get; set; } = 1;

    /// <summary>
    /// Gets or sets the type of matrix decomposition used in the estimation algorithm.
    /// </summary>
    /// <value>A value from the MatrixDecompositionType enumeration, defaulting to MatrixDecompositionType.Lu.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the type of matrix decomposition used in the estimation algorithm for the VAR model. 
    /// Matrix decomposition is used to solve the system of equations that arises when estimating the model parameters. 
    /// Different decomposition methods have different numerical properties and computational requirements. LU 
    /// decomposition (the default) is efficient for general matrices and is suitable for most VAR applications. 
    /// Other options might include Cholesky decomposition (for positive definite matrices), QR decomposition 
    /// (more stable for ill-conditioned matrices), or SVD (Singular Value Decomposition, the most stable but also 
    /// the most computationally expensive). The optimal choice depends on the numerical properties of the specific 
    /// problem and the desired trade-off between numerical stability and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical method used to solve equations during model estimation.
    /// 
    /// The matrix decomposition type:
    /// - Affects how certain mathematical operations are performed during estimation
    /// - Different methods have different trade-offs between speed and numerical stability
    /// - Most users don't need to change this setting
    /// 
    /// The default value of Lu means:
    /// - The algorithm uses LU decomposition for matrix operations
    /// - This method is efficient and works well for most VAR models
    /// 
    /// Common alternatives include:
    /// - Cholesky: Efficient for positive definite matrices
    /// - QR: More stable for difficult numerical problems
    /// - SVD: Most numerically stable, but significantly slower
    /// 
    /// When to adjust this value:
    /// - Change to QR or SVD if you encounter numerical stability issues
    /// - Change to Cholesky if you know your matrices are positive definite
    /// - Keep the default for most applications
    /// 
    /// For example, if your VAR estimation process fails with numerical errors,
    /// you might try changing this to MatrixDecompositionType.Svd for better stability.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;
}
