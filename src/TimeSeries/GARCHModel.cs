using AiDotNet.Extensions;
using Newtonsoft.Json;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for time series with changing volatility.
/// </summary>
/// <remarks>
/// <para>
/// The GARCH model is specifically designed for time series data that exhibits volatility clustering, where periods of high 
/// volatility are followed by periods of high volatility, and periods of low volatility are followed by periods of low volatility.
/// It combines a mean model (typically an ARIMA model) with a variance model that captures the conditional heteroskedasticity.
/// </para>
/// <para><b>For Beginners:</b> GARCH models help predict both the value and the uncertainty of financial data.
/// 
/// Think of it like weather forecasting:
/// - Regular forecasting predicts the temperature (the mean value)
/// - GARCH also predicts how much the temperature might vary (the volatility)
/// 
/// For example, with stock prices:
/// - Some days, prices barely change (low volatility)
/// - Other days, prices swing wildly up and down (high volatility)
/// - Often, volatile periods tend to cluster together (volatility clustering)
/// 
/// GARCH helps model this behavior by:
/// - Using one model to predict the average price (mean model)
/// - Using another model to predict how much prices might fluctuate (volatility model)
/// 
/// This is especially useful for financial risk management, option pricing, and trading strategies.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GARCHModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// The configuration options for the GARCH model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains the settings for the GARCH model, such as the orders of the ARCH and GARCH components,
    /// the mean model to use, and other configuration parameters that control the model's behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This holds all the settings that define how the model works.
    /// 
    /// These options include:
    /// - How many past squared errors to consider (ARCH order)
    /// - How many past volatilities to consider (GARCH order)
    /// - What model to use for predicting average values
    /// - Other technical settings that control the model's behavior
    /// 
    /// It's like a recipe that specifies all the ingredients and proportions
    /// needed to create the model.
    /// </para>
    /// </remarks>
    private GARCHModelOptions<T> _garchOptions;

    /// <summary>
    /// The model used to forecast the mean (average value) of the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is typically an ARIMA model that captures the average behavior of the time series.
    /// The GARCH model then models the variance of the residuals (errors) from this mean model.
    /// </para>
    /// <para><b>For Beginners:</b> This predicts the expected value, not the volatility.
    /// 
    /// The mean model:
    /// - Forecasts the average or expected value at each time point
    /// - Is usually an ARIMA model, which captures patterns in the data values
    /// - Works independently of the volatility modeling
    /// 
    /// For example, with stock prices, the mean model might predict that
    /// tomorrow's price will be $100, while the GARCH component predicts
    /// how much that price might vary from the $100 prediction.
    /// </para>
    /// </remarks>
    private ITimeSeriesModel<T> _meanModel;

    /// <summary>
    /// The constant term in the GARCH variance equation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter represents the baseline level of variance in the GARCH model.
    /// It ensures that the conditional variance always has a minimum level, even when
    /// past squared residuals and past variances are close to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This is the minimum level of volatility in the model.
    /// 
    /// Omega (Ï‰):
    /// - Sets a baseline or minimum level of volatility
    /// - Ensures the model never predicts zero volatility
    /// - Represents the long-term average contribution to volatility
    /// 
    /// Think of it as the "background" volatility that's always present,
    /// even during the calmest market periods.
    /// </para>
    /// </remarks>
    private Vector<T> _omega; // Constant term in variance equation

    /// <summary>
    /// The coefficients for the ARCH terms in the variance equation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These coefficients determine how much the conditional variance depends on past squared residuals.
    /// Higher values indicate a stronger reaction of volatility to recent shocks or surprises in the data.
    /// </para>
    /// <para><b>For Beginners:</b> These determine how much recent surprises affect volatility.
    /// 
    /// Alpha (Î±) coefficients:
    /// - Measure how sensitive volatility is to recent surprises or shocks
    /// - Higher values mean volatility reacts strongly to new information
    /// - Lower values mean volatility is more stable
    /// 
    /// For example, if a stock has high alpha values, a sudden price jump
    /// will cause the model to predict higher volatility in the near future.
    /// </para>
    /// </remarks>
    private Vector<T> _alpha; // ARCH coefficients

    /// <summary>
    /// The coefficients for the GARCH terms in the variance equation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These coefficients determine how much the conditional variance depends on past conditional variances.
    /// Higher values indicate a stronger persistence of volatility over time.
    /// </para>
    /// <para><b>For Beginners:</b> These determine how persistent volatility is over time.
    /// 
    /// Beta (Î²) coefficients:
    /// - Measure how much current volatility depends on past volatility
    /// - Higher values mean volatility persists for longer periods
    /// - Lower values mean volatility dissipates more quickly
    /// 
    /// For example, if a stock has high beta values, a period of high volatility
    /// will likely be followed by more periods of high volatility.
    /// </para>
    /// </remarks>
    private Vector<T> _beta;  // GARCH coefficients

    /// <summary>
    /// The residuals (errors) from the mean model's predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the differences between the actual observations and the mean model's predictions.
    /// The GARCH model focuses on modeling the variance of these residuals.
    /// </para>
    /// <para><b>For Beginners:</b> These are the prediction errors from the mean model.
    /// 
    /// Residuals:
    /// - The differences between actual values and predicted values
    /// - Represent the "surprises" or unpredicted components
    /// - The GARCH model studies how these surprises vary over time
    /// 
    /// For example, if the mean model predicted $100 for a stock price
    /// but the actual price was $103, the residual would be $3.
    /// </para>
    /// </remarks>
    private Vector<T> _residuals;

    /// <summary>
    /// The estimated conditional variances for each time point in the series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the estimated variances for each time point based on the GARCH model.
    /// They represent the expected level of volatility at each point in the time series.
    /// </para>
    /// <para><b>For Beginners:</b> These are the estimated volatility levels at each time point.
    /// 
    /// Conditional variances:
    /// - Measure how much uncertainty exists at each time point
    /// - Higher values indicate higher expected volatility
    /// - Change over time based on recent data patterns
    /// 
    /// These values answer the question "How much might the actual value
    /// differ from our prediction?" at each point in time.
    /// </para>
    /// </remarks>
    private Vector<T> _conditionalVariances;

    /// <summary>
    /// Initializes a new instance of the <see cref="GARCHModel{T}"/> class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the GARCH model. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the GARCH model with the provided configuration options or default options if none
    /// are specified. The options determine parameters such as the ARCH and GARCH orders, the mean model to use,
    /// and various other settings that control the model's behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your GARCH model with your chosen settings.
    /// 
    /// When creating the model, you can specify:
    /// - ARCHOrder: How many past squared errors to consider
    /// - GARCHOrder: How many past volatilities to consider
    /// - MeanModel: The model used to predict the average value
    /// 
    /// For example:
    /// - GARCH(1,1) is common, meaning 1 past error term and 1 past volatility term
    /// - Higher orders (like GARCH(2,2)) capture more complex patterns but need more data
    /// 
    /// If you don't provide options, the model uses sensible defaults.
    /// </para>
    /// </remarks>
    public GARCHModel(GARCHModelOptions<T>? options = null) : base(options ?? new GARCHModelOptions<T>())
    {
        _garchOptions = (GARCHModelOptions<T>)Options;
        _meanModel = _garchOptions.MeanModel ?? new ARIMAModel<T>();
        _omega = new Vector<T>(1);
        _alpha = new Vector<T>(_garchOptions.ARCHOrder);
        _beta = new Vector<T>(_garchOptions.GARCHOrder);
        _residuals = new Vector<T>(0);
        _conditionalVariances = new Vector<T>(0);
    }

    /// <summary>
    /// Generates predictions for the given input data, including both mean and volatility forecasts.
    /// </summary>
    /// <param name="xNew">The input features matrix for the forecast period.</param>
    /// <returns>A vector containing the predicted values that incorporate both the mean forecast and volatility.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for future time periods by first predicting the mean using the mean model,
    /// then forecasting the conditional variance using the GARCH parameters, and finally generating residuals
    /// based on the forecasted variance to create the final predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This makes predictions that include both expected values and uncertainty.
    /// 
    /// The prediction process:
    /// 1. Predicts the average value for each future period using the mean model
    /// 2. Estimates how much uncertainty (volatility) to expect for each period
    /// 3. Generates realistic random variations based on this volatility
    /// 4. Combines the average prediction with these variations
    /// 
    /// This gives you not just a prediction, but a realistic scenario that includes
    /// the natural randomness seen in volatile data.
    /// 
    /// For example, instead of just predicting "tomorrow's stock price will be $100,"
    /// it might predict "tomorrow's stock price will be $100 with a likely range of $95-$105."
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> xNew)
    {
        int forecastHorizon = xNew.Rows;
        Vector<T> predictions = new Vector<T>(forecastHorizon);
        Vector<T> variances = new Vector<T>(forecastHorizon);

        // Predict mean using the mean model
        Vector<T> meanPredictions = _meanModel.Predict(xNew);

        // Initialize with the last known values
        T lastVariance = _conditionalVariances[_conditionalVariances.Length - 1];
        T lastResidual = _residuals[_residuals.Length - 1];

        for (int t = 0; t < forecastHorizon; t++)
        {
            // Calculate conditional variance
            T variance = _omega[0];
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                if (t - i - 1 >= 0)
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(lastResidual, lastResidual)));
                }
                else
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(lastResidual, lastResidual)));
                }
            }
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                if (t - i - 1 >= 0)
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], variances[t - i - 1]));
                }
                else
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], lastVariance));
                }
            }

            variances[t] = variance;
            lastVariance = variance;

            // Generate prediction
            T standardNormal = GenerateStandardNormal();
            T residual = NumOps.Multiply(NumOps.Sqrt(variance), standardNormal);
            predictions[t] = NumOps.Add(meanPredictions[t], residual);

            // Update last residual for the next iteration
            lastResidual = residual;
        }

        return predictions;
    }

    /// <summary>
    /// Generates a random number from a standard normal distribution.
    /// </summary>
    /// <returns>A random value from a standard normal distribution (mean 0, variance 1).</returns>
    /// <remarks>
    /// <para>
    /// This method uses the Box-Muller transform to generate a random number from a standard normal distribution.
    /// The transform converts uniformly distributed random numbers to normally distributed ones.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a random number that follows a bell curve pattern.
    /// 
    /// The standard normal distribution (also called a bell curve):
    /// - Has most values clustered around zero
    /// - Values farther from zero become increasingly rare
    /// - About 68% of values fall between -1 and 1
    /// - About 95% of values fall between -2 and 2
    /// 
    /// This random number generator is used to simulate realistic market movements,
    /// where small changes are common but extreme changes occasionally happen.
    /// </para>
    /// </remarks>
    private T GenerateStandardNormal()
    {
        Random random = RandomHelper.CreateSecureRandom();
        return NumOps.FromDouble(random.NextGaussian());
    }

    /// <summary>
    /// Initializes the GARCH model parameters with reasonable starting values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets initial values for the GARCH parameters (omega, alpha, beta) before the optimization process.
    /// The values are chosen to be consistent with commonly observed patterns in financial time series, with a small
    /// positive value for omega, a moderate value for alpha, and a larger value for beta.
    /// </para>
    /// <para><b>For Beginners:</b> This sets starting values for the model before fine-tuning.
    /// 
    /// The method sets:
    /// - Omega (the constant term): A small positive value (0.01)
    /// - Alpha (the ARCH coefficients): Moderate values (0.05)
    /// - Beta (the GARCH coefficients): Larger values (0.85)
    /// 
    /// These starting values are inspired by typical patterns seen in financial data,
    /// where current volatility depends heavily on recent volatility (high beta)
    /// and somewhat on recent shocks (moderate alpha).
    /// 
    /// Think of it as giving the model a head start in the right direction
    /// before it begins finding the optimal parameters.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Initialize parameters with small positive values
        _omega[0] = NumOps.FromDouble(0.01);
        for (int i = 0; i < _alpha.Length; i++)
        {
            _alpha[i] = NumOps.FromDouble(0.05);
        }
        for (int i = 0; i < _beta.Length; i++)
        {
            _beta[i] = NumOps.FromDouble(0.85);
        }
    }

    /// <summary>
    /// Estimates the optimal GARCH parameters using a gradient-based optimization approach.
    /// </summary>
    /// <param name="y">The residuals from the mean model.</param>
    /// <remarks>
    /// <para>
    /// This method optimizes the GARCH parameters (omega, alpha, beta) to maximize the log-likelihood
    /// of the observed data. It uses a gradient-based approach with momentum and adaptive learning rate,
    /// and ensures that parameters remain within valid ranges throughout the optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This finds the best settings to model your data's volatility patterns.
    /// 
    /// The optimization process:
    /// - Starts with initial parameter values
    /// - Repeatedly adjusts these values to better fit the data
    /// - Uses "gradient descent," which finds the direction of improvement
    /// - Includes "momentum" to avoid getting stuck in suboptimal solutions
    /// - Adapts the learning rate (step size) based on progress
    /// - Constrains parameters to maintain valid model properties
    /// 
    /// It's like climbing down a mountain in fog - you feel which way is downhill (the gradient),
    /// keep some momentum so you don't get stuck in small depressions, and take smaller steps
    /// when the terrain gets tricky.
    /// </para>
    /// </remarks>
    private void EstimateParameters(Vector<T> y)
    {
        int maxIterations = 10000;
        T initialLearningRate = NumOps.FromDouble(0.01);
        T minLearningRate = NumOps.FromDouble(1e-6);
        T convergenceThreshold = NumOps.FromDouble(1e-6);
        T momentumFactor = NumOps.FromDouble(0.9);

        Vector<T> previousOmega = _omega.Clone();
        Vector<T> previousAlpha = _alpha.Clone();
        Vector<T> previousBeta = _beta.Clone();

        Vector<T> velocityOmega = new Vector<T>(_omega.Length);
        Vector<T> velocityAlpha = new Vector<T>(_alpha.Length);
        Vector<T> velocityBeta = new Vector<T>(_beta.Length);

        T previousLogLikelihood = CalculateLogLikelihood(y);
        T currentLearningRate = initialLearningRate;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            Vector<T> gradientOmega = CalculateGradient(y, GradientType.Omega);
            Vector<T> gradientAlpha = CalculateGradient(y, GradientType.Alpha);
            Vector<T> gradientBeta = CalculateGradient(y, GradientType.Beta);

            // Update velocities (momentum)
            var momentumVecOmega = new Vector<T>(velocityOmega.Length);
            for (int i = 0; i < momentumVecOmega.Length; i++) momentumVecOmega[i] = momentumFactor;
            var lrVecOmega = new Vector<T>(gradientOmega.Length);
            for (int i = 0; i < lrVecOmega.Length; i++) lrVecOmega[i] = currentLearningRate;
            velocityOmega = (Vector<T>)Engine.Add((Vector<T>)Engine.Multiply(velocityOmega, momentumVecOmega),
                                                  (Vector<T>)Engine.Multiply(gradientOmega, lrVecOmega));

            var momentumVecAlpha = new Vector<T>(velocityAlpha.Length);
            for (int i = 0; i < momentumVecAlpha.Length; i++) momentumVecAlpha[i] = momentumFactor;
            var lrVecAlpha = new Vector<T>(gradientAlpha.Length);
            for (int i = 0; i < lrVecAlpha.Length; i++) lrVecAlpha[i] = currentLearningRate;
            velocityAlpha = (Vector<T>)Engine.Add((Vector<T>)Engine.Multiply(velocityAlpha, momentumVecAlpha),
                                                  (Vector<T>)Engine.Multiply(gradientAlpha, lrVecAlpha));

            var momentumVecBeta = new Vector<T>(velocityBeta.Length);
            for (int i = 0; i < momentumVecBeta.Length; i++) momentumVecBeta[i] = momentumFactor;
            var lrVecBeta = new Vector<T>(gradientBeta.Length);
            for (int i = 0; i < lrVecBeta.Length; i++) lrVecBeta[i] = currentLearningRate;
            velocityBeta = (Vector<T>)Engine.Add((Vector<T>)Engine.Multiply(velocityBeta, momentumVecBeta),
                                                 (Vector<T>)Engine.Multiply(gradientBeta, lrVecBeta));

            // Update parameters
            _omega = (Vector<T>)Engine.Subtract(_omega, velocityOmega);
            _alpha = (Vector<T>)Engine.Subtract(_alpha, velocityAlpha);
            _beta = (Vector<T>)Engine.Subtract(_beta, velocityBeta);

            // Ensure parameters stay within valid ranges
            ConstrainParameters();

            // Check for convergence
            T currentLogLikelihood = CalculateLogLikelihood(y);
            T improvement = NumOps.Subtract(currentLogLikelihood, previousLogLikelihood);

            if (NumOps.LessThan(NumOps.Abs(improvement), convergenceThreshold))
            {
                break;
            }

            // Adaptive learning rate
            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                currentLearningRate = NumOps.Multiply(currentLearningRate, NumOps.FromDouble(1.05)); // Increase learning rate
                previousOmega = _omega.Clone();
                previousAlpha = _alpha.Clone();
                previousBeta = _beta.Clone();
                previousLogLikelihood = currentLogLikelihood;
            }
            else
            {
                currentLearningRate = NumOps.Multiply(currentLearningRate, NumOps.FromDouble(0.5)); // Decrease learning rate
                _omega = previousOmega.Clone();
                _alpha = previousAlpha.Clone();
                _beta = previousBeta.Clone();

                if (NumOps.LessThan(currentLearningRate, minLearningRate))
                {
                    break; // Stop if learning rate becomes too small
                }
            }
        }
    }

    /// <summary>
    /// Calculates the gradient of the log-likelihood function with respect to the model parameters.
    /// </summary>
    /// <param name="y">The residuals from the mean model.</param>
    /// <param name="gradientType">The type of parameter to calculate the gradient for (Omega, Alpha, or Beta).</param>
    /// <returns>A vector containing the gradient values.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the gradient of the log-likelihood function with respect to the specified parameter type
    /// using numerical differentiation. The gradient indicates the direction and magnitude of the steepest increase
    /// in the log-likelihood, which guides the parameter updates during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how to adjust the parameters to better fit the data.
    /// 
    /// The gradient is like a compass that shows:
    /// - Which direction to move each parameter
    /// - How big of a step to take in that direction
    /// 
    /// The method:
    /// - Slightly increases each parameter and checks if the model fits better
    /// - Slightly decreases each parameter and checks if the model fits better
    /// - Uses these tests to figure out which way and how far to move
    /// 
    /// This process is called "numerical differentiation" and it's like testing the waters
    /// in different directions to see which way leads to improvement.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateGradient(Vector<T> y, GradientType gradientType)
    {
        T epsilon = NumOps.FromDouble(1e-6); // Small value for numerical differentiation
        T twoEpsilon = NumOps.Multiply(epsilon, NumOps.FromDouble(2));
        T logLikelihood = CalculateLogLikelihood(y);

        if (gradientType == GradientType.Omega)
        {
            Vector<T> gradient = new Vector<T>(1);
            T originalOmega = _omega[0];

            _omega[0] = NumOps.Add(originalOmega, epsilon);
            T logLikelihoodPlus = CalculateLogLikelihood(y);

            _omega[0] = NumOps.Subtract(originalOmega, epsilon);
            T logLikelihoodMinus = CalculateLogLikelihood(y);

            gradient[0] = NumOps.Divide(NumOps.Subtract(logLikelihoodPlus, logLikelihoodMinus), twoEpsilon);
            _omega[0] = originalOmega; // Restore original value

            return gradient;
        }
        else if (gradientType == GradientType.Alpha)
        {
            Vector<T> gradient = new Vector<T>(_garchOptions.ARCHOrder);
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                T originalAlpha = _alpha[i];

                _alpha[i] = NumOps.Add(originalAlpha, epsilon);
                T logLikelihoodPlus = CalculateLogLikelihood(y);

                _alpha[i] = NumOps.Subtract(originalAlpha, epsilon);
                T logLikelihoodMinus = CalculateLogLikelihood(y);

                gradient[i] = NumOps.Divide(NumOps.Subtract(logLikelihoodPlus, logLikelihoodMinus), twoEpsilon);
                _alpha[i] = originalAlpha; // Restore original value
            }
            return gradient;
        }
        else // beta
        {
            Vector<T> gradient = new Vector<T>(_garchOptions.GARCHOrder);
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                T originalBeta = _beta[i];

                _beta[i] = NumOps.Add(originalBeta, epsilon);
                T logLikelihoodPlus = CalculateLogLikelihood(y);

                _beta[i] = NumOps.Subtract(originalBeta, epsilon);
                T logLikelihoodMinus = CalculateLogLikelihood(y);

                gradient[i] = NumOps.Divide(NumOps.Subtract(logLikelihoodPlus, logLikelihoodMinus), twoEpsilon);
                _beta[i] = originalBeta; // Restore original value
            }
            return gradient;
        }
    }

    /// <summary>
    /// Calculates the log-likelihood of the data given the current model parameters.
    /// </summary>
    /// <param name="y">The residuals from the mean model.</param>
    /// <returns>The log-likelihood value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the log-likelihood of the observed data given the current model parameters. The log-likelihood
    /// is a measure of how well the model fits the data, with higher values indicating a better fit. It is used as the
    /// objective function during parameter optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how well the current model settings fit the data.
    /// 
    /// Log-likelihood:
    /// - Is a mathematical way to measure how well a model fits data
    /// - Higher values mean better fit (unlike error measures where lower is better)
    /// - Takes into account both the prediction error and the expected volatility
    /// 
    /// For example:
    /// - A large error during a high-volatility period might be acceptable
    /// - The same error during a low-volatility period would be penalized more heavily
    /// 
    /// This measure helps the model properly account for changing volatility over time,
    /// rather than treating all errors equally.
    /// </para>
    /// </remarks>
    private T CalculateLogLikelihood(Vector<T> y)
    {
        T logLikelihood = NumOps.Zero;
        int n = y.Length;
        Vector<T> conditionalVariances = CalculateConditionalVariances(y);

        for (int t = Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t < n; t++)
        {
            T variance = conditionalVariances[t];
            T residual = y[t]; // Assuming zero mean for simplicity
            T term = NumOps.Add(NumOps.Log(variance), NumOps.Divide(NumOps.Multiply(residual, residual), variance));
            logLikelihood = NumOps.Add(logLikelihood, term);
        }

        return NumOps.Multiply(NumOps.FromDouble(-0.5), logLikelihood);
    }

    /// <summary>
    /// Calculates the conditional variances for the time series given the current model parameters.
    /// </summary>
    /// <param name="y">The residuals from the mean model.</param>
    /// <returns>A vector containing the conditional variances for each time point.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the conditional variances for the time series based on the current GARCH parameters.
    /// The conditional variance at each time point depends on the unconditional variance (for initialization) and
    /// the ARCH and GARCH terms as specified by the model parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This estimates how volatile the data is at each point in time.
    /// 
    /// Conditional variance:
    /// - Measures how much uncertainty or volatility exists at each time point
    /// - "Conditional" means it depends on what happened in previous periods
    /// - Changes over time based on recent data patterns
    /// 
    /// The calculation:
    /// - Starts with an estimate of the long-term average variance (unconditional variance)
    /// - Updates this based on recent squared errors (ARCH terms) 
    /// - And recent variance estimates (GARCH terms)
    /// 
    /// This captures the tendency of volatility to cluster - high volatility periods
    /// tend to be followed by high volatility periods.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateConditionalVariances(Vector<T> y)
    {
        int n = y.Length;
        Vector<T> conditionalVariances = new Vector<T>(n);
        T unconditionalVariance = CalculateUnconditionalVariance(y);

        // Initialize with unconditional variance
        for (int t = 0; t < Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t++)
        {
            conditionalVariances[t] = unconditionalVariance;
        }

        // Calculate conditional variances
        for (int t = Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t < n; t++)
        {
            T variance = _omega[0];
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(y[t - i - 1], y[t - i - 1])));
            }
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], conditionalVariances[t - i - 1]));
            }
            conditionalVariances[t] = variance;
        }

        return conditionalVariances;
    }

    /// <summary>
    /// Constrains the GARCH parameters to ensure they remain within valid ranges.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method enforces constraints on the GARCH parameters to ensure they remain within valid ranges.
    /// Specifically, it ensures that all parameters are non-negative and that the sum of ARCH and GARCH
    /// coefficients is less than 1, which is a necessary condition for the model to be stationary.
    /// </para>
    /// <para><b>For Beginners:</b> This keeps the model parameters within sensible limits.
    /// 
    /// The constraints ensure:
    /// - All parameters are positive (negative values wouldn't make statistical sense)
    /// - The sum of alpha and beta coefficients is less than 1 (typically 0.99)
    /// 
    /// The second constraint is particularly important:
    /// - If the sum exceeds 1, the model becomes "non-stationary"
    /// - Non-stationary models can produce forecasts with infinitely increasing volatility
    /// - This constraint keeps the model well-behaved and realistic
    /// 
    /// Think of it like guardrails that prevent the optimization process
    /// from wandering into invalid or unstable territory.
    /// </para>
    /// </remarks>
    private void ConstrainParameters()
    {
        // Ensure all parameters are non-negative
        _omega[0] = MathHelper.Max(_omega[0], NumOps.Zero);
        for (int i = 0; i < _alpha.Length; i++)
        {
            _alpha[i] = MathHelper.Max(_alpha[i], NumOps.Zero);
        }
        for (int i = 0; i < _beta.Length; i++)
        {
            _beta[i] = MathHelper.Max(_beta[i], NumOps.Zero);
        }

        // Ensure the sum of ARCH and GARCH coefficients is less than 1 for stationarity
        T alphaSum = Engine.Sum(_alpha);
        T betaSum = Engine.Sum(_beta);
        T sum = NumOps.Add(alphaSum, betaSum);
        if (NumOps.GreaterThan(sum, NumOps.One))
        {
            T scaleFactor = NumOps.Divide(NumOps.FromDouble(0.99), sum);
            var scaleVecAlpha = new Vector<T>(_alpha.Length);
            for (int i = 0; i < scaleVecAlpha.Length; i++) scaleVecAlpha[i] = scaleFactor;
            _alpha = (Vector<T>)Engine.Multiply(_alpha, scaleVecAlpha);

            var scaleVecBeta = new Vector<T>(_beta.Length);
            for (int i = 0; i < scaleVecBeta.Length; i++) scaleVecBeta[i] = scaleFactor;
            _beta = (Vector<T>)Engine.Multiply(_beta, scaleVecBeta);
        }
    }

    /// <summary>
    /// Calculates the final residuals and conditional variances based on the trained model parameters.
    /// </summary>
    /// <param name="residuals">The residuals from the mean model.</param>
    /// <remarks>
    /// <para>
    /// This method calculates the final residuals and conditional variances for the time series based on the
    /// trained model parameters. These values are stored for use during forecasting and evaluation.
    /// </para>
    /// <para><b>For Beginners:</b> This computes the final volatility estimates based on the trained model.
    /// 
    /// After finding the best parameters (omega, alpha, beta), this method:
    /// - Calculates the final volatility estimate for each time point in your data
    /// - Stores these estimates for later use in forecasting
    /// 
    /// These volatility estimates show:
    /// - Where volatility was high in your historical data
    /// - Where volatility was low
    /// - How volatility patterns changed over time
    /// 
    /// These patterns provide the foundation for forecasting future volatility.
    /// </para>
    /// </remarks>
    private void CalculateResidualsAndVariances(Vector<T> residuals)
    {
        int n = residuals.Length;
        _conditionalVariances = new Vector<T>(n);

        // Initialize with unconditional variance
        T unconditionalVariance = CalculateUnconditionalVariance(residuals);
        for (int t = 0; t < Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t++)
        {
            _conditionalVariances[t] = unconditionalVariance;
        }

        // Calculate conditional variances
        for (int t = Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t < n; t++)
        {
            T variance = _omega[0];
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(residuals[t - i - 1], residuals[t - i - 1])));
            }
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], _conditionalVariances[t - i - 1]));
            }
            _conditionalVariances[t] = variance;
        }
    }

    /// <summary>
    /// Calculates the unconditional variance of the time series data.
    /// </summary>
    /// <param name="y">The residuals from the mean model.</param>
    /// <returns>The unconditional variance of the time series.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the unconditional variance of the time series, which represents the long-term average
    /// level of volatility in the data. It is used for initializing the conditional variances at the beginning of the series.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates the average long-term volatility in your data.
    /// 
    /// Unconditional variance:
    /// - Is the average or baseline volatility in your data
    /// - Doesn't change over time (unlike conditional variance)
    /// - Serves as a starting point for variance calculations
    /// 
    /// Think of it as the "normal" level of variability in your data.
    /// For example, a stock might have an unconditional variance corresponding to
    /// about 1% daily price changes on average, but the conditional variance
    /// might be higher during market stress or lower during calm periods.
    /// </para>
    /// </remarks>
    private T CalculateUnconditionalVariance(Vector<T> y)
    {
        return StatisticsHelper<T>.CalculateVariance(y);
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">The test input features matrix.</param>
    /// <param name="yTest">The test target values vector.</param>
    /// <returns>A dictionary containing various evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates the model's performance on test data by generating predictions and calculating
    /// various error metrics. The returned metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    /// Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
    /// </para>
    /// <para><b>For Beginners:</b> This measures how accurate the model's predictions are.
    /// 
    /// The evaluation:
    /// - Makes predictions for data the model hasn't seen before
    /// - Compares these predictions to the actual values
    /// - Calculates different types of error measurements:
    ///   - MSE (Mean Squared Error): Average of squared differences
    ///   - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as your data
    ///   - MAE (Mean Absolute Error): Average of absolute differences
    ///   - MAPE (Mean Absolute Percentage Error): Average percentage difference
    /// 
    /// These metrics help you understand how well your model is likely to perform
    /// when forecasting new data. Lower values indicate better performance.
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
    /// This method writes the model's essential parameters to a binary stream, allowing the model to be saved
    /// to a file or database. The serialized parameters include the GARCH parameters (omega, alpha, beta),
    /// the residuals, the conditional variances, and the model options.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the model so you can use it later.
    /// 
    /// The method:
    /// - Converts the model's parameters to a format that can be saved
    /// - Writes these values to a file or database
    /// - Includes all the information needed to recreate the model exactly
    /// 
    /// This allows you to:
    /// - Save a trained model for future use
    /// - Share the model with others
    /// - Use the model in other applications
    /// 
    /// It's like saving a document so you can open it again later without
    /// having to start from scratch.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        SerializationHelper<T>.SerializeVector(writer, _omega);
        SerializationHelper<T>.SerializeVector(writer, _alpha);
        SerializationHelper<T>.SerializeVector(writer, _beta);
        SerializationHelper<T>.SerializeVector(writer, _residuals);
        SerializationHelper<T>.SerializeVector(writer, _conditionalVariances);

        writer.Write(JsonConvert.SerializeObject(_garchOptions));

        // Serialize the mean model so it can be restored on deserialization
        byte[] meanModelBytes = _meanModel.Serialize();
        writer.Write(meanModelBytes.Length);
        writer.Write(meanModelBytes);
    }

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the model's essential parameters from a binary stream, allowing a previously saved model
    /// to be loaded from a file or database. The deserialized parameters include the GARCH parameters (omega, alpha, beta),
    /// the residuals, the conditional variances, and the model options.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved model.
    /// 
    /// The method:
    /// - Reads the saved model data from a file or database
    /// - Converts this data back into the model's parameters
    /// - Reconstructs the model exactly as it was when saved
    /// 
    /// This is particularly useful when:
    /// - You want to use a model that took a long time to train
    /// - You want to ensure consistent results across different runs
    /// - You need to deploy the model in a production environment
    /// 
    /// Think of it like opening a document you previously saved, allowing you
    /// to continue using the model without having to train it again.
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        _omega = SerializationHelper<T>.DeserializeVector(reader);
        _alpha = SerializationHelper<T>.DeserializeVector(reader);
        _beta = SerializationHelper<T>.DeserializeVector(reader);
        _residuals = SerializationHelper<T>.DeserializeVector(reader);
        _conditionalVariances = SerializationHelper<T>.DeserializeVector(reader);

        string optionsJson = reader.ReadString();
        _garchOptions = JsonConvert.DeserializeObject<GARCHModelOptions<T>>(optionsJson) ?? new();

        // Deserialize the mean model to restore its trained state
        int meanModelBytesLength = reader.ReadInt32();
        byte[] meanModelBytes = reader.ReadBytes(meanModelBytesLength);
        _meanModel = _garchOptions.MeanModel ?? new ARIMAModel<T>();
        _meanModel.Deserialize(meanModelBytes);
    }

    /// <summary>
    /// Resets the model to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the GARCH model to its initial state, clearing any learned parameters and returning to the
    /// initial configuration provided in the options. This is useful when you want to retrain the model from scratch.
    /// </para>
    /// <para><b>For Beginners:</b> This resets the model to start fresh.
    /// 
    /// Resetting the model:
    /// - Clears all learned parameters (omega, alpha, beta)
    /// - Reinitializes the mean model
    /// - Empties stored residuals and variances
    /// - Returns the model to its original state before training
    /// 
    /// This is useful when you want to:
    /// - Train the model on different data
    /// - Try different settings or approaches
    /// - Start with a clean slate after experimentation
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        _omega = new Vector<T>(1);
        _alpha = new Vector<T>(_garchOptions.ARCHOrder);
        _beta = new Vector<T>(_garchOptions.GARCHOrder);
        _residuals = new Vector<T>(0);
        _conditionalVariances = new Vector<T>(0);

        // Initialize with reasonable default values
        InitializeParameters();
    }

    /// <summary>
    /// Creates a new instance of the GARCH model with the same options.
    /// </summary>
    /// <returns>A new instance of the GARCH model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the GARCH model with the same configuration options as the current instance.
    /// This is useful for creating copies or clones of the model for purposes like cross-validation or ensemble modeling.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new copy of the model with the same settings.
    /// 
    /// Creating a new instance:
    /// - Makes a fresh copy of the model with the same configuration
    /// - The new copy hasn't been trained yet
    /// - You can train and use the copy independently from the original
    /// 
    /// This is helpful when you want to:
    /// - Train multiple versions of the same model on different data subsets
    /// - Compare different training approaches while keeping the base configuration the same
    /// - Create an ensemble of similar models
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new GARCHModel<T>(_garchOptions);
    }

    /// <summary>
    /// Returns metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed metadata about the GARCH model, including its type, current parameters (omega, alpha, beta),
    /// and configuration options. This metadata can be used for model selection, comparison, documentation, and serialization purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about your model's settings and state.
    /// 
    /// The metadata includes:
    /// - The type of model (GARCH)
    /// - Current parameter values (omega, alpha, beta)
    /// - Information about the mean model
    /// - Configuration settings from when you created the model
    /// - A serialized version of the entire model
    /// 
    /// This information is useful for:
    /// - Keeping track of different models you've created
    /// - Comparing model configurations
    /// - Documenting which settings worked best
    /// - Sharing model information with others
    /// - Storing model details in a database or registry
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.GARCHModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include the actual model state variables
                { "Omega", _omega },
                { "Alpha", _alpha },
                { "Beta", _beta },
                { "MeanModel", _meanModel.GetType().Name },
            
                // Include model configuration as well
                { "ARCHOrder", _garchOptions.ARCHOrder },
                { "GARCHOrder", _garchOptions.GARCHOrder },
                { "UseMeanModel", _meanModel != null },
            },
            ModelData = this.Serialize()
        };
        return metadata;
    }

    /// <summary>
    /// Core implementation of the training logic for the GARCH model.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector (the time series data to model).</param>
    /// <remarks>
    /// <para>
    /// This method implements the core training mechanism for the GARCH model, which involves
    /// training the mean model first, then estimating the parameters of the variance equation
    /// based on the residuals. It follows a multi-step process that captures both the average
    /// behavior of the series and its changing volatility patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine that powers the model's learning process.
    /// 
    /// This method:
    /// 1. Trains the model that predicts the average values (mean model)
    /// 2. Calculates the errors in these predictions (residuals)
    /// 3. Sets up initial GARCH parameters to model the volatility
    /// 4. Finds the optimal parameter values to best explain the volatility patterns
    /// 5. Calculates the final volatility estimates for the historical data
    /// 
    /// It's the actual "learning" part of the model, where it discovers patterns
    /// in both the values and the volatility of your time series data.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Train the mean model
        _meanModel.Train(x, y);

        // Step 2: Calculate residuals from the mean model
        Vector<T> meanPredictions = _meanModel.Predict(x);
        _residuals = (Vector<T>)Engine.Subtract(y, meanPredictions);

        // Step 3: Initialize GARCH parameters
        InitializeParameters();

        // Step 4: Estimate GARCH parameters using Maximum Likelihood Estimation
        EstimateParameters(_residuals);

        // Step 5: Calculate final residuals and conditional variances
        CalculateResidualsAndVariances(_residuals);
    }

    /// <summary>
    /// Predicts a single value based on the input vector.
    /// </summary>
    /// <param name="input">The input vector containing features for the prediction.</param>
    /// <returns>The predicted value for the given input.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a single prediction by combining the mean prediction from the mean model
    /// with a simulated residual based on the estimated volatility. It creates appropriate context
    /// for the prediction and ensures consistency with the model's learned patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This generates a prediction for a single point in time.
    /// 
    /// The method:
    /// 1. Creates a context for generating a single prediction
    /// 2. Gets the mean prediction (expected value) from the mean model
    /// 3. Estimates the volatility (uncertainty) for this time point
    /// 4. Generates a realistic random variation based on this volatility
    /// 5. Combines the mean prediction with the random variation
    /// 
    /// This approach provides not just an expected value, but also incorporates
    /// the appropriate level of randomness based on current volatility conditions.
    /// It's like predicting that tomorrow's temperature will be 75Â°F, but with 
    /// a range of Â±3Â° because weather conditions are currently stable.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Create a matrix with a single row for the prediction
        Matrix<T> inputMatrix = new Matrix<T>(1, input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            inputMatrix[0, i] = input[i];
        }

        // Make sure we have some volatility estimate to work with
        if (_conditionalVariances.Length == 0)
        {
            throw new InvalidOperationException("Model has not been trained. Cannot make predictions without volatility estimates.");
        }

        // Get the mean prediction from the mean model
        T meanPrediction = _meanModel.PredictSingle(input);

        // Get the last known conditional variance
        T lastVariance = _conditionalVariances[_conditionalVariances.Length - 1];

        // Generate a random residual based on the estimated variance
        T standardNormal = GenerateStandardNormal();
        T residual = NumOps.Multiply(NumOps.Sqrt(lastVariance), standardNormal);

        // Combine the mean prediction with the residual
        return NumOps.Add(meanPrediction, residual);
    }
}
