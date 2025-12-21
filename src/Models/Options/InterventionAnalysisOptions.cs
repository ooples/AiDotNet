namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Intervention Analysis, which is a time series modeling technique used to
/// assess the impact of specific events or interventions on a time series.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Intervention Analysis extends time series regression by explicitly modeling the effects of known
/// interventions or events (like policy changes, marketing campaigns, or natural disasters) on a time
/// series. It combines ARIMA (AutoRegressive Integrated Moving Average) modeling with additional
/// components that represent these interventions, allowing you to quantify their impact while accounting
/// for the underlying time series patterns.
/// </para>
/// <para><b>For Beginners:</b> Intervention Analysis helps you measure how specific events affected your
/// time series data. For example, if you're tracking daily sales and launched a major marketing campaign
/// on a specific date, intervention analysis can help you determine how much that campaign actually
/// boosted your sales while accounting for other factors like seasonal patterns or existing trends.
/// 
/// Think of it like trying to measure the effect of a new medication on a patient's health while
/// accounting for their normal day-to-day fluctuations in health metrics. Just looking at the raw
/// numbers before and after might be misleading - intervention analysis gives you a more accurate
/// picture of the true impact.
/// 
/// This class inherits from TimeSeriesRegressionOptions, so all the general time series regression
/// settings are also available. The additional settings specific to intervention analysis let you
/// configure how the model handles both the time series patterns and the intervention effects.</para>
/// </remarks>
public class InterventionAnalysisOptions<T, TInput, TOutput> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the order of the AutoRegressive (AR) component in the ARIMA model.
    /// </summary>
    /// <value>The AR order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The AR order specifies how many previous time points directly influence the current value in the
    /// time series model. Higher values allow the model to capture more complex patterns of autocorrelation
    /// but increase the risk of overfitting and computational complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many previous time periods your model
    /// should look at to predict the current value. With the default value of 1, the model considers
    /// only the immediately previous time period when making predictions.
    /// 
    /// Think of it like predicting tomorrow's weather based on today's weather (AR order = 1) versus
    /// predicting it based on today's, yesterday's, and the day before yesterday's weather (AR order = 3).
    /// A higher AR order can capture more complex patterns but might also pick up on random fluctuations
    /// that don't actually help with prediction.
    /// 
    /// For example, with daily sales data:
    /// - AR order = 1: Today's sales depend on yesterday's sales
    /// - AR order = 7: Today's sales depend on sales from each of the previous 7 days
    /// 
    /// Start with the default value of 1 and increase it if you believe there are longer-term dependencies
    /// in your data. For many business time series, values between 1 and 7 are common, often corresponding
    /// to dependencies on the previous day, week, or month.</para>
    /// </remarks>
    public int AROrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the Moving Average (MA) component in the ARIMA model.
    /// </summary>
    /// <value>The MA order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The MA order specifies how many previous error terms (differences between predicted and actual values)
    /// influence the current value in the time series model. Higher values allow the model to account for
    /// more complex patterns in the error terms but increase the risk of overfitting and computational complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many previous prediction errors your model
    /// should consider when making the current prediction. With the default value of 1, the model adjusts
    /// its current prediction based on how wrong its previous prediction was.
    /// 
    /// While the AR component looks at previous actual values, the MA component looks at previous prediction
    /// errors. This helps the model correct for systematic errors it might be making.
    /// 
    /// Think of it like a weather forecaster who not only looks at previous weather (AR) but also adjusts
    /// based on how wrong their recent forecasts were (MA). If they consistently underestimated temperatures
    /// last week, they might adjust this week's predictions upward.
    /// 
    /// For example, with daily sales data:
    /// - MA order = 1: Today's prediction is adjusted based on yesterday's prediction error
    /// - MA order = 3: Today's prediction is adjusted based on prediction errors from the past 3 days
    /// 
    /// The default value of 1 works well for many applications. Higher values might help if there are
    /// complex patterns in how your model's errors behave over time.</para>
    /// </remarks>
    public int MAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the list of interventions to be analyzed in the time series.
    /// </summary>
    /// <value>A list of intervention information objects, defaulting to an empty list.</value>
    /// <remarks>
    /// <para>
    /// Each intervention is defined by its timing (when it occurred) and type (how it affected the time series).
    /// Common intervention types include pulse (temporary effect), step (permanent level change), and ramp
    /// (gradual change). The model will estimate the magnitude and significance of each intervention's effect.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you specify the events or changes you want to analyze.
    /// Each intervention needs information about when it happened and what type of effect you expect it to have.
    /// 
    /// For example, if you're analyzing website traffic data and launched a new marketing campaign on
    /// June 1, 2023, you would add an intervention with that date. You'd also specify whether you expect:
    /// - A temporary spike in traffic (pulse intervention)
    /// - A permanent increase in the baseline traffic level (step intervention)
    /// - A gradual increase that builds over time (ramp intervention)
    /// 
    /// You can add multiple interventions to analyze several events at once. For instance, you might want
    /// to analyze both a marketing campaign and a website redesign that happened on different dates.
    /// 
    /// The model will then estimate how much each intervention actually affected your time series,
    /// accounting for other factors like seasonal patterns and existing trends.</para>
    /// </remarks>
    public List<InterventionInfo> Interventions { get; set; } = new List<InterventionInfo>();

    /// <summary>
    /// Gets or sets the optimizer used to find the best parameters for the intervention analysis model.
    /// </summary>
    /// <value>The optimizer instance, defaulting to null (which will use a default optimizer).</value>
    /// <remarks>
    /// <para>
    /// The optimizer is responsible for finding the parameter values that minimize the error between the
    /// model's predictions and the actual time series values. Different optimizers may have different
    /// performance characteristics depending on the specific time series and interventions being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you choose the algorithm that will be used to find
    /// the best possible model parameters. If you leave it as null (the default), the system will choose
    /// an appropriate optimizer for you.
    /// 
    /// Think of the optimizer as the "learner" that figures out exactly how much each intervention affected
    /// your time series and how strong the time series patterns are. Different optimizers use different
    /// strategies to find these values, similar to how different people might use different strategies to
    /// find the lowest price for a product.
    /// 
    /// Most beginners should leave this as null and let the system choose an appropriate optimizer.
    /// Advanced users might want to specify a particular optimizer if they have specific requirements
    /// regarding speed, precision, or if they're dealing with unusual time series patterns.</para>
    /// </remarks>
    public IOptimizer<T, TInput, TOutput>? Optimizer { get; set; }
}
