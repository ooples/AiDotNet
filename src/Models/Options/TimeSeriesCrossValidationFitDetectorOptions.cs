namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for detecting overfitting, underfitting, and model stability in time series models
/// using cross-validation techniques.
/// </summary>
/// <remarks>
/// <para>
/// Time series cross-validation is a technique for evaluating the performance and generalization ability of time 
/// series forecasting models. Unlike standard cross-validation used for non-time series data, time series 
/// cross-validation respects the temporal order of observations, typically using a rolling window or expanding 
/// window approach. This class provides configuration options for thresholds used to detect common modeling 
/// issues such as overfitting (where the model performs well on training data but poorly on validation data), 
/// underfitting (where the model performs poorly on both training and validation data), and high variance 
/// (where model performance varies significantly across different validation periods). These thresholds help 
/// automate the process of model evaluation and selection for time series forecasting tasks.
/// </para>
/// <para><b>For Beginners:</b> This class helps you detect common problems when training time series forecasting models.
/// 
/// When building time series forecasting models:
/// - Overfitting: Model learns patterns specific to historical data that don't generalize to future data
/// - Underfitting: Model is too simple to capture important patterns in the data
/// - High variance: Model performance changes dramatically across different time periods
/// 
/// Time series cross-validation:
/// - Tests your model on multiple time periods
/// - Respects the temporal nature of the data (unlike regular cross-validation)
/// - Usually involves training on earlier data and testing on later data
/// - Helps assess how well your model will perform on future, unseen data
/// 
/// This class provides thresholds to automatically detect these issues based on
/// cross-validation results, helping you diagnose and fix model training problems.
/// </para>
/// </remarks>
public class TimeSeriesCrossValidationFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.2.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum acceptable ratio of training error to validation error. If the validation 
    /// error is more than this threshold times the training error, the model is considered to be overfitting. For 
    /// example, with the default value of 1.2, if the training error is 10 and the validation error is greater than 
    /// 12, the model would be flagged as overfitting. This threshold is expressed as a ratio rather than an absolute 
    /// difference because the scale of errors can vary widely across different time series. A smaller threshold is 
    /// more strict, flagging smaller differences as overfitting, while a larger threshold is more lenient. The 
    /// appropriate value depends on the specific application and the expected difference between in-sample and 
    /// out-of-sample performance for a well-fitted model.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much worse a model can perform on validation data versus training data before it's considered overfitting.
    /// 
    /// Overfitting in time series occurs when:
    /// - A model performs significantly worse on validation periods than on training periods
    /// - It has essentially "memorized" the training data rather than learning general patterns
    /// 
    /// The default value of 1.2 means:
    /// - If validation error is more than 20% higher than training error, the model is overfitting
    /// - For example, if training RMSE is 10 and validation RMSE is 13, that's overfitting
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 1.1): More strict, flags smaller differences as overfitting
    /// - Higher values (e.g., 1.5): More lenient, allows larger differences before flagging overfitting
    /// 
    /// When to adjust this value:
    /// - Decrease it when working with stable time series where training and validation should be very close
    /// - Increase it for volatile time series where some gap is expected and acceptable
    /// 
    /// For example, in financial time series that are known to be volatile,
    /// you might increase this to 1.3-1.5 since some gap between training and
    /// validation performance is normal.
    /// </para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 1.2;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum acceptable performance relative to a naive benchmark model. The 
    /// performance is typically measured as the ratio of the model's error to the benchmark model's error, so 
    /// lower values indicate better performance. If this ratio exceeds the threshold, the model is considered 
    /// to be underfitting. For example, with the default value of 0.5, if the benchmark model has an error of 
    /// 100 and the evaluated model has an error greater than 50, the model would be flagged as underfitting. 
    /// Common benchmark models for time series include the naive forecast (using the last observed value) or 
    /// seasonal naive forecast (using the value from the same season in the previous period). A lower threshold 
    /// is more strict, requiring better performance relative to the benchmark, while a higher threshold is more 
    /// lenient. The appropriate value depends on the specific application and the minimum acceptable improvement 
    /// over the benchmark for the model to be considered useful.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how well a model must perform compared to a simple benchmark to avoid being considered underfitting.
    /// 
    /// Underfitting in time series occurs when:
    /// - A model doesn't perform much better than simple benchmark methods
    /// - It's too simple to capture the underlying patterns in the data
    /// 
    /// The default value of 0.5 means:
    /// - The model's error should be at most 50% of the benchmark error
    /// - For example, if a naive forecast has RMSE of 100, your model should have RMSE of 50 or less
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 0.3): More strict, requires better performance compared to benchmark
    /// - Higher values (e.g., 0.7): More lenient, allows performance closer to benchmark
    /// 
    /// When to adjust this value:
    /// - Decrease it for problems where sophisticated models should significantly outperform benchmarks
    /// - Increase it for difficult forecasting problems where even modest improvements are valuable
    /// 
    /// For example, in retail demand forecasting where patterns are clear,
    /// you might decrease this to 0.4 to ensure your model provides substantial
    /// improvement over simple methods.
    /// </para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the threshold for detecting high variance.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum acceptable coefficient of variation (standard deviation divided by mean) 
    /// of the model's performance across different validation periods. If the coefficient of variation exceeds this 
    /// threshold, the model is considered to have high variance. For example, with the default value of 1.1, if the 
    /// mean error across validation periods is 10 and the standard deviation is greater than 11, the model would be 
    /// flagged as having high variance. A smaller threshold is more strict, requiring more consistent performance 
    /// across different periods, while a larger threshold is more lenient. The appropriate value depends on the 
    /// specific application and the expected stability of the time series across different periods. For highly 
    /// volatile or non-stationary time series, a larger threshold might be appropriate, while for more stable series, 
    /// a smaller threshold might be preferred.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how consistent a model's performance must be across different time periods.
    /// 
    /// High variance in time series models occurs when:
    /// - Performance varies significantly across different validation periods
    /// - The model is too sensitive to specific time periods
    /// - It can't maintain consistent performance across the entire time range
    /// 
    /// The default value of 1.1 means:
    /// - If the coefficient of variation (standard deviation ÷ mean) of errors exceeds 1.1, the model has high variance
    /// - This indicates the model's performance is too inconsistent across different periods
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 0.8): More strict, requires more consistent performance across periods
    /// - Higher values (e.g., 1.5): More lenient, allows more variation across periods
    /// 
    /// When to adjust this value:
    /// - Decrease it when consistent performance across time periods is critical
    /// - Increase it for highly volatile time series where some variation in performance is expected
    /// 
    /// For example, in utility load forecasting where consistent reliability is essential,
    /// you might decrease this to 0.9 to ensure the model performs consistently
    /// across different seasons and conditions.
    /// </para>
    /// </remarks>
    public double HighVarianceThreshold { get; set; } = 1.1;

    /// <summary>
    /// Gets or sets the threshold for determining a good fit.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.8.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum acceptable improvement over a naive benchmark model for the model to be 
    /// considered a good fit. The improvement is typically measured as 1 minus the ratio of the model's error to 
    /// the benchmark model's error, so higher values indicate better performance. If this improvement is below the 
    /// threshold, the model is not considered a good fit, even if it doesn't exhibit overfitting, underfitting, or 
    /// high variance. For example, with the default value of 0.8, the model's error must be at most 20% of the 
    /// benchmark error (an 80% improvement) to be considered a good fit. A higher threshold is more strict, requiring 
    /// better performance relative to the benchmark, while a lower threshold is more lenient. The appropriate value 
    /// depends on the specific application and the minimum acceptable improvement over the benchmark for the model 
    /// to be considered useful.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much better than a benchmark a model must perform to be considered a good fit.
    /// 
    /// A good fit in time series means:
    /// - The model significantly outperforms simple benchmark methods
    /// - It captures the important patterns in the data
    /// - It generalizes well to validation periods
    /// 
    /// The default value of 0.8 means:
    /// - The model must achieve at least an 80% improvement over the benchmark
    /// - For example, if benchmark RMSE is 100, model RMSE must be 20 or less
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.9): More strict, requires excellent performance compared to benchmark
    /// - Lower values (e.g., 0.6): More lenient, accepts more modest improvements over benchmark
    /// 
    /// When to adjust this value:
    /// - Increase it for applications where high forecast accuracy is critical
    /// - Decrease it for difficult forecasting problems where even modest improvements are valuable
    /// 
    /// For example, in inventory optimization where forecast accuracy directly impacts costs,
    /// you might increase this to 0.85-0.9 to ensure your model provides substantial
    /// improvement over simple methods.
    /// </para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.8;
}
