namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Learning Curve Fit Detector, which analyzes training progress
/// to determine when a model has converged or is unlikely to improve further.
/// </summary>
/// <remarks>
/// <para>
/// The Learning Curve Fit Detector monitors the training progress of a machine learning model by
/// analyzing the pattern of error reduction over time. It fits a mathematical curve to the error
/// values and uses this to predict whether continued training is likely to yield significant
/// improvements. This can help automatically determine when to stop training.
/// </para>
/// <para><b>For Beginners:</b> When training a machine learning model, it's important to know when
/// to stop. Training for too long can waste time or cause overfitting (where the model performs
/// well on training data but poorly on new data), while stopping too early might leave the model
/// under-trained.
/// 
/// The Learning Curve Fit Detector is like a smart assistant that watches how your model's performance
/// improves during training. It looks at the pattern of improvement and tries to predict whether
/// continuing to train will give meaningful benefits or if the model has already learned as much as
/// it can from the data.
/// 
/// Think of it like watching someone learn a new skill:
/// - At first, they improve quickly (steep learning curve)
/// - Over time, the rate of improvement slows down (flattening curve)
/// - Eventually, they reach a plateau where more practice yields minimal improvement
/// 
/// This class lets you configure how the detector decides when that plateau has been reached,
/// allowing you to automatically stop training at the right time.</para>
/// </remarks>
public class LearningCurveFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold that determines when the model is considered to have converged.
    /// </summary>
    /// <value>The convergence threshold, defaulting to 0.01 (1%).</value>
    /// <remarks>
    /// <para>
    /// This threshold represents the predicted percentage improvement in the error metric if training
    /// were to continue. When the predicted improvement falls below this threshold, the detector
    /// considers the model to have converged. Lower values require more evidence of convergence
    /// before stopping training.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strict the detector is when deciding
    /// if your model has finished learning.
    /// 
    /// The convergence threshold represents how much improvement we expect to see if training continues.
    /// For example, with the default value of 0.01 (or 1%):
    /// 
    /// - If the detector predicts that continuing to train would improve your model's performance by
    ///   less than 1%, it will suggest stopping because further training isn't worth the time.
    /// - If it predicts an improvement of more than 1%, it will suggest continuing training.
    /// 
    /// You can adjust this value based on your needs:
    /// - Lower values (like 0.001 or 0.1%) make the detector more patient, continuing training until
    ///   even tiny improvements are unlikely. This might give slightly better results but takes longer.
    /// - Higher values (like 0.05 or 5%) make the detector stop training earlier, saving time but
    ///   potentially leaving some performance on the table.
    /// 
    /// The default of 1% is a good balance for most problems, ensuring significant improvements are
    /// captured while avoiding diminishing returns.</para>
    /// </remarks>
    public double ConvergenceThreshold { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the minimum number of data points (training iterations) required before
    /// the detector will attempt to predict convergence.
    /// </summary>
    /// <value>The minimum number of data points, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// To make reliable predictions about convergence, the detector needs a sufficient history of
    /// error values from previous training iterations. This parameter specifies how many data points
    /// must be collected before the detector will begin making predictions. Setting this too low may
    /// result in premature convergence detection, while setting it too high delays detection.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many training steps must complete
    /// before the detector starts making decisions about stopping.
    /// 
    /// To predict whether your model has converged, the detector needs to see a pattern in how the
    /// performance changes over time. This setting specifies how many measurements it needs before
    /// it starts making these predictions.
    /// 
    /// With the default value of 5:
    /// - During the first 5 training iterations, the detector will always recommend continuing
    ///   because it doesn't have enough information yet
    /// - After 5 iterations, it will start analyzing the pattern to decide if training should continue
    /// 
    /// Think of it like trying to predict the weather:
    /// - With only 1-2 measurements, it's hard to see a reliable pattern
    /// - With 5+ measurements, you can start to see trends and make better predictions
    /// 
    /// You might want to increase this value (to 10 or more) for complex models or noisy training
    /// processes where performance fluctuates a lot between iterations. For simpler, more stable
    /// training processes, the default of 5 is usually sufficient.</para>
    /// </remarks>
    public int MinDataPoints { get; set; } = 5;
}
