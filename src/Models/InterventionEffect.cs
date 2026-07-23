namespace AiDotNet.Models;

/// <summary>
/// Represents the effect of an intervention in a time series or sequential data, capturing the starting point,
/// duration, and magnitude of the effect.
/// </summary>
/// <remarks>
/// <para>
/// This class models an intervention effect, which is a change or treatment applied to a system at a specific point 
/// in time that continues for a certain duration and produces a measurable effect. Interventions are commonly analyzed 
/// in time series analysis, causal inference, and experimental studies to understand how specific actions or events 
/// affect outcomes over time. The class captures the essential information about an intervention: when it started, 
/// how long it lasted, and how strong its effect was.
/// </para>
/// <para><b>For Beginners:</b> This class represents a change or treatment that was applied at a specific point in time and had some measurable effect.
/// 
/// For example, you might use this to model:
/// - The effect of a marketing campaign that ran for 2 weeks
/// - The impact of a policy change that was implemented on a specific date
/// - The result of a medical treatment that was administered for a certain period
/// 
/// The class stores three key pieces of information:
/// - When the intervention started (as an index in a sequence or time series)
/// - How long the intervention lasted
/// - How strong the effect was (positive or negative)
/// 
/// This information is useful for analyzing cause-and-effect relationships in data
/// and understanding how specific actions impact outcomes over time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class InterventionEffect<T>
{
    /// <summary>
    /// Gets or sets the starting index of the intervention in the time series or sequence.
    /// </summary>
    /// <value>An integer representing the position where the intervention begins.</value>
    /// <remarks>
    /// <para>
    /// This property represents the point in the sequence or time series where the intervention begins. The index is 
    /// typically zero-based, meaning that an index of 0 refers to the first element in the sequence. In time series 
    /// analysis, this might correspond to a specific date or time point when a treatment or change was introduced. 
    /// The starting index is crucial for determining when the intervention's effects begin to manifest in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you when the intervention started.
    /// 
    /// The start index:
    /// - Indicates the position in your data where the intervention began
    /// - Is typically zero-based (0 means the first element, 1 means the second, etc.)
    /// - Helps locate the intervention in your time series or sequence
    /// 
    /// For example, if you have daily sales data and implemented a new marketing strategy
    /// on day 30, the StartIndex would be 29 (since counting starts at 0).
    /// 
    /// This value is important for:
    /// - Identifying when the effects of the intervention should begin
    /// - Comparing data before and after the intervention
    /// - Visualizing the intervention's timing on charts or graphs
    /// </para>
    /// </remarks>
    public int StartIndex { get; set; }

    /// <summary>
    /// Gets or sets the duration of the intervention in time units or sequence steps.
    /// </summary>
    /// <value>An integer representing how long the intervention lasted.</value>
    /// <remarks>
    /// <para>
    /// This property represents the length of time or number of sequence steps that the intervention lasted. A duration 
    /// of 1 indicates that the intervention affected only a single time point or sequence element, while larger values 
    /// indicate interventions that persisted over multiple time points. The duration helps define the window during which 
    /// the intervention's effects should be considered active, which is important for properly attributing changes in the 
    /// data to the intervention.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how long the intervention lasted.
    /// 
    /// The duration:
    /// - Measures how many time periods or data points the intervention affected
    /// - Is expressed in the same units as your data sequence (days, weeks, etc.)
    /// - Defines the window during which the intervention was active
    /// 
    /// For example, if a marketing campaign ran for 14 days, the Duration would be 14.
    /// 
    /// This value is important for:
    /// - Determining when the intervention ended
    /// - Calculating the total impact over the entire intervention period
    /// - Distinguishing between short-term and long-term interventions
    /// </para>
    /// </remarks>
    public int Duration { get; set; }

    /// <summary>
    /// Gets or sets the magnitude of the intervention's effect on the outcome variable.
    /// </summary>
    /// <value>A double representing the strength and direction of the effect.</value>
    /// <remarks>
    /// <para>
    /// This property represents the magnitude of the intervention's effect on the outcome variable. Positive values indicate 
    /// that the intervention increased the outcome, while negative values indicate that it decreased the outcome. The absolute 
    /// value represents the strength of the effect. This value might be estimated through various statistical methods such as 
    /// interrupted time series analysis, difference-in-differences, or causal impact analysis. The effect size is crucial for 
    /// understanding the practical significance of the intervention.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how strong the intervention's impact was.
    /// 
    /// The effect:
    /// - Measures the size and direction of the intervention's impact
    /// - Positive values mean the intervention increased the outcome
    /// - Negative values mean the intervention decreased the outcome
    /// - Larger absolute values indicate stronger effects
    /// 
    /// For example, an Effect of 12.5 might mean that a marketing campaign
    /// increased daily sales by an average of 12.5 units.
    /// 
    /// This value is important for:
    /// - Evaluating whether the intervention was beneficial or harmful
    /// - Comparing the relative impact of different interventions
    /// - Determining if the effect was practically significant
    /// </para>
    /// </remarks>
    public double Effect { get; set; }
}
