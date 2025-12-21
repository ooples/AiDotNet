namespace AiDotNet.Models;

/// <summary>
/// Represents information about an intervention in a time series or sequential data, specifying when it started
/// and how long it lasted.
/// </summary>
/// <remarks>
/// <para>
/// This class provides a simple representation of an intervention's timing and duration in a time series or sequential 
/// dataset. An intervention is a deliberate change, treatment, or event that occurs at a specific point in time and may 
/// continue for a certain duration. This class captures only the timing aspects of the intervention without including 
/// information about its effect or magnitude. It is useful for defining when interventions occurred in causal analysis, 
/// time series experiments, or when modeling the impact of specific events.
/// </para>
/// <para><b>For Beginners:</b> This class describes when a change or treatment was applied and how long it lasted.
/// 
/// For example, you might use this to specify:
/// - When a marketing campaign started and how long it ran
/// - When a policy change was implemented and whether it's still in effect
/// - When a medical treatment began and its duration
/// 
/// Unlike the InterventionEffect class, this class only stores information about
/// timing and duration, not about the strength or direction of the effect.
/// This makes it useful for planning analyses or defining intervention periods
/// before measuring their impacts.
/// </para>
/// </remarks>
public class InterventionInfo
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
    /// The starting index is crucial for determining when to begin considering the intervention's effects in the data.
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
    /// <value>An integer representing how long the intervention lasted, with 0 indicating a permanent intervention.</value>
    /// <remarks>
    /// <para>
    /// This property represents the length of time or number of sequence steps that the intervention lasted. A duration 
    /// of 1 indicates that the intervention affected only a single time point or sequence element, while larger values 
    /// indicate interventions that persisted over multiple time points. A special value of 0 is used to indicate that 
    /// the intervention is permanent, meaning it continues indefinitely from the start index onward. This distinction is 
    /// important for modeling interventions that represent one-time events versus those that represent permanent changes 
    /// to the system.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how long the intervention lasted.
    /// 
    /// The duration:
    /// - Measures how many time periods or data points the intervention affected
    /// - Is expressed in the same units as your data sequence (days, weeks, etc.)
    /// - A value of 0 has special meaning: it indicates a permanent intervention
    /// 
    /// For example:
    /// - Duration = 14: A marketing campaign that ran for exactly 14 days
    /// - Duration = 0: A policy change that, once implemented, remains in effect indefinitely
    /// 
    /// This value is important for:
    /// - Determining when the intervention ended (if it did)
    /// - Distinguishing between temporary and permanent interventions
    /// - Properly modeling the intervention's effects over time
    /// </para>
    /// </remarks>
    public int Duration { get; set; }
}
