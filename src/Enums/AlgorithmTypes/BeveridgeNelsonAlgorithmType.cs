namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Beveridge-Nelson decomposition of time series data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Beveridge-Nelson decomposition is a technique used in economics and finance to separate 
/// time series data (like stock prices or GDP over time) into two main components:
/// 
/// 1. Permanent Component (Trend) - The long-lasting changes that persist indefinitely
/// 2. Temporary Component (Cycle) - The short-term fluctuations that eventually fade away
/// 
/// Unlike other decomposition methods that might look at regular patterns (like seasonality), 
/// Beveridge-Nelson focuses on distinguishing between changes that will have lasting effects 
/// versus those that will eventually reverse.
/// 
/// For example, when analyzing a company's stock price:
/// - A permanent component might be fundamental improvements in the company's business model
/// - A temporary component might be short-term market excitement that will eventually subside
/// 
/// This enum lists different algorithmic approaches to performing this type of decomposition.
/// </para>
/// </remarks>
public enum BeveridgeNelsonAlgorithmType
{
    /// <summary>
    /// The standard implementation of the Beveridge-Nelson decomposition algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Standard approach implements the original Beveridge-Nelson method as described
    /// in their 1981 paper. It uses statistical techniques to identify which changes in your data are
    /// likely to be permanent and which are likely to be temporary.
    /// 
    /// This approach is like separating a river into its main channel (the permanent component that keeps
    /// flowing in the same direction) and its eddies and whirlpools (temporary fluctuations that eventually
    /// dissipate).
    /// 
    /// The standard method works well for many basic time series but may not capture all the complexities
    /// in more sophisticated data.
    /// </para>
    /// </remarks>
    Standard,

    /// <summary>
    /// Uses ARIMA (AutoRegressive Integrated Moving Average) models to perform Beveridge-Nelson decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ARIMA stands for AutoRegressive Integrated Moving Average, which is a popular
    /// statistical method for analyzing and forecasting time series data.
    /// 
    /// This approach combines the Beveridge-Nelson decomposition with ARIMA modeling to better capture
    /// the patterns in your data. It's like having a more sophisticated tool that can detect subtle
    /// patterns in your time series.
    /// 
    /// Think of it as using a high-powered microscope instead of a magnifying glass - you can see more
    /// details and make better distinctions between what's permanent and what's temporary in your data.
    /// 
    /// The ARIMA approach is particularly useful when your data has complex patterns that the standard
    /// method might miss.
    /// </para>
    /// </remarks>
    ARIMA,

    /// <summary>
    /// Extends the Beveridge-Nelson decomposition to handle multiple related time series simultaneously.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Multivariate approach allows you to analyze multiple related time series at once,
    /// taking into account how they influence each other.
    /// 
    /// Imagine you're analyzing both unemployment rates and inflation - these economic indicators affect
    /// each other, so analyzing them together can reveal insights that looking at each separately would miss.
    /// 
    /// This is like studying an ecosystem rather than a single species - you get a more complete picture
    /// by seeing how different elements interact.
    /// 
    /// The Multivariate approach is more complex but can be much more powerful when you have multiple
    /// related data series that might share common permanent and temporary components.
    /// </para>
    /// </remarks>
    Multivariate
}
