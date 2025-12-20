namespace AiDotNet.Enums;

/// <summary>
/// Defines different weight functions used in robust statistical methods and machine learning algorithms.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Weight functions are special mathematical formulas that help AI models handle unusual 
/// or extreme data points (outliers).
/// 
/// In regular statistics, outliers can significantly throw off your results. For example, if you're 
/// calculating the average income in a neighborhood and one billionaire lives there, the average 
/// would be misleadingly high.
/// 
/// Weight functions solve this problem by automatically giving less importance (lower weight) to data 
/// points that seem unusual or extreme. This makes your AI models more robust and reliable when 
/// dealing with real-world data that might contain errors or unusual values.
/// 
/// Each weight function has different characteristics that make it suitable for different situations:
/// - Some are gentler with outliers (like Huber)
/// - Others are more aggressive in downweighting extreme values (like Bisquare)
/// 
/// Choosing the right weight function depends on how much you expect your data to contain outliers 
/// and how you want to handle them.
/// </remarks>
public enum WeightFunction
{
    /// <summary>
    /// The Huber weight function, which provides a balance between efficiency and robustness.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The Huber function is like a "middle ground" approach to handling outliers.
    /// 
    /// How it works:
    /// - For data points that are close to the expected pattern, it treats them normally (with full weight)
    /// - For data points that are far from the expected pattern, it reduces their importance (weight) 
    ///   gradually as they get more extreme
    /// 
    /// Think of Huber as a "diplomatic" approach - it doesn't completely ignore outliers, but it doesn't 
    /// let them dominate either. It's like saying "I'll listen to unusual opinions, but I won't let them 
    /// completely change my mind."
    /// 
    /// When to use Huber:
    /// - When you expect some outliers in your data but don't want to be too aggressive in removing their influence
    /// - When you want a good balance between accuracy and robustness
    /// - As a safe default choice when you're not sure which weight function to use
    /// 
    /// The Huber function has a tuning parameter that controls how quickly it starts to downweight outliers.
    /// </remarks>
    Huber,

    /// <summary>
    /// The Bisquare (also known as Tukey's biweight) weight function, which completely downweights extreme outliers.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The Bisquare function is more aggressive in handling outliers than Huber.
    /// 
    /// How it works:
    /// - For data points close to the expected pattern, it treats them normally (with full weight)
    /// - As data points get farther from the expected pattern, it reduces their weight more quickly than Huber
    /// - Beyond a certain threshold, it gives zero weight to extreme outliers (completely ignores them)
    /// 
    /// Think of Bisquare as a "strict" approach - it's willing to completely ignore data points that 
    /// seem too unusual. It's like saying "If your opinion is too extreme, I'm not going to consider it at all."
    /// 
    /// When to use Bisquare:
    /// - When you know your data contains significant outliers that should have no influence
    /// - When you want to ensure extreme values don't affect your model
    /// - In situations where you need high robustness against outliers
    /// 
    /// The Bisquare function has a tuning parameter that determines the threshold beyond which 
    /// outliers are completely ignored.
    /// </remarks>
    Bisquare,

    /// <summary>
    /// The Andrews weight function, which uses a sine wave to handle outliers.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The Andrews function uses a wave-like pattern (specifically, a sine wave) 
    /// to determine how much weight to give to different data points.
    /// 
    /// How it works:
    /// - For data points close to the expected pattern, it gives them nearly full weight
    /// - As data points get farther from the pattern, their weight oscillates (goes up and down) 
    ///   but generally decreases
    /// - For very extreme outliers, it gives them zero weight
    /// 
    /// Think of Andrews as a "nuanced" approach - it doesn't just gradually reduce weight like Huber, 
    /// or cut off outliers like Bisquare. Instead, it has a more complex pattern of weighting that can 
    /// sometimes preserve more information from the data.
    /// 
    /// When to use Andrews:
    /// - In specialized statistical applications where you need the specific mathematical properties 
    ///   of the sine function
    /// - When dealing with periodic data or data with wave-like patterns
    /// - When recommended by a statistical expert for your specific application
    /// 
    /// The Andrews function is less commonly used than Huber or Bisquare but has useful properties 
    /// in certain specialized applications.
    /// </remarks>
    Andrews
}
