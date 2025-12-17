namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Empirical Mode Decomposition (EMD).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Empirical Mode Decomposition (EMD) is a technique used to break down complex data signals 
/// into simpler components called Intrinsic Mode Functions (IMFs).
/// 
/// Imagine you're listening to an orchestra. The music you hear is a complex mixture of sounds from many 
/// instruments playing together. EMD is like having a special ability to hear each instrument separately, 
/// even though they're all playing at once. It helps you understand how each instrument contributes to the 
/// overall music.
/// 
/// In data analysis and AI, EMD helps us:
/// 
/// 1. Analyze non-stationary data (data that changes its statistical properties over time), like stock 
///    market prices, weather patterns, or brain signals
/// 
/// 2. Extract meaningful patterns from noisy data
/// 
/// 3. Identify hidden cycles or trends in complex time series data
/// 
/// 4. Preprocess data for machine learning models to improve their performance
/// 
/// Unlike Fourier transforms (another common technique) which assume data patterns repeat regularly, 
/// EMD adapts to the data itself, making it particularly useful for real-world data that often contains 
/// irregular patterns and sudden changes.
/// 
/// This enum lists different variations of the EMD algorithm, each with specific strengths for different 
/// types of data analysis problems.
/// </para>
/// </remarks>
public enum EMDAlgorithmType
{
    /// <summary>
    /// Uses the standard Empirical Mode Decomposition algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Standard EMD is the original version of the algorithm that decomposes a signal 
    /// into a collection of Intrinsic Mode Functions (IMFs).
    /// 
    /// It works by identifying local extremes (peaks and valleys) in your data, connecting them with smooth 
    /// curves, and then subtracting these curves from the original data. This process is repeated multiple 
    /// times until the signal is fully decomposed.
    /// 
    /// Think of it like peeling an onion - you remove one layer at a time, with each layer representing a 
    /// different oscillation pattern in your data.
    /// 
    /// The Standard EMD works well for many applications but has some limitations with certain types of data, 
    /// particularly when dealing with signals that have similar frequencies appearing at different times 
    /// (known as mode mixing).
    /// </para>
    /// </remarks>
    Standard,

    /// <summary>
    /// Uses the Ensemble Empirical Mode Decomposition (EEMD) algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensemble EMD improves on the standard algorithm by adding small amounts of noise to 
    /// the original signal and performing multiple decompositions.
    /// 
    /// Imagine you're trying to find a path through a foggy forest. If you make the journey just once, you 
    /// might get lost. But if you make the journey many times with slightly different starting points, and 
    /// then average all your paths, you'll likely find the best route. EEMD works similarly by adding random 
    /// noise to the signal multiple times and then averaging the results.
    /// 
    /// This approach helps solve the "mode mixing" problem of standard EMD, where components with similar 
    /// frequencies can get mixed together. By adding noise and averaging multiple decompositions, EEMD can 
    /// better separate these mixed components.
    /// 
    /// EEMD is particularly useful for analyzing complex signals like climate data, biomedical signals, or 
    /// financial time series where different patterns may overlap.
    /// </para>
    /// </remarks>
    Ensemble,

    /// <summary>
    /// Uses the Complete Ensemble Empirical Mode Decomposition (CEEMD) algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Complete Ensemble EMD is an enhanced version of EEMD that adds noise in pairs (positive 
    /// and negative) to ensure that the added noise cancels out completely in the final result.
    /// 
    /// While EEMD adds random noise many times and averages the results, some residual noise can still remain 
    /// in the final decomposition. CEEMD solves this by adding pairs of noise with opposite signs, ensuring 
    /// that the noise cancels out perfectly when averaged.
    /// 
    /// Think of it like taking multiple photographs of a scene with different random camera shakes, but making 
    /// sure that for every photo with a shake to the left, there's another with an equal shake to the right. 
    /// When you average all the photos, the shakes cancel out completely, giving you a perfectly clear image.
    /// 
    /// CEEMD provides more accurate decompositions than EEMD while still solving the mode mixing problem. 
    /// It's particularly valuable when high precision is required, such as in medical signal analysis or 
    /// high-frequency financial data analysis.
    /// </para>
    /// </remarks>
    CompleteEnsemble,

    /// <summary>
    /// Uses the Multivariate Empirical Mode Decomposition (MEMD) algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multivariate EMD extends the EMD concept to handle multiple related signals simultaneously.
    /// 
    /// While standard EMD, EEMD, and CEEMD work on a single data series (like temperature over time), MEMD can 
    /// analyze multiple related data series together (like temperature, humidity, and pressure over time).
    /// 
    /// Imagine you're analyzing a symphony orchestra again, but now instead of just separating instruments, 
    /// you want to identify when all instruments are playing the same melody or theme, even if they're playing 
    /// at different volumes or with slight variations. MEMD helps you find these common patterns across 
    /// multiple channels of information.
    /// 
    /// This is particularly useful in:
    /// 
    /// 1. Brain signal analysis (EEG) where data comes from multiple sensors
    /// 
    /// 2. Environmental studies where multiple variables interact
    /// 
    /// 3. Financial markets where different stocks or indicators may show related patterns
    /// 
    /// 4. Motion capture data where movements in different dimensions are related
    /// 
    /// MEMD preserves the relationships between different channels of data, allowing you to discover patterns 
    /// that might be missed if each channel were analyzed separately.
    /// </para>
    /// </remarks>
    Multivariate
}
