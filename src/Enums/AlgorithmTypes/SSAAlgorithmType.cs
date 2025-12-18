namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Singular Spectrum Analysis (SSA).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Singular Spectrum Analysis (SSA) is a powerful technique used to analyze time series data 
/// by breaking it down into meaningful components. Think of it as taking apart a complex musical piece to 
/// identify the individual instruments playing.
/// 
/// Here's how SSA works in simple terms:
/// 
/// 1. Embedding: First, we take our time series (a sequence of values over time) and create a matrix by sliding 
///    a window of a certain length through the data. Each column of this matrix represents a segment of our 
///    original time series.
/// 
/// 2. Decomposition: Next, we perform a mathematical operation called Singular Value Decomposition (SVD) on this 
///    matrix. This breaks down our matrix into simpler components, each capturing different patterns in the data.
/// 
/// 3. Grouping: We then group these components based on their properties. Some might represent trends, others 
///    seasonal patterns, and some just random noise.
/// 
/// 4. Reconstruction: Finally, we can reconstruct our time series using only the components we're interested in, 
///    effectively filtering out unwanted patterns.
/// 
/// Why is SSA important in AI and machine learning?
/// 
/// 1. Noise Reduction: It can clean up noisy data by separating signal from noise
/// 
/// 2. Trend Extraction: It can identify and isolate long-term trends in data
/// 
/// 3. Seasonality Detection: It can extract seasonal patterns of various frequencies
/// 
/// 4. Feature Engineering: The components extracted can serve as features for machine learning models
/// 
/// 5. Forecasting: By understanding the underlying patterns, we can make better predictions
/// 
/// 6. Anomaly Detection: Unusual patterns that don't fit the main components can be identified as anomalies
/// 
/// This enum specifies which specific algorithm variant to use for SSA, as different methods have different 
/// performance characteristics and may be more suitable for certain types of data or analysis goals.
/// </para>
/// </remarks>
public enum SSAAlgorithmType
{
    /// <summary>
    /// Uses the standard basic implementation of Singular Spectrum Analysis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Basic SSA algorithm follows the classic four-step approach described above: embedding, 
    /// decomposition, grouping, and reconstruction.
    /// 
    /// Imagine sorting through a box of mixed Lego pieces. The Basic SSA approach would be like:
    /// 1. Organizing the pieces into rows based on their size (embedding)
    /// 2. Identifying common patterns or shapes among the pieces (decomposition)
    /// 3. Grouping similar pieces together (grouping)
    /// 4. Using selected groups to build something new (reconstruction)
    /// 
    /// The Basic approach:
    /// 
    /// 1. Is the most straightforward implementation of SSA
    /// 
    /// 2. Works well for most standard time series analysis tasks
    /// 
    /// 3. Provides a good balance between computational efficiency and accuracy
    /// 
    /// 4. Is easier to understand and interpret
    /// 
    /// 5. Serves as a foundation for more specialized variants
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You're new to SSA and want to start with the standard approach
    /// 
    /// 2. Your time series is well-behaved (not too noisy or irregular)
    /// 
    /// 3. You want results that are easy to interpret
    /// 
    /// 4. You're exploring the data to get a general understanding of its components
    /// 
    /// In machine learning applications, the Basic SSA algorithm provides a solid foundation for feature extraction 
    /// from time series data, helping to identify patterns that can improve the performance of predictive models.
    /// </para>
    /// </remarks>
    Basic,

    /// <summary>
    /// Uses a sequential implementation of SSA that processes data in a step-by-step manner.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Sequential SSA algorithm processes the time series data in a step-by-step manner, 
    /// updating the decomposition as new data points become available.
    /// 
    /// Think of it like reading a book one page at a time and continuously updating your understanding of the 
    /// story, rather than reading the whole book at once:
    /// 
    /// 1. You start with a small portion of the data
    /// 2. Perform the SSA steps on this portion
    /// 3. When new data arrives, you update your analysis without starting over
    /// 4. This continues as more data becomes available
    /// 
    /// The Sequential approach:
    /// 
    /// 1. Is more efficient for processing streaming or real-time data
    /// 
    /// 2. Requires less memory since it doesn't need to store the entire dataset at once
    /// 
    /// 3. Can adapt to changes in the data patterns over time
    /// 
    /// 4. Is suitable for online learning scenarios
    /// 
    /// 5. May sacrifice some accuracy compared to processing all data at once
    /// 
    /// This method is particularly valuable when:
    /// 
    /// 1. You're working with streaming data that arrives continuously
    /// 
    /// 2. You have limited memory resources
    /// 
    /// 3. You need to update your analysis in real-time
    /// 
    /// 4. The patterns in your data might evolve over time
    /// 
    /// In machine learning applications, Sequential SSA enables real-time feature extraction and pattern recognition, 
    /// making it useful for applications like predictive maintenance, anomaly detection in IoT sensors, or financial 
    /// market analysis where immediate insights are valuable.
    /// </para>
    /// </remarks>
    Sequential,

    /// <summary>
    /// Uses a Toeplitz matrix approach for SSA, which exploits the structure of time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Toeplitz SSA algorithm uses a special type of matrix called a Toeplitz matrix during 
    /// the embedding step, which has a unique pattern where each diagonal contains the same value.
    /// 
    /// Imagine a staircase where each step has the same pattern repeated. A Toeplitz matrix has a similar 
    /// repeating pattern, which makes calculations more efficient:
    /// 
    /// 1. Instead of creating the full trajectory matrix, it uses the special Toeplitz structure
    /// 2. This structure allows for faster computations and less memory usage
    /// 3. The mathematical properties of Toeplitz matrices enable specialized, efficient algorithms
    /// 
    /// The Toeplitz approach:
    /// 
    /// 1. Is computationally more efficient than the basic approach, especially for large datasets
    /// 
    /// 2. Reduces memory requirements by exploiting the matrix structure
    /// 
    /// 3. Produces results that are mathematically equivalent to the basic approach
    /// 
    /// 4. Works particularly well for stationary time series (those with consistent statistical properties)
    /// 
    /// 5. Can handle longer time series more effectively
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You're working with very large time series datasets
    /// 
    /// 2. Computational efficiency is important
    /// 
    /// 3. Your data has relatively stable statistical properties
    /// 
    /// 4. You need to process many time series quickly
    /// 
    /// In machine learning applications, the Toeplitz SSA algorithm enables efficient processing of large-scale 
    /// time series data, making it practical to extract features from extensive historical datasets or to apply 
    /// SSA to high-frequency data like audio signals or high-resolution sensor readings.
    /// </para>
    /// </remarks>
    Toeplitz
}
