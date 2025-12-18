namespace AiDotNet.Data.Structures;

/// <summary>
/// Enumeration of pooling strategies for graph neural networks and attention mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// Pooling strategies reduce variable-sized inputs to fixed-size representations
/// by aggregating information across dimensions. They are essential in meta-learning
/// for handling tasks with different numbers of examples or features.
/// </para>
/// <para><b>For Beginners:</b> Pooling is like summarizing information:</para>
///
/// Think of reading a book:
/// - <b>Average pooling:</b> Take the average sentence meaning
/// - <b>Max pooling:</b> Remember only the most important point
/// - <b>Attention pooling:</b> Focus on relevant parts based on context
/// - <b>Sum pooling:</b> Add up all the meanings together
///
/// Each method captures different aspects of the information.
/// </remarks>
public enum PoolingStrategy
{
    /// <summary>
    /// No pooling (identity operation).
    /// </summary>
    /// <remarks>
    /// Returns the input as-is without any aggregation.
    /// Used when pooling is not needed or input is already fixed-size.
    /// </remarks>
    None,

    /// <summary>
    /// Average pooling: computes the mean of all values.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Takes all values in the pooling region
    /// - Computes their arithmetic mean
    /// - Smooths out variations
    ///
    /// <para><b>When to use:</b></para>
    /// - When all information is equally important
    /// - When you want a stable representation
    /// - When dealing with normally distributed data
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: 2.5 (average)
    /// </remarks>
    Average,

    /// <summary>
    /// Maximum pooling: selects the maximum value.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Scans all values in the pooling region
    /// - Returns the maximum value
    /// - Preserves the strongest features
    ///
    /// <para><b>When to use:</b></para>
    /// - When you want to detect presence of features
    /// - When maximum activation is meaningful
    /// - In convolutional networks for spatial invariance
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: 4 (maximum)
    /// </remarks>
    Max,

    /// <summary>
    /// Minimum pooling: selects the minimum value.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Scans all values in the pooling region
    /// - Returns the minimum value
    /// - Preserves the weakest features
    ///
    /// <para><b>When to use:</b></para>
    /// - When minimum values carry important information
    /// - For certain distance or error metrics
    /// - Less common than max pooling
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: 1 (minimum)
    /// </remarks>
    Min,

    /// <summary>
    /// Sum pooling: computes the sum of all values.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Adds all values in the pooling region
    /// - Preserves total magnitude
    /// - Sensitive to number of elements
    ///
    /// <para><b>When to use:</b></para>
    /// - When total quantity matters
    /// - For count-like features
    /// - When preserving energy or mass
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: 10 (sum)
    /// </remarks>
    Sum,

    /// <summary>
    /// Attention pooling: weighted sum using attention weights.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// 1. Computes attention weights for each element
    /// 2. Weights sum to 1 (softmax activation)
    /// 3. Returns weighted sum of elements
    /// 4. Elements can be selectively emphasized
    ///
    /// <para><b>When to use:</b></para>
    /// - When some elements are more important than others
    /// - When you need to learn what to focus on
    /// - For variable-length sequences
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] with weights [0.1, 0.2, 0.3, 0.4]
    /// Output: 1×0.1 + 2×0.2 + 3×0.3 + 4×0.4 = 3.0
    /// </remarks>
    Attention,

    /// <summary>
    /// L2 norm pooling: computes the Euclidean norm.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Takes square root of sum of squares
    /// - Measures magnitude of the vector
    /// - Always positive
    ///
    /// <para><b>When to use:</b></para>
    /// - When you need vector magnitude
    /// - For feature normalization
    /// - In similarity calculations
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: sqrt(1²+2²+3²+4²) = sqrt(30) ≈ 5.48
    /// </remarks>
    L2Norm,

    /// <summary>
    /// L1 norm pooling: computes the Manhattan norm (sum of absolute values).
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Sum of absolute values
    /// - Less sensitive to outliers than L2
    /// - Sparse feature preservation
    ///
    /// <para><b>When to use:</b></para>
    /// - With sparse features
    /// - When robustness to outliers matters
    /// - For taxicab geometry distances
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: |1|+|2|+|3|+|4| = 10
    /// </remarks>
    L1Norm,

    /// <summary>
    /// Standard deviation pooling: computes the standard deviation.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Measures spread of values
    /// - Complement to average pooling
    /// - Captures variability
    ///
    /// <para><b>When to use:</b></para>
    /// - When variability is important
    /// - For uncertainty estimation
    /// - To complement mean features
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: 1.29 (standard deviation)
    /// </remarks>
    StandardDeviation,

    /// <summary>
    /// Variance pooling: computes the variance (square of standard deviation).
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Measures squared deviation from mean
    /// - Emphasizes large deviations
    /// - Differentiable optimization friendly
    ///
    /// <para><b>When to use:</b></para>
    /// - When optimizing gradients
    /// - For feature selection
    /// - In neural network layers
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: 1.67 (variance)
    /// </remarks>
    Variance,

    /// <summary>
    /// Root Mean Square (RMS) pooling.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Square root of mean of squares
    /// - Between average and max in effect
    /// - Preserves magnitude information
    ///
    /// <para><b>When to use:</b></para>
    /// - For signal processing
    /// - Audio feature extraction
    /// - When RMS value is meaningful
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] → Output: sqrt((1²+2²+3²+4²)/4) = 2.74
    /// </remarks>
    RootMeanSquare,

    /// <summary>
    /// Learned pooling with trainable parameters.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// - Uses trainable weights for pooling
    /// - Weights learned during training
    /// - Can implement complex pooling functions
    ///
    /// <para><b>When to use:</b></para>
    /// - When data-specific pooling needed
    /// - In deep neural networks
    /// - For maximum flexibility
    ///
    /// <para><b>Example:</b></para>
    /// Input: [1, 2, 3, 4] with learned weights [w1, w2, w3, w4]
    /// Output: w1×1 + w2×2 + w3×3 + w4×4 (weights learned)
    /// </remarks>
    Learned
}