namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for SMOTE-NC (Synthetic Minority Over-sampling Technique for
/// Nominal and Continuous features), a k-NN based oversampling method that generates
/// synthetic minority samples by interpolating between existing ones.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// SMOTE-NC extends the original SMOTE algorithm to handle mixed-type data:
/// - <b>Continuous features</b>: Linear interpolation between a sample and its k-nearest neighbor
/// - <b>Categorical features</b>: Majority vote among the k-nearest neighbors
/// - <b>Distance metric</b>: Euclidean for continuous + Value Difference Metric for categoricals
/// </para>
/// <para>
/// <b>For Beginners:</b> SMOTE-NC helps when your data is imbalanced
/// (e.g., 99% normal transactions, 1% fraud).
///
/// How it works:
/// 1. Find each minority sample's k nearest neighbors (similar samples)
/// 2. Pick one neighbor randomly
/// 3. Create a new sample "in between" the original and the neighbor
///    - For numbers: pick a random point on the line between them
///    - For categories: use the most common category among the neighbors
///
/// This produces new minority samples that are realistic because they're
/// based on actual data relationships, not just random copies.
///
/// Example:
/// <code>
/// var options = new SMOTENCOptions&lt;double&gt;
/// {
///     K = 5,
///     LabelColumnIndex = 4,
///     MinorityClassValue = 1
/// };
/// var smote = new SMOTENCGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)
/// </para>
/// </remarks>
public class SMOTENCOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of nearest neighbors to consider.
    /// </summary>
    /// <value>K value, defaulting to 5.</value>
    public int K { get; set; } = 5;

    /// <summary>
    /// Gets or sets the index of the label column for identifying minority class.
    /// </summary>
    /// <value>Label column index, defaulting to -1 (last column).</value>
    public int LabelColumnIndex { get; set; } = -1;

    /// <summary>
    /// Gets or sets the minority class value to oversample.
    /// </summary>
    /// <value>Minority class value, defaulting to 1.</value>
    public int MinorityClassValue { get; set; } = 1;
}
