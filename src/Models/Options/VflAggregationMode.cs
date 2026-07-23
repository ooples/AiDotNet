namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how embeddings from multiple parties are combined in vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In vertical FL, each party computes a local embedding (a compressed
/// representation of its features). These embeddings must be combined before the top model can
/// make a prediction. This enum controls how that combination happens.</para>
/// </remarks>
public enum VflAggregationMode
{
    /// <summary>
    /// Concatenate embeddings side by side. If Party A produces a 32-dim embedding and Party B
    /// produces a 32-dim embedding, the result is a 64-dim vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like merging two spreadsheet rows by placing columns side by side.
    /// This preserves all information but increases the input dimension for the top model.</para>
    /// </remarks>
    Concatenation,

    /// <summary>
    /// Element-wise sum of embeddings. All parties must produce the same embedding dimension.
    /// The result has the same dimension as each individual embedding.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like adding two columns of numbers together.
    /// This keeps the dimension small but may lose information through summation.</para>
    /// </remarks>
    Sum,

    /// <summary>
    /// Attention-weighted combination of embeddings. Learns which party's features are most
    /// important for each prediction.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like having a smart weighting system that decides how much to trust
    /// each party's contribution for each sample. Some samples might rely more on Party A's features,
    /// while others rely more on Party B's features.</para>
    /// </remarks>
    Attention,

    /// <summary>
    /// Gating mechanism that learns a sigmoid gate to blend embeddings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like a dimmer switch that controls how much of each party's
    /// contribution to use. The gate is learned during training.</para>
    /// </remarks>
    Gating
}
