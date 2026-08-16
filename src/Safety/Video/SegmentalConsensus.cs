namespace AiDotNet.Safety.Video;

/// <summary>
/// The segmental consensus function <c>G</c> of a Temporal Segment Network, which aggregates
/// per-snippet class scores into one video-level score.
/// </summary>
/// <remarks>
/// <para>
/// Wang et al., "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition"
/// (ECCV 2016, arXiv:1608.00859) evaluate three forms of <c>G</c>. On UCF101 split 1 (Table 3),
/// spatial / temporal / two-stream accuracy:
/// </para>
/// <list type="bullet">
/// <item><description>Max — 85.0% / 86.0% / 91.6%</description></item>
/// <item><description>Average — 85.7% / 87.9% / <b>93.5%</b></description></item>
/// <item><description>Weighted average — 86.2% / 87.7% / 92.4%</description></item>
/// </list>
/// <para>
/// The paper concludes "average pooling function achieves the best performance", which is why
/// <see cref="Average"/> is the default here.
/// </para>
/// <para>
/// <b>For Beginners:</b> The video is chopped into a few equal chunks and one frame is examined
/// from each. This setting decides how those separate opinions are combined into a single verdict
/// for the whole video — average them, take the most extreme, or weight some chunks more heavily.
/// </para>
/// </remarks>
public enum SegmentalConsensus
{
    /// <summary>
    /// Even average across snippets — the paper's best-performing consensus and the default.
    /// </summary>
    Average = 0,

    /// <summary>
    /// Maximum across snippets. Most sensitive to a single strong snippet, so it detects content
    /// confined to one moment but is correspondingly easier to trip on a single noisy frame.
    /// </summary>
    Max = 1,

    /// <summary>
    /// Linearly increasing weights across snippets, normalized to sum to one, so later segments
    /// count for more. The paper's third variant.
    /// </summary>
    WeightedAverage = 2,
}
