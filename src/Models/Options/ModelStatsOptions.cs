namespace AiDotNet.Models.Options;

public class ModelStatsOptions
{
    public ConditionNumberMethod ConditionNumberMethod { get; set; } = ConditionNumberMethod.SVD;
    public double MulticollinearityThreshold { get; set; } = 0.8;
    public int MaxVIF { get; set; } = 10;

    /// <summary>
    /// The number of top items to consider when calculating Mean Average Precision (MAP).
    /// </summary>
    public int MapTopK { get; set; } = 10;

    /// <summary>
    /// The number of top items to consider when calculating Normalized Discounted Cumulative Gain (NDCG).
    /// </summary>
    public int NdcgTopK { get; set; } = 10;

    /// <summary>
    /// The maximum lag to use for the ACF calculation. Default is 20.
    /// </summary>
    public int AcfMaxLag { get; set; } = 20;

    /// <summary>
    /// The maximum lag to use for the PACF calculation. Default is 20.
    /// </summary>
    public int PacfMaxLag { get; set; } = 20;
}