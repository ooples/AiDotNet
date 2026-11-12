namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Which candidate AttentiveNAS trains from the pool drawn under a FLOPs target (Wang et al. 2021,
    /// "AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling", Sec. 3.2).
    /// </summary>
    public enum AttentiveSamplingStrategy
    {
        /// <summary>Train the candidate with the highest predicted performance (the Pareto-best).</summary>
        BestUp,

        /// <summary>Train the candidate with the lowest predicted performance (the Pareto-worst).</summary>
        WorstUp
    }
}
