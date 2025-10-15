using AiDotNet.Models.Options;

namespace AiDotNet.Ensemble
{
    /// <summary>
    /// Configuration options specific to voting ensembles.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class VotingEnsembleOptions<T> : EnsembleOptions<T>
    {
        /// <summary>
        /// Gets or sets the type of voting to use.
        /// </summary>
        public VotingType VotingType { get; set; } = VotingType.Weighted;
        
        /// <summary>
        /// Gets or sets whether to check for consensus among models.
        /// </summary>
        /// <remarks>
        /// When enabled, the ensemble will verify that a sufficient proportion of models
        /// agree on the prediction before returning it.
        /// </remarks>
        public bool RequireConsensus { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the minimum consensus threshold (0-1).
        /// </summary>
        /// <remarks>
        /// Only used when RequireConsensus is true. Specifies the minimum proportion
        /// of models that must agree on a prediction.
        /// </remarks>
        public double ConsensusThreshold { get; set; } = 0.7;
        
        /// <summary>
        /// Gets or sets the power scaling for performance-based weights.
        /// </summary>
        /// <remarks>
        /// Values greater than 1 amplify differences in performance, giving more weight
        /// to better-performing models. Values less than 1 reduce differences, making
        /// the weighting more uniform. A value of 1 uses linear weighting.
        /// </remarks>
        public double PerformancePowerScaling { get; set; } = 1.0;
    }
}