namespace AiDotNet.Models
{
    /// <summary>
    /// Represents the interaction effect between two features
    /// </summary>
    public class InteractionEffect
    {
        /// <summary>
        /// Gets or sets the first feature index
        /// </summary>
        public int Feature1Index { get; set; }
        
        /// <summary>
        /// Gets or sets the second feature index
        /// </summary>
        public int Feature2Index { get; set; }
        
        /// <summary>
        /// Gets or sets the interaction strength
        /// </summary>
        public double InteractionStrength { get; set; }
        
        /// <summary>
        /// Gets or sets the p-value for the interaction
        /// </summary>
        public double PValue { get; set; }
    }
}