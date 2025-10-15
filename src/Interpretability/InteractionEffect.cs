namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents an interaction effect between two features
    /// </summary>
    public class InteractionEffect
    {
        /// <summary>
        /// Index of the first feature
        /// </summary>
        public int Feature1Index { get; set; }

        /// <summary>
        /// Index of the second feature
        /// </summary>
        public int Feature2Index { get; set; }

        /// <summary>
        /// Strength of the interaction (-1 to 1)
        /// </summary>
        public double InteractionStrength { get; set; }

        /// <summary>
        /// Statistical significance (p-value) of the interaction
        /// </summary>
        public double PValue { get; set; }

        /// <summary>
        /// Name of the first feature (if available)
        /// </summary>
        public string Feature1Name { get; set; } = string.Empty;

        /// <summary>
        /// Name of the second feature (if available)
        /// </summary>
        public string Feature2Name { get; set; } = string.Empty;

        /// <summary>
        /// Description of the interaction effect
        /// </summary>
        public string Description { get; set; } = string.Empty;
    }
}