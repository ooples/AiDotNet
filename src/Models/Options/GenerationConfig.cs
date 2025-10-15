using System;
using System.Collections.Generic;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Generation-specific configuration
    /// </summary>
    public class GenerationConfig
    {
        /// <summary>
        /// Default temperature for generation
        /// </summary>
        public double DefaultTemperature { get; set; } = 1.0;

        /// <summary>
        /// Default top-p value
        /// </summary>
        public double DefaultTopP { get; set; } = 1.0;

        /// <summary>
        /// Default top-k value
        /// </summary>
        public int? DefaultTopK { get; set; }

        /// <summary>
        /// Repetition penalty
        /// </summary>
        public double RepetitionPenalty { get; set; } = 1.0;

        /// <summary>
        /// Length penalty for beam search
        /// </summary>
        public double LengthPenalty { get; set; } = 1.0;

        /// <summary>
        /// Number of beams for beam search
        /// </summary>
        public int NumBeams { get; set; } = 1;

        /// <summary>
        /// Early stopping for beam search
        /// </summary>
        public bool EarlyStopping { get; set; } = false;

        /// <summary>
        /// Bad words to avoid in generation
        /// </summary>
        public List<string> BadWords { get; set; } = new List<string>();

        /// <summary>
        /// Forced decoder IDs
        /// </summary>
        public Dictionary<int, int> ForcedDecoderIds { get; set; } = new Dictionary<int, int>();

        /// <summary>
        /// Validates the generation configuration
        /// </summary>
        public void Validate()
        {
            if (DefaultTemperature <= 0)
            {
                throw new InvalidOperationException("DefaultTemperature must be greater than 0");
            }

            if (DefaultTopP <= 0 || DefaultTopP > 1)
            {
                throw new InvalidOperationException("DefaultTopP must be between 0 and 1");
            }

            if (DefaultTopK.HasValue && DefaultTopK.Value <= 0)
            {
                throw new InvalidOperationException("DefaultTopK must be greater than 0");
            }

            if (RepetitionPenalty < 0)
            {
                throw new InvalidOperationException("RepetitionPenalty must be non-negative");
            }

            if (NumBeams <= 0)
            {
                throw new InvalidOperationException("NumBeams must be greater than 0");
            }
        }
    }
}