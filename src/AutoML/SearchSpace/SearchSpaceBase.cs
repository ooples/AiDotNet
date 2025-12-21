using System.Collections.Generic;

namespace AiDotNet.AutoML.SearchSpace
{
    /// <summary>
    /// Defines the search space for neural architecture search.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class SearchSpaceBase<T>
    {
        /// <summary>
        /// Available operations for the search.
        /// </summary>
        public List<string> Operations { get; set; } = new List<string>
        {
            "identity",
            "conv3x3",
            "conv5x5",
            "maxpool3x3",
            "avgpool3x3"
        };

        /// <summary>
        /// Maximum number of nodes in the architecture.
        /// </summary>
        public int MaxNodes { get; set; } = 8;

        /// <summary>
        /// Number of input channels.
        /// </summary>
        public int InputChannels { get; set; } = 1;

        /// <summary>
        /// Number of output channels.
        /// </summary>
        public int OutputChannels { get; set; } = 1;
    }
}

