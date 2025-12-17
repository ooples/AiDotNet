using System.Collections.Generic;

namespace AiDotNet.AutoML.SearchSpace
{
    /// <summary>
    /// Defines the MobileNet-based search space for neural architecture search.
    /// Includes inverted residual blocks, depthwise separable convolutions, and squeeze-excitation.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class MobileNetSearchSpace<T> : SearchSpaceBase<T>
    {
        public MobileNetSearchSpace()
        {
            Operations = new List<string>
            {
                "identity",
                "conv1x1",
                "conv3x3",
                "depthwise_conv3x3",
                "inverted_residual_3x3_e3",  // Expansion ratio 3
                "inverted_residual_3x3_e6",  // Expansion ratio 6
                "inverted_residual_5x5_e3",
                "inverted_residual_5x5_e6",
                "se_block"  // Squeeze-and-Excitation
            };

            MaxNodes = 20;  // MobileNet typically has more layers
            InputChannels = 3;  // RGB images
            OutputChannels = 1000;  // ImageNet classes

            // MobileNet-specific parameters
            ExpansionRatios = new List<int> { 3, 6 };
            KernelSizes = new List<int> { 3, 5 };
            DepthMultiplier = 1.0;
            WidthMultiplier = 1.0;
        }

        /// <summary>
        /// Expansion ratios for inverted residual blocks
        /// </summary>
        public List<int> ExpansionRatios { get; set; }

        /// <summary>
        /// Kernel sizes to search over
        /// </summary>
        public List<int> KernelSizes { get; set; }

        /// <summary>
        /// Depth multiplier for scaling network depth
        /// </summary>
        public double DepthMultiplier { get; set; }

        /// <summary>
        /// Width multiplier for scaling channel counts
        /// </summary>
        public double WidthMultiplier { get; set; }
    }
}
