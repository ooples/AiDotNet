using System.Collections.Generic;

namespace AiDotNet.AutoML.SearchSpace
{
    /// <summary>
    /// Defines the ResNet-based search space for neural architecture search.
    /// Includes residual blocks, bottleneck blocks, and various skip connections.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class ResNetSearchSpace<T> : SearchSpaceBase<T>
    {
        public ResNetSearchSpace()
        {
            Operations = new List<string>
            {
                "identity",
                "conv1x1",
                "conv3x3",
                "conv5x5",
                "residual_block_basic",      // Basic residual block (2 conv layers)
                "residual_block_bottleneck", // Bottleneck block (1x1, 3x3, 1x1)
                "maxpool3x3",
                "avgpool3x3",
                "grouped_conv3x3"  // ResNeXt-style grouped convolutions
            };

            MaxNodes = 16;
            InputChannels = 3;  // RGB images
            OutputChannels = 1000;  // ImageNet classes

            // ResNet-specific parameters
            BottleneckRatio = 4;  // Channel reduction in bottleneck blocks
            GroupCount = 32;  // For ResNeXt grouped convolutions
            BlockDepths = new List<int> { 2, 3, 4, 6, 8 };  // Possible block repetitions
        }

        /// <summary>
        /// Bottleneck ratio for channel reduction
        /// </summary>
        public int BottleneckRatio { get; set; }

        /// <summary>
        /// Number of groups for grouped convolutions (ResNeXt)
        /// </summary>
        public int GroupCount { get; set; }

        /// <summary>
        /// Possible depths for residual blocks
        /// </summary>
        public List<int> BlockDepths { get; set; }
    }
}
