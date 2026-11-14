namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Search settings of <see cref="PCDARTS{T}"/>. The defaults are those of Xu et al. 2020, "PC-DARTS: Partial
    /// Channel Connections for Memory-Efficient Architecture Search" (Sec. 4.2) and its reference implementation.
    /// </summary>
    /// <remarks>
    /// <para>The optimizer settings reuse <see cref="DartsOptimizerOptions"/>, with PC-DARTS' own defaults: the
    /// architecture parameters (alpha and the edge-normalization parameters beta) are trained with Adam at learning
    /// rate 6e-4, betas (0.5, 0.999) and weight decay 1e-3; the network weights with SGD, momentum 0.9, weight
    /// decay 3e-4, gradient-norm clipping at 5 and a learning rate cosine-annealed from 0.1 down to 0.</para>
    /// <para><b>For Beginners:</b> PC-DARTS alternates two steps: one trains the network's weights on the
    /// training data, the other trains the architecture choices on the validation data. During the first
    /// <see cref="WarmupEpochs"/> epochs only the weights are trained, so the architecture choices are not
    /// judged by an untrained network. The defaults reproduce the paper's CIFAR-10 search; a smaller dataset or
    /// search space may need fewer epochs or a smaller batch.</para>
    /// </remarks>
    public class PCDARTSOptions : DartsOptimizerOptions
    {
        /// <summary>
        /// Creates the options with PC-DARTS' defaults.
        /// </summary>
        public PCDARTSOptions()
        {
            ArchitectureLearningRate = 6e-4;
            WeightMaxLearningRate = 0.1;
            WeightMinLearningRate = 0.0;
        }

        /// <summary>Number of search epochs (default: 50).</summary>
        public int Epochs { get; set; } = 50;

        /// <summary>
        /// Number of leading epochs that train only the network weights, leaving alpha and beta unchanged
        /// (default: 15).
        /// </summary>
        public int WarmupEpochs { get; set; } = 15;

        /// <summary>Mini-batch size of both the weight and the architecture steps (default: 256).</summary>
        public int BatchSize { get; set; } = 256;

        /// <summary>
        /// The partial-channel factor K: each edge's operation mixture sees 1/K of the channels, the rest bypass
        /// it, and a channel shuffle with K groups follows (default: 4).
        /// </summary>
        public int PartialChannelK { get; set; } = 4;

        /// <summary>
        /// The number of intermediate nodes of the searched cell (4 in Xu et al. 2020, as in DARTS).
        /// </summary>
        public int NumNodes { get; set; } = 4;

        /// <summary>
        /// Whether to weight each edge by softmax(beta), the edge normalization of PC-DARTS (Sec. 3.3).
        /// </summary>
        public bool UseEdgeNormalization { get; set; } = true;
    }
}
