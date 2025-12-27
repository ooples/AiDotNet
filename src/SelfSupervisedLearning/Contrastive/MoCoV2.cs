using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;
using AiDotNet.SelfSupervisedLearning.Infrastructure.ProjectorHeads;

namespace AiDotNet.SelfSupervisedLearning.Contrastive;

/// <summary>
/// MoCo v2: Improved Baselines with Momentum Contrastive Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MoCo v2 improves on MoCo v1 by incorporating ideas from SimCLR:
/// an MLP projection head and stronger augmentations. This combines MoCo's memory efficiency
/// with SimCLR's representation quality improvements.</para>
///
/// <para><b>Key improvements over MoCo v1:</b></para>
/// <list type="bullet">
/// <item><b>MLP projection head:</b> 2-layer MLP instead of linear (from SimCLR)</item>
/// <item><b>Stronger augmentations:</b> Added blur and more color distortion (from SimCLR)</item>
/// <item><b>Cosine learning rate schedule:</b> Better training dynamics</item>
/// </list>
///
/// <para><b>Result:</b> MoCo v2 matches SimCLR performance with much smaller batch sizes
/// (256 vs 4096-8192).</para>
///
/// <para><b>Reference:</b> Chen et al., "Improved Baselines with Momentum Contrastive Learning"
/// (arXiv 2020)</para>
/// </remarks>
public class MoCoV2<T> : MoCo<T>
{
    /// <inheritdoc />
    public override string Name => "MoCo v2";

    /// <summary>
    /// Initializes a new instance of the MoCoV2 class.
    /// </summary>
    /// <param name="encoder">The online encoder network.</param>
    /// <param name="momentumEncoder">The momentum encoder.</param>
    /// <param name="projector">The MLP projection head for online encoder.</param>
    /// <param name="momentumProjector">The MLP projection head for momentum encoder.</param>
    /// <param name="embeddingDim">Dimension of the output embeddings.</param>
    /// <param name="config">Optional SSL configuration.</param>
    public MoCoV2(
        INeuralNetwork<T> encoder,
        IMomentumEncoder<T> momentumEncoder,
        IProjectorHead<T> projector,
        IProjectorHead<T> momentumProjector,
        int embeddingDim = 128,
        SSLConfig? config = null)
        : base(encoder, momentumEncoder, projector, momentumProjector, embeddingDim,
              config ?? CreateMoCoV2Config())
    {
    }

    private static SSLConfig CreateMoCoV2Config()
    {
        var config = SSLConfig.ForMoCo();
        config.MoCo ??= new MoCoConfig();
        config.MoCo.UseMLPProjector = true;
        config.UseCosineDecay = true;
        return config;
    }

    /// <summary>
    /// Creates a MoCo v2 instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder.</param>
    /// <param name="createEncoderCopy">Function to create a copy of the encoder for momentum.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 128).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 2048).</param>
    /// <param name="queueSize">Size of the memory queue (default: 65536).</param>
    /// <returns>A configured MoCo v2 instance.</returns>
    public static MoCoV2<T> Create(
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createEncoderCopy,
        int encoderOutputDim,
        int projectionDim = 128,
        int hiddenDim = 2048,
        int queueSize = 65536)
    {
        // Create projectors
        var projector = new MLPProjector<T>(encoderOutputDim, hiddenDim, projectionDim);
        var momentumProjector = new MLPProjector<T>(encoderOutputDim, hiddenDim, projectionDim);

        // Copy projector parameters
        momentumProjector.SetParameters(projector.GetParameters());

        // Create momentum encoder
        var encoderCopy = createEncoderCopy(encoder);
        var momentumEncoder = new Infrastructure.MomentumEncoder<T>(encoderCopy, 0.999);

        var config = CreateMoCoV2Config();
        config.MoCo!.QueueSize = queueSize;

        return new MoCoV2<T>(encoder, momentumEncoder, projector, momentumProjector, projectionDim, config);
    }
}
