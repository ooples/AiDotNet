using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for DAM-VSR, appearance and motion disentanglement for video super-resolution on Stable Video
/// Diffusion.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Kong et al., SIGGRAPH 2025 (arXiv 2507.01012), Sec. 3-4, and Stable Video Diffusion as diffusers
/// releases it (<c>stabilityai/stable-video-diffusion-img2vid</c>): the 1.5B-parameter spatio-temporal U-Net (320, 640,
/// 1280, 1280 channels; 5, 10, 20, 20 heads), its VAE with the temporal decoder, OpenCLIP ViT-H/14 as the image encoder,
/// 14-frame clips and a x4 output. Sampling is 30 Euler steps on the Karras schedule (σ from 700 to 0.002) entered by
/// SDEdit 60% of the way down, with motion-aligned bidirectional sampling. Training is AdamW at a constant 8e-5: 60,000
/// iterations of the ControlNet, 45,000 of the backward pass's temporal value and output projections, and 10,000 of the
/// VAE decoder on L2, perceptual and GAN losses (1, 1, 0.025) after 8 sampling steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> DAM-VSR upscales video with a video diffusion model. An image upscaler sharpens one reference
/// frame (the appearance), the low-resolution video steers how everything moves (the motion), and the diffusion model
/// fills in the rest of the frames consistently.
/// </para>
/// </remarks>
public class DAMVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public DAMVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DAMVSROptions(DAMVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        UNetChannels = (int[])other.UNetChannels.Clone();
        UNetHeads = (int[])other.UNetHeads.Clone();
        LayersPerBlock = other.LayersPerBlock;
        TransformerDepth = other.TransformerDepth;
        AddedTimeEmbedDim = other.AddedTimeEmbedDim;
        VaeChannels = (int[])other.VaeChannels.Clone();
        VaeLayersPerBlock = other.VaeLayersPerBlock;
        LatentScale = other.LatentScale;
        ImageEncoderSize = other.ImageEncoderSize;
        ImageEncoderPatchSize = other.ImageEncoderPatchSize;
        ImageEncoderWidth = other.ImageEncoderWidth;
        ImageEncoderDepth = other.ImageEncoderDepth;
        ImageEncoderHeads = other.ImageEncoderHeads;
        ImageEncoderMlpDim = other.ImageEncoderMlpDim;
        ImageEmbedDim = other.ImageEmbedDim;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        Fps = other.Fps;
        MotionBucketId = other.MotionBucketId;
        NoiseAugmentationStrength = other.NoiseAugmentationStrength;
        NumInferenceSteps = other.NumInferenceSteps;
        SdeditStrength = other.SdeditStrength;
        SigmaMin = other.SigmaMin;
        SigmaMax = other.SigmaMax;
        KarrasRho = other.KarrasRho;
        BidirectionalSampling = other.BidirectionalSampling;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        ControlNetIterations = other.ControlNetIterations;
        BackwardIterations = other.BackwardIterations;
        DecoderSamplingSteps = other.DecoderSamplingSteps;
        PerceptualWeight = other.PerceptualWeight;
        GANWeight = other.GANWeight;
        DiscriminatorChannels = other.DiscriminatorChannels;
        TrainingSigmaMean = other.TrainingSigmaMean;
        TrainingSigmaStd = other.TrainingSigmaStd;
    }

    #region Architecture

    /// <summary>Gets or sets the U-Net stage widths (320, 640, 1280, 1280); each must divide into 32 groups.</summary>
    public int[] UNetChannels { get; set; } = { 320, 640, 1280, 1280 };

    /// <summary>Gets or sets the attention heads per U-Net stage (5, 10, 20, 20).</summary>
    public int[] UNetHeads { get; set; } = { 5, 10, 20, 20 };

    /// <summary>Gets or sets the ResNets per down stage (2; up stages have one more).</summary>
    public int LayersPerBlock { get; set; } = 2;

    /// <summary>Gets or sets the transformer layers per attention (1).</summary>
    public int TransformerDepth { get; set; } = 1;

    /// <summary>Gets or sets the width each added time id is embedded at (256).</summary>
    public int AddedTimeEmbedDim { get; set; } = 256;

    /// <summary>Gets or sets the VAE stage widths (128, 256, 512, 512); the latent is 2^(stages - 1) times smaller.</summary>
    public int[] VaeChannels { get; set; } = { 128, 256, 512, 512 };

    /// <summary>Gets or sets the VAE encoder's residual blocks per stage (2).</summary>
    public int VaeLayersPerBlock { get; set; } = 2;

    /// <summary>Gets or sets the latent scale (0.18215).</summary>
    public double LatentScale { get; set; } = 0.18215;

    /// <summary>Gets or sets the image encoder's input side (224).</summary>
    public int ImageEncoderSize { get; set; } = 224;

    /// <summary>Gets or sets the image encoder's patch side (14).</summary>
    public int ImageEncoderPatchSize { get; set; } = 14;

    /// <summary>Gets or sets the image encoder's width (1,280).</summary>
    public int ImageEncoderWidth { get; set; } = 1280;

    /// <summary>Gets or sets the image encoder's transformer layers (32).</summary>
    public int ImageEncoderDepth { get; set; } = 32;

    /// <summary>Gets or sets the image encoder's attention heads (16).</summary>
    public int ImageEncoderHeads { get; set; } = 16;

    /// <summary>Gets or sets the image encoder's MLP width (5,120).</summary>
    public int ImageEncoderMlpDim { get; set; } = 5120;

    /// <summary>Gets or sets the image embedding the U-Net cross-attends to (1,024).</summary>
    public int ImageEmbedDim { get; set; } = 1024;

    /// <summary>Gets or sets the spatial upscaling factor (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the frames of a clip (14, SVD's length).</summary>
    public int NumFrames { get; set; } = 14;

    #endregion

    #region Conditioning and sampling

    /// <summary>Gets or sets the frame rate conditioning (7; the first added time id is fps - 1, as SVD's pipeline passes it).</summary>
    public int Fps { get; set; } = 7;

    /// <summary>Gets or sets SVD's motion bucket conditioning (127).</summary>
    public int MotionBucketId { get; set; } = 127;

    /// <summary>Gets or sets the noise added to the reference image before encoding (0.02).</summary>
    public double NoiseAugmentationStrength { get; set; } = 0.02;

    /// <summary>Gets or sets the sampling schedule's steps (30).</summary>
    public int NumInferenceSteps { get; set; } = 30;

    /// <summary>Gets or sets the fraction of the schedule SDEdit enters at (0.6, leaving 18 of 30 steps).</summary>
    public double SdeditStrength { get; set; } = 0.6;

    /// <summary>Gets or sets the smallest σ of the Karras schedule (0.002).</summary>
    public double SigmaMin { get; set; } = 0.002;

    /// <summary>Gets or sets the largest σ of the Karras schedule (700).</summary>
    public double SigmaMax { get; set; } = 700.0;

    /// <summary>Gets or sets the Karras schedule's ρ (7).</summary>
    public double KarrasRho { get; set; } = 7.0;

    /// <summary>Gets or sets whether sampling averages the forward pass with the motion-aligned backward pass (true).</summary>
    public bool BidirectionalSampling { get; set; } = true;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets AdamW's constant rate (8e-5).</summary>
    public double LearningRate { get; set; } = 8e-5;

    /// <summary>Gets or sets AdamW's decoupled weight decay (0.01, PyTorch's default; the paper states none).</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>Gets or sets the iterations that train the ControlNet with the U-Net frozen (60,000).</summary>
    public int ControlNetIterations { get; set; } = 60000;

    /// <summary>Gets or sets the iterations that then train the backward pass's temporal value and output projections (45,000).</summary>
    public int BackwardIterations { get; set; } = 45000;

    /// <summary>Gets or sets the sampling steps whose output trains the VAE decoder in the last stage (8).</summary>
    public int DecoderSamplingSteps { get; set; } = 8;

    /// <summary>Gets or sets the decoder stage's perceptual loss weight α (1; 0 leaves the VGG19 out).</summary>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the decoder stage's GAN loss weight β (0.025).</summary>
    public double GANWeight { get; set; } = 0.025;

    /// <summary>Gets or sets the decoder stage discriminator's base width (64, RealBasicVSR's U-Net discriminator).</summary>
    public int DiscriminatorChannels { get; set; } = 64;

    /// <summary>Gets or sets the mean of log σ for training noise (0.7, SVD's high-resolution fine-tuning).</summary>
    public double TrainingSigmaMean { get; set; } = 0.7;

    /// <summary>Gets or sets the standard deviation of log σ for training noise (1.6).</summary>
    public double TrainingSigmaStd { get; set; } = 1.6;

    #endregion
}
