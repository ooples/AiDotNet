using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Shared configuration for generative adversarial networks (DCGAN, WGAN, WGAN-GP, BigGAN,
/// SAGAN, StyleGAN, ProgressiveGAN, InfoGAN, Pix2Pix, CycleGAN).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A GAN trains two networks against each other. The generator invents
/// images; the discriminator (sometimes called the critic) tries to tell real images from
/// invented ones. Each gets better because the other does. The settings here describe how
/// big each of the two networks is and how they are trained.
/// </para>
/// <para>
/// Consumers that derive from this base must assign their own required dimensions and rates.
/// This base does not construct a GAN or replace the separate options and scalar constructor
/// parameters used by existing GAN models.
/// See <see cref="ModelHyperparameterOptions"/> for why these properties are non-nullable.
/// </para>
/// </remarks>
public abstract class GanOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the length of the random noise vector the generator starts from.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The generator turns a list of random numbers into an image.
    /// This is how many random numbers it starts with — the "seed idea" for one image. 100 and
    /// 128 are common; StyleGAN uses 512.</para>
    /// </remarks>
    public int LatentSize { get; set; }

    /// <summary>
    /// Gets or sets the base channel count of the generator. Layers scale up from this.
    /// </summary>
    /// <value>A positive channel count supplied by the consumer. This abstract base has no model-specific default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Roughly how wide the image-making network is. Bigger means
    /// more detail and slower training.</para>
    /// </remarks>
    public int GeneratorChannels { get; set; }

    /// <summary>
    /// Gets or sets the base channel count of the discriminator or critic.
    /// </summary>
    /// <value>A positive channel count supplied by the consumer. This abstract base has no model-specific default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the width of the network judging generated images.
    /// It is independent of the generator's width and must be configured by the model using these options.</para>
    /// </remarks>
    public int DiscriminatorChannels { get; set; }

    /// <summary>
    /// Gets or sets the number of colour channels in the generated image.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 3 for colour, 1 for greyscale.</para>
    /// </remarks>
    public int ImageChannels { get; set; } = 3;

    /// <summary>
    /// Gets or sets how many discriminator or critic updates run per generator update.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Wasserstein GANs train the critic several times for each
    /// time they train the generator, which keeps the two from getting out of step. The WGAN
    /// paper uses 5. Models that update both once per step leave this unset.</para>
    /// </remarks>
    public int CriticIterations { get; set; }

    /// <summary>
    /// Gets or sets the learning rate the adversarial pair starts training at.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the networks take when correcting themselves.
    /// GANs are unusually sensitive to this; 0.0002 with the Adam optimizer is the value most
    /// GAN papers settled on.</para>
    /// </remarks>
    public double InitialLearningRate { get; set; }

    /// <summary>
    /// Validates the required dimensions and initial rate for callers of this shared base.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, or the initial learning rate is
    /// non-finite or nonpositive. Required values must be supplied by the concrete consumer.
    /// </exception>
    protected void ValidateCore()
    {
        Require(LatentSize, nameof(LatentSize));
        Require(GeneratorChannels, nameof(GeneratorChannels));
        Require(DiscriminatorChannels, nameof(DiscriminatorChannels));
        Require(ImageChannels, nameof(ImageChannels));
        Require(InitialLearningRate, nameof(InitialLearningRate));
    }
}
