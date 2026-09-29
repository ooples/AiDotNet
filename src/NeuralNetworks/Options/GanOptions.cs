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
/// Derived options classes assign their paper's values in their parameterless constructor.
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
    /// <remarks>
    /// <para><b>For Beginners:</b> Roughly how wide the image-making network is. Bigger means
    /// more detail and slower training.</para>
    /// </remarks>
    public int GeneratorChannels { get; set; }

    /// <summary>
    /// Gets or sets the base channel count of the discriminator or critic.
    /// </summary>
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
    /// <value>
    /// Defaults to 0.0002, the Adam learning rate the DCGAN paper settled on and which most
    /// later GAN papers inherited.
    /// </value>
    public double InitialLearningRate { get; set; } = 0.0002;

    /// <summary>
    /// Throws if a dimension every GAN requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    /// <param name="requireLatentSize">
    /// False for a model that takes its latent size from a constructor argument rather than
    /// from these options, which then legitimately leave it unset.
    /// </param>
    /// <param name="requireChannels">
    /// False for a model that builds its generator and discriminator widths from its supplied
    /// architectures rather than from these options.
    /// </param>
    protected void ValidateCore(bool requireLatentSize = true, bool requireChannels = true)
    {
        // Strict by default: a model whose paper defaults live here must have assigned them.
        // Only a model that genuinely does not read a value from its options opts out of it, so
        // requiring it at the shared boundary would throw at construction for a valid model.
        if (requireLatentSize)
            Require(LatentSize, nameof(LatentSize));
        if (requireChannels)
        {
            Require(GeneratorChannels, nameof(GeneratorChannels));
            Require(DiscriminatorChannels, nameof(DiscriminatorChannels));
        }
        Require(ImageChannels, nameof(ImageChannels));
        Require(InitialLearningRate, nameof(InitialLearningRate));
    }
}
