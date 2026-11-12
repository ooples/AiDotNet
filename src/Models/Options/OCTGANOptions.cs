namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for OCT-GAN (Kim et al. 2021, "OCT-GAN: Neural ODE-based Conditional Tabular GANs"), a
/// conditional tabular GAN whose generator maps <c>z ⊕ c</c> through a neural ODE and whose discriminator classifies
/// the trajectory of a neural ODE at learnable time points.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Defaults follow section 4.1.4 of the paper: 256-wide hidden layers in both networks, m = 3 trajectory segments,
/// one shared learning rate λ = λ_G = λ_D = λ_t = 2e-4 from the paper's grid (the time points share the critic's
/// optimizer, as in the authors' code), a decay factor ψ = 0.97
/// every ξ = 2 epochs, 300 epochs, a mini-batch of 500 and 10 VGM modes.
/// </para>
/// <para>
/// <b>For Beginners:</b> OCT-GAN learns to produce new table rows that look like the real ones. You can ask it for
/// rows with a given category in a discrete column (the "condition"), and it learns to honour that request.
///
/// Example:
/// <code>
/// var options = new OCTGANOptions&lt;double&gt; { Epochs = 300, TimePoints = 3 };
/// var octgan = new OCTGANGenerator&lt;double&gt;(architecture, options);
/// octgan.Fit(data, columns);
/// var synthetic = octgan.Generate(1000);
/// </code>
/// </para>
/// </remarks>
public class OCTGANOptions<T> : SyntheticDataGeneratorOptions<T>
{
    /// <summary>
    /// Gets or sets the dimension of the noise vector <c>z</c>.
    /// </summary>
    /// <value>Defaults to 128, the width of the CTGAN code the paper builds on.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the output widths of the generator's two residual blocks (FC1 and FC2, Eqs. 12-13).
    /// </summary>
    /// <value>Defaults to [256, 256] (section 4.1.4).</value>
    public int[] GeneratorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the output widths of the discriminator's layers before its ODE (FC1 and FC2, Eq. 2). The last
    /// one is the width of the ODE state <c>h(t)</c>.
    /// </summary>
    /// <value>Defaults to [256, 256] (section 4.1.4).</value>
    public int[] DiscriminatorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the number m of segments the discriminator's ODE trajectory on [0, 1] is read at: t_0 = 0 and
    /// t_m = 1 are fixed and the m − 1 interior points t_1..t_{m-1} are learned.
    /// </summary>
    /// <remarks>
    /// <para>The paper searches m ∈ {3, 5, 7}. The interior points start evenly spaced, t_i = i/m (OCT-GAN(fixed)),
    /// and are trained with the critic.</para>
    /// <para><b>For Beginners:</b> The discriminator watches a hidden vector evolve over time and looks at it at m
    /// moments; it learns which moments best tell real rows from fake ones.</para>
    /// </remarks>
    /// <value>Defaults to 3.</value>
    public int TimePoints { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of fourth-order Runge-Kutta steps used for each ODE integral (the generator's [0, 1]
    /// and each of the discriminator's [t_{i-1}, t_i]).
    /// </summary>
    /// <remarks>
    /// The authors integrate with the adaptive Dormand-Prince solver; a fixed-step RK4 keeps the cost of every
    /// training step predictable and gives the same time-point gradients (Proposition 3.1) by differentiating through
    /// the solver.
    /// </remarks>
    /// <value>Defaults to 4.</value>
    public int OdeSolverSteps { get; set; } = 4;

    /// <summary>
    /// Gets or sets the Gumbel-softmax temperature of the generator's mode and discrete-column heads (Eqs. 15-16).
    /// </summary>
    /// <value>Defaults to 0.2, the temperature of the CTGAN code the paper builds on.</value>
    public double GumbelTemperature { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>Defaults to 500, the smallest of the paper's {500, 1000, 1500}.</value>
    public int BatchSize { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Defaults to 300 (section 4.1.4).</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the generator and discriminator learning rate λ_G = λ_D.
    /// </summary>
    /// <value>Defaults to 2e-4, from the paper's grid {2e-3, 2e-4, 2e-5, 2e-6}.</value>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Gets or sets the decay factor ψ applied to λ_G and λ_D (and so to the time points) every <see cref="LearningRateDecayPeriod"/> epochs.
    /// </summary>
    /// <value>Defaults to 0.97 (section 4.1.4).</value>
    public double LearningRateDecay { get; set; } = 0.97;

    /// <summary>
    /// Gets or sets the decay period ξ in epochs.
    /// </summary>
    /// <value>Defaults to 2 (section 4.1.4).</value>
    public int LearningRateDecayPeriod { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column encoding.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> VGM (Variational Gaussian Mixture) splits each numeric column
    /// into a mixture of bell curves, which helps the GAN learn multi-modal distributions
    /// (columns with multiple peaks, like age distributions with clusters at 25 and 55).</para>
    /// </remarks>
    /// <value>Defaults to 10, the smallest of the paper's {10, 20, 30}.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate of the discriminator's first two layers (Eq. 2).
    /// </summary>
    /// <value>Defaults to 0.5, the rate of the CTGAN code the paper builds on.</value>
    public double DiscriminatorDropout { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the gradient-penalty weight λ of the WGAN-GP loss (Eq. 1).
    /// </summary>
    /// <value>Defaults to 10 (Gulrajani et al. 2017).</value>
    public double GradientPenaltyWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the number of discriminator updates per generator update.
    /// </summary>
    /// <value>Defaults to 1: Algorithm 1 trains the discriminator once per mini-batch.</value>
    public int DiscriminatorSteps { get; set; } = 1;
}
