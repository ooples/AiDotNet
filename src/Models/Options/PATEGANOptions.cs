namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for PATE-GAN (Jordon, Yoon and van der Schaar, ICLR 2019), a GAN whose discriminator is a
/// student trained on generated samples labelled by a noisy vote of teachers, each of which sees only one disjoint
/// partition of the data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Defaults follow the paper's appendix ("Hyper-parameter optimization"): depth-3 generator and student with
/// <c>d</c> and <c>d/2</c> hidden units (<c>d</c> the encoded row width), depth-1 teachers, ReLU hidden layers and
/// sigmoid outputs, batch size 64, n_T = n_S = 5, Adam with learning rate 1e-4. The privacy budget (ε = 1,
/// δ = 1e-5), the noise size λ = 1 and k = 10 teachers are the defaults of the authors' released code; the paper
/// selects k by cross-validation from {N/10, N/50, ..., N/10000}.
/// </para>
/// <para>
/// <b>For Beginners:</b> PATE-GAN never lets one network see all the real data. Each "teacher" sees a small slice;
/// a "student" learns only from the teachers' noisy votes about generated rows; the generator learns only from the
/// student. Training stops once the privacy budget ε is spent.
/// <code>
/// var options = new PATEGANOptions&lt;double&gt; { Epsilon = 1.0, NumTeachers = 10 };
/// var pategan = new PATEGANGenerator&lt;double&gt;(architecture, options);
/// pategan.Fit(data, columns);
/// </code>
/// </para>
/// </remarks>
public class PATEGANOptions<T> : SyntheticDataGeneratorOptions<T>
{
    /// <summary>
    /// Gets or sets the number k of teachers, each trained on one of k disjoint partitions of the data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More teachers make each vote count for less, so the noise matters less and the
    /// privacy guarantee is tighter, but each teacher then learns from fewer rows.</para>
    /// </remarks>
    /// <value>Defaults to 10, the authors' released code.</value>
    public int NumTeachers { get; set; } = 10;

    /// <summary>
    /// Gets or sets the scale λ of the Laplace noise added to each vote count (Eq. 4).
    /// </summary>
    /// <remarks>
    /// <para>A single noisy vote is (1/λ, 0)-differentially private (section 3), so larger noise spends less of the
    /// budget per query.</para>
    /// </remarks>
    /// <value>Defaults to 1.0, the authors' released code.</value>
    public double LaplaceScale { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the privacy budget ε; training stops once the moments accountant's ε̂ reaches it (Algorithm 1).
    /// </summary>
    /// <value>Defaults to 1.0.</value>
    public double Epsilon { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets δ of the (ε, δ) guarantee.
    /// </summary>
    /// <value>Defaults to 1e-5, the value fixed throughout the paper's experiments.</value>
    public double Delta { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the number L of moments the accountant tracks (Algorithm 1, line 2).
    /// </summary>
    /// <value>Defaults to 20, the authors' released code.</value>
    public int MomentOrders { get; set; } = 20;

    /// <summary>
    /// Gets or sets the width of the generator's noise z ~ Unif([0, 1]^n); null uses the encoded row width d, as the
    /// paper does (section 4.2.2 relies on G([0, 1]^d) covering [0, 1]^d).
    /// </summary>
    /// <value>Defaults to null.</value>
    public int? EmbeddingDimension { get; set; }

    /// <summary>
    /// Gets or sets the generator's hidden widths; null uses the paper's [d, d/2].
    /// </summary>
    /// <value>Defaults to null.</value>
    public int[]? GeneratorDimensions { get; set; }

    /// <summary>
    /// Gets or sets the teachers' hidden widths; empty is the paper's depth-1 teacher (a logistic regression).
    /// </summary>
    /// <value>Defaults to an empty array.</value>
    public int[] TeacherDimensions { get; set; } = [];

    /// <summary>
    /// Gets or sets the student's hidden widths; null uses the paper's [d, d/2].
    /// </summary>
    /// <value>Defaults to null.</value>
    public int[]? StudentDimensions { get; set; }

    /// <summary>
    /// Gets or sets the mini-batch size n for the teachers, the student and the generator.
    /// </summary>
    /// <value>Defaults to 64 (paper appendix).</value>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the maximum number of generator iterations; training also stops when ε is spent.
    /// </summary>
    /// <value>Defaults to 10000, the authors' released code's iteration cap.</value>
    public int Epochs { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the Adam learning rate of every network.
    /// </summary>
    /// <value>Defaults to 1e-4 (paper appendix).</value>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the number n_T of teacher updates per generator iteration.
    /// </summary>
    /// <value>Defaults to 5 (paper appendix).</value>
    public int TeacherSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number n_S of student updates per generator iteration.
    /// </summary>
    /// <value>Defaults to 5 (paper appendix).</value>
    public int StudentSteps { get; set; } = 5;
}
