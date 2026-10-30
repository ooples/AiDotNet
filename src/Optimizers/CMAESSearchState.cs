namespace AiDotNet.Optimizers;

/// <summary>
/// The complete state of a CMA-ES search between generations: the distribution's mean, step size and covariance,
/// the two evolution paths, and the generation count.
/// </summary>
/// <remarks>
/// <para>
/// Created by <see cref="CMAESOptimizer{T, TInput, TOutput}.CreateSearchState"/> and advanced one generation at a
/// time by <see cref="CMAESOptimizer{T, TInput, TOutput}.Ask"/> and
/// <see cref="CMAESOptimizer{T, TInput, TOutput}.Tell"/>. Everything the next generation depends on is held here as
/// plain numbers, so a caller can keep a search running across calls, copy it, or serialize it and resume it later.
/// The recombination weights and learning rates are not stored: they are fixed functions of
/// <see cref="Dimension"/> and <see cref="PopulationSize"/> (Hansen 2016, Table 1) and are recomputed.
/// </para>
/// <para><b>For Beginners:</b> CMA-ES keeps a "cloud" of where good solutions probably are: its centre (the mean),
/// its size (sigma) and its shape (the covariance). This object is that cloud plus a little memory of where it has
/// been moving, so the search can be paused after any round and continued exactly where it stopped.
/// </para>
/// </remarks>
public sealed class CMAESSearchState
{
    /// <summary>Gets or sets the number of parameters being searched over, n.</summary>
    public int Dimension { get; set; }

    /// <summary>Gets or sets the number of candidates sampled per generation, lambda.</summary>
    public int PopulationSize { get; set; }

    /// <summary>Gets or sets the number of generations completed so far.</summary>
    public int Generation { get; set; }

    /// <summary>Gets or sets the mean of the sampling distribution, m.</summary>
    public double[] Mean { get; set; } = Array.Empty<double>();

    /// <summary>Gets or sets the global step size, sigma.</summary>
    public double Sigma { get; set; }

    /// <summary>Gets or sets the conjugate evolution path p_sigma that controls the step size.</summary>
    public double[] PathSigma { get; set; } = Array.Empty<double>();

    /// <summary>Gets or sets the evolution path p_c that drives the rank-one covariance update.</summary>
    public double[] PathC { get; set; } = Array.Empty<double>();

    /// <summary>Gets or sets the covariance matrix C, stored as rows.</summary>
    public double[][] Covariance { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Whether the step size is still a usable positive, finite number. A search whose sigma has collapsed to zero,
    /// overflowed or become NaN cannot sample a meaningful generation.
    /// </summary>
    public bool IsDegenerate => double.IsNaN(Sigma) || double.IsInfinity(Sigma) || Sigma <= 0.0;

    /// <summary>Returns an independent deep copy of this state.</summary>
    public CMAESSearchState Clone()
    {
        var covariance = new double[Covariance.Length][];
        for (int i = 0; i < Covariance.Length; i++) covariance[i] = (double[])Covariance[i].Clone();
        return new CMAESSearchState
        {
            Dimension = Dimension,
            PopulationSize = PopulationSize,
            Generation = Generation,
            Mean = (double[])Mean.Clone(),
            Sigma = Sigma,
            PathSigma = (double[])PathSigma.Clone(),
            PathC = (double[])PathC.Clone(),
            Covariance = covariance,
        };
    }
}

/// <summary>
/// One CMA-ES generation sampled by <see cref="CMAESOptimizer{T, TInput, TOutput}.Ask"/>: the candidate points to
/// evaluate, plus what <see cref="CMAESOptimizer{T, TInput, TOutput}.Tell"/> needs to update the distribution from
/// their fitness.
/// </summary>
/// <typeparam name="T">The numeric type of the candidate points.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Evaluate every point in <see cref="Points"/> (lower is better), then hand the values
/// back to <c>Tell</c> together with this object.</para>
/// </remarks>
public sealed class CMAESGeneration<T>
{
    internal CMAESGeneration(Vector<T>[] points, double[][] draws, double[,] factor, int generation)
    {
        Points = points;
        Draws = draws;
        Factor = factor;
        GenerationIndex = generation;
    }

    /// <summary>Gets the candidate points, m + sigma * y_k, in sampling order.</summary>
    public IReadOnlyList<Vector<T>> Points { get; }

    /// <summary>The draws y_k ~ N(0, C) the points were made from.</summary>
    internal double[][] Draws { get; }

    /// <summary>The lower Cholesky factor of C the draws were made with.</summary>
    internal double[,] Factor { get; }

    /// <summary>The state's generation count when this generation was sampled.</summary>
    internal int GenerationIndex { get; }

    /// <summary>Whether <c>Tell</c> has consumed this generation.</summary>
    internal bool Told { get; set; }
}
