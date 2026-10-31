using AiDotNet.Clustering.Options;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// The sizes and schedules of a self-organizing map.
/// </summary>
/// <param name="Width">Neurons per grid row.</param>
/// <param name="Height">Grid rows.</param>
/// <param name="InputDimension">The length of an input vector.</param>
/// <param name="Steps">The training length T: the step count over which the learning rate falls to zero and the
/// radius to one.</param>
/// <param name="InitialLearningRate">α(0).</param>
/// <param name="InitialRadius">The neighborhood radius at step 0; non-positive means half the larger grid side.</param>
/// <param name="Topology">Rectangular, hexagonal (SOM_PAK's lattice) or toroidal (rectangular, wrapping at the edges).</param>
/// <param name="Neighborhood">The neighborhood kernel h(d, r).</param>
internal sealed record SelfOrganizingMapConfiguration(int Width, int Height, int InputDimension, int Steps,
    double InitialLearningRate, double InitialRadius = -1, SOMTopology Topology = SOMTopology.Rectangular,
    NeighborhoodFunction Neighborhood = NeighborhoodFunction.Gaussian)
{
    public int Neurons => Width * Height;

    public double StartRadius => InitialRadius > 0 ? InitialRadius : Math.Max(Width, Height) / 2.0;
}

/// <summary>
/// Kohonen's self-organizing map (Kohonen 1982) trained as SOM_PAK, the reference package from Kohonen's laboratory
/// (Kohonen, Hynninen, Kangas &amp; Laaksonen 1996), trains it: one input at a time, the best-matching unit c is the
/// codebook vector nearest the input, and every neuron i moves toward the input by
/// <c>w_i ← w_i + α(t)·h(d(c, i), r(t))·(x − w_i)</c>.
/// </summary>
/// <remarks>
/// <para>The paper asks only that the gain α and the neighborhood shrink with time; SOM_PAK's schedules fill that in:
/// <c>α(t) = α(0)·(1 − t/T)</c> and <c>r(t) = 1 + (r(0) − 1)·(1 − t/T)</c>. Grid distance is Euclidean between lattice
/// positions: (column, row) on a rectangular grid, and on SOM_PAK's hexagonal lattice odd rows are shifted half a unit
/// right and rows are √0.75 apart. The Gaussian kernel is <c>exp(−d²/2r²)</c> and the bubble kernel is 1 within the
/// radius and 0 beyond it.</para>
/// <para>The clustering model and the neural-network model share this map; each adds its own reading of it.</para>
/// </remarks>
internal sealed class SelfOrganizingMapCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public SelfOrganizingMapCore(SelfOrganizingMapConfiguration configuration, Tensor<T>? codebook = null)
    {
        Configuration = configuration;
        if (configuration.Width <= 0 || configuration.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(configuration), "The grid needs at least one neuron.");
        if (configuration.InputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(configuration), "The input dimension must be positive.");
        if (configuration.Steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(configuration), "The training length must be positive.");
        Codebook = codebook ?? new Tensor<T>(new[] { configuration.Neurons, configuration.InputDimension });
        if (Codebook.Shape[0] != configuration.Neurons || Codebook.Shape[1] != configuration.InputDimension)
            throw new ArgumentException("The codebook must be [neurons, input dimension].", nameof(codebook));
    }

    public SelfOrganizingMapConfiguration Configuration { get; }

    /// <summary>The codebook <c>[Width·Height, InputDimension]</c>; neuron <c>row·Width + column</c>.</summary>
    public Tensor<T> Codebook { get; }

    /// <summary>The lattice position (x, y) of a neuron.</summary>
    public (double X, double Y) Position(int neuron)
    {
        int row = neuron / Configuration.Width, column = neuron % Configuration.Width;
        return Configuration.Topology == SOMTopology.Hexagonal
            ? (column + ((row & 1) == 1 ? 0.5 : 0.0), row * Math.Sqrt(0.75))
            : (column, row);
    }

    /// <summary>The grid distance between two neurons (wrapping at the edges on a toroidal map).</summary>
    public double GridDistance(int a, int b)
    {
        var (ax, ay) = Position(a);
        var (bx, by) = Position(b);
        double dx = Math.Abs(ax - bx), dy = Math.Abs(ay - by);
        if (Configuration.Topology == SOMTopology.Toroidal)
        {
            dx = Math.Min(dx, Configuration.Width - dx);
            dy = Math.Min(dy, Configuration.Height - dy);
        }
        return Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>α(t) = α(0)·(1 − t/T), never below zero.</summary>
    public double LearningRate(int step)
        => Configuration.InitialLearningRate * Math.Max(0.0, 1.0 - (double)step / Configuration.Steps);

    /// <summary>r(t) = 1 + (r(0) − 1)·(1 − t/T).</summary>
    public double Radius(int step)
        => 1.0 + (Configuration.StartRadius - 1.0) * Math.Max(0.0, 1.0 - (double)step / Configuration.Steps);

    /// <summary>The neighborhood kernel at grid distance <paramref name="distance"/> and radius <paramref name="radius"/>.</summary>
    public double Neighborhood(double distance, double radius) => Configuration.Neighborhood switch
    {
        NeighborhoodFunction.Bubble => distance <= radius ? 1.0 : 0.0,
        // The Ricker wavelet: excitation near the winner, inhibition in a ring around it.
        NeighborhoodFunction.MexicanHat => (1.0 - distance * distance / (radius * radius))
            * Math.Exp(-distance * distance / (2.0 * radius * radius)),
        _ => Math.Exp(-distance * distance / (2.0 * radius * radius)),
    };

    /// <summary>The codebook vector nearest <paramref name="input"/> (squared Euclidean distance unless a metric is
    /// given); the first on a tie.</summary>
    public int BestMatchingUnit(ReadOnlySpan<T> input, Func<Vector<T>, Vector<T>, T>? distance = null)
    {
        int d = Configuration.InputDimension;
        var codebook = Codebook.AsSpan();
        int best = 0;
        double bestDistance = double.PositiveInfinity;
        Vector<T>? inputVector = distance is null ? null : new Vector<T>(input.ToArray());
        for (int n = 0; n < Configuration.Neurons; n++)
        {
            double value;
            if (distance is null)
            {
                value = 0;
                for (int j = 0; j < d; j++)
                {
                    double diff = NumOps.ToDouble(input[j]) - NumOps.ToDouble(codebook[n * d + j]);
                    value += diff * diff;
                }
            }
            else
            {
                value = NumOps.ToDouble(distance(inputVector!, new Vector<T>(codebook.Slice(n * d, d).ToArray())));
            }
            if (value < bestDistance)
            {
                bestDistance = value;
                best = n;
            }
        }
        return best;
    }

    /// <summary>
    /// One training step at time <paramref name="step"/>: finds the best-matching unit and moves every neuron toward
    /// the input. Returns the unit, and the per-weight update direction <c>h·(w − x)</c> the step descended (the
    /// gradient of the neighborhood-weighted quantization objective, for owners that publish gradients).
    /// </summary>
    public (int Winner, Vector<T> Direction) Step(ReadOnlySpan<T> input, int step, Func<Vector<T>, Vector<T>, T>? distance = null)
    {
        int winner = BestMatchingUnit(input, distance);
        double rate = LearningRate(step), radius = Radius(step);
        int d = Configuration.InputDimension;
        var codebook = Codebook.AsWritableSpan();
        var direction = new Vector<T>(Configuration.Neurons * d);
        for (int n = 0; n < Configuration.Neurons; n++)
        {
            double h = Neighborhood(GridDistance(winner, n), radius);
            if (h == 0.0) continue;
            for (int j = 0; j < d; j++)
            {
                int k = n * d + j;
                double w = NumOps.ToDouble(codebook[k]), x = NumOps.ToDouble(input[j]);
                direction[k] = NumOps.FromDouble(h * (w - x));
                codebook[k] = NumOps.FromDouble(w + rate * h * (x - w));
            }
        }
        return (winner, direction);
    }

    /// <summary>SOM_PAK's random initialization: every codebook component uniform within that component's range in
    /// the data (a degenerate range spans one unit).</summary>
    public void InitializeWithinRange(double[] minimum, double[] maximum, Random random)
    {
        int d = Configuration.InputDimension;
        var codebook = Codebook.AsWritableSpan();
        for (int n = 0; n < Configuration.Neurons; n++)
            for (int j = 0; j < d; j++)
            {
                double range = maximum[j] - minimum[j];
                if (range < 1e-10) range = 1.0;
                codebook[n * d + j] = NumOps.FromDouble(minimum[j] + random.NextDouble() * range);
            }
    }
}
