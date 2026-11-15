namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// nerfacc's <c>OccupancyGrid</c> (v0.3, the version instant-nsr-pl pins) over the box <c>[−r, r]³</c> with AABB
/// contraction: a running occupancy per cell, refreshed every few iterations from the field, and the binary mask the
/// ray marcher uses to skip empty space (Müller et al. 2022, Sec. 5.4).
/// </summary>
/// <remarks>
/// <para>
/// An update evaluates the occupancy at one jittered point per chosen cell, <c>x = (cell + U[0,1)³) / R</c> mapped to
/// the box. During the warm-up every cell is chosen; afterwards <c>R³ / 4</c> uniformly drawn cells plus as many
/// occupied cells (all of them when fewer). Each chosen cell becomes <c>max(decay · old, new)</c>, computed from the
/// values before the update (with repeats the last write wins, as PyTorch's indexed assignment does), and a cell is
/// occupied when its value exceeds <c>min(mean of all cells, threshold)</c>. Cells are indexed
/// <c>x·R² + y·R + z</c>.
/// </para>
/// <para>
/// Until the first update the grid marks nothing empty, so an untrained field is marched densely (nerfacc's grid starts
/// all-empty, but instant-nsr-pl always updates it before the first rendered batch).
/// </para>
/// </remarks>
internal sealed class NeuSOccupancyGrid
{
    private readonly double[] _occupancy;
    private readonly bool[] _binary;

    /// <summary>Creates an empty grid.</summary>
    /// <param name="resolution">The cells per axis R.</param>
    /// <param name="radius">The box half extent r.</param>
    public NeuSOccupancyGrid(int resolution, double radius)
    {
        if (resolution <= 0 || resolution > 1024) throw new ArgumentOutOfRangeException(nameof(resolution));
        if (!(radius > 0) || double.IsInfinity(radius)) throw new ArgumentOutOfRangeException(nameof(radius));
        Resolution = resolution;
        Radius = radius;
        int cells = resolution * resolution * resolution;
        _occupancy = new double[cells];
        _binary = new bool[cells];
    }

    /// <summary>The cells per axis.</summary>
    public int Resolution { get; }

    /// <summary>The box half extent.</summary>
    public double Radius { get; }

    /// <summary>The cell count R³.</summary>
    public int CellCount => _occupancy.Length;

    /// <summary>Whether the grid has been updated at least once.</summary>
    public bool IsInitialized { get; private set; }

    /// <summary>Each cell's running occupancy.</summary>
    public IReadOnlyList<double> Occupancy => _occupancy;

    /// <summary>Each cell's occupied flag.</summary>
    public IReadOnlyList<bool> Binary => _binary;

    /// <summary>The flat index of cell <c>(x, y, z)</c>.</summary>
    public int CellIndex(int x, int y, int z) => (x * Resolution + y) * Resolution + z;

    /// <summary>
    /// Whether the point lies in an occupied cell (nerfacc <c>grid_occupied_at</c>): false outside the box; every
    /// point in the box counts as occupied before the first update.
    /// </summary>
    public bool IsOccupied(double x, double y, double z)
    {
        double scale = 1.0 / (2.0 * Radius);
        double ux = (x + Radius) * scale, uy = (y + Radius) * scale, uz = (z + Radius) * scale;
        if (ux < 0 || ux > 1 || uy < 0 || uy > 1 || uz < 0 || uz > 1) return false;
        if (!IsInitialized) return true;
        int ix = Math.Min((int)(ux * Resolution), Resolution - 1);
        int iy = Math.Min((int)(uy * Resolution), Resolution - 1);
        int iz = Math.Min((int)(uz * Resolution), Resolution - 1);
        return _binary[CellIndex(ix, iy, iz)];
    }

    /// <summary>
    /// nerfacc <c>every_n_step</c>: on iterations divisible by <paramref name="interval"/>, refreshes the grid from
    /// <paramref name="evaluate"/>; returns whether it did.
    /// </summary>
    /// <param name="iteration">The training iteration.</param>
    /// <param name="interval">The update interval n.</param>
    /// <param name="evaluate">The occupancy at <c>N</c> box points (<c>N · 3</c> values), <c>N</c> values.</param>
    /// <param name="threshold">The occupancy threshold.</param>
    /// <param name="emaDecay">The running-occupancy decay.</param>
    /// <param name="warmUpIterations">The iterations that refresh every cell.</param>
    /// <param name="random">The cell and jitter generator.</param>
    public bool EveryNStep(int iteration, int interval, Func<double[], double[]> evaluate, double threshold,
        double emaDecay, int warmUpIterations, Random random)
    {
        if (interval <= 0) throw new ArgumentOutOfRangeException(nameof(interval));
        if (iteration % interval != 0) return false;
        Update(iteration, evaluate, threshold, emaDecay, warmUpIterations, random);
        return true;
    }

    /// <summary>One refresh (nerfacc <c>OccupancyGrid._update</c>).</summary>
    /// <param name="iteration">The training iteration.</param>
    /// <param name="evaluate">The occupancy at <c>N</c> box points (<c>N · 3</c> values), <c>N</c> values.</param>
    /// <param name="threshold">The occupancy threshold.</param>
    /// <param name="emaDecay">The running-occupancy decay.</param>
    /// <param name="warmUpIterations">The iterations that refresh every cell.</param>
    /// <param name="random">The cell and jitter generator.</param>
    public void Update(int iteration, Func<double[], double[]> evaluate, double threshold, double emaDecay,
        int warmUpIterations, Random random)
    {
        if (evaluate is null) throw new ArgumentNullException(nameof(evaluate));
        if (random is null) throw new ArgumentNullException(nameof(random));
        var cells = iteration < warmUpIterations ? AllCells() : UniformAndOccupiedCells(random);
        int r = Resolution, r2 = r * r;
        var points = new double[cells.Length * 3];
        for (int k = 0; k < cells.Length; k++)
        {
            int c = cells[k];
            int x = c / r2, y = c / r % r, z = c % r;
            // (grid_coords + rand) / R in the unit cube, then contract_inv to the box.
            points[k * 3] = ((x + random.NextDouble()) / r * 2.0 - 1.0) * Radius;
            points[k * 3 + 1] = ((y + random.NextDouble()) / r * 2.0 - 1.0) * Radius;
            points[k * 3 + 2] = ((z + random.NextDouble()) / r * 2.0 - 1.0) * Radius;
        }

        var values = evaluate(points);
        if (values is null || values.Length != cells.Length)
            throw new InvalidOperationException("The occupancy evaluation must return one value per point.");
        Apply(cells, values, threshold, emaDecay);
    }

    /// <summary>
    /// The EMA write and re-thresholding of an update: <c>occs[cells] = max(occs[cells] · decay, values)</c> from the
    /// values before the write (last repeat wins), then <c>binary = occs &gt; min(mean(occs), threshold)</c>.
    /// </summary>
    /// <param name="cells">The chosen cells.</param>
    /// <param name="values">Their new occupancies.</param>
    /// <param name="threshold">The occupancy threshold.</param>
    /// <param name="emaDecay">The running-occupancy decay.</param>
    internal void Apply(int[] cells, double[] values, double threshold, double emaDecay)
    {
        if (cells is null) throw new ArgumentNullException(nameof(cells));
        if (values is null || values.Length != cells.Length) throw new ArgumentException("One value per cell is needed.", nameof(values));
        var updated = new double[cells.Length];
        for (int k = 0; k < cells.Length; k++) updated[k] = Math.Max(_occupancy[cells[k]] * emaDecay, values[k]);
        for (int k = 0; k < cells.Length; k++) _occupancy[cells[k]] = updated[k];
        double mean = 0;
        foreach (double v in _occupancy) mean += v;
        mean /= _occupancy.Length;
        double cut = Math.Min(mean, threshold);
        for (int i = 0; i < _binary.Length; i++) _binary[i] = _occupancy[i] > cut;
        IsInitialized = true;
    }

    private int[] AllCells()
    {
        var cells = new int[CellCount];
        for (int i = 0; i < cells.Length; i++) cells[i] = i;
        return cells;
    }

    // _sample_uniform_and_occupied_cells(R³ // 4).
    private int[] UniformAndOccupiedCells(Random random)
    {
        int n = CellCount / 4;
        var occupied = new List<int>();
        for (int i = 0; i < _binary.Length; i++)
            if (_binary[i]) occupied.Add(i);
        var cells = new List<int>(n + Math.Min(n, occupied.Count));
        for (int k = 0; k < n; k++) cells.Add(random.Next(CellCount));
        if (n < occupied.Count)
            for (int k = 0; k < n; k++) cells.Add(occupied[random.Next(occupied.Count)]);
        else
            cells.AddRange(occupied);
        return cells.ToArray();
    }
}
