using AiDotNet.Attributes;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralRadianceFields.Models;

/// <summary>
/// Generative optimisation of the Gaussians, as DreamGaussian's <c>GaussianModel</c> does it (Tang et al. 2024,
/// section 3.1, and the official <c>gs_renderer.py</c>): random initialisation inside a sphere, per-attribute Adam in
/// the activated parameterisation (centre, log scale, raw quaternion, opacity logit, colour), and adaptive density
/// control (clone, split, prune, opacity reset).
/// </summary>
/// <remarks>
/// <para>
/// While a generative session runs, the scene's parameters live in a structure-of-arrays state that the
/// differentiable rasterizer reads and the optimiser writes; every step writes them back to the Gaussians, so
/// <see cref="RenderImage"/>, <see cref="ToGaussianCloud"/> and serialization always see the current cloud. Only
/// spherical-harmonic degree 0 is supported here (DreamGaussian's <c>sh_degree: 0</c>); the colour of a Gaussian is
/// then <c>max(0, C0 f)</c> with f its DC coefficient, the convention of this class's other renderers (the
/// reference's <c>C0 f + 0.5</c> is the same parameterisation shifted by a constant, which Adam is invariant to).
/// </para>
/// </remarks>
public partial class GaussianSplatting<T>
{
    /// <summary>The degree-0 spherical-harmonic constant this class's renderers use.</summary>
    private const double ShDcConstant = 0.282095;

    /// <summary>Attributes per Gaussian in the optimiser state: 3 centre, 3 log scale, 4 quaternion, 1 opacity, 3 colour.</summary>
    private const int GenerativeAttributeCount = 14;

    private GenerativeState? _generativeState;

    /// <summary>The structure-of-arrays state of a generative session.</summary>
    private sealed class GenerativeState
    {
        public GenerativeState(int count)
        {
            Resize(count);
            Position = new double[3 * count];
            LogScale = new double[3 * count];
            Rotation = new double[4 * count];
            OpacityLogit = new double[count];
            Color = new double[3 * count];
            FirstMoment = new double[GenerativeAttributeCount * count];
            SecondMoment = new double[GenerativeAttributeCount * count];
            GradientAccumulator = new double[count];
            Denominator = new int[count];
            MaxRadii = new double[count];
        }

        public int Count { get; private set; }

        /// <summary>Incremented on every parameter change, so a stale render is rejected.</summary>
        public int Version { get; set; }

        /// <summary>Adam's step count; one per parameter group in the reference, shared by every Gaussian.</summary>
        public int AdamStep { get; set; }

        public double[] Position { get; set; }
        public double[] LogScale { get; set; }
        public double[] Rotation { get; set; }
        public double[] OpacityLogit { get; set; }
        public double[] Color { get; set; }
        public double[] FirstMoment { get; set; }
        public double[] SecondMoment { get; set; }
        public double[] GradientAccumulator { get; set; }
        public int[] Denominator { get; set; }
        public double[] MaxRadii { get; set; }

        public void Resize(int count) => Count = count;
    }

    /// <summary>The colour of a DC coefficient: C0 with spherical harmonics, 1 for raw colours.</summary>
    private double GenerativeColorScale => _useSphericalHarmonics ? ShDcConstant : 1.0;

    /// <summary>
    /// DreamGaussian's <c>Renderer.initialize(num_pts, radius)</c>: <paramref name="count"/> centres uniform in the
    /// ball of <paramref name="radius"/> (<c>r cbrt(u)</c> with a uniform direction), colours <c>SH2RGB(u / 255)</c>,
    /// then <c>create_from_pcd</c>: scale the square root of the mean squared distance to the three nearest
    /// neighbours (floored at 1e-7 squared), identity rotation and opacity <paramref name="initialOpacity"/>.
    /// </summary>
    /// <param name="options">Scene options; spherical harmonics must be off or of degree 0.</param>
    /// <param name="count">Number of Gaussians. DreamGaussian: 5000.</param>
    /// <param name="radius">Ball radius. DreamGaussian: 0.5.</param>
    /// <param name="initialOpacity">Initial opacity. Reference: 0.1.</param>
    /// <param name="random">Random source.</param>
    /// <returns>The scene, with its generative session started.</returns>
    [ScalarKernel("Samples each initial Gaussian and its three nearest neighbours one at a time, as the reference's initializer does.")]
    internal static GaussianSplatting<T> CreateInSphere(
        GaussianSplattingOptions options, int count, double radius, double initialOpacity, Random random)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (count < 1) throw new ArgumentOutOfRangeException(nameof(count), "At least one Gaussian is needed.");
        if (!(radius > 0.0)) throw new ArgumentOutOfRangeException(nameof(radius), "The radius must be positive.");
        if (!(initialOpacity > 0.0 && initialOpacity < 1.0))
            throw new ArgumentOutOfRangeException(nameof(initialOpacity), "The initial opacity must be in (0, 1).");
        if (options.UseSphericalHarmonics && options.ShDegree != 0)
            throw new ArgumentException("Generative optimisation supports spherical-harmonic degree 0 only.", nameof(options));

        var positions = new double[3 * count];
        var colors = new double[3 * count];
        for (int i = 0; i < count; i++)
        {
            double phi = random.NextDouble() * 2.0 * Math.PI;
            double cosTheta = (random.NextDouble() * 2.0) - 1.0;
            double theta = Math.Acos(cosTheta);
            double r = radius * Math.Pow(random.NextDouble(), 1.0 / 3.0);
            positions[3 * i] = r * Math.Sin(theta) * Math.Cos(phi);
            positions[(3 * i) + 1] = r * Math.Sin(theta) * Math.Sin(phi);
            positions[(3 * i) + 2] = r * Math.Cos(theta);
        }

        for (int i = 0; i < 3 * count; i++)
        {
            // shs = U[0, 1) / 255, colors = SH2RGB(shs) = shs C0 + 0.5.
            colors[i] = ((random.NextDouble() / 255.0) * 0.28209479177387814) + 0.5;
        }

        var pointCloud = new Tensor<double>(new[] { count, 3 }, new Vector<double>(positions)).Cast<T>().ToMatrix();
        var colorMatrix = new Tensor<double>(new[] { count, 3 }, new Vector<double>(colors)).Cast<T>().ToMatrix();
        var scene = new GaussianSplatting<T>(options, pointCloud, colorMatrix);
        if (scene._gaussians.Count != count)
            throw new InvalidOperationException($"The scene initialised {scene._gaussians.Count} of {count} Gaussians.");

        var state = scene.ExportGenerativeState();
        double logit = Math.Log(initialOpacity / (1.0 - initialOpacity));
        var meanSquared = MeanSquaredNeighborDistance(positions, count, 3);
        for (int i = 0; i < count; i++)
        {
            double logScale = Math.Log(Math.Sqrt(Math.Max(meanSquared[i], 0.0000001)));
            for (int k = 0; k < 3; k++)
                state.LogScale[(3 * i) + k] = logScale;
            state.Rotation[4 * i] = 1.0;
            state.Rotation[(4 * i) + 1] = 0.0;
            state.Rotation[(4 * i) + 2] = 0.0;
            state.Rotation[(4 * i) + 3] = 0.0;
            state.OpacityLogit[i] = logit;
            for (int k = 0; k < 3; k++)
                state.Color[(3 * i) + k] = colors[(3 * i) + k] / scene.GenerativeColorScale;
        }

        scene._generativeState = state;
        scene.CommitGenerativeState(structureChanged: false);
        return scene;
    }

    /// <summary>
    /// The mean squared distance from each point to its <paramref name="neighbors"/> nearest other points
    /// (<c>simple_knn.distCUDA2</c>), with a uniform grid so the search is near linear.
    /// </summary>
    internal static double[] MeanSquaredNeighborDistance(double[] positions, int count, int neighbors)
    {
        var result = new double[count];
        if (count < 2)
            return result;

        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
        for (int i = 0; i < count; i++)
        {
            minX = Math.Min(minX, positions[3 * i]);
            maxX = Math.Max(maxX, positions[3 * i]);
            minY = Math.Min(minY, positions[(3 * i) + 1]);
            maxY = Math.Max(maxY, positions[(3 * i) + 1]);
            minZ = Math.Min(minZ, positions[(3 * i) + 2]);
            maxZ = Math.Max(maxZ, positions[(3 * i) + 2]);
        }

        double extent = Math.Max(maxX - minX, Math.Max(maxY - minY, maxZ - minZ));
        int cellsPerAxis = Math.Max(1, (int)Math.Pow(count / 2.0, 1.0 / 3.0));
        double cell = extent > 0.0 ? extent / cellsPerAxis : 1.0;
        var grid = new Dictionary<long, List<int>>();
        long Key(int x, int y, int z) => ((long)x * 73856093L) ^ ((long)y * 19349663L) ^ ((long)z * 83492791L);
        int CellOf(double value, double origin) => Math.Min(cellsPerAxis - 1, Math.Max(0, (int)((value - origin) / cell)));
        for (int i = 0; i < count; i++)
        {
            long key = Key(CellOf(positions[3 * i], minX), CellOf(positions[(3 * i) + 1], minY), CellOf(positions[(3 * i) + 2], minZ));
            if (!grid.TryGetValue(key, out var list))
            {
                list = new List<int>();
                grid[key] = list;
            }

            list.Add(i);
        }

        int k = Math.Min(neighbors, count - 1);
        System.Threading.Tasks.Parallel.For(0, count, i =>
        {
            var best = new double[k];
            for (int j = 0; j < k; j++) best[j] = double.MaxValue;
            int cx = CellOf(positions[3 * i], minX), cy = CellOf(positions[(3 * i) + 1], minY), cz = CellOf(positions[(3 * i) + 2], minZ);
            for (int ring = 0; ring <= cellsPerAxis; ring++)
            {
                for (int x = cx - ring; x <= cx + ring; x++)
                {
                    for (int y = cy - ring; y <= cy + ring; y++)
                    {
                        for (int z = cz - ring; z <= cz + ring; z++)
                        {
                            if (Math.Max(Math.Abs(x - cx), Math.Max(Math.Abs(y - cy), Math.Abs(z - cz))) != ring)
                                continue;
                            if (x < 0 || y < 0 || z < 0 || x >= cellsPerAxis || y >= cellsPerAxis || z >= cellsPerAxis)
                                continue;
                            if (!grid.TryGetValue(Key(x, y, z), out var list))
                                continue;
                            foreach (int j in list)
                            {
                                if (j == i) continue;
                                double dx = positions[3 * j] - positions[3 * i];
                                double dy = positions[(3 * j) + 1] - positions[(3 * i) + 1];
                                double dz = positions[(3 * j) + 2] - positions[(3 * i) + 2];
                                double d2 = (dx * dx) + (dy * dy) + (dz * dz);
                                if (d2 >= best[k - 1]) continue;
                                int slot = k - 1;
                                while (slot > 0 && best[slot - 1] > d2)
                                {
                                    best[slot] = best[slot - 1];
                                    slot--;
                                }

                                best[slot] = d2;
                            }
                        }
                    }
                }

                // Every point beyond this ring is at least ring * cell away.
                double reach = ring * cell;
                if (best[k - 1] < double.MaxValue && best[k - 1] <= reach * reach)
                    break;
            }

            double sum = 0.0;
            for (int j = 0; j < k; j++) sum += best[j];
            result[i] = sum / k;
        });

        return result;
    }

    /// <summary>
    /// Starts (or restarts) a generative session from the current Gaussians, with fresh optimiser and densification
    /// statistics.
    /// </summary>
    internal void BeginGenerativeSession()
    {
        if (_useSphericalHarmonics && _shDegree != 0)
            throw new InvalidOperationException("Generative optimisation supports spherical-harmonic degree 0 only.");
        _generativeState = ExportGenerativeState();
    }

    /// <summary>The number of Gaussians in the generative state.</summary>
    internal int GenerativeCount => EnsureGenerativeState().Count;

    private GenerativeState EnsureGenerativeState()
    {
        if (_generativeState is null || _generativeState.Count != _gaussians.Count)
            BeginGenerativeSession();
        return _generativeState ?? throw new InvalidOperationException("The generative state could not be created.");
    }

    [ScalarKernel("Copies each Gaussian's small attribute vectors into the optimiser's arrays one Gaussian at a time.")]
    private GenerativeState ExportGenerativeState()
    {
        int count = _gaussians.Count;
        var state = new GenerativeState(count);
        int basis = GetShBasisCount();
        for (int i = 0; i < count; i++)
        {
            var g = _gaussians[i];
            for (int k = 0; k < 3; k++)
            {
                state.Position[(3 * i) + k] = NumOps.ToDouble(g.Position[k]);
                state.LogScale[(3 * i) + k] = Math.Log(Math.Max(1e-12, Math.Abs(NumOps.ToDouble(g.Scale[k]))));
                state.Color[(3 * i) + k] = NumOps.ToDouble(g.Color[_useSphericalHarmonics ? k * basis : k]);
            }

            for (int k = 0; k < 4; k++)
                state.Rotation[(4 * i) + k] = NumOps.ToDouble(g.Rotation[k]);
            state.OpacityLogit[i] = NumOps.ToDouble(g.Opacity);
        }

        return state;
    }

    /// <summary>
    /// Writes the generative state back to the Gaussians (rebuilding them when densification changed the count) and
    /// refreshes their covariances.
    /// </summary>
    [ScalarKernel("Writes the optimiser's arrays back into each Gaussian's small attribute vectors one Gaussian at a time.")]
    private void CommitGenerativeState(bool structureChanged)
    {
        var state = _generativeState;
        if (state is null)
            return;
        int basis = GetShBasisCount();
        int colorDim = _useSphericalHarmonics ? 3 * basis : 3;
        if (structureChanged || _gaussians.Count != state.Count)
        {
            _gaussians.Clear();
            for (int i = 0; i < state.Count; i++)
                _gaussians.Add(new Gaussian(colorDim, NumOps));
        }

        for (int i = 0; i < state.Count; i++)
        {
            var g = _gaussians[i];
            for (int k = 0; k < 3; k++)
            {
                g.Position[k] = NumOps.FromDouble(state.Position[(3 * i) + k]);
                g.Scale[k] = NumOps.FromDouble(Math.Exp(state.LogScale[(3 * i) + k]));
                g.Color[_useSphericalHarmonics ? k * basis : k] = NumOps.FromDouble(state.Color[(3 * i) + k]);
            }

            for (int k = 0; k < 4; k++)
                g.Rotation[k] = NumOps.FromDouble(state.Rotation[(4 * i) + k]);
            g.Opacity = NumOps.FromDouble(state.OpacityLogit[i]);
            ComputeCovariance(g);
        }

        MarkSpatialIndexDirty();
        if (structureChanged)
            InvalidateParameterCountCache();
    }

    /// <summary>
    /// One Adam step of every attribute with <paramref name="gradients"/>, at the centre learning rate of
    /// <paramref name="iteration"/> (the reference's <c>update_learning_rate(step)</c> then <c>optimizer.step()</c>).
    /// </summary>
    internal void GenerativeAdamStep(GaussianSplatGradients gradients, GaussianAdamSettings settings, int iteration)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (settings is null) throw new ArgumentNullException(nameof(settings));
        var state = EnsureGenerativeState();
        if (gradients.Count != state.Count)
            throw new ArgumentException("The gradients do not match the Gaussians.", nameof(gradients));

        state.AdamStep++;
        int step = state.AdamStep;
        double b1 = settings.Beta1, b2 = settings.Beta2, eps = settings.Epsilon;
        double correction1 = 1.0 - Math.Pow(b1, step);
        double correction2 = 1.0 - Math.Pow(b2, step);
        double positionRate = settings.PositionLearningRate(iteration);

        System.Threading.Tasks.Parallel.For(0, state.Count, i =>
        {
            int m = GenerativeAttributeCount * i;
            for (int k = 0; k < 3; k++)
                state.Position[(3 * i) + k] -= AdamDelta(state, m + k, gradients.Position[(3 * i) + k], positionRate, b1, b2, eps, correction1, correction2);
            for (int k = 0; k < 3; k++)
                state.LogScale[(3 * i) + k] -= AdamDelta(state, m + 3 + k, gradients.LogScale[(3 * i) + k], settings.ScalingLearningRate, b1, b2, eps, correction1, correction2);
            for (int k = 0; k < 4; k++)
                state.Rotation[(4 * i) + k] -= AdamDelta(state, m + 6 + k, gradients.Rotation[(4 * i) + k], settings.RotationLearningRate, b1, b2, eps, correction1, correction2);
            state.OpacityLogit[i] -= AdamDelta(state, m + 10, gradients.OpacityLogit[i], settings.OpacityLearningRate, b1, b2, eps, correction1, correction2);
            for (int k = 0; k < 3; k++)
                state.Color[(3 * i) + k] -= AdamDelta(state, m + 11 + k, gradients.Color[(3 * i) + k], settings.FeatureLearningRate, b1, b2, eps, correction1, correction2);
        });

        state.Version++;
        CommitGenerativeState(structureChanged: false);
    }

    private static double AdamDelta(
        GenerativeState state, int slot, double gradient, double rate, double b1, double b2, double eps, double correction1, double correction2)
    {
        // torch.optim.Adam: p -= lr * m_hat / (sqrt(v_hat) + eps).
        double m = (b1 * state.FirstMoment[slot]) + ((1.0 - b1) * gradient);
        double v = (b2 * state.SecondMoment[slot]) + ((1.0 - b2) * gradient * gradient);
        state.FirstMoment[slot] = m;
        state.SecondMoment[slot] = v;
        return rate * (m / correction1) / (Math.Sqrt(v / correction2) + eps);
    }

    /// <summary>
    /// The reference's per-step bookkeeping inside the densification window: <c>max_radii2D</c> takes the larger
    /// screen radius of every visible Gaussian, and <c>add_densification_stats</c> adds the norm of its NDC centre
    /// gradient and counts the view.
    /// </summary>
    internal void AccumulateDensificationStatistics(GaussianSplatRender render, GaussianSplatGradients gradients)
    {
        if (render is null) throw new ArgumentNullException(nameof(render));
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        var state = EnsureGenerativeState();
        if (render.GaussianCount != state.Count || gradients.Count != state.Count)
            throw new ArgumentException("The render and gradients must match the Gaussians.");
        for (int i = 0; i < state.Count; i++)
        {
            if (render.Radii[i] <= 0)
                continue;
            state.MaxRadii[i] = Math.Max(state.MaxRadii[i], render.Radii[i]);
            double gx = gradients.MeanNdc[2 * i], gy = gradients.MeanNdc[(2 * i) + 1];
            state.GradientAccumulator[i] += Math.Sqrt((gx * gx) + (gy * gy));
            state.Denominator[i]++;
        }
    }

    /// <summary>
    /// The reference's <c>densify_and_prune</c>: clone the small high-gradient Gaussians, split the large ones into
    /// <see cref="GaussianDensificationSettings.SplitCount"/> children sampled from the parent and shrunk by
    /// <c>0.8 N</c>, reset the statistics, then prune the transparent, screen-large and world-large ones.
    /// </summary>
    /// <remarks>
    /// The order is the reference's: densification resets <c>max_radii2D</c> (<c>densification_postfix</c>) before the
    /// prune reads it (the postfix runs even when nothing was cloned or split), so with DreamGaussian's call order the
    /// screen-size rule never removes a Gaussian. New Gaussians start with zero Adam moments, as <c>cat_tensors_to_optimizer</c> gives them.
    /// </remarks>
    [ScalarKernel("Clones, splits and prunes Gaussians one at a time over the optimiser's arrays, as the reference's masked copies do.")]
    internal GaussianDensificationReport DensifyAndPruneGenerative(GaussianDensificationSettings settings, Random random)
    {
        if (settings is null) throw new ArgumentNullException(nameof(settings));
        if (random is null) throw new ArgumentNullException(nameof(random));
        var state = EnsureGenerativeState();
        int original = state.Count;
        var grads = new double[original];
        for (int i = 0; i < original; i++)
            grads[i] = state.Denominator[i] > 0 ? state.GradientAccumulator[i] / state.Denominator[i] : 0.0;

        double denseLimit = settings.PercentDense * settings.SceneExtent;
        var rows = new List<GenerativeRow>(original);
        for (int i = 0; i < original; i++)
            rows.Add(GenerativeRow.From(state, i));

        // densify_and_clone.
        int cloned = 0;
        for (int i = 0; i < original; i++)
        {
            if (grads[i] >= settings.GradientThreshold && rows[i].MaxScale <= denseLimit)
            {
                rows.Add(rows[i].CopyWithFreshMoments());
                cloned++;
            }
        }

        // densify_and_split over the cloned set; clones carry a zero gradient (padded_grad).
        int beforeSplit = rows.Count;
        var splitMask = new bool[beforeSplit];
        int splitCount = 0;
        int children = Math.Max(1, settings.SplitCount);
        for (int i = 0; i < beforeSplit; i++)
        {
            double g = i < original ? grads[i] : 0.0;
            if (g >= settings.GradientThreshold && rows[i].MaxScale > denseLimit)
            {
                splitMask[i] = true;
                splitCount++;
            }
        }

        // The reference repeats the selection N times ([sel; sel; ...]), so children are appended copy by copy.
        for (int copy = 0; copy < children; copy++)
        {
            for (int i = 0; i < beforeSplit; i++)
            {
                if (!splitMask[i]) continue;
                rows.Add(rows[i].SplitChild(random, settings.SplitScaleDivisor * children));
            }
        }

        var survivors = new List<GenerativeRow>(rows.Count);
        for (int i = 0; i < rows.Count; i++)
        {
            if (i < beforeSplit && splitMask[i]) continue;
            survivors.Add(rows[i]);
        }

        // densification_postfix reset the statistics (max_radii2D included) before the prune.
        int pruned = 0;
        var kept = new List<GenerativeRow>(survivors.Count);
        foreach (var row in survivors)
        {
            double opacity = 1.0 / (1.0 + Math.Exp(-row.OpacityLogit));
            bool prune = opacity < settings.MinOpacity;
            if (settings.MaxScreenSize > 0.0)
            {
                // big_points_vs = max_radii2D > max_screen_size is false for every Gaussian here: the postfix zeroed
                // max_radii2D. big_points_ws still applies.
                prune = prune || row.MaxScale > settings.WorldSizePruneFraction * settings.SceneExtent;
            }

            if (prune) pruned++;
            else kept.Add(row);
        }

        WriteRows(state, kept);
        CommitGenerativeState(structureChanged: true);
        return new GaussianDensificationReport(cloned, splitCount, pruned, kept.Count);
    }

    /// <summary>
    /// The reference's <c>prune(min_opacity, extent, max_screen_size)</c> without densification: here the screen-size
    /// rule reads the accumulated radii.
    /// </summary>
    [ScalarKernel("Prunes Gaussians one at a time over the optimiser's arrays.")]
    internal int PruneGenerative(GaussianDensificationSettings settings)
    {
        if (settings is null) throw new ArgumentNullException(nameof(settings));
        var state = EnsureGenerativeState();
        var kept = new List<GenerativeRow>(state.Count);
        int pruned = 0;
        for (int i = 0; i < state.Count; i++)
        {
            var row = GenerativeRow.From(state, i);
            double opacity = 1.0 / (1.0 + Math.Exp(-row.OpacityLogit));
            bool prune = opacity < settings.MinOpacity;
            if (settings.MaxScreenSize > 0.0)
            {
                prune = prune || state.MaxRadii[i] > settings.MaxScreenSize
                    || row.MaxScale > settings.WorldSizePruneFraction * settings.SceneExtent;
            }

            if (prune) pruned++;
            else kept.Add(row);
        }

        WriteRows(state, kept);
        CommitGenerativeState(structureChanged: true);
        return pruned;
    }

    /// <summary>
    /// The reference's <c>reset_opacity</c>: every opacity becomes <c>min(opacity, ceiling)</c> and its Adam moments
    /// restart at zero (<c>replace_tensor_to_optimizer</c>).
    /// </summary>
    internal void ResetGenerativeOpacity(double ceiling)
    {
        if (!(ceiling > 0.0 && ceiling < 1.0)) throw new ArgumentOutOfRangeException(nameof(ceiling));
        var state = EnsureGenerativeState();
        double limit = Math.Log(ceiling / (1.0 - ceiling));
        for (int i = 0; i < state.Count; i++)
        {
            state.OpacityLogit[i] = Math.Min(state.OpacityLogit[i], limit);
            state.FirstMoment[(GenerativeAttributeCount * i) + 10] = 0.0;
            state.SecondMoment[(GenerativeAttributeCount * i) + 10] = 0.0;
        }

        state.Version++;
        CommitGenerativeState(structureChanged: false);
    }

    /// <summary>
    /// The Gaussian centres with their colours, [N, 6] (x, y, z, r, g, b), colours <c>clamp(C0 f, 0, 1)</c>.
    /// </summary>
    internal Tensor<T> ExportColoredCenters()
    {
        var state = EnsureGenerativeState();
        var data = new double[6 * state.Count];
        double scale = GenerativeColorScale;
        for (int i = 0; i < state.Count; i++)
        {
            for (int k = 0; k < 3; k++)
            {
                data[(6 * i) + k] = state.Position[(3 * i) + k];
                data[(6 * i) + 3 + k] = Math.Min(1.0, Math.Max(0.0, scale * state.Color[(3 * i) + k]));
            }
        }

        return new Tensor<double>(new[] { state.Count, 6 }, new Vector<double>(data)).Cast<T>();
    }

    /// <summary>Test access to the densification statistics.</summary>
    internal void SetDensificationStatistics(double[] gradientAccumulator, int[] denominator, double[] maxRadii)
    {
        var state = EnsureGenerativeState();
        if (gradientAccumulator.Length != state.Count || denominator.Length != state.Count || maxRadii.Length != state.Count)
            throw new ArgumentException("The statistics must have one entry per Gaussian.");
        Array.Copy(gradientAccumulator, state.GradientAccumulator, state.Count);
        Array.Copy(denominator, state.Denominator, state.Count);
        Array.Copy(maxRadii, state.MaxRadii, state.Count);
    }

    /// <summary>Test and export access to the parameters: centre, log scale, raw quaternion, opacity logit, DC colour.</summary>
    internal (double[] Position, double[] LogScale, double[] Rotation, double[] OpacityLogit, double[] Color) GetGenerativeParameters()
    {
        var state = EnsureGenerativeState();
        return ((double[])state.Position.Clone(), (double[])state.LogScale.Clone(), (double[])state.Rotation.Clone(),
            (double[])state.OpacityLogit.Clone(), (double[])state.Color.Clone());
    }

    /// <summary>Replaces the parameters (test fixtures and loaded checkpoints); the optimiser state restarts.</summary>
    internal void SetGenerativeParameters(double[] position, double[] logScale, double[] rotation, double[] opacityLogit, double[] color)
    {
        if (position is null) throw new ArgumentNullException(nameof(position));
        if (logScale is null) throw new ArgumentNullException(nameof(logScale));
        if (rotation is null) throw new ArgumentNullException(nameof(rotation));
        if (opacityLogit is null) throw new ArgumentNullException(nameof(opacityLogit));
        if (color is null) throw new ArgumentNullException(nameof(color));
        int count = opacityLogit.Length;
        if (position.Length != 3 * count || logScale.Length != 3 * count || rotation.Length != 4 * count || color.Length != 3 * count)
            throw new ArgumentException("The parameter arrays disagree on the Gaussian count.");
        var state = new GenerativeState(count)
        {
            Position = (double[])position.Clone(),
            LogScale = (double[])logScale.Clone(),
            Rotation = (double[])rotation.Clone(),
            OpacityLogit = (double[])opacityLogit.Clone(),
            Color = (double[])color.Clone(),
        };
        _generativeState = state;
        CommitGenerativeState(structureChanged: true);
    }

    private static void WriteRows(GenerativeState state, List<GenerativeRow> rows)
    {
        int count = rows.Count;
        state.Resize(count);
        state.Position = new double[3 * count];
        state.LogScale = new double[3 * count];
        state.Rotation = new double[4 * count];
        state.OpacityLogit = new double[count];
        state.Color = new double[3 * count];
        state.FirstMoment = new double[GenerativeAttributeCount * count];
        state.SecondMoment = new double[GenerativeAttributeCount * count];
        state.GradientAccumulator = new double[count];
        state.Denominator = new int[count];
        state.MaxRadii = new double[count];
        for (int i = 0; i < count; i++)
        {
            var row = rows[i];
            Array.Copy(row.Position, 0, state.Position, 3 * i, 3);
            Array.Copy(row.LogScale, 0, state.LogScale, 3 * i, 3);
            Array.Copy(row.Rotation, 0, state.Rotation, 4 * i, 4);
            state.OpacityLogit[i] = row.OpacityLogit;
            Array.Copy(row.Color, 0, state.Color, 3 * i, 3);
            Array.Copy(row.FirstMoment, 0, state.FirstMoment, GenerativeAttributeCount * i, GenerativeAttributeCount);
            Array.Copy(row.SecondMoment, 0, state.SecondMoment, GenerativeAttributeCount * i, GenerativeAttributeCount);
        }

        state.Version++;
    }

    /// <summary>One Gaussian's parameters and Adam moments, for densification's gather and scatter.</summary>
    private sealed class GenerativeRow
    {
        public double[] Position { get; private set; } = new double[3];
        public double[] LogScale { get; private set; } = new double[3];
        public double[] Rotation { get; private set; } = new double[4];
        public double OpacityLogit { get; private set; }
        public double[] Color { get; private set; } = new double[3];
        public double[] FirstMoment { get; private set; } = new double[GenerativeAttributeCount];
        public double[] SecondMoment { get; private set; } = new double[GenerativeAttributeCount];

        public double MaxScale => Math.Exp(Math.Max(LogScale[0], Math.Max(LogScale[1], LogScale[2])));

        public static GenerativeRow From(GenerativeState state, int i)
        {
            var row = new GenerativeRow { OpacityLogit = state.OpacityLogit[i] };
            Array.Copy(state.Position, 3 * i, row.Position, 0, 3);
            Array.Copy(state.LogScale, 3 * i, row.LogScale, 0, 3);
            Array.Copy(state.Rotation, 4 * i, row.Rotation, 0, 4);
            Array.Copy(state.Color, 3 * i, row.Color, 0, 3);
            Array.Copy(state.FirstMoment, GenerativeAttributeCount * i, row.FirstMoment, 0, GenerativeAttributeCount);
            Array.Copy(state.SecondMoment, GenerativeAttributeCount * i, row.SecondMoment, 0, GenerativeAttributeCount);
            return row;
        }

        public GenerativeRow CopyWithFreshMoments() => new()
        {
            Position = (double[])Position.Clone(),
            LogScale = (double[])LogScale.Clone(),
            Rotation = (double[])Rotation.Clone(),
            OpacityLogit = OpacityLogit,
            Color = (double[])Color.Clone(),
        };

        /// <summary>
        /// A split child: centre <c>R(q) n + mu</c> with <c>n ~ N(0, diag(s))</c> (the reference's
        /// <c>torch.normal(mean=0, std=stds)</c>), scale <c>s / divisor</c>, the rest copied, fresh moments.
        /// </summary>
        public GenerativeRow SplitChild(Random random, double divisor)
        {
            var child = CopyWithFreshMoments();
            double w = Rotation[0], x = Rotation[1], y = Rotation[2], z = Rotation[3];
            double norm = Math.Max(1e-12, Math.Sqrt((w * w) + (x * x) + (y * y) + (z * z)));
            w /= norm;
            x /= norm;
            y /= norm;
            z /= norm;
            var r = new[]
            {
                1.0 - (2.0 * ((y * y) + (z * z))), 2.0 * ((x * y) - (w * z)), 2.0 * ((x * z) + (w * y)),
                2.0 * ((x * y) + (w * z)), 1.0 - (2.0 * ((x * x) + (z * z))), 2.0 * ((y * z) - (w * x)),
                2.0 * ((x * z) - (w * y)), 2.0 * ((y * z) + (w * x)), 1.0 - (2.0 * ((x * x) + (y * y))),
            };
            var sample = new double[3];
            for (int k = 0; k < 3; k++)
                sample[k] = StandardNormal(random) * Math.Exp(LogScale[k]);
            for (int a = 0; a < 3; a++)
                child.Position[a] = Position[a] + (r[3 * a] * sample[0]) + (r[(3 * a) + 1] * sample[1]) + (r[(3 * a) + 2] * sample[2]);
            double shift = Math.Log(divisor);
            for (int k = 0; k < 3; k++)
                child.LogScale[k] = LogScale[k] - shift;
            return child;
        }
    }

    /// <summary>A standard normal sample (Box-Muller).</summary>
    private static double StandardNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
