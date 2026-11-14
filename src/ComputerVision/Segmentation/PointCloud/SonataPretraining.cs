using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>Pointcept's <c>CosineScheduler</c>: a linear warm-up then a half cosine to the final value.</summary>
internal sealed class SonataCosineSchedule
{
    private readonly double[] _schedule;
    private readonly double _final;

    internal SonataCosineSchedule(double baseValue, double finalValue, int totalIterations, double startValue = 0.0,
        int warmupIterations = 0)
    {
        if (totalIterations < 1) throw new ArgumentOutOfRangeException(nameof(totalIterations));
        warmupIterations = Math.Max(0, Math.Min(warmupIterations, totalIterations));
        _final = finalValue;
        _schedule = new double[totalIterations];
        // np.linspace(start, base, warmup): endpoint included.
        for (int i = 0; i < warmupIterations; i++)
            _schedule[i] = warmupIterations == 1 ? startValue : startValue + (baseValue - startValue) * i / (warmupIterations - 1);
        int length = totalIterations - warmupIterations;
        for (int i = 0; i < length; i++)
            _schedule[warmupIterations + i] = finalValue + 0.5 * (baseValue - finalValue) * (1 + Math.Cos(Math.PI * i / length));
    }

    internal double Get(int iteration) => iteration >= _schedule.Length ? _final : _schedule[Math.Max(0, iteration)];
}

/// <summary>Several crops of one scene, concatenated, with the view each point belongs to.</summary>
internal sealed class SonataViewSet
{
    internal SonataViewSet(double[] coordinates, double[] origin, float[] features, int width, int[] view, int viewCount)
    {
        Coordinates = coordinates;
        Origin = origin;
        Features = features;
        Width = width;
        View = view;
        ViewCount = viewCount;
    }

    /// <summary>Augmented coordinates, <c>[points * 3]</c>.</summary>
    internal double[] Coordinates { get; }

    /// <summary>Coordinates before augmentation, used to match points across views.</summary>
    internal double[] Origin { get; }

    /// <summary>Per-point features, <c>[points * width]</c>: augmented xyz then the remaining channels.</summary>
    internal float[] Features { get; }

    internal int Width { get; }

    internal int[] View { get; }

    internal int ViewCount { get; }

    internal int Count => View.Length;
}

/// <summary>
/// The pieces of Sonata's self-distillation that are data preparation rather than network:
/// multi-view cropping and augmentation (<c>MultiViewGenerator</c>), patch masking
/// (<c>generate_mask</c>), cross-view matching (<c>match_neighbour</c>) and the EMA teacher update.
/// </summary>
internal static class SonataPretraining
{
    /// <summary>
    /// Builds the global and local crops of a <c>[points, width]</c> scene whose first three channels
    /// are xyz, following Pointcept's <c>MultiViewGenerator</c>: the major global view is centred on
    /// a random point, the other global views on points of the major view, and the local views on
    /// points of the major view not yet covered by an earlier local view.
    /// </summary>
    internal static (SonataViewSet Global, SonataViewSet Local) GenerateViews(float[] scene, int width, Random random,
        int globalViews, double globalMin, double globalMax, int localViews, double localMin, double localMax, int maxSize)
    {
        int n = scene.Length / width;
        var coordinates = new double[n * 3];
        for (int i = 0; i < n; i++)
        {
            coordinates[i * 3] = scene[i * width];
            coordinates[i * 3 + 1] = scene[i * width + 1];
            coordinates[i * 3 + 2] = scene[i * width + 2];
        }

        int limit = Math.Min(maxSize, n);
        var major = Crop(coordinates, coordinates, random.Next(n), globalMin, globalMax, limit, random);
        var globals = new List<int[]> { major };
        for (int v = 1; v < globalViews; v++)
            globals.Add(Crop(coordinates, coordinates, major[random.Next(major.Length)], globalMin, globalMax, limit, random));

        var covered = new HashSet<int>();
        var locals = new List<int[]>();
        for (int v = 0; v < localViews; v++)
        {
            if (covered.Count >= major.Length) covered.Clear();
            var candidates = new List<int>();
            foreach (int index in major)
            {
                if (!covered.Contains(index)) candidates.Add(index);
            }

            var crop = Crop(coordinates, coordinates, candidates[random.Next(candidates.Count)], localMin, localMax, limit, random);
            locals.Add(crop);
            var majorSet = new HashSet<int>(major);
            foreach (int index in crop)
            {
                if (majorSet.Contains(index)) covered.Add(index);
            }
        }

        return (Assemble(scene, width, coordinates, globals, random), Assemble(scene, width, coordinates, locals, random));
    }

    /// <summary>The <c>size</c> nearest points to the centre, <c>size = U(min, max) * limit</c>.</summary>
    private static int[] Crop(double[] coordinates, double[] reference, int center, double min, double max, int limit,
        Random random)
    {
        int n = coordinates.Length / 3;
        int size = 0;
        for (int attempt = 0; attempt < 10 && size <= 0; attempt++)
            size = (int)((min + (max - min) * random.NextDouble()) * limit);
        if (size <= 0) size = Math.Max(10, (int)(max * limit));
        size = Math.Min(size, n);
        double cx = reference[center * 3], cy = reference[center * 3 + 1], cz = reference[center * 3 + 2];
        var distance = new double[n];
        var order = new int[n];
        for (int i = 0; i < n; i++)
        {
            double dx = coordinates[i * 3] - cx, dy = coordinates[i * 3 + 1] - cy, dz = coordinates[i * 3 + 2] - cz;
            distance[i] = dx * dx + dy * dy + dz * dz;
            order[i] = i;
        }

        Array.Sort(distance, order);
        var crop = new int[size];
        Array.Copy(order, crop, size);
        return crop;
    }

    /// <summary>Applies the per-view geometric augmentation and concatenates the views.</summary>
    private static SonataViewSet Assemble(float[] scene, int width, double[] coordinates, List<int[]> views, Random random)
    {
        int total = 0;
        foreach (var view in views) total += view.Length;
        var augmented = new double[total * 3];
        var origin = new double[total * 3];
        var features = new float[total * width];
        var viewOf = new int[total];
        int row = 0;
        for (int v = 0; v < views.Count; v++)
        {
            var index = views[v];
            var transform = RandomTransform(random, out double scale);
            // CenterShift(apply_z=True): centre x and y on the box midpoint, z on its floor.
            double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
            double maxX = double.MinValue, maxY = double.MinValue;
            foreach (int i in index)
            {
                minX = Math.Min(minX, coordinates[i * 3]);
                maxX = Math.Max(maxX, coordinates[i * 3]);
                minY = Math.Min(minY, coordinates[i * 3 + 1]);
                maxY = Math.Max(maxY, coordinates[i * 3 + 1]);
                minZ = Math.Min(minZ, coordinates[i * 3 + 2]);
            }

            double sx = (minX + maxX) / 2, sy = (minY + maxY) / 2;
            double brightness = random.NextDouble() < 0.95 ? (random.NextDouble() - 0.5) * 2 * 0.05 : 0.0;
            foreach (int i in index)
            {
                double x = (coordinates[i * 3] - sx) * scale, y = (coordinates[i * 3 + 1] - sy) * scale,
                    z = (coordinates[i * 3 + 2] - minZ) * scale;
                var p = Apply(transform, x, y, z);
                for (int a = 0; a < 3; a++)
                {
                    // RandomJitter(sigma=0.005, clip=0.02).
                    double jitter = Math.Max(-0.02, Math.Min(0.02, 0.005 * Gaussian(random)));
                    augmented[row * 3 + a] = p[a] + jitter;
                    origin[row * 3 + a] = coordinates[i * 3 + a];
                    features[row * width + a] = (float)augmented[row * 3 + a];
                }

                for (int c = 3; c < width; c++) features[row * width + c] = scene[i * width + c];
                if (width >= 6)
                {
                    // ChromaticTranslation on colours normalised to [-1, 1].
                    for (int c = 3; c < 6; c++)
                        features[row * width + c] = (float)Math.Max(-1.0, Math.Min(1.0, features[row * width + c] + brightness));
                }

                if (width >= 9)
                {
                    // Normals follow the rotation and flips, not the scale or shift.
                    var normal = Apply(transform, scene[i * width + 6], scene[i * width + 7], scene[i * width + 8]);
                    for (int a = 0; a < 3; a++) features[row * width + 6 + a] = (float)normal[a];
                }

                viewOf[row] = v;
                row++;
            }
        }

        return new SonataViewSet(augmented, origin, features, width, viewOf, views.Count);
    }

    /// <summary>
    /// RandomRotate about z (angle U(-1, 1) * pi, p = 0.8), about x and y (U(-1/64, 1/64) * pi,
    /// p = 0.8 each), RandomFlip of x and y (p = 0.5 each); also draws RandomScale U(0.9, 1.1).
    /// </summary>
    private static double[] RandomTransform(Random random, out double scale)
    {
        scale = 0.9 + 0.2 * random.NextDouble();
        double az = random.NextDouble() < 0.8 ? (random.NextDouble() * 2 - 1) * Math.PI : 0.0;
        double ax = random.NextDouble() < 0.8 ? (random.NextDouble() * 2 - 1) / 64 * Math.PI : 0.0;
        double ay = random.NextDouble() < 0.8 ? (random.NextDouble() * 2 - 1) / 64 * Math.PI : 0.0;
        var m = Multiply(Rotation(ay, 1), Multiply(Rotation(ax, 0), Rotation(az, 2)));
        double fx = random.NextDouble() < 0.5 ? -1 : 1, fy = random.NextDouble() < 0.5 ? -1 : 1;
        for (int c = 0; c < 3; c++)
        {
            m[c] *= fx;
            m[3 + c] *= fy;
        }

        return m;
    }

    private static double[] Rotation(double angle, int axis)
    {
        double c = Math.Cos(angle), s = Math.Sin(angle);
        return axis switch
        {
            0 => new[] { 1, 0, 0, 0, c, -s, 0, s, c },
            1 => new[] { c, 0, s, 0, 1, 0, -s, 0, c },
            _ => new[] { c, -s, 0, s, c, 0, 0, 0, 1 }
        };
    }

    private static double[] Multiply(double[] a, double[] b)
    {
        var r = new double[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }

        return r;
    }

    private static double[] Apply(double[] m, double x, double y, double z)
        => new[] { m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z };

    internal static double Gaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Sonata <c>generate_mask</c>: groups each view's points into cubes of <paramref name="maskSize"/>
    /// metres and masks a random <paramref name="maskRatio"/> of the cubes.
    /// </summary>
    internal static bool[] GenerateMask(SonataViewSet views, double maskSize, double maskRatio, Random random)
    {
        int n = views.Count;
        var min = new double[views.ViewCount * 3];
        for (int i = 0; i < min.Length; i++) min[i] = double.MaxValue;
        for (int i = 0; i < n; i++)
        {
            for (int a = 0; a < 3; a++)
                min[views.View[i] * 3 + a] = Math.Min(min[views.View[i] * 3 + a], views.Coordinates[i * 3 + a]);
        }

        var patches = new Dictionary<(int, long, long, long), int>();
        var cluster = new int[n];
        for (int i = 0; i < n; i++)
        {
            int v = views.View[i];
            var key = (v, Cell(views.Coordinates[i * 3], min[v * 3], maskSize),
                Cell(views.Coordinates[i * 3 + 1], min[v * 3 + 1], maskSize),
                Cell(views.Coordinates[i * 3 + 2], min[v * 3 + 2], maskSize));
            if (!patches.TryGetValue(key, out int id))
            {
                id = patches.Count;
                patches[key] = id;
            }

            cluster[i] = id;
        }

        int count = patches.Count;
        var permutation = PointTransformerV3Encoder<double>.RandomPermutation(random, count);
        var masked = new bool[count];
        int maskCount = (int)(count * maskRatio);
        for (int j = 0; j < maskCount; j++) masked[permutation[j]] = true;
        var mask = new bool[n];
        for (int i = 0; i < n; i++) mask[i] = masked[cluster[i]];
        return mask;
    }

    private static long Cell(double value, double min, double size) => (long)Math.Floor((value - min) / size);

    /// <summary>
    /// Sonata <c>match_neighbour</c>: for every query point, the nearest reference point (by original
    /// coordinates) in the reference view paired with the query's view, kept when closer than
    /// <paramref name="radius"/>. Returns (query, reference) index pairs.
    /// </summary>
    /// <param name="queryOrigin">Query coordinates before augmentation.</param>
    /// <param name="queryView">The view of each query point.</param>
    /// <param name="referenceOrigin">Reference coordinates before augmentation.</param>
    /// <param name="referenceView">The view of each reference point.</param>
    /// <param name="pairedView">For each query view, the reference view it is matched against.</param>
    /// <param name="radius">Largest accepted distance (Sonata <c>match_max_r</c>).</param>
    internal static (int[] Query, int[] Reference) MatchNeighbour(double[] queryOrigin, int[] queryView,
        double[] referenceOrigin, int[] referenceView, int[] pairedView, double radius)
    {
        var cells = new Dictionary<(int, long, long, long), List<int>>();
        for (int j = 0; j < referenceView.Length; j++)
        {
            var key = (referenceView[j], Cell(referenceOrigin[j * 3], 0, radius), Cell(referenceOrigin[j * 3 + 1], 0, radius),
                Cell(referenceOrigin[j * 3 + 2], 0, radius));
            if (!cells.TryGetValue(key, out var list))
            {
                list = new List<int>();
                cells[key] = list;
            }

            list.Add(j);
        }

        var query = new List<int>();
        var reference = new List<int>();
        double r2 = radius * radius;
        for (int i = 0; i < queryView.Length; i++)
        {
            int view = pairedView[queryView[i]];
            double x = queryOrigin[i * 3], y = queryOrigin[i * 3 + 1], z = queryOrigin[i * 3 + 2];
            long cx = Cell(x, 0, radius), cy = Cell(y, 0, radius), cz = Cell(z, 0, radius);
            int best = -1;
            double bestDistance = double.MaxValue;
            for (long dx = -1; dx <= 1; dx++)
            {
                for (long dy = -1; dy <= 1; dy++)
                {
                    for (long dz = -1; dz <= 1; dz++)
                    {
                        if (!cells.TryGetValue((view, cx + dx, cy + dy, cz + dz), out var list)) continue;
                        foreach (int j in list)
                        {
                            double ex = referenceOrigin[j * 3] - x, ey = referenceOrigin[j * 3 + 1] - y,
                                ez = referenceOrigin[j * 3 + 2] - z;
                            double d = ex * ex + ey * ey + ez * ez;
                            if (d < bestDistance || (d == bestDistance && j < best))
                            {
                                bestDistance = d;
                                best = j;
                            }
                        }
                    }
                }
            }

            if (best >= 0 && bestDistance < r2)
            {
                query.Add(i);
                reference.Add(best);
            }
        }

        return (query.ToArray(), reference.ToArray());
    }

    /// <summary>The EMA teacher update <c>teacher = m * teacher + (1 - m) * student</c>, without gradient.</summary>
    internal static void UpdateTeacher<T>(IReadOnlyList<Tensor<T>> teacher, IReadOnlyList<Tensor<T>> student, double momentum)
    {
        if (teacher.Count != student.Count) throw new ArgumentException("Teacher and student must have the same parameters.");
        var engine = AiDotNetEngine.Current;
        var weight = MathHelper.GetNumericOperations<T>().FromDouble(1.0 - momentum);
        using var noGrad = new NoGradScope<T>();
        for (int i = 0; i < teacher.Count; i++) engine.TensorLerpInPlace(teacher[i], student[i], weight);
    }

    /// <summary>Copies the student's values into the teacher.</summary>
    internal static void CopyInto<T>(IReadOnlyList<Tensor<T>> teacher, IReadOnlyList<Tensor<T>> student)
    {
        if (teacher.Count != student.Count) throw new ArgumentException("Teacher and student must have the same parameters.");
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        for (int i = 0; i < teacher.Count; i++) engine.TensorCopyInPlace(teacher[i], student[i]);
    }
}
