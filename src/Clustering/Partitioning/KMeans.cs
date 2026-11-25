using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// K-Means clustering algorithm implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// K-Means is one of the most widely used clustering algorithms. It partitions
/// n observations into k clusters by minimizing within-cluster variance (inertia).
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Initialize k cluster centers (randomly or using k-means++)
/// 2. Assign each point to the nearest center
/// 3. Update each center as the mean of its assigned points
/// 4. Repeat steps 2-3 until convergence or max iterations
/// </para>
/// <para>
/// Convergence follows scikit-learn's Lloyd iteration: it stops when an assignment leaves every label unchanged, or
/// when the sum over clusters of each center's squared shift is at most
/// <see cref="Clustering.Options.ClusteringOptions{T}.Tolerance"/> times the mean per-feature variance of the data (so
/// the default 1e-4 means what it means in scikit-learn). Unless the labels were already unchanged, a final assignment
/// makes the labels those of the returned centers. <see cref="NumIterations"/> counts the update steps run.
/// </para>
/// <para><b>For Beginners:</b> K-Means finds k groups in your data by:
/// - Starting with k "center" points
/// - Grouping data points by their closest center
/// - Moving each center to the middle of its group
/// - Repeating until centers stop moving
///
/// It works best when:
/// - You know how many clusters to look for
/// - Clusters are roughly spherical and similar in size
/// - Data doesn't have many outliers
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new KMeans&lt;double&gt;(new KMeansOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // one cluster index per row
/// var assignments = result.Predict(dataMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Some methods for classification and analysis of multivariate observations", "https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fifth-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/0/bsmsp/1200512992", Year = 1967, Authors = "James MacQueen")]
public partial class KMeans<T> : ClusteringBase<T>
{
    private readonly KMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Random _random;
    private int _numIterations;

    /// <summary>
    /// Initializes a new KMeans instance with the specified options.
    /// </summary>
    /// <param name="options">The KMeans configuration options.</param>
    public KMeans(KMeansOptions<T>? options = null)
        : base(options ??= new KMeansOptions<T>())
    {
        _options = options;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Set distance metric if not specified
        if (_options.DistanceMetric is null)
        {
            _options.DistanceMetric = new EuclideanDistance<T>();
        }

        // Set NumClusters from options
        NumClusters = _options.NumClusters;
    }

    /// <summary>
    /// Gets the number of iterations from the last fit.
    /// </summary>
    public int NumIterations => _numIterations;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (KMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputData(x);
        NumFeatures = x.Columns;

        // The data, its row norms and the convergence threshold serve every initialization.
        var data = new KMeansData(x, UsesEuclideanDistance, AbsoluteTolerance(x));

        // Run multiple initializations and keep the best (the first one with the lowest inertia)
        int numInits = _options.InitMethod == KMeansInitMethod.Custom ? 1 : _options.NumInitializations;
        var runs = new List<(Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations)>(numInits);
        for (int init = 0; init < numInits; init++)
        {
            runs.Add(FitSingle(data));
        }

        Matrix<T>? bestCenters = null;
        Vector<T>? bestLabels = null;
        T bestInertia = NumOps.MaxValue;
        if (runs.Count > 0)
        {
            int best = ClusteringTensorOps.ArgMin(Vector<T>.FromEnumerable(runs.Select(run => run.Inertia)));
            bestCenters = runs[best].Centers;
            bestLabels = runs[best].Labels;
            bestInertia = runs[best].Inertia;
            _numIterations = runs[best].Iterations;
        }

        ClusterCenters = bestCenters;
        Labels = bestLabels;
        Inertia = bestInertia;
        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null) return new Vector<T>(x.Rows);
        return AssignLabels(new KMeansData(x, UsesEuclideanDistance, NumOps.Zero), ClusterCenters);
    }

    /// <inheritdoc />
    public override Matrix<T> Transform(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null) return new Matrix<T>(x.Rows, NumClusters);
        return ComputeDistancesToCenters(x, ClusterCenters);
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// True when distances are plain Euclidean, so the assignment can use the batched
    /// <c>||x||^2 - 2 x.c + ||c||^2</c> form. A custom metric, including a subclass of
    /// <see cref="EuclideanDistance{T}"/>, goes through the metric itself.
    /// </summary>
    private bool UsesEuclideanDistance =>
        _options.DistanceMetric is null || _options.DistanceMetric.GetType() == typeof(EuclideanDistance<T>);

    /// <summary>
    /// The data of one fit or prediction: the matrix, its rank-2 tensor, each row's squared norm (Euclidean only) and the
    /// absolute convergence threshold.
    /// </summary>
    private sealed class KMeansData
    {
        public KMeansData(Matrix<T> x, bool euclidean, T tolerance)
        {
            X = x;
            Raw = ClusteringTensorOps.ToTensor(x);
            Values = Raw;
            Euclidean = euclidean;
            Tolerance = tolerance;
            if (euclidean && x.Rows > 0)
            {
                // The norm expansion subtracts numbers of the size of ||x||^2 to get a squared distance, so it is
                // evaluated on the data shifted to its column means: distances are unchanged by the shift, and the
                // cancellation error then scales with the spread of the data instead of its offset from the
                // origin (in single precision, unit-spread clusters around 1e4 are otherwise unresolvable).
                var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
                Origin = ClusteringTensorOps.ColumnMeans(x);
                var centered = ClusteringTensorOps.SubtractRow(x, Origin);
                Scale = OverflowSafeScale(engine.TensorMaxValue(engine.TensorAbs(centered)), x.Columns);
                Values = engine.TensorDivideScalar(centered, Scale);
                RowNorms = engine.ReduceSum(engine.TensorSquare(Values), new[] { 1 }, true);
            }
        }

        /// <summary>
        /// One when every squared norm of the centered data fits in <typeparamref name="T"/>; otherwise the power of
        /// two at or above the largest centered magnitude. In half precision a centered coordinate above ~256
        /// already squares past 65504, which would make the norm expansion inf or NaN. Dividing by a power of two is
        /// exact, and the nearest center and the k-means++ sampling weights are unchanged by a positive scale.
        /// </summary>
        private static T OverflowSafeScale(T largestMagnitude, int columns)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            double magnitude = numOps.ToDouble(largestMagnitude);
            double bound = 4.0 * magnitude * magnitude * Math.Max(1, columns);
            double roundTrip = numOps.ToDouble(numOps.FromDouble(bound));
            if (magnitude <= 0 || double.IsNaN(magnitude) || double.IsInfinity(magnitude)
                || (!double.IsInfinity(roundTrip) && !double.IsNaN(roundTrip)))
            {
                return numOps.One;
            }

            return numOps.FromDouble(Math.Pow(2.0, Math.Ceiling(Math.Log(magnitude, 2.0))));
        }

        /// <summary>The column means subtracted from <see cref="Values"/>; null unless Euclidean.</summary>
        public Vector<T>? Origin { get; }

        /// <summary>The power of two the centered data is divided by (one unless its squares would overflow T).</summary>
        public T Scale { get; } = MathHelper.GetNumericOperations<T>().One;

        public Matrix<T> X { get; }

        /// <summary><see cref="X"/> as a [rows, columns] tensor, copied once per fit instead of once per iteration.</summary>
        public Tensor<T> Raw { get; }

        /// <summary>
        /// The data as a [rows, columns] tensor, shifted to its column means (and divided by <see cref="Scale"/>)
        /// when Euclidean.
        /// </summary>
        public Tensor<T> Values { get; }

        public bool Euclidean { get; }

        public T Tolerance { get; }

        /// <summary>Each row's squared norm as [rows, 1]; null unless Euclidean.</summary>
        public Tensor<T>? RowNorms { get; }
    }

    private (Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations) FitSingle(KMeansData data)
    {
        var x = data.X;
        var centers = InitializeCenters(data);

        // Labels stay cluster indices (int) between iterations: converting them to and from T every iteration cost
        // more than the assignment itself.
        int[] labels = new int[x.Rows];
        int[]? previousLabels = null;
        bool unchanged = false;
        int iterations = 0;
        while (iterations < _options.MaxIterations)
        {
            // Assignment step: assign each point to nearest center
            labels = AssignLabelIndices(data, centers);
            iterations++;

            // Update step: compute new centers, and whether the sum of the centers' squared shifts is within tolerance
            var newCenters = UpdateCenters(data, labels, centers);
            bool shiftWithinTolerance = HasConverged(data, centers, newCenters);

            centers = newCenters;
            if (previousLabels is not null && labels.AsSpan().SequenceEqual(previousLabels))
            {
                unchanged = true;
                break;
            }

            if (shiftWithinTolerance) break;
            previousLabels = labels;
        }

        // Label every row by the centers returned, unless the last assignment already did.
        if (iterations > 0 && !unchanged)
            labels = AssignLabelIndices(data, centers);

        var labelVector = ClusteringTensorOps.FromIntArray<T>(labels);
        T inertia = iterations > 0 ? ComputeInertia(data, labelVector, centers) : NumOps.Zero;
        return (centers, labelVector, inertia, iterations);
    }

    /// <summary>
    /// The convergence threshold on the summed squared center shift: the tolerance times the mean over features of
    /// each feature's (population) variance, as scikit-learn's KMeans.
    /// </summary>
    private T AbsoluteTolerance(Matrix<T> x)
    {
        if (x.Rows == 0 || x.Columns == 0) return NumOps.Zero;

        // Mean over all entries of the squared deviation from the column mean = mean of the per-feature variances.
        var deviations = ClusteringTensorOps.SubtractRow(x, ClusteringTensorOps.ColumnMeans(x));
        T meanVariance = Engine.TensorMean(Engine.TensorSquare(deviations));
        return NumOps.Multiply(NumOps.FromDouble(_options.Tolerance), meanVariance);
    }

    private Matrix<T> InitializeCenters(KMeansData data)
    {
        return _options.InitMethod switch
        {
            KMeansInitMethod.Custom when _options.InitialCenters is not null => _options.InitialCenters,
            KMeansInitMethod.KMeansPlusPlus => InitializeKMeansPlusPlus(data),
            _ => InitializeRandom(data.X)
        };
    }

    private Matrix<T> InitializeRandom(Matrix<T> x)
    {
        var selectedIndices = new HashSet<int>();
        var order = new int[_options.NumClusters];

        for (int k = 0; k < _options.NumClusters; k++)
        {
            int idx;
            do
            {
                idx = _random.Next(x.Rows);
            } while (selectedIndices.Contains(idx));

            selectedIndices.Add(idx);
            order[k] = idx;
        }

        return ClusteringTensorOps.SelectRows(x, order);
    }

    /// <summary>
    /// Candidates per k-means++ center: <see cref="KMeansOptions{T}.KMeansPlusPlusLocalTrials"/>, or scikit-learn's
    /// <c>2 + floor(ln k)</c>.
    /// </summary>
    private int LocalTrials(int clusters)
    {
        if (_options.KMeansPlusPlusLocalTrials is int trials)
        {
            if (trials < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(KMeansOptions<T>.KMeansPlusPlusLocalTrials), trials,
                    "k-means++ needs at least one candidate per center.");
            }

            return trials;
        }

        return 2 + (int)Math.Log(clusters);
    }

    /// <summary>
    /// Greedy k-means++ (Arthur and Vassilvitskii, 2007; scikit-learn's default): each new center is the best of
    /// <see cref="LocalTrials"/> rows drawn with probability proportional to the squared distance to the nearest chosen
    /// center, the one whose potential (the total of those squared distances once it is added) is lowest, the first drawn
    /// on a tie. One trial is classic k-means++, which takes the drawn row.
    /// </summary>
    private Matrix<T> InitializeKMeansPlusPlus(KMeansData data)
    {
        var x = data.X;
        int rows = x.Rows;
        var chosen = new int[_options.NumClusters];

        // Choose first center randomly
        chosen[0] = _random.Next(rows);

        int trials = LocalTrials(_options.NumClusters);
#if NET7_0_OR_GREATER
        if (typeof(T) == typeof(double) && data.Euclidean && data.RowNorms is not null && data.Values.IsContiguous
            && data.RowNorms.IsContiguous)
        {
            KMeansPlusPlusOfDoubles(data, chosen, trials);
            return ClusteringTensorOps.SelectRows(x, chosen);
        }
#endif
        Vector<T>? minDistances = null;
        for (int k = 1; k < _options.NumClusters; k++)
        {
            // The first center sets the minimum squared distances; each later pick lowers them below
            if (minDistances is null)
            {
                minDistances = Engine.Reshape(
                    SquaredDistancesToRows(data, new[] { chosen[k - 1] }), new[] { rows }).ToVector();
            }

            // Select next center with probability proportional to distance squared, keeping the best of the trials
            var candidates = ClusteringTensorOps.SampleProportionalMany(minDistances, _random, trials);
            if (trials == 1)
            {
                chosen[k] = candidates[0];
                var toChosen = SquaredDistancesToRows(data, candidates);
                minDistances = Engine.TensorMin(minDistances.AsTensor(), Engine.Reshape(toChosen, new[] { rows })).ToVector();
                continue;
            }

            // [rows, trials]: every row's minimum squared distance once each candidate is a center
            var lowered = Engine.TensorMin(
                Engine.TensorBroadcastTo(Tensor<T>.FromVector(minDistances, new[] { rows, 1 }), new[] { rows, trials }),
                SquaredDistancesToRows(data, candidates));
            var potentials = Engine.ReduceSum(lowered, new[] { 0 }, false).ToVector();
            int best = ClusteringTensorOps.ArgMin(potentials);
            chosen[k] = candidates[best];
            minDistances = Engine.Reshape(Engine.TensorSlice(lowered, new[] { 0, best }, new[] { rows, 1 }), new[] { rows }).ToVector();
        }

        return ClusteringTensorOps.SelectRows(x, chosen);
    }

#if NET7_0_OR_GREATER
    /// <summary>
    /// <see cref="InitializeKMeansPlusPlus"/> for double Euclidean data, filling <paramref name="chosen"/> after its
    /// first entry. It keeps one running nearest-distance array, lowered in O(rows) per new center, and draws each
    /// candidate from the running sum in place.
    /// </summary>
    /// <remarks>
    /// Every value is the engine form's, operation for operation: the engine's cumulative sum, column sums, row norms
    /// and matrix product all accumulate in index order without fused multiply-adds, its sorted search returns the first
    /// running sum at or above the threshold, and its arg-min the first minimum. So the draws, the chosen centers and
    /// the fit are identical to that form's for a given seed. As engine operations (a gather, a transpose, a product,
    /// broadcasts, a cumulative sum and a search per draw) seeding took a quarter of a k-means fit (#2524).
    /// </remarks>
    private void KMeansPlusPlusOfDoubles(KMeansData data, int[] chosen, int trials)
    {
        var values = data.Values;
        var rowNorms = data.RowNorms!;
        int rows = values.Shape[0];
        int columns = values.Shape[1];
        int blocks = (rows + NearestBlockRows - 1) / NearestBlockRows;

        var minDistances = new double[rows];
        var cumulative = new double[rows];
        var lowered = new double[rows * trials];
        var potentials = new double[trials];
        var candidates = new int[trials];
        var candidateNorms = new double[trials];

        // [rows, count] of min(current nearest, squared distance to candidate t), or the bare distance on the first
        // pass, written into lowered; rows are independent, so blocks of them run in parallel.
        void LowerTo(int count, bool first)
        {
            var points = ClusteringTensorOps.AsDoubles(values.AsSpan());
            for (int t = 0; t < count; t++)
            {
                var point = points.Slice(candidates[t] * columns, columns);
                double norm = 0.0;
                for (int j = 0; j < columns; j++) norm += point[j] * point[j];
                candidateNorms[t] = norm;
            }

            AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, blocks, (long)rows * columns * count, block =>
            {
                var x = ClusteringTensorOps.AsDoubles(values.AsSpan());
                var norms = ClusteringTensorOps.AsDoubles(rowNorms.AsSpan());
                int end = Math.Min(rows, (block + 1) * NearestBlockRows);
                for (int r = block * NearestBlockRows; r < end; r++)
                {
                    var row = x.Slice(r * columns, columns);
                    for (int t = 0; t < count; t++)
                    {
                        var point = x.Slice(candidates[t] * columns, columns);
                        double dot = 0.0;
                        for (int j = 0; j < columns; j++) dot += row[j] * point[j];

                        // ||x||^2 + ||c||^2 - 2 x.c, clamped at zero (rounding can leave a coincident row below it).
                        double distance = Math.Max(norms[r] + candidateNorms[t] - dot * 2.0, 0.0);
                        lowered[r * count + t] = first ? distance : Math.Min(minDistances[r], distance);
                    }
                }
            });
        }

        candidates[0] = chosen[0];
        LowerTo(1, first: true);
        Array.Copy(lowered, minDistances, rows);

        for (int k = 1; k < chosen.Length; k++)
        {
            // Candidates with probability proportional to the squared distance to the nearest chosen center: draw t is
            // the first row whose running sum reaches u_t * total, or row 0 when none does.
            double running = 0.0;
            for (int r = 0; r < rows; r++) cumulative[r] = running += minDistances[r];
            double total = rows > 0 ? cumulative[rows - 1] : 0.0;
            for (int t = 0; t < trials; t++)
            {
                double threshold = _random.NextDouble() * total;
                int low = 0, high = rows;
                while (low < high)
                {
                    int middle = (low + high) >> 1;
                    if (cumulative[middle] < threshold) low = middle + 1;
                    else high = middle;
                }

                candidates[t] = low < rows ? low : 0;
            }

            LowerTo(trials, first: false);

            // The candidate whose potential (the summed lowered distances) is lowest, the first drawn on a tie.
            int best = 0;
            if (trials > 1)
            {
                Array.Clear(potentials, 0, trials);
                for (int r = 0; r < rows; r++)
                    for (int t = 0; t < trials; t++)
                        potentials[t] += lowered[r * trials + t];

                double bestPotential = double.IsNaN(potentials[0]) ? double.PositiveInfinity : potentials[0];
                for (int t = 1; t < trials; t++)
                {
                    double potential = double.IsNaN(potentials[t]) ? double.PositiveInfinity : potentials[t];
                    if (potential < bestPotential)
                    {
                        bestPotential = potential;
                        best = t;
                    }
                }
            }

            chosen[k] = candidates[best];
            for (int r = 0; r < rows; r++) minDistances[r] = lowered[r * trials + best];
        }
    }
#endif

    /// <summary>
    /// Squared distance from every row of the data to the data rows at <paramref name="rowIndices"/>, as
    /// [rows, rowIndices.Length]: one matrix product for Euclidean distance, the square of the configured metric
    /// otherwise.
    /// </summary>
    private Tensor<T> SquaredDistancesToRows(KMeansData data, int[] rowIndices)
    {
        if (data.Euclidean && data.RowNorms is not null)
        {
            // Gathered from the shifted tensor, so neither the data nor the points are copied or re-shifted.
            // Replicating every row against every point instead costs rows * points * columns elements per
            // call, which dominated k-means++ on large inputs.
            var pointsT = Engine.TensorTranspose(
                Engine.TensorIndexSelect(data.Values, Vector<int>.Wrap(rowIndices).AsTensor(), 0));
            var pointNorms = Engine.Reshape(
                Engine.ReduceSum(Engine.TensorSquare(pointsT), new[] { 0 }, false), new[] { 1, rowIndices.Length });
            return Engine.TensorMax(
                SquaredDistanceScores(data.Values, data.RowNorms, pointsT, pointNorms), NumOps.Zero);
        }

        return SquaredDistancesTo(data, ClusteringTensorOps.SelectRows(data.X, rowIndices));
    }

    /// <summary>
    /// Squared distance from every row of the data to every row of <paramref name="points"/>, as [rows, points]:
    /// computed directly for Euclidean distance, as the square of the configured metric otherwise.
    /// </summary>
    private Tensor<T> SquaredDistancesTo(KMeansData data, Matrix<T> points)
    {
        if (data.Euclidean)
        {
            return ClusteringTensorOps.ToTensor(ClusteringTensorOps.SquaredDistances(data.X, points));
        }

        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        return Engine.TensorSquare(ClusteringTensorOps.ToTensor(distanceMetric.ComputePairwise(data.X, points)));
    }

    /// <summary>
    /// Nearest-center label (the lowest index on a tie) for every row. Euclidean distance ranks the centers by
    /// <c>||x||^2 + ||c||^2 - 2 x.c</c>, one matrix product for all rows and centers; any other metric by its own
    /// pairwise distances.
    /// </summary>
    private Vector<T> AssignLabels(KMeansData data, Matrix<T> centers)
        => ClusteringTensorOps.FromIntArray<T>(AssignLabelIndices(data, centers));

    /// <summary><see cref="AssignLabels"/> as cluster indices.</summary>
    private int[] AssignLabelIndices(KMeansData data, Matrix<T> centers)
    {
        var x = data.X;
        if (!data.Euclidean || data.RowNorms is null || data.Origin is null || centers.Rows == 0)
        {
            var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
            return ClusteringTensorOps.NearestCenters(distanceMetric, x, centers);
        }

        int rows = x.Rows;
        int clusters = centers.Rows;
        var centersT = Engine.TensorTranspose(
            Engine.TensorDivideScalar(ClusteringTensorOps.SubtractRow(centers, data.Origin), data.Scale));
        // ||x||^2 is the same for every center of a row, so the nearest center minimizes ||c||^2 / 2 - x.c.
        var halfCenterNorms = Engine.TensorMultiplyScalar(
            Engine.Reshape(Engine.ReduceSum(Engine.TensorSquare(centersT), new[] { 0 }, false), new[] { 1, clusters }),
            NumOps.FromDouble(0.5));

        // Rows are scored in cache-sized blocks so the [rows, clusters] intermediates stay small.
        int rowsPerBlock = Math.Max(1, Math.Min(rows, AssignmentBlockElements / clusters));
        if (rowsPerBlock >= rows)
        {
            return NearestCenterIndices(data.Values, centersT, halfCenterNorms);
        }

        var nearest = new int[rows];
        for (int start = 0; start < rows; start += rowsPerBlock)
        {
            int count = Math.Min(rowsPerBlock, rows - start);
            var values = Engine.TensorSlice(data.Values, new[] { start, 0 }, new[] { count, x.Columns });
            Array.Copy(NearestCenterIndices(values, centersT, halfCenterNorms), 0, nearest, start, count);
        }

        return nearest;
    }

    /// <summary>Upper bound on the elements of one [rows, clusters] block of the Euclidean assignment.</summary>
    private const int AssignmentBlockElements = 1 << 18;

    /// <summary>
    /// Index of the center minimizing <c>||c||^2 / 2 - x.c</c> (the lowest index on a tie) for every row of
    /// <paramref name="values"/> ([rows, d]) against the transposed centers [d, clusters] and their halved squared
    /// norms [1, clusters].
    /// </summary>
    private int[] NearestCenterIndices(Tensor<T> values, Tensor<T> centersT, Tensor<T> halfCenterNorms)
    {
        int rows = values.Shape[0];
        int clusters = centersT.Shape[1];
#if NET7_0_OR_GREATER
        if (typeof(T) == typeof(double) && values.IsContiguous && centersT.IsContiguous && halfCenterNorms.IsContiguous)
            return NearestCentersOfDoubles(values, centersT, halfCenterNorms);
#endif
        var scores = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(halfCenterNorms, new[] { rows, clusters }), Engine.TensorMatMul(values, centersT));
        return Engine.TensorArgMin(scores, 1).ToArray();
    }

#if NET7_0_OR_GREATER
    /// <summary>Rows per parallel block of <see cref="NearestCentersOfDoubles"/>.</summary>
    private const int NearestBlockRows = 1024;

    /// <summary>
    /// <see cref="NearestCenterIndices"/> for double in one pass per row: the row's products with every center (a SIMD
    /// vector of centers per feature), its scores <c>||c||^2 / 2 - x.c</c> and their minimum, the lowest index on a
    /// tie; blocks of rows run in parallel. As engine operations (a matrix product, a broadcast, a subtraction and an
    /// arg-min over [rows, clusters], each with its own buffer and dispatch) one k-means iteration took 4x as long
    /// (#2524).
    /// </summary>
    private static int[] NearestCentersOfDoubles(Tensor<T> values, Tensor<T> centersT, Tensor<T> halfCenterNorms)
    {
        int rows = values.Shape[0];
        int columns = values.Shape[1];
        int clusters = centersT.Shape[1];

        // Centers by feature, the clusters padded to whole SIMD vectors; a padding cluster is never read back.
        int width = System.Numerics.Vector<double>.Count;
        int vectors = (clusters + width - 1) / width;
        int padded = vectors * width;
        var centers = ClusteringTensorOps.AsDoubles(centersT.AsSpan());
        var byFeature = new double[columns * padded];
        for (int j = 0; j < columns; j++)
            centers.Slice(j * clusters, clusters).CopyTo(byFeature.AsSpan(j * padded, clusters));
        var halfNorms = ClusteringTensorOps.AsDoubles(halfCenterNorms.AsSpan()).Slice(0, clusters).ToArray();

        var nearest = new int[rows];
        int blocks = (rows + NearestBlockRows - 1) / NearestBlockRows;
        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, blocks, (long)rows * columns * clusters, block =>
        {
            var x = ClusteringTensorOps.AsDoubles(values.AsSpan());
            var centerVectors = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(
                new ReadOnlySpan<double>(byFeature));
            Span<double> dots = stackalloc double[padded];
            var dotVectors = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(dots);
            int end = Math.Min(rows, (block + 1) * NearestBlockRows);
            for (int r = block * NearestBlockRows; r < end; r++)
            {
                dotVectors.Clear();
                var row = x.Slice(r * columns, columns);
                for (int j = 0; j < columns; j++)
                {
                    var value = new System.Numerics.Vector<double>(row[j]);
                    var featureCenters = centerVectors.Slice(j * vectors, vectors);
                    for (int v = 0; v < vectors; v++) dotVectors[v] += value * featureCenters[v];
                }

                int best = 0;
                double bestScore = halfNorms[0] - dots[0];
                for (int c = 1; c < clusters; c++)
                {
                    double score = halfNorms[c] - dots[c];
                    if (score < bestScore)
                    {
                        bestScore = score;
                        best = c;
                    }
                }

                nearest[r] = best;
            }
        });

        return nearest;
    }
#endif

    /// <summary>
    /// <c>||x||^2 + ||c||^2 - 2 x.c</c> as [rows, clusters]: the squared distances up to rounding (which can
    /// leave a slightly negative value for a row that coincides with a center).
    /// </summary>
    private Tensor<T> SquaredDistanceScores(Tensor<T> values, Tensor<T> rowNorms, Tensor<T> centersT, Tensor<T> centerNorms)
    {
        int rows = values.Shape[0];
        int clusters = centersT.Shape[1];
        var dots = Engine.TensorMatMul(values, centersT);
        var norms = Engine.TensorAdd(
            Engine.TensorBroadcastTo(rowNorms, new[] { rows, clusters }),
            Engine.TensorBroadcastTo(centerNorms, new[] { rows, clusters }));
        return Engine.TensorSubtract(norms, Engine.TensorMultiplyScalar(dots, NumOps.FromDouble(2.0)));
    }

    private Matrix<T> UpdateCenters(KMeansData data, int[] labels, Matrix<T> oldCenters)
    {
        // Mean of each cluster; an empty cluster keeps its old center
        var means = ClusteringTensorOps.ClusterMeans(data.Raw, labels, _options.NumClusters, out var counts);
        return ClusteringTensorOps.RowsWherePositive(counts, means, oldCenters);
    }

    /// <summary>True when the sum over clusters of each center's squared shift is at most the absolute tolerance.</summary>
    private bool HasConverged(KMeansData data, Matrix<T> oldCenters, Matrix<T> newCenters)
        => NumOps.LessThanOrEquals(ComputeCenterShift(data, oldCenters, newCenters), data.Tolerance);

    /// <summary>The sum over clusters of each center's squared shift.</summary>
    private T ComputeCenterShift(KMeansData data, Matrix<T> oldCenters, Matrix<T> newCenters)
    {
        if (oldCenters.Rows == 0) return NumOps.Zero;

        if (data.Euclidean)
        {
            return Engine.TensorSum(ClusteringTensorOps.RowSquaredEuclidean(
                ClusteringTensorOps.ToTensor(oldCenters), ClusteringTensorOps.ToTensor(newCenters)));
        }

        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var shifts = ClusteringTensorOps.RowwiseDistances(distanceMetric, oldCenters, newCenters);
        return Engine.DotProduct(shifts, shifts);
    }

    /// <summary>Sum of squared distances from every row to the center its label names.</summary>
    private T ComputeInertia(KMeansData data, Vector<T> labels, Matrix<T> centers)
    {
        if (data.Euclidean)
        {
            return ClusteringTensorOps.Inertia(data.X, labels, centers);
        }

        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var distances = ClusteringTensorOps.AssignedDistances(distanceMetric, data.X, labels, centers);
        return Engine.DotProduct(distances, distances);
    }

    private Matrix<T> ComputeDistancesToCenters(Matrix<T> x, Matrix<T> centers)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var distances = distanceMetric.ComputePairwise(x, centers);
        if (centers.Rows >= _options.NumClusters)
        {
            return distances;
        }

        // Fewer centers than configured clusters (after merging): the remaining columns stay zero
        var padded = new Matrix<T>(x.Rows, _options.NumClusters);
        padded.SetSubMatrix(0, 0, distances);
        return padded;
    }

    private void ValidateInputData(Matrix<T> x)
    {
        if (x.Rows < _options.NumClusters)
        {
            throw new ArgumentException(
                $"Number of samples ({x.Rows}) must be at least the number of clusters ({_options.NumClusters}).");
        }

        if (_options.InitMethod == KMeansInitMethod.Custom && _options.InitialCenters is not null)
        {
            if (_options.InitialCenters.Rows != _options.NumClusters)
            {
                throw new ArgumentException(
                    $"Initial centers rows ({_options.InitialCenters.Rows}) must equal NumClusters ({_options.NumClusters}).");
            }
            if (_options.InitialCenters.Columns != x.Columns)
            {
                throw new ArgumentException(
                    $"Initial centers columns ({_options.InitialCenters.Columns}) must equal data columns ({x.Columns}).");
            }
        }

        if (ClusteringTensorOps.ContainsNonFinite(x))
        {
            throw new ArgumentException("Input data contains NaN or Infinity values.", nameof(x));
        }
    }

    private void ValidatePredictInput(Matrix<T> x)
    {
        if (ClusterCenters is not null && x.Columns != ClusterCenters.Columns)
        {
            throw new ArgumentException(
                $"Input columns ({x.Columns}) must match training data columns ({ClusterCenters.Columns}).");
        }
    }
}
