using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Neural;

/// <summary>
/// Self-Organizing Map (SOM) / Kohonen Network implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A Self-Organizing Map is an unsupervised neural network that learns a
/// low-dimensional representation of high-dimensional input data. The map
/// preserves topological relationships: similar inputs activate nearby neurons.
/// </para>
/// <para>
/// Algorithm:
/// 1. Initialize neuron weights randomly
/// 2. For each training sample:
///    a. Find the Best Matching Unit (BMU) - closest neuron
///    b. Update BMU and neighbors to be more like the input
///    c. Decay learning rate and neighborhood radius over time
/// 3. Repeat for many iterations
/// </para>
/// <para><b>For Beginners:</b> SOM creates a "neural map" of your data.
///
/// Think of it like training a room full of students:
/// - Each student (neuron) specializes in a topic
/// - Students sitting near each other learn similar topics
/// - Over time, each student becomes expert in their area
///
/// The result is a 2D map where:
/// - Each neuron represents a prototype pattern
/// - Nearby neurons represent similar patterns
/// - You can visualize high-dimensional data on a 2D grid
///
/// Great for:
/// - Data visualization
/// - Finding cluster structure
/// - Understanding data topology
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SelfOrganizingMap&lt;double&gt;(new SOMOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// Vector&lt;double&gt;? labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Clustering)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Self-Organized Formation of Topologically Correct Feature Maps", "https://doi.org/10.1007/BF00337288", Year = 1982, Authors = "Teuvo Kohonen")]
[TaskVariantOf(typeof(AiDotNet.NeuralNetworks.SelfOrganizingMap<>), "clustering")]
public partial class SelfOrganizingMap<T> : ClusteringBase<T>
{
    private readonly SOMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[,][]? _weights;
    private T[,][] FittedWeights => _weights ?? throw new InvalidOperationException("SOM not fitted. Call Fit() or FitPredict() first.");
    private int[]? _neuronLabels;

    /// <summary>
    /// Initializes a new SOM instance.
    /// </summary>
    /// <param name="options">The SOM options.</param>
    public SelfOrganizingMap(SOMOptions<T>? options = null)
        : base(options ??= new SOMOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the neuron weight vectors (GridHeight x GridWidth x NumFeatures).
    /// </summary>
    public T[,][]? Weights => _weights;

    /// <summary>
    /// Gets the cluster label assigned to each neuron (GridHeight x GridWidth).
    /// </summary>
    public int[]? NeuronLabels => _neuronLabels;

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (SelfOrganizingMap<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        int width = _options.GridWidth;
        int height = _options.GridHeight;

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Compute data range per feature for proper weight initialization
        var minVals = new double[d];
        var maxVals = new double[d];
        for (int j = 0; j < d; j++)
        {
            minVals[j] = double.MaxValue;
            maxVals[j] = double.MinValue;
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double val = NumOps.ToDouble(x[i, j]);
                if (val < minVals[j]) minVals[j] = val;
                if (val > maxVals[j]) maxVals[j] = val;
            }
        }

        // SOM_PAK's random initialization within the data range, then its schedules on the shared map.
        var map = new NeuralNetworks.SelfOrganizingMapCore<T>(new NeuralNetworks.SelfOrganizingMapConfiguration(
            width, height, d, Options.MaxIterations, _options.InitialLearningRate, _options.InitialNeighborhoodRadius,
            _options.Topology, _options.NeighborhoodType));
        map.InitializeWithinRange(minVals, maxVals, rand);
        var drawn = new T[d];
        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // A random training sample each step (SOM_PAK's -rand order).
            int sampleIdx = rand.Next(n);
            for (int j = 0; j < d; j++) drawn[j] = x[sampleIdx, j];
            map.Step(drawn, iter, Distance);
        }
        _weights = new T[height, width][];
        var codebook = map.Codebook.AsSpan();
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                _weights[r, c] = codebook.Slice((r * width + c) * d, d).ToArray();

        // Cluster the neurons using K-Means on weight vectors
        ClusterNeurons(width, height, d);

        // Set labels for each training sample
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            var sample = new T[d];
            for (int j = 0; j < d; j++)
            {
                sample[j] = x[i, j];
            }

            var (bmuRow, bmuCol) = FindBMU(sample, d);
            int neuronIdx = bmuRow * width + bmuCol;
            if (_neuronLabels is not null)
            {
                Labels[i] = NumOps.FromDouble(_neuronLabels[neuronIdx]);
            }
        }

        // Recompute ClusterCenters from actual data (not neuron weights)
        // so MergeDegenerateClusters can accurately assess inter-cluster distances
        var dataCenters = new Matrix<T>(NumClusters, d);
        var clusterCounts = new int[NumClusters];
        for (int i = 0; i < n; i++)
        {
            int c = (int)Math.Round(NumOps.ToDouble(Labels[i]));
            if (c >= 0 && c < NumClusters)
            {
                for (int j = 0; j < d; j++)
                    dataCenters[c, j] = NumOps.Add(dataCenters[c, j], x[i, j]);
                clusterCounts[c]++;
            }
        }
        for (int c = 0; c < NumClusters; c++)
        {
            if (clusterCounts[c] > 0)
            {
                for (int j = 0; j < d; j++)
                    dataCenters[c, j] = NumOps.Divide(dataCenters[c, j], NumOps.FromDouble(clusterCounts[c]));
            }
        }
        ClusterCenters = dataCenters;

        int premergeK = NumClusters;
        MergeDegenerateClusters(x);

        // Update _neuronLabels to match merged cluster IDs so Predict() is consistent
        if (NumClusters < premergeK && _neuronLabels is not null)
        {
            // Build mapping from old labels to new labels using the merged Labels vector
            var labelMap = new Dictionary<int, int>();
            for (int i = 0; i < n; i++)
            {
                int neuronIdx2 = 0;
                var sample2 = new T[d];
                for (int j = 0; j < d; j++) sample2[j] = x[i, j];
                var (br, bc) = FindBMU(sample2, d);
                neuronIdx2 = br * width + bc;
                int oldLabel = _neuronLabels[neuronIdx2];
                int newLabel = Labels is not null ? (int)Math.Round(NumOps.ToDouble(Labels[i])) : 0;
                labelMap[oldLabel] = newLabel;
            }

            // Remap ALL neurons (not just BMU winners) to valid post-merge labels.
            // Neurons that never won a BMU could retain stale labels >= NumClusters.
            for (int ni = 0; ni < _neuronLabels.Length; ni++)
            {
                if (labelMap.TryGetValue(_neuronLabels[ni], out int mapped))
                {
                    _neuronLabels[ni] = mapped;
                }
                else
                {
                    // This neuron was never a BMU winner — assign it to the nearest cluster center
                    int nr = ni / width;
                    int nc = ni % width;
                    T[] neuronWeights = FittedWeights[nr, nc];
                    int bestCluster = 0;
                    T bestDist = NumOps.MaxValue;
                    for (int ci = 0; ci < ClusterCenters.Rows; ci++)
                    {
                        T dist = NumOps.Zero;
                        for (int fi = 0; fi < d; fi++)
                        {
                            T diff = NumOps.Subtract(neuronWeights[fi], ClusterCenters[ci, fi]);
                            dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                        }
                        if (NumOps.ToDouble(dist) < NumOps.ToDouble(bestDist))
                        {
                            bestDist = dist;
                            bestCluster = ci;
                        }
                    }
                    _neuronLabels[ni] = bestCluster;
                }
            }
        }

        IsTrained = true;
    }

    /// <summary>The configured distance, or null for SOM_PAK's Euclidean distance.</summary>
    private Func<Vector<T>, Vector<T>, T>? Distance => _options.DistanceMetric is { } metric ? metric.Compute : null;

    private (int row, int col) FindBMU(T[] sample, int d)
    {
        int width = _options.GridWidth, height = _options.GridHeight;
        var codebook = new Tensor<T>(new[] { width * height, d });
        var span = codebook.AsWritableSpan();
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                FittedWeights[r, c].AsSpan().CopyTo(span.Slice((r * width + c) * d, d));
        var map = new NeuralNetworks.SelfOrganizingMapCore<T>(new NeuralNetworks.SelfOrganizingMapConfiguration(
            width, height, d, Math.Max(1, Options.MaxIterations), _options.InitialLearningRate), codebook);
        int unit = map.BestMatchingUnit(sample, Distance);
        return (unit / width, unit % width);
    }

    private void ClusterNeurons(int width, int height, int d)
    {
        // Flatten weights to a list for clustering
        int numNeurons = width * height;
        var neuronWeights = new T[numNeurons][];

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                neuronWeights[r * width + c] = FittedWeights[r, c];
            }
        }

        // Determine optimal number of clusters using simple heuristic
        int k = Math.Min((int)Math.Sqrt(numNeurons), 10);
        NumClusters = k;

        // Simple K-Means clustering of neurons
        _neuronLabels = new int[numNeurons];
        var centers = new T[k][];
        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize centers randomly from neurons
        var indices = Enumerable.Range(0, numNeurons).OrderBy(_ => rand.Next()).Take(k).ToArray();
        for (int i = 0; i < k; i++)
        {
            centers[i] = (T[])neuronWeights[indices[i]].Clone();
        }

        // K-Means iterations
        for (int iter = 0; iter < 20; iter++)
        {
            // Assign neurons to nearest center
            for (int i = 0; i < numNeurons; i++)
            {
                T minDist = NumOps.MaxValue;
                int bestCluster = 0;

                for (int c = 0; c < k; c++)
                {
                    T dist = NumOps.Zero;
                    for (int j = 0; j < d; j++)
                    {
                        T diff = NumOps.Subtract(neuronWeights[i][j], centers[c][j]);
                        dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                    }

                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                _neuronLabels[i] = bestCluster;
            }

            // Update centers
            var counts = new int[k];
            var newCenters = new T[k][];
            for (int c = 0; c < k; c++)
            {
                newCenters[c] = new T[d];
                for (int j = 0; j < d; j++)
                {
                    newCenters[c][j] = NumOps.Zero;
                }
            }

            for (int i = 0; i < numNeurons; i++)
            {
                int cluster = _neuronLabels[i];
                counts[cluster]++;
                for (int j = 0; j < d; j++)
                {
                    newCenters[cluster][j] = NumOps.Add(newCenters[cluster][j], neuronWeights[i][j]);
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (counts[c] > 0)
                {
                    T countT = NumOps.FromDouble(counts[c]);
                    for (int j = 0; j < d; j++)
                    {
                        centers[c][j] = NumOps.Divide(newCenters[c][j], countT);
                    }
                }
            }
        }

        // Set cluster centers
        ClusterCenters = new Matrix<T>(k, d);
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = centers[c][j];
            }
        }
    }

    /// <summary>
    /// Gets the 2D grid position for a data point.
    /// </summary>
    /// <param name="point">The data point.</param>
    /// <returns>Grid position (row, column).</returns>
    public (int row, int col) GetGridPosition(Vector<T> point)
    {
        ValidateIsTrained();

        int d = NumFeatures;
        var sample = new T[d];
        for (int j = 0; j < d; j++)
        {
            sample[j] = point[j];
        }

        return FindBMU(sample, d);
    }

    /// <summary>
    /// Gets the U-Matrix (unified distance matrix) for visualization.
    /// </summary>
    /// <returns>Matrix of average distances to neighbors.</returns>
    public T[,] GetUMatrix()
    {
        ValidateIsTrained();

        int width = _options.GridWidth;
        int height = _options.GridHeight;
        int d = NumFeatures;
        var uMatrix = new T[height, width];

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                T sum = NumOps.Zero;
                int count = 0;

                // Check all neighbors
                for (int dr = -1; dr <= 1; dr++)
                {
                    for (int dc = -1; dc <= 1; dc++)
                    {
                        if (dr == 0 && dc == 0) continue;

                        int nr = r + dr;
                        int nc = c + dc;

                        if (_options.Topology == SOMTopology.Toroidal)
                        {
                            nr = (nr + height) % height;
                            nc = (nc + width) % width;
                        }
                        else if (nr < 0 || nr >= height || nc < 0 || nc >= width)
                        {
                            continue;
                        }

                        T distSq = NumOps.Zero;
                        for (int j = 0; j < d; j++)
                        {
                            T diff = NumOps.Subtract(FittedWeights[r, c][j], FittedWeights[nr, nc][j]);
                            distSq = NumOps.Add(distSq, NumOps.Multiply(diff, diff));
                        }

                        sum = NumOps.Add(sum, NumOps.Sqrt(distSq));
                        count++;
                    }
                }

                uMatrix[r, c] = count > 0 ? NumOps.Divide(sum, NumOps.FromDouble(count)) : NumOps.Zero;
            }
        }

        return uMatrix;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        // Return stored labels for in-sample prediction (preserves merge results)
        if (Labels is not null && ReferenceEquals(x, TrainingDataRef))
            return new Vector<T>(Labels);

        int n = x.Rows;
        int d = NumFeatures;
        int width = _options.GridWidth;
        var labels = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            var sample = new T[d];
            for (int j = 0; j < d; j++)
            {
                sample[j] = x[i, j];
            }

            var (bmuRow, bmuCol) = FindBMU(sample, d);
            int neuronIdx = bmuRow * width + bmuCol;
            if (_neuronLabels is not null)
            {
                labels[i] = NumOps.FromDouble(_neuronLabels[neuronIdx]);
            }
        }

        return labels;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }
}
