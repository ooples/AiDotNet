using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

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
public class SelfOrganizingMap<T> : ClusteringBase<T>
{
    private readonly SOMOptions<T> _options;
    private double[,][]? _weights;
    private int[]? _neuronLabels;

    /// <summary>
    /// Initializes a new SOM instance.
    /// </summary>
    /// <param name="options">The SOM options.</param>
    public SelfOrganizingMap(SOMOptions<T>? options = null)
        : base(options ?? new SOMOptions<T>())
    {
        _options = options ?? new SOMOptions<T>();
    }

    /// <summary>
    /// Gets the neuron weight vectors (GridHeight x GridWidth x NumFeatures).
    /// </summary>
    public double[,][]? Weights => _weights;

    /// <summary>
    /// Gets the cluster label assigned to each neuron (GridHeight x GridWidth).
    /// </summary>
    public int[]? NeuronLabels => _neuronLabels;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SelfOrganizingMap<T>(new SOMOptions<T>
        {
            GridWidth = _options.GridWidth,
            GridHeight = _options.GridHeight,
            InitialLearningRate = _options.InitialLearningRate,
            InitialNeighborhoodRadius = _options.InitialNeighborhoodRadius,
            NeighborhoodType = _options.NeighborhoodType,
            Topology = _options.Topology,
            MaxIterations = _options.MaxIterations,
            DistanceMetric = _options.DistanceMetric
        });
    }

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

        var rand = Options.RandomState.HasValue ? new Random(Options.RandomState.Value) : new Random();

        // Initialize weights randomly
        _weights = new double[height, width][];
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                _weights[r, c] = new double[d];
                for (int j = 0; j < d; j++)
                {
                    _weights[r, c][j] = rand.NextDouble();
                }
            }
        }

        // Get initial parameters
        double initialRadius = _options.InitialNeighborhoodRadius > 0
            ? _options.InitialNeighborhoodRadius
            : Math.Max(width, height) / 2.0;

        double timeConstant = Options.MaxIterations / Math.Log(initialRadius);

        // Training
        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Decay parameters
            double radius = initialRadius * Math.Exp(-iter / timeConstant);
            double learningRate = _options.InitialLearningRate * Math.Exp(-(double)iter / Options.MaxIterations);

            // Select a random training sample
            int sampleIdx = rand.Next(n);
            var sample = new double[d];
            for (int j = 0; j < d; j++)
            {
                sample[j] = NumOps.ToDouble(x[sampleIdx, j]);
            }

            // Find BMU
            var (bmuRow, bmuCol) = FindBMU(sample, d);

            // Update BMU and neighbors
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    double distance = GetGridDistance(bmuRow, bmuCol, r, c);
                    double influence = GetNeighborhoodInfluence(distance, radius);

                    if (influence > 0.001)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            _weights[r, c][j] += learningRate * influence *
                                (sample[j] - _weights[r, c][j]);
                        }
                    }
                }
            }
        }

        // Cluster the neurons using K-Means on weight vectors
        ClusterNeurons(width, height, d);

        // Set labels for each training sample
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            var sample = new double[d];
            for (int j = 0; j < d; j++)
            {
                sample[j] = NumOps.ToDouble(x[i, j]);
            }

            var (bmuRow, bmuCol) = FindBMU(sample, d);
            int neuronIdx = bmuRow * width + bmuCol;
            Labels[i] = NumOps.FromDouble(_neuronLabels![neuronIdx]);
        }

        IsTrained = true;
    }

    private (int row, int col) FindBMU(double[] sample, int d)
    {
        int bmuRow = 0;
        int bmuCol = 0;
        double minDist = double.MaxValue;

        for (int r = 0; r < _options.GridHeight; r++)
        {
            for (int c = 0; c < _options.GridWidth; c++)
            {
                double dist = 0;
                for (int j = 0; j < d; j++)
                {
                    double diff = sample[j] - _weights![r, c][j];
                    dist += diff * diff;
                }

                if (dist < minDist)
                {
                    minDist = dist;
                    bmuRow = r;
                    bmuCol = c;
                }
            }
        }

        return (bmuRow, bmuCol);
    }

    private double GetGridDistance(int r1, int c1, int r2, int c2)
    {
        int width = _options.GridWidth;
        int height = _options.GridHeight;

        double dr = r1 - r2;
        double dc = c1 - c2;

        if (_options.Topology == SOMTopology.Toroidal)
        {
            // Handle wrapping
            if (Math.Abs(dr) > height / 2.0) dr = height - Math.Abs(dr);
            if (Math.Abs(dc) > width / 2.0) dc = width - Math.Abs(dc);
        }

        if (_options.Topology == SOMTopology.Hexagonal)
        {
            // Adjust for hexagonal grid distance
            double xDist = dc + (r1 % 2 - r2 % 2) * 0.5;
            return Math.Sqrt(dr * dr + xDist * xDist);
        }

        return Math.Sqrt(dr * dr + dc * dc);
    }

    private double GetNeighborhoodInfluence(double distance, double radius)
    {
        switch (_options.NeighborhoodType)
        {
            case NeighborhoodFunction.Gaussian:
                return Math.Exp(-(distance * distance) / (2 * radius * radius));

            case NeighborhoodFunction.Bubble:
                return distance <= radius ? 1.0 : 0.0;

            case NeighborhoodFunction.MexicanHat:
                double normalized = distance / radius;
                return (1 - 2 * normalized * normalized) * Math.Exp(-normalized * normalized);

            default:
                return Math.Exp(-(distance * distance) / (2 * radius * radius));
        }
    }

    private void ClusterNeurons(int width, int height, int d)
    {
        // Flatten weights to a list for clustering
        int numNeurons = width * height;
        var neuronWeights = new double[numNeurons][];

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                neuronWeights[r * width + c] = _weights![r, c];
            }
        }

        // Determine optimal number of clusters using simple heuristic
        int k = Math.Min((int)Math.Sqrt(numNeurons), 10);
        NumClusters = k;

        // Simple K-Means clustering of neurons
        _neuronLabels = new int[numNeurons];
        var centers = new double[k][];
        var rand = Options.RandomState.HasValue ? new Random(Options.RandomState.Value) : new Random();

        // Initialize centers randomly from neurons
        var indices = Enumerable.Range(0, numNeurons).OrderBy(_ => rand.Next()).Take(k).ToArray();
        for (int i = 0; i < k; i++)
        {
            centers[i] = (double[])neuronWeights[indices[i]].Clone();
        }

        // K-Means iterations
        for (int iter = 0; iter < 20; iter++)
        {
            // Assign neurons to nearest center
            for (int i = 0; i < numNeurons; i++)
            {
                double minDist = double.MaxValue;
                int bestCluster = 0;

                for (int c = 0; c < k; c++)
                {
                    double dist = 0;
                    for (int j = 0; j < d; j++)
                    {
                        double diff = neuronWeights[i][j] - centers[c][j];
                        dist += diff * diff;
                    }

                    if (dist < minDist)
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                _neuronLabels[i] = bestCluster;
            }

            // Update centers
            var counts = new int[k];
            var newCenters = new double[k][];
            for (int c = 0; c < k; c++)
            {
                newCenters[c] = new double[d];
            }

            for (int i = 0; i < numNeurons; i++)
            {
                int cluster = _neuronLabels[i];
                counts[cluster]++;
                for (int j = 0; j < d; j++)
                {
                    newCenters[cluster][j] += neuronWeights[i][j];
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (counts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        centers[c][j] = newCenters[c][j] / counts[c];
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
                ClusterCenters[c, j] = NumOps.FromDouble(centers[c][j]);
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
        var sample = new double[d];
        for (int j = 0; j < d; j++)
        {
            sample[j] = NumOps.ToDouble(point[j]);
        }

        return FindBMU(sample, d);
    }

    /// <summary>
    /// Gets the U-Matrix (unified distance matrix) for visualization.
    /// </summary>
    /// <returns>Matrix of average distances to neighbors.</returns>
    public double[,] GetUMatrix()
    {
        ValidateIsTrained();

        int width = _options.GridWidth;
        int height = _options.GridHeight;
        int d = NumFeatures;
        var uMatrix = new double[height, width];

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                double sum = 0;
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

                        double dist = 0;
                        for (int j = 0; j < d; j++)
                        {
                            double diff = _weights![r, c][j] - _weights[nr, nc][j];
                            dist += diff * diff;
                        }

                        sum += Math.Sqrt(dist);
                        count++;
                    }
                }

                uMatrix[r, c] = count > 0 ? sum / count : 0;
            }
        }

        return uMatrix;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int d = NumFeatures;
        int width = _options.GridWidth;
        var labels = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            var sample = new double[d];
            for (int j = 0; j < d; j++)
            {
                sample[j] = NumOps.ToDouble(x[i, j]);
            }

            var (bmuRow, bmuCol) = FindBMU(sample, d);
            int neuronIdx = bmuRow * width + bmuCol;
            labels[i] = NumOps.FromDouble(_neuronLabels![neuronIdx]);
        }

        return labels;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }
}
