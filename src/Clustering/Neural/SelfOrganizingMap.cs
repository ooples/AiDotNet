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
public partial class SelfOrganizingMap<T> : ClusteringBase<T>
{
    private readonly SOMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Matrix<T>? _weights;
    private Matrix<T> FittedWeights => _weights ?? throw new InvalidOperationException("SOM not fitted. Call Fit() or FitPredict() first.");
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
    public Tensor<T>? Weights => _weights is null
        ? null
        : Engine.Reshape(Tensor<T>.FromMatrix(_weights), new[] { _options.GridHeight, _options.GridWidth, _weights.Columns });

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
        int numNeurons = width * height;

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Compute data range per feature for proper weight initialization
        var data = Tensor<T>.FromMatrix(x);
        var minVals = Engine.TensorAmin(data, new[] { 0 }, true);
        var range = Engine.TensorSubtract(Engine.ReduceMax(data, new[] { 0 }, true), minVals);
        range = Engine.TensorWhere(
            Engine.TensorLessThan(range, NumOps.FromDouble(1e-10)), Engine.TensorOnesLike(range), range);

        // Initialize weights randomly within the data range (neuron r * width + c, row by row)
        var uniform = Tensor<T>.FromMatrix(RandomUniform(numNeurons, d, rand));
        var shape = new[] { numNeurons, d };
        _weights = Engine.TensorAdd(
            Engine.TensorBroadcastTo(minVals, shape),
            Engine.TensorMultiply(uniform, Engine.TensorBroadcastTo(range, shape))).ToMatrix();

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
            var sample = x.GetRow(sampleIdx);

            // Find BMU
            int bmu = FindBMU(sample);

            // Update BMU and neighbors with influence above 0.001: w += lr * influence * (sample - w)
            var influence = GetNeighborhoodInfluence(GetGridDistances(bmu / width, bmu % width), radius);
            var scale = Engine.TensorMultiplyScalar(
                Engine.TensorMultiply(influence, Engine.TensorGreaterThan(influence, NumOps.FromDouble(0.001))),
                NumOps.FromDouble(learningRate));
            var weights = Tensor<T>.FromMatrix(_weights);
            var toSample = Engine.TensorSubtract(ClusteringTensorOps.TileRow(sample, numNeurons), weights);
            _weights = Engine.TensorAdd(
                weights,
                Engine.TensorMultiply(toSample, Engine.TensorBroadcastTo(Engine.Reshape(scale, new[] { numNeurons, 1 }), shape))).ToMatrix();
        }

        // Cluster the neurons using K-Means on weight vectors
        ClusterNeurons(width, height, d);

        // Set labels for each training sample
        var bmus = FindBMUs(x);
        var trainedNeuronLabels = _neuronLabels;
        Labels = trainedNeuronLabels is not null && n > 0
            ? ClusteringTensorOps.FromIntArray<T>(bmus.Select(b => trainedNeuronLabels[b]).ToArray())
            : new Vector<T>(n);

        // Recompute ClusterCenters from actual data (not neuron weights)
        // so MergeDegenerateClusters can accurately assess inter-cluster distances
        ClusterCenters = NumClusters > 0
            ? ClusteringTensorOps.ClusterMeans(x, Labels, NumClusters, out _)
            : new Matrix<T>(0, d);

        int premergeK = NumClusters;
        MergeDegenerateClusters(x);

        // Update _neuronLabels to match merged cluster IDs so Predict() is consistent
        if (NumClusters < premergeK && _neuronLabels is not null)
        {
            // Build mapping from old labels to new labels using the merged Labels vector
            var mergedLabels = Labels is not null ? ClusteringTensorOps.ToIntArray(Labels) : new int[n];
            var labelMap = new Dictionary<int, int>();
            for (int i = 0; i < n; i++)
            {
                labelMap[_neuronLabels[bmus[i]]] = mergedLabels[i];
            }

            // Remap ALL neurons (not just BMU winners) to valid post-merge labels.
            // Neurons that never won a BMU could retain stale labels >= NumClusters:
            // they are assigned to the nearest cluster center instead.
            var neuronLabels = _neuronLabels;
            var unmapped = Enumerable.Range(0, neuronLabels.Length).Where(ni => !labelMap.ContainsKey(neuronLabels[ni])).ToArray();
            var nearest = unmapped.Length > 0 && ClusterCenters is not null && ClusterCenters.Rows > 0
                ? ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SquaredDistances(
                    ClusteringTensorOps.SelectRows(FittedWeights, unmapped), ClusterCenters))
                : new int[unmapped.Length];

            for (int ni = 0; ni < _neuronLabels.Length; ni++)
            {
                if (labelMap.TryGetValue(_neuronLabels[ni], out int mapped))
                {
                    _neuronLabels[ni] = mapped;
                }
            }

            for (int u = 0; u < unmapped.Length; u++)
            {
                _neuronLabels[unmapped[u]] = nearest[u];
            }
        }

        IsTrained = true;
    }

    /// <summary>Uniform [0, 1) draws from <paramref name="rand"/> filling a [rows, columns] matrix row by row.</summary>
    private Matrix<T> RandomUniform(int rows, int columns, Random rand)
    {
        var values = new Matrix<T>(rows, columns);
        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < columns; j++)
            {
                values[r, j] = NumOps.FromDouble(rand.NextDouble());
            }
        }

        return values;
    }

    private int FindBMU(Vector<T> sample)
    {
        // Neuron (r * width + c) with the smallest squared distance, first on ties
        return ClusteringTensorOps.ArgMin(ClusteringTensorOps.SquaredDistancesToPoint(sample, FittedWeights));
    }

    /// <summary>Best-matching neuron of every row.</summary>
    private int[] FindBMUs(Matrix<T> x)
        => x.Rows == 0 ? Array.Empty<int>() : ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SquaredDistances(x, FittedWeights));

    private Tensor<T> GetGridDistances(int bmuRow, int bmuCol)
    {
        // Grid distance from the BMU to every neuron (r * width + c)
        int width = _options.GridWidth;
        int height = _options.GridHeight;
        int numNeurons = width * height;

        var rowIds = new int[numNeurons];
        var columnIds = new int[numNeurons];
        var rowParity = new int[numNeurons];
        for (int i = 0; i < numNeurons; i++)
        {
            rowIds[i] = i / width;
            columnIds[i] = i % width;
            rowParity[i] = bmuRow % 2 - (i / width) % 2;
        }

        var dr = Engine.TensorSubtractScalar(ClusteringTensorOps.FromIntArray<T>(rowIds).AsTensor(), NumOps.FromDouble(bmuRow));
        var dc = Engine.TensorSubtractScalar(ClusteringTensorOps.FromIntArray<T>(columnIds).AsTensor(), NumOps.FromDouble(bmuCol));

        if (_options.Topology == SOMTopology.Toroidal)
        {
            // Handle wrapping
            dr = Wrap(dr, height);
            dc = Wrap(dc, width);
        }

        if (_options.Topology == SOMTopology.Hexagonal)
        {
            // Adjust for hexagonal grid distance: (c1 - c2) + (r1 % 2 - r2 % 2) / 2
            var shift = Engine.TensorMultiplyScalar(ClusteringTensorOps.FromIntArray<T>(rowParity).AsTensor(), NumOps.FromDouble(0.5));
            dc = Engine.TensorAdd(Engine.TensorNegate(dc), shift);
        }

        return Engine.TensorSqrt(Engine.TensorAdd(Engine.TensorSquare(dr), Engine.TensorSquare(dc)));
    }

    /// <summary>Wrapped offset magnitude: size - |delta| where |delta| exceeds half the size.</summary>
    private Tensor<T> Wrap(Tensor<T> delta, int size)
    {
        var magnitude = Engine.TensorAbs(delta);
        var far = Engine.TensorGreaterThan(magnitude, NumOps.FromDouble(size / 2.0));
        return Engine.TensorWhere(far, Engine.ScalarMinusTensor(NumOps.FromDouble(size), magnitude), delta);
    }

    private Tensor<T> GetNeighborhoodInfluence(Tensor<T> distance, double radius)
    {
        switch (_options.NeighborhoodType)
        {
            case NeighborhoodFunction.Bubble:
                return Engine.ScalarMinusTensor(NumOps.One, Engine.TensorGreaterThan(distance, NumOps.FromDouble(radius)));

            case NeighborhoodFunction.MexicanHat:
            {
                // (1 - 2 u^2) exp(-u^2) with u = distance / radius
                var normalizedSq = Engine.TensorSquare(Engine.TensorDivideScalar(distance, NumOps.FromDouble(radius)));
                return Engine.TensorMultiply(
                    Engine.ScalarMinusTensor(NumOps.One, Engine.TensorMultiplyScalar(normalizedSq, NumOps.FromDouble(2))),
                    Engine.TensorExp(Engine.TensorNegate(normalizedSq)));
            }

            default:
                // Gaussian: exp(-distance^2 / (2 radius^2))
                return Engine.TensorExp(Engine.TensorDivideScalar(
                    Engine.TensorNegate(Engine.TensorSquare(distance)), NumOps.FromDouble(2 * radius * radius)));
        }
    }

    private void ClusterNeurons(int width, int height, int d)
    {
        int numNeurons = width * height;
        var neuronWeights = FittedWeights;

        // Determine optimal number of clusters using simple heuristic
        int k = Math.Min((int)Math.Sqrt(numNeurons), 10);
        NumClusters = k;

        // Simple K-Means clustering of neurons
        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize centers randomly from neurons
        var indices = Enumerable.Range(0, numNeurons).OrderBy(_ => rand.Next()).Take(k).ToArray();
        var centers = ClusteringTensorOps.SelectRows(neuronWeights, indices);
        _neuronLabels = new int[numNeurons];

        // K-Means iterations
        for (int iter = 0; iter < 20 && k > 0; iter++)
        {
            // Assign neurons to nearest center
            _neuronLabels = ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SquaredDistances(neuronWeights, centers));

            // Update centers (empty clusters keep their center)
            var means = ClusteringTensorOps.ClusterMeans(
                neuronWeights, ClusteringTensorOps.FromIntArray<T>(_neuronLabels), k, out var counts);
            centers = ClusteringTensorOps.RowsWherePositive(counts, means, centers);
        }

        // Set cluster centers
        ClusterCenters = centers;
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
        int bmu = FindBMU(point.Length == d ? point : point.GetSubVector(0, d));
        return (bmu / _options.GridWidth, bmu % _options.GridWidth);
    }

    /// <summary>
    /// Gets the U-Matrix (unified distance matrix) for visualization.
    /// </summary>
    /// <returns>Matrix of average distances to neighbors.</returns>
    public Matrix<T> GetUMatrix()
    {
        ValidateIsTrained();

        int width = _options.GridWidth;
        int height = _options.GridHeight;
        int numNeurons = width * height;
        var weights = FittedWeights;

        var sum = Tensor<T>.CreateZeros(new[] { numNeurons });
        var count = new int[numNeurons];

        // Average distance to the (up to 8) grid neighbors
        for (int dr = -1; dr <= 1; dr++)
        {
            for (int dc = -1; dc <= 1; dc++)
            {
                if (dr == 0 && dc == 0) continue;

                var neighbor = new int[numNeurons];
                var valid = new int[numNeurons];
                for (int i = 0; i < numNeurons; i++)
                {
                    int nr = i / width + dr;
                    int nc = i % width + dc;

                    if (_options.Topology == SOMTopology.Toroidal)
                    {
                        nr = (nr + height) % height;
                        nc = (nc + width) % width;
                    }
                    else if (nr < 0 || nr >= height || nc < 0 || nc >= width)
                    {
                        continue;
                    }

                    neighbor[i] = nr * width + nc;
                    valid[i] = 1;
                    count[i]++;
                }

                var distances = ClusteringTensorOps.RowEuclidean(
                    Tensor<T>.FromMatrix(weights), Tensor<T>.FromMatrix(ClusteringTensorOps.SelectRows(weights, neighbor)));
                sum = Engine.TensorAdd(sum, Engine.TensorMultiply(distances, ClusteringTensorOps.FromIntArray<T>(valid).AsTensor()));
            }
        }

        var counts = ClusteringTensorOps.FromIntArray<T>(count).AsTensor();
        var hasNeighbors = Engine.TensorGreaterThan(counts, NumOps.Zero);
        var average = Engine.TensorDivide(sum, Engine.TensorWhere(hasNeighbors, counts, Engine.TensorOnesLike(counts)));
        var uMatrix = Engine.TensorWhere(hasNeighbors, average, Engine.TensorZerosLike(average));
        return Engine.Reshape(uMatrix, new[] { height, width }).ToMatrix();
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        // Return stored labels for in-sample prediction (preserves merge results)
        if (Labels is not null && ReferenceEquals(x, TrainingDataRef))
            return new Vector<T>(Labels);

        if (_neuronLabels is null || x.Rows == 0)
        {
            return new Vector<T>(x.Rows);
        }

        var neuronLabels = _neuronLabels;
        return ClusteringTensorOps.FromIntArray<T>(FindBMUs(x).Select(b => neuronLabels[b]).ToArray());
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }
}
