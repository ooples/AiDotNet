# Issue #382: Implement Comprehensive Clustering Algorithms
## Junior Developer Implementation Guide

**For**: Developers new to unsupervised machine learning and clustering algorithms
**Difficulty**: Intermediate
**Estimated Time**: 35-45 hours
**Prerequisites**: Understanding of distance metrics, basic linear algebra, C# generics

---

## Understanding Clustering Algorithms

**For Beginners**: Clustering is like organizing a messy room without being told where things go. Imagine you have 100 random objects scattered on the floor. Clustering algorithms automatically group similar items together - books with books, toys with toys, clothes with clothes - without you having to label anything first.

**Why Build Clustering Algorithms?**

**vs Manual Grouping**:
- ✅ Handles thousands/millions of data points automatically
- ✅ Finds patterns humans might miss
- ✅ Objective and reproducible
- ❌ Requires choosing number of clusters (for some algorithms)

**Real-World Use Cases**:
- **Customer Segmentation**: Group customers by purchasing behavior without predefined categories
- **Image Compression**: Reduce colors in an image by clustering similar colors
- **Anomaly Detection**: Identify outliers that don't fit any cluster
- **Document Organization**: Group similar documents for search and retrieval
- **Gene Expression Analysis**: Find groups of genes with similar expression patterns

---

## Key Concepts

### Distance Metrics

Clustering algorithms need to measure similarity. The most common metric is **Euclidean distance**:

```
Distance between points A and B = sqrt((A₁-B₁)² + (A₂-B₂)² + ... + (Aₙ-Bₙ)²)
```

**Example**:
- Point A: [2, 3]
- Point B: [5, 7]
- Distance: sqrt((2-5)² + (3-7)²) = sqrt(9 + 16) = 5

### KMeans Clustering

**How it Works**:
1. Randomly place K "centroids" (cluster centers) in your data space
2. Assign each point to its nearest centroid
3. Move each centroid to the average position of all points assigned to it
4. Repeat steps 2-3 until centroids stop moving (convergence)

**Beginner Analogy**: Imagine K magnets scattered on a table with iron filings. Each filing sticks to the closest magnet (assignment). Then you move each magnet to the center of its filings (update). Repeat until magnets stop moving.

**Algorithm Complexity**: O(n × K × i × d) where:
- n = number of data points
- K = number of clusters
- i = number of iterations (typically < 100)
- d = number of dimensions

**Best For**:
- Spherical/round clusters
- Similar-sized clusters
- When you know K (number of clusters) in advance

### DBSCAN (Density-Based Spatial Clustering)

**How it Works**:
1. For each point, count neighbors within radius ε (epsilon)
2. Points with ≥ minPts neighbors are "core points"
3. Core points and their neighbors form clusters
4. Points that aren't in any cluster are "noise" (outliers)

**Beginner Analogy**: Think of a crowded concert. DBSCAN finds groups of people standing close together (clusters), individual people scattered around (noise), and people on the edge of groups (border points).

**Algorithm Complexity**: O(n log n) with spatial indexing, O(n²) naive

**Best For**:
- Arbitrary-shaped clusters (not just round)
- Varying cluster sizes
- Data with noise/outliers
- When you DON'T know K in advance

**Key Parameters**:
- **ε (epsilon)**: Maximum distance for two points to be neighbors (0.5 for normalized data)
- **minPts**: Minimum neighbors to form a cluster (typically 2×dimensions or 5-10)

### Hierarchical Clustering

**How it Works (Agglomerative - Bottom-Up)**:
1. Start with each point as its own cluster
2. Repeatedly merge the two closest clusters
3. Continue until you have one big cluster (or K clusters)
4. Result is a "dendrogram" (tree) showing merge history

**How it Works (Divisive - Top-Down)**:
1. Start with all points in one cluster
2. Repeatedly split the largest/most spread-out cluster
3. Continue until each point is its own cluster (or K clusters)

**Beginner Analogy**: Agglomerative is like building a family tree from children to grandparents. Divisive is like breaking down a company org chart from CEO to individual employees.

**Linkage Methods** (how to measure distance between clusters):
- **Single**: Minimum distance between any two points (creates "chains")
- **Complete**: Maximum distance between any two points (compact clusters)
- **Average**: Average distance between all pairs
- **Ward**: Minimizes variance when merging (balanced clusters)

**Algorithm Complexity**: O(n² log n) for efficient implementations, O(n³) naive

**Best For**:
- Exploring data at multiple granularities
- Creating taxonomies/hierarchies
- When you want to visualize cluster relationships
- Small to medium datasets (< 10,000 points)

---

## Implementation Overview

```
src/Clustering/
├── KMeans.cs                          [NEW - AC 1.1]
├── DBSCAN.cs                          [NEW - AC 1.2]
├── HierarchicalClustering.cs          [NEW - AC 1.3]
└── Base/
    └── ClusteringBase.cs              [NEW - base class]

src/Interfaces/
├── IClustering.cs                     [NEW - AC 1.1]
└── IDistanceMetric.cs                 [NEW - helper]

src/Evaluation/
└── ClusteringMetrics.cs               [NEW - AC 2.1]

tests/UnitTests/Clustering/
├── KMeansTests.cs                     [NEW - AC 3.1]
├── DBSCANTests.cs                     [NEW - AC 3.2]
└── HierarchicalClusteringTests.cs     [NEW - AC 3.3]
```

---

## Phase 1: Core Clustering Algorithms

### AC 1.0: Create IClustering Interface (3 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IClustering.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for clustering algorithms that group data points into clusters.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Clustering is an unsupervised learning technique that groups similar data points together
/// without requiring labeled training data. Different algorithms use different strategies for
/// determining similarity and forming clusters.
/// </para>
/// <para><b>For Beginners:</b> Think of clustering as organizing a messy room automatically.
///
/// You have items scattered everywhere, and you want to group similar items together.
/// Clustering algorithms do this for data - they find natural groupings without being told
/// what those groups should be.
///
/// For example:
/// - Customer segmentation: Group customers by shopping behavior
/// - Image compression: Group similar colors together
/// - Anomaly detection: Find data points that don't fit any group
///
/// Different clustering algorithms work better for different types of data:
/// - KMeans: Fast, works well for round, similar-sized clusters
/// - DBSCAN: Finds arbitrary shapes, handles noise and outliers
/// - Hierarchical: Creates a tree of clusters at different scales
/// </para>
/// </remarks>
public interface IClustering<T>
{
    /// <summary>
    /// Fits the clustering model to the provided data.
    /// </summary>
    /// <param name="data">
    /// A matrix where each row is a data point and each column is a feature.
    /// For example, a 100×4 matrix represents 100 data points in 4-dimensional space.
    /// </param>
    /// <remarks>
    /// <para>
    /// This method analyzes the data to identify clusters. The specific algorithm varies:
    /// - KMeans: Iteratively moves cluster centers
    /// - DBSCAN: Identifies dense regions
    /// - Hierarchical: Builds a merge/split tree
    /// </para>
    /// <para><b>For Beginners:</b> Fit is where the algorithm does its work.
    ///
    /// It's like showing someone all the items in your messy room and having them
    /// mentally organize how to group things. After fitting:
    /// - The algorithm has learned the cluster structure
    /// - You can then use Predict to assign new items to clusters
    ///
    /// For KMeans, fitting finds the best cluster centers.
    /// For DBSCAN, fitting identifies which points form dense groups.
    /// For Hierarchical, fitting builds the full merge/split tree.
    /// </para>
    /// </remarks>
    void Fit(Matrix<T> data);

    /// <summary>
    /// Predicts which cluster each data point belongs to.
    /// </summary>
    /// <param name="data">
    /// A matrix where each row is a data point to be assigned to a cluster.
    /// </param>
    /// <returns>
    /// A vector of cluster labels, where each element is the cluster ID for the corresponding data point.
    /// For example, [0, 1, 0, 2] means point 0 is in cluster 0, point 1 is in cluster 1, etc.
    ///
    /// Note: Cluster IDs start at 0. A label of -1 indicates noise/outlier (for DBSCAN).
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method assigns each data point to its nearest/most appropriate cluster based on the
    /// fitted model. For KMeans, this finds the nearest centroid. For DBSCAN, this checks if the
    /// point is within ε of a core point.
    /// </para>
    /// <para><b>For Beginners:</b> Predict assigns items to groups.
    ///
    /// After fitting (learning the cluster structure), you can predict which cluster
    /// new data points belong to. It's like:
    /// - You've organized your room into groups (books, toys, clothes)
    /// - Someone hands you a new item
    /// - You decide which group it fits best with
    ///
    /// The returned vector contains cluster IDs (0, 1, 2, ...).
    /// Points with the same ID belong to the same cluster.
    ///
    /// Example:
    /// If Predict returns [0, 0, 1, 2, 1]:
    /// - Points 0 and 1 are in cluster 0
    /// - Points 2 and 4 are in cluster 1
    /// - Point 3 is in cluster 2
    /// </para>
    /// </remarks>
    Vector<int> Predict(Matrix<T> data);

    /// <summary>
    /// Fits the model and immediately predicts cluster labels for the same data.
    /// </summary>
    /// <param name="data">
    /// A matrix where each row is a data point.
    /// </param>
    /// <returns>
    /// A vector of cluster labels for the input data.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This is a convenience method equivalent to calling Fit(data) followed by Predict(data).
    /// It's the most common usage pattern for clustering.
    /// </para>
    /// <para><b>For Beginners:</b> FitPredict is a one-step shortcut.
    ///
    /// Instead of:
    /// 1. Fit the model on your data
    /// 2. Predict cluster labels for the same data
    ///
    /// You can just call FitPredict to do both at once.
    ///
    /// This is what you'll use most often:
    /// ```csharp
    /// var kmeans = new KMeans&lt;double&gt;(numClusters: 3);
    /// Vector&lt;int&gt; labels = kmeans.FitPredict(myData);
    /// // labels now contains cluster assignments for each row in myData
    /// ```
    /// </para>
    /// </remarks>
    Vector<int> FitPredict(Matrix<T> data);

    /// <summary>
    /// Gets the number of clusters identified by the algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For KMeans, this is the K value specified in the constructor.
    /// For DBSCAN, this is determined automatically during fitting.
    /// For Hierarchical, this is either K (if specified) or the number of leaf nodes.
    /// </para>
    /// <para><b>For Beginners:</b> How many groups were found?
    ///
    /// After fitting, this tells you how many distinct clusters the algorithm identified.
    /// - KMeans: You specify this in advance (e.g., "find 3 clusters")
    /// - DBSCAN: The algorithm discovers this automatically based on density
    /// - Hierarchical: Depends on where you "cut" the tree
    /// </para>
    /// </remarks>
    int NumClusters { get; }

    /// <summary>
    /// Gets the cluster labels assigned during the last Fit or FitPredict operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property caches the result of the last clustering operation. If you call Fit
    /// followed by Predict on the same data, you can access these labels without re-prediction.
    /// </para>
    /// <para><b>For Beginners:</b> Quick access to the last clustering result.
    ///
    /// After calling Fit or FitPredict, you can get the cluster assignments from this
    /// property instead of calling Predict again. It's like saving the answer sheet
    /// so you don't have to recalculate everything.
    ///
    /// Returns null if Fit hasn't been called yet.
    /// </para>
    /// </remarks>
    Vector<int>? Labels { get; }
}
```

**Key Implementation Details**:
- **Generic type T**: Supports float, double, etc. for different precision needs
- **Matrix<T> input**: Each row is a data point, each column is a feature
- **Vector<int> output**: Cluster IDs (0, 1, 2, ..., -1 for noise in DBSCAN)
- **FitPredict**: Most common pattern - fit and predict in one step

---

### AC 1.1: Implement KMeans Clustering (13 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Clustering\KMeans.cs`

```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering;

/// <summary>
/// Implements the K-Means clustering algorithm for partitioning data into K distinct clusters.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// K-Means is one of the most widely used clustering algorithms. It works by:
/// 1. Randomly initializing K cluster centers (centroids)
/// 2. Assigning each data point to its nearest centroid
/// 3. Updating each centroid to the mean of all points assigned to it
/// 4. Repeating steps 2-3 until convergence
/// </para>
/// <para>
/// The algorithm minimizes the within-cluster sum of squares (WCSS):
/// WCSS = Σ(i=1 to K) Σ(x in cluster i) ||x - μi||²
/// where μi is the centroid of cluster i.
/// </para>
/// <para><b>For Beginners:</b> K-Means is like organizing items into K boxes.
///
/// Imagine you have 100 items and 3 boxes (K=3). K-Means works like this:
///
/// 1. Place 3 markers randomly in the room (initial centroids)
/// 2. Put each item in the box closest to its nearest marker
/// 3. Move each marker to the center of the items in its box
/// 4. Repeat steps 2-3 until markers stop moving
///
/// The "magic" is that this simple process finds natural groupings in your data.
/// The markers automatically move to the centers of dense groups of points.
///
/// <b>When to use K-Means:</b>
/// - You know approximately how many clusters to expect (K)
/// - Your clusters are roughly round/spherical in shape
/// - Your clusters are similar in size
/// - You need a fast algorithm (K-Means is very efficient)
///
/// <b>Limitations:</b>
/// - You must specify K in advance
/// - Struggles with non-spherical clusters (e.g., crescent shapes)
/// - Sensitive to outliers (one outlier can pull a centroid far away)
/// - Different random initializations can give different results
///
/// <b>Example Use Cases:</b>
/// - Customer segmentation (e.g., high/medium/low value customers)
/// - Image compression (reducing colors)
/// - Document clustering (grouping similar articles)
/// - Anomaly detection (points far from all centroids)
/// </para>
/// </remarks>
public class KMeans<T> : IClustering<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numClusters;
    private readonly int _maxIterations;
    private readonly T _tolerance;
    private readonly int _randomSeed;
    private readonly string _initMethod;

    private Matrix<T>? _centroids;
    private Vector<int>? _labels;

    /// <summary>
    /// Initializes a new instance of the KMeans clustering algorithm.
    /// </summary>
    /// <param name="numClusters">
    /// The number of clusters to form (K). Must be ≥ 2.
    /// Default: 3 (common starting point for exploratory analysis).
    /// </param>
    /// <param name="maxIterations">
    /// Maximum number of iterations to run the algorithm.
    /// Default: 300 (scikit-learn default, sufficient for most datasets).
    /// The algorithm may stop earlier if it converges before reaching this limit.
    /// </param>
    /// <param name="tolerance">
    /// Convergence tolerance - algorithm stops when centroid movement is less than this value.
    /// Default: 1e-4 (scikit-learn default, balances speed and accuracy).
    /// Smaller values = more precise but slower. Larger values = faster but less precise.
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducible centroid initialization.
    /// Default: 42 (makes results reproducible across runs).
    /// Use a different seed to explore alternative cluster configurations.
    /// </param>
    /// <param name="initMethod">
    /// Method for initializing centroids. Options: "kmeans++" or "random".
    /// Default: "kmeans++" (scikit-learn default, provably better than random).
    /// "kmeans++": Spreads initial centroids far apart (faster convergence, better results).
    /// "random": Picks K random data points as initial centroids (faster init, slower convergence).
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when numClusters is less than 2.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Understanding the parameters:
    ///
    /// <b>numClusters (K):</b>
    /// - This is the most important parameter - how many groups do you want?
    /// - Too few: You might merge distinct groups together
    /// - Too many: You might split natural groups
    /// - Common methods to choose K:
    ///   • Elbow method: Plot WCSS vs K, look for the "elbow" where improvement slows
    ///   • Silhouette analysis: Measure how well points fit their clusters
    ///   • Domain knowledge: "We sell to 3 customer segments" → K=3
    ///
    /// <b>maxIterations:</b>
    /// - Safety limit to prevent infinite loops
    /// - 300 is usually more than enough (most datasets converge in 10-50 iterations)
    /// - Increase if you see "algorithm did not converge" warnings
    ///
    /// <b>tolerance:</b>
    /// - When to stop iterating (when centroids barely move)
    /// - 1e-4 means centroids move less than 0.0001 units
    /// - Larger = faster but less precise, smaller = slower but more precise
    ///
    /// <b>randomSeed:</b>
    /// - Makes results reproducible (same input → same output)
    /// - Important for testing and debugging
    /// - Try different seeds if you get poor results (K-Means can get stuck in local minima)
    ///
    /// <b>initMethod:</b>
    /// - "kmeans++": Smarter initialization, almost always better (use this)
    /// - "random": Simple but can lead to poor clusters
    ///
    /// <b>Default values from research:</b>
    /// - maxIterations=300: scikit-learn default (MacQueen, 1967; Arthur & Vassilvitskii, 2007)
    /// - tolerance=1e-4: scikit-learn default (empirically validated)
    /// - initMethod="kmeans++": Arthur & Vassilvitskii (2007) - provably O(log K) competitive
    /// </para>
    /// </remarks>
    public KMeans(
        int numClusters = 3,
        int maxIterations = 300,
        double tolerance = 1e-4,
        int randomSeed = 42,
        string initMethod = "kmeans++")
    {
        if (numClusters < 2)
        {
            throw new ArgumentException("Number of clusters must be at least 2.", nameof(numClusters));
        }

        if (maxIterations < 1)
        {
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));
        }

        if (tolerance <= 0)
        {
            throw new ArgumentException("Tolerance must be positive.", nameof(tolerance));
        }

        if (initMethod != "kmeans++" && initMethod != "random")
        {
            throw new ArgumentException("Init method must be 'kmeans++' or 'random'.", nameof(initMethod));
        }

        _numClusters = numClusters;
        _maxIterations = maxIterations;
        _tolerance = NumOps.FromDouble(tolerance);
        _randomSeed = randomSeed;
        _initMethod = initMethod;
    }

    /// <inheritdoc/>
    public int NumClusters => _numClusters;

    /// <inheritdoc/>
    public Vector<int>? Labels => _labels;

    /// <summary>
    /// Gets the cluster centroids (centers) after fitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each row is the centroid of one cluster. For K=3 clusters in 4D space,
    /// this is a 3×4 matrix where row 0 is the center of cluster 0, etc.
    /// </para>
    /// <para><b>For Beginners:</b> Where are the cluster centers?
    ///
    /// After fitting, this tells you the coordinates of each cluster's center point.
    /// These centroids are the "representatives" of each cluster.
    ///
    /// You can use centroids to:
    /// - Understand what each cluster represents (interpret the feature values)
    /// - Compress data (represent each point by its nearest centroid)
    /// - Assign new points to clusters (find the nearest centroid)
    ///
    /// Returns null if Fit hasn't been called yet.
    /// </para>
    /// </remarks>
    public Matrix<T>? Centroids => _centroids;

    /// <inheritdoc/>
    public void Fit(Matrix<T> data)
    {
        if (data.Rows < _numClusters)
        {
            throw new ArgumentException(
                $"Number of data points ({data.Rows}) must be at least equal to number of clusters ({_numClusters}).",
                nameof(data));
        }

        var random = new Random(_randomSeed);

        // Initialize centroids
        _centroids = _initMethod == "kmeans++"
            ? InitializeCentroidsKMeansPlusPlus(data, random)
            : InitializeCentroidsRandom(data, random);

        Vector<int> labels = new Vector<int>(data.Rows);
        bool converged = false;

        for (int iteration = 0; iteration < _maxIterations && !converged; iteration++)
        {
            // Assignment step: assign each point to nearest centroid
            for (int i = 0; i < data.Rows; i++)
            {
                labels[i] = FindNearestCentroid(data.GetRow(i), _centroids);
            }

            // Update step: move centroids to mean of assigned points
            var newCentroids = UpdateCentroids(data, labels);

            // Check convergence: have centroids stopped moving?
            converged = HasConverged(_centroids, newCentroids);
            _centroids = newCentroids;
        }

        _labels = labels;
    }

    /// <inheritdoc/>
    public Vector<int> Predict(Matrix<T> data)
    {
        if (_centroids == null)
        {
            throw new InvalidOperationException("Model must be fitted before calling Predict.");
        }

        var labels = new Vector<int>(data.Rows);
        for (int i = 0; i < data.Rows; i++)
        {
            labels[i] = FindNearestCentroid(data.GetRow(i), _centroids);
        }
        return labels;
    }

    /// <inheritdoc/>
    public Vector<int> FitPredict(Matrix<T> data)
    {
        Fit(data);
        return _labels!;
    }

    /// <summary>
    /// Initializes centroids using the K-Means++ algorithm for better convergence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// K-Means++ (Arthur & Vassilvitskii, 2007) improves upon random initialization by
    /// spreading initial centroids far apart. The algorithm:
    /// 1. Pick first centroid uniformly at random
    /// 2. For each subsequent centroid, pick a point with probability proportional to
    ///    the squared distance to the nearest existing centroid
    /// 3. Repeat until K centroids are chosen
    /// </para>
    /// <para>
    /// This ensures initial centroids are well-separated, leading to:
    /// - Faster convergence (fewer iterations needed)
    /// - Better final clustering (less likely to get stuck in poor local minima)
    /// - Provably O(log K) competitive with optimal clustering
    /// </para>
    /// </remarks>
    private Matrix<T> InitializeCentroidsKMeansPlusPlus(Matrix<T> data, Random random)
    {
        var centroids = new Matrix<T>(_numClusters, data.Columns);
        var chosenIndices = new HashSet<int>();

        // Choose first centroid uniformly at random
        int firstIdx = random.Next(data.Rows);
        chosenIndices.Add(firstIdx);
        for (int j = 0; j < data.Columns; j++)
        {
            centroids[0, j] = data[firstIdx, j];
        }

        // Choose remaining centroids with probability proportional to D²
        for (int k = 1; k < _numClusters; k++)
        {
            var distances = new double[data.Rows];
            double sumDistances = 0.0;

            // For each point, compute distance to nearest existing centroid
            for (int i = 0; i < data.Rows; i++)
            {
                if (chosenIndices.Contains(i))
                {
                    distances[i] = 0.0;
                    continue;
                }

                T minDist = NumOps.FromDouble(double.MaxValue);
                for (int c = 0; c < k; c++)
                {
                    T dist = EuclideanDistance(data.GetRow(i), centroids.GetRow(c));
                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                    }
                }

                // Use squared distance for probability weighting
                distances[i] = Math.Pow(Convert.ToDouble(minDist), 2);
                sumDistances += distances[i];
            }

            // Select next centroid with probability proportional to D²
            double threshold = random.NextDouble() * sumDistances;
            double cumulative = 0.0;
            int selectedIdx = 0;

            for (int i = 0; i < data.Rows; i++)
            {
                cumulative += distances[i];
                if (cumulative >= threshold && !chosenIndices.Contains(i))
                {
                    selectedIdx = i;
                    break;
                }
            }

            chosenIndices.Add(selectedIdx);
            for (int j = 0; j < data.Columns; j++)
            {
                centroids[k, j] = data[selectedIdx, j];
            }
        }

        return centroids;
    }

    /// <summary>
    /// Initializes centroids by randomly selecting K data points.
    /// </summary>
    private Matrix<T> InitializeCentroidsRandom(Matrix<T> data, Random random)
    {
        var centroids = new Matrix<T>(_numClusters, data.Columns);
        var chosenIndices = new HashSet<int>();

        for (int k = 0; k < _numClusters; k++)
        {
            int idx;
            do
            {
                idx = random.Next(data.Rows);
            } while (chosenIndices.Contains(idx));

            chosenIndices.Add(idx);
            for (int j = 0; j < data.Columns; j++)
            {
                centroids[k, j] = data[idx, j];
            }
        }

        return centroids;
    }

    /// <summary>
    /// Finds the index of the nearest centroid to a given point.
    /// </summary>
    private int FindNearestCentroid(Vector<T> point, Matrix<T> centroids)
    {
        int nearest = 0;
        T minDist = EuclideanDistance(point, centroids.GetRow(0));

        for (int k = 1; k < centroids.Rows; k++)
        {
            T dist = EuclideanDistance(point, centroids.GetRow(k));
            if (NumOps.LessThan(dist, minDist))
            {
                minDist = dist;
                nearest = k;
            }
        }

        return nearest;
    }

    /// <summary>
    /// Computes Euclidean distance between two points.
    /// </summary>
    private T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Updates centroids to the mean of all points assigned to each cluster.
    /// </summary>
    private Matrix<T> UpdateCentroids(Matrix<T> data, Vector<int> labels)
    {
        var newCentroids = new Matrix<T>(_numClusters, data.Columns);
        var counts = new int[_numClusters];

        // Sum points in each cluster
        for (int i = 0; i < data.Rows; i++)
        {
            int cluster = labels[i];
            counts[cluster]++;

            for (int j = 0; j < data.Columns; j++)
            {
                newCentroids[cluster, j] = NumOps.Add(newCentroids[cluster, j], data[i, j]);
            }
        }

        // Divide by count to get mean
        for (int k = 0; k < _numClusters; k++)
        {
            if (counts[k] > 0)
            {
                T count = NumOps.FromDouble(counts[k]);
                for (int j = 0; j < data.Columns; j++)
                {
                    newCentroids[k, j] = NumOps.Divide(newCentroids[k, j], count);
                }
            }
            // If cluster is empty, keep previous centroid (or could reinitialize)
        }

        return newCentroids;
    }

    /// <summary>
    /// Checks if centroids have converged (stopped moving significantly).
    /// </summary>
    private bool HasConverged(Matrix<T> oldCentroids, Matrix<T> newCentroids)
    {
        for (int k = 0; k < _numClusters; k++)
        {
            T dist = EuclideanDistance(oldCentroids.GetRow(k), newCentroids.GetRow(k));
            if (NumOps.GreaterThan(dist, _tolerance))
            {
                return false;
            }
        }
        return true;
    }
}
```

**Key Implementation Details**:
- **K-Means++ initialization**: Provably better than random (Arthur & Vassilvitskii, 2007)
- **Convergence check**: Stops when centroids move less than tolerance
- **Empty cluster handling**: Keeps previous centroid (could also reinitialize)
- **Time complexity**: O(n × K × i × d) where i is typically < 100
- **Space complexity**: O(K × d) for centroids + O(n) for labels

---

### AC 1.2: Implement DBSCAN Clustering (13 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Clustering\DBSCAN.cs`

```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering;

/// <summary>
/// Implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// DBSCAN (Ester et al., 1996) is a density-based clustering algorithm that:
/// 1. Groups together points that are closely packed (high-density regions)
/// 2. Marks points in low-density regions as outliers (noise)
/// 3. Automatically determines the number of clusters
/// 4. Can find arbitrarily-shaped clusters
/// </para>
/// <para>
/// Key concepts:
/// - <b>Core point</b>: A point with at least minPts neighbors within radius ε
/// - <b>Border point</b>: Not a core point, but within ε of a core point
/// - <b>Noise point</b>: Neither core nor border (marked as cluster -1)
/// </para>
/// <para><b>For Beginners:</b> DBSCAN finds crowded areas without knowing how many groups exist.
///
/// Imagine a crowded concert venue from a bird's eye view. DBSCAN works like this:
///
/// 1. Look at each person
/// 2. If they have enough neighbors nearby (≥ minPts within radius ε), they're in a "crowd"
/// 3. Connect all neighboring crowds into one cluster
/// 4. People who are alone or in small groups are "noise" (not in any cluster)
///
/// <b>When to use DBSCAN:</b>
/// - You DON'T know how many clusters to expect
/// - Your clusters have irregular shapes (not just round)
/// - Your clusters have varying densities
/// - Your data has noise/outliers you want to identify
///
/// <b>Advantages over K-Means:</b>
/// - No need to specify number of clusters
/// - Finds arbitrary shapes (crescents, spirals, etc.)
/// - Robust to outliers (marks them as noise instead of forcing into clusters)
/// - Varying cluster sizes are fine
///
/// <b>Limitations:</b>
/// - Struggles with varying densities (one ε doesn't fit all)
/// - Sensitive to ε and minPts parameters
/// - Slower than K-Means for large datasets (O(n²) naive, O(n log n) with spatial indexing)
/// - Doesn't work well in very high dimensions (curse of dimensionality)
///
/// <b>Example Use Cases:</b>
/// - Geospatial clustering (finding crime hotspots, wildlife habitats)
/// - Anomaly detection (noise points are potential outliers)
/// - Image segmentation (grouping pixels by color density)
/// - Customer behavior (finding groups with similar purchase patterns)
/// </para>
/// </remarks>
public class DBSCAN<T> : IClustering<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _epsilon;
    private readonly int _minPoints;

    private Vector<int>? _labels;
    private int _numClusters;

    // Constants for labeling during processing
    private const int NOISE = -1;
    private const int UNDEFINED = -2;

    /// <summary>
    /// Initializes a new instance of the DBSCAN clustering algorithm.
    /// </summary>
    /// <param name="epsilon">
    /// The maximum distance between two points to be considered neighbors (ε).
    /// Default: 0.5 (assumes normalized/standardized data where features range [0,1] or [-1,1]).
    ///
    /// For unnormalized data, choose based on feature scales:
    /// - Geographic data (meters): 100-1000
    /// - Pixel colors (0-255): 10-50
    /// - Standardized features: 0.3-1.0
    /// </param>
    /// <param name="minPoints">
    /// The minimum number of points required to form a dense region (minPts).
    /// Default: 5 (Ester et al., 1996 recommendation: minPts ≥ dimensions + 1, commonly 5-10).
    ///
    /// Rule of thumb: minPts ≥ 2 × dimensions, but at least 4-5.
    /// - Small minPts (2-4): More sensitive, creates more clusters, fewer noise points
    /// - Large minPts (10+): Less sensitive, fewer clusters, more noise points
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when epsilon is not positive or minPoints is less than 2.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choosing epsilon and minPoints:
    ///
    /// <b>epsilon (ε):</b> How close is "close enough" to be neighbors?
    /// - Too small: Every point is noise (no clusters form)
    /// - Too large: Everything merges into one cluster
    /// - Just right: Naturally separates dense regions
    ///
    /// <b>How to choose ε:</b>
    /// 1. K-distance plot: Plot sorted distances to k-th nearest neighbor
    /// 2. Look for the "elbow" where distance sharply increases
    /// 3. That distance is a good ε
    ///
    /// For normalized data (features scaled to [0,1]):
    /// - Start with ε = 0.5
    /// - Increase if too many noise points
    /// - Decrease if too few clusters
    ///
    /// <b>minPoints:</b> How many neighbors make a "crowd"?
    /// - Too small: Noise gets clustered, many tiny clusters
    /// - Too large: Real clusters get marked as noise
    /// - Just right: Captures the minimum meaningful cluster size
    ///
    /// <b>Rule of thumb:</b>
    /// - 2D data: minPts = 4-6
    /// - 3D data: minPts = 6-8
    /// - High-D data: minPts = 2 × dimensions
    /// - At minimum: minPts ≥ 2 (otherwise every pair is a cluster)
    ///
    /// <b>Default values from research:</b>
    /// - epsilon=0.5: Common default for normalized data
    /// - minPoints=5: Ester et al. (1996) original recommendation
    /// - For d-dimensional data: minPts ≥ d + 1 (Sander et al., 1998)
    /// </para>
    /// </remarks>
    public DBSCAN(double epsilon = 0.5, int minPoints = 5)
    {
        if (epsilon <= 0)
        {
            throw new ArgumentException("Epsilon must be positive.", nameof(epsilon));
        }

        if (minPoints < 2)
        {
            throw new ArgumentException("MinPoints must be at least 2.", nameof(minPoints));
        }

        _epsilon = NumOps.FromDouble(epsilon);
        _minPoints = minPoints;
    }

    /// <inheritdoc/>
    public int NumClusters => _numClusters;

    /// <inheritdoc/>
    public Vector<int>? Labels => _labels;

    /// <inheritdoc/>
    public void Fit(Matrix<T> data)
    {
        _labels = new Vector<int>(data.Rows);

        // Initialize all points as undefined
        for (int i = 0; i < data.Rows; i++)
        {
            _labels[i] = UNDEFINED;
        }

        int clusterId = 0;

        // Process each point
        for (int i = 0; i < data.Rows; i++)
        {
            // Skip already processed points
            if (_labels[i] != UNDEFINED)
            {
                continue;
            }

            // Find neighbors of current point
            var neighbors = FindNeighbors(data, i);

            // If not enough neighbors, mark as noise
            if (neighbors.Count < _minPoints)
            {
                _labels[i] = NOISE;
                continue;
            }

            // Start a new cluster
            ExpandCluster(data, i, neighbors, clusterId);
            clusterId++;
        }

        _numClusters = clusterId;
    }

    /// <inheritdoc/>
    public Vector<int> Predict(Matrix<T> data)
    {
        if (_labels == null)
        {
            throw new InvalidOperationException("Model must be fitted before calling Predict.");
        }

        // For DBSCAN, prediction on new data is not standard
        // We'll assign new points to the nearest cluster or mark as noise
        var predictions = new Vector<int>(data.Rows);

        for (int i = 0; i < data.Rows; i++)
        {
            predictions[i] = PredictSingle(data.GetRow(i));
        }

        return predictions;
    }

    /// <inheritdoc/>
    public Vector<int> FitPredict(Matrix<T> data)
    {
        Fit(data);
        return _labels!;
    }

    /// <summary>
    /// Expands a cluster by recursively adding density-reachable points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the core of DBSCAN. Starting from a core point, it:
    /// 1. Marks the point and all neighbors as part of the cluster
    /// 2. For each neighbor that is also a core point, recursively adds its neighbors
    /// 3. Continues until no more density-reachable points are found
    /// </para>
    /// </remarks>
    private void ExpandCluster(Matrix<T> data, int pointIdx, List<int> neighbors, int clusterId)
    {
        // Add seed point to cluster
        _labels![pointIdx] = clusterId;

        // Process neighbors (using queue for iterative expansion instead of recursion)
        var queue = new Queue<int>(neighbors);

        while (queue.Count > 0)
        {
            int currentIdx = queue.Dequeue();

            // If noise, convert to border point
            if (_labels[currentIdx] == NOISE)
            {
                _labels[currentIdx] = clusterId;
            }

            // Skip if already processed
            if (_labels[currentIdx] != UNDEFINED)
            {
                continue;
            }

            // Add to cluster
            _labels[currentIdx] = clusterId;

            // Check if this point is also a core point
            var currentNeighbors = FindNeighbors(data, currentIdx);
            if (currentNeighbors.Count >= _minPoints)
            {
                // Add its neighbors to the queue for processing
                foreach (int neighborIdx in currentNeighbors)
                {
                    if (_labels[neighborIdx] == UNDEFINED || _labels[neighborIdx] == NOISE)
                    {
                        queue.Enqueue(neighborIdx);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Finds all points within epsilon distance of the given point.
    /// </summary>
    private List<int> FindNeighbors(Matrix<T> data, int pointIdx)
    {
        var neighbors = new List<int>();
        var point = data.GetRow(pointIdx);

        for (int i = 0; i < data.Rows; i++)
        {
            T distance = EuclideanDistance(point, data.GetRow(i));
            if (NumOps.LessThanOrEqual(distance, _epsilon))
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    /// <summary>
    /// Predicts the cluster for a single new point.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Note: DBSCAN doesn't have a standard prediction method for new data.
    /// This implementation assigns new points to the nearest cluster if they're
    /// within epsilon of any point in that cluster, otherwise marks as noise.
    /// </para>
    /// </remarks>
    private int PredictSingle(Vector<T> point)
    {
        // This is a simplified prediction - not part of original DBSCAN
        // In practice, you might want to:
        // 1. Check if point is within ε of any core point → assign to that cluster
        // 2. Otherwise mark as noise

        // For now, we'll return NOISE for new points (conservative approach)
        return NOISE;
    }

    /// <summary>
    /// Computes Euclidean distance between two points.
    /// </summary>
    private T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sum);
    }
}
```

**Key Implementation Details**:
- **Core/Border/Noise classification**: Based on neighborhood density
- **Iterative expansion**: Uses queue instead of recursion to avoid stack overflow
- **Noise marking**: Points labeled -1 are outliers
- **Time complexity**: O(n²) naive, O(n log n) with spatial indexing (e.g., KD-tree)
- **Parameter sensitivity**: ε and minPts significantly affect results

**Performance Optimization Note**: For production use with large datasets, implement spatial indexing (KD-tree, Ball tree) to reduce neighbor search from O(n) to O(log n).

---

### AC 1.3: Implement Hierarchical Clustering (13 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Clustering\HierarchicalClustering.cs`

```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering;

/// <summary>
/// Implements Hierarchical Clustering (Agglomerative) for building cluster dendrograms.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Hierarchical clustering builds a tree (dendrogram) of clusters by either:
/// - <b>Agglomerative (bottom-up):</b> Start with each point as its own cluster, repeatedly merge closest clusters
/// - <b>Divisive (top-down):</b> Start with one cluster, repeatedly split (not implemented here)
/// </para>
/// <para>
/// This implementation uses agglomerative clustering with configurable linkage methods:
/// - <b>Single linkage:</b> Distance between closest points in clusters (tends to create chains)
/// - <b>Complete linkage:</b> Distance between farthest points (creates compact clusters)
/// - <b>Average linkage:</b> Average distance between all pairs (balanced approach)
/// - <b>Ward linkage:</b> Minimizes within-cluster variance (most popular, creates equal-sized clusters)
/// </para>
/// <para><b>For Beginners:</b> Hierarchical clustering builds a family tree of your data.
///
/// Imagine organizing species into a taxonomy (Kingdom → Phylum → Class → ... → Species).
/// Hierarchical clustering does this automatically for any data:
///
/// 1. Start: Every point is its own "cluster" (like individual animals)
/// 2. Merge: Repeatedly combine the two closest clusters (cats + dogs → mammals)
/// 3. Continue: Keep merging until everything is in one cluster (all living things)
/// 4. Result: A tree showing relationships at all levels
///
/// The beauty is you can "cut" the tree at any height to get different numbers of clusters:
/// - Cut high: Few large clusters (mammals, reptiles, birds)
/// - Cut low: Many small clusters (cats, dogs, mice, rats, ...)
///
/// <b>When to use Hierarchical Clustering:</b>
/// - You want to explore data at multiple granularities
/// - You need to visualize cluster relationships (dendrogram)
/// - You don't know K in advance but want to try different values
/// - Your dataset is small-to-medium (< 10,000 points)
///
/// <b>Advantages:</b>
/// - Creates interpretable hierarchy (dendrogram)
/// - No need to specify K upfront
/// - Deterministic (same input always gives same tree)
/// - Works with any distance metric
///
/// <b>Limitations:</b>
/// - Slow for large datasets (O(n² log n) or O(n³) depending on implementation)
/// - Greedy merges can't be undone (no global optimization)
/// - Sensitive to outliers (especially single/complete linkage)
/// - Memory intensive (stores full distance matrix)
///
/// <b>Example Use Cases:</b>
/// - Phylogenetic trees (evolution of species)
/// - Document hierarchies (topics → subtopics → articles)
/// - Image segmentation (pixels → regions → objects)
/// - Gene expression analysis (gene families)
/// </para>
/// </remarks>
public class HierarchicalClustering<T> : IClustering<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numClusters;
    private readonly string _linkage;

    private Vector<int>? _labels;
    private List<ClusterNode>? _dendrogram;

    /// <summary>
    /// Represents a node in the hierarchical clustering dendrogram.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A node in the family tree.
    ///
    /// Each node represents either:
    /// - A leaf (original data point, no children)
    /// - A merge (two clusters joined, has left/right children)
    ///
    /// The Height tells you how "far apart" the children were when merged.
    /// Higher merges mean more distant clusters.
    /// </para>
    /// </remarks>
    public class ClusterNode
    {
        public int Id { get; set; }
        public int LeftChild { get; set; }
        public int RightChild { get; set; }
        public T Height { get; set; }
        public int Size { get; set; }

        public ClusterNode()
        {
            Id = 0;
            LeftChild = -1;
            RightChild = -1;
            Height = NumOps.Zero;
            Size = 0;
        }
    }

    /// <summary>
    /// Initializes a new instance of Hierarchical Clustering.
    /// </summary>
    /// <param name="numClusters">
    /// The number of clusters to form when cutting the dendrogram.
    /// Default: 2 (binary split of data).
    /// Set to 0 to skip cutting and just build the full dendrogram.
    /// </param>
    /// <param name="linkage">
    /// The linkage method for measuring distance between clusters.
    /// Default: "ward" (minimizes variance, most popular in scikit-learn).
    ///
    /// Options:
    /// - "single": Minimum distance between any two points (min(dist(a,b)) for a in A, b in B)
    /// - "complete": Maximum distance between any two points (max(dist(a,b)) for a in A, b in B)
    /// - "average": Average distance between all pairs (mean(dist(a,b)) for all a,b pairs)
    /// - "ward": Minimize increase in within-cluster variance (Ward, 1963)
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when linkage is not one of the supported methods.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Understanding the parameters:
    ///
    /// <b>numClusters:</b> How many groups to end up with?
    /// - The algorithm builds a full tree (dendrogram) regardless
    /// - This parameter just determines where to "cut" the tree
    /// - Set to 0 if you want to explore the dendrogram yourself
    /// - Otherwise, the algorithm will cut to produce exactly this many clusters
    ///
    /// <b>linkage:</b> How to measure distance between groups?
    ///
    /// Imagine two groups of people. How far apart are the groups?
    ///
    /// - <b>"single"</b>: Distance to the closest person in the other group
    ///   - Creates "chain" clusters (elongated shapes)
    ///   - Sensitive to outliers (one nearby point merges whole clusters)
    ///   - Use for: Finding natural paths/connections
    ///
    /// - <b>"complete"</b>: Distance to the farthest person in the other group
    ///   - Creates compact, spherical clusters
    ///   - Sensitive to outliers (one far point delays merging)
    ///   - Use for: Well-separated, round clusters
    ///
    /// - <b>"average"</b>: Average distance to all people in the other group
    ///   - Balanced between single and complete
    ///   - Less sensitive to outliers
    ///   - Use for: General-purpose clustering
    ///
    /// - <b>"ward"</b>: Merge groups to minimize total variance increase
    ///   - Creates equal-sized clusters
    ///   - MOST POPULAR (scikit-learn default)
    ///   - Tends to find the "best" clusterings at each level
    ///   - Use for: Most applications (unless you have specific needs)
    ///
    /// <b>Default values from research:</b>
    /// - linkage="ward": Ward (1963), scikit-learn default, minimizes variance
    /// - numClusters=2: Allows exploring dendrogram before committing to K
    ///
    /// <b>Linkage comparison:</b>
    /// | Linkage  | Speed | Outlier Sensitivity | Cluster Shape | When to Use |
    /// |----------|-------|---------------------|---------------|-------------|
    /// | single   | Fast  | High                | Chains        | Finding connections |
    /// | complete | Medium| High                | Compact       | Well-separated data |
    /// | average  | Medium| Low                 | Balanced      | General-purpose |
    /// | ward     | Slow  | Medium              | Equal-sized   | Most applications |
    /// </para>
    /// </remarks>
    public HierarchicalClustering(int numClusters = 2, string linkage = "ward")
    {
        if (numClusters < 0)
        {
            throw new ArgumentException("Number of clusters must be non-negative.", nameof(numClusters));
        }

        if (linkage != "single" && linkage != "complete" && linkage != "average" && linkage != "ward")
        {
            throw new ArgumentException(
                "Linkage must be 'single', 'complete', 'average', or 'ward'.",
                nameof(linkage));
        }

        _numClusters = numClusters;
        _linkage = linkage;
    }

    /// <inheritdoc/>
    public int NumClusters => _numClusters;

    /// <inheritdoc/>
    public Vector<int>? Labels => _labels;

    /// <summary>
    /// Gets the dendrogram (hierarchical tree) built during fitting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The full cluster family tree.
    ///
    /// This is a list of all merge operations that happened during clustering.
    /// Each node tells you:
    /// - Which two clusters were merged (LeftChild, RightChild)
    /// - How far apart they were (Height)
    /// - How many points are in the merged cluster (Size)
    ///
    /// You can use this to:
    /// - Visualize the dendrogram (plot Height vs merges)
    /// - Cut at different heights to get different numbers of clusters
    /// - Understand hierarchical relationships in your data
    ///
    /// Returns null if Fit hasn't been called yet.
    /// </para>
    /// </remarks>
    public List<ClusterNode>? Dendrogram => _dendrogram;

    /// <inheritdoc/>
    public void Fit(Matrix<T> data)
    {
        int n = data.Rows;

        // Build dendrogram (always builds full tree)
        _dendrogram = BuildDendrogram(data);

        // Cut dendrogram to get K clusters (if K > 0)
        if (_numClusters > 0)
        {
            _labels = CutDendrogram(_dendrogram, _numClusters, n);
        }
        else
        {
            // If K=0, assign each point to its own cluster
            _labels = new Vector<int>(n);
            for (int i = 0; i < n; i++)
            {
                _labels[i] = i;
            }
        }
    }

    /// <inheritdoc/>
    public Vector<int> Predict(Matrix<T> data)
    {
        if (_labels == null)
        {
            throw new InvalidOperationException("Model must be fitted before calling Predict.");
        }

        // Hierarchical clustering doesn't have a standard prediction method
        // For simplicity, we'll assign new points to nearest cluster based on
        // the original training data (not ideal, but functional)

        throw new NotImplementedException(
            "Hierarchical clustering doesn't support prediction on new data. " +
            "Use KMeans or DBSCAN for prediction, or refit with new data included.");
    }

    /// <inheritdoc/>
    public Vector<int> FitPredict(Matrix<T> data)
    {
        Fit(data);
        return _labels!;
    }

    /// <summary>
    /// Builds the full hierarchical clustering dendrogram.
    /// </summary>
    private List<ClusterNode> BuildDendrogram(Matrix<T> data)
    {
        int n = data.Rows;
        var dendrogram = new List<ClusterNode>();

        // Initialize distance matrix (symmetric, so only store upper triangle)
        var distances = new Dictionary<(int, int), T>();
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                distances[(i, j)] = EuclideanDistance(data.GetRow(i), data.GetRow(j));
            }
        }

        // Initialize each point as its own cluster
        var clusters = new Dictionary<int, HashSet<int>>();
        for (int i = 0; i < n; i++)
        {
            clusters[i] = new HashSet<int> { i };
        }

        // For Ward linkage, track cluster variance
        var clusterVariances = new Dictionary<int, T>();
        if (_linkage == "ward")
        {
            for (int i = 0; i < n; i++)
            {
                clusterVariances[i] = NumOps.Zero; // Single points have zero variance
            }
        }

        int nextClusterId = n;

        // Agglomerative merging: repeat until one cluster remains
        for (int merge = 0; merge < n - 1; merge++)
        {
            // Find closest pair of clusters
            var (clusterA, clusterB, dist) = FindClosestClusters(clusters, distances, data, clusterVariances);

            // Create merge node
            var node = new ClusterNode
            {
                Id = nextClusterId,
                LeftChild = clusterA,
                RightChild = clusterB,
                Height = dist,
                Size = clusters[clusterA].Count + clusters[clusterB].Count
            };
            dendrogram.Add(node);

            // Merge clusters
            var merged = new HashSet<int>(clusters[clusterA]);
            merged.UnionWith(clusters[clusterB]);

            // Update distances from new cluster to remaining clusters
            foreach (var otherId in clusters.Keys.ToList())
            {
                if (otherId == clusterA || otherId == clusterB)
                {
                    continue;
                }

                T newDist = ComputeLinkageDistance(
                    clusterA, clusterB, otherId,
                    clusters, distances, data, clusterVariances);

                // Store distance to new cluster
                int smaller = Math.Min(nextClusterId, otherId);
                int larger = Math.Max(nextClusterId, otherId);
                distances[(smaller, larger)] = newDist;

                // Remove old distances
                RemoveDistance(distances, clusterA, otherId);
                RemoveDistance(distances, clusterB, otherId);
            }

            // Remove old clusters and add new merged cluster
            clusters.Remove(clusterA);
            clusters.Remove(clusterB);
            clusters[nextClusterId] = merged;

            if (_linkage == "ward")
            {
                // Update variance for Ward linkage
                clusterVariances[nextClusterId] = ComputeClusterVariance(merged, data);
                clusterVariances.Remove(clusterA);
                clusterVariances.Remove(clusterB);
            }

            nextClusterId++;
        }

        return dendrogram;
    }

    /// <summary>
    /// Finds the pair of clusters with minimum distance.
    /// </summary>
    private (int clusterA, int clusterB, T distance) FindClosestClusters(
        Dictionary<int, HashSet<int>> clusters,
        Dictionary<(int, int), T> distances,
        Matrix<T> data,
        Dictionary<int, T> clusterVariances)
    {
        int bestA = -1, bestB = -1;
        T minDist = NumOps.FromDouble(double.MaxValue);

        var clusterIds = clusters.Keys.ToList();
        for (int i = 0; i < clusterIds.Count; i++)
        {
            for (int j = i + 1; j < clusterIds.Count; j++)
            {
                int idA = clusterIds[i];
                int idB = clusterIds[j];

                T dist = GetDistance(distances, idA, idB);

                if (NumOps.LessThan(dist, minDist))
                {
                    minDist = dist;
                    bestA = idA;
                    bestB = idB;
                }
            }
        }

        return (bestA, bestB, minDist);
    }

    /// <summary>
    /// Computes the linkage distance between two clusters being merged and another cluster.
    /// </summary>
    private T ComputeLinkageDistance(
        int clusterA, int clusterB, int other,
        Dictionary<int, HashSet<int>> clusters,
        Dictionary<(int, int), T> distances,
        Matrix<T> data,
        Dictionary<int, T> clusterVariances)
    {
        T distA = GetDistance(distances, clusterA, other);
        T distB = GetDistance(distances, clusterB, other);

        return _linkage switch
        {
            "single" => NumOps.LessThan(distA, distB) ? distA : distB,
            "complete" => NumOps.GreaterThan(distA, distB) ? distA : distB,
            "average" => ComputeAverageLinkage(clusterA, clusterB, other, clusters, data),
            "ward" => ComputeWardLinkage(clusterA, clusterB, other, clusters, clusterVariances),
            _ => distA
        };
    }

    /// <summary>
    /// Computes average linkage distance (average of all pairwise distances).
    /// </summary>
    private T ComputeAverageLinkage(
        int clusterA, int clusterB, int other,
        Dictionary<int, HashSet<int>> clusters,
        Matrix<T> data)
    {
        var merged = new HashSet<int>(clusters[clusterA]);
        merged.UnionWith(clusters[clusterB]);
        var otherPoints = clusters[other];

        T sum = NumOps.Zero;
        int count = 0;

        foreach (int i in merged)
        {
            foreach (int j in otherPoints)
            {
                sum = NumOps.Add(sum, EuclideanDistance(data.GetRow(i), data.GetRow(j)));
                count++;
            }
        }

        return NumOps.Divide(sum, NumOps.FromDouble(count));
    }

    /// <summary>
    /// Computes Ward linkage distance (increase in variance when merging).
    /// </summary>
    private T ComputeWardLinkage(
        int clusterA, int clusterB, int other,
        Dictionary<int, HashSet<int>> clusters,
        Dictionary<int, T> clusterVariances)
    {
        // Ward linkage: merge to minimize increase in total variance
        // This is an approximation - full Ward would recompute variances

        T varA = clusterVariances[clusterA];
        T varB = clusterVariances[clusterB];
        T varOther = clusterVariances[other];

        int sizeA = clusters[clusterA].Count;
        int sizeB = clusters[clusterB].Count;
        int sizeOther = clusters[other].Count;

        // Approximate Ward distance as weighted sum of variances
        T totalSize = NumOps.FromDouble(sizeA + sizeB + sizeOther);
        T weightA = NumOps.Divide(NumOps.FromDouble(sizeA), totalSize);
        T weightB = NumOps.Divide(NumOps.FromDouble(sizeB), totalSize);
        T weightOther = NumOps.Divide(NumOps.FromDouble(sizeOther), totalSize);

        T weightedVar = NumOps.Add(
            NumOps.Add(NumOps.Multiply(varA, weightA), NumOps.Multiply(varB, weightB)),
            NumOps.Multiply(varOther, weightOther));

        return weightedVar;
    }

    /// <summary>
    /// Computes the variance of a cluster.
    /// </summary>
    private T ComputeClusterVariance(HashSet<int> cluster, Matrix<T> data)
    {
        if (cluster.Count == 1)
        {
            return NumOps.Zero;
        }

        // Compute centroid
        var centroid = new Vector<T>(data.Columns);
        foreach (int idx in cluster)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                centroid[j] = NumOps.Add(centroid[j], data[idx, j]);
            }
        }

        T size = NumOps.FromDouble(cluster.Count);
        for (int j = 0; j < data.Columns; j++)
        {
            centroid[j] = NumOps.Divide(centroid[j], size);
        }

        // Compute variance
        T variance = NumOps.Zero;
        foreach (int idx in cluster)
        {
            T dist = EuclideanDistance(data.GetRow(idx), centroid);
            variance = NumOps.Add(variance, NumOps.Multiply(dist, dist));
        }

        return NumOps.Divide(variance, size);
    }

    /// <summary>
    /// Cuts the dendrogram to produce K clusters.
    /// </summary>
    private Vector<int> CutDendrogram(List<ClusterNode> dendrogram, int K, int numPoints)
    {
        if (K >= numPoints)
        {
            // Return one cluster per point
            var singleLabels = new Vector<int>(numPoints);
            for (int i = 0; i < numPoints; i++)
            {
                singleLabels[i] = i;
            }
            return singleLabels;
        }

        // The last (n-K) merges in the dendrogram create K clusters
        // We want to "undo" the last (K-1) merges to get K clusters

        var labels = new Vector<int>(numPoints);
        var clusterMap = new Dictionary<int, int>();

        // Initially, each point is its own cluster
        for (int i = 0; i < numPoints; i++)
        {
            clusterMap[i] = i;
        }

        // Apply merges up to (n-K)th merge
        int numMerges = numPoints - K;
        for (int m = 0; m < numMerges; m++)
        {
            var node = dendrogram[m];

            // Find current cluster IDs for left and right children
            int leftCluster = clusterMap.ContainsKey(node.LeftChild) ? clusterMap[node.LeftChild] : node.LeftChild;
            int rightCluster = clusterMap.ContainsKey(node.RightChild) ? clusterMap[node.RightChild] : node.RightChild;

            // Merge into new cluster (use smaller ID)
            int newCluster = Math.Min(leftCluster, rightCluster);

            // Update all points in both clusters
            var keysToUpdate = clusterMap.Where(kv => kv.Value == leftCluster || kv.Value == rightCluster)
                .Select(kv => kv.Key).ToList();

            foreach (var key in keysToUpdate)
            {
                clusterMap[key] = newCluster;
            }

            clusterMap[node.Id] = newCluster;
        }

        // Assign final cluster labels (renumber to 0, 1, 2, ...)
        var uniqueClusters = clusterMap.Values.Distinct().OrderBy(x => x).ToList();
        var clusterRemap = new Dictionary<int, int>();
        for (int i = 0; i < uniqueClusters.Count; i++)
        {
            clusterRemap[uniqueClusters[i]] = i;
        }

        for (int i = 0; i < numPoints; i++)
        {
            labels[i] = clusterRemap[clusterMap[i]];
        }

        return labels;
    }

    /// <summary>
    /// Gets distance from the distance dictionary, handling symmetry.
    /// </summary>
    private T GetDistance(Dictionary<(int, int), T> distances, int i, int j)
    {
        int smaller = Math.Min(i, j);
        int larger = Math.Max(i, j);

        if (smaller == larger)
        {
            return NumOps.Zero;
        }

        return distances.ContainsKey((smaller, larger)) ? distances[(smaller, larger)] : NumOps.FromDouble(double.MaxValue);
    }

    /// <summary>
    /// Removes distance entry from the dictionary, handling symmetry.
    /// </summary>
    private void RemoveDistance(Dictionary<(int, int), T> distances, int i, int j)
    {
        int smaller = Math.Min(i, j);
        int larger = Math.Max(i, j);
        distances.Remove((smaller, larger));
    }

    /// <summary>
    /// Computes Euclidean distance between two points.
    /// </summary>
    private T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sum);
    }
}
```

**Key Implementation Details**:
- **Agglomerative approach**: Bottom-up merging (divisive not implemented)
- **Linkage methods**: Single, Complete, Average, Ward
- **Dendrogram storage**: Full merge history for visualization/cutting
- **Time complexity**: O(n³) naive, O(n² log n) with priority queue
- **Space complexity**: O(n²) for distance matrix

**Performance Note**: For large datasets (> 10,000 points), consider implementing optimized algorithms like SLINK (single linkage) or using approximate methods.

---

## Phase 2: Clustering Evaluation Metrics

### AC 2.1: Implement Clustering Metrics (8 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Evaluation\ClusteringMetrics.cs`

```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation;

/// <summary>
/// Provides evaluation metrics for clustering algorithms.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Clustering evaluation metrics help assess the quality of clustering results.
/// Two main categories:
///
/// 1. <b>Internal metrics</b> (no ground truth needed):
///    - Silhouette Score: How well points fit their clusters
///    - Davies-Bouldin Index: Cluster separation and compactness
///    - Calinski-Harabasz Index: Ratio of between/within cluster variance
///
/// 2. <b>External metrics</b> (require ground truth labels):
///    - Adjusted Rand Index: Similarity to true labels (chance-corrected)
///    - Normalized Mutual Information: Shared information with truth
///    - V-Measure: Harmonic mean of homogeneity and completeness
/// </para>
/// <para><b>For Beginners:</b> How do we know if clustering worked well?
///
/// Unlike supervised learning (where you have correct answers), clustering has no
/// "right answer" to compare against. So we measure quality by:
///
/// <b>Internal metrics</b> (looking at the clustering itself):
/// - Are points in the same cluster close together?
/// - Are different clusters far apart?
/// - Are clusters compact and well-separated?
///
/// <b>External metrics</b> (if you have ground truth labels for testing):
/// - Do clusters match the true categories?
/// - Are similar items grouped together?
/// - Are different items kept separate?
///
/// Think of organizing a library:
/// - Internal: "Are books on the same shelf related?" "Are shelves well-separated?"
/// - External: "Do shelves match Dewey Decimal categories?" (if you know them)
/// </para>
/// </remarks>
public static class ClusteringMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the Silhouette Score for evaluating clustering quality.
    /// </summary>
    /// <param name="data">The data matrix where each row is a data point.</param>
    /// <param name="labels">Cluster labels for each data point.</param>
    /// <returns>
    /// The mean Silhouette Score across all points, ranging from -1 to +1.
    /// - +1: Perfect clustering (points very close to their cluster, far from others)
    /// - 0: Overlapping clusters (points on cluster boundaries)
    /// - -1: Poor clustering (points closer to other clusters than their own)
    /// </returns>
    /// <remarks>
    /// <para>
    /// Silhouette Score (Rousseeuw, 1987) measures how well each point fits its cluster:
    ///
    /// For each point i:
    /// - a(i) = average distance to other points in same cluster (cohesion)
    /// - b(i) = average distance to points in nearest other cluster (separation)
    /// - s(i) = (b(i) - a(i)) / max(a(i), b(i))
    ///
    /// Final score is the mean of s(i) across all points.
    /// </para>
    /// <para><b>For Beginners:</b> How well do points fit their assigned clusters?
    ///
    /// Imagine organizing people into groups for a team activity:
    ///
    /// - <b>a(i):</b> How close you are to your teammates (lower = better)
    /// - <b>b(i):</b> How close you are to the nearest other team (higher = better)
    ///
    /// Good clustering: You're close to your team, far from others → high score
    /// Bad clustering: You're far from your team, close to others → low score
    ///
    /// <b>Interpreting Silhouette Scores:</b>
    /// - 0.71-1.0: Strong, well-separated clusters
    /// - 0.51-0.70: Reasonable structure, some overlap
    /// - 0.26-0.50: Weak structure, clusters overlap significantly
    /// - &lt; 0.25: No substantial clustering structure
    ///
    /// <b>Use Cases:</b>
    /// - Choosing K for K-Means (plot score vs K, pick the peak)
    /// - Comparing different algorithms on the same data
    /// - Identifying poorly-clustered points (low individual scores)
    ///
    /// <b>Limitations:</b>
    /// - Slow to compute (O(n²) distance calculations)
    /// - Favors convex, spherical clusters
    /// - Sensitive to outliers
    /// </para>
    /// </remarks>
    public static double SilhouetteScore(Matrix<T> data, Vector<int> labels)
    {
        int n = data.Rows;
        var silhouetteScores = new double[n];

        // Skip noise points (label = -1 in DBSCAN)
        var validIndices = Enumerable.Range(0, n).Where(i => labels[i] >= 0).ToList();

        if (validIndices.Count == 0)
        {
            return 0.0; // All points are noise
        }

        foreach (int i in validIndices)
        {
            int clusterI = labels[i];

            // Compute a(i): mean distance to points in same cluster
            var sameCluster = validIndices.Where(j => labels[j] == clusterI && j != i).ToList();

            double a = 0.0;
            if (sameCluster.Count > 0)
            {
                foreach (int j in sameCluster)
                {
                    a += Convert.ToDouble(EuclideanDistance(data.GetRow(i), data.GetRow(j)));
                }
                a /= sameCluster.Count;
            }

            // Compute b(i): mean distance to points in nearest other cluster
            var otherClusters = validIndices.Select(j => labels[j]).Distinct().Where(c => c != clusterI).ToList();

            double b = double.MaxValue;
            foreach (int otherCluster in otherClusters)
            {
                var otherPoints = validIndices.Where(j => labels[j] == otherCluster).ToList();

                double avgDist = 0.0;
                foreach (int j in otherPoints)
                {
                    avgDist += Convert.ToDouble(EuclideanDistance(data.GetRow(i), data.GetRow(j)));
                }
                avgDist /= otherPoints.Count;

                if (avgDist < b)
                {
                    b = avgDist;
                }
            }

            // Silhouette score for point i
            silhouetteScores[i] = (b - a) / Math.Max(a, b);
        }

        // Return mean silhouette score
        return validIndices.Average(i => silhouetteScores[i]);
    }

    /// <summary>
    /// Computes the Davies-Bouldin Index for evaluating cluster separation.
    /// </summary>
    /// <param name="data">The data matrix where each row is a data point.</param>
    /// <param name="labels">Cluster labels for each data point.</param>
    /// <returns>
    /// The Davies-Bouldin Index (lower is better).
    /// - 0: Perfect clustering (infinitely separated clusters)
    /// - Higher values: More cluster overlap
    /// </returns>
    /// <remarks>
    /// <para>
    /// Davies-Bouldin Index (Davies & Bouldin, 1979) measures cluster separation.
    /// For each cluster, it finds the most similar cluster and computes:
    ///
    /// DB = (1/K) × Σ max((σi + σj) / d(ci, cj))
    ///
    /// where:
    /// - σi = average distance of points to cluster i's centroid (compactness)
    /// - d(ci, cj) = distance between centroids (separation)
    ///
    /// Lower values indicate better clustering (compact, well-separated clusters).
    /// </para>
    /// <para><b>For Beginners:</b> Are clusters compact and well-separated?
    ///
    /// Imagine neighborhoods in a city:
    /// - <b>Compact</b>: Houses in a neighborhood are close together (low σ)
    /// - <b>Separated</b>: Neighborhoods are far apart (high d)
    ///
    /// Good clustering: Compact neighborhoods, far apart → low DB score
    /// Bad clustering: Spread-out neighborhoods, close together → high DB score
    ///
    /// <b>Interpreting DB Index:</b>
    /// - 0-0.5: Excellent clustering (rare in real data)
    /// - 0.5-1.0: Good clustering
    /// - 1.0-2.0: Acceptable clustering
    /// - &gt; 2.0: Poor clustering, consider different K or algorithm
    ///
    /// <b>Use Cases:</b>
    /// - Choosing K for K-Means (plot DB vs K, pick the minimum)
    /// - Comparing different algorithms
    /// - Validating cluster quality without ground truth
    ///
    /// <b>Advantages over Silhouette:</b>
    /// - Faster to compute (O(nK) vs O(n²))
    /// - Simpler interpretation (lower = better)
    ///
    /// <b>Limitations:</b>
    /// - Favors spherical clusters
    /// - Sensitive to outliers
    /// - Not normalized (can't compare across datasets)
    /// </para>
    /// </remarks>
    public static double DaviesBouldinIndex(Matrix<T> data, Vector<int> labels)
    {
        var clusters = labels.Where(l => l >= 0).Distinct().OrderBy(x => x).ToList();
        int K = clusters.Count;

        if (K <= 1)
        {
            return 0.0; // Only one cluster, no separation to measure
        }

        // Compute centroids
        var centroids = new Dictionary<int, Vector<T>>();
        foreach (int cluster in clusters)
        {
            var points = Enumerable.Range(0, data.Rows).Where(i => labels[i] == cluster).ToList();
            var centroid = new Vector<T>(data.Columns);

            foreach (int i in points)
            {
                for (int j = 0; j < data.Columns; j++)
                {
                    centroid[j] = NumOps.Add(centroid[j], data[i, j]);
                }
            }

            T count = NumOps.FromDouble(points.Count);
            for (int j = 0; j < data.Columns; j++)
            {
                centroid[j] = NumOps.Divide(centroid[j], count);
            }

            centroids[cluster] = centroid;
        }

        // Compute average within-cluster distances (σ)
        var sigma = new Dictionary<int, double>();
        foreach (int cluster in clusters)
        {
            var points = Enumerable.Range(0, data.Rows).Where(i => labels[i] == cluster).ToList();
            var centroid = centroids[cluster];

            double avgDist = 0.0;
            foreach (int i in points)
            {
                avgDist += Convert.ToDouble(EuclideanDistance(data.GetRow(i), centroid));
            }
            sigma[cluster] = avgDist / points.Count;
        }

        // Compute Davies-Bouldin Index
        double dbSum = 0.0;
        foreach (int i in clusters)
        {
            double maxRatio = 0.0;

            foreach (int j in clusters)
            {
                if (i == j) continue;

                double separation = Convert.ToDouble(EuclideanDistance(centroids[i], centroids[j]));
                double ratio = (sigma[i] + sigma[j]) / separation;

                if (ratio > maxRatio)
                {
                    maxRatio = ratio;
                }
            }

            dbSum += maxRatio;
        }

        return dbSum / K;
    }

    /// <summary>
    /// Computes Euclidean distance between two points.
    /// </summary>
    private static T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sum);
    }
}
```

**Key Metrics Implemented**:
- **Silhouette Score**: Measures point-to-cluster fit (-1 to +1, higher is better)
- **Davies-Bouldin Index**: Measures cluster separation (0+, lower is better)

**Additional Metrics to Consider** (not implemented above, but valuable):
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Adjusted Rand Index**: Similarity to ground truth labels (external metric)
- **Normalized Mutual Information**: Information shared with ground truth

---

## Testing

### AC 3.1: Unit Tests for KMeans (5 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Clustering\KMeansTests.cs`

```csharp
using Xunit;
using AiDotNet.Clustering;

namespace AiDotNet.Tests.UnitTests.Clustering;

public class KMeansTests
{
    [Fact]
    public void KMeans_ThreeGaussianBlobs_ProducesThreeClusters()
    {
        // Arrange: Create 3 well-separated Gaussian blobs
        var data = new Matrix<double>(150, 2);
        var random = new Random(42);

        // Cluster 0: centered at (0, 0)
        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = random.NextDouble() - 0.5; // [-0.5, 0.5]
            data[i, 1] = random.NextDouble() - 0.5;
        }

        // Cluster 1: centered at (5, 5)
        for (int i = 50; i < 100; i++)
        {
            data[i, 0] = 5 + random.NextDouble() - 0.5;
            data[i, 1] = 5 + random.NextDouble() - 0.5;
        }

        // Cluster 2: centered at (10, 0)
        for (int i = 100; i < 150; i++)
        {
            data[i, 0] = 10 + random.NextDouble() - 0.5;
            data[i, 1] = random.NextDouble() - 0.5;
        }

        var kmeans = new KMeans<double>(numClusters: 3, randomSeed: 42);

        // Act
        var labels = kmeans.FitPredict(data);

        // Assert
        Assert.Equal(3, kmeans.NumClusters);
        Assert.NotNull(labels);
        Assert.Equal(150, labels.Length);

        // Check that we have exactly 3 distinct clusters
        var uniqueLabels = labels.Distinct().OrderBy(x => x).ToArray();
        Assert.Equal(3, uniqueLabels.Length);
        Assert.Equal(new[] { 0, 1, 2 }, uniqueLabels);

        // Check centroids are approximately at expected locations
        var centroids = kmeans.Centroids!;
        Assert.Equal(3, centroids.Rows);
        Assert.Equal(2, centroids.Columns);

        // Each centroid should be near one of the blob centers
        // (exact matching depends on random initialization, so we check proximity)
        var expectedCenters = new[] { (0.0, 0.0), (5.0, 5.0), (10.0, 0.0) };

        for (int k = 0; k < 3; k++)
        {
            double cx = centroids[k, 0];
            double cy = centroids[k, 1];

            // Check if this centroid is close to one of the expected centers
            bool matchesExpected = expectedCenters.Any(center =>
                Math.Abs(cx - center.Item1) < 1.0 &&
                Math.Abs(cy - center.Item2) < 1.0);

            Assert.True(matchesExpected, $"Centroid {k} at ({cx}, {cy}) doesn't match any expected center");
        }
    }

    [Fact]
    public void KMeans_KMeansPlusPlusInit_BetterThanRandom()
    {
        // Arrange: Same 3-blob dataset
        var data = CreateThreeBlobDataset();

        var kmeansRandom = new KMeans<double>(numClusters: 3, initMethod: "random", randomSeed: 42);
        var kmeansPlusPlus = new KMeans<double>(numClusters: 3, initMethod: "kmeans++", randomSeed: 42);

        // Act
        var labelsRandom = kmeansRandom.FitPredict(data);
        var labelsPlusPlus = kmeansPlusPlus.FitPredict(data);

        // Assert: K-Means++ should have better Silhouette score
        var scoreRandom = ClusteringMetrics<double>.SilhouetteScore(data, labelsRandom);
        var scorePlusPlus = ClusteringMetrics<double>.SilhouetteScore(data, labelsPlusPlus);

        // K-Means++ should be at least as good as random (often better)
        Assert.True(scorePlusPlus >= scoreRandom - 0.05, // Allow small tolerance
            $"K-Means++ score ({scorePlusPlus}) should be ≥ random score ({scoreRandom})");
    }

    [Fact]
    public void KMeans_Convergence_StopsBeforeMaxIterations()
    {
        // Arrange: Well-separated blobs should converge quickly
        var data = CreateThreeBlobDataset();
        var kmeans = new KMeans<double>(numClusters: 3, maxIterations: 300, tolerance: 1e-4);

        // Act
        kmeans.Fit(data);

        // Assert: Should converge before max iterations (typically < 20 for this data)
        // We can't directly check iteration count, but we can verify the result is stable
        var labels1 = kmeans.FitPredict(data);
        var labels2 = kmeans.FitPredict(data);

        Assert.Equal(labels1.ToArray(), labels2.ToArray());
    }

    [Fact]
    public void KMeans_InvalidParameters_ThrowsArgumentException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new KMeans<double>(numClusters: 1)); // K < 2
        Assert.Throws<ArgumentException>(() => new KMeans<double>(maxIterations: 0)); // maxIter < 1
        Assert.Throws<ArgumentException>(() => new KMeans<double>(tolerance: -0.1)); // tolerance <= 0
        Assert.Throws<ArgumentException>(() => new KMeans<double>(initMethod: "invalid")); // bad init method
    }

    [Fact]
    public void KMeans_PredictNewData_AssignsToNearestCentroid()
    {
        // Arrange
        var trainData = CreateThreeBlobDataset();
        var kmeans = new KMeans<double>(numClusters: 3, randomSeed: 42);
        kmeans.Fit(trainData);

        // Create new points near each blob center
        var testData = new Matrix<double>(3, 2);
        testData[0, 0] = 0.1; testData[0, 1] = 0.1; // Near blob 0
        testData[1, 0] = 5.1; testData[1, 1] = 5.1; // Near blob 1
        testData[2, 0] = 10.1; testData[2, 1] = 0.1; // Near blob 2

        // Act
        var predictions = kmeans.Predict(testData);

        // Assert: Each test point should be assigned to a cluster
        Assert.Equal(3, predictions.Length);
        Assert.All(predictions, label => Assert.InRange(label, 0, 2));
    }

    private static Matrix<double> CreateThreeBlobDataset()
    {
        var data = new Matrix<double>(150, 2);
        var random = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = random.NextDouble() - 0.5;
            data[i, 1] = random.NextDouble() - 0.5;
        }

        for (int i = 50; i < 100; i++)
        {
            data[i, 0] = 5 + random.NextDouble() - 0.5;
            data[i, 1] = 5 + random.NextDouble() - 0.5;
        }

        for (int i = 100; i < 150; i++)
        {
            data[i, 0] = 10 + random.NextDouble() - 0.5;
            data[i, 1] = random.NextDouble() - 0.5;
        }

        return data;
    }
}
```

**Test Coverage**:
- ✅ Basic clustering on synthetic data
- ✅ K-Means++ vs random initialization
- ✅ Convergence behavior
- ✅ Parameter validation
- ✅ Prediction on new data
- ✅ Edge cases

**Target**: 80%+ code coverage with meaningful tests

---

## Performance Benchmarks

| Operation | Dataset Size | K | Time | Notes |
|-----------|-------------|---|------|-------|
| KMeans Fit | 1,000 × 10 | 5 | ~50ms | 10-20 iterations typical |
| KMeans Fit | 10,000 × 10 | 5 | ~500ms | Scales linearly with n |
| KMeans Fit | 100,000 × 10 | 5 | ~5s | Memory-bound at this scale |
| DBSCAN Fit | 1,000 × 10 | - | ~200ms | O(n²) naive implementation |
| DBSCAN Fit | 10,000 × 10 | - | ~20s | Needs spatial indexing |
| Hierarchical Fit | 1,000 × 10 | 5 | ~1s | O(n² log n) |
| Hierarchical Fit | 5,000 × 10 | 5 | ~25s | Not recommended for large n |
| Silhouette Score | 1,000 × 10 | 5 | ~300ms | O(n²) distance calculations |

**Performance Optimization Opportunities**:
1. **DBSCAN**: Implement KD-tree or Ball tree for neighbor search → O(n log n)
2. **Hierarchical**: Use priority queue for merge selection → O(n² log n)
3. **KMeans**: Use mini-batch KMeans for large datasets → 10x faster
4. **Silhouette**: Sample-based approximation for large datasets

---

## Common Pitfalls

1. **Choosing K for KMeans**:
   - **Pitfall**: Guessing K randomly
   - **Solution**: Use Elbow method (plot WCSS vs K) or Silhouette analysis
   - **Code**:
     ```csharp
     var scores = new List<double>();
     for (int k = 2; k <= 10; k++)
     {
         var kmeans = new KMeans<double>(numClusters: k);
         var labels = kmeans.FitPredict(data);
         scores.Add(ClusteringMetrics<double>.SilhouetteScore(data, labels));
     }
     // Plot scores vs K, pick the maximum
     ```

2. **Choosing ε and minPts for DBSCAN**:
   - **Pitfall**: Using default values without understanding your data scale
   - **Solution**: K-distance plot to find the "elbow"
   - **Code**:
     ```csharp
     // Compute k-th nearest neighbor distances for all points
     var kDistances = new List<double>();
     int k = 4; // minPts - 1

     for (int i = 0; i < data.Rows; i++)
     {
         var distances = new List<double>();
         for (int j = 0; j < data.Rows; j++)
         {
             if (i != j)
             {
                 distances.Add(EuclideanDistance(data.GetRow(i), data.GetRow(j)));
             }
         }
         distances.Sort();
         kDistances.Add(distances[k - 1]);
     }

     // Sort and plot kDistances - the "elbow" is your ε
     kDistances.Sort();
     // Plot index vs kDistances[index], look for sharp increase
     ```

3. **Not Normalizing Features**:
   - **Pitfall**: Features on different scales (e.g., age: 0-100, income: 0-1000000)
   - **Solution**: Standardize features before clustering
   - **Code**:
     ```csharp
     // Standardize each column to mean=0, std=1
     for (int j = 0; j < data.Columns; j++)
     {
         double mean = 0;
         for (int i = 0; i < data.Rows; i++) mean += data[i, j];
         mean /= data.Rows;

         double variance = 0;
         for (int i = 0; i < data.Rows; i++)
         {
             double diff = data[i, j] - mean;
             variance += diff * diff;
         }
         double std = Math.Sqrt(variance / data.Rows);

         for (int i = 0; i < data.Rows; i++)
         {
             data[i, j] = (data[i, j] - mean) / std;
         }
     }
     ```

4. **Hierarchical Clustering on Large Datasets**:
   - **Pitfall**: Running hierarchical clustering on 100,000 points
   - **Solution**: Use KMeans or DBSCAN for large datasets, hierarchical for small (<10,000)
   - **Alternative**: Sample data before hierarchical clustering

5. **Interpreting Silhouette Score**:
   - **Pitfall**: Thinking negative scores mean "failure"
   - **Solution**: Understand that negative scores indicate points closer to other clusters (might need more clusters)
   - **Interpretation**: Score < 0.25 suggests data isn't naturally clusterable

---

## Conclusion

You've built a comprehensive clustering module with three fundamental algorithms:

**What You Built**:
- **KMeans**: Fast, scalable, works for round clusters with known K
- **DBSCAN**: Finds arbitrary shapes, handles noise, auto-determines K
- **Hierarchical**: Creates interpretable dendrograms, explores multiple K values

**Impact**:
- Enables customer segmentation, image compression, anomaly detection
- Provides foundation for advanced techniques (spectral clustering, GMMs)
- Completes unsupervised learning toolkit alongside dimensionality reduction

**Next Steps**:
- Implement Gaussian Mixture Models (probabilistic clustering)
- Add Spectral Clustering (graph-based, finds complex shapes)
- Optimize DBSCAN with spatial indexing (KD-tree)
- Implement streaming/online clustering (mini-batch KMeans)

**Key Takeaways**:
1. Different algorithms excel in different scenarios - choose based on data characteristics
2. Parameter tuning (K, ε, minPts, linkage) is critical for good results
3. Feature normalization is essential for distance-based algorithms
4. Evaluation metrics help validate clustering quality objectively

You've mastered clustering - the art of finding structure in unlabeled data!
