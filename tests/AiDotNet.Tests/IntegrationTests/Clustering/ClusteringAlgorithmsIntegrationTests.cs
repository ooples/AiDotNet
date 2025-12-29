using AiDotNet.Clustering.AutoK;
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Ensemble;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Clustering.Neural;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Probabilistic;
using AiDotNet.Clustering.SemiSupervised;
using AiDotNet.Clustering.Spectral;
using AiDotNet.Clustering.Subspace;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

public class PartitioningClusteringIntegrationTests
{
    private const double Tolerance = 1e-3;

    [Fact]
    public void KMeans_FitPredict_AssignsExpectedClusters()
    {
        var dataset = ClusteringTestData.CreateThreeClusterBlobs();
        var options = new KMeansOptions<double>
        {
            NumClusters = 3,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 42,
            InitMethod = KMeansInitMethod.KMeansPlusPlus
        };

        var kmeans = new KMeans<double>(options);
        var labels = kmeans.FitPredict(dataset.Data);

        Assert.Equal(dataset.Data.Rows, labels.Length);
        ClusteringTestHelpers.AssertAllAssigned(labels);
        Assert.Equal(3, ClusteringTestHelpers.CountClusters(labels));
        ClusteringTestHelpers.AssertPairwiseAgreement(dataset.Labels, labels);
    }

    [Fact]
    public void KMeans_Centroids_CloseToExpectedMeans()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 42
        };

        var kmeans = new KMeans<double>(options);
        kmeans.Train(dataset.Data);

        var centers = ClusteringTestHelpers.RequireNotNull(kmeans.ClusterCenters, "ClusterCenters");

        var centerList = new List<(double X, double Y)>
        {
            (centers[0, 0], centers[0, 1]),
            (centers[1, 0], centers[1, 1])
        };

        centerList.Sort((a, b) => a.X.CompareTo(b.X));

        Assert.Equal(0.3, centerList[0].X, Tolerance);
        Assert.Equal(0.3, centerList[0].Y, Tolerance);
        Assert.Equal(10.3, centerList[1].X, Tolerance);
        Assert.Equal(10.3, centerList[1].Y, Tolerance);
    }

    [Fact]
    public void KMeans_ReproducibleWithFixedSeed()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 123
        };

        var first = new KMeans<double>(options);
        var second = new KMeans<double>(options);

        var labels1 = first.FitPredict(dataset.Data);
        var labels2 = second.FitPredict(dataset.Data);

        Assert.Equal(labels1.Length, labels2.Length);
        for (int i = 0; i < labels1.Length; i++)
        {
            Assert.Equal(labels1[i], labels2[i], Tolerance);
        }
    }

    [Fact]
    public void KMeans_Transform_ReturnsExpectedShape()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var kmeans = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 20,
            NumInitializations = 1,
            RandomState = 7
        });

        kmeans.Train(dataset.Data);
        var distances = kmeans.Transform(dataset.Data);

        Assert.Equal(dataset.Data.Rows, distances.Rows);
        Assert.Equal(2, distances.Columns);
    }

    [Fact]
    public void KMeans_DifferentDistanceMetrics_RunSuccessfully()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 20,
            NumInitializations = 1
        };

        var euclidean = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = options.NumClusters,
            MaxIterations = options.MaxIterations,
            NumInitializations = options.NumInitializations,
            DistanceMetric = new EuclideanDistance<double>()
        });

        var manhattan = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = options.NumClusters,
            MaxIterations = options.MaxIterations,
            NumInitializations = options.NumInitializations,
            DistanceMetric = new ManhattanDistance<double>()
        });

        var euclidLabels = euclidean.FitPredict(dataset.Data);
        var manhattanLabels = manhattan.FitPredict(dataset.Data);

        Assert.Equal(dataset.Data.Rows, euclidLabels.Length);
        Assert.Equal(dataset.Data.Rows, manhattanLabels.Length);
    }

    [Fact]
    public void KMeans_InitializationMethods_RunSuccessfully()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var randomOptions = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 20,
            NumInitializations = 1,
            RandomState = 21,
            InitMethod = KMeansInitMethod.Random
        };

        var plusOptions = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 20,
            NumInitializations = 1,
            RandomState = 21,
            InitMethod = KMeansInitMethod.KMeansPlusPlus
        };

        var randomModel = new KMeans<double>(randomOptions);
        var plusModel = new KMeans<double>(plusOptions);

        randomModel.Train(dataset.Data);
        plusModel.Train(dataset.Data);

        _ = ClusteringTestHelpers.RequireNotNull(randomModel.ClusterCenters, "ClusterCenters");
        _ = ClusteringTestHelpers.RequireNotNull(plusModel.ClusterCenters, "ClusterCenters");
    }

    [Fact]
    public void KMeans_Wcss_DecreasesWithMoreClusters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();

        var k1 = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 1,
            MaxIterations = 30,
            NumInitializations = 1,
            RandomState = 42
        });

        var k2 = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 30,
            NumInitializations = 1,
            RandomState = 42
        });

        k1.Train(dataset.Data);
        k2.Train(dataset.Data);

        double inertia1 = k1.Inertia;
        double inertia2 = k2.Inertia;

        Assert.True(!double.IsNaN(inertia1) && !double.IsInfinity(inertia1));
        Assert.True(!double.IsNaN(inertia2) && !double.IsInfinity(inertia2));

        Assert.True(inertia2 < inertia1);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(12)]
    public void KMeans_VaryingDatasetSizes_Converges(int pointsPerCluster)
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: pointsPerCluster);
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 30,
            NumInitializations = 1,
            RandomState = 42
        };

        var kmeans = new KMeans<double>(options);
        kmeans.Train(dataset.Data);

        Assert.True(kmeans.NumIterations <= options.MaxIterations);
        double inertia = kmeans.Inertia;
        Assert.True(!double.IsNaN(inertia) && !double.IsInfinity(inertia));
    }

    [Fact]
    public void MiniBatchKMeans_Train_ProducesValidLabels()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new MiniBatchKMeansOptions<double>
        {
            NumClusters = 2,
            BatchSize = 4,
            MaxIterations = 20,
            NumInitializations = 1,
            RandomState = 42
        };

        var miniBatch = new MiniBatchKMeans<double>(options);
        miniBatch.Train(dataset.Data);

        var labels = ClusteringTestHelpers.RequireNotNull(miniBatch.Labels, "Labels");
        Assert.Equal(dataset.Data.Rows, labels.Length);
        Assert.Equal(2, miniBatch.NumClusters);
    }

    [Fact]
    public void MiniBatchKMeans_PartialFit_InitializesCenters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new MiniBatchKMeansOptions<double>
        {
            NumClusters = 2,
            BatchSize = 4,
            MaxIterations = 5,
            NumInitializations = 1,
            RandomState = 11
        };

        var miniBatch = new MiniBatchKMeans<double>(options);
        miniBatch.PartialFit(dataset.Data);

        var centers = ClusteringTestHelpers.RequireNotNull(miniBatch.ClusterCenters, "ClusterCenters");
        Assert.Equal(2, centers.Rows);
    }

    [Fact]
    public void OnlineKMeans_Train_TracksPointsSeen()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new OnlineKMeansOptions<double>
        {
            NumClusters = 2,
            LearningRate = 0.2,
            MaxIterations = 1,
            RandomState = 99
        };

        var online = new AiDotNet.Clustering.Streaming.OnlineKMeans<double>(options);
        online.Train(dataset.Data);

        Assert.Equal(dataset.Data.Rows, online.TotalPointsSeen);
        _ = ClusteringTestHelpers.RequireNotNull(online.ClusterCenters, "ClusterCenters");
    }

    [Fact]
    public void OnlineKMeans_PartialFit_UpdatesCounts()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new OnlineKMeansOptions<double>
        {
            NumClusters = 2,
            LearningRate = 0.3,
            MaxIterations = 1,
            RandomState = 15
        };

        var online = new AiDotNet.Clustering.Streaming.OnlineKMeans<double>(options);
        online.Train(dataset.Data);

        long before = online.TotalPointsSeen;
        var point = new Vector<double>(new[] { 0.1, 0.1 });
        int cluster = online.PartialFit(point);

        Assert.True(cluster >= 0 && cluster < online.NumClusters);
        Assert.Equal(before + 1, online.TotalPointsSeen);
    }

    [Fact]
    public void KMedoids_MedoidIndices_ReferToOriginalPoints()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new KMedoidsOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 20
        };

        var kmedoids = new KMedoids<double>(options);
        kmedoids.Train(dataset.Data);

        var medoidIndices = ClusteringTestHelpers.RequireNotNull(kmedoids.MedoidIndices, "MedoidIndices");
        Assert.Equal(2, medoidIndices.Length);

        var medoids = kmedoids.GetMedoids(dataset.Data);
        for (int i = 0; i < medoids.Rows; i++)
        {
            int idx = medoidIndices[i];
            Assert.Equal(dataset.Data[idx, 0], medoids[i, 0], Tolerance);
            Assert.Equal(dataset.Data[idx, 1], medoids[i, 1], Tolerance);
        }
    }

    [Fact]
    public void FuzzyCMeans_MembershipRowsSumToOne()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new FuzzyCMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 50,
            RandomState = 42
        };

        var fcm = new FuzzyCMeans<double>(options);
        fcm.Train(dataset.Data);

        var memberships = ClusteringTestHelpers.RequireNotNull(fcm.MembershipMatrix, "MembershipMatrix");

        for (int i = 0; i < memberships.GetLength(0); i++)
        {
            double sum = 0;
            for (int j = 0; j < memberships.GetLength(1); j++)
            {
                sum += memberships[i, j];
            }
            Assert.Equal(1.0, sum, 1e-6);
        }
    }

    [Fact]
    public void SeededKMeans_RespectsSeedsWhenConstrained()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var seeds = new Dictionary<int, int>
        {
            { 0, 0 },
            { 4, 1 }
        };

        var options = new SeededKMeansOptions<double>
        {
            Seeds = seeds,
            NumClusters = 2,
            ConstrainSeeds = true,
            MaxIterations = 50
        };

        var seeded = new SeededKMeans<double>(options);
        seeded.Train(dataset.Data);

        var labels = ClusteringTestHelpers.RequireNotNull(seeded.Labels, "Labels");
        Assert.Equal(0.0, labels[0], Tolerance);
        Assert.Equal(1.0, labels[4], Tolerance);
    }

    [Fact]
    public void COPKMeans_RespectsMustAndCannotLink()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new COPKMeansOptions<double>
        {
            NumClusters = 2,
            MustLink = new List<(int, int)> { (0, 1) },
            CannotLink = new List<(int, int)> { (0, 4) },
            MaxIterations = 50
        };

        var cop = new COPKMeans<double>(options);
        cop.Train(dataset.Data);

        var labels = ClusteringTestHelpers.RequireNotNull(cop.Labels, "Labels");
        Assert.True(cop.ConstraintsSatisfied);
        Assert.Equal(0, cop.ConstraintViolations);
        Assert.Equal(labels[0], labels[1]);
        Assert.NotEqual(labels[0], labels[4]);
    }

    [Fact]
    public void GMeans_DetectsTwoClusters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 12, spacing: 12.0);
        var options = new GMeansOptions<double>
        {
            MinClusters = 1,
            MaxClusters = 4,
            RandomState = 42,
            SignificanceLevel = 0.05
        };

        var gmeans = new GMeans<double>(options);
        gmeans.Train(dataset.Data);

        Assert.Equal(2, gmeans.NumClusters);
    }

    [Fact]
    public void XMeans_DetectsTwoClusters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 12, spacing: 12.0);
        var options = new XMeansOptions<double>
        {
            MinClusters = 2,
            MaxClusters = 2,
            RandomState = 42
        };

        var xmeans = new XMeans<double>(options);
        xmeans.Train(dataset.Data);

        Assert.Equal(2, xmeans.NumClusters);
    }

    [Fact]
    public void BisectingKMeans_BuildsHierarchy()
    {
        var dataset = ClusteringTestData.CreateThreeClusterBlobs();
        var options = new BisectingKMeansOptions<double>
        {
            NumClusters = 3,
            BuildHierarchy = true,
            RandomState = 42,
            MaxIterations = 50
        };

        var bisect = new BisectingKMeans<double>(options);
        bisect.Train(dataset.Data);

        Assert.Equal(3, bisect.NumClusters);
        var hierarchy = ClusteringTestHelpers.RequireNotNull(bisect.Hierarchy, "Hierarchy");
        Assert.True(hierarchy.Count >= 1);
    }

    [Fact]
    public void CLARANS_ProducesMedoids()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new CLARANSOptions<double>
        {
            NumClusters = 2,
            NumLocal = 2,
            MaxNeighbor = 5
        };

        var clarans = new CLARANS<double>(options);
        clarans.Train(dataset.Data);

        var medoidIndices = ClusteringTestHelpers.RequireNotNull(clarans.MedoidIndices, "MedoidIndices");
        Assert.Equal(2, medoidIndices.Length);
        Assert.True(clarans.BestCost > 0);
    }

    [Fact]
    public void KMeans_SerializeDeserialize_PreservesPredictions()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 21
        };

        var model = new KMeans<double>(options);
        model.Train(dataset.Data);
        var original = model.Predict(dataset.Data);

        var payload = model.Serialize();
        var loaded = new KMeans<double>(options);
        loaded.Deserialize(payload);

        var roundTripped = loaded.Predict(dataset.Data);
        Assert.Equal(original.Length, roundTripped.Length);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], roundTripped[i], Tolerance);
        }
    }

    [Fact]
    public void KMeans_Clone_ProducesSamePredictions()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 21
        };

        var model = new KMeans<double>(options);
        model.Train(dataset.Data);
        var original = model.Predict(dataset.Data);

        var clone = (KMeans<double>)model.Clone();
        var cloned = clone.Predict(dataset.Data);

        Assert.Equal(original.Length, cloned.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], cloned[i], Tolerance);
        }
    }
}

public class DensityClusteringIntegrationTests
{
    [Fact]
    public void DBSCAN_FindsClustersAndNoise()
    {
        var dataset = ClusteringTestData.CreateWithOutlier();
        var options = new DBSCANOptions<double>
        {
            Epsilon = 1.6,
            MinPoints = 3,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var dbscan = new DBSCAN<double>(options);
        var labels = dbscan.FitPredict(dataset.Data);

        Assert.Equal(dataset.Data.Rows, labels.Length);
        Assert.Equal(2, dbscan.NumClusters);
        Assert.True(dbscan.GetNoiseCount() >= 1);
    }

    [Fact]
    public void DBSCAN_Epsilon_ControlsNoiseCount()
    {
        var dataset = ClusteringTestData.CreateWithOutlier();
        var tight = new DBSCAN<double>(new DBSCANOptions<double>
        {
            Epsilon = 0.6,
            MinPoints = 3,
            Algorithm = NeighborAlgorithm.BruteForce
        });

        var loose = new DBSCAN<double>(new DBSCANOptions<double>
        {
            Epsilon = 3.0,
            MinPoints = 3,
            Algorithm = NeighborAlgorithm.BruteForce
        });

        tight.Train(dataset.Data);
        loose.Train(dataset.Data);

        int tightNoise = tight.GetNoiseCount();
        int looseNoise = loose.GetNoiseCount();

        Assert.True(tightNoise >= looseNoise);
    }

    [Fact]
    public void DBSCAN_CoreSampleIndices_NotEmpty()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new DBSCANOptions<double>
        {
            Epsilon = 1.6,
            MinPoints = 3,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var dbscan = new DBSCAN<double>(options);
        dbscan.Train(dataset.Data);

        var cores = dbscan.GetCoreSampleIndices();
        Assert.True(cores.Length > 0);
    }

    [Fact]
    public void DBSCAN_Circles_ProducesLabels()
    {
        var dataset = ClusteringTestData.CreateCircles(pointsPerCircle: 8);
        var options = new DBSCANOptions<double>
        {
            Epsilon = 1.5,
            MinPoints = 3,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var dbscan = new DBSCAN<double>(options);
        var labels = dbscan.FitPredict(dataset.Data);

        Assert.Equal(dataset.Data.Rows, labels.Length);
        Assert.True(dbscan.NumClusters >= 1);
    }

    [Fact]
    public void HDBSCAN_ProducesProbabilitiesAndOutlierScores()
    {
        var dataset = ClusteringTestData.CreateWithOutlier();
        var options = new HDBSCANOptions<double>
        {
            MinClusterSize = 3,
            MinSamples = 3
        };

        var hdbscan = new HDBSCAN<double>(options);
        hdbscan.Train(dataset.Data);

        var probabilities = ClusteringTestHelpers.RequireNotNull(hdbscan.Probabilities, "Probabilities");
        var outlierScores = ClusteringTestHelpers.RequireNotNull(hdbscan.OutlierScores, "OutlierScores");
        Assert.Equal(dataset.Data.Rows, probabilities.Length);
        Assert.Equal(dataset.Data.Rows, outlierScores.Length);
        for (int i = 0; i < probabilities.Length; i++)
        {
            Assert.True(probabilities[i] >= 0.0 && probabilities[i] <= 1.0);
            Assert.True(outlierScores[i] >= 0.0 && outlierScores[i] <= 1.0);
        }
    }

    [Fact]
    public void OPTICS_OrderingAndReachabilityLengths()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new OPTICSOptions<double>
        {
            MinSamples = 3,
            MaxEpsilon = 2.0,
            ExtractionMethod = OPTICSExtractionMethod.DbscanStyle,
            ClusterEpsilon = 1.6,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var optics = new OPTICS<double>(options);
        optics.Train(dataset.Data);

        var ordering = ClusteringTestHelpers.RequireNotNull(optics.Ordering, "Ordering");
        var reachability = ClusteringTestHelpers.RequireNotNull(optics.ReachabilityDistances, "ReachabilityDistances");
        Assert.Equal(dataset.Data.Rows, ordering.Length);
        Assert.Equal(dataset.Data.Rows, reachability.Length);
        Assert.True(optics.NumClusters >= 1);
    }

    [Fact]
    public void OPTICS_ExtractClustersAtEpsilon_ProducesLabels()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new OPTICSOptions<double>
        {
            MinSamples = 3,
            MaxEpsilon = 2.0,
            ExtractionMethod = OPTICSExtractionMethod.DbscanStyle,
            ClusterEpsilon = 1.6,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var optics = new OPTICS<double>(options);
        optics.Train(dataset.Data);

        var labels = optics.ExtractClustersAtEpsilon(1.6);
        Assert.Equal(dataset.Data.Rows, labels.Length);
    }

    [Fact]
    public void Denclue_FindsAttractors()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new DenclueOptions<double>
        {
            Bandwidth = 1.0,
            MinDensity = 0.01,
            AttractorMergeThreshold = 0.5,
            MaxIterations = 50
        };

        var denclue = new Denclue<double>(options);
        denclue.Train(dataset.Data);

        var attractors = ClusteringTestHelpers.RequireNotNull(denclue.Attractors, "Attractors");
        _ = ClusteringTestHelpers.RequireNotNull(denclue.AttractorDensities, "AttractorDensities");
        Assert.True(attractors.Length > 0);
    }

    [Fact]
    public void MeanShift_FindsCenters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new MeanShiftOptions<double>
        {
            Bandwidth = 2.0,
            ClusterMergeThreshold = 1.0,
            BinSeeding = false,
            MaxIterations = 30
        };

        var meanShift = new MeanShift<double>(options);
        meanShift.Train(dataset.Data);

        Assert.True(meanShift.NumClusters >= 1);
        Assert.NotNull(meanShift.ClusterCenters);
    }
}

public class HierarchicalClusteringIntegrationTests
{
    [Theory]
    [InlineData(LinkageMethod.Single)]
    [InlineData(LinkageMethod.Complete)]
    [InlineData(LinkageMethod.Average)]
    [InlineData(LinkageMethod.Ward)]
    public void AgglomerativeClustering_Linkages_ProduceClusters(LinkageMethod linkage)
    {
        var dataset = ClusteringTestData.CreateThreeClusterBlobs();
        var options = new HierarchicalOptions<double>
        {
            NumClusters = 3,
            Linkage = linkage
        };

        var agg = new AgglomerativeClustering<double>(options);
        agg.Train(dataset.Data);

        Assert.Equal(3, agg.NumClusters);
        Assert.NotNull(agg.Dendrogram);
    }

    [Fact]
    public void AgglomerativeClustering_GetLabelsForNClusters_ReturnsExpectedCount()
    {
        var dataset = ClusteringTestData.CreateThreeClusterBlobs();
        var options = new HierarchicalOptions<double>
        {
            NumClusters = 2,
            Linkage = LinkageMethod.Ward
        };

        var agg = new AgglomerativeClustering<double>(options);
        agg.Train(dataset.Data);

        var labels = agg.GetLabelsForNClusters(3);
        Assert.Equal(3, ClusteringTestHelpers.CountClusters(labels, ignoreNoise: false));
    }

    [Fact]
    public void BIRCH_ComputesLeafEntriesAndCenters()
    {
        var dataset = ClusteringTestData.CreateThreeClusterBlobs();
        var options = new BIRCHOptions<double>
        {
            Threshold = 1.5,
            BranchingFactor = 10,
            NumClusters = 3,
            ComputeLabels = true
        };

        var birch = new BIRCH<double>(options);
        birch.Train(dataset.Data);

        Assert.NotNull(birch.LeafEntries);
        Assert.Equal(3, birch.NumClusters);
        Assert.NotNull(birch.ClusterCenters);
    }

    [Fact]
    public void CURE_ProducesClusters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new CUREOptions<double>
        {
            NumClusters = 2,
            NumRepresentatives = 2,
            RandomState = 42,
            MaxIterations = 10
        };

        var cure = new CURE<double>(options);
        cure.Train(dataset.Data);

        Assert.Equal(2, cure.NumClusters);
        Assert.NotNull(cure.ClusterCenters);
    }
}

public class SpectralSubspaceClusteringIntegrationTests
{
    [Fact]
    public void SpectralClustering_RbfAffinity_ClustersMoons()
    {
        var dataset = ClusteringTestData.CreateMoons(pointsPerMoon: 10);
        var options = new SpectralOptions<double>
        {
            NumClusters = 2,
            Affinity = AffinityType.RBF,
            Gamma = 1.0,
            EigenSolver = EigenSolver.Full,
            Normalization = LaplacianNormalization.Normalized,
            AssignLabels = SpectralAssignment.KMeans,
            RandomState = 42
        };

        var spectral = new SpectralClustering<double>(options);
        spectral.Train(dataset.Data);

        Assert.Equal(2, spectral.NumClusters);
        Assert.NotNull(spectral.Embedding);
    }

    [Fact]
    public void SpectralClustering_PrecomputedAffinity_Clusters()
    {
        var affinity = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
        {
            affinity[i, i] = 1.0;
        }

        affinity[0, 1] = 0.9; affinity[1, 0] = 0.9;
        affinity[2, 3] = 0.9; affinity[3, 2] = 0.9;
        affinity[0, 2] = 0.1; affinity[2, 0] = 0.1;
        affinity[0, 3] = 0.1; affinity[3, 0] = 0.1;
        affinity[1, 2] = 0.1; affinity[2, 1] = 0.1;
        affinity[1, 3] = 0.1; affinity[3, 1] = 0.1;

        var options = new SpectralOptions<double>
        {
            NumClusters = 2,
            Affinity = AffinityType.Precomputed,
            EigenSolver = EigenSolver.Full,
            AssignLabels = SpectralAssignment.KMeans,
            RandomState = 5
        };

        var spectral = new SpectralClustering<double>(options);
        spectral.Train(affinity);

        Assert.Equal(2, spectral.NumClusters);
        Assert.NotNull(spectral.AffinityMatrix);
    }

    [Fact]
    public void CLIQUE_FindsClustersInGrid()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new CLIQUEOptions<double>
        {
            NumIntervals = 2,
            MinPoints = 2,
            MaxSubspaceDimensions = 2
        };

        var clique = new CLIQUE<double>(options);
        clique.Train(dataset.Data);

        Assert.True(clique.NumClusters > 0);
        Assert.NotNull(clique.ClusterCenters);
    }

    [Fact]
    public void SUBCLU_FindsClusters()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new SUBCLUOptions<double>
        {
            Epsilon = 1.6,
            MinPoints = 2,
            MaxSubspaceDimensions = 2,
            MinClusterSize = 2
        };

        var subclu = new SUBCLU<double>(options);
        subclu.Train(dataset.Data);

        Assert.True(subclu.NumClusters > 0);
        Assert.NotNull(subclu.ClusterCenters);
    }
}

public class ProbabilisticClusteringIntegrationTests
{
    [Fact]
    public void GaussianMixtureModel_ProbabilitiesSumToOne()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new GMMOptions<double>
        {
            NumComponents = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 42
        };

        var gmm = new GaussianMixtureModel<double>(options);
        gmm.Train(dataset.Data);

        var probs = gmm.PredictProba(dataset.Data);
        for (int i = 0; i < probs.GetLength(0); i++)
        {
            double sum = 0;
            for (int j = 0; j < probs.GetLength(1); j++)
            {
                sum += probs[i, j];
            }
            Assert.Equal(1.0, sum, 1e-6);
        }
    }

    [Fact]
    public void GaussianMixtureModel_WeightsSumToOne()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new GMMOptions<double>
        {
            NumComponents = 2,
            MaxIterations = 50,
            NumInitializations = 1,
            RandomState = 42
        };

        var gmm = new GaussianMixtureModel<double>(options);
        gmm.Train(dataset.Data);

        var weights = ClusteringTestHelpers.RequireNotNull(gmm.Weights, "Weights");
        double sum = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            sum += weights[i];
        }
        Assert.Equal(1.0, sum, 1e-6);
        Assert.True(!double.IsNaN(gmm.LowerBound) && !double.IsInfinity(gmm.LowerBound));
    }

    [Fact]
    public void AffinityPropagation_ProducesExemplars()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new AffinityPropagationOptions<double>
        {
            MaxIterations = 50,
            ConvergenceIterations = 5,
            Damping = 0.7
        };

        var affinity = new AffinityPropagation<double>(options);
        affinity.Train(dataset.Data);

        _ = ClusteringTestHelpers.RequireNotNull(affinity.ExemplarIndices, "ExemplarIndices");
        _ = ClusteringTestHelpers.RequireNotNull(affinity.SimilarityMatrix, "SimilarityMatrix");
        Assert.True(affinity.NumClusters >= 1);
    }
}

public class EnsembleNeuralClusteringIntegrationTests
{
    [Fact]
    public void ConsensusClustering_ProducesCoAssociationMatrix()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new ConsensusClusteringOptions<double>
        {
            NumBaseClusterings = 5,
            NumClusters = 2,
            RandomSeed = 42
        };

        var consensus = new ConsensusClustering<double>(options);
        consensus.Train(dataset.Data);

        _ = ClusteringTestHelpers.RequireNotNull(consensus.CoAssociationMatrix, "CoAssociationMatrix");
        Assert.Equal(2, consensus.NumClusters);
    }

    [Fact]
    public void SelfOrganizingMap_ProducesUmatAndLabels()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs();
        var options = new SOMOptions<double>
        {
            GridWidth = 3,
            GridHeight = 3,
            MaxIterations = 100,
            RandomState = 7
        };

        var som = new SelfOrganizingMap<double>(options);
        som.Train(dataset.Data);

        _ = ClusteringTestHelpers.RequireNotNull(som.Weights, "Weights");
        _ = ClusteringTestHelpers.RequireNotNull(som.NeuronLabels, "NeuronLabels");
        var umat = som.GetUMatrix();
        Assert.Equal(options.GridHeight, umat.GetLength(0));
        Assert.Equal(options.GridWidth, umat.GetLength(1));
    }
}

public class ClusteringEdgeCaseIntegrationTests
{
    [Fact]
    public void KMeans_SinglePoint_Works()
    {
        var dataset = ClusteringTestData.CreateSinglePoint();
        var options = new KMeansOptions<double>
        {
            NumClusters = 1,
            MaxIterations = 10,
            NumInitializations = 1
        };

        var kmeans = new KMeans<double>(options);
        kmeans.Train(dataset.Data);

        var labels = ClusteringTestHelpers.RequireNotNull(kmeans.Labels, "Labels");
        Assert.Equal(0.0, labels[0], 1e-6);
        _ = ClusteringTestHelpers.RequireNotNull(kmeans.ClusterCenters, "ClusterCenters");
    }

    [Fact]
    public void KMeans_HighDimensionalData_ProducesLabels()
    {
        var dataset = ClusteringTestData.CreateHighDimensional(pointsPerCluster: 4, dimensions: 30);
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 30,
            NumInitializations = 1,
            RandomState = 10
        };

        var kmeans = new KMeans<double>(options);
        kmeans.Train(dataset.Data);

        var labels = ClusteringTestHelpers.RequireNotNull(kmeans.Labels, "Labels");
        Assert.Equal(dataset.Data.Rows, labels.Length);
    }

    [Fact]
    public void KMeans_ImbalancedClusters_AssignsAllPoints()
    {
        var dataset = ClusteringTestData.CreateImbalancedClusters(largeClusterSize: 10, smallClusterSize: 2);
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 30,
            NumInitializations = 1,
            RandomState = 19
        };

        var kmeans = new KMeans<double>(options);
        var labels = kmeans.FitPredict(dataset.Data);

        ClusteringTestHelpers.AssertAllAssigned(labels);
        Assert.Equal(2, ClusteringTestHelpers.CountClusters(labels));
    }

    [Fact]
    public void KMeans_LargeValueRange_ProducesLabels()
    {
        var data = new Matrix<double>(4, 2);
        data[0, 0] = 1e-9; data[0, 1] = 1e-9;
        data[1, 0] = 2e-9; data[1, 1] = 2e-9;
        data[2, 0] = 1e6; data[2, 1] = 1e6;
        data[3, 0] = 1e6 + 1.0; data[3, 1] = 1e6 + 1.0;

        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            MaxIterations = 20,
            NumInitializations = 1,
            RandomState = 3
        };

        var kmeans = new KMeans<double>(options);
        kmeans.Train(data);

        var labels = ClusteringTestHelpers.RequireNotNull(kmeans.Labels, "Labels");
        Assert.Equal(data.Rows, labels.Length);
    }

    [Fact]
    public void DBSCAN_AllIdenticalPoints_OneCluster()
    {
        var dataset = ClusteringTestData.CreateIdenticalPoints(count: 5);
        var options = new DBSCANOptions<double>
        {
            Epsilon = 0.1,
            MinPoints = 2,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var dbscan = new DBSCAN<double>(options);
        dbscan.Train(dataset.Data);

        Assert.Equal(1, dbscan.NumClusters);
        Assert.Equal(0, dbscan.GetNoiseCount());
    }

    [Fact]
    public void KMeans_WithNaN_ThrowsArgumentException()
    {
        var data = new Matrix<double>(2, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = double.NaN; data[1, 1] = 1.0;

        var options = new KMeansOptions<double>
        {
            NumClusters = 1,
            MaxIterations = 5,
            NumInitializations = 1
        };

        var kmeans = new KMeans<double>(options);
        var ex = Assert.Throws<ArgumentException>(() => kmeans.Train(data));
        Assert.Contains("NaN", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
