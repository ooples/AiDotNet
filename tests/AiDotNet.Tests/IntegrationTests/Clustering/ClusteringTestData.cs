using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

internal readonly struct ClusteringDataset
{
    public ClusteringDataset(Matrix<double> data, Vector<double> labels)
    {
        Data = data;
        Labels = labels;
    }

    public Matrix<double> Data { get; }
    public Vector<double> Labels { get; }
}

internal static class ClusteringTestData
{
    public static ClusteringDataset CreateTwoClusterBlobs(int pointsPerCluster = 4, double spacing = 10.0)
    {
        if (pointsPerCluster <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pointsPerCluster));
        }

        int total = pointsPerCluster * 2;
        var data = new Matrix<double>(total, 2);
        var labels = new Vector<double>(total);

        for (int i = 0; i < pointsPerCluster; i++)
        {
            double offset = i * 0.2;
            data[i, 0] = offset;
            data[i, 1] = offset;
            labels[i] = 0.0;

            int row = pointsPerCluster + i;
            data[row, 0] = spacing + offset;
            data[row, 1] = spacing + offset;
            labels[row] = 1.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateThreeClusterBlobs(int pointsPerCluster = 3, double spacing = 5.0)
    {
        if (pointsPerCluster <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pointsPerCluster));
        }

        int total = pointsPerCluster * 3;
        var data = new Matrix<double>(total, 2);
        var labels = new Vector<double>(total);

        for (int i = 0; i < pointsPerCluster; i++)
        {
            double offset = i * 0.2;
            data[i, 0] = offset;
            data[i, 1] = offset;
            labels[i] = 0.0;

            int row = pointsPerCluster + i;
            data[row, 0] = spacing + offset;
            data[row, 1] = spacing + offset;
            labels[row] = 1.0;

            int row2 = (pointsPerCluster * 2) + i;
            data[row2, 0] = (spacing * 2) + offset;
            data[row2, 1] = offset;
            labels[row2] = 2.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateOverlappingClusters(int pointsPerCluster = 4, double offset = 0.5)
    {
        if (pointsPerCluster <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pointsPerCluster));
        }

        int total = pointsPerCluster * 2;
        var data = new Matrix<double>(total, 2);
        var labels = new Vector<double>(total);

        for (int i = 0; i < pointsPerCluster; i++)
        {
            double value = i * 0.3;
            data[i, 0] = value;
            data[i, 1] = value;
            labels[i] = 0.0;

            int row = pointsPerCluster + i;
            data[row, 0] = value + offset;
            data[row, 1] = value + offset;
            labels[row] = 1.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateMoons(int pointsPerMoon = 20)
    {
        if (pointsPerMoon <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pointsPerMoon));
        }

        int total = pointsPerMoon * 2;
        var data = new Matrix<double>(total, 2);
        var labels = new Vector<double>(total);

        int denom = Math.Max(pointsPerMoon - 1, 1);
        for (int i = 0; i < pointsPerMoon; i++)
        {
            double angle = Math.PI * i / denom;
            double x1 = Math.Cos(angle);
            double y1 = Math.Sin(angle);
            data[i, 0] = x1;
            data[i, 1] = y1;
            labels[i] = 0.0;

            double x2 = 1.0 - Math.Cos(angle);
            double y2 = -Math.Sin(angle) - 0.5;
            int row = pointsPerMoon + i;
            data[row, 0] = x2;
            data[row, 1] = y2;
            labels[row] = 1.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateCircles(int pointsPerCircle = 20, double innerRadius = 1.0, double outerRadius = 3.0)
    {
        if (pointsPerCircle <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pointsPerCircle));
        }

        int total = pointsPerCircle * 2;
        var data = new Matrix<double>(total, 2);
        var labels = new Vector<double>(total);

        int denom = Math.Max(pointsPerCircle, 1);
        for (int i = 0; i < pointsPerCircle; i++)
        {
            double angle = 2 * Math.PI * i / denom;
            double innerX = innerRadius * Math.Cos(angle);
            double innerY = innerRadius * Math.Sin(angle);
            data[i, 0] = innerX;
            data[i, 1] = innerY;
            labels[i] = 0.0;

            double outerX = outerRadius * Math.Cos(angle);
            double outerY = outerRadius * Math.Sin(angle);
            int row = pointsPerCircle + i;
            data[row, 0] = outerX;
            data[row, 1] = outerY;
            labels[row] = 1.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateHighDimensional(int pointsPerCluster = 5, int dimensions = 50, double spacing = 10.0)
    {
        if (pointsPerCluster <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pointsPerCluster));
        }

        if (dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions));
        }

        int total = pointsPerCluster * 2;
        var data = new Matrix<double>(total, dimensions);
        var labels = new Vector<double>(total);

        for (int i = 0; i < pointsPerCluster; i++)
        {
            for (int d = 0; d < dimensions; d++)
            {
                data[i, d] = (i * 0.01) + (d * 0.001);
            }
            labels[i] = 0.0;

            int row = pointsPerCluster + i;
            for (int d = 0; d < dimensions; d++)
            {
                data[row, d] = spacing + (i * 0.01) + (d * 0.001);
            }
            labels[row] = 1.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateImbalancedClusters(int largeClusterSize = 8, int smallClusterSize = 2, double spacing = 10.0)
    {
        if (largeClusterSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(largeClusterSize));
        }

        if (smallClusterSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(smallClusterSize));
        }

        int total = largeClusterSize + smallClusterSize;
        var data = new Matrix<double>(total, 2);
        var labels = new Vector<double>(total);

        for (int i = 0; i < largeClusterSize; i++)
        {
            double offset = i * 0.1;
            data[i, 0] = offset;
            data[i, 1] = offset;
            labels[i] = 0.0;
        }

        for (int i = 0; i < smallClusterSize; i++)
        {
            double offset = i * 0.1;
            int row = largeClusterSize + i;
            data[row, 0] = spacing + offset;
            data[row, 1] = spacing + offset;
            labels[row] = 1.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateSinglePoint()
    {
        var data = new Matrix<double>(1, 2);
        data[0, 0] = 0.0;
        data[0, 1] = 0.0;

        var labels = new Vector<double>(1);
        labels[0] = 0.0;

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateIdenticalPoints(int count, double value = 1.0)
    {
        if (count <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        var data = new Matrix<double>(count, 2);
        var labels = new Vector<double>(count);

        for (int i = 0; i < count; i++)
        {
            data[i, 0] = value;
            data[i, 1] = value;
            labels[i] = 0.0;
        }

        return new ClusteringDataset(data, labels);
    }

    public static ClusteringDataset CreateWithOutlier()
    {
        var baseData = CreateTwoClusterBlobs(pointsPerCluster: 4);
        int total = baseData.Data.Rows + 1;
        var data = new Matrix<double>(total, baseData.Data.Columns);
        var labels = new Vector<double>(total);

        for (int i = 0; i < baseData.Data.Rows; i++)
        {
            data[i, 0] = baseData.Data[i, 0];
            data[i, 1] = baseData.Data[i, 1];
            labels[i] = baseData.Labels[i];
        }

        data[total - 1, 0] = 30.0;
        data[total - 1, 1] = 30.0;
        labels[total - 1] = -1.0;

        return new ClusteringDataset(data, labels);
    }
}

internal static class ClusteringTestHelpers
{
    public static T RequireNotNull<T>(T? value, string name) where T : class
    {
        if (value is null)
        {
            throw new InvalidOperationException($"{name} was null.");
        }

        return value;
    }

    public static int CountClusters(Vector<double> labels, bool ignoreNoise = true)
    {
        var unique = new HashSet<int>();
        for (int i = 0; i < labels.Length; i++)
        {
            int label = ToLabel(labels[i]);
            if (ignoreNoise && label < 0)
            {
                continue;
            }

            unique.Add(label);
        }

        return unique.Count;
    }

    public static void AssertAllAssigned(Vector<double> labels, bool allowNoise = false)
    {
        for (int i = 0; i < labels.Length; i++)
        {
            double label = labels[i];
            Assert.False(double.IsNaN(label));
            if (!allowNoise)
            {
                Assert.True(label >= 0.0);
            }
        }
    }

    public static void AssertPairwiseAgreement(Vector<double> expected, Vector<double> actual, double minAgreement = 1.0)
    {
        Assert.Equal(expected.Length, actual.Length);

        int totalPairs = 0;
        int matchingPairs = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            for (int j = i + 1; j < expected.Length; j++)
            {
                bool expectedSame = ToLabel(expected[i]) == ToLabel(expected[j]);
                bool actualSame = ToLabel(actual[i]) == ToLabel(actual[j]);

                if (expectedSame == actualSame)
                {
                    matchingPairs++;
                }

                totalPairs++;
            }
        }

        double agreement = totalPairs == 0 ? 1.0 : (double)matchingPairs / totalPairs;
        Assert.True(agreement >= minAgreement, $"Pairwise agreement {agreement:F3} is below {minAgreement:F3}.");
    }

    private static int ToLabel(double value)
    {
        return (int)Math.Round(value, MidpointRounding.AwayFromZero);
    }
}
