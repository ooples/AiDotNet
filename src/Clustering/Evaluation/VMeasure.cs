using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Computes the V-Measure for cluster-label agreement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// V-Measure is the harmonic mean of homogeneity and completeness, providing
/// a single metric that balances both properties. It requires ground truth labels.
/// </para>
/// <para>
/// Components:
/// - Homogeneity: Each cluster contains only members of a single class
/// - Completeness: All members of a given class are assigned to the same cluster
/// - V-Measure: 2 * (homogeneity * completeness) / (homogeneity + completeness)
/// </para>
/// <para><b>For Beginners:</b> V-Measure combines two important properties:
///
/// Homogeneity: "Is each cluster pure?"
/// - A cluster is pure if it contains only one type of item
/// - Example: A "cats" cluster should have only cats
///
/// Completeness: "Is each class together?"
/// - All items of the same type should be in the same cluster
/// - Example: All cats should be in the same cluster
///
/// V-Measure balances these:
/// - High homogeneity + low completeness = many small pure clusters
/// - Low homogeneity + high completeness = one big impure cluster
/// - High V-Measure = both are high (ideal)
///
/// V-Measure ranges from 0 to 1, where 1 is perfect agreement.
/// </para>
/// </remarks>
public class VMeasure<T> : IClusterMetric<T>, IExternalClusterMetric<T>
{
    private readonly double _beta;

    /// <summary>
    /// Initializes a new VMeasure instance.
    /// </summary>
    /// <param name="beta">Weight for completeness vs homogeneity. Default is 1.0 (equal weight).</param>
    public VMeasure(double beta = 1.0)
    {
        _beta = beta;
    }

    /// <inheritdoc />
    public string Name => "V-Measure";

    /// <inheritdoc />
    public bool HigherIsBetter => true;

    /// <summary>
    /// Computes V-Measure comparing predicted labels to true labels.
    /// </summary>
    /// <param name="data">The data matrix (not used, can be null).</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <param name="trueLabels">The ground truth class labels.</param>
    /// <returns>V-Measure score between 0 and 1.</returns>
    public double ComputeWithTrueLabels(Matrix<T>? data, Vector<T> predictedLabels, Vector<T> trueLabels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = predictedLabels.Length;

        if (n != trueLabels.Length)
        {
            throw new ArgumentException("Predicted and true labels must have the same length.");
        }

        // Build contingency table
        var trueClasses = new Dictionary<int, int>();
        var predClusters = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)numOps.ToDouble(predictedLabels[i]);

            if (!trueClasses.ContainsKey(trueLabel))
                trueClasses[trueLabel] = trueClasses.Count;
            if (!predClusters.ContainsKey(predLabel))
                predClusters[predLabel] = predClusters.Count;
        }

        int numClasses = trueClasses.Count;
        int numClusters = predClusters.Count;

        // Contingency matrix: contingency[cluster][class] = count
        var contingency = new int[numClusters, numClasses];
        var clusterSizes = new int[numClusters];
        var classSizes = new int[numClasses];

        for (int i = 0; i < n; i++)
        {
            int trueIdx = trueClasses[(int)numOps.ToDouble(trueLabels[i])];
            int predIdx = predClusters[(int)numOps.ToDouble(predictedLabels[i])];

            contingency[predIdx, trueIdx]++;
            clusterSizes[predIdx]++;
            classSizes[trueIdx]++;
        }

        // Compute entropy of classes H(C)
        double entropyC = 0;
        for (int c = 0; c < numClasses; c++)
        {
            if (classSizes[c] > 0)
            {
                double p = (double)classSizes[c] / n;
                entropyC -= p * Math.Log(p);
            }
        }

        // Compute entropy of clusters H(K)
        double entropyK = 0;
        for (int k = 0; k < numClusters; k++)
        {
            if (clusterSizes[k] > 0)
            {
                double p = (double)clusterSizes[k] / n;
                entropyK -= p * Math.Log(p);
            }
        }

        // Compute conditional entropy H(C|K)
        double entropyCGivenK = 0;
        for (int k = 0; k < numClusters; k++)
        {
            if (clusterSizes[k] > 0)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    if (contingency[k, c] > 0)
                    {
                        double p = (double)contingency[k, c] / clusterSizes[k];
                        entropyCGivenK -= ((double)clusterSizes[k] / n) * p * Math.Log(p);
                    }
                }
            }
        }

        // Compute conditional entropy H(K|C)
        double entropyKGivenC = 0;
        for (int c = 0; c < numClasses; c++)
        {
            if (classSizes[c] > 0)
            {
                for (int k = 0; k < numClusters; k++)
                {
                    if (contingency[k, c] > 0)
                    {
                        double p = (double)contingency[k, c] / classSizes[c];
                        entropyKGivenC -= ((double)classSizes[c] / n) * p * Math.Log(p);
                    }
                }
            }
        }

        // Compute homogeneity: h = 1 - H(C|K) / H(C)
        double homogeneity = entropyC > 0 ? 1 - entropyCGivenK / entropyC : 1.0;

        // Compute completeness: c = 1 - H(K|C) / H(K)
        double completeness = entropyK > 0 ? 1 - entropyKGivenC / entropyK : 1.0;

        // Compute V-Measure
        double vMeasure;
        if (homogeneity + completeness == 0)
        {
            vMeasure = 0;
        }
        else
        {
            vMeasure = (1 + _beta) * homogeneity * completeness /
                       (_beta * homogeneity + completeness);
        }

        return vMeasure;
    }

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        // V-Measure requires true labels; without them, return 0
        return 0;
    }

    /// <inheritdoc />
    double IExternalClusterMetric<T>.Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        return ComputeWithTrueLabels(null, predictedLabels, trueLabels);
    }

    /// <summary>
    /// Computes homogeneity score.
    /// </summary>
    public double ComputeHomogeneity(Vector<T> predictedLabels, Vector<T> trueLabels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = predictedLabels.Length;

        var trueClasses = new Dictionary<int, int>();
        var predClusters = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)numOps.ToDouble(predictedLabels[i]);

            if (!trueClasses.ContainsKey(trueLabel))
                trueClasses[trueLabel] = trueClasses.Count;
            if (!predClusters.ContainsKey(predLabel))
                predClusters[predLabel] = predClusters.Count;
        }

        int numClasses = trueClasses.Count;
        int numClusters = predClusters.Count;

        var contingency = new int[numClusters, numClasses];
        var clusterSizes = new int[numClusters];
        var classSizes = new int[numClasses];

        for (int i = 0; i < n; i++)
        {
            int trueIdx = trueClasses[(int)numOps.ToDouble(trueLabels[i])];
            int predIdx = predClusters[(int)numOps.ToDouble(predictedLabels[i])];

            contingency[predIdx, trueIdx]++;
            clusterSizes[predIdx]++;
            classSizes[trueIdx]++;
        }

        double entropyC = 0;
        for (int c = 0; c < numClasses; c++)
        {
            if (classSizes[c] > 0)
            {
                double p = (double)classSizes[c] / n;
                entropyC -= p * Math.Log(p);
            }
        }

        double entropyCGivenK = 0;
        for (int k = 0; k < numClusters; k++)
        {
            if (clusterSizes[k] > 0)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    if (contingency[k, c] > 0)
                    {
                        double p = (double)contingency[k, c] / clusterSizes[k];
                        entropyCGivenK -= ((double)clusterSizes[k] / n) * p * Math.Log(p);
                    }
                }
            }
        }

        return entropyC > 0 ? 1 - entropyCGivenK / entropyC : 1.0;
    }

    /// <summary>
    /// Computes completeness score.
    /// </summary>
    public double ComputeCompleteness(Vector<T> predictedLabels, Vector<T> trueLabels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = predictedLabels.Length;

        var trueClasses = new Dictionary<int, int>();
        var predClusters = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)numOps.ToDouble(predictedLabels[i]);

            if (!trueClasses.ContainsKey(trueLabel))
                trueClasses[trueLabel] = trueClasses.Count;
            if (!predClusters.ContainsKey(predLabel))
                predClusters[predLabel] = predClusters.Count;
        }

        int numClasses = trueClasses.Count;
        int numClusters = predClusters.Count;

        var contingency = new int[numClusters, numClasses];
        var clusterSizes = new int[numClusters];
        var classSizes = new int[numClasses];

        for (int i = 0; i < n; i++)
        {
            int trueIdx = trueClasses[(int)numOps.ToDouble(trueLabels[i])];
            int predIdx = predClusters[(int)numOps.ToDouble(predictedLabels[i])];

            contingency[predIdx, trueIdx]++;
            clusterSizes[predIdx]++;
            classSizes[trueIdx]++;
        }

        double entropyK = 0;
        for (int k = 0; k < numClusters; k++)
        {
            if (clusterSizes[k] > 0)
            {
                double p = (double)clusterSizes[k] / n;
                entropyK -= p * Math.Log(p);
            }
        }

        double entropyKGivenC = 0;
        for (int c = 0; c < numClasses; c++)
        {
            if (classSizes[c] > 0)
            {
                for (int k = 0; k < numClusters; k++)
                {
                    if (contingency[k, c] > 0)
                    {
                        double p = (double)contingency[k, c] / classSizes[c];
                        entropyKGivenC -= ((double)classSizes[c] / n) * p * Math.Log(p);
                    }
                }
            }
        }

        return entropyK > 0 ? 1 - entropyKGivenC / entropyK : 1.0;
    }
}
