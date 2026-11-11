using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.DimensionalityReduction;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for dimensionality reduction bugs fixed in the element-wise sweep. Expected values were
/// produced with scikit-learn 1.5.2 (scipy 1.18.1, numpy 2.5.4); eigenvector signs are arbitrary, so
/// embeddings are compared column by column up to sign.
/// </summary>
public class DimensionalityReductionReferenceTests
{
    private static readonly double[,] Data =
    {
        { 2.5, 2.4, 0.5 }, { 0.5, 0.7, 1.1 }, { 2.2, 2.9, 0.3 }, { 1.9, 2.2, 0.8 }, { 3.1, 3.0, 0.2 },
        { 2.3, 2.7, 0.9 }, { 2.0, 1.6, 1.4 }, { 1.0, 1.1, 0.6 }, { 1.5, 1.6, 1.0 }, { 1.1, 0.9, 1.3 },
    };

    // sklearn.decomposition.PCA(2, whiten=True, svd_solver='full').fit_transform(Data); IncrementalPCA(2,
    // whiten=True, batch_size=10) gives the same values.
    private static readonly double[,] WhitenedReference =
    {
        { 0.752545997927, -0.018461316818 }, { -1.537672564524, -0.646778032567 },
        { 0.933611604440, -1.232348036313 }, { 0.230574141332, -0.020909405772 },
        { 1.520122249926, -0.194191836135 }, { 0.740568505771, 0.627730630567 },
        { -0.208073352741, 2.017347869380 }, { -0.906823256921, -1.423957453224 },
        { -0.403945557703, 0.185333577908 }, { -1.120907767505, 0.706234002974 },
    };

    [Fact(Timeout = 120000)]
    public async Task PCA_Whiten_MatchesSklearn()
    {
        await Task.Yield();
        var result = new PCA<double>(nComponents: 2, whiten: true).FitTransform(new Matrix<double>(Data));
        AssertColumnsEqualUpToSign(WhitenedReference, result, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task IncrementalPCA_Whiten_MatchesSklearn()
    {
        await Task.Yield();
        var result = new IncrementalPCA<double>(nComponents: 2, batchSize: 10, whiten: true)
            .FitTransform(new Matrix<double>(Data));
        AssertColumnsEqualUpToSign(WhitenedReference, result, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task Isomap_Embedding_MatchesSklearn()
    {
        await Task.Yield();

        // sklearn.manifold.Isomap(n_neighbors=5, n_components=2, path_method='D').fit_transform(Data)
        var expected = new double[,]
        {
            { -0.940503266044, 0.063767984244 }, { 1.815281378785, -0.148519160586 },
            { -1.131406459882, -0.394680162389 }, { -0.261685460489, -0.097386934939 },
            { -1.811554112659, 0.034749965632 }, { -0.864205173252, 0.071272264562 },
            { 0.243270076705, 0.665743583928 }, { 1.141307903482, -0.351679250338 },
            { 0.481814701398, -0.015870286011 }, { 1.327680411957, 0.172601995896 },
        };

        var result = new Isomap<double>(nComponents: 2, nNeighbors: 5, pathAlgorithm: IsomapPathAlgorithm.Dijkstra)
            .FitTransform(new Matrix<double>(Data));
        AssertColumnsEqualUpToSign(expected, result, 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task KernelPCA_TransformOfNewRows_MatchesSklearn()
    {
        await Task.Yield();

        // sklearn.decomposition.KernelPCA(2, kernel='rbf', gamma=0.5, eigen_solver='dense')
        //     .fit(Data[:8]).transform(Data[8:])
        var expected = new double[,]
        {
            { 0.414511964062, 0.436989242824 },
            { 0.774775863460, -0.021600300997 },
        };

        var train = new Matrix<double>(Rows(0, 8));
        var test = new Matrix<double>(Rows(8, 2));
        var kpca = new KernelPCA<double>(
            nComponents: 2, kernel: AiDotNet.Preprocessing.DimensionalityReduction.KernelType.RBF, gamma: 0.5);
        kpca.Fit(train);
        AssertColumnsEqualUpToSign(expected, kpca.Transform(test), 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task TSNE_PcaInitialization_MatchesSklearnScaling()
    {
        await Task.Yield();

        // sklearn TSNE(init='pca'): pca = PCA(2, svd_solver='full').fit_transform(Data);
        // init = pca / np.std(pca[:, 0]) * 1e-4
        var expected = new double[,]
        {
            { 7.932531324976742e-05, -5.341786591712599e-07 },
            { -1.620849199816420e-04, -1.871453838431589e-05 },
            { 9.841130399978662e-05, -3.565802093941009e-05 },
            { 2.430464853823290e-05, -6.050141736638023e-07 },
            { 1.602349543889037e-04, -5.618945586099933e-06 },
            { 7.806277472079983e-05, 1.816340133590916e-05 },
            { -2.193285716835552e-05, 5.837201054307661e-05 },
            { -9.558756423613075e-05, -4.120224416130173e-05 },
            { -4.257960043499210e-05, 5.362631664999215e-06 },
            { -1.181540530763701e-04, 2.043489835997777e-05 },
        };

        var tsne = new TSNE<double>(
            nComponents: 2, perplexity: 3.0, nIter: 0, initialization: TSNEInitialization.PCA, randomState: 0);
        AssertColumnsEqualUpToSign(expected, tsne.FitTransform(new Matrix<double>(Data)), 1e-15);
    }

    [Fact(Timeout = 120000)]
    public async Task ProbabilisticPCA_Score_IsGaussianLogLikelihoodOfFittedModel()
    {
        await Task.Yield();

        // Tipping and Bishop (1999) / sklearn PCA.score_samples: x ~ N(mu, C), C = W W^T + sigma^2 I.
        // Score returns the sum over rows. The previous code used only the diagonal of C for log|C|.
        var data = new Matrix<double>(Data);
        var ppca = new ProbabilisticPCA<double>(nComponents: 1, maxIter: 200, randomState: 0);
        ppca.Fit(data);
        var w = ppca.LoadingMatrix ?? throw new InvalidOperationException("PPCA was not fitted.");
        double sigma2 = ppca.NoiseVariance;

        var c = new double[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int q = 0; q < w.Columns; q++)
                {
                    c[i, j] += w[i, q] * w[j, q];
                }
            }

            c[i, i] += sigma2;
        }

        double det = c[0, 0] * (c[1, 1] * c[2, 2] - c[1, 2] * c[2, 1])
                   - c[0, 1] * (c[1, 0] * c[2, 2] - c[1, 2] * c[2, 0])
                   + c[0, 2] * (c[1, 0] * c[2, 1] - c[1, 1] * c[2, 0]);
        var inverse = new double[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int r0 = (j + 1) % 3, r1 = (j + 2) % 3, c0 = (i + 1) % 3, c1 = (i + 2) % 3;
                inverse[i, j] = (c[r0, c0] * c[r1, c1] - c[r0, c1] * c[r1, c0]) / det;
            }
        }

        int n = Data.GetLength(0);
        var mean = new double[3];
        for (int r = 0; r < n; r++)
        {
            for (int j = 0; j < 3; j++)
            {
                mean[j] += Data[r, j] / n;
            }
        }

        double expected = 0;
        for (int r = 0; r < n; r++)
        {
            double mahalanobis = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    mahalanobis += (Data[r, i] - mean[i]) * inverse[i, j] * (Data[r, j] - mean[j]);
                }
            }

            expected += -1.5 * Math.Log(2 * Math.PI) - 0.5 * Math.Log(det) - 0.5 * mahalanobis;
        }

        Assert.Equal(expected, ppca.Score(data), 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task FastICA_InverseTransformOfTransform_ReconstructsData()
    {
        await Task.Yield();

        // sklearn FastICA: mixing_ = pinv(components_), so inverse_transform(transform(X)) == X when k == p.
        var data = new Matrix<double>(Data);
        var ica = new FastICA<double>(nComponents: 3, randomState: 0);
        var reconstructed = ica.InverseTransform(ica.FitTransform(data));
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(Data[i, j], reconstructed[i, j], 1e-8);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task HessianLLE_PlaneEmbedding_IsAffineInTrueCoordinates()
    {
        await Task.Yield();

        // sklearn LocallyLinearEmbedding(n_neighbors=8, n_components=2, method='hessian') on this plane gives an
        // embedding whose residual after an affine fit on (u, v) is 1.1e-15 (max |embedding| 0.367).
        var (plane, u, v) = Plane();
        var embedding = new HessianLLE<double>(nComponents: 2, nNeighbors: 8, randomState: 0).FitTransform(plane);
        AssertAffineIn(embedding, u, v, 1e-6);
    }

    [Fact(Timeout = 120000)]
    public async Task ModifiedLLE_MoreNeighborsThanFeatures_IsAffineInTrueCoordinates()
    {
        await Task.Yield();

        // sklearn LocallyLinearEmbedding(n_neighbors=8, n_components=2, method='modified') on this plane (3
        // features < 8 neighbours): affine-fit residual 3.1e-4 with max |embedding| 0.334.
        var (plane, u, v) = Plane();
        var embedding = new LocallyLinearEmbedding<double>(nComponents: 2, nNeighbors: 8, method: LLEMethod.Modified)
            .FitTransform(plane);
        AssertAffineIn(embedding, u, v, 1e-2);
    }

    [Fact(Timeout = 120000)]
    public async Task NCA_MoreIterations_IncreaseObjective()
    {
        await Task.Yield();

        // Goldberger et al. (2005): NCA maximises f = sum_i sum_{j in class(i), j != i} p_ij. Gradient ascent with
        // the correct sign must not lower f; the previous code stepped along -grad f.
        var x = new Matrix<double>(new double[,]
        {
            { 0.0, 3.1 }, { 0.2, -2.4 }, { 0.4, 1.7 }, { 0.1, -0.6 }, { 0.3, 2.2 }, { 0.5, -3.0 },
            { 1.0, -2.8 }, { 1.2, 2.5 }, { 1.4, -1.1 }, { 1.1, 0.4 }, { 1.3, -2.0 }, { 1.5, 2.9 },
        });
        var labels = new[] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };

        double before = NcaObjective(
            new NCA<double>(nComponents: 2, maxIter: 1, tol: 1e-12, randomState: 0).FitTransformSupervised(x, labels),
            labels);
        double after = NcaObjective(
            new NCA<double>(nComponents: 2, maxIter: 50, tol: 1e-12, randomState: 0).FitTransformSupervised(x, labels),
            labels);
        Assert.True(after > before, $"NCA objective fell from {before} to {after}.");
    }

    [Fact(Timeout = 120000)]
    public async Task LDA_Batch_DisjointVocabularies_SeparateTopics()
    {
        await Task.Yield();

        // sklearn LatentDirichletAllocation(2, learning_method='batch', max_iter=20, random_state=0) puts 0.944
        // of each topic's normalised mass on one vocabulary block. The previous batch code started every topic
        // from the same constant, so all topics stayed identical.
        var docs = new Matrix<double>(new double[,]
        {
            { 3, 2, 1, 0, 0, 0 }, { 1, 3, 2, 0, 0, 0 }, { 2, 1, 3, 0, 0, 0 }, { 2, 2, 2, 0, 0, 0 },
            { 0, 0, 0, 3, 1, 2 }, { 0, 0, 0, 1, 2, 3 }, { 0, 0, 0, 2, 3, 1 }, { 0, 0, 0, 2, 2, 2 },
        });
        var lda = new LatentDirichletAllocation<double>(
            nComponents: 2, learningMethod: LdaLearningMethod.Batch, maxIter: 20, randomState: 0);
        lda.Fit(docs);
        var components = lda.Components ?? throw new InvalidOperationException("LDA was not fitted.");

        var firstBlockShare = new double[2];
        for (int t = 0; t < 2; t++)
        {
            double total = 0, first = 0;
            for (int w = 0; w < 6; w++)
            {
                total += components[t, w];
                first += w < 3 ? components[t, w] : 0;
            }

            firstBlockShare[t] = first / total;
        }

        Assert.True(Math.Max(firstBlockShare[0], firstBlockShare[1]) > 0.9, "No topic owns the first vocabulary.");
        Assert.True(Math.Min(firstBlockShare[0], firstBlockShare[1]) < 0.1, "No topic owns the second vocabulary.");
    }

    private static double[,] Rows(int start, int count)
    {
        var rows = new double[count, Data.GetLength(1)];
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < Data.GetLength(1); j++)
            {
                rows[i, j] = Data[start + i, j];
            }
        }

        return rows;
    }

    /// <summary>6 x 6 grid (u, v) mapped linearly into 3-D: (u, v + 0.3u, 0.5u - 0.2v).</summary>
    private static (Matrix<double> Plane, double[] U, double[] V) Plane()
    {
        var plane = new double[36, 3];
        var u = new double[36];
        var v = new double[36];
        for (int i = 0; i < 36; i++)
        {
            u[i] = i % 6;
            v[i] = i / 6;
            plane[i, 0] = u[i];
            plane[i, 1] = v[i] + 0.3 * u[i];
            plane[i, 2] = 0.5 * u[i] - 0.2 * v[i];
        }

        return (new Matrix<double>(plane), u, v);
    }

    /// <summary>Least-squares fit of every embedding column on (1, u, v); max residual / max |value| below tol.</summary>
    private static void AssertAffineIn(Matrix<double> embedding, double[] u, double[] v, double relativeTolerance)
    {
        int n = u.Length;
        var basis = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            basis[i, 0] = 1;
            basis[i, 1] = u[i];
            basis[i, 2] = v[i];
        }

        for (int col = 0; col < embedding.Columns; col++)
        {
            // Normal equations A^T A c = A^T y, solved by Gaussian elimination with partial pivoting
            var m = new double[3, 4];
            double scale = 0;
            for (int i = 0; i < n; i++)
            {
                scale = Math.Max(scale, Math.Abs(embedding[i, col]));
                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++)
                    {
                        m[a, b] += basis[i, a] * basis[i, b];
                    }

                    m[a, 3] += basis[i, a] * embedding[i, col];
                }
            }

            for (int p = 0; p < 3; p++)
            {
                int pivot = p;
                for (int r = p + 1; r < 3; r++)
                {
                    if (Math.Abs(m[r, p]) > Math.Abs(m[pivot, p])) pivot = r;
                }

                for (int c = 0; c < 4; c++)
                {
                    (m[p, c], m[pivot, c]) = (m[pivot, c], m[p, c]);
                }

                for (int r = 0; r < 3; r++)
                {
                    if (r == p) continue;
                    double factor = m[r, p] / m[p, p];
                    for (int c = p; c < 4; c++)
                    {
                        m[r, c] -= factor * m[p, c];
                    }
                }
            }

            Assert.True(scale > 0, $"Embedding column {col} is all zero.");
            for (int i = 0; i < n; i++)
            {
                double fitted = m[0, 3] / m[0, 0] + m[1, 3] / m[1, 1] * u[i] + m[2, 3] / m[2, 2] * v[i];
                double residual = Math.Abs(embedding[i, col] - fitted);
                Assert.True(residual <= relativeTolerance * scale,
                    $"Column {col}, row {i}: residual {residual} exceeds {relativeTolerance} x {scale}.");
            }
        }
    }

    private static double NcaObjective(Matrix<double> y, int[] labels)
    {
        int n = y.Rows;
        double objective = 0;
        for (int i = 0; i < n; i++)
        {
            double total = 0, same = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                double d = 0;
                for (int c = 0; c < y.Columns; c++)
                {
                    double diff = y[i, c] - y[j, c];
                    d += diff * diff;
                }

                double e = Math.Exp(-d);
                total += e;
                if (labels[j] == labels[i]) same += e;
            }

            objective += total > 0 ? same / total : 0;
        }

        return objective;
    }

    private static void AssertColumnsEqualUpToSign(double[,] expected, Matrix<double> actual, double tolerance)
    {
        Assert.Equal(expected.GetLength(0), actual.Rows);
        Assert.Equal(expected.GetLength(1), actual.Columns);
        for (int col = 0; col < actual.Columns; col++)
        {
            double sign = Math.Sign(expected[0, col]) == Math.Sign(actual[0, col]) ? 1 : -1;
            for (int row = 0; row < actual.Rows; row++)
            {
                Assert.Equal(expected[row, col], sign * actual[row, col], tolerance);
            }
        }
    }
}
