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

    [Fact(Timeout = 120000)]
    public async Task FastICA_Parallel_RecoversSklearnSources()
    {
        await Task.Yield();

        // t = 0.1 i (i < 60), sources (sin 2t, sign sin 3t) mixed by [[1, 1], [0.5, 2]]. Reference:
        // sklearn.decomposition.FastICA(2, algorithm='parallel', whiten='unit-variance', fun='logcosh', tol=1e-10,
        // max_iter=1000, random_state=0).fit_transform(X), each column divided by its (ddof 0) std. Other seeds
        // agree with it to 1.2e-6 up to order and sign.
        var expected = new double[,]
        {
            { -0.084956108348, -0.025138672860 }, { 0.934165951042, 0.261745076932 }, { 0.940478753393, 0.527389816965 },
            { 0.946277762771, 0.771413984539 }, { 0.951331790973, 0.984089106154 }, { 0.955439349842, 1.156936495847 },
            { 0.958436683970, 1.283065273642 }, { 0.960204299102, 1.357447083191 }, { 0.960671726001, 1.377116556472 },
            { 0.959820329832, 1.341289533657 }, { 0.957684053074, 1.251394325087 }, { -1.070746208809, 1.090597059461 },
            { -1.075148918097, 0.905329645855 }, { -1.080442823812, 0.682560401820 }, { -1.086416874640, 0.431170434126 },
            { -1.092832904026, 0.161181867479 }, { -1.099435125126, -0.116641705968 }, { -1.105960328218, -0.391224337063 },
            { -1.112148374048, -0.651619282802 }, { -1.117752564754, -0.887445418421 }, { -1.122549478940, -1.089301100086 },
            { 0.898746392375, -1.228721269175 }, { 0.896097937109, -1.340169113034 }, { 0.894705011963, -1.398783848238 },
            { 0.894623148468, -1.402228690262 }, { 0.895855610263, -1.350366304126 }, { 0.898353262986, -1.245264279505 },
            { 0.902016533103, -1.091112702452 }, { 0.906699377597, -0.894057109889 }, { 0.912215106233, -0.661953486448 },
            { 0.918343824318, -0.404055071150 }, { 0.924841199202, -0.130643459967 }, { -1.093646070097, 0.126963579378 },
            { -1.087192841596, 0.398517496673 }, { -1.081150656299, 0.652774584588 }, { -1.075760397068, 0.879598415383 },
            { -1.071236956526, 1.069946238708 }, { -1.067760669976, 1.216229487525 }, { -1.065470125990, 1.312616310326 },
            { -1.064456641330, 1.355264068653 }, { -1.064760620431, 1.342472530963 }, { -1.066369944605, 1.274751655500 },
            { 0.955873815973, 1.175218969445 }, { 0.951895759866, 1.007821097380 }, { 0.946948750767, 0.799649373793 },
            { 0.941230010317, 0.559002948380 }, { 0.934967526652, 0.295475634718 }, { 0.928410965232, 0.019573435195 },
            { 0.921821715471, -0.257704300155 }, { 0.915462469965, -0.525303383023 }, { 0.909586751762, -0.772555482433 },
            { 0.904428807206, -0.989603437431 }, { 0.900194267270, -1.167794230985 }, { -1.128042321451, -1.320441668055 },
            { -1.129967142374, -1.401438747512 }, { -1.130597730111, -1.427974080983 }, { -1.129908945120, -1.398989788460 },
            { -1.127928247084, -1.315641382220 }, { -1.124734600183, -1.181251700200 }, { -1.120455325042, -1.001178434930 },
        };

        var x = new double[60, 2];
        for (int i = 0; i < 60; i++)
        {
            double t = 0.1 * i;
            double s1 = Math.Sin(2 * t);
            double s2 = Math.Sign(Math.Sin(3 * t));
            x[i, 0] = s1 + s2;
            x[i, 1] = 0.5 * s1 + 2 * s2;
        }

        var ica = new FastICA<double>(
            nComponents: 2, algorithm: ICAAlgorithm.Parallel, maxIterations: 1000, tolerance: 1e-10, randomState: 0);
        var sources = Standardize(ica.FitTransform(new Matrix<double>(x)));

        // Match the component order by correlation, then compare up to sign
        double direct = Math.Abs(Correlation(sources, 0, expected, 0)) + Math.Abs(Correlation(sources, 1, expected, 1));
        double swapped = Math.Abs(Correlation(sources, 0, expected, 1)) + Math.Abs(Correlation(sources, 1, expected, 0));
        var order = direct >= swapped ? new[] { 0, 1 } : new[] { 1, 0 };
        for (int c = 0; c < 2; c++)
        {
            double sign = Correlation(sources, order[c], expected, c) >= 0 ? 1 : -1;
            for (int i = 0; i < 60; i++)
            {
                Assert.Equal(expected[i, c], sign * sources[i, order[c]], 1e-5);
            }
        }
    }

    [Theory(Timeout = 120000)]
    [InlineData(1.0, 0.1, 1.5769434607659205, 0.8950608782088149)]
    [InlineData(2.0, 0.5, 0.2588787407930196, 1.057499700228411)]
    public async Task UMAP_FindAbParams_MatchesUmapLearn(double spread, double minDist, double expectedA, double expectedB)
    {
        await Task.Yield();

        // umap-learn 0.6.0 umap.umap_.find_ab_params(spread, min_dist) (scipy 1.18.1 curve_fit). scipy stops at
        // its default ftol, 1.6e-7 from the tight least-squares optimum, hence the 1e-6 tolerance.
        var (a, b) = UMAP<double>.FindAbParams(spread, minDist);
        Assert.Equal(expectedA, a, 1e-6);
        Assert.Equal(expectedB, b, 1e-6);
    }

    [Fact(Timeout = 120000)]
    public async Task LaplacianEigenmaps_Rbf_MatchesSklearnSpectralEmbedding()
    {
        await Task.Yield();

        // sklearn.manifold.SpectralEmbedding(2, affinity='rbf', gamma=0.5, random_state=0).fit_transform(Data)
        // (norm_laplacian=True: eigenvectors of the normalized Laplacian scaled by D^(-1/2))
        var expected = new double[,]
        {
            { -0.153003174602, 0.040916844969 }, { 0.283193490985, 0.267735157612 },
            { -0.181120433049, 0.126434866121 }, { -0.056882259004, -0.163944630329 },
            { -0.227533456210, 0.278522881152 }, { -0.147400387874, -0.012511377851 },
            { 0.046636490441, -0.264265255972 }, { 0.209662920119, 0.104497930563 },
            { 0.094264528077, -0.163372702115 }, { 0.231562413785, 0.099632204160 },
        };

        var embedding = new LaplacianEigenmaps<double>(
            nComponents: 2, affinity: LaplacianAffinityType.RBF, gamma: 0.5).FitTransform(new Matrix<double>(Data));
        AssertColumnsEqualUpToSign(expected, embedding, 1e-8);
    }

    // Every PCA component of this fixture has a negative entry sum but a positive largest-magnitude entry, so a
    // sum-based sign rule and scikit-learn's svd_flip give opposite signs.
    private static readonly double[,] SignData =
    {
        { 3.7, 2.1, 0.2 }, { 1.7, 2.9, 0.6 }, { 0.2, 2.8, 3.5 }, { 0.7, 0.1, 2.4 },
        { 3.2, 0.8, 1.9 }, { 1.4, 2.5, 1.9 }, { 2.7, 1.1, 0.2 }, { 3.5, 0.4, 3.0 },
    };

    // sklearn.decomposition.PCA(2, svd_solver='full').fit(SignData).components_ (svd_flip, u_based_decision=False);
    // PCA(2, svd_solver='randomized', random_state=0) and IncrementalPCA(2, batch_size=8) give the same rows.
    private static readonly double[,] SignComponents =
    {
        { 0.767701768260, -0.168799646714, -0.618175277959 },
        { -0.270911202331, 0.788738488936, -0.551814023491 },
    };

    [Fact(Timeout = 120000)]
    public async Task PCA_ComponentSigns_FollowSklearnSvdFlip()
    {
        await Task.Yield();

        // sklearn.decomposition.PCA(2, svd_solver='full').fit_transform(SignData)
        var expected = new double[,]
        {
            { 2.048014301878, 0.815548432468 }, { 0.130300936803, 1.767636018883 },
            { -2.797080056996, 0.494868305362 }, { -1.277477320983, -1.163185790092 },
            { 0.832704985946, -1.012439841919 }, { -0.836117596336, 0.816055753469 },
            { 1.449112180332, 0.297721145863 }, { 0.450542569355, -2.016204024034 },
        };

        var pca = new PCA<double>(nComponents: 2);
        var result = pca.FitTransform(new Matrix<double>(SignData));
        AssertMatrixEqual(expected, result, 1e-9);
        AssertMatrixEqual(SignComponents, pca.Components, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task IncrementalPCA_ComponentSigns_FollowSklearnSvdFlip()
    {
        await Task.Yield();
        var ipca = new IncrementalPCA<double>(nComponents: 2, batchSize: 8);
        ipca.Fit(new Matrix<double>(SignData));
        AssertMatrixEqual(SignComponents, ipca.Components, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task RandomizedPCA_ComponentSigns_FollowSklearnSvdFlip()
    {
        await Task.Yield();
        var rpca = new RandomizedPCA<double>(nComponents: 2, randomState: 0);
        rpca.Fit(new Matrix<double>(SignData));
        AssertMatrixEqual(SignComponents, rpca.Components, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task TruncatedSVD_ComponentsAndExplainedVariance_MatchSklearn()
    {
        await Task.Yield();

        // sklearn.decomposition.TruncatedSVD(2, algorithm='arpack').fit(SignData): explained_variance_ is the variance
        // of each projected column and the ratio is over np.var(X, axis=0).sum(), not s^2 / ||X||^2
        var expectedComponents = new double[,]
        {
            { 0.691155087877, 0.489524576040, 0.531667503196 },
            { -0.687598570111, 0.218907507222, 0.692306080909 },
        };

        var svd = new TruncatedSVD<double>(nComponents: 2);
        svd.Fit(new Matrix<double>(SignData));
        AssertMatrixEqual(expectedComponents, svd.Components, 1e-9);
        AssertVectorEqual(new[] { 9.169451345107, 4.147514608579 }, svd.SingularValues, 1e-9);
        AssertVectorEqual(new[] { 0.492978753976, 2.146221723676 }, svd.ExplainedVariance, 1e-9);
        AssertVectorEqual(new[] { 0.122189846460, 0.531963093278 }, svd.ExplainedVarianceRatio, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task KernelPCA_Signs_FollowSklearnSvdFlip()
    {
        await Task.Yield();

        // sklearn.decomposition.KernelPCA(2, kernel='rbf', gamma=0.5, eigen_solver='dense')
        //     .fit(SignData[:6]).transform(SignData[6:]); eigenvectors_ go through svd_flip (u-based). The centred
        // kernel's eigenvectors sum to zero, so only the largest-magnitude rule fixes their sign.
        var expected = new double[,]
        {
            { -0.170496307912, -0.251310071341 },
            { -0.258933656833, -0.114992136705 },
        };

        var kpca = new KernelPCA<double>(
            nComponents: 2, kernel: AiDotNet.Preprocessing.DimensionalityReduction.KernelType.RBF, gamma: 0.5);
        kpca.Fit(new Matrix<double>(SignRows(0, 6)));
        AssertMatrixEqual(expected, kpca.Transform(new Matrix<double>(SignRows(6, 2))), 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task Isomap_Signs_FollowSklearnKernelPcaFlip()
    {
        await Task.Yield();

        // sklearn.manifold.Isomap(n_neighbors=4, n_components=2, path_method='D').fit_transform(SignData)
        var expected = new double[,]
        {
            { -2.058730468239, -0.906665779542 }, { -0.180794457901, -1.702932846116 },
            { 2.813747912854, -1.129914063237 }, { 1.976787151642, 1.687443048832 },
            { -0.680055206323, 0.982411069744 }, { 0.799143305935, -0.618055922955 },
            { -1.711225608510, -0.597363971672 }, { -0.958872629459, 2.285078464946 },
        };

        var result = new Isomap<double>(nComponents: 2, nNeighbors: 4, pathAlgorithm: IsomapPathAlgorithm.Dijkstra)
            .FitTransform(new Matrix<double>(SignData));
        AssertMatrixEqual(expected, result, 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task SpectralEmbedding_Rbf_MatchesSklearn()
    {
        await Task.Yield();

        // sklearn.manifold.SpectralEmbedding(2, affinity='rbf', gamma=0.5).fit_transform(SignData): the Laplacian
        // ignores self loops, eigenvectors are divided by sqrt(degree) and sign-flipped (largest entry positive)
        var expected = new double[,]
        {
            { -0.056346198972, 0.755095630050 }, { 0.534542937956, 0.018107718432 },
            { 0.847765252319, -0.744304288895 }, { 0.136382007942, -0.678891203990 },
            { -0.504718869659, -0.256018235630 }, { 0.634421647935, -0.321580118087 },
            { -0.087529712449, 0.611966907158 }, { -0.663063124567, -0.528530563781 },
        };

        var result = new SpectralEmbedding<double>(nComponents: 2, affinity: SpectralAffinity.RBF, gamma: 0.5)
            .FitTransform(new Matrix<double>(SignData));
        AssertMatrixEqual(expected, result, 1e-7);
    }

    [Fact(Timeout = 120000)]
    public async Task SpectralEmbedding_NearestNeighbors_MatchesSklearn()
    {
        await Task.Yield();

        // sklearn.manifold.SpectralEmbedding(2, affinity='nearest_neighbors', n_neighbors=4).fit_transform(SignData):
        // 0/1 kneighbors_graph(include_self=True) symmetrized as (A + A^T) / 2
        var expected = new double[,]
        {
            { -0.217338616381, -0.172145719346 }, { 0.030045657412, -0.257209558926 },
            { 0.386114008833, -0.076719677106 }, { 0.157654014760, 0.308900027044 },
            { -0.140753688257, 0.120330401967 }, { 0.286991963188, -0.092102682922 },
            { -0.210848405159, -0.086126269290 }, { -0.071799489939, 0.357715313718 },
        };

        var result = new SpectralEmbedding<double>(
                nComponents: 2, affinity: SpectralAffinity.NearestNeighbors, nNeighbors: 4)
            .FitTransform(new Matrix<double>(SignData));
        AssertMatrixEqual(expected, result, 1e-7);
    }

    [Fact(Timeout = 120000)]
    public async Task NearestNeighbors_EqualDistances_KeepIndexOrder()
    {
        await Task.Yield();

        // Points 0, 1, -1, 2, -2 on a line: from point 0 the distances to 1 and 2 tie (1) as do those to 3 and 4 (2),
        // so the neighbours must come in index order, like LINQ OrderBy (the engine sort is not documented as stable)
        var x = new Tensor<double>(new[] { 0.0, 1.0, -1.0, 2.0, -2.0 }, new[] { 5, 1 });
        var neighbors = DimensionalityReductionOps.NearestNeighbors(DimensionalityReductionOps.EuclideanDistances(x, x), 4);

        Assert.Equal(new[] { 1, 2, 3, 4 }, new[] { neighbors[0, 0], neighbors[0, 1], neighbors[0, 2], neighbors[0, 3] });
    }

    [Fact(Timeout = 120000)]
    public async Task SymmetricEigen_RepeatedEigenvalues_KeepIndexOrderAndSvdFlipSigns()
    {
        await Task.Yield();

        // diag(2, 5, 5, 1): the tied eigenvalue 5 keeps its eigenvectors e1, e2 in index order and every vector's
        // largest entry is positive
        var a = new Tensor<double>(new[]
        {
            2.0, 0.0, 0.0, 0.0,
            0.0, 5.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        }, new[] { 4, 4 });
        var (values, vectors) = DimensionalityReductionOps.SymmetricEigen(a, descending: true);

        Assert.Equal(new[] { 5.0, 5.0, 2.0, 1.0 }, values.ToArray());
        int[] expectedAxis = { 1, 2, 0, 3 };
        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 4; row++)
            {
                Assert.Equal(row == expectedAxis[col] ? 1.0 : 0.0, vectors[row, col], 1e-12);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Tolerance_Half_StaysPositive()
    {
        await Task.Yield();

        // Half rounds 1e-10 to 0, which would switch every "above epsilon" guard off
        var half = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(0.0, half.ToDouble(half.FromDouble(1e-10)));
        Assert.Equal(6.103515625e-05, half.ToDouble(DimensionalityReductionOps.Tolerance<Half>(1e-10)));
        Assert.Equal(1e-10, DimensionalityReductionOps.Tolerance<double>(1e-10));
        Assert.Equal(1e-3, half.ToDouble(DimensionalityReductionOps.Tolerance<Half>(1e-3)), 1e-6);
    }

    [Fact]
    public async Task PCA_VarianceRatio_ExactlyReached_KeepsOneMoreLikeSklearn()
    {
        await Task.Yield();

        // Equal variance on both axes gives explained_variance_ratio_ = [0.5, 0.5] exactly. scikit-learn
        // (PCA(n_components=0.5)) picks searchsorted([0.5, 1.0], 0.5, side="right") + 1 = 2 components.
        var data = new Matrix<double>(new double[,] { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } });
        var pca = new PCA<double>(varianceRatio: 0.5);
        pca.Fit(data);

        Assert.Equal(new[] { 0.5, 0.5 }, pca.ExplainedVarianceRatio?.ToArray());
        Assert.Equal(2, pca.NComponentsOut);
    }

    [Fact]
    public async Task StableOrder_Half_MoreThan2048DistinctValues_SortsThemAll()
    {
        await Task.Yield();

        // 3000 distinct Half integers (1500 down to -1499) in descending index order: ascending order is the
        // reversed index list. Dense ranks counted in Half stop at 2048, which would sort every value past the
        // 2048th distinct one by index instead of by value.
        var half = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        const int n = 3000;
        var values = new Half[n];
        for (int j = 0; j < n; j++)
        {
            values[j] = half.FromDouble(1500 - j);
        }

        var order = AiDotNet.AnomalyDetection.AnomalyTensorOps.StableOrder(
            new AiDotNet.Tensors.LinearAlgebra.Tensor<Half>(values, new[] { n }), descending: false).ToArray();

        for (int p = 0; p < n; p++)
        {
            Assert.Equal(n - 1 - p, order[p]);
        }
    }

    [Fact]
    public async Task PaCMAP_Half_NearestPairsFollowDistanceBeyondSquareOverflow()
    {
        await Task.Yield();

        // Half points 0, 1000 and 300 on a line: the nearest neighbour of point 0 is point 2 (300 away), not point 1
        // (1000 away). Their squared distances (90000 and 1e6) both overflow Half to infinity, which would tie them
        // and pick point 1 by index.
        var half = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        var x = new AiDotNet.Tensors.LinearAlgebra.Tensor<Half>(
            new[] { half.Zero, half.FromDouble(1000), half.FromDouble(300) }, new[] { 3, 1 });
        var pacmap = new PaCMAP<Half>(nComponents: 1, nNeighbors: 1, randomState: 0);
        var method = typeof(PaCMAP<Half>).GetMethod(
            "GeneratePairs", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(method);

        var result = method?.Invoke(pacmap, new object[] { x, 3, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1) });
        Assert.NotNull(result);
        var nearby = ((System.ValueTuple<
            (AiDotNet.Tensors.LinearAlgebra.Tensor<int> Heads, AiDotNet.Tensors.LinearAlgebra.Tensor<int> Tails),
            (AiDotNet.Tensors.LinearAlgebra.Tensor<int> Heads, AiDotNet.Tensors.LinearAlgebra.Tensor<int> Tails),
            (AiDotNet.Tensors.LinearAlgebra.Tensor<int> Heads, AiDotNet.Tensors.LinearAlgebra.Tensor<int> Tails)>)(result ?? new object())).Item1;

        Assert.Equal(new[] { 0, 1, 2 }, nearby.Heads.ToArray());
        Assert.Equal(new[] { 2, 2, 0 }, nearby.Tails.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task HessianLLE_CurvedSurface_MatchesSklearn()
    {
        await Task.Yield();

        // sklearn 1.9.1 LocallyLinearEmbedding(n_neighbors=7, n_components=2, method='hessian',
        // eigen_solver='dense').fit_transform(X), compared column by column up to sign. scikit-learn divides every
        // local Hessian column by its sum (hessian_tol 1e-4); without that the embedding differs in the second
        // decimal (row 0 would be (0.0653, 0.4912)).
        var x = new double[,]
        {
            { 0.0, 0.0, 0.10 }, { 1.0, 0.2, 0.90 }, { 2.1, 0.1, 4.30 }, { 0.1, 1.0, -0.8 },
            { 1.2, 1.1, 0.40 }, { 2.0, 0.9, 3.20 }, { 0.2, 2.1, -2.0 }, { 1.1, 2.0, -0.7 },
            { 2.2, 2.2, 2.60 }, { 3.0, 0.4, 8.70 }, { 3.1, 1.6, 7.90 }, { 2.9, 2.8, 5.60 },
            { 0.5, 3.1, -2.9 }, { 1.6, 3.0, -0.4 },
        };
        var expected = new double[,]
        {
            { 0.05342888776842297, -0.4917728309031661 },
            { 0.004587378675076775, -0.33611827793832494 },
            { -0.24219838103521124, -0.23296552633791687 },
            { 0.17038456916384617, -0.265655136735507 },
            { 0.08782570931381038, -0.12068101488690054 },
            { -0.12147384592582469, -0.07038600431288754 },
            { 0.3133604230009105, -0.016395559876013543 },
            { 0.2138043263668885, 0.06240510968871538 },
            { -0.009955093415199418, 0.23511055505965495 },
            { -0.5373880675759429, -0.035147497684842294 },
            { -0.41685148634514774, 0.2379224344146773 },
            { -0.19012959356745046, 0.46804929758947106 },
            { 0.42871244508534556, 0.23540218219583142 },
            { 0.2458927285204894, 0.3302322697449402 },
        };

        var embedding = new HessianLLE<double>(nComponents: 2, nNeighbors: 7).FitTransform(new Matrix<double>(x));

        for (int c = 0; c < 2; c++)
        {
            double sign = Math.Sign(embedding[0, c]) == Math.Sign(expected[0, c]) ? 1.0 : -1.0;
            for (int i = 0; i < expected.GetLength(0); i++)
            {
                Assert.Equal(expected[i, c], sign * embedding[i, c], 6);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DiffusionMaps_Embedding_IsRightEigenvectorsOfTheMarkovMatrix()
    {
        await Task.Yield();

        // numpy reference (Coifman and Lafon 2006): K = exp(-d^2 / median(d)^2), alpha = 0.5 normalisation,
        // P = D^-1 K_alpha, right eigenvectors psi of P (np.linalg.eig) normalised so that psi_0 = 1 (sum of
        // pi psi^2 = 1 with pi = d / sum d), embedding lambda_k psi_k for k = 1, 2; compared up to column sign. The
        // symmetric eigenvectors D^(1/2) psi_k are not diffusion coordinates.
        var expected = new double[,]
        {
            { 0.6436337210550851, -0.007101490621882962 },
            { -1.1269572482442622, 0.45210199710973675 },
            { 0.7619197836092492, 0.11560995973009273 },
            { 0.25558232554860183, -0.2927252370685659 },
            { 0.9979601611632956, 0.5581308283488806 },
            { 0.6186344005106518, -0.09175726168309226 },
            { -0.15424225405626862, -0.4528671754185853 },
            { -0.8096286829285488, 0.08277908914902861 },
            { -0.3417395536719599, -0.28807291461045176 },
            { -0.888072540155848, 0.09184996069358281 },
        };

        var embedding = new DiffusionMaps<double>(nComponents: 2).FitTransform(new Matrix<double>(Data));

        for (int c = 0; c < 2; c++)
        {
            double sign = Math.Sign(embedding[0, c]) == Math.Sign(expected[0, c]) ? 1.0 : -1.0;
            for (int i = 0; i < expected.GetLength(0); i++)
            {
                Assert.Equal(expected[i, c], sign * embedding[i, c], 9);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DiffusionMaps_Half_LargeDistances_KernelStaysFinite()
    {
        await Task.Yield();

        // Points 0, 300, 700 and 1000 on a line: every squared distance and the squared median overflow Half, so
        // exp(-d^2 / eps) would be exp(-inf / inf) = NaN everywhere.
        var half = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        var data = new Matrix<Half>(4, 1);
        data[0, 0] = half.Zero;
        data[1, 0] = half.FromDouble(300);
        data[2, 0] = half.FromDouble(700);
        data[3, 0] = half.FromDouble(1000);

        var embedding = new DiffusionMaps<Half>(nComponents: 1).FitTransform(data);

        for (int i = 0; i < 4; i++)
        {
            double value = half.ToDouble(embedding[i, 0]);
            Assert.False(double.IsNaN(value) || double.IsInfinity(value), $"row {i} is {value}");
        }

        // The 1-D chain is ordered by its first diffusion coordinate
        Assert.True(half.ToDouble(embedding[0, 0]) * half.ToDouble(embedding[3, 0]) < 0);
    }

    [Fact(Timeout = 120000)]
    public async Task NCA_SameLabelMask_Half_KeepsLabelsHalfCannotTellApart()
    {
        await Task.Yield();

        // Labels 2048 and 2049 are different classes; converted to Half both are 2048 and would count as the same
        // class.
        var half = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        var same = NCA<Half>.SameLabelMask(new[] { 2048, 2049, 2048 });

        Assert.Equal(1.0, half.ToDouble(same[0, 0]));
        Assert.Equal(0.0, half.ToDouble(same[0, 1]));
        Assert.Equal(1.0, half.ToDouble(same[0, 2]));
        Assert.Equal(0.0, half.ToDouble(same[1, 2]));
    }

    private static double[,] SignRows(int start, int count)
    {
        var rows = new double[count, SignData.GetLength(1)];
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < SignData.GetLength(1); j++)
            {
                rows[i, j] = SignData[start + i, j];
            }
        }

        return rows;
    }

    private static void AssertMatrixEqual(double[,] expected, Matrix<double>? actual, double tolerance)
    {
        Assert.NotNull(actual);
        if (actual is null)
        {
            return;
        }

        Assert.Equal(expected.GetLength(0), actual.Rows);
        Assert.Equal(expected.GetLength(1), actual.Columns);
        for (int row = 0; row < actual.Rows; row++)
        {
            for (int col = 0; col < actual.Columns; col++)
            {
                Assert.Equal(expected[row, col], actual[row, col], tolerance);
            }
        }
    }

    private static void AssertVectorEqual(double[] expected, Vector<double>? actual, double tolerance)
    {
        Assert.NotNull(actual);
        if (actual is null)
        {
            return;
        }

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], tolerance);
        }
    }

    private static double[,] Standardize(Matrix<double> m)
    {
        var result = new double[m.Rows, m.Columns];
        for (int c = 0; c < m.Columns; c++)
        {
            double mean = 0;
            for (int i = 0; i < m.Rows; i++) mean += m[i, c] / m.Rows;
            double variance = 0;
            for (int i = 0; i < m.Rows; i++) variance += (m[i, c] - mean) * (m[i, c] - mean) / m.Rows;
            double std = Math.Sqrt(variance);
            for (int i = 0; i < m.Rows; i++) result[i, c] = (m[i, c] - mean) / std;
        }

        return result;
    }

    private static double Correlation(double[,] a, int ca, double[,] b, int cb)
    {
        double sum = 0;
        for (int i = 0; i < a.GetLength(0); i++) sum += a[i, ca] * b[i, cb];
        return sum / a.GetLength(0);
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
