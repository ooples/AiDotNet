// Copyright (c) AiDotNet. All rights reserved.
// Feature-based Bayesian tuner with ARD kernel for GEMM configuration selection.

using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// Feature-based Bayesian tuner for GEMM kernel configuration selection.
/// Uses Gaussian Process regression with ARD (Automatic Relevance Determination) kernel
/// that learns the importance of each configuration parameter.
/// </summary>
internal sealed class GemmFeatureBayesianTuner
{
    private readonly Random _random;
    private readonly List<double[]> _observedFeatures;
    private readonly List<double> _observedValues;
    private readonly GemmConfig[] _configSpace;
    private double[,]? _covarianceMatrix;
    private double[,]? _covarianceMatrixInverse;

    // Feature indices for easy reference
    private const int FEAT_TILE_M = 0;
    private const int FEAT_TILE_N = 1;
    private const int FEAT_TILE_K = 2;
    private const int FEAT_THREAD_TILE_M = 3;
    private const int FEAT_THREAD_TILE_N = 4;
    private const int FEAT_VECTOR_WIDTH_M = 5;
    private const int FEAT_VECTOR_WIDTH_N = 6;
    private const int FEAT_DOUBLE_BUFFERING = 7;
    private const int FEAT_VECTORIZED_LOADS = 8;
    private const int FEAT_KREG = 9;
    private const int FEAT_KUNROLL = 10;
    private const int FEAT_SUBGROUP_OPS = 11;
    private const int FEAT_STRIDE_M = 12;
    private const int FEAT_STRIDE_N = 13;
    private const int NUM_FEATURES = 14;

    // ARD length scales per feature (learned)
    private readonly double[] _lengthScales;

    // Signal variance
    private double _signalVariance = 1.0;
    private const double NoiseVariance = 0.01;

    // Feature normalization parameters
    private readonly double[] _featureMeans;
    private readonly double[] _featureStds;

    public double BestObservedValue => _observedValues.Count > 0 ? _observedValues.Max() : 0;

    public GemmFeatureBayesianTuner(GemmConfig[] configSpace, int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _configSpace = configSpace;
        _observedFeatures = new List<double[]>();
        _observedValues = new List<double>();

        // Initialize ARD length scales (will be optimized)
        _lengthScales = new double[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            _lengthScales[i] = 1.0;
        }

        // Compute feature normalization parameters from config space
        _featureMeans = new double[NUM_FEATURES];
        _featureStds = new double[NUM_FEATURES];
        ComputeFeatureNormalization();
    }

    private void ComputeFeatureNormalization()
    {
        // Extract all feature vectors
        var allFeatures = new double[_configSpace.Length][];
        for (int i = 0; i < _configSpace.Length; i++)
        {
            allFeatures[i] = ExtractFeatures(_configSpace[i]);
        }

        // Compute mean and std for each feature
        for (int f = 0; f < NUM_FEATURES; f++)
        {
            double sum = 0;
            for (int i = 0; i < allFeatures.Length; i++)
            {
                sum += allFeatures[i][f];
            }
            _featureMeans[f] = sum / allFeatures.Length;

            double varSum = 0;
            for (int i = 0; i < allFeatures.Length; i++)
            {
                double diff = allFeatures[i][f] - _featureMeans[f];
                varSum += diff * diff;
            }
            _featureStds[f] = Math.Max(0.01, Math.Sqrt(varSum / allFeatures.Length));
        }
    }

    private double[] ExtractFeatures(GemmConfig config)
    {
        return new double[]
        {
            config.TileM,
            config.TileN,
            config.TileK,
            config.ThreadTileM,
            config.ThreadTileN,
            config.VectorWidthM,
            config.VectorWidthN,
            config.UseDoubleBuffering ? 1.0 : 0.0,
            config.UseVectorizedLoads ? 1.0 : 0.0,
            config.KReg,       // Register tiling in K dimension (1, 2, 4)
            config.KUnroll,    // K loop unroll factor (1, 2, 4, 8)
            config.UseSubgroupOps ? 1.0 : 0.0,  // Subgroup/wave operations
            config.StrideM ? 1.0 : 0.0,  // STRM: Strided A tile stores for bank conflict avoidance
            config.StrideN ? 1.0 : 0.0   // STRN: Strided B tile stores for bank conflict avoidance
        };
    }

    private double[] NormalizeFeatures(double[] features)
    {
        var normalized = new double[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            normalized[i] = (features[i] - _featureMeans[i]) / _featureStds[i];
        }
        return normalized;
    }

    public int SampleRandomIndex(int totalConfigs, HashSet<int> excluded)
    {
        int idx;
        int attempts = 0;
        do
        {
            idx = _random.Next(totalConfigs);
            attempts++;
        } while (excluded.Contains(idx) && attempts < totalConfigs * 2);

        if (excluded.Contains(idx))
        {
            for (int i = 0; i < totalConfigs; i++)
            {
                if (!excluded.Contains(i))
                    return i;
            }
        }

        return idx;
    }

    public void AddObservation(int configIndex, double gflops)
    {
        var features = ExtractFeatures(_configSpace[configIndex]);
        var normalized = NormalizeFeatures(features);
        _observedFeatures.Add(normalized);
        _observedValues.Add(gflops);
    }

    public void UpdateModel()
    {
        if (_observedFeatures.Count < 2)
            return;

        // Update covariance matrix using ARD kernel
        int n = _observedFeatures.Count;
        _covarianceMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _covarianceMatrix[i, j] = ArdKernel(_observedFeatures[i], _observedFeatures[j]);
                if (i == j)
                    _covarianceMatrix[i, j] += NoiseVariance;
            }
        }

        _covarianceMatrixInverse = InvertMatrixCholesky(_covarianceMatrix);

        // Optimize ARD length scales periodically
        if (n % 5 == 0 && n >= 5)
        {
            OptimizeArdLengthScales();
        }
    }

    public int SelectNextPoint(int totalConfigs, HashSet<int> excluded)
    {
        if (_observedFeatures.Count < 2 || _covarianceMatrixInverse == null)
        {
            return SampleRandomIndex(totalConfigs, excluded);
        }

        double bestAcquisition = double.NegativeInfinity;
        int bestIndex = -1;

        for (int idx = 0; idx < totalConfigs; idx++)
        {
            if (excluded.Contains(idx))
                continue;

            double acquisition = ComputeExpectedImprovement(idx);
            if (acquisition > bestAcquisition)
            {
                bestAcquisition = acquisition;
                bestIndex = idx;
            }
        }

        return bestIndex >= 0 ? bestIndex : SampleRandomIndex(totalConfigs, excluded);
    }

    private double ComputeExpectedImprovement(int candidateIdx)
    {
        var (mean, variance) = PredictGP(candidateIdx);
        double std = Math.Sqrt(Math.Max(0, variance) + 1e-9);

        if (std < 1e-9)
            return 0;

        double bestValue = BestObservedValue;
        double z = (mean - bestValue) / std;

        double phi = NormalPdf(z);
        double Phi = NormalCdf(z);

        return std * (z * Phi + phi);
    }

    private (double mean, double variance) PredictGP(int candidateIdx)
    {
        if (_observedFeatures.Count == 0 || _covarianceMatrixInverse == null)
        {
            return (0.0, _signalVariance);
        }

        var candidateFeatures = ExtractFeatures(_configSpace[candidateIdx]);
        var candidateNormalized = NormalizeFeatures(candidateFeatures);

        int n = _observedFeatures.Count;
        var kStar = new double[n];

        for (int i = 0; i < n; i++)
        {
            kStar[i] = ArdKernel(candidateNormalized, _observedFeatures[i]);
        }

        // Mean prediction: k* @ K^-1 @ y
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * _observedValues[j];
            }
            mean += kStar[i] * sum;
        }

        // Variance prediction: k** - k* @ K^-1 @ k*^T
        double kStarStar = ArdKernel(candidateNormalized, candidateNormalized);
        double variance = kStarStar;

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * kStar[j];
            }
            variance -= kStar[i] * sum;
        }

        return (mean, Math.Max(0, variance));
    }

    /// <summary>
    /// ARD (Automatic Relevance Determination) kernel.
    /// Each feature has its own length scale, allowing the model to learn feature importance.
    /// k(x, x') = σ² * exp(-0.5 * Σ (xi - xi')² / li²)
    /// </summary>
    private double ArdKernel(double[] x1, double[] x2)
    {
        double sqDist = 0;
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            double diff = x1[i] - x2[i];
            sqDist += diff * diff / (_lengthScales[i] * _lengthScales[i]);
        }
        return _signalVariance * Math.Exp(-0.5 * sqDist);
    }

    private void OptimizeArdLengthScales()
    {
        // Grid search over length scales for each feature
        double bestLl = double.NegativeInfinity;
        var bestScales = (double[])_lengthScales.Clone();

        // Candidate length scales
        double[] candidates = { 0.5, 1.0, 2.0, 5.0 };

        // For efficiency, optimize a subset of features per update
        // Focus on the most impactful features including register tiling and unroll factors
        int[] featuresToOptimize = { FEAT_TILE_M, FEAT_TILE_N, FEAT_TILE_K, FEAT_THREAD_TILE_M, FEAT_THREAD_TILE_N,
                                     FEAT_VECTOR_WIDTH_M, FEAT_VECTOR_WIDTH_N, FEAT_KREG, FEAT_KUNROLL };

        foreach (int feat in featuresToOptimize)
        {
            double originalScale = _lengthScales[feat];

            foreach (double scale in candidates)
            {
                _lengthScales[feat] = scale;
                UpdateCovarianceMatrix();
                double ll = ComputeLogMarginalLikelihood();

                if (ll > bestLl)
                {
                    bestLl = ll;
                    bestScales[feat] = scale;
                }
            }

            _lengthScales[feat] = bestScales[feat];
        }

        // Apply best scales and update covariance
        Array.Copy(bestScales, _lengthScales, NUM_FEATURES);
        UpdateCovarianceMatrix();
    }

    private void UpdateCovarianceMatrix()
    {
        if (_observedFeatures.Count < 2)
            return;

        int n = _observedFeatures.Count;
        _covarianceMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _covarianceMatrix[i, j] = ArdKernel(_observedFeatures[i], _observedFeatures[j]);
                if (i == j)
                    _covarianceMatrix[i, j] += NoiseVariance;
            }
        }

        _covarianceMatrixInverse = InvertMatrixCholesky(_covarianceMatrix);
    }

    private double ComputeLogMarginalLikelihood()
    {
        if (_covarianceMatrix == null || _covarianceMatrixInverse == null)
            return double.NegativeInfinity;

        int n = _observedFeatures.Count;
        double[] y = _observedValues.ToArray();

        double dataFit = 0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * y[j];
            }
            dataFit += y[i] * sum;
        }

        double logDet = LogDeterminant(_covarianceMatrix);

        return -0.5 * (dataFit + logDet + n * Math.Log(2 * Math.PI));
    }

    /// <summary>
    /// Gets the learned feature relevances (inverse of length scales squared).
    /// Higher values indicate more important features.
    /// </summary>
    public Dictionary<string, double> GetFeatureRelevances()
    {
        var relevances = new Dictionary<string, double>();
        string[] featureNames = { "TileM", "TileN", "TileK", "ThreadTileM", "ThreadTileN",
                                  "VectorWidthM", "VectorWidthN", "DoubleBuffering", "VectorizedLoads",
                                  "KReg", "KUnroll", "SubgroupOps", "StrideM", "StrideN" };

        for (int i = 0; i < NUM_FEATURES; i++)
        {
            // Relevance = 1 / lengthScale² (smaller length scale = more relevant)
            relevances[featureNames[i]] = 1.0 / (_lengthScales[i] * _lengthScales[i]);
        }

        return relevances;
    }

    #region Math Helpers

    private static double NormalPdf(double x)
    {
        return Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
    }

    private static double NormalCdf(double x)
    {
        return 0.5 * (1 + Erf(x / Math.Sqrt(2)));
    }

    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }

    private static double[,] InvertMatrixCholesky(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var L = new double[n, n];
        var inverse = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double diag = matrix[i, i] - sum;
                    L[i, j] = Math.Sqrt(Math.Max(1e-10, diag));
                }
                else
                {
                    L[i, j] = (matrix[i, j] - sum) / L[j, j];
                }
            }
        }

        var Linv = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            Linv[i, i] = 1.0 / L[i, i];
            for (int j = 0; j < i; j++)
            {
                double sum = 0;
                for (int k = j; k < i; k++)
                {
                    sum += L[i, k] * Linv[k, j];
                }
                Linv[i, j] = -sum / L[i, i];
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = i; k < n; k++)
                {
                    sum += Linv[k, i] * Linv[k, j];
                }
                inverse[i, j] = sum;
                inverse[j, i] = sum;
            }
        }

        return inverse;
    }

    private static double LogDeterminant(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var L = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double diag = matrix[i, i] - sum;
                    L[i, j] = Math.Sqrt(Math.Max(1e-10, diag));
                }
                else
                {
                    L[i, j] = (matrix[i, j] - sum) / L[j, j];
                }
            }
        }

        double logDet = 0;
        for (int i = 0; i < n; i++)
        {
            logDet += Math.Log(Math.Max(1e-10, L[i, i]));
        }

        return 2 * logDet;
    }

    #endregion
}
