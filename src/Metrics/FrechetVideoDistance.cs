using System;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Metrics;

/// <summary>
/// Fréchet Video Distance (FVD) - A metric for evaluating the quality of generated videos.
/// </summary>
/// <remarks>
/// <para>
/// FVD extends Fréchet Inception Distance (FID) to videos by using a 3D video feature extractor
/// (typically Inflated 3D ConvNet, I3D) to compare the distribution of generated videos
/// against real videos.
/// </para>
/// <para>
/// The algorithm:
/// 1. Extract spatiotemporal features from video clips using a 3D CNN
/// 2. Compute statistics (mean and covariance) for real and generated video features
/// 3. Compute the Fréchet distance between the two Gaussian distributions
/// </para>
/// <para>
/// Formula: FVD = ||mu_1 - mu_2||^2 + Tr(Sigma_1 + Sigma_2 - 2 * sqrt(Sigma_1 * Sigma_2))
/// </para>
/// <para>
/// Typical FVD scores:
/// - FVD &lt; 50: Excellent quality (hard to distinguish from real)
/// - FVD 50-100: Good quality
/// - FVD 100-300: Moderate quality
/// - FVD &gt; 300: Poor quality
/// </para>
/// <para>
/// Based on "Towards Accurate Generative Models of Video: A New Metric and Challenges"
/// by Unterthiner et al. (2018)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations</typeparam>
public class FrechetVideoDistance<T>
{
    private readonly INumericOperations<T> _numOps;
    private IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the 3D feature extraction network (e.g., I3D) used for video representations.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> FeatureNetwork { get; }

    /// <summary>
    /// Gets the dimensionality of extracted video features.
    /// </summary>
    public int FeatureDimension { get; }

    /// <summary>
    /// Gets or sets the number of frames per video clip for feature extraction.
    /// Default is 16 frames, which is standard for I3D models.
    /// </summary>
    public int FramesPerClip { get; set; }

    /// <summary>
    /// Gets or sets the frame sampling strategy.
    /// </summary>
    public FrameSamplingStrategy SamplingStrategy { get; set; }

    /// <summary>
    /// Initializes a new instance of FVD calculator.
    /// </summary>
    /// <param name="featureNetwork">Pre-trained 3D CNN (I3D or similar) for video feature extraction</param>
    /// <param name="featureDimension">Dimension of extracted features (typically 400 or 1024 for I3D)</param>
    /// <param name="framesPerClip">Number of frames per clip (default 16)</param>
    public FrechetVideoDistance(
        ConvolutionalNeuralNetwork<T> featureNetwork,
        int featureDimension = 400,
        int framesPerClip = 16)
    {
        if (featureNetwork == null)
        {
            throw new ArgumentNullException(nameof(featureNetwork),
                "A pre-trained 3D feature extraction network is required for FVD computation");
        }
        if (featureDimension < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(featureDimension),
                "Feature dimension must be at least 1");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        FeatureNetwork = featureNetwork;
        FeatureDimension = featureDimension;
        FramesPerClip = framesPerClip;
        SamplingStrategy = FrameSamplingStrategy.Uniform;
    }

    /// <summary>
    /// Computes the FVD score between real and generated videos.
    /// </summary>
    /// <param name="realVideos">Tensor of real videos [N, T, C, H, W] or [N, C, T, H, W]</param>
    /// <param name="generatedVideos">Tensor of generated videos with same format</param>
    /// <returns>FVD score (lower is better)</returns>
    public double ComputeFVD(Tensor<T> realVideos, Tensor<T> generatedVideos)
    {
        if (realVideos.Shape.Length < 5 || generatedVideos.Shape.Length < 5)
        {
            throw new ArgumentException("Videos must be 5D tensors [N, T, C, H, W] or [N, C, T, H, W]");
        }

        var realFeatures = ExtractVideoFeatures(realVideos);
        var generatedFeatures = ExtractVideoFeatures(generatedVideos);

        var (realMean, realCov) = ComputeStatistics(realFeatures);
        var (genMean, genCov) = ComputeStatistics(generatedFeatures);

        return ComputeFrechetDistance(realMean, realCov, genMean, genCov);
    }

    /// <summary>
    /// Computes FVD using pre-computed statistics for the real distribution.
    /// </summary>
    /// <param name="realMean">Mean of real video features</param>
    /// <param name="realCov">Covariance of real video features</param>
    /// <param name="generatedVideos">Generated videos to evaluate</param>
    /// <returns>FVD score</returns>
    public double ComputeFVDWithStats(
        Vector<T> realMean,
        Matrix<T> realCov,
        Tensor<T> generatedVideos)
    {
        var generatedFeatures = ExtractVideoFeatures(generatedVideos);
        var (genMean, genCov) = ComputeStatistics(generatedFeatures);
        return ComputeFrechetDistance(realMean, realCov, genMean, genCov);
    }

    /// <summary>
    /// Pre-computes statistics for a set of videos.
    /// </summary>
    /// <param name="videos">Videos to compute statistics for</param>
    /// <returns>Tuple of (mean, covariance)</returns>
    public (Vector<T> mean, Matrix<T> covariance) PrecomputeStatistics(Tensor<T> videos)
    {
        var features = ExtractVideoFeatures(videos);
        return ComputeStatistics(features);
    }

    /// <summary>
    /// Extracts features from videos using the 3D feature network.
    /// </summary>
    private Matrix<T> ExtractVideoFeatures(Tensor<T> videos)
    {
        int numVideos = videos.Shape[0];
        var features = new Matrix<T>(numVideos, FeatureDimension);

        bool originalTrainingMode = FeatureNetwork.IsTrainingMode;
        FeatureNetwork.SetTrainingMode(false);

        try
        {
            // Determine video format (NTCHW or NCTHW)
            bool isNTCHW = videos.Shape[1] > 3; // If second dim > 3, likely temporal

            for (int v = 0; v < numVideos; v++)
            {
                // Extract and sample frames from video
                var videoClip = ExtractVideoClip(videos, v, isNTCHW);

                // Run through 3D CNN
                var output = FeatureNetwork.Predict(videoClip);

                // Store features (global average pooled if needed)
                var videoFeature = GlobalAveragePool(output);
                for (int j = 0; j < Math.Min(videoFeature.Length, FeatureDimension); j++)
                {
                    features[v, j] = videoFeature[j];
                }
            }

            return features;
        }
        finally
        {
            FeatureNetwork.SetTrainingMode(originalTrainingMode);
        }
    }

    /// <summary>
    /// Extracts a single video clip from the batch tensor.
    /// </summary>
    private Tensor<T> ExtractVideoClip(Tensor<T> videos, int videoIdx, bool isNTCHW)
    {
        int totalFrames;
        int channels;
        int height;
        int width;

        if (isNTCHW)
        {
            // Format: [N, T, C, H, W]
            totalFrames = videos.Shape[1];
            channels = videos.Shape[2];
            height = videos.Shape[3];
            width = videos.Shape[4];
        }
        else
        {
            // Format: [N, C, T, H, W]
            channels = videos.Shape[1];
            totalFrames = videos.Shape[2];
            height = videos.Shape[3];
            width = videos.Shape[4];
        }

        // Sample frames according to strategy
        int[] frameIndices = SampleFrameIndices(totalFrames);

        // Create output tensor for the clip [1, C, T, H, W]
        var clipShape = new[] { 1, channels, FramesPerClip, height, width };
        var clip = new Tensor<T>(clipShape);

        for (int t = 0; t < FramesPerClip; t++)
        {
            int srcFrame = frameIndices[t];

            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        T value;
                        if (isNTCHW)
                        {
                            value = videos[videoIdx, srcFrame, c, h, w];
                        }
                        else
                        {
                            value = videos[videoIdx, c, srcFrame, h, w];
                        }
                        clip[0, c, t, h, w] = value;
                    }
                }
            }
        }

        return clip;
    }

    /// <summary>
    /// Samples frame indices according to the sampling strategy.
    /// </summary>
    private int[] SampleFrameIndices(int totalFrames)
    {
        var indices = new int[FramesPerClip];

        switch (SamplingStrategy)
        {
            case FrameSamplingStrategy.Uniform:
                // Uniform sampling across the video
                for (int i = 0; i < FramesPerClip; i++)
                {
                    indices[i] = Math.Min(
                        i * totalFrames / FramesPerClip,
                        totalFrames - 1);
                }
                break;

            case FrameSamplingStrategy.Random:
                // Random sampling
                var random = new Random(42);
                for (int i = 0; i < FramesPerClip; i++)
                {
                    indices[i] = random.Next(totalFrames);
                }
                Array.Sort(indices);
                break;

            case FrameSamplingStrategy.CenterCrop:
                // Sample from the center of the video
                int startFrame = Math.Max(0, (totalFrames - FramesPerClip) / 2);
                for (int i = 0; i < FramesPerClip; i++)
                {
                    indices[i] = Math.Min(startFrame + i, totalFrames - 1);
                }
                break;

            default:
                // Default to uniform
                for (int i = 0; i < FramesPerClip; i++)
                {
                    indices[i] = Math.Min(
                        i * totalFrames / FramesPerClip,
                        totalFrames - 1);
                }
                break;
        }

        return indices;
    }

    /// <summary>
    /// Applies global average pooling to reduce spatial and temporal dimensions.
    /// </summary>
    private T[] GlobalAveragePool(Tensor<T> tensor)
    {
        // Assuming tensor is [1, C, T, H, W] or [1, C, H, W] or [1, D]
        if (tensor.Rank <= 2)
        {
            // Already reduced, just return as array
            var result = new T[tensor.Shape[tensor.Rank - 1]];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = tensor.GetFlat(i);
            }
            return result;
        }

        int channels = tensor.Shape[1];
        var pooled = new T[channels];
        int spatiotemporalSize = tensor.Length / channels;

        for (int c = 0; c < channels; c++)
        {
            T sum = _numOps.Zero;
            int count = 0;

            // Sum over all spatial and temporal dimensions
            for (int idx = 0; idx < tensor.Length / channels; idx++)
            {
                int flatIdx = c * (tensor.Length / channels) + idx;
                if (flatIdx < tensor.Length)
                {
                    sum = _numOps.Add(sum, tensor.GetFlat(flatIdx));
                    count++;
                }
            }

            pooled[c] = count > 0 ? _numOps.Divide(sum, _numOps.FromDouble(count)) : _numOps.Zero;
        }

        return pooled;
    }

    /// <summary>
    /// Computes mean and covariance matrix of feature vectors.
    /// </summary>
    private (Vector<T> mean, Matrix<T> covariance) ComputeStatistics(Matrix<T> features)
    {
        int numSamples = features.Rows;
        int dim = features.Columns;

        if (numSamples < 2)
        {
            throw new ArgumentException(
                $"Need at least 2 samples to compute covariance, got {numSamples}.",
                nameof(features));
        }

        var mean = new Vector<T>(dim);
        T nInv = _numOps.FromDouble(1.0 / numSamples);

        // Compute mean
        for (int j = 0; j < dim; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < numSamples; i++)
            {
                sum = _numOps.Add(sum, features[i, j]);
            }
            mean[j] = _numOps.Multiply(sum, nInv);
        }

        // Compute centered features
        var centered = new Matrix<T>(numSamples, dim);
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                centered[i, j] = _numOps.Subtract(features[i, j], mean[j]);
            }
        }

        // Compute covariance: (1/(n-1)) * centered^T * centered
        T nMinusOneInv = _numOps.FromDouble(1.0 / (numSamples - 1));
        var centeredT = centered.Transpose();
        var covariance = (Matrix<T>)Engine.MatrixMultiply(centeredT, centered);

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                covariance[i, j] = _numOps.Multiply(covariance[i, j], nMinusOneInv);
            }
        }

        return (mean, covariance);
    }

    /// <summary>
    /// Computes the Fréchet distance between two Gaussian distributions.
    /// </summary>
    private double ComputeFrechetDistance(
        Vector<T> mean1,
        Matrix<T> cov1,
        Vector<T> mean2,
        Matrix<T> cov2)
    {
        var meanDiff = (Vector<T>)Engine.Subtract(mean1, mean2);
        var meanDiffSq = Engine.DotProduct(meanDiff, meanDiff);

        var trace1 = ComputeTrace(cov1);
        var trace2 = ComputeTrace(cov2);
        var traceCov = _numOps.Add(trace1, trace2);

        var traceSqrtCovProduct = ComputeTraceSqrtCovProduct(cov1, cov2);

        var fvd = _numOps.Add(meanDiffSq, traceCov);
        fvd = _numOps.Subtract(fvd, _numOps.Multiply(_numOps.FromDouble(2.0), traceSqrtCovProduct));

        return _numOps.ToDouble(fvd);
    }

    /// <summary>
    /// Computes the trace of a matrix.
    /// </summary>
    private T ComputeTrace(Matrix<T> matrix)
    {
        var trace = _numOps.Zero;
        int n = Math.Min(matrix.Rows, matrix.Columns);
        for (int i = 0; i < n; i++)
        {
            trace = _numOps.Add(trace, matrix[i, i]);
        }
        return trace;
    }

    /// <summary>
    /// Computes Tr(sqrt(cov1 * cov2)) using Newton-Schulz iteration.
    /// </summary>
    private T ComputeTraceSqrtCovProduct(Matrix<T> cov1, Matrix<T> cov2)
    {
        int n = cov1.Rows;

        var product = (Matrix<T>)Engine.MatrixMultiply(cov1, cov2);

        // Symmetrize the product
        var symProduct = new Matrix<T>(n, n);
        T half = _numOps.FromDouble(0.5);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                symProduct[i, j] = _numOps.Multiply(
                    _numOps.Add(product[i, j], product[j, i]),
                    half);
            }
        }

        // Add regularization for numerical stability
        T eps = _numOps.FromDouble(1e-6);
        for (int i = 0; i < n; i++)
        {
            symProduct[i, i] = _numOps.Add(symProduct[i, i], eps);
        }

        // Compute Frobenius norm for scaling
        T frobNormSq = _numOps.Zero;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                frobNormSq = _numOps.Add(frobNormSq,
                    _numOps.Multiply(symProduct[i, j], symProduct[i, j]));
            }
        }
        T frobNorm = _numOps.Sqrt(frobNormSq);

        if (_numOps.LessThan(frobNorm, _numOps.FromDouble(1e-10)))
        {
            return _numOps.Zero;
        }

        // Scale for better numerical stability
        T scale = _numOps.Sqrt(frobNorm);
        T scaleInv = _numOps.Divide(_numOps.One, scale);

        var A = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i, j] = _numOps.Multiply(symProduct[i, j], scaleInv);
            }
        }

        // Newton-Schulz iteration for matrix square root
        var Y = Matrix<T>.CreateIdentity(n);
        var identity = Matrix<T>.CreateIdentity(n);
        T three = _numOps.FromDouble(3.0);

        const int maxIterations = 15;
        const double convergenceTolerance = 1e-10;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var YY = (Matrix<T>)Engine.MatrixMultiply(Y, Y);
            var AYY = (Matrix<T>)Engine.MatrixMultiply(A, YY);

            var threeIMinusAYY = new Matrix<T>(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    threeIMinusAYY[i, j] = _numOps.Subtract(
                        _numOps.Multiply(three, identity[i, j]),
                        AYY[i, j]);
                }
            }

            var newY = (Matrix<T>)Engine.MatrixMultiply(Y, threeIMinusAYY);

            T updateNormSq = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    T newVal = _numOps.Multiply(half, newY[i, j]);
                    T diff = _numOps.Subtract(newVal, Y[i, j]);
                    updateNormSq = _numOps.Add(updateNormSq, _numOps.Multiply(diff, diff));
                    Y[i, j] = newVal;
                }
            }

            if (_numOps.ToDouble(_numOps.Sqrt(updateNormSq)) < convergenceTolerance)
            {
                break;
            }
        }

        var AY = (Matrix<T>)Engine.MatrixMultiply(A, Y);
        T sqrtScale = _numOps.Sqrt(scale);
        T traceAY = ComputeTrace(AY);

        return _numOps.Multiply(sqrtScale, traceAY);
    }
}

/// <summary>
/// Frame sampling strategies for video feature extraction.
/// </summary>
public enum FrameSamplingStrategy
{
    /// <summary>
    /// Uniformly sample frames across the entire video.
    /// </summary>
    Uniform,

    /// <summary>
    /// Randomly sample frames from the video.
    /// </summary>
    Random,

    /// <summary>
    /// Sample consecutive frames from the center of the video.
    /// </summary>
    CenterCrop
}
