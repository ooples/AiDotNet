using System;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;

using AiDotNet.Tensors.Helpers;
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
    public T ComputeFVD(Tensor<T> realVideos, Tensor<T> generatedVideos)
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
    public T ComputeFVDWithStats(
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
            // Determine video format (NTCHW or NCTHW)            // Heuristic: If second dimension > 3, assume temporal (NTCHW format)            // since channels are typically 1-3. For videos with exactly 3 frames,            // this may misidentify the format - ensure input uses proper format.
            bool isNTCHW = videos.Shape[1] > 3;

            for (int v = 0; v < numVideos; v++)
            {
                // Extract and sample frames from video
                var videoClip = ExtractVideoClip(videos, v, isNTCHW);

                // Run through 3D CNN
                var output = FeatureNetwork.Predict(videoClip);

                // Store features (global average pooled if needed)
                var videoFeature = GlobalAveragePool(output);
                Engine.SetRow(features, v, FrechetDistanceHelper<T>.FeatureRow(videoFeature, FeatureDimension));
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

        // Gather the sampled frames into a [1, C, T, H, W] clip.
        var engine = Engine;
        var frames = new Tensor<int>(frameIndices, new[] { FramesPerClip });
        Tensor<T> clip;
        if (isNTCHW)
        {
            // [T, C, H, W] -> sampled [T', C, H, W] -> [C, T', H, W]
            var video = engine.Reshape(
                engine.TensorSlice(videos, new[] { videoIdx, 0, 0, 0, 0 }, new[] { 1, totalFrames, channels, height, width }),
                new[] { totalFrames, channels, height, width });
            clip = engine.TensorPermute(engine.TensorIndexSelect(video, frames, 0), new[] { 1, 0, 2, 3 });
        }
        else
        {
            // [C, T, H, W] -> sampled [C, T', H, W]
            var video = engine.Reshape(
                engine.TensorSlice(videos, new[] { videoIdx, 0, 0, 0, 0 }, new[] { 1, channels, totalFrames, height, width }),
                new[] { channels, totalFrames, height, width });
            clip = engine.TensorIndexSelect(video, frames, 1);
        }

        return engine.Reshape(clip, new[] { 1, channels, FramesPerClip, height, width });
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
                var random = RandomHelper.CreateSeededRandom(42);
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
    private Tensor<T> GlobalAveragePool(Tensor<T> tensor)
    {
        // Assuming tensor is [1, C, T, H, W] or [1, C, H, W] or [1, D]
        var engine = Engine;
        if (tensor.Rank <= 2)
        {
            // Already reduced: the first Shape[last] values
            int featureCount = tensor.Shape[tensor.Rank - 1];
            return engine.TensorSlice(engine.Reshape(tensor, new[] { tensor.Length }), new[] { 0 }, new[] { featureCount });
        }

        // Mean over all spatial and temporal positions of each channel
        int channels = tensor.Shape[1];
        return engine.ReduceMean(engine.Reshape(tensor, new[] { channels, tensor.Length / channels }), new[] { 1 }, keepDims: false);
    }

    /// <summary>
    /// Computes mean and covariance matrix of feature vectors.
    /// </summary>
    private (Vector<T> mean, Matrix<T> covariance) ComputeStatistics(Matrix<T> features)
    {
        return FrechetDistanceHelper<T>.ComputeStatistics(features);
    }

    /// <summary>
    /// Computes the Fréchet distance between two Gaussian distributions.
    /// </summary>
    private T ComputeFrechetDistance(
        Vector<T> mean1,
        Matrix<T> cov1,
        Vector<T> mean2,
        Matrix<T> cov2)
    {
        return FrechetDistanceHelper<T>.FrechetDistance(mean1, cov1, mean2, cov2);
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
