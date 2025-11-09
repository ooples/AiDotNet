# Issue #398: Junior Developer Implementation Guide
## Video AI Models (TimeSformer, VideoMAE, CLIP)

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Video Processing Fundamentals](#video-processing-fundamentals)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Strategy](#implementation-strategy)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Understanding the Problem

### What Are We Building?

We're implementing support for **video AI models** that can:
- **Video Classification**: Recognize actions, scenes, events (TimeSformer, VideoMAE)
- **Video-Text Alignment**: Match videos with descriptions (CLIP for video)
- **Temporal Reasoning**: Understand motion, transitions, and sequences
- **Video Retrieval**: Search videos by text or similarity
- **Action Detection**: Identify what's happening and when

### Why Video Models Are Special

Video adds complexity beyond images:
1. **Temporal dimension**: Understanding motion and change over time
2. **Massive data**: 30 FPS × 10 seconds = 300 frames to process
3. **Spatial-temporal patterns**: Need to model both "what" and "when"
4. **Computational cost**: Processing hundreds of frames is expensive

### Real-World Use Cases

- **Content moderation**: Detect inappropriate content in videos
- **Sports analysis**: Recognize actions (goal, foul, tackle)
- **Security**: Detect anomalies, track people
- **Video search**: "Find clips of a cat jumping"
- **Video understanding**: Question answering about video content
- **Autonomous driving**: Understand dynamic scenes

---

## Video Processing Fundamentals

### Understanding Video Data

#### 1. Video as Frame Sequence
```csharp
/// <summary>
/// Video represented as a sequence of image frames.
/// Shape: [time_frames, channels, height, width]
/// </summary>
/// <remarks>
/// For Beginners:
/// A video is just a series of images (frames) shown quickly.
/// - Time: Which frame (0, 1, 2, ...)
/// - Channels: RGB (3 channels)
/// - Height, Width: Image dimensions
///
/// Example:
/// - 10-second video at 30 FPS = 300 frames
/// - Each frame: [3, 224, 224] (RGB, 224x224 pixels)
/// - Total shape: [300, 3, 224, 224]
///
/// Challenges:
/// - 300 frames is a lot of data to process
/// - Most models sample frames (e.g., take every 4th frame)
/// - Trade-off: more frames = better temporal understanding but slower
/// </remarks>
public class VideoTensor<T>
{
    // Shape: [num_frames, channels, height, width]
    // Example: [300, 3, 224, 224] for 10 seconds at 30 FPS
    public Tensor<T> Frames { get; set; } = new Tensor<T>(new[] { 0, 3, 224, 224 });

    public int NumFrames { get; set; }     // Total frames
    public int Channels { get; set; }      // 3 for RGB, 1 for grayscale
    public int Height { get; set; }        // Frame height in pixels
    public int Width { get; set; }         // Frame width in pixels
    public double FPS { get; set; }        // Frames per second

    public double DurationSeconds => NumFrames / FPS;
}
```

#### 2. Frame Sampling Strategies
```csharp
/// <summary>
/// Different strategies for sampling frames from video.
/// </summary>
/// <remarks>
/// For Beginners:
/// We can't process all 300 frames - too slow and memory-intensive.
/// Instead, we sample a subset:
///
/// 1. Uniform Sampling: Take every Nth frame
///    - Example: [0, 10, 20, 30, ...] (every 10th frame)
///    - Simple, but might miss important events
///
/// 2. Random Sampling: Pick frames randomly
///    - Good for training (data augmentation)
///    - Not deterministic (different each time)
///
/// 3. Dense Sampling: Take multiple clips
///    - Example: 3 clips of 16 frames each
///    - Better coverage, but more computation
///
/// 4. Adaptive Sampling: Sample more where motion happens
///    - Smart, but requires motion detection first
/// </remarks>
public enum FrameSamplingStrategy
{
    Uniform,      // Evenly spaced frames
    Random,       // Random frames
    Dense,        // Multiple dense clips
    Center        // Frames around center of video
}

public class VideoFrameSampler<T>
{
    public Tensor<T> Sample(
        VideoTensor<T> video,
        int targetFrames,
        FrameSamplingStrategy strategy)
    {
        Guard.NotNull(video, nameof(video));
        Guard.Positive(targetFrames, nameof(targetFrames));

        return strategy switch
        {
            FrameSamplingStrategy.Uniform => UniformSample(video, targetFrames),
            FrameSamplingStrategy.Random => RandomSample(video, targetFrames),
            FrameSamplingStrategy.Center => CenterSample(video, targetFrames),
            FrameSamplingStrategy.Dense => DenseSample(video, targetFrames),
            _ => throw new ArgumentException($"Unknown strategy: {strategy}")
        };
    }

    private Tensor<T> UniformSample(VideoTensor<T> video, int targetFrames)
    {
        // Take evenly spaced frames
        var sampled = new Tensor<T>(new[]
        {
            targetFrames,
            video.Channels,
            video.Height,
            video.Width
        });

        for (int i = 0; i < targetFrames; i++)
        {
            // Map target frame index to source frame index
            int sourceIdx = (int)((double)i * video.NumFrames / targetFrames);
            sourceIdx = Math.Min(sourceIdx, video.NumFrames - 1);

            // Copy frame
            for (int c = 0; c < video.Channels; c++)
            {
                for (int h = 0; h < video.Height; h++)
                {
                    for (int w = 0; w < video.Width; w++)
                    {
                        sampled[i, c, h, w] = video.Frames[sourceIdx, c, h, w];
                    }
                }
            }
        }

        return sampled;
    }

    private Tensor<T> RandomSample(VideoTensor<T> video, int targetFrames)
    {
        var random = new Random();
        var indices = Enumerable.Range(0, video.NumFrames)
            .OrderBy(x => random.Next())
            .Take(targetFrames)
            .OrderBy(x => x)  // Sort to maintain temporal order
            .ToArray();

        var sampled = new Tensor<T>(new[]
        {
            targetFrames,
            video.Channels,
            video.Height,
            video.Width
        });

        for (int i = 0; i < targetFrames; i++)
        {
            int sourceIdx = indices[i];

            for (int c = 0; c < video.Channels; c++)
            {
                for (int h = 0; h < video.Height; h++)
                {
                    for (int w = 0; w < video.Width; w++)
                    {
                        sampled[i, c, h, w] = video.Frames[sourceIdx, c, h, w];
                    }
                }
            }
        }

        return sampled;
    }

    private Tensor<T> CenterSample(VideoTensor<T> video, int targetFrames)
    {
        // Take frames centered around the middle of the video
        int centerFrame = video.NumFrames / 2;
        int startFrame = Math.Max(0, centerFrame - targetFrames / 2);
        int endFrame = Math.Min(video.NumFrames, startFrame + targetFrames);

        var sampled = new Tensor<T>(new[]
        {
            targetFrames,
            video.Channels,
            video.Height,
            video.Width
        });

        for (int i = 0; i < targetFrames; i++)
        {
            int sourceIdx = startFrame + i;
            if (sourceIdx >= video.NumFrames)
                sourceIdx = video.NumFrames - 1;

            for (int c = 0; c < video.Channels; c++)
            {
                for (int h = 0; h < video.Height; h++)
                {
                    for (int w = 0; w < video.Width; w++)
                    {
                        sampled[i, c, h, w] = video.Frames[sourceIdx, c, h, w];
                    }
                }
            }
        }

        return sampled;
    }

    private Tensor<T> DenseSample(VideoTensor<T> video, int targetFrames)
    {
        // Sample multiple short clips densely
        // This is a simplified version - real implementation would return multiple clips
        return UniformSample(video, targetFrames);
    }
}
```

#### 3. Video Preprocessing Pipeline
```csharp
/// <summary>
/// Standard preprocessing pipeline for video models.
/// </summary>
/// <remarks>
/// For Beginners:
/// Converting raw video to model input:
///
/// 1. Decode video → frames (raw pixels)
/// 2. Sample frames → reduce from 300 to 16-32 frames
/// 3. Resize frames → standardize to 224x224
/// 4. Normalize pixels → scale to [-1, 1] or [0, 1]
/// 5. Rearrange dimensions → match model input format
///
/// Common input formats:
/// - TimeSformer: [batch, frames, channels, height, width]
/// - I3D: [batch, channels, frames, height, width]
/// - VideoMAE: [batch, frames, channels, height, width]
/// </remarks>
public class VideoPreprocessor<T>
{
    private readonly int _targetFrames;
    private readonly int _targetHeight;
    private readonly int _targetWidth;
    private readonly FrameSamplingStrategy _samplingStrategy;
    private readonly VideoFrameSampler<T> _sampler;
    private readonly ImageNormalizer<T> _normalizer;

    public VideoPreprocessor(
        int targetFrames = 16,
        int targetHeight = 224,
        int targetWidth = 224,
        FrameSamplingStrategy samplingStrategy = FrameSamplingStrategy.Uniform)
    {
        Guard.Positive(targetFrames, nameof(targetFrames));
        Guard.Positive(targetHeight, nameof(targetHeight));
        Guard.Positive(targetWidth, nameof(targetWidth));

        _targetFrames = targetFrames;
        _targetHeight = targetHeight;
        _targetWidth = targetWidth;
        _samplingStrategy = samplingStrategy;

        _sampler = new VideoFrameSampler<T>();
        _normalizer = new ImageNormalizer<T>(
            mean: new[] { 0.485, 0.456, 0.406 },  // ImageNet mean
            std: new[] { 0.229, 0.224, 0.225 });   // ImageNet std
    }

    public Tensor<T> Process(VideoTensor<T> video)
    {
        Guard.NotNull(video, nameof(video));

        // Step 1: Sample frames
        var sampled = _sampler.Sample(video, _targetFrames, _samplingStrategy);

        // Step 2: Resize each frame
        var resized = ResizeFrames(sampled, _targetHeight, _targetWidth);

        // Step 3: Normalize pixels
        var normalized = _normalizer.Normalize(resized);

        return normalized;
    }

    private Tensor<T> ResizeFrames(Tensor<T> frames, int targetH, int targetW)
    {
        int numFrames = frames.Shape[0];
        int channels = frames.Shape[1];
        int srcH = frames.Shape[2];
        int srcW = frames.Shape[3];

        var resized = new Tensor<T>(new[] { numFrames, channels, targetH, targetW });

        // Resize each frame using bilinear interpolation
        for (int f = 0; f < numFrames; f++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        // Map target pixel to source pixel
                        double srcY = h * (double)srcH / targetH;
                        double srcX = w * (double)srcW / targetW;

                        // Bilinear interpolation
                        int y0 = (int)srcY;
                        int y1 = Math.Min(y0 + 1, srcH - 1);
                        int x0 = (int)srcX;
                        int x1 = Math.Min(x0 + 1, srcW - 1);

                        double fy = srcY - y0;
                        double fx = srcX - x0;

                        dynamic v00 = frames[f, c, y0, x0];
                        dynamic v01 = frames[f, c, y0, x1];
                        dynamic v10 = frames[f, c, y1, x0];
                        dynamic v11 = frames[f, c, y1, x1];

                        dynamic interpolated =
                            v00 * (1 - fx) * (1 - fy) +
                            v01 * fx * (1 - fy) +
                            v10 * (1 - fx) * fy +
                            v11 * fx * fy;

                        resized[f, c, h, w] = (T)(object)interpolated;
                    }
                }
            }
        }

        return resized;
    }
}
```

---

## Architecture Overview

### Model Taxonomy

```
Video AI Models
├── Frame-based (Process frames independently, then aggregate)
│   ├── Two-Stream Networks (RGB + Optical Flow)
│   ├── C3D (3D Convolutions)
│   └── I3D (Inflated 3D ConvNets)
│
├── Transformer-based (Attention across space and time)
│   ├── TimeSformer (Divided space-time attention)
│   ├── VideoMAE (Masked autoencoder for video)
│   ├── ViViT (Video Vision Transformer)
│   └── MViT (Multiscale Vision Transformers)
│
└── Video-Text Models (Joint video-text understanding)
    ├── CLIP for Video (Contrastive learning)
    ├── VideoCLIP (Temporal extensions)
    └── VIOLET (Video-Language Transformer)
```

### TimeSformer Architecture

```csharp
/// <summary>
/// TimeSformer: Space-time attention model for video understanding.
/// Uses divided attention (separate spatial and temporal attention).
/// </summary>
/// <remarks>
/// For Beginners:
/// TimeSformer processes video in three stages:
///
/// 1. Patch Embedding:
///    - Split each frame into patches (like ViT)
///    - Each patch becomes a token
///    - Add positional and temporal encodings
///
/// 2. Divided Attention:
///    - Spatial attention: Attend to patches in same frame
///    - Temporal attention: Attend to same patch across frames
///    - More efficient than full space-time attention
///
/// 3. Classification:
///    - Global average pooling over time and space
///    - Linear classifier for action labels
///
/// Input: [batch, frames, channels, height, width]
/// Example: [1, 16, 3, 224, 224]
/// → Patch embedding → [batch, frames, num_patches, hidden_dim]
/// → Attention blocks → [batch, frames, num_patches, hidden_dim]
/// → Pooling → [batch, hidden_dim]
/// → Classifier → [batch, num_classes]
/// </remarks>
public class TimeSformerModel<T> : IVideoModel<T>
{
    private readonly TimeSformerConfig _config;
    private readonly PatchEmbedding<T> _patchEmbed;
    private readonly PositionalEncoding<T> _posEncoding;
    private readonly TemporalEncoding<T> _temporalEncoding;
    private readonly List<DividedAttentionBlock<T>> _blocks;
    private readonly LayerNorm<T> _norm;
    private readonly LinearLayer<T> _classifier;

    public TimeSformerModel(TimeSformerConfig config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        // Patch embedding: [frames, 3, 224, 224] → [frames, num_patches, hidden_dim]
        _patchEmbed = new PatchEmbedding<T>(
            imageSize: config.ImageSize,
            patchSize: config.PatchSize,
            inChannels: 3,
            embedDim: config.HiddenSize);

        int numPatches = (config.ImageSize / config.PatchSize) *
                        (config.ImageSize / config.PatchSize);

        _posEncoding = new PositionalEncoding<T>(
            maxLen: numPatches + 1,  // +1 for CLS token
            hiddenSize: config.HiddenSize);

        _temporalEncoding = new TemporalEncoding<T>(
            maxFrames: config.NumFrames,
            hiddenSize: config.HiddenSize);

        // Divided attention blocks
        _blocks = new List<DividedAttentionBlock<T>>();
        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks.Add(new DividedAttentionBlock<T>(
                hiddenSize: config.HiddenSize,
                numHeads: config.NumAttentionHeads,
                mlpDim: config.IntermediateSize,
                dropoutRate: config.DropoutRate));
        }

        _norm = new LayerNorm<T>(config.HiddenSize);

        _classifier = new LinearLayer<T>(
            inputSize: config.HiddenSize,
            outputSize: config.NumClasses);
    }

    public VideoModelOutput<T> Forward(Tensor<T> video)
    {
        Guard.NotNull(video, nameof(video));

        // video: [batch, frames, channels, height, width]
        int batch = video.Shape[0];
        int frames = video.Shape[1];

        // Step 1: Embed each frame independently
        // Process each frame: [batch*frames, channels, height, width]
        var flattenedVideo = ReshapeForPatching(video);

        // Patch embedding: [batch*frames, num_patches, hidden_dim]
        var patches = _patchEmbed.Forward(flattenedVideo);

        // Add CLS token to each frame
        var withCls = AddClassToken(patches, batch, frames);

        // withCls: [batch, frames, num_patches+1, hidden_dim]

        // Step 2: Add positional and temporal encodings
        var encoded = AddEncodings(withCls);

        // Step 3: Apply divided attention blocks
        var x = encoded;
        foreach (var block in _blocks)
        {
            x = block.Forward(x, numFrames: frames);
        }

        // Step 4: Layer norm
        x = _norm.Forward(x);

        // Step 5: Extract CLS tokens and average over time
        // x: [batch, frames, num_patches+1, hidden_dim]
        // Take CLS token (index 0) from each frame
        var clsTokens = ExtractClassTokens(x, batch, frames);

        // Average over time: [batch, frames, hidden_dim] → [batch, hidden_dim]
        var pooled = TemporalPooling(clsTokens);

        // Step 6: Classification
        var logits = _classifier.Forward(pooled);

        return new VideoModelOutput<T>
        {
            Logits = logits,
            HiddenStates = x,
            Pooled = pooled
        };
    }

    private Tensor<T> ReshapeForPatching(Tensor<T> video)
    {
        // [batch, frames, channels, height, width]
        // → [batch*frames, channels, height, width]
        int batch = video.Shape[0];
        int frames = video.Shape[1];
        int channels = video.Shape[2];
        int height = video.Shape[3];
        int width = video.Shape[4];

        var reshaped = new Tensor<T>(new[]
        {
            batch * frames,
            channels,
            height,
            width
        });

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < frames; f++)
            {
                int idx = b * frames + f;
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            reshaped[idx, c, h, w] = video[b, f, c, h, w];
                        }
                    }
                }
            }
        }

        return reshaped;
    }

    private Tensor<T> AddClassToken(Tensor<T> patches, int batch, int frames)
    {
        // patches: [batch*frames, num_patches, hidden_dim]
        // Add CLS token as first token
        int numPatches = patches.Shape[1];
        int hiddenDim = patches.Shape[2];

        var withCls = new Tensor<T>(new[]
        {
            batch * frames,
            numPatches + 1,
            hiddenDim
        });

        // Initialize CLS tokens (learnable parameter in production)
        for (int i = 0; i < batch * frames; i++)
        {
            for (int d = 0; d < hiddenDim; d++)
            {
                withCls[i, 0, d] = (T)(object)0.0;  // CLS token
            }

            for (int p = 0; p < numPatches; p++)
            {
                for (int d = 0; d < hiddenDim; d++)
                {
                    withCls[i, p + 1, d] = patches[i, p, d];
                }
            }
        }

        return withCls;
    }

    private Tensor<T> AddEncodings(Tensor<T> tokens)
    {
        // Add positional encoding (spatial) and temporal encoding
        var withPos = _posEncoding.Forward(tokens);
        var withTemporal = _temporalEncoding.Forward(withPos);
        return withTemporal;
    }

    private Tensor<T> ExtractClassTokens(Tensor<T> x, int batch, int frames)
    {
        // x: [batch*frames, num_patches+1, hidden_dim]
        // Extract CLS tokens: [batch, frames, hidden_dim]
        int hiddenDim = x.Shape[2];

        var clsTokens = new Tensor<T>(new[] { batch, frames, hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < frames; f++)
            {
                int idx = b * frames + f;
                for (int d = 0; d < hiddenDim; d++)
                {
                    clsTokens[b, f, d] = x[idx, 0, d];  // CLS is at index 0
                }
            }
        }

        return clsTokens;
    }

    private Tensor<T> TemporalPooling(Tensor<T> clsTokens)
    {
        // clsTokens: [batch, frames, hidden_dim]
        // Average over time: [batch, hidden_dim]
        int batch = clsTokens.Shape[0];
        int frames = clsTokens.Shape[1];
        int hiddenDim = clsTokens.Shape[2];

        var pooled = new Tensor<T>(new[] { batch, hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int d = 0; d < hiddenDim; d++)
            {
                dynamic sum = (T)(object)0.0;
                for (int f = 0; f < frames; f++)
                {
                    sum += clsTokens[b, f, d];
                }
                pooled[b, d] = (T)(object)(sum / frames);
            }
        }

        return pooled;
    }
}

public class TimeSformerConfig
{
    public int ImageSize { get; set; } = 224;
    public int PatchSize { get; set; } = 16;
    public int NumFrames { get; set; } = 16;
    public int HiddenSize { get; set; } = 768;
    public int NumLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 12;
    public int IntermediateSize { get; set; } = 3072;
    public int NumClasses { get; set; } = 400;  // Kinetics-400
    public double DropoutRate { get; set; } = 0.1;
}
```

### Divided Attention Block

```csharp
/// <summary>
/// Divided attention: separate temporal and spatial attention.
/// More efficient than joint space-time attention.
/// </summary>
/// <remarks>
/// For Beginners:
/// Instead of attending to all patches in all frames (expensive),
/// we divide attention into two steps:
///
/// 1. Temporal Attention:
///    - For each spatial position, attend across time
///    - "What is this patch doing over time?"
///    - Example: Same location in frame 1, 2, 3, ...
///
/// 2. Spatial Attention:
///    - Within each frame, attend to other patches
///    - "What objects are in this frame?"
///    - Example: All patches in frame 5
///
/// Benefits:
/// - Reduces complexity from O(T²P²) to O(TP² + T²P)
/// - T = num frames, P = num patches
/// - Still captures space-time relationships
/// </remarks>
public class DividedAttentionBlock<T>
{
    private readonly MultiHeadAttention<T> _temporalAttention;
    private readonly MultiHeadAttention<T> _spatialAttention;
    private readonly LayerNorm<T> _norm1;
    private readonly LayerNorm<T> _norm2;
    private readonly LayerNorm<T> _norm3;
    private readonly FeedForwardNetwork<T> _ffn;

    public DividedAttentionBlock(
        int hiddenSize,
        int numHeads,
        int mlpDim,
        double dropoutRate)
    {
        Guard.Positive(hiddenSize, nameof(hiddenSize));
        Guard.Positive(numHeads, nameof(numHeads));

        _temporalAttention = new MultiHeadAttention<T>(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            dropoutRate: dropoutRate);

        _spatialAttention = new MultiHeadAttention<T>(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            dropoutRate: dropoutRate);

        _norm1 = new LayerNorm<T>(hiddenSize);
        _norm2 = new LayerNorm<T>(hiddenSize);
        _norm3 = new LayerNorm<T>(hiddenSize);

        _ffn = new FeedForwardNetwork<T>(
            inputSize: hiddenSize,
            hiddenSize: mlpDim,
            outputSize: hiddenSize,
            dropoutRate: dropoutRate);
    }

    public Tensor<T> Forward(Tensor<T> x, int numFrames)
    {
        Guard.NotNull(x, nameof(x));

        // x: [batch*frames, num_patches+1, hidden_dim]

        // Step 1: Temporal attention (attend across frames)
        var temporalOut = TemporalAttention(x, numFrames);
        x = x + temporalOut;  // Residual connection
        x = _norm1.Forward(x);

        // Step 2: Spatial attention (attend within frames)
        var spatialOut = _spatialAttention.Forward(x, x, x);
        x = x + spatialOut;  // Residual connection
        x = _norm2.Forward(x);

        // Step 3: Feed-forward network
        var ffnOut = _ffn.Forward(x);
        x = x + ffnOut;  // Residual connection
        x = _norm3.Forward(x);

        return x;
    }

    private Tensor<T> TemporalAttention(Tensor<T> x, int numFrames)
    {
        // x: [batch*frames, num_patches+1, hidden_dim]
        // Rearrange to [batch*num_patches, frames, hidden_dim]
        int batchFrames = x.Shape[0];
        int batch = batchFrames / numFrames;
        int numPatches = x.Shape[1];
        int hiddenDim = x.Shape[2];

        // Rearrange for temporal attention
        var rearranged = new Tensor<T>(new[]
        {
            batch * numPatches,
            numFrames,
            hiddenDim
        });

        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    int srcIdx = b * numFrames + f;
                    int dstIdx = b * numPatches + p;

                    for (int d = 0; d < hiddenDim; d++)
                    {
                        rearranged[dstIdx, f, d] = x[srcIdx, p, d];
                    }
                }
            }
        }

        // Apply temporal attention
        var attended = _temporalAttention.Forward(
            rearranged,
            rearranged,
            rearranged);

        // Rearrange back to [batch*frames, num_patches, hidden_dim]
        var output = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    int srcIdx = b * numPatches + p;
                    int dstIdx = b * numFrames + f;

                    for (int d = 0; d < hiddenDim; d++)
                    {
                        output[dstIdx, p, d] = attended[srcIdx, f, d];
                    }
                }
            }
        }

        return output;
    }
}
```

### VideoMAE Architecture

```csharp
/// <summary>
/// VideoMAE: Masked Autoencoder for video self-supervised learning.
/// Masks patches in space and time, learns to reconstruct them.
/// </summary>
/// <remarks>
/// For Beginners:
/// VideoMAE learns video representations by:
///
/// 1. Masking: Hide 90% of patches (very high masking ratio)
/// 2. Encoding: Process visible patches with transformer
/// 3. Decoding: Predict masked patches from visible ones
/// 4. Reconstruction: Match original video
///
/// Why it works:
/// - Forces model to understand temporal relationships
/// - Can't just copy neighbors (too many masked)
/// - Learns motion, object permanence, physics
///
/// After pretraining, use encoder for downstream tasks:
/// - Action recognition
/// - Video retrieval
/// - Anomaly detection
/// </remarks>
public class VideoMAEModel<T> : IVideoModel<T>
{
    private readonly VideoMAEConfig _config;
    private readonly PatchEmbedding<T> _patchEmbed;
    private readonly PositionalEncoding<T> _posEncoding;
    private readonly TransformerEncoder<T> _encoder;
    private readonly TransformerDecoder<T> _decoder;
    private readonly LinearLayer<T> _reconstructionHead;

    public VideoMAEModel(VideoMAEConfig config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        _patchEmbed = new PatchEmbedding<T>(
            imageSize: config.ImageSize,
            patchSize: config.PatchSize,
            inChannels: 3,
            embedDim: config.HiddenSize);

        int totalPatches = config.NumFrames *
            (config.ImageSize / config.PatchSize) *
            (config.ImageSize / config.PatchSize);

        _posEncoding = new PositionalEncoding<T>(
            maxLen: totalPatches,
            hiddenSize: config.HiddenSize);

        _encoder = new TransformerEncoder<T>(
            numLayers: config.EncoderLayers,
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            intermediateSize: config.IntermediateSize,
            dropoutRate: config.DropoutRate);

        _decoder = new TransformerDecoder<T>(
            numLayers: config.DecoderLayers,
            hiddenSize: config.DecoderHiddenSize,
            numHeads: config.DecoderNumHeads,
            intermediateSize: config.DecoderIntermediateSize,
            dropoutRate: config.DropoutRate);

        int patchDim = 3 * config.PatchSize * config.PatchSize;
        _reconstructionHead = new LinearLayer<T>(
            inputSize: config.DecoderHiddenSize,
            outputSize: patchDim);
    }

    public VideoMAEOutput<T> Forward(
        Tensor<T> video,
        double maskRatio = 0.9,
        bool returnReconstruction = true)
    {
        Guard.NotNull(video, nameof(video));

        // video: [batch, frames, channels, height, width]

        // Step 1: Patchify video
        var patches = PatchifyVideo(video);
        // patches: [batch, num_patches, patch_dim]

        // Step 2: Random masking
        var (visiblePatches, maskIndices) = RandomMask(patches, maskRatio);

        // Step 3: Add positional encoding to visible patches
        var encoded = _posEncoding.Forward(visiblePatches);

        // Step 4: Encode visible patches
        var encoderOutput = _encoder.Forward(encoded);

        if (!returnReconstruction)
        {
            return new VideoMAEOutput<T>
            {
                EncoderOutput = encoderOutput,
                MaskIndices = maskIndices
            };
        }

        // Step 5: Decode to reconstruct all patches
        var decoderInput = InsertMaskTokens(
            encoderOutput,
            maskIndices,
            patches.Shape[1]);

        var decoderOutput = _decoder.Forward(decoderInput);

        // Step 6: Predict pixel values
        var reconstruction = _reconstructionHead.Forward(decoderOutput);

        // Step 7: Compute reconstruction loss (only on masked patches)
        var loss = ComputeReconstructionLoss(
            reconstruction,
            patches,
            maskIndices);

        return new VideoMAEOutput<T>
        {
            EncoderOutput = encoderOutput,
            Reconstruction = reconstruction,
            Loss = loss,
            MaskIndices = maskIndices
        };
    }

    private Tensor<T> PatchifyVideo(Tensor<T> video)
    {
        // Convert video to sequence of patches
        // [batch, frames, channels, height, width]
        // → [batch, frames*num_patches_per_frame, patch_dim]

        int batch = video.Shape[0];
        int frames = video.Shape[1];
        int patchSize = _config.PatchSize;
        int patchesPerSide = _config.ImageSize / patchSize;
        int patchesPerFrame = patchesPerSide * patchesPerSide;
        int totalPatches = frames * patchesPerFrame;
        int patchDim = 3 * patchSize * patchSize;

        var patches = new Tensor<T>(new[] { batch, totalPatches, patchDim });

        for (int b = 0; b < batch; b++)
        {
            int patchIdx = 0;
            for (int f = 0; f < frames; f++)
            {
                for (int py = 0; py < patchesPerSide; py++)
                {
                    for (int px = 0; px < patchesPerSide; px++)
                    {
                        int dim = 0;
                        for (int c = 0; c < 3; c++)
                        {
                            for (int h = 0; h < patchSize; h++)
                            {
                                for (int w = 0; w < patchSize; w++)
                                {
                                    int y = py * patchSize + h;
                                    int x = px * patchSize + w;
                                    patches[b, patchIdx, dim++] =
                                        video[b, f, c, y, x];
                                }
                            }
                        }
                        patchIdx++;
                    }
                }
            }
        }

        return patches;
    }

    private (Tensor<T> visible, int[] maskIndices) RandomMask(
        Tensor<T> patches,
        double maskRatio)
    {
        int batch = patches.Shape[0];
        int numPatches = patches.Shape[1];
        int patchDim = patches.Shape[2];

        int numMasked = (int)(numPatches * maskRatio);
        int numVisible = numPatches - numMasked;

        // Random shuffle indices
        var random = new Random();
        var indices = Enumerable.Range(0, numPatches)
            .OrderBy(x => random.Next())
            .ToArray();

        var visibleIndices = indices.Take(numVisible).ToArray();
        var maskIndices = indices.Skip(numVisible).ToArray();

        // Extract visible patches
        var visible = new Tensor<T>(new[] { batch, numVisible, patchDim });

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < numVisible; i++)
            {
                int srcIdx = visibleIndices[i];
                for (int d = 0; d < patchDim; d++)
                {
                    visible[b, i, d] = patches[b, srcIdx, d];
                }
            }
        }

        return (visible, maskIndices);
    }

    private Tensor<T> InsertMaskTokens(
        Tensor<T> encoded,
        int[] maskIndices,
        int totalPatches)
    {
        // Insert learnable mask tokens at masked positions
        // This is simplified - real implementation uses learnable parameters
        int batch = encoded.Shape[0];
        int hiddenDim = encoded.Shape[2];

        var full = new Tensor<T>(new[] { batch, totalPatches, hiddenDim });

        // Fill with mask tokens (zero for simplicity)
        // In production, these would be learnable parameters

        return full;
    }

    private Tensor<T> ComputeReconstructionLoss(
        Tensor<T> reconstruction,
        Tensor<T> original,
        int[] maskIndices)
    {
        // MSE loss on masked patches only
        // This is a placeholder - real implementation needed
        return new Tensor<T>(new[] { 1 });
    }
}

public class VideoMAEConfig
{
    public int ImageSize { get; set; } = 224;
    public int PatchSize { get; set; } = 16;
    public int NumFrames { get; set; } = 16;
    public int HiddenSize { get; set; } = 768;
    public int EncoderLayers { get; set; } = 12;
    public int DecoderLayers { get; set; } = 4;
    public int NumAttentionHeads { get; set; } = 12;
    public int DecoderNumHeads { get; set; } = 8;
    public int IntermediateSize { get; set; } = 3072;
    public int DecoderHiddenSize { get; set; } = 384;
    public int DecoderIntermediateSize { get; set; } = 1536;
    public double DropoutRate { get; set; } = 0.1;
}
```

### CLIP for Video

```csharp
/// <summary>
/// CLIP adapted for video: learns joint video-text embeddings.
/// Enables zero-shot video classification and video-text retrieval.
/// </summary>
/// <remarks>
/// For Beginners:
/// CLIP for video learns to match videos with text descriptions:
///
/// 1. Video Encoder:
///    - Encodes video → vector representation
///    - Uses TimeSformer or similar architecture
///
/// 2. Text Encoder:
///    - Encodes text description → vector representation
///    - Uses transformer (BERT-like)
///
/// 3. Contrastive Learning:
///    - Match video embeddings with correct text embeddings
///    - Push away incorrect matches
///
/// Applications:
/// - Zero-shot classification: "Does this video show 'person dancing'?"
/// - Video search: Find videos matching "cat playing piano"
/// - Video captioning: Generate descriptions
/// </remarks>
public class VideoCLIPModel<T> : IVideoTextModel<T>
{
    private readonly VideoCLIPConfig _config;
    private readonly TimeSformerModel<T> _videoEncoder;
    private readonly TextTransformer<T> _textEncoder;
    private readonly LinearLayer<T> _videoProjection;
    private readonly LinearLayer<T> _textProjection;

    public VideoCLIPModel(VideoCLIPConfig config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        // Video encoder (TimeSformer or similar)
        _videoEncoder = new TimeSformerModel<T>(
            new TimeSformerConfig
            {
                NumFrames = config.NumFrames,
                HiddenSize = config.VideoHiddenSize,
                NumLayers = config.VideoNumLayers,
                NumClasses = config.EmbedDim  // Project to shared embedding space
            });

        // Text encoder (BERT-like)
        _textEncoder = new TextTransformer<T>(
            vocabSize: config.TextVocabSize,
            hiddenSize: config.TextHiddenSize,
            numLayers: config.TextNumLayers,
            numHeads: config.TextNumHeads);

        // Projection layers to shared embedding space
        _videoProjection = new LinearLayer<T>(
            inputSize: config.VideoHiddenSize,
            outputSize: config.EmbedDim);

        _textProjection = new LinearLayer<T>(
            inputSize: config.TextHiddenSize,
            outputSize: config.EmbedDim);
    }

    public VideoCLIPOutput<T> Forward(
        Tensor<T> video,
        Tensor<T> textTokens)
    {
        Guard.NotNull(video, nameof(video));
        Guard.NotNull(textTokens, nameof(textTokens));

        // Step 1: Encode video
        var videoOutput = _videoEncoder.Forward(video);
        var videoFeatures = videoOutput.Pooled;

        // Step 2: Encode text
        var textOutput = _textEncoder.Forward(textTokens);
        var textFeatures = textOutput.Pooled;

        // Step 3: Project to shared embedding space
        var videoEmbeddings = _videoProjection.Forward(videoFeatures);
        var textEmbeddings = _textProjection.Forward(textFeatures);

        // Step 4: Normalize embeddings
        videoEmbeddings = L2Normalize(videoEmbeddings);
        textEmbeddings = L2Normalize(textEmbeddings);

        // Step 5: Compute similarity matrix
        var similarity = ComputeSimilarity(videoEmbeddings, textEmbeddings);

        return new VideoCLIPOutput<T>
        {
            VideoEmbeddings = videoEmbeddings,
            TextEmbeddings = textEmbeddings,
            Similarity = similarity
        };
    }

    public Tensor<T> EncodeVideo(Tensor<T> video)
    {
        var output = _videoEncoder.Forward(video);
        var projected = _videoProjection.Forward(output.Pooled);
        return L2Normalize(projected);
    }

    public Tensor<T> EncodeText(Tensor<T> textTokens)
    {
        var output = _textEncoder.Forward(textTokens);
        var projected = _textProjection.Forward(output.Pooled);
        return L2Normalize(projected);
    }

    private Tensor<T> L2Normalize(Tensor<T> embeddings)
    {
        // Normalize each embedding to unit length
        int batch = embeddings.Shape[0];
        int dim = embeddings.Shape[1];

        var normalized = new Tensor<T>(embeddings.Shape);

        for (int b = 0; b < batch; b++)
        {
            // Compute L2 norm
            double norm = 0;
            for (int d = 0; d < dim; d++)
            {
                double val = Convert.ToDouble(embeddings[b, d]);
                norm += val * val;
            }
            norm = Math.Sqrt(norm);

            if (norm < 1e-10)
                norm = 1.0;

            // Normalize
            for (int d = 0; d < dim; d++)
            {
                normalized[b, d] = (T)(object)(
                    Convert.ToDouble(embeddings[b, d]) / norm);
            }
        }

        return normalized;
    }

    private Tensor<T> ComputeSimilarity(
        Tensor<T> videoEmbeddings,
        Tensor<T> textEmbeddings)
    {
        // Cosine similarity: video_embed · text_embed
        // Since embeddings are L2-normalized, this is just dot product

        int numVideos = videoEmbeddings.Shape[0];
        int numTexts = textEmbeddings.Shape[0];
        int dim = videoEmbeddings.Shape[1];

        var similarity = new Tensor<T>(new[] { numVideos, numTexts });

        for (int v = 0; v < numVideos; v++)
        {
            for (int t = 0; t < numTexts; t++)
            {
                double sim = 0;
                for (int d = 0; d < dim; d++)
                {
                    sim += Convert.ToDouble(videoEmbeddings[v, d]) *
                           Convert.ToDouble(textEmbeddings[t, d]);
                }
                similarity[v, t] = (T)(object)sim;
            }
        }

        return similarity;
    }
}

public class VideoCLIPConfig
{
    public int NumFrames { get; set; } = 16;
    public int VideoHiddenSize { get; set; } = 768;
    public int VideoNumLayers { get; set; } = 12;
    public int TextVocabSize { get; set; } = 49408;
    public int TextHiddenSize { get; set; } = 512;
    public int TextNumLayers { get; set; } = 12;
    public int TextNumHeads { get; set; } = 8;
    public int EmbedDim { get; set; } = 512;  // Shared embedding space
}
```

---

## Implementation Strategy

### Project Structure

```
src/
├── Video/
│   ├── IVideoModel.cs
│   ├── VideoTensor.cs
│   ├── VideoFrameSampler.cs
│   └── Preprocessing/
│       ├── VideoPreprocessor.cs
│       ├── FrameExtractor.cs
│       ├── VideoNormalizer.cs
│       └── VideoAugmentation.cs
│
├── Video/Models/
│   ├── TimeSformer/
│   │   ├── TimeSformerModel.cs
│   │   ├── TimeSformerConfig.cs
│   │   ├── DividedAttentionBlock.cs
│   │   ├── TemporalEncoding.cs
│   │   └── TimeSformerProcessor.cs
│   │
│   ├── VideoMAE/
│   │   ├── VideoMAEModel.cs
│   │   ├── VideoMAEConfig.cs
│   │   ├── VideoMAEPretraining.cs
│   │   └── VideoMAEProcessor.cs
│   │
│   └── VideoCLIP/
│       ├── VideoCLIPModel.cs
│       ├── VideoCLIPConfig.cs
│       ├── VideoEncoder.cs
│       ├── TextEncoder.cs
│       └── ContrastiveLearning.cs
│
└── Video/Utils/
    ├── VideoIO.cs (Load/save videos)
    ├── OpticalFlow.cs (Motion estimation)
    └── TemporalAugmentation.cs
```

---

## Testing Strategy

### Unit Tests

```csharp
namespace AiDotNetTests.Video;

public class VideoPreprocessorTests
{
    [Fact]
    public void Process_ValidVideo_ReturnsProcessedFrames()
    {
        // Arrange
        var preprocessor = new VideoPreprocessor<double>(
            targetFrames: 16,
            targetHeight: 224,
            targetWidth: 224);

        var video = CreateTestVideo(
            numFrames: 300,
            height: 480,
            width: 640);

        // Act
        var processed = preprocessor.Process(video);

        // Assert
        Assert.NotNull(processed);
        Assert.Equal(16, processed.Shape[0]);  // 16 frames
        Assert.Equal(3, processed.Shape[1]);   // RGB
        Assert.Equal(224, processed.Shape[2]); // Height
        Assert.Equal(224, processed.Shape[3]); // Width
    }

    [Theory]
    [InlineData(FrameSamplingStrategy.Uniform)]
    [InlineData(FrameSamplingStrategy.Random)]
    [InlineData(FrameSamplingStrategy.Center)]
    public void FrameSampler_DifferentStrategies_ReturnsCorrectCount(
        FrameSamplingStrategy strategy)
    {
        // Arrange
        var sampler = new VideoFrameSampler<double>();
        var video = CreateTestVideo(100, 224, 224);

        // Act
        var sampled = sampler.Sample(video, targetFrames: 16, strategy);

        // Assert
        Assert.Equal(16, sampled.Shape[0]);
    }

    private VideoTensor<double> CreateTestVideo(
        int numFrames,
        int height,
        int width)
    {
        var frames = new Tensor<double>(new[]
        {
            numFrames,
            3,  // RGB
            height,
            width
        });

        // Fill with test pattern
        for (int f = 0; f < numFrames; f++)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        frames[f, c, h, w] = (f + c) / 255.0;
                    }
                }
            }
        }

        return new VideoTensor<double>
        {
            Frames = frames,
            NumFrames = numFrames,
            Channels = 3,
            Height = height,
            Width = width,
            FPS = 30.0
        };
    }
}
```

### Integration Tests

```csharp
public class TimeSformerIntegrationTests
{
    [Fact]
    public void TimeSformer_ProcessVideo_ReturnsLogits()
    {
        // Arrange
        var config = new TimeSformerConfig
        {
            NumFrames = 16,
            ImageSize = 224,
            NumClasses = 400
        };
        var model = new TimeSformerModel<double>(config);

        var video = CreateTestVideo(
            numFrames: 16,
            height: 224,
            width: 224);

        // Add batch dimension
        var batch = video.Frames.Unsqueeze(0);

        // Act
        var output = model.Forward(batch);

        // Assert
        Assert.NotNull(output.Logits);
        Assert.Equal(2, output.Logits.Rank);  // [batch, num_classes]
        Assert.Equal(1, output.Logits.Shape[0]);  // Batch size
        Assert.Equal(400, output.Logits.Shape[1]);  // Kinetics-400
    }

    [Fact]
    public void VideoCLIP_VideoTextMatch_HighSimilarity()
    {
        // Arrange
        var config = new VideoCLIPConfig();
        var model = new VideoCLIPModel<double>(config);

        var video = CreateTestVideo(16, 224, 224);
        var videoBatch = video.Frames.Unsqueeze(0);

        var textTokens = CreateTestTextTokens("person dancing");
        var textBatch = textTokens.Unsqueeze(0);

        // Act
        var output = model.Forward(videoBatch, textBatch);

        // Assert
        Assert.NotNull(output.Similarity);
        Assert.Equal(2, output.Similarity.Rank);  // [num_videos, num_texts]

        // Similarity should be between -1 and 1 (cosine similarity)
        double sim = Convert.ToDouble(output.Similarity[0, 0]);
        Assert.InRange(sim, -1.0, 1.0);
    }

    private Tensor<long> CreateTestTextTokens(string text)
    {
        // Simplified tokenization
        var tokens = new Tensor<long>(new[] { 10 });  // Max 10 tokens
        for (int i = 0; i < 10; i++)
        {
            tokens[i] = i + 1;  // Dummy tokens
        }
        return tokens;
    }
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Core Video Infrastructure (6 hours)

#### AC 1.1: Video Data Structures
**File**: `src/Video/VideoTensor.cs`

Implement VideoTensor class with metadata.

#### AC 1.2: Frame Sampling
**File**: `src/Video/VideoFrameSampler.cs`

Implement all sampling strategies.

#### AC 1.3: Video Preprocessing
**File**: `src/Video/Preprocessing/VideoPreprocessor.cs`

Implement complete preprocessing pipeline.

**Tests**: `tests/Video/VideoPreprocessorTests.cs`

### Phase 2: TimeSformer Implementation (10 hours)

#### AC 2.1: Divided Attention Block
**File**: `src/Video/Models/TimeSformer/DividedAttentionBlock.cs`

Implement temporal and spatial attention.

#### AC 2.2: Temporal Encoding
**File**: `src/Video/Models/TimeSformer/TemporalEncoding.cs`

Add time-aware positional encodings.

#### AC 2.3: Complete TimeSformer
**File**: `src/Video/Models/TimeSformer/TimeSformerModel.cs`

Integrate all components.

**Tests**: `tests/Video/Models/TimeSformer/TimeSformerTests.cs`

### Phase 3: VideoMAE Implementation (12 hours)

#### AC 3.1: Patch Masking
Implement random masking with high ratio (90%).

#### AC 3.2: Encoder-Decoder
Implement MAE architecture with reconstruction.

#### AC 3.3: Pretraining Loop
Create self-supervised pretraining pipeline.

**Tests**: `tests/Video/Models/VideoMAE/VideoMAETests.cs`

### Phase 4: VideoCLIP Implementation (14 hours)

#### AC 4.1: Video Encoder
Adapt TimeSformer for CLIP.

#### AC 4.2: Text Encoder
Implement transformer text encoder.

#### AC 4.3: Contrastive Learning
Implement contrastive loss and training.

**Tests**: `tests/Video/Models/VideoCLIP/VideoCLIPTests.cs`

### Phase 5: Documentation and Examples (4 hours)

#### AC 5.1: XML Documentation
Complete API documentation.

#### AC 5.2: Usage Examples
Create examples for action recognition, video search.

---

## Checklist Summary

### Phase 1: Core Infrastructure (6 hours)
- [ ] Implement VideoTensor and metadata
- [ ] Implement frame sampling strategies
- [ ] Implement video preprocessing pipeline
- [ ] Write unit tests for preprocessing
- [ ] Test with real video files

### Phase 2: TimeSformer (10 hours)
- [ ] Implement divided attention block
- [ ] Implement temporal encoding
- [ ] Create TimeSformerModel
- [ ] Write integration tests
- [ ] Test on Kinetics-400 dataset

### Phase 3: VideoMAE (12 hours)
- [ ] Implement patch masking
- [ ] Implement encoder-decoder
- [ ] Create pretraining pipeline
- [ ] Write integration tests
- [ ] Test reconstruction quality

### Phase 4: VideoCLIP (14 hours)
- [ ] Adapt TimeSformer as video encoder
- [ ] Implement text encoder
- [ ] Implement contrastive learning
- [ ] Write integration tests
- [ ] Test zero-shot classification

### Phase 5: Documentation (4 hours)
- [ ] Add XML documentation
- [ ] Create usage examples
- [ ] Write performance benchmarks

### Total Estimated Time: 46 hours

---

## Success Criteria

1. **Preprocessing**: Videos correctly sampled and normalized
2. **TimeSformer**: Achieves >70% top-1 accuracy on Kinetics-400
3. **VideoMAE**: Learns meaningful representations via reconstruction
4. **VideoCLIP**: Enables zero-shot video classification
5. **Tests**: 80%+ coverage, all integration tests pass
6. **Performance**: Can process videos in reasonable time
7. **Documentation**: Complete XML docs and examples

---

## Common Pitfalls

### Pitfall 1: Memory Overflow
**Problem**: Loading all 300 frames into memory.
**Solution**: Sample frames first, don't load entire video.

### Pitfall 2: Incorrect Dimension Order
**Problem**: [B, C, T, H, W] vs [B, T, C, H, W].
**Solution**: Document expected format clearly.

### Pitfall 3: Forgetting Temporal Context
**Problem**: Treating video as independent frames.
**Solution**: Use temporal attention or 3D convolutions.

### Pitfall 4: Slow Inference
**Problem**: Processing too many frames.
**Solution**: Sample fewer frames or use efficient attention.

---

## Resources

- [TimeSformer Paper](https://arxiv.org/abs/2102.05095)
- [VideoMAE Paper](https://arxiv.org/abs/2203.12602)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Kinetics Dataset](https://deepmind.com/research/open-source/kinetics)
- [Video Understanding Survey](https://arxiv.org/abs/2012.06567)
