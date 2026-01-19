# Video Generation with Stable Video Diffusion

This sample demonstrates text-to-video and image-to-video generation using Stable Video Diffusion (SVD) and related models.

## What You'll Learn

- How to configure video generation parameters
- How to generate videos from text prompts
- How to animate images with motion
- How to control motion intensity and style
- How to use negative prompts for quality
- How to export videos in different formats
- Best practices for prompt engineering

## What is Stable Video Diffusion?

Stable Video Diffusion (SVD) is a latent video diffusion model that generates high-quality video clips from text prompts or images. Key features:

- **High Quality**: Produces realistic, coherent video frames
- **Temporal Consistency**: Maintains smooth motion between frames
- **Flexible Input**: Works with text prompts or images
- **Controllable Motion**: Adjust motion intensity via motion bucket ID

## Available Models

| Model | Input | Frames | Resolution | Best For |
|-------|-------|--------|------------|----------|
| SVD | Image | 14-25 | 576x320 | Image animation |
| SVD-XT | Image | 25 | 576x320 | Extended motion |
| AnimateDiff | Text+LoRA | 16-32 | 512x512 | Stylized animation |
| ModelScope | Text | 16 | 256x256 | Fast generation |
| ZeroScope | Text | 24 | 576x320 | High quality |

## Running the Sample

```bash
cd samples/video/VideoGeneration
dotnet run
```

## Expected Output

```
=== AiDotNet Video Generation ===
Text-to-Video with Stable Video Diffusion

Available Video Generation Models:
  Stable Video Diffusion (SVD)     - High-quality video from image/text
  Stable Video Diffusion XT        - Extended temporal coherence
  AnimateDiff                      - Animation-focused generation
  ModelScope Text2Video            - Text-to-video generation

Configuration:
  Model: StableVideoDiffusion
  Resolution: 576x320
  Frame Count: 25
  FPS: 8
  Duration: 3.13 seconds
  Inference Steps: 25
  Guidance Scale: 7.5
  GPU Acceleration: True

Loading video generation model...
  Model loaded successfully

=== Demo 1: Text-to-Video Generation ===

Prompt: "A serene mountain lake at sunset, with golden light reflecting..."

Generating video...
  [==================================================] 100% Complete

Generation Result:
  Frames Generated: 25
  Frame Size: 576x320
  Total Pixels: 4,608,000
  Generation Time: 45.23 seconds
  Time per Frame: 1809.20ms

Frame Timeline:
  Time:    0ms  125ms  250ms  375ms  500ms ...
  Frame:    O     O      @      @      O   ...

=== Demo 4: Parameter Effects ===

Guidance Scale Effect (CFG):
------------------------------------------------------------
| Value | Effect                                      |
------------------------------------------------------------
|  1-3  | Creative, may not follow prompt closely     |
|  5-7  | Balanced quality and prompt adherence       |
| 8-12  | Strong prompt following, may be over-sharp  |
| 15+   | Very strict, can cause artifacts            |
------------------------------------------------------------
```

## Code Highlights

### Basic Text-to-Video

```csharp
// Configure video generation
var config = new VideoGenerationConfig
{
    Model = VideoModel.StableVideoDiffusion,
    Width = 576,
    Height = 320,
    NumFrames = 25,
    Fps = 8,
    NumInferenceSteps = 25,
    GuidanceScale = 7.5f
};

// Create generator
var generator = new VideoGenerator(config);

// Generate from text prompt
var result = generator.Generate(
    "A serene lake at sunset, golden reflections, cinematic quality");

// Export video
generator.ExportMp4(result, "output.mp4");
```

### Image-to-Video Animation

```csharp
// Load or create input image
var inputImage = LoadImage("photo.jpg");

// Configure animation
var animConfig = new VideoGenerationConfig
{
    Model = VideoModel.StableVideoDiffusionXT,
    MotionBucketId = 127,  // Controls motion amount (0-255)
    NumFrames = 25
};

// Animate the image
var result = generator.AnimateImage(
    inputImage,
    "Gentle camera pan with parallax",
    animConfig);
```

### Using Negative Prompts

```csharp
var options = new GenerationOptions
{
    NegativePrompt = "blurry, low quality, artifacts, jittery, " +
                     "flickering, watermark, text, deformed",
    Seed = 42  // For reproducibility
};

var result = generator.Generate(prompt, options);
```

### Progress Callback

```csharp
var result = generator.Generate(prompt, progress: (step, total) =>
{
    var percent = (int)((float)step / total * 100);
    Console.WriteLine($"Step {step}/{total} ({percent}%)");
});
```

## Configuration Parameters

### Resolution Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Width | 576 | Output video width (must be divisible by 8) |
| Height | 320 | Output video height (must be divisible by 8) |
| NumFrames | 25 | Number of frames to generate |
| Fps | 8 | Frames per second |

### Generation Quality

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| NumInferenceSteps | 25 | 10-100 | More steps = higher quality, slower |
| GuidanceScale | 7.5 | 1-20 | Higher = stricter prompt following |
| MotionBucketId | 127 | 0-255 | Higher = more motion (SVD-XT) |
| NoiseAugStrength | 0.02 | 0-1 | Noise augmentation for variety |

### Guidance Scale Guide

| Value | Effect | Use Case |
|-------|--------|----------|
| 1-3 | Very creative, may ignore prompt | Artistic exploration |
| 4-6 | Balanced creativity | General use |
| 7-9 | Strong prompt adherence | Standard quality |
| 10-15 | Very strict, sharp | Precise prompts |
| 15+ | Over-constrained | Usually avoid |

## Prompt Engineering

### Effective Prompt Structure

```
[Subject] + [Action/Motion] + [Environment] + [Style] + [Quality Keywords]
```

### Example Prompts

**Nature Scene:**
```
A majestic waterfall cascading down mossy rocks, mist rising into
golden sunlight, lush rainforest, slow motion, cinematic, 4K quality
```

**Urban Scene:**
```
Busy city intersection at night, neon signs reflecting on wet
pavement, cars with light trails, cyberpunk aesthetic, rain falling
```

**Abstract:**
```
Flowing liquid metal transforming into geometric shapes, iridescent
colors, smooth morphing transitions, abstract art, seamless loop
```

### Negative Prompt Keywords

| Category | Keywords |
|----------|----------|
| Quality | blurry, low quality, artifacts, noise, grainy, pixelated |
| Motion | jittery, flickering, choppy, unstable, stuttering |
| Content | watermark, text, logo, border, frame, signature |
| Distortion | deformed, distorted, warped, morphing errors |
| Style | oversaturated, underexposed, overexposed |

## Motion Control (SVD-XT)

### Motion Bucket ID Guide

| Range | Motion Level | Best For |
|-------|--------------|----------|
| 0-50 | Minimal | Subtle atmosphere, sky timelapse |
| 50-100 | Moderate | Gentle movement, portraits |
| 100-150 | Normal | Standard scenes |
| 150-200 | Active | Action scenes |
| 200-255 | Maximum | High motion (may lose coherence) |

```csharp
var config = new VideoGenerationConfig
{
    MotionBucketId = 80,  // Moderate motion
    NoiseAugStrength = 0.02f
};
```

## Video Export

### Supported Formats

| Format | Codec | Quality | Size | Use Case |
|--------|-------|---------|------|----------|
| MP4 | H.264 | Good | Small | Web, general |
| MP4 | H.265 | Better | Smaller | High-quality web |
| WebM | VP9 | Good | Small | Web browsers |
| MOV | ProRes | Best | Large | Professional |
| GIF | - | Limited | Medium | Social media |
| PNG | - | Lossless | Large | Post-processing |

### Export Examples

```csharp
// Export as MP4 with H.264
generator.ExportMp4(result, "output.mp4", new ExportOptions
{
    Codec = VideoCodec.H264,
    Bitrate = 8_000_000,  // 8 Mbps
    Quality = 23  // CRF (lower = higher quality)
});

// Export as GIF
generator.ExportGif(result, "output.gif", new GifOptions
{
    Width = 320,  // Reduce size for GIF
    ColorPalette = 256,
    Dithering = DitheringMethod.FloydSteinberg,
    LoopCount = 0  // Infinite loop
});

// Export individual frames
generator.ExportFrames(result, "frames/", ImageFormat.Png);
```

## Frame Interpolation

Increase FPS using frame interpolation:

```csharp
// Generate base video at 8 FPS
var config = new VideoGenerationConfig { Fps = 8, NumFrames = 25 };
var result = generator.Generate(prompt, config);

// Interpolate to 24 FPS (3x frames)
var smoothResult = generator.Interpolate(result,
    targetFps: 24,
    method: InterpolationMethod.RIFE);
```

### Interpolation Methods

| Method | Quality | Speed | Description |
|--------|---------|-------|-------------|
| Linear | Low | Fast | Simple blending |
| Optical Flow | Good | Medium | Motion-based |
| RIFE | Great | Medium | AI-based interpolation |
| Film | Best | Slow | Google's FILM model |

## GPU Memory Requirements

| Resolution | Frames | Min VRAM | Recommended |
|------------|--------|----------|-------------|
| 256x144 | 25 | 4 GB | 6 GB |
| 512x288 | 25 | 8 GB | 12 GB |
| 576x320 | 25 | 12 GB | 16 GB |
| 1024x576 | 25 | 24 GB | 32 GB |

### Memory Optimization

```csharp
var config = new VideoGenerationConfig
{
    UseGpu = true,
    Precision = Precision.FP16,  // Half precision
    EnableAttentionSlicing = true,  // Lower VRAM usage
    EnableVaeTiling = true  // For high resolution
};
```

## Architecture Overview

```
Text/Image Input
       |
       v
+------------------+
| CLIP Encoder     |  Text/image to embeddings
| (ViT-H/14)       |  768-dim latent vectors
+------------------+
       |
       v
+------------------+
| Temporal UNet    |  3D convolutions
| - Down blocks    |  Temporal attention layers
| - Mid block      |  Cross-attention with text
| - Up blocks      |  Motion modules
+------------------+
       |
       v
+------------------+
| Temporal VAE     |  Latent to pixel space
| Decoder          |  Frame-by-frame decoding
+------------------+
       |
       v
Video Frames (576x320 x 25)
```

### Key Components

- **Temporal Attention**: Maintains consistency across frames
- **Motion Modules**: Learned motion patterns
- **Cross-Frame Attention**: Links related content between frames
- **3D Convolutions**: Process spatial and temporal dimensions

## Best Practices

1. **Start with lower resolution** for prompt iteration
2. **Use motion bucket 100-150** for natural motion
3. **Include quality keywords** in prompts (cinematic, 4K, etc.)
4. **Use negative prompts** to avoid common artifacts
5. **Set seed** for reproducible results during development
6. **Export at native resolution** then upscale if needed

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of memory | Resolution too high | Reduce resolution or use FP16 |
| Flickering | Low motion coherence | Reduce motion bucket ID |
| Static video | Motion too low | Increase motion bucket ID |
| Artifacts | High guidance scale | Reduce to 7-9 range |
| Blurry output | Low inference steps | Increase to 25-50 |

## Next Steps

- [ImageGeneration](../../image/ImageGeneration/) - Generate still images
- [VideoUpscaling](../VideoUpscaling/) - Enhance video quality
- [AudioGeneration](../../audio/AudioGeneration/) - Add sound to videos

## Resources

- [Stable Video Diffusion Paper](https://arxiv.org/abs/2311.15127)
- [AnimateDiff Paper](https://arxiv.org/abs/2307.04725)
- [Video Diffusion Models](https://arxiv.org/abs/2204.03458)
