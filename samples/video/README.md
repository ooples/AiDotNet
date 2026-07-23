# Video Processing Samples

This directory contains examples of video processing models in AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [VideoGeneration](./VideoGeneration/) | Generate videos from text/images |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Video;

// Video Generation
var generator = new VideoGenerationModel<float>();
var video = await generator.GenerateAsync(
    prompt: "A cat playing piano",
    numFrames: 16,
    fps: 8);

// Action Recognition
var classifier = new ActionRecognitionModel<float>();
var action = classifier.Classify("video.mp4");
```

## Video Models (34+)

### Video Generation
- Stable Video Diffusion
- AnimateDiff
- VideoLDM

### Action Recognition
- Video Swin Transformer
- TimeSformer
- SlowFast

### Object Tracking
- SORT
- DeepSORT
- ByteTrack

### Optical Flow
- RAFT
- FlowNet2
- PWC-Net

## Learn More

- [Video Tutorial](/docs/tutorials/video/)
- [API Reference](/api/AiDotNet.Video/)
