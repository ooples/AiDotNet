# Issue #268: Junior Developer Implementation Guide

## Video Upscaling, Frame Interpolation, and Optical Flow Aids

**This issue creates NEW functionality for video enhancement, focusing initially on video upscaling.**

### What You're Building:
1. **VideoHelper**: Utilities for reading video files into frame tensors and assembling frames back into video files
2. **VideoUpscalerModel**: A wrapper that processes videos frame-by-frame through a neural upscaling model
3. **Frame-by-Frame Processing**: Memory-efficient video processing using streaming (IEnumerable)

---

## Understanding Video Processing Concepts

### What is Video Upscaling?

**Video Upscaling** (also called **super-resolution**) increases the resolution of video while attempting to add realistic detail.

**Simple Upscaling (Naive Approach)**:
```
Input:  64×64 pixel video frame
Output: 128×128 pixel frame (each pixel repeated 2×2)

Problem: Blocky, pixelated result (no new information added)
```

**Neural Upscaling (This Implementation)**:
```
Input:  64×64 pixel video frame
        ↓ Neural Network
Output: 128×128 pixel frame (with inferred details)

Result: Sharper, more realistic high-resolution video
```

**Key Differences**:
- **Naive upscaling**: Just repeats pixels (like zooming in)
- **Neural upscaling**: Uses AI to infer missing details based on learned patterns
- **Video upscaling**: Must maintain temporal consistency between frames

**Use Cases**:
- Enhancing old low-resolution videos for modern displays
- Improving video quality for streaming
- Upscaling security camera footage for better detail
- Restoring archived video content

### What is Frame-by-Frame Processing?

**Frame-by-Frame Processing** means handling video one frame at a time rather than loading the entire video into memory.

**Memory-Inefficient Approach (Load All Frames)**:
```csharp
// BAD: Loads entire video into memory
var frames = new List<Tensor<T>>();
for (int i = 0; i < totalFrames; i++)
{
    frames.Add(LoadFrame(i));  // All frames in memory!
}
// Memory usage: 1000 frames × 1920×1080×3 × 4 bytes = 24 GB!
```

**Memory-Efficient Approach (Streaming with IEnumerable)**:
```csharp
// GOOD: Process one frame at a time
public IEnumerable<Tensor<T>> ExtractFrames(string videoPath)
{
    for (int i = 0; i < totalFrames; i++)
    {
        yield return LoadFrame(i);  // Only one frame in memory at a time!
    }
}
// Memory usage: 1 frame × 1920×1080×3 × 4 bytes = 24 MB
```

**Why This Matters**:
- Videos are **large**: A 10-second 1080p video at 30fps = 300 frames × 1920×1080×3 = 1.8 GB uncompressed
- **Streaming** allows processing videos larger than available RAM
- **IEnumerable** with `yield return` provides lazy evaluation (compute on demand)

### What is FFMpeg and Why Do We Need It?

**FFMpeg** is a powerful, cross-platform library for reading, writing, and processing video/audio files.

**Why We Can't Just Use Built-in C# APIs**:
- C# doesn't have native video codec support
- Videos come in many formats: MP4, AVI, MOV, WebM, etc.
- Each format has different codecs: H.264, H.265, VP9, etc.
- FFMpeg handles all of this complexity for us

**What FFMpegCore Does**:
```csharp
// Read video metadata
var videoInfo = FFMpeg.GetMediaInfo("input.mp4");
double frameRate = videoInfo.VideoStreams[0].FrameRate;
int width = videoInfo.VideoStreams[0].Width;
int height = videoInfo.VideoStreams[0].Height;

// Extract frames as images
FFMpeg.ExtractFrames("input.mp4", outputDirectory);

// Encode frames back into video
FFMpeg.JoinImageSequence("output.mp4", frameRate, imageFiles);
```

**Installation**:
```bash
dotnet add package FFMpegCore
```

**Note**: FFMpegCore requires the FFMpeg executable to be installed on the system or distributed with your application.

### Understanding Tensor Shape for Video Frames

**Video Frame Tensor Shapes**:
```
Single Frame (RGB):
Shape: [height, width, channels]
Example: [1080, 1920, 3] for 1920×1080 (Full HD) RGB image

Video Sequence:
Shape: [num_frames, height, width, channels]
Example: [300, 1080, 1920, 3] for 10 seconds at 30fps

Batch of Frames (for neural network):
Shape: [batch_size, height, width, channels]
Example: [8, 256, 256, 3] for batch of 8 frames at 256×256
```

**Channel Order**:
- **RGB**: Red, Green, Blue (most common in ML)
- **BGR**: Blue, Green, Red (used by OpenCV)
- **Grayscale**: Single channel (for black and white)

**Value Range**:
- **Pixel values**: Typically 0-255 (uint8) or 0.0-1.0 (float normalized)
- Neural networks often expect normalized values [0, 1] or [-1, 1]

---

## Phase 1: Step-by-Step Implementation

### Understanding the Architecture

**Before coding**, understand the complete pipeline:

```
Input Video File (input.mp4)
      ↓
[VideoHelper.ExtractFrames] → IEnumerable<Tensor<T>> (streaming)
      ↓
[For each frame]
    ↓
[Neural Upscaler Model] → Upscaled Frame Tensor
    (64×64 → 128×128)
      ↓
[VideoHelper.AssembleVideo] → Output Video File (output.mp4)
```

**Key Design Decisions**:
1. **Streaming**: Use `IEnumerable` to avoid loading entire video into memory
2. **Frame Rate Preservation**: Extract and reuse the original video's frame rate
3. **ONNX Model**: Use pre-trained model for upscaling (no training code needed)
4. **Cross-Platform**: FFMpeg works on Windows, Linux, and macOS

### AC 1.1: Add Video Processing Dependency

**File**: Modify `src/AiDotNet.csproj`

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <!-- Other properties... -->
  </PropertyGroup>

  <ItemGroup>
    <!-- Existing packages... -->

    <!-- Add FFMpeg wrapper for video processing -->
    <PackageReference Include="FFMpegCore" Version="5.1.0" />
  </ItemGroup>
</Project>
```

**Understanding FFMpegCore**:
- Version 5.1.0 is stable and well-maintained
- Provides a clean C# API over FFMpeg command-line tool
- Handles video encoding/decoding, frame extraction, metadata reading

**Installation Verification**:
```bash
dotnet restore
dotnet build
```

### AC 1.2: Implement VideoHelper

**File**: `src/Video/VideoHelper.cs` (NEW FILE - create `src/Video` directory)

**Step 1: Create the basic structure**

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using FFMpegCore;
using FFMpegCore.Enums;
using FFMpegCore.Pipes;
using System.Drawing;
using System.Drawing.Imaging;

namespace AiDotNet.Video;

/// <summary>
/// Provides utilities for video frame extraction and assembly.
/// </summary>
/// <remarks>
/// <para>
/// This class abstracts the complexities of video I/O using FFMpeg, providing simple methods
/// to convert between video files and tensor sequences.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "video toolkit" that handles the messy details of video files.
///
/// Video files are complex:
/// - Different formats (MP4, AVI, MOV)
/// - Different codecs (H.264, H.265, VP9)
/// - Compression, metadata, audio tracks
///
/// VideoHelper hides all this complexity, giving you simple methods:
/// - ExtractFrames: Video file → Stream of frame tensors
/// - AssembleVideo: Stream of frame tensors → Video file
///
/// It's like having a video cassette player that can convert tapes to digital frames and back.
/// </para>
/// </remarks>
public static class VideoHelper
{
    /// <summary>
    /// Extracts frames from a video file as a stream of tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements (e.g., float, double).</typeparam>
    /// <param name="videoPath">Path to the input video file.</param>
    /// <returns>An enumerable sequence of frame tensors with shape [height, width, channels].</returns>
    /// <remarks>
    /// <para>
    /// This method uses lazy evaluation (yield return) to stream frames one at a time,
    /// avoiding the need to load the entire video into memory. This is crucial for processing
    /// large videos efficiently.
    /// </para>
    /// <para><b>For Beginners:</b> This method reads a video and gives you frames one at a time.
    ///
    /// Think of it like reading a flip book:
    /// - You don't need to see all pages at once
    /// - You look at one page (frame) at a time
    /// - When you're done with a page, you move to the next
    ///
    /// This saves memory because only one frame is loaded at any time.
    ///
    /// Usage example:
    /// ```csharp
    /// foreach (var frame in VideoHelper.ExtractFrames<float>("video.mp4"))
    /// {
    ///     // Process this frame
    ///     var upscaled = upscaler.Process(frame);
    ///     // Frame is automatically cleaned up after this iteration
    /// }
    /// ```
    /// </para>
    /// </remarks>
    public static IEnumerable<Tensor<T>> ExtractFrames<T>(string videoPath)
    {
        if (!File.Exists(videoPath))
            throw new FileNotFoundException($"Video file not found: {videoPath}");

        // Get video metadata
        var mediaInfo = FFProbe.Analyse(videoPath);
        var videoStream = mediaInfo.VideoStreams.FirstOrDefault()
            ?? throw new InvalidOperationException("No video stream found in file");

        int width = videoStream.Width;
        int height = videoStream.Height;
        int frameCount = (int)(videoStream.Duration.TotalSeconds * videoStream.FrameRate);

        // Create temporary directory for extracted frames
        string tempDir = Path.Combine(Path.GetTempPath(), $"video_frames_{Guid.NewGuid()}");
        Directory.CreateDirectory(tempDir);

        try
        {
            // Extract all frames to temporary directory
            FFMpeg.ExtractFrames(
                videoPath,
                tempDir,
                captureTime: null,
                outputFormat: ImageFormat.Png);

            // Stream frames as tensors
            var frameFiles = Directory.GetFiles(tempDir, "*.png").OrderBy(f => f);

            foreach (var frameFile in frameFiles)
            {
                yield return LoadImageAsTensor<T>(frameFile, width, height);
            }
        }
        finally
        {
            // Cleanup temporary files
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    /// <summary>
    /// Loads an image file as a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="targetWidth">Target width for the tensor.</param>
    /// <param name="targetHeight">Target height for the tensor.</param>
    /// <returns>A tensor with shape [height, width, 3] containing RGB values normalized to [0, 1].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts an image file into a tensor of numbers.
    ///
    /// Process:
    /// 1. Load image from file
    /// 2. Convert to RGB format
    /// 3. Normalize pixel values from [0, 255] to [0.0, 1.0]
    /// 4. Store in tensor format
    ///
    /// Normalization is important because neural networks expect inputs in [0, 1] range.
    /// </para>
    /// </remarks>
    private static Tensor<T> LoadImageAsTensor<T>(string imagePath, int targetWidth, int targetHeight)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        using var bitmap = new Bitmap(imagePath);

        // Ensure correct dimensions
        if (bitmap.Width != targetWidth || bitmap.Height != targetHeight)
        {
            throw new InvalidOperationException(
                $"Image dimensions {bitmap.Width}×{bitmap.Height} don't match expected {targetWidth}×{targetHeight}");
        }

        var tensor = new Tensor<T>(new[] { targetHeight, targetWidth, 3 });

        // Convert bitmap to tensor
        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                Color pixel = bitmap.GetPixel(x, y);

                // Normalize to [0, 1] range
                tensor[y, x, 0] = numOps.FromDouble(pixel.R / 255.0); // Red
                tensor[y, x, 1] = numOps.FromDouble(pixel.G / 255.0); // Green
                tensor[y, x, 2] = numOps.FromDouble(pixel.B / 255.0); // Blue
            }
        }

        return tensor;
    }

    /// <summary>
    /// Assembles a sequence of frame tensors into a video file.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="frames">Sequence of frame tensors with shape [height, width, channels].</param>
    /// <param name="outputPath">Path for the output video file.</param>
    /// <param name="frameRate">Frame rate for the output video (frames per second).</param>
    /// <remarks>
    /// <para>
    /// This method consumes the frame sequence and encodes it into a video file using FFMpeg.
    /// The frames are processed sequentially without loading all frames into memory simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a video file from individual frame images.
    ///
    /// Think of it like making a flip book into an animated GIF:
    /// - You have a stack of images (frames)
    /// - FFMpeg stitches them together with timing (frame rate)
    /// - Result is a video file that plays back smoothly
    ///
    /// Frame rate determines playback speed:
    /// - 24 fps: Standard cinema
    /// - 30 fps: Standard video
    /// - 60 fps: Smooth video (gaming, sports)
    ///
    /// Usage example:
    /// ```csharp
    /// var frames = GetUpscaledFrames();
    /// VideoHelper.AssembleVideo(frames, "output.mp4", frameRate: 30.0);
    /// ```
    /// </para>
    /// </remarks>
    public static void AssembleVideo<T>(
        IEnumerable<Tensor<T>> frames,
        string outputPath,
        double frameRate)
    {
        if (frameRate <= 0)
            throw new ArgumentException("Frame rate must be positive", nameof(frameRate));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Create temporary directory for frame images
        string tempDir = Path.Combine(Path.GetTempPath(), $"video_assembly_{Guid.NewGuid()}");
        Directory.CreateDirectory(tempDir);

        try
        {
            // Convert tensors to image files
            int frameIndex = 0;
            foreach (var frame in frames)
            {
                string framePath = Path.Combine(tempDir, $"frame_{frameIndex:D6}.png");
                SaveTensorAsImage(frame, framePath, numOps);
                frameIndex++;
            }

            if (frameIndex == 0)
                throw new InvalidOperationException("No frames provided to assemble");

            // Encode image sequence as video
            var frameFiles = Directory.GetFiles(tempDir, "*.png").OrderBy(f => f).ToArray();

            FFMpeg.JoinImageSequence(
                outputPath,
                frameRate: frameRate,
                images: frameFiles);

            if (!File.Exists(outputPath))
                throw new InvalidOperationException("Failed to create output video file");
        }
        finally
        {
            // Cleanup temporary files
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    /// <summary>
    /// Saves a tensor as an image file.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="tensor">Tensor with shape [height, width, 3] containing normalized RGB values.</param>
    /// <param name="outputPath">Path for the output image file.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a tensor back into an image file.
    ///
    /// Process:
    /// 1. Create blank image
    /// 2. Denormalize tensor values from [0, 1] to [0, 255]
    /// 3. Set pixel colors in image
    /// 4. Save to file
    ///
    /// This is the reverse of LoadImageAsTensor.
    /// </para>
    /// </remarks>
    private static void SaveTensorAsImage<T>(Tensor<T> tensor, string outputPath, INumericOperations<T> numOps)
    {
        if (tensor.Shape.Length != 3 || tensor.Shape[2] != 3)
            throw new ArgumentException("Tensor must have shape [height, width, 3]", nameof(tensor));

        int height = tensor.Shape[0];
        int width = tensor.Shape[1];

        using var bitmap = new Bitmap(width, height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Denormalize from [0, 1] to [0, 255]
                double r = numOps.ToDouble(tensor[y, x, 0]) * 255.0;
                double g = numOps.ToDouble(tensor[y, x, 1]) * 255.0;
                double b = numOps.ToDouble(tensor[y, x, 2]) * 255.0;

                // Clamp to valid range
                int rByte = Math.Clamp((int)Math.Round(r), 0, 255);
                int gByte = Math.Clamp((int)Math.Round(g), 0, 255);
                int bByte = Math.Clamp((int)Math.Round(b), 0, 255);

                bitmap.SetPixel(x, y, Color.FromArgb(rByte, gByte, bByte));
            }
        }

        bitmap.Save(outputPath, ImageFormat.Png);
    }
}
```

**Understanding the Implementation**:

1. **ExtractFrames**:
   - Uses FFProbe to get video metadata (dimensions, frame rate, duration)
   - Creates temporary directory for extracted frames
   - Uses FFMpeg to extract all frames as PNG images
   - Yields tensors one at a time (lazy evaluation)
   - Cleans up temporary files after use

2. **AssembleVideo**:
   - Consumes frame tensors sequentially
   - Saves each tensor as PNG image in temporary directory
   - Uses FFMpeg to encode image sequence into video
   - Cleans up temporary files after use

3. **Why Temporary Files?**:
   - FFMpeg works with files, not in-memory data
   - Temporary files provide an interface between tensors and FFMpeg
   - Automatic cleanup prevents disk space issues

### AC 1.3: Implement VideoUpscalerModel

**File**: `src/Models/Video/VideoUpscalerModel.cs` (NEW FILE)

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.Video;
using FFMpegCore;

namespace AiDotNet.Models.Video;

/// <summary>
/// Upscales low-resolution videos to higher resolution using a neural network.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class processes video files frame-by-frame through a pre-trained super-resolution
/// ONNX model, increasing the spatial resolution while preserving temporal consistency.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "video enhancer" that makes blurry videos sharper.
///
/// What it does:
/// 1. Reads a low-resolution video (e.g., 480p)
/// 2. Sends each frame through an AI upscaling model
/// 3. The AI infers missing details based on learned patterns
/// 4. Outputs a high-resolution video (e.g., 1080p)
///
/// Unlike simple interpolation (which just repeats pixels), neural upscaling:
/// - Adds realistic details that weren't in the original
/// - Reduces blur and artifacts
/// - Maintains sharpness and clarity
///
/// It's like having an expert photo restorer enhance each frame of your video.
/// </para>
/// </remarks>
public class VideoUpscalerModel<T>
{
    private readonly OnnxModel<T> _upscaler;

    /// <summary>
    /// Initializes a new instance of the VideoUpscalerModel class.
    /// </summary>
    /// <param name="upscalerOnnxPath">Path to the pre-trained video upscaling ONNX model.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor loads the AI model that will upscale your video.
    ///
    /// The ONNX model file contains:
    /// - Neural network architecture
    /// - Pre-trained weights (learned from thousands of low/high resolution image pairs)
    /// - Input/output specifications
    ///
    /// Example models:
    /// - ESRGAN: High-quality photo-realistic upscaling
    /// - Real-ESRGAN: Optimized for real-world images
    /// - BasicVSR: Designed specifically for video (temporal aware)
    ///
    /// You don't need to train the model yourself - use a pre-trained one.
    /// </para>
    /// </remarks>
    public VideoUpscalerModel(string upscalerOnnxPath)
    {
        if (!File.Exists(upscalerOnnxPath))
            throw new FileNotFoundException($"Upscaler model not found: {upscalerOnnxPath}");

        _upscaler = new OnnxModel<T>(upscalerOnnxPath);
    }

    /// <summary>
    /// Upscales a video file to higher resolution.
    /// </summary>
    /// <param name="inputVideoPath">Path to the input video file.</param>
    /// <param name="outputVideoPath">Path for the output upscaled video file.</param>
    /// <remarks>
    /// <para>
    /// This method orchestrates the complete upscaling pipeline:
    /// 1. Extracts metadata (frame rate) from input video
    /// 2. Streams frames one at a time from input
    /// 3. Upscales each frame through the neural network
    /// 4. Assembles upscaled frames into output video
    ///
    /// Memory efficiency is achieved through lazy evaluation - only one frame
    /// is processed at a time, allowing upscaling of videos larger than available RAM.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main method that upscales your entire video.
    ///
    /// Pipeline:
    /// ```
    /// Input Video (low-res)
    ///     ↓ VideoHelper.ExtractFrames
    /// Frame Stream (one at a time)
    ///     ↓ For each frame
    /// [Neural Network Upscaling]
    ///     ↓ Upscaled frame
    /// Frame Stream (high-res)
    ///     ↓ VideoHelper.AssembleVideo
    /// Output Video (high-res)
    /// ```
    ///
    /// Example usage:
    /// ```csharp
    /// var upscaler = new VideoUpscalerModel<float>("esrgan_4x.onnx");
    /// upscaler.Upscale("input_480p.mp4", "output_1080p.mp4");
    /// // Input: 640×480 → Output: 2560×1920 (4x upscaling)
    /// ```
    /// </para>
    /// </remarks>
    public void Upscale(string inputVideoPath, string outputVideoPath)
    {
        if (!File.Exists(inputVideoPath))
            throw new FileNotFoundException($"Input video not found: {inputVideoPath}");

        // Step 1: Get frame rate from input video
        var mediaInfo = FFProbe.Analyse(inputVideoPath);
        var videoStream = mediaInfo.VideoStreams.FirstOrDefault()
            ?? throw new InvalidOperationException("No video stream found in input file");

        double frameRate = videoStream.FrameRate;

        // Step 2: Extract frames as a stream
        IEnumerable<Tensor<T>> inputFrames = VideoHelper.ExtractFrames<T>(inputVideoPath);

        // Step 3: Upscale each frame (lazy evaluation - processed one at a time)
        IEnumerable<Tensor<T>> upscaledFrames = inputFrames.Select(frame => UpscaleFrame(frame));

        // Step 4: Assemble upscaled frames into output video
        VideoHelper.AssembleVideo(upscaledFrames, outputVideoPath, frameRate);
    }

    /// <summary>
    /// Upscales a single frame using the neural network model.
    /// </summary>
    /// <param name="frame">Input frame tensor with shape [height, width, channels].</param>
    /// <returns>Upscaled frame tensor with larger spatial dimensions.</returns>
    /// <remarks>
    /// <para>
    /// This method prepares the frame for the model, runs inference, and extracts the result.
    /// The actual upscaling factor depends on the pre-trained model (typically 2x, 4x, or 8x).
    /// </para>
    /// <para><b>For Beginners:</b> This processes a single frame through the AI model.
    ///
    /// Steps:
    /// 1. Prepare input: Add batch dimension [height, width, 3] → [1, height, width, 3]
    /// 2. Run model: Neural network inference
    /// 3. Extract output: Remove batch dimension [1, new_height, new_width, 3] → [new_height, new_width, 3]
    ///
    /// The model has learned patterns like:
    /// - Edge reconstruction (sharp lines from blurry ones)
    /// - Texture synthesis (realistic details)
    /// - Color refinement (reducing compression artifacts)
    ///
    /// Example:
    /// - Input: 256×256 blurry frame
    /// - Output: 1024×1024 sharp frame (4x upscale)
    /// </para>
    /// </remarks>
    private Tensor<T> UpscaleFrame(Tensor<T> frame)
    {
        // Add batch dimension: [H, W, C] → [1, H, W, C]
        var batchedFrame = AddBatchDimension(frame);

        // Prepare model inputs
        var inputs = new Dictionary<string, Tensor<T>>
        {
            ["input"] = batchedFrame
        };

        // Run inference
        var outputs = _upscaler.Forward(inputs);
        var upscaledBatched = outputs["output"];

        // Remove batch dimension: [1, H', W', C] → [H', W', C]
        return RemoveBatchDimension(upscaledBatched);
    }

    /// <summary>
    /// Adds a batch dimension to a frame tensor.
    /// </summary>
    /// <param name="frame">Frame tensor with shape [height, width, channels].</param>
    /// <returns>Batched tensor with shape [1, height, width, channels].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Most neural networks expect a "batch" dimension as the first axis.
    ///
    /// Think of a batch as a stack of images:
    /// - Single image: [height, width, channels]
    /// - Batch of images: [batch_size, height, width, channels]
    ///
    /// Even when processing one image, we create a batch of size 1: [1, height, width, channels]
    ///
    /// This is like putting a single photo in a photo album - the album (batch) contains one photo.
    /// </para>
    /// </remarks>
    private Tensor<T> AddBatchDimension(Tensor<T> frame)
    {
        int height = frame.Shape[0];
        int width = frame.Shape[1];
        int channels = frame.Shape[2];

        var batched = new Tensor<T>(new[] { 1, height, width, channels });

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int c = 0; c < channels; c++)
                {
                    batched[0, h, w, c] = frame[h, w, c];
                }
            }
        }

        return batched;
    }

    /// <summary>
    /// Removes the batch dimension from a tensor.
    /// </summary>
    /// <param name="batched">Batched tensor with shape [1, height, width, channels].</param>
    /// <returns>Frame tensor with shape [height, width, channels].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This extracts the single image from a batch of size 1.
    ///
    /// After the model processes the batch [1, H, W, C], we extract the single frame [H, W, C].
    ///
    /// This is like taking the photo out of the album after processing.
    /// </para>
    /// </remarks>
    private Tensor<T> RemoveBatchDimension(Tensor<T> batched)
    {
        if (batched.Shape[0] != 1)
            throw new ArgumentException("Expected batch size of 1", nameof(batched));

        int height = batched.Shape[1];
        int width = batched.Shape[2];
        int channels = batched.Shape[3];

        var frame = new Tensor<T>(new[] { height, width, channels });

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int c = 0; c < channels; c++)
                {
                    frame[h, w, c] = batched[0, h, w, c];
                }
            }
        }

        return frame;
    }
}
```

**Understanding the Upscaling Pipeline**:

1. **Memory Efficiency**: Using `IEnumerable` with LINQ's `Select`:
   ```csharp
   IEnumerable<Tensor<T>> upscaledFrames = inputFrames.Select(frame => UpscaleFrame(frame));
   ```
   - No intermediate list created
   - Each frame is upscaled on-demand as AssembleVideo consumes the sequence
   - Only one input frame and one output frame in memory at a time

2. **Frame Rate Preservation**:
   - Extract frame rate from input video metadata
   - Use same frame rate for output video
   - Ensures temporal consistency (video plays at correct speed)

3. **Batch Dimension**:
   - Neural networks expect batch dimension as first axis
   - We add it before inference, remove it after
   - Allows same model to process single frames or batches

---

## Phase 2: Testing and Validation

### AC 2.1: Unit Tests

**File**: `tests/UnitTests/Models/VideoUpscalerModelTests.cs`

```csharp
using Xunit;
using Moq;
using AiDotNet.Models.Video;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Unit tests for VideoUpscalerModel.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These tests verify the upscaler's logic without real video files.
///
/// We use mocks to simulate:
/// - VideoHelper (without actual FFMpeg)
/// - OnnxModel (without actual neural network)
///
/// This makes tests:
/// - Fast (no video encoding/decoding)
/// - Reliable (no external dependencies)
/// - Focused (test logic, not library behavior)
/// </para>
/// </remarks>
public class VideoUpscalerModelTests
{
    [Fact]
    public void Upscale_ProcessesCorrectNumberOfFrames()
    {
        // Arrange
        var mockOnnxModel = new Mock<IOnnxModel<float>>();
        int frameCount = 3;

        // Create dummy input frames
        var inputFrames = Enumerable.Range(0, frameCount)
            .Select(_ => new Tensor<float>(new[] { 64, 64, 3 }))
            .ToList();

        // Setup mock to return upscaled frames (2x larger)
        mockOnnxModel
            .Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
            .Returns((Dictionary<string, Tensor<float>> inputs) =>
            {
                // Return a 2x upscaled frame
                return new Dictionary<string, Tensor<float>>
                {
                    ["output"] = new Tensor<float>(new[] { 1, 128, 128, 3 })
                };
            });

        // Create upscaler with mock
        var upscaler = new VideoUpscalerModel<float>(mockOnnxModel.Object);

        // Act
        // In real implementation, this would call VideoHelper
        // For unit test, we test UpscaleFrame directly
        var upscaledFrames = inputFrames.Select(f => upscaler.UpscaleFrame(f)).ToList();

        // Assert
        mockOnnxModel.Verify(
            m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()),
            Times.Exactly(frameCount),
            "Model should be called once per frame");

        Assert.Equal(frameCount, upscaledFrames.Count);
        Assert.All(upscaledFrames, frame =>
        {
            Assert.Equal(128, frame.Shape[0]); // height doubled
            Assert.Equal(128, frame.Shape[1]); // width doubled
            Assert.Equal(3, frame.Shape[2]);   // channels unchanged
        });
    }

    [Fact]
    public void UpscaleFrame_AddsBatchDimensionCorrectly()
    {
        // Arrange
        var mockOnnxModel = new Mock<IOnnxModel<float>>();
        Dictionary<string, Tensor<float>>? capturedInputs = null;

        mockOnnxModel
            .Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
            .Callback<Dictionary<string, Tensor<float>>>(inputs => capturedInputs = inputs)
            .Returns(new Dictionary<string, Tensor<float>>
            {
                ["output"] = new Tensor<float>(new[] { 1, 128, 128, 3 })
            });

        var upscaler = new VideoUpscalerModel<float>(mockOnnxModel.Object);
        var inputFrame = new Tensor<float>(new[] { 64, 64, 3 });

        // Act
        var upscaled = upscaler.UpscaleFrame(inputFrame);

        // Assert
        Assert.NotNull(capturedInputs);
        var modelInput = capturedInputs["input"];

        Assert.Equal(4, modelInput.Shape.Length);
        Assert.Equal(1, modelInput.Shape[0]);  // batch size = 1
        Assert.Equal(64, modelInput.Shape[1]); // height
        Assert.Equal(64, modelInput.Shape[2]); // width
        Assert.Equal(3, modelInput.Shape[3]);  // channels
    }
}
```

### AC 2.2: Integration Test

**File**: `tests/IntegrationTests/Models/VideoUpscalerIntegrationTests.cs`

```csharp
using Xunit;
using AiDotNet.Models.Video;
using FFMpegCore;

namespace AiDotNet.Tests.IntegrationTests.Models;

/// <summary>
/// Integration tests for VideoUpscalerModel with real video files and ONNX models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Integration tests verify the complete end-to-end pipeline.
///
/// These tests require:
/// - FFMpeg executable installed on system
/// - Pre-trained upscaler ONNX model
/// - Test video file
///
/// They're slower but test real-world scenarios.
/// </para>
/// </remarks>
public class VideoUpscalerIntegrationTests
{
    private const string ModelPath = "test_data/esrgan_2x.onnx";
    private const string TestVideoPath = "test_data/low_res_video.mp4";

    [Fact(Skip = "Requires FFMpeg and ONNX model")]
    public void Upscale_WithRealVideo_IncreasesResolution()
    {
        // Skip if dependencies not available
        if (!File.Exists(ModelPath) || !File.Exists(TestVideoPath))
        {
            return;
        }

        // Arrange
        var upscaler = new VideoUpscalerModel<float>(ModelPath);
        string outputPath = "test_output_upscaled.mp4";

        // Get input video metadata
        var inputInfo = FFProbe.Analyse(TestVideoPath);
        var inputVideo = inputInfo.VideoStreams[0];
        int inputWidth = inputVideo.Width;
        int inputHeight = inputVideo.Height;

        // Act
        upscaler.Upscale(TestVideoPath, outputPath);

        // Assert
        Assert.True(File.Exists(outputPath), "Output video file should be created");

        // Verify output video metadata
        var outputInfo = FFProbe.Analyse(outputPath);
        var outputVideo = outputInfo.VideoStreams[0];

        Assert.True(outputVideo.Width > inputWidth, "Output width should be larger");
        Assert.True(outputVideo.Height > inputHeight, "Output height should be larger");

        // For 2x upscaler, expect dimensions to double
        Assert.Equal(inputWidth * 2, outputVideo.Width);
        Assert.Equal(inputHeight * 2, outputVideo.Height);

        // Verify frame rate is preserved
        Assert.Equal(inputVideo.FrameRate, outputVideo.FrameRate);

        // Manual verification required:
        // 1. Play test_output_upscaled.mp4
        // 2. Compare with test_data/low_res_video.mp4
        // 3. Verify increased sharpness and detail
        // 4. Check for temporal consistency (no flickering)
    }

    [Fact]
    public void VideoHelper_ExtractFrames_ReturnsCorrectCount()
    {
        // This test can run if test video exists (doesn't require ONNX model)
        if (!File.Exists(TestVideoPath))
        {
            return;
        }

        // Arrange
        var info = FFProbe.Analyse(TestVideoPath);
        var expectedFrameCount = (int)(info.VideoStreams[0].Duration.TotalSeconds * info.VideoStreams[0].FrameRate);

        // Act
        var frames = VideoHelper.ExtractFrames<float>(TestVideoPath).ToList();

        // Assert
        Assert.Equal(expectedFrameCount, frames.Count);
        Assert.All(frames, frame =>
        {
            Assert.Equal(3, frame.Shape.Length); // [H, W, C]
            Assert.Equal(3, frame.Shape[2]);      // RGB channels
        });
    }
}
```

**Creating Test Data**:

To run integration tests, you'll need:

1. **Test Video** (`test_data/low_res_video.mp4`):
   ```bash
   # Create a 1-second 128×128 test video using FFMpeg
   ffmpeg -f lavfi -i testsrc=duration=1:size=128x128:rate=15 test_data/low_res_video.mp4
   ```

2. **Upscaler Model** (`test_data/esrgan_2x.onnx`):
   - Download pre-trained ESRGAN or Real-ESRGAN model
   - Convert to ONNX format if needed
   - Place in `test_data/` directory

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Loading Entire Video into Memory

**Wrong Approach**:
```csharp
// BAD: Loads all frames at once
var allFrames = VideoHelper.ExtractFrames<float>(videoPath).ToList();
foreach (var frame in allFrames)
{
    ProcessFrame(frame);
}
```

**Problem**: Uses memory proportional to video length (10GB+ for long videos)

**Correct Approach**:
```csharp
// GOOD: Streams frames one at a time
foreach (var frame in VideoHelper.ExtractFrames<float>(videoPath))
{
    ProcessFrame(frame);
    // Previous frames are garbage collected
}
```

### Pitfall 2: Forgetting to Preserve Frame Rate

**Wrong Approach**:
```csharp
// Uses default frame rate (may not match input)
VideoHelper.AssembleVideo(frames, "output.mp4", frameRate: 30.0);
```

**Problem**: If input was 24fps, output at 30fps will play faster

**Correct Approach**:
```csharp
// Extract frame rate from input, use for output
var info = FFProbe.Analyse(inputPath);
double frameRate = info.VideoStreams[0].FrameRate;
VideoHelper.AssembleVideo(frames, outputPath, frameRate);
```

### Pitfall 3: Incorrect Tensor Normalization

**Wrong Approach**:
```csharp
// Pixel values in [0, 255] range
tensor[y, x, c] = numOps.FromDouble(pixel.R); // 0-255
```

**Problem**: Neural networks expect normalized inputs [0, 1] or [-1, 1]

**Correct Approach**:
```csharp
// Normalize to [0, 1]
tensor[y, x, c] = numOps.FromDouble(pixel.R / 255.0); // 0.0-1.0
```

### Pitfall 4: Not Handling Batch Dimensions

**Wrong Approach**:
```csharp
// Passing [H, W, C] directly to model
var output = model.Forward(new Dictionary<string, Tensor<T>>
{
    ["input"] = frame  // Wrong shape!
});
```

**Problem**: Model expects [1, H, W, C] (batch dimension)

**Correct Approach**:
```csharp
// Add batch dimension: [H, W, C] → [1, H, W, C]
var batched = AddBatchDimension(frame);
var output = model.Forward(new Dictionary<string, Tensor<T>>
{
    ["input"] = batched  // Correct shape!
});
// Remove batch dimension from output: [1, H', W', C] → [H', W', C]
var result = RemoveBatchDimension(output["output"]);
```

---

## Testing Checklist

Before submitting your PR:

- [ ] Unit tests pass with mocked components
- [ ] Unit tests verify frame count and shapes
- [ ] Integration test creates output video file
- [ ] Integration test verifies resolution increase
- [ ] Frame rate is preserved in output
- [ ] Manual verification shows improved quality
- [ ] Code coverage >= 90%
- [ ] Memory usage is reasonable (streaming works)
- [ ] XML documentation for all public methods
- [ ] No hardcoded numeric types (use generic T)
- [ ] Proper use of INumericOperations<T>

---

## Future Extensions (Not in This Issue)

This implementation focuses on **video upscaling**. Future issues will add:

1. **Frame Interpolation**: Generate intermediate frames for slow-motion
   ```
   Input: [Frame 0, Frame 2, Frame 4] (3 frames)
   Output: [Frame 0, Frame 1, Frame 2, Frame 3, Frame 4] (5 frames, smoother)
   ```

2. **Optical Flow**: Estimate motion between frames
   ```
   Used for: Motion compensation, temporal stabilization, frame interpolation
   ```

3. **Temporal Denoising**: Remove noise while preserving motion
4. **Video Stabilization**: Reduce camera shake
5. **Frame Rate Conversion**: Convert between different frame rates (24fps → 60fps)

---

## Further Learning Resources

### Papers
1. **"Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" (2021)**: Practical video/image upscaling
2. **"BasicVSR: The Search for Essential Components in Video Super-Resolution" (2021)**: Video-specific super-resolution
3. **"RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation" (2022)**: Frame interpolation

### Concepts to Explore
- **Temporal Super-Resolution**: Using adjacent frames to improve upscaling quality
- **Optical Flow Networks**: Learning motion estimation end-to-end
- **Recurrent Video Enhancement**: Using RNNs to propagate information across frames
- **Video Codec Artifacts**: Understanding and removing compression artifacts

### Example Pre-trained Models
- **Real-ESRGAN**: General-purpose upscaling (ONNX available)
- **BasicVSR++**: State-of-the-art video super-resolution
- **RIFE**: Real-time frame interpolation
- **TecoGAN**: Temporally coherent video upscaling

---

## Summary

You've implemented a **VideoUpscalerModel** that:
1. Reads video files using FFMpeg
2. Streams frames one at a time (memory-efficient)
3. Upscales each frame through a neural network
4. Assembles frames back into a high-resolution video
5. Preserves temporal properties (frame rate)

This is the foundation for video enhancement, with applications in content restoration, streaming optimization, and quality improvement. Excellent work!
