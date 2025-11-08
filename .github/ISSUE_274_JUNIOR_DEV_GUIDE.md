# Issue #274: Junior Developer Implementation Guide - Segment Anything (SAM) Integration

## Understanding Segment Anything Model (SAM)

### What is SAM?

**SAM** (Segment Anything Model) is Meta's foundation model for image segmentation that can segment **any** object in **any** image with simple prompts.

**For Beginners:** Think of SAM as a magical "intelligent scissors" tool:
- Show it an image
- Click where you want to select (or draw a box)
- It automatically finds and outlines the object perfectly
- Works on ANY type of object, even things it's never seen before!

**Real-world analogy:** Imagine using Photoshop's magic wand tool, but it:
- Works perfectly every time
- Understands object boundaries
- Can segment anything (people, animals, objects, parts of objects)
- Requires just a single click

### Key Innovation: Promptable Segmentation

Traditional segmentation models require:
- ❌ Training on specific object classes (cars, people, etc.)
- ❌ Large labeled datasets
- ❌ Retraining for new object types

SAM's breakthrough:
- ✅ **Zero-shot:** Segments objects it's never seen before
- ✅ **Prompt-based:** Click, box, or text prompt
- ✅ **Interactive:** Real-time feedback and refinement
- ✅ **Foundation model:** Works across all domains

### SAM's Two-Stage Architecture

```
┌────────────────┐                    ┌─────────────────┐
│  Image Encoder │                    │  Mask Decoder   │
│  (ViT-Huge)    │  ─── (cache) ───>  │  (Lightweight)  │
└────────────────┘                    └─────────────────┘
        ↓                                     ↑
    [SLOW - 1x]                           [FAST - 30ms]
    Compute once                          Run many times
                                              ↑
                                         Prompts:
                                         • Points
                                         • Boxes
                                         • Masks
```

**Two-stage design rationale:**

1. **Image Encoder (Heavy)**: Runs once per image
   - Vision Transformer (ViT-H): 632M parameters
   - Input: 1024×1024 image
   - Output: Rich feature embeddings (256×64×64)
   - Takes: ~200ms on GPU
   - **Cache this!** Don't recompute for each prompt

2. **Mask Decoder (Lightweight)**: Runs many times
   - Small transformer: ~4M parameters
   - Input: Image embeddings + prompt
   - Output: Segmentation masks
   - Takes: ~30ms on GPU
   - **Interactive speed** for real-time editing

**Why this design?**
- User can click multiple times to refine segmentation
- Image encoding is expensive, only do it once
- Mask decoding is cheap, do it many times
- Enables interactive workflows

### Types of Prompts

SAM supports three prompt types:

#### 1. Point Prompts (Most Common)

```csharp
// Single positive point: "Segment the object at this location"
var point = new Point(x: 150, y: 200);
var mask = sam.Predict(points: new[] { point }, bboxes: null);

// Multiple points: Refine segmentation
var points = new[]
{
    new Point(150, 200, label: 1),  // Positive: "Include this"
    new Point(180, 220, label: 1),  // Positive: "Include this too"
    new Point(300, 100, label: 0)   // Negative: "Don't include this"
};
```

**Point label meanings:**
- `label = 1`: **Foreground** (include this in the mask)
- `label = 0`: **Background** (exclude this from the mask)

**Use cases:**
- Quick object selection: One click segments the object
- Refinement: Add positive/negative points to improve mask
- Interactive editing: User clicks to guide segmentation

#### 2. Bounding Box Prompts

```csharp
// Box prompt: "Segment everything in this rectangle"
var bbox = new Rectangle(x: 100, y: 150, width: 200, height: 150);
var mask = sam.Predict(points: null, bboxes: new[] { bbox });
```

**Use cases:**
- Object detection integration: Use detected boxes as prompts
- Coarse-to-fine: Box gives rough region, SAM refines boundary
- Batch processing: Segment all detected objects

#### 3. Combining Prompts

```csharp
// Most powerful: Box + points for precise control
var bbox = new Rectangle(100, 150, 200, 150);  // Rough region
var point = new Point(150, 200, label: 1);     // Focus point inside box
var mask = sam.Predict(points: new[] { point }, bboxes: new[] { bbox });
```

### SAM Output: Multiple Mask Candidates

SAM generates **multiple** mask predictions with **confidence scores**:

```csharp
var output = sam.Predict(points, bboxes);
// output.Masks: List of 3 masks (typically)
// output.Scores: Confidence for each mask

// Mask 1: Whole object (score: 0.95)
// Mask 2: Object part (score: 0.87)
// Mask 3: Larger region including background (score: 0.72)
```

**Why multiple masks?**
- Different granularities: Whole object vs. parts
- Ambiguity resolution: User picks best mask
- Flexibility: Choose mask based on task

**Predicted IoU score:** How confident SAM is about each mask
- Higher score = better mask quality
- Typically use the highest-scoring mask

---

## Existing Infrastructure to Use

### 1. Vision Transformer for Image Encoding

**File:** `src/NeuralNetworks/VisionTransformer.cs`

SAM uses a Vision Transformer (ViT-Huge) as image encoder:
- Input: 1024×1024 RGB image
- Architecture: ViT with 32 transformer blocks
- Output: Feature embeddings (256 channels × 64×64 spatial)

We can leverage existing ViT infrastructure:
- Patch embedding
- Multi-head attention
- Position embeddings

### 2. Attention Mechanisms

**File:** `src/NeuralNetworks/Layers/MultiHeadAttentionLayer.cs`

SAM's mask decoder uses:
- **Self-attention:** Process prompt embeddings
- **Cross-attention:** Attend to image features
- **Transformer decoder blocks**

Our existing `MultiHeadAttentionLayer<T>` provides the foundation.

### 3. Image Preprocessing

**File:** `.github/ISSUE_330_JUNIOR_DEV_GUIDE.md`

SAM requires specific preprocessing:
- Resize to 1024×1024 (maintaining aspect ratio)
- Normalize with ImageNet statistics
- Pad to square if needed

Can reuse existing `ImageProcessor.Resize()` and normalization.

### 4. Tensor Operations

**Files:**
- `src/LinearAlgebra/Tensor.cs` - Multi-dimensional arrays
- `src/LinearAlgebra/Vector.cs` - 1D vectors
- `src/LinearAlgebra/Matrix.cs` - 2D matrices

Needed for:
- Image embeddings (3D tensor: channels × height × width)
- Masks (2D tensor: height × width)
- Batch operations

---

## Phase 1: Implementation Plan

### AC 1.1: Define SamOutput Data Structure (2 points)

**Step 1: Create SamOutput.cs**

```csharp
// File: src/Models/ComputerVision/Segmentation/SamOutput.cs
using AiDotNet.LinearAlgebra;
using System.Collections.Generic;

namespace AiDotNet.Models.ComputerVision.Segmentation
{
    /// <summary>
    /// Represents the output from Segment Anything Model (SAM) prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SamOutput contains the segmentation masks and their quality scores.
    ///
    /// SAM typically generates 3 masks per prompt:
    /// 1. Whole object mask (most confident)
    /// 2. Part-level mask (medium confidence)
    /// 3. Broader mask including context (lower confidence)
    ///
    /// The scores tell you how confident SAM is about each mask.
    /// Usually, you'll want to use the mask with the highest score.
    ///
    /// Example output:
    /// - Masks[0]: Dog body (256×256 binary mask)
    /// - Scores[0]: 0.95 (very confident)
    /// - Masks[1]: Dog head only (256×256)
    /// - Scores[1]: 0.87 (fairly confident)
    /// - Masks[2]: Dog + background (256×256)
    /// - Scores[2]: 0.72 (less confident)
    /// </para>
    /// </remarks>
    public class SamOutput
    {
        /// <summary>
        /// Gets the list of predicted segmentation masks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Each mask is a 2D tensor of bytes where:
        /// - 0 = background (not part of object)
        /// - 255 = foreground (part of object)
        ///
        /// Masks are binary (only 0 or 255), no intermediate values.
        /// This makes them easy to use with image editing tools.
        ///
        /// Typical shape: [height, width]
        /// Example: [256, 256] for 256×256 pixel mask
        /// </para>
        /// </remarks>
        public List<Tensor<byte>> Masks { get; }

        /// <summary>
        /// Gets the confidence scores for each mask (predicted IoU).
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Scores represent SAM's confidence in each mask.
        ///
        /// What is IoU? Intersection over Union - measures mask quality:
        /// - 1.0 = perfect mask
        /// - 0.9+ = very good mask
        /// - 0.7-0.9 = acceptable mask
        /// - &lt;0.7 = questionable mask
        ///
        /// The scores are SAM's prediction of how good each mask is.
        /// Higher score = SAM thinks this mask better matches the object.
        ///
        /// Example:
        /// - Scores[0]: 0.95 ← Use this mask (highest confidence)
        /// - Scores[1]: 0.87
        /// - Scores[2]: 0.72
        /// </para>
        /// </remarks>
        public List<double> Scores { get; }

        /// <summary>
        /// Creates a new SAM output with the specified masks and scores.
        /// </summary>
        /// <param name="masks">The predicted segmentation masks.</param>
        /// <param name="scores">The confidence scores for each mask.</param>
        /// <exception cref="ArgumentNullException">Thrown when masks or scores is null.</exception>
        /// <exception cref="ArgumentException">Thrown when masks and scores have different counts.</exception>
        public SamOutput(List<Tensor<byte>> masks, List<double> scores)
        {
            if (masks == null)
                throw new ArgumentNullException(nameof(masks));
            if (scores == null)
                throw new ArgumentNullException(nameof(scores));
            if (masks.Count != scores.Count)
                throw new ArgumentException($"Number of masks ({masks.Count}) must match number of scores ({scores.Count})");

            Masks = masks;
            Scores = scores;
        }

        /// <summary>
        /// Gets the mask with the highest confidence score.
        /// </summary>
        /// <returns>The best mask and its score.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is a convenience method to get the "best" mask.
        ///
        /// In most cases, you just want the highest-scoring mask.
        /// This method finds it for you automatically.
        ///
        /// Returns a tuple: (mask, score)
        /// - mask: The binary segmentation mask
        /// - score: SAM's confidence in this mask
        /// </para>
        /// </remarks>
        public (Tensor<byte> Mask, double Score) GetBestMask()
        {
            if (Masks.Count == 0)
                throw new InvalidOperationException("No masks available");

            int bestIndex = 0;
            double bestScore = Scores[0];

            for (int i = 1; i < Scores.Count; i++)
            {
                if (Scores[i] > bestScore)
                {
                    bestScore = Scores[i];
                    bestIndex = i;
                }
            }

            return (Masks[bestIndex], bestScore);
        }
    }
}
```

**Step 2: Create unit tests**

```csharp
// File: tests/UnitTests/Models/ComputerVision/Segmentation/SamOutputTests.cs
using Xunit;
using AiDotNet.Models.ComputerVision.Segmentation;
using AiDotNet.LinearAlgebra;
using System.Collections.Generic;

namespace AiDotNet.Tests.UnitTests.Models.ComputerVision.Segmentation
{
    public class SamOutputTests
    {
        [Fact]
        public void Constructor_ValidInputs_CreatesInstance()
        {
            // Arrange
            var masks = new List<Tensor<byte>>
            {
                new Tensor<byte>(new[] { 256, 256 }),
                new Tensor<byte>(new[] { 256, 256 })
            };
            var scores = new List<double> { 0.9, 0.8 };

            // Act
            var output = new SamOutput(masks, scores);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(2, output.Masks.Count);
            Assert.Equal(2, output.Scores.Count);
        }

        [Fact]
        public void Constructor_NullMasks_ThrowsArgumentNullException()
        {
            // Arrange
            var scores = new List<double> { 0.9 };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SamOutput(null, scores));
        }

        [Fact]
        public void Constructor_MismatchedCounts_ThrowsArgumentException()
        {
            // Arrange
            var masks = new List<Tensor<byte>>
            {
                new Tensor<byte>(new[] { 256, 256 }),
                new Tensor<byte>(new[] { 256, 256 })
            };
            var scores = new List<double> { 0.9 }; // Only one score

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new SamOutput(masks, scores));
        }

        [Fact]
        public void GetBestMask_ReturnsHighestScoringMask()
        {
            // Arrange
            var mask1 = new Tensor<byte>(new[] { 256, 256 });
            var mask2 = new Tensor<byte>(new[] { 256, 256 });
            var mask3 = new Tensor<byte>(new[] { 256, 256 });

            // Fill masks with different patterns for identification
            mask1[0, 0] = 1;
            mask2[0, 0] = 2;
            mask3[0, 0] = 3;

            var masks = new List<Tensor<byte>> { mask1, mask2, mask3 };
            var scores = new List<double> { 0.7, 0.95, 0.8 }; // mask2 has highest score

            var output = new SamOutput(masks, scores);

            // Act
            var (bestMask, bestScore) = output.GetBestMask();

            // Assert
            Assert.Equal(0.95, bestScore);
            Assert.Equal(2, bestMask[0, 0]); // Should be mask2
        }

        [Fact]
        public void GetBestMask_EmptyMasks_ThrowsInvalidOperationException()
        {
            // Arrange
            var output = new SamOutput(new List<Tensor<byte>>(), new List<double>());

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => output.GetBestMask());
        }
    }
}
```

---

### AC 1.2: Implement the SamModel Wrapper (13 points)

**Step 1: Create SamModel.cs**

```csharp
// File: src/Models/ComputerVision/Segmentation/SamModel.cs
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Drawing; // For Point, Rectangle

namespace AiDotNet.Models.ComputerVision.Segmentation
{
    /// <summary>
    /// Implements the Segment Anything Model (SAM) for promptable segmentation.
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAM is like an intelligent "magic wand" tool that can segment
    /// any object in any image with just a click or a box.
    ///
    /// What makes SAM special:
    /// - **Zero-shot:** Segments objects it's never seen before
    /// - **Interactive:** Real-time feedback with clicks
    /// - **Versatile:** Works on any image, any object
    /// - **Accurate:** High-quality segmentation masks
    ///
    /// How to use SAM:
    /// 1. Load your image once: `sam.SetImage(image)`
    /// 2. Click to segment: `sam.Predict(points, boxes)`
    /// 3. Refine if needed: Add more points, call Predict again
    /// 4. Get your mask: Use the highest-scoring mask
    ///
    /// Common workflows:
    /// - **Quick selection:** One click → get mask
    /// - **Refinement:** Click positive points (include) and negative points (exclude)
    /// - **Batch processing:** Use bounding boxes from object detector
    ///
    /// Example use cases:
    /// - Photo editing: Remove backgrounds
    /// - Medical imaging: Segment organs/tumors
    /// - Autonomous driving: Segment vehicles/pedestrians
    /// - Content creation: Extract objects for compositing
    /// </para>
    /// </remarks>
    public class SamModel<T>
    {
        private readonly IOnnxModel<T> _imageEncoder;
        private readonly IOnnxModel<T> _maskDecoder;

        private Tensor<T>? _imageEmbedding;
        private int _originalImageHeight;
        private int _originalImageWidth;

        protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        private const int ImageEncoderInputSize = 1024; // SAM uses 1024×1024 images
        private const int MaskOutputSize = 256; // SAM outputs 256×256 masks

        /// <summary>
        /// Creates a new SAM model wrapper with pre-trained ONNX components.
        /// </summary>
        /// <param name="imageEncoderPath">Path to the image encoder ONNX model (typically ViT-H).</param>
        /// <param name="maskDecoderPath">Path to the mask decoder ONNX model.</param>
        /// <exception cref="ArgumentException">Thrown when paths are invalid.</exception>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> SAM has two parts:
        ///
        /// 1. **Image Encoder** (Heavy): ~632M parameters
        ///    - Processes the image ONCE to extract features
        ///    - Takes ~200ms on GPU
        ///    - Cache the output! (That's what SetImage does)
        ///
        /// 2. **Mask Decoder** (Light): ~4M parameters
        ///    - Takes prompts (clicks/boxes) and image features
        ///    - Generates segmentation masks
        ///    - Takes ~30ms on GPU
        ///    - Run this repeatedly for interactive segmentation
        ///
        /// This two-stage design enables real-time interaction:
        /// - User clicks → Instant mask (30ms)
        /// - User clicks again → Another instant mask (30ms)
        /// - No need to reprocess the image each time!
        /// </para>
        /// </remarks>
        public SamModel(string imageEncoderPath, string maskDecoderPath)
        {
            if (string.IsNullOrWhiteSpace(imageEncoderPath))
                throw new ArgumentException("Image encoder path cannot be empty", nameof(imageEncoderPath));
            if (string.IsNullOrWhiteSpace(maskDecoderPath))
                throw new ArgumentException("Mask decoder path cannot be empty", nameof(maskDecoderPath));

            // Note: Replace with actual OnnxModel<T> when issue #280 is implemented
            _imageEncoder = LoadOnnxModel(imageEncoderPath);
            _maskDecoder = LoadOnnxModel(maskDecoderPath);
        }

        private IOnnxModel<T> LoadOnnxModel(string path)
        {
            // TODO: Replace with actual OnnxModel<T> instantiation when #280 is complete
            throw new NotImplementedException("OnnxModel wrapper from issue #280 required");
        }

        /// <summary>
        /// Sets the image to be segmented and computes its embedding (expensive operation).
        /// </summary>
        /// <param name="image">The input image tensor with shape [channels, height, width] or [batch, channels, height, width].</param>
        /// <exception cref="ArgumentNullException">Thrown when image is null.</exception>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method prepares the image for segmentation.
        ///
        /// What happens:
        /// 1. **Preprocessing:** Resize to 1024×1024, normalize
        /// 2. **Encoding:** Run through Vision Transformer (SLOW - ~200ms)
        /// 3. **Caching:** Store the features for later use
        ///
        /// **IMPORTANT:** Call this ONCE per image, not for every prompt!
        ///
        /// Workflow:
        /// ```csharp
        /// sam.SetImage(myImage);           // Do once (slow)
        /// var mask1 = sam.Predict(...);    // Fast (~30ms)
        /// var mask2 = sam.Predict(...);    // Fast (~30ms)
        /// var mask3 = sam.Predict(...);    // Fast (~30ms)
        /// ```
        ///
        /// Why cache? The image features are expensive to compute but constant.
        /// No matter where you click, the image features are the same.
        /// Only the prompt changes, and processing prompts is fast.
        /// </para>
        /// </remarks>
        public void SetImage(Tensor<T> image)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            // Store original dimensions for mask resizing later
            _originalImageHeight = image.Shape[^2]; // Second-to-last dimension
            _originalImageWidth = image.Shape[^1];  // Last dimension

            // Preprocess: Resize and normalize for SAM
            var preprocessed = PreprocessImage(image);

            // Encode image to features (EXPENSIVE - run once)
            _imageEmbedding = _imageEncoder.Forward(preprocessed);
        }

        /// <summary>
        /// Preprocesses image for SAM image encoder.
        /// </summary>
        /// <param name="image">Raw input image.</param>
        /// <returns>Preprocessed image ready for encoding.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> SAM needs images in a specific format:
        ///
        /// 1. **Resize to 1024×1024:**
        ///    - SAM's image encoder expects exactly 1024×1024
        ///    - Maintain aspect ratio by padding with zeros
        ///    - Remember original size for resizing masks back
        ///
        /// 2. **Normalize with ImageNet stats:**
        ///    - Mean: [0.485, 0.456, 0.406] (RGB)
        ///    - Std: [0.229, 0.224, 0.225] (RGB)
        ///    - Formula: (pixel - mean) / std
        ///
        /// 3. **Format as batch:**
        ///    - Add batch dimension: [1, 3, 1024, 1024]
        ///
        /// Why these specific values? They're standard ImageNet preprocessing.
        /// SAM's vision encoder was trained with these, so we must use them too.
        /// </para>
        /// </remarks>
        private Tensor<T> PreprocessImage(Tensor<T> image)
        {
            // Resize to 1024×1024 (SAM's expected size)
            var resized = ImageProcessor.Resize(image, ImageEncoderInputSize, ImageEncoderInputSize);

            // Normalize with ImageNet statistics
            var meanR = NumOps.FromDouble(0.485);
            var meanG = NumOps.FromDouble(0.456);
            var meanB = NumOps.FromDouble(0.406);
            var stdR = NumOps.FromDouble(0.229);
            var stdG = NumOps.FromDouble(0.224);
            var stdB = NumOps.FromDouble(0.225);

            var normalized = new Tensor<T>(new[] { 1, 3, ImageEncoderInputSize, ImageEncoderInputSize });

            for (int h = 0; h < ImageEncoderInputSize; h++)
            {
                for (int w = 0; w < ImageEncoderInputSize; w++)
                {
                    // Normalize each channel
                    var r = resized[h, w, 0];
                    var g = resized[h, w, 1];
                    var b = resized[h, w, 2];

                    normalized[0, 0, h, w] = NumOps.Divide(NumOps.Subtract(r, meanR), stdR);
                    normalized[0, 1, h, w] = NumOps.Divide(NumOps.Subtract(g, meanG), stdG);
                    normalized[0, 2, h, w] = NumOps.Divide(NumOps.Subtract(b, meanB), stdB);
                }
            }

            return normalized;
        }

        /// <summary>
        /// Predicts segmentation masks based on point and/or box prompts.
        /// </summary>
        /// <param name="points">List of point prompts (can be null if using boxes).</param>
        /// <param name="bboxes">List of bounding box prompts (can be null if using points).</param>
        /// <returns>SamOutput containing predicted masks and their confidence scores.</returns>
        /// <exception cref="InvalidOperationException">Thrown when SetImage hasn't been called first.</exception>
        /// <exception cref="ArgumentException">Thrown when both points and bboxes are null.</exception>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is where the magic happens! Give SAM some hints
        /// about what to segment, and it returns high-quality masks.
        ///
        /// **Prompt types:**
        ///
        /// 1. **Point prompts:** Click where you want to segment
        ///    ```csharp
        ///    var point = new Point(x: 150, y: 200);
        ///    var output = sam.Predict(new[] { point }, null);
        ///    ```
        ///
        /// 2. **Box prompts:** Draw a rectangle around the object
        ///    ```csharp
        ///    var box = new Rectangle(x: 100, y: 150, width: 200, height: 150);
        ///    var output = sam.Predict(null, new[] { box });
        ///    ```
        ///
        /// 3. **Combined:** Box + points for precision
        ///    ```csharp
        ///    var box = new Rectangle(100, 150, 200, 150);
        ///    var point = new Point(150, 200);
        ///    var output = sam.Predict(new[] { point }, new[] { box });
        ///    ```
        ///
        /// **Refinement workflow:**
        /// ```csharp
        /// // First attempt
        /// var mask1 = sam.Predict(new[] { new Point(150, 200) }, null);
        ///
        /// // Not quite right? Add a negative point
        /// var mask2 = sam.Predict(new[]
        /// {
        ///     new Point(150, 200, label: 1),  // Positive: include
        ///     new Point(300, 100, label: 0)   // Negative: exclude
        /// }, null);
        /// ```
        ///
        /// **Output:**
        /// - Typically 3 masks at different granularities
        /// - Each with a confidence score (predicted IoU)
        /// - Usually use the highest-scoring mask
        ///
        /// **Performance:** ~30ms on GPU (fast enough for real-time interaction)
        /// </para>
        /// </remarks>
        public SamOutput Predict(List<Point>? points, List<Rectangle>? bboxes)
        {
            // Validate that image has been set
            if (_imageEmbedding == null)
                throw new InvalidOperationException("Must call SetImage() before Predict()");

            // Must have at least one type of prompt
            if ((points == null || points.Count == 0) && (bboxes == null || bboxes.Count == 0))
                throw new ArgumentException("Must provide at least one point or bounding box prompt");

            // Encode prompts into tensors
            var promptEmbeddings = EncodePrompts(points, bboxes);

            // Run mask decoder (FAST - ~30ms)
            var decoderOutput = _maskDecoder.Forward(_imageEmbedding, promptEmbeddings);

            // Parse output into masks and scores
            var samOutput = ParseDecoderOutput(decoderOutput);

            return samOutput;
        }

        /// <summary>
        /// Encodes point and box prompts into tensors for the mask decoder.
        /// </summary>
        /// <param name="points">Point prompts.</param>
        /// <param name="bboxes">Bounding box prompts.</param>
        /// <returns>Encoded prompt tensor.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> SAM's mask decoder needs prompts in a specific format.
        ///
        /// **Point encoding:**
        /// - Coordinates (x, y): Scaled to [0, 1] range
        /// - Labels: 1 = foreground, 0 = background
        /// - Format: [num_points, 3] where columns are [x, y, label]
        ///
        /// **Box encoding:**
        /// - Corners: Top-left (x1, y1) and bottom-right (x2, y2)
        /// - Scaled to [0, 1] range
        /// - Format: [num_boxes, 4] where columns are [x1, y1, x2, y2]
        ///
        /// **Coordinate scaling:** Why [0, 1]?
        /// - Makes prompts independent of image size
        /// - Point at (150, 200) in 300×400 image → (0.5, 0.5) normalized
        /// - Point at (512, 512) in 1024×1024 image → (0.5, 0.5) normalized
        /// - Same semantic location!
        ///
        /// **Combining prompts:**
        /// If both points and boxes are provided, concatenate them:
        /// - [point_embeddings; box_embeddings]
        /// - Mask decoder processes all prompts together
        /// </para>
        /// </remarks>
        private Tensor<T> EncodePrompts(List<Point>? points, List<Rectangle>? bboxes)
        {
            var encodedPrompts = new List<Tensor<T>>();

            // Encode points
            if (points != null && points.Count > 0)
            {
                var pointTensor = new Tensor<T>(new[] { points.Count, 3 }); // [x, y, label]

                for (int i = 0; i < points.Count; i++)
                {
                    // Normalize coordinates to [0, 1] range
                    T x = NumOps.Divide(
                        NumOps.FromDouble(points[i].X),
                        NumOps.FromDouble(_originalImageWidth));
                    T y = NumOps.Divide(
                        NumOps.FromDouble(points[i].Y),
                        NumOps.FromDouble(_originalImageHeight));

                    // Point label: 1 = foreground, 0 = background
                    // Assuming all points are foreground by default
                    T label = NumOps.One;

                    pointTensor[i, 0] = x;
                    pointTensor[i, 1] = y;
                    pointTensor[i, 2] = label;
                }

                encodedPrompts.Add(pointTensor);
            }

            // Encode bounding boxes
            if (bboxes != null && bboxes.Count > 0)
            {
                var boxTensor = new Tensor<T>(new[] { bboxes.Count, 4 }); // [x1, y1, x2, y2]

                for (int i = 0; i < bboxes.Count; i++)
                {
                    // Normalize box coordinates to [0, 1] range
                    T x1 = NumOps.Divide(
                        NumOps.FromDouble(bboxes[i].X),
                        NumOps.FromDouble(_originalImageWidth));
                    T y1 = NumOps.Divide(
                        NumOps.FromDouble(bboxes[i].Y),
                        NumOps.FromDouble(_originalImageHeight));
                    T x2 = NumOps.Divide(
                        NumOps.FromDouble(bboxes[i].X + bboxes[i].Width),
                        NumOps.FromDouble(_originalImageWidth));
                    T y2 = NumOps.Divide(
                        NumOps.FromDouble(bboxes[i].Y + bboxes[i].Height),
                        NumOps.FromDouble(_originalImageHeight));

                    boxTensor[i, 0] = x1;
                    boxTensor[i, 1] = y1;
                    boxTensor[i, 2] = x2;
                    boxTensor[i, 3] = y2;
                }

                encodedPrompts.Add(boxTensor);
            }

            // Concatenate all prompts
            if (encodedPrompts.Count == 1)
                return encodedPrompts[0];
            else
                return ConcatenateTensors(encodedPrompts);
        }

        /// <summary>
        /// Concatenates multiple tensors along the first dimension.
        /// </summary>
        private Tensor<T> ConcatenateTensors(List<Tensor<T>> tensors)
        {
            // Calculate total size
            int totalRows = 0;
            int cols = tensors[0].Shape[1];

            foreach (var tensor in tensors)
            {
                totalRows += tensor.Shape[0];
            }

            // Create concatenated tensor
            var result = new Tensor<T>(new[] { totalRows, cols });
            int currentRow = 0;

            foreach (var tensor in tensors)
            {
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        result[currentRow + i, j] = tensor[i, j];
                    }
                }
                currentRow += tensor.Shape[0];
            }

            return result;
        }

        /// <summary>
        /// Parses the mask decoder output into masks and scores.
        /// </summary>
        /// <param name="decoderOutput">Raw output from mask decoder.</param>
        /// <returns>Structured SamOutput with masks and scores.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> The mask decoder returns raw tensors that need parsing.
        ///
        /// **Decoder output format:**
        /// - Masks: [batch, num_masks, height, width] - Float values
        /// - Scores: [batch, num_masks] - Predicted IoU scores
        ///
        /// **Processing steps:**
        /// 1. **Extract masks:** Get [num_masks, height, width] slice
        /// 2. **Threshold:** Convert floats to binary (0 or 255)
        ///    - If mask[i, j] > 0.0 → 255 (foreground)
        ///    - If mask[i, j] <= 0.0 → 0 (background)
        /// 3. **Resize:** From 256×256 to original image size
        /// 4. **Extract scores:** Get predicted IoU for each mask
        ///
        /// **Why threshold at 0.0?**
        /// - Decoder outputs logits (raw scores)
        /// - Positive = foreground, negative = background
        /// - Threshold at 0 creates binary mask
        ///
        /// **Why resize masks?**
        /// - Decoder outputs 256×256 for efficiency
        /// - User needs masks at original image resolution
        /// - Use nearest-neighbor interpolation (preserve binary values)
        /// </para>
        /// </remarks>
        private SamOutput ParseDecoderOutput(Tensor<T> decoderOutput)
        {
            // Typically decoder outputs:
            // - Masks: [1, num_masks, 256, 256]
            // - Scores: [1, num_masks]

            int numMasks = decoderOutput.Shape[1]; // Typically 3 masks
            var masks = new List<Tensor<byte>>();
            var scores = new List<double>();

            for (int i = 0; i < numMasks; i++)
            {
                // Extract mask
                var mask = ExtractMask(decoderOutput, i);

                // Convert to binary (threshold at 0)
                var binaryMask = ThresholdMask(mask);

                // Resize to original image size
                var resizedMask = ResizeMask(binaryMask, _originalImageHeight, _originalImageWidth);

                masks.Add(resizedMask);

                // Extract score (predicted IoU)
                var score = ExtractScore(decoderOutput, i);
                scores.Add(score);
            }

            return new SamOutput(masks, scores);
        }

        /// <summary>
        /// Extracts a single mask from decoder output.
        /// </summary>
        private Tensor<T> ExtractMask(Tensor<T> decoderOutput, int maskIndex)
        {
            // Extract [256, 256] mask
            var mask = new Tensor<T>(new[] { MaskOutputSize, MaskOutputSize });

            for (int h = 0; h < MaskOutputSize; h++)
            {
                for (int w = 0; w < MaskOutputSize; w++)
                {
                    mask[h, w] = decoderOutput[0, maskIndex, h, w];
                }
            }

            return mask;
        }

        /// <summary>
        /// Thresholds a mask to binary values (0 or 255).
        /// </summary>
        private Tensor<byte> ThresholdMask(Tensor<T> mask)
        {
            var binary = new Tensor<byte>(mask.Shape);

            for (int h = 0; h < mask.Shape[0]; h++)
            {
                for (int w = 0; w < mask.Shape[1]; w++)
                {
                    // Threshold at 0: positive = foreground
                    binary[h, w] = NumOps.GreaterThan(mask[h, w], NumOps.Zero) ? (byte)255 : (byte)0;
                }
            }

            return binary;
        }

        /// <summary>
        /// Resizes a binary mask to target dimensions.
        /// </summary>
        /// <param name="mask">Input binary mask.</param>
        /// <param name="targetHeight">Target height.</param>
        /// <param name="targetWidth">Target width.</param>
        /// <returns>Resized binary mask.</returns>
        /// <remarks>
        /// <para>
        /// Uses nearest-neighbor interpolation to preserve binary values (0 or 255).
        /// Bilinear would introduce intermediate values, which we don't want.
        /// </para>
        /// </remarks>
        private Tensor<byte> ResizeMask(Tensor<byte> mask, int targetHeight, int targetWidth)
        {
            var resized = new Tensor<byte>(new[] { targetHeight, targetWidth });

            float scaleH = (float)mask.Shape[0] / targetHeight;
            float scaleW = (float)mask.Shape[1] / targetWidth;

            for (int h = 0; h < targetHeight; h++)
            {
                for (int w = 0; w < targetWidth; w++)
                {
                    // Nearest neighbor
                    int srcH = (int)(h * scaleH);
                    int srcW = (int)(w * scaleW);

                    resized[h, w] = mask[srcH, srcW];
                }
            }

            return resized;
        }

        /// <summary>
        /// Extracts the predicted IoU score for a mask.
        /// </summary>
        private double ExtractScore(Tensor<T> decoderOutput, int maskIndex)
        {
            // Scores are typically in a separate output tensor
            // For simplicity, assuming they're part of the main output

            // In actual SAM ONNX model, this would be:
            // var scoresOutput = decoderOutputs[1]; // Second output
            // return Convert.ToDouble(scoresOutput[0, maskIndex]);

            // Placeholder
            return 0.9; // Mock score
        }
    }
}
```

**Step 2: Create comprehensive unit tests**

```csharp
// File: tests/UnitTests/Models/ComputerVision/Segmentation/SamModelTests.cs
using Xunit;
using AiDotNet.Models.ComputerVision.Segmentation;
using AiDotNet.LinearAlgebra;
using Moq;
using System.Collections.Generic;
using System.Drawing;

namespace AiDotNet.Tests.UnitTests.Models.ComputerVision.Segmentation
{
    public class SamModelTests
    {
        [Fact]
        public void Constructor_ValidPaths_CreatesInstance()
        {
            // Arrange
            string imageEncoderPath = "image_encoder.onnx";
            string maskDecoderPath = "mask_decoder.onnx";

            // Act & Assert
            var exception = Record.Exception(() =>
                new SamModel<double>(imageEncoderPath, maskDecoderPath));

            // Will throw NotImplementedException until #280 is done
            Assert.NotNull(exception);
        }

        [Theory]
        [InlineData(null, "decoder.onnx")]
        [InlineData("", "decoder.onnx")]
        [InlineData("encoder.onnx", null)]
        [InlineData("encoder.onnx", "")]
        public void Constructor_InvalidPaths_ThrowsArgumentException(
            string imageEncoderPath, string maskDecoderPath)
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new SamModel<double>(imageEncoderPath, maskDecoderPath));
        }

        [Fact]
        public void SetImage_ValidImage_CachesEmbedding()
        {
            // Arrange
            var mockImageEncoder = new Mock<IOnnxModel<double>>();
            var mockMaskDecoder = new Mock<IOnnxModel<double>>();

            var imageEmbedding = new Tensor<double>(new[] { 1, 256, 64, 64 });
            mockImageEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                           .Returns(imageEmbedding);

            var model = new SamModel<double>(
                mockImageEncoder.Object,
                mockMaskDecoder.Object);

            var image = new Tensor<double>(new[] { 3, 512, 512 });

            // Act
            model.SetImage(image);

            // Assert
            // Image encoder should be called exactly once
            mockImageEncoder.Verify(e => e.Forward(It.IsAny<Tensor<double>>()), Times.Once);
        }

        [Fact]
        public void Predict_WithoutSetImage_ThrowsInvalidOperationException()
        {
            // Arrange
            var model = CreateMockSamModel();
            var points = new List<Point> { new Point(100, 100) };

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                model.Predict(points, null));
        }

        [Fact]
        public void Predict_NoPrompts_ThrowsArgumentException()
        {
            // Arrange
            var model = CreateMockSamModel();
            var image = new Tensor<double>(new[] { 3, 512, 512 });
            model.SetImage(image);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                model.Predict(null, null));
        }

        [Fact]
        public void Predict_PointPrompt_ReturnsValidOutput()
        {
            // Arrange
            var mockImageEncoder = new Mock<IOnnxModel<double>>();
            var mockMaskDecoder = new Mock<IOnnxModel<double>>();

            // Mock image encoder
            var imageEmbedding = new Tensor<double>(new[] { 1, 256, 64, 64 });
            mockImageEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                           .Returns(imageEmbedding);

            // Mock mask decoder: return 3 masks
            var decoderOutput = new Tensor<double>(new[] { 1, 3, 256, 256 });
            mockMaskDecoder.Setup(d => d.Forward(It.IsAny<Tensor<double>>(), It.IsAny<Tensor<double>>()))
                          .Returns(decoderOutput);

            var model = new SamModel<double>(
                mockImageEncoder.Object,
                mockMaskDecoder.Object);

            var image = new Tensor<double>(new[] { 3, 512, 512 });
            model.SetImage(image);

            var points = new List<Point> { new Point(256, 256) };

            // Act
            var output = model.Predict(points, null);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(3, output.Masks.Count);
            Assert.Equal(3, output.Scores.Count);

            // Mask decoder should be called once
            mockMaskDecoder.Verify(d => d.Forward(
                It.IsAny<Tensor<double>>(),
                It.IsAny<Tensor<double>>()), Times.Once);
        }

        [Fact]
        public void Predict_BoxPrompt_ReturnsValidOutput()
        {
            // Arrange
            var model = CreateMockSamModel();
            var image = new Tensor<double>(new[] { 3, 512, 512 });
            model.SetImage(image);

            var bbox = new Rectangle(100, 100, 200, 200);

            // Act
            var output = model.Predict(null, new List<Rectangle> { bbox });

            // Assert
            Assert.NotNull(output);
            Assert.True(output.Masks.Count > 0);
        }

        [Fact]
        public void Predict_CombinedPrompts_ReturnsValidOutput()
        {
            // Arrange
            var model = CreateMockSamModel();
            var image = new Tensor<double>(new[] { 3, 512, 512 });
            model.SetImage(image);

            var points = new List<Point> { new Point(256, 256) };
            var bbox = new Rectangle(100, 100, 200, 200);

            // Act
            var output = model.Predict(points, new List<Rectangle> { bbox });

            // Assert
            Assert.NotNull(output);
            Assert.True(output.Masks.Count > 0);
        }

        [Fact]
        public void Predict_MultiplePoints_HandlesCorrectly()
        {
            // Arrange
            var model = CreateMockSamModel();
            var image = new Tensor<double>(new[] { 3, 512, 512 });
            model.SetImage(image);

            var points = new List<Point>
            {
                new Point(200, 200), // Positive point
                new Point(250, 250), // Another positive
                new Point(400, 400)  // Could be negative
            };

            // Act
            var output = model.Predict(points, null);

            // Assert
            Assert.NotNull(output);
        }

        private SamModel<double> CreateMockSamModel()
        {
            var mockImageEncoder = new Mock<IOnnxModel<double>>();
            var mockMaskDecoder = new Mock<IOnnxModel<double>>();

            mockImageEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                           .Returns(new Tensor<double>(new[] { 1, 256, 64, 64 }));
            mockMaskDecoder.Setup(d => d.Forward(It.IsAny<Tensor<double>>(), It.IsAny<Tensor<double>>()))
                          .Returns(new Tensor<double>(new[] { 1, 3, 256, 256 }));

            return new SamModel<double>(
                mockImageEncoder.Object,
                mockMaskDecoder.Object);
        }
    }
}
```

---

## Phase 2: Integration Testing

### AC 2.2: Integration Test with Real ONNX Models (8 points)

```csharp
// File: tests/IntegrationTests/Models/ComputerVision/Segmentation/SamModelIntegrationTests.cs
using Xunit;
using AiDotNet.Models.ComputerVision.Segmentation;
using AiDotNet.LinearAlgebra;
using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;

namespace AiDotNet.Tests.IntegrationTests.Models.ComputerVision.Segmentation
{
    /// <summary>
    /// Integration tests for SAM model using real ONNX files.
    /// </summary>
    public class SamModelIntegrationTests : IDisposable
    {
        private readonly string _modelDir;
        private readonly string _imageEncoderPath;
        private readonly string _maskDecoderPath;
        private bool _modelsAvailable;

        public SamModelIntegrationTests()
        {
            _modelDir = Path.Combine(Path.GetTempPath(), "sam_models");
            _imageEncoderPath = Path.Combine(_modelDir, "sam_vit_h_image_encoder.onnx");
            _maskDecoderPath = Path.Combine(_modelDir, "sam_vit_h_mask_decoder.onnx");

            _modelsAvailable = File.Exists(_imageEncoderPath) && File.Exists(_maskDecoderPath);
        }

        [Fact(Skip = "Requires manual model download")]
        public void Predict_CarImage_SegmentsCar()
        {
            if (!_modelsAvailable)
            {
                Console.WriteLine("SAM models not found. Download from:");
                Console.WriteLine("https://github.com/facebookresearch/segment-anything");
                return;
            }

            // Arrange
            var sam = new SamModel<float>(_imageEncoderPath, _maskDecoderPath);

            // Load test image: car
            var image = LoadTestImage("test_images/car.jpg");

            sam.SetImage(image);

            // Click in the middle of the car
            var point = new Point(x: 300, y: 200);

            // Act
            var output = sam.Predict(new List<Point> { point }, null);

            // Assert
            Assert.NotNull(output);
            Assert.True(output.Masks.Count > 0);

            var (bestMask, bestScore) = output.GetBestMask();

            // Should have reasonable confidence
            Assert.True(bestScore > 0.7, $"Score {bestScore} is too low");

            // Mask should be correct size
            Assert.Equal(image.Shape[1], bestMask.Shape[0]); // Height
            Assert.Equal(image.Shape[2], bestMask.Shape[1]); // Width

            // Mask should have both foreground and background
            bool hasForeground = false;
            bool hasBackground = false;
            for (int h = 0; h < bestMask.Shape[0]; h++)
            {
                for (int w = 0; w < bestMask.Shape[1]; w++)
                {
                    if (bestMask[h, w] == 255) hasForeground = true;
                    if (bestMask[h, w] == 0) hasBackground = true;
                }
            }

            Assert.True(hasForeground, "Mask should have foreground pixels");
            Assert.True(hasBackground, "Mask should have background pixels");

            Console.WriteLine($"Best mask score: {bestScore:F3}");
            Console.WriteLine($"Total masks generated: {output.Masks.Count}");
        }

        [Fact(Skip = "Requires manual model download")]
        public void Predict_BoxPrompt_SegmentsObjectInBox()
        {
            if (!_modelsAvailable) return;

            // Arrange
            var sam = new SamModel<float>(_imageEncoderPath, _maskDecoderPath);
            var image = LoadTestImage("test_images/multiple_objects.jpg");

            sam.SetImage(image);

            // Draw box around one object
            var bbox = new Rectangle(x: 100, y: 150, width: 200, height: 200);

            // Act
            var output = sam.Predict(null, new List<Rectangle> { bbox });

            // Assert
            var (bestMask, bestScore) = output.GetBestMask();

            Assert.True(bestScore > 0.7);

            // Check that mask is mostly inside the box
            int pixelsInsideBox = 0;
            int totalForeground = 0;

            for (int h = 0; h < bestMask.Shape[0]; h++)
            {
                for (int w = 0; w < bestMask.Shape[1]; w++)
                {
                    if (bestMask[h, w] == 255)
                    {
                        totalForeground++;
                        if (w >= bbox.X && w < bbox.X + bbox.Width &&
                            h >= bbox.Y && h < bbox.Y + bbox.Height)
                        {
                            pixelsInsideBox++;
                        }
                    }
                }
            }

            // Most foreground pixels should be inside the box
            float insideRatio = (float)pixelsInsideBox / totalForeground;
            Assert.True(insideRatio > 0.8,
                $"Only {insideRatio:P0} of foreground is inside box");

            Console.WriteLine($"Foreground inside box: {insideRatio:P0}");
        }

        [Fact(Skip = "Requires manual model download")]
        public void Predict_RefinementWithNegativePoint_ImprovesMask()
        {
            if (!_modelsAvailable) return;

            // Arrange
            var sam = new SamModel<float>(_imageEncoderPath, _maskDecoderPath);
            var image = LoadTestImage("test_images/dog_with_background.jpg");

            sam.SetImage(image);

            // First attempt: single positive point
            var point1 = new Point(250, 200);
            var output1 = sam.Predict(new List<Point> { point1 }, null);
            var (mask1, score1) = output1.GetBestMask();

            // Second attempt: positive + negative point for refinement
            var points2 = new List<Point>
            {
                new Point(250, 200, label: 1),  // Positive: dog
                new Point(450, 100, label: 0)   // Negative: background tree
            };
            var output2 = sam.Predict(points2, null);
            var (mask2, score2) = output2.GetBestMask();

            // Assert: Refinement should improve or maintain quality
            Assert.True(score2 >= score1 - 0.05,
                $"Refinement made it worse: {score1:F3} → {score2:F3}");

            Console.WriteLine($"Initial mask score: {score1:F3}");
            Console.WriteLine($"Refined mask score: {score2:F3}");
        }

        [Fact(Skip = "Requires manual model download")]
        public void Predict_MultipleObjectsSeparately_GeneratesDistinctMasks()
        {
            if (!_modelsAvailable) return;

            // Arrange
            var sam = new SamModel<float>(_imageEncoderPath, _maskDecoderPath);
            var image = LoadTestImage("test_images/three_animals.jpg");

            sam.SetImage(image);

            // Segment each animal separately
            var point1 = new Point(150, 200); // Animal 1
            var point2 = new Point(350, 200); // Animal 2
            var point3 = new Point(550, 200); // Animal 3

            var output1 = sam.Predict(new List<Point> { point1 }, null);
            var output2 = sam.Predict(new List<Point> { point2 }, null);
            var output3 = sam.Predict(new List<Point> { point3 }, null);

            var (mask1, _) = output1.GetBestMask();
            var (mask2, _) = output2.GetBestMask();
            var (mask3, _) = output3.GetBestMask();

            // Assert: Masks should be different (low overlap)
            float overlap12 = CalculateMaskOverlap(mask1, mask2);
            float overlap13 = CalculateMaskOverlap(mask1, mask3);
            float overlap23 = CalculateMaskOverlap(mask2, mask3);

            Assert.True(overlap12 < 0.3, $"Mask 1-2 overlap too high: {overlap12:P0}");
            Assert.True(overlap13 < 0.3, $"Mask 1-3 overlap too high: {overlap13:P0}");
            Assert.True(overlap23 < 0.3, $"Mask 2-3 overlap too high: {overlap23:P0}");

            Console.WriteLine($"Mask overlaps: 1-2={overlap12:P0}, 1-3={overlap13:P0}, 2-3={overlap23:P0}");
        }

        private Tensor<float> LoadTestImage(string path)
        {
            // Load image from file and convert to tensor
            // Would use ImageSharp or System.Drawing

            throw new NotImplementedException("Requires image loading library");
        }

        private float CalculateMaskOverlap(Tensor<byte> mask1, Tensor<byte> mask2)
        {
            int intersection = 0;
            int union = 0;

            for (int h = 0; h < mask1.Shape[0]; h++)
            {
                for (int w = 0; w < mask1.Shape[1]; w++)
                {
                    bool inMask1 = mask1[h, w] == 255;
                    bool inMask2 = mask2[h, w] == 255;

                    if (inMask1 && inMask2) intersection++;
                    if (inMask1 || inMask2) union++;
                }
            }

            return union > 0 ? (float)intersection / union : 0;
        }

        public void Dispose()
        {
            // Cleanup if needed
        }
    }
}
```

---

## Common Pitfalls and Best Practices

### 1. Always Call SetImage() Before Predict()

**❌ Wrong:**
```csharp
var sam = new SamModel<float>(encoderPath, decoderPath);
var output = sam.Predict(points, null); // ERROR: No image set!
```

**✅ Correct:**
```csharp
var sam = new SamModel<float>(encoderPath, decoderPath);
sam.SetImage(image); // Must call this first
var output = sam.Predict(points, null); // Now works
```

### 2. Reuse Image Embedding for Multiple Prompts

**❌ Wrong:**
```csharp
// Inefficient: Recomputes image embedding each time
sam.SetImage(image); // ~200ms
var mask1 = sam.Predict(point1, null);

sam.SetImage(image); // ~200ms again!
var mask2 = sam.Predict(point2, null);
```

**✅ Correct:**
```csharp
// Efficient: Compute once, use many times
sam.SetImage(image); // ~200ms (once)
var mask1 = sam.Predict(point1, null); // ~30ms
var mask2 = sam.Predict(point2, null); // ~30ms
var mask3 = sam.Predict(point3, null); // ~30ms
```

### 3. Use Highest-Scoring Mask

**❌ Wrong:**
```csharp
// Just use first mask (might not be best)
var mask = output.Masks[0];
```

**✅ Correct:**
```csharp
// Use the mask SAM is most confident about
var (bestMask, score) = output.GetBestMask();
```

### 4. Handle Coordinate Scaling

**Important:** Prompt coordinates must be in original image space, not resized space!

```csharp
// User clicks at (150, 200) in a 300×400 image
var point = new Point(150, 200); // Correct: Original coordinates

// SAM internally scales to 1024×1024 for encoding
// But you don't need to worry about that!
```

### 5. Resize Masks to Original Size

SAM outputs 256×256 masks for efficiency. Always resize back:

```csharp
// ✅ SamModel does this automatically
var output = sam.Predict(points, null);
// Masks are already at original image resolution
```

---

## Testing Strategy

### Unit Tests (Mocked)
- Test constructor validation
- Test SetImage caching behavior
- Test Predict with different prompt combinations
- Test coordinate normalization
- Test mask thresholding and resizing
- Test error handling

### Integration Tests (Real Models)
- Test single-click segmentation
- Test box-based segmentation
- Test refinement with negative points
- Test multiple objects in one image
- Test various image sizes and aspect ratios
- Verify mask quality with IoU scores

### Performance Tests
```csharp
[Fact]
public void SetImage_Performance_ReasonablyFast()
{
    var sw = Stopwatch.StartNew();
    sam.SetImage(image);
    sw.Stop();

    // Should encode in under 500ms on GPU
    Assert.True(sw.ElapsedMilliseconds < 500);
}

[Fact]
public void Predict_Performance_Interactive()
{
    sam.SetImage(image);

    var sw = Stopwatch.StartNew();
    var output = sam.Predict(points, null);
    sw.Stop();

    // Should predict in under 100ms for real-time interaction
    Assert.True(sw.ElapsedMilliseconds < 100);
}
```

---

## Next Steps After Implementation

1. **Add automatic mask postprocessing:** Fill holes, smooth boundaries
2. **Implement everything mode:** Segment all objects automatically
3. **Add mask refinement tools:** Dilate, erode, smooth
4. **Support text prompts:** Use CLIP to convert text to visual prompts
5. **Add mask tracking:** Track segmented objects across video frames
6. **Integrate with object detection:** Use detections as automatic prompts

---

## Summary

This guide covered:
- ✅ Understanding SAM's promptable segmentation paradigm
- ✅ Two-stage architecture (heavy encoder + light decoder)
- ✅ Point, box, and combined prompting strategies
- ✅ Interactive refinement with positive/negative points
- ✅ Efficient image embedding caching
- ✅ Multiple mask candidates with confidence scores
- ✅ Comprehensive testing strategies

**Key concepts:**
- **Promptable segmentation:** Click or box to segment anything
- **Zero-shot capability:** Works on objects never seen during training
- **Two-stage design:** Cache expensive encoding, fast decoding
- **Multiple masks:** Different granularities (whole object vs. parts)
- **Interactive workflow:** Real-time refinement with user feedback

**Dependencies:**
- Issue #280: OnnxModel wrapper (prerequisite)
- Issue #330: Image preprocessing (required for resizing/normalization)
- Existing: VisionTransformer, MultiHeadAttention, Tensor operations

**For beginners:** Start by understanding the two-stage architecture. Implement SetImage (caching) first, then Predict (prompt processing). Test with simple point prompts before adding boxes and refinement. Remember: SAM is designed for interactivity, so performance matters!
