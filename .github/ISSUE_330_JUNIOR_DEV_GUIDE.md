# Issue #330: Junior Developer Implementation Guide

## Understanding Tensor<T> for Images

**File**: `src/LinearAlgebra/Tensor.cs`

### What is a Tensor?
A Tensor is a multi-dimensional array. For images:
- **Shape**: `[height, width, channels]`
- **Example RGB image (224x224)**: `new Tensor<double>(224, 224, 3)`
- **Example Grayscale (28x28)**: `new Tensor<double>(28, 28, 1)`

### Key Tensor Properties/Methods:
```csharp
// Accessing dimensions
int height = tensor.Dimensions[0];  // Height
int width = tensor.Dimensions[1];   // Width
int channels = tensor.Dimensions[2]; // Channels (3 for RGB, 1 for grayscale)

// Accessing individual elements
T value = tensor[row, col, channel];

// Creating new tensor
var newTensor = new Tensor<T>(height, width, channels);
```

---

## Phase 1: Step-by-Step Implementation

### AC 1.1: ImageProcessor.Resize() - WRAPPER AROUND EXISTING CODE

**Existing Code to Use**: `src/Interpolation/BilinearInterpolation.cs`

**Step 1**: Understand existing BilinearInterpolation
```bash
# Read the existing interpolation code
cat src/Interpolation/BilinearInterpolation.cs
```

**Step 2**: Create ImageProcessor.cs
```csharp
// File: src/Images/ImageProcessor.cs
namespace AiDotNet.Images;

public static class ImageProcessor
{
    /// <summary>
    /// Resizes an image tensor to specified dimensions using bilinear interpolation.
    /// </summary>
    /// <param name="image">Input image as Tensor with shape [height, width, channels]</param>
    /// <param name="newHeight">Target height in pixels</param>
    /// <param name="newWidth">Target width in pixels</param>
    /// <returns>Resized image tensor with shape [newHeight, newWidth, channels]</returns>
    public static Tensor<T> Resize<T>(Tensor<T> image, int newHeight, int newWidth)
    {
        // Validate input
        if (image.Dimensions.Length != 3)
            throw new ArgumentException("Image tensor must have 3 dimensions [height, width, channels]");

        int oldHeight = image.Dimensions[0];
        int oldWidth = image.Dimensions[1];
        int channels = image.Dimensions[2];

        // Create output tensor
        var result = new Tensor<T>(newHeight, newWidth, channels);

        // Use existing BilinearInterpolation for each channel
        var interpolator = new BilinearInterpolation<T>();

        // Process each channel separately
        for (int c = 0; c < channels; c++)
        {
            // Extract channel as Matrix
            var channelMatrix = new Matrix<T>(oldHeight, oldWidth);
            for (int r = 0; r < oldHeight; r++)
                for (int col = 0; col < oldWidth; col++)
                    channelMatrix[r, col] = image[r, col, c];

            // Interpolate using existing code
            var resizedChannel = interpolator.Interpolate(channelMatrix, newHeight, newWidth);

            // Copy back to result tensor
            for (int r = 0; r < newHeight; r++)
                for (int col = 0; col < newWidth; col++)
                    result[r, col, c] = resizedChannel[r, col];
        }

        return result;
    }
}
```

**Step 3**: Create unit test
```csharp
// File: tests/UnitTests/Images/ImageProcessorTests.cs
[Fact]
public void Resize_ValidImage_ReturnsCorrectDimensions()
{
    // Arrange
    var image = new Tensor<double>(100, 100, 3); // 100x100 RGB
    // Fill with test data...

    // Act
    var resized = ImageProcessor.Resize(image, 50, 50);

    // Assert
    Assert.Equal(50, resized.Dimensions[0]); // Height
    Assert.Equal(50, resized.Dimensions[1]); // Width
    Assert.Equal(3, resized.Dimensions[2]);  // Channels preserved
}
```

---

### AC 1.1: ImageProcessor.Grayscale() - NEW IMPLEMENTATION NEEDED

**No existing code - you must implement RGB to Gray conversion**

**Standard Formula**: `Gray = 0.299*R + 0.587*G + 0.114*B`

**Implementation**:
```csharp
/// <summary>
/// Converts RGB image to grayscale using standard luminosity formula.
/// </summary>
/// <param name="image">Input RGB image with shape [height, width, 3]</param>
/// <returns>Grayscale image with shape [height, width, 1]</returns>
public static Tensor<T> Grayscale<T>(Tensor<T> image)
{
    var numOps = NumericOperations<T>.Instance;

    // Validate RGB input
    if (image.Dimensions[2] != 3)
        throw new ArgumentException("Input must be RGB image (3 channels)");

    int height = image.Dimensions[0];
    int width = image.Dimensions[1];

    var result = new Tensor<T>(height, width, 1);

    // Luminosity weights (ITU-R BT.601 standard)
    T weightR = numOps.FromDouble(0.299);
    T weightG = numOps.FromDouble(0.587);
    T weightB = numOps.FromDouble(0.114);

    // Convert each pixel
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            T red = image[r, c, 0];
            T green = image[r, c, 1];
            T blue = image[r, c, 2];

            // Weighted sum
            T gray = numOps.Add(
                numOps.Add(
                    numOps.Multiply(red, weightR),
                    numOps.Multiply(green, weightG)
                ),
                numOps.Multiply(blue, weightB)
            );

            result[r, c, 0] = gray;
        }
    }

    return result;
}
```

---

## Phase 2: Geometric Augmentations

### AC 2.1: ImageAugmenter.Flip() - NEW IMPLEMENTATION

**Horizontal Flip**: Reverse column order
**Vertical Flip**: Reverse row order

**Implementation**:
```csharp
// File: src/Images/ImageAugmenter.cs
public static class ImageAugmenter
{
    /// <summary>
    /// Flips image horizontally and/or vertically.
    /// </summary>
    /// <param name="image">Input image tensor</param>
    /// <param name="horizontal">If true, flip left-right</param>
    /// <param name="vertical">If true, flip top-bottom</param>
    /// <returns>Flipped image tensor</returns>
    public static Tensor<T> Flip<T>(Tensor<T> image, bool horizontal, bool vertical)
    {
        int height = image.Dimensions[0];
        int width = image.Dimensions[1];
        int channels = image.Dimensions[2];

        var result = new Tensor<T>(height, width, channels);

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                // Determine source coordinates
                int srcRow = vertical ? (height - 1 - r) : r;
                int srcCol = horizontal ? (width - 1 - c) : c;

                // Copy all channels
                for (int ch = 0; ch < channels; ch++)
                    result[r, c, ch] = image[srcRow, srcCol, ch];
            }
        }

        return result;
    }
}
```

---

## Common Pitfalls to Avoid:

1. **DON'T use `default(T)`** - Use `NumOps.Zero` instead
2. **DON'T hardcode doubles** - Use `NumOps.FromDouble()`
3. **DON'T assume `double` type** - Support all `T` where `INumericOperations<T>` exists
4. **DON'T forget channel dimension** - Images are 3D `[H, W, C]` not 2D
5. **DO validate input dimensions** - Check tensor has exactly 3 dimensions
6. **DO use existing infrastructure** - BilinearInterpolation, MinMaxNormalizer
7. **DO write comprehensive tests** - Test multiple image sizes, channels, types

---

## Testing Strategy:

1. **Unit Tests**: Test each method independently
2. **Integration Tests**: Test with ConvolutionalNeuralNetwork
3. **Type Tests**: Test with `double`, `float` generic types
4. **Edge Cases**: Test 1x1 images, single channel, large images

**Next Steps**: Start with Resize() wrapper, then Grayscale(), then Flip(). Build incrementally!
