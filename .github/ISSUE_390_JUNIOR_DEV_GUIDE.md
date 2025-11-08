# Issue #390: Junior Developer Implementation Guide - Data Augmentation

## Understanding Data Augmentation

### What is Data Augmentation?
Data augmentation artificially increases the size and diversity of your training dataset by applying transformations that preserve the label. This helps models:

1. **Generalize Better**: Learn robust features that work on varied inputs
2. **Reduce Overfitting**: More diverse training data prevents memorization
3. **Handle Class Imbalance**: Generate more examples of underrepresented classes

### Types of Augmentation

**Image Augmentation**:
- Geometric: Rotation, flipping, cropping, scaling
- Color: Brightness, contrast, saturation, hue
- Noise: Gaussian noise, blur, dropout

**Text Augmentation**:
- Synonym replacement: Replace words with synonyms
- Back-translation: Translate to another language and back
- Random insertion/deletion/swap of words

**Tabular Augmentation**:
- SMOTE: Synthetic Minority Oversampling
- Gaussian noise: Add small random noise to numeric features
- Mixup: Create convex combinations of examples

---

## Phase 1: Image Augmentation

### AC 1.1: Base ImageAugmenter Class

**File**: `src/Images/ImageAugmenter.cs`

```csharp
namespace AiDotNet.Images;

/// <summary>
/// Provides image augmentation techniques to increase training data diversity.
/// </summary>
/// <remarks>
/// <para>
/// Image augmentation creates modified versions of training images while preserving labels.
/// This helps neural networks generalize better and reduces overfitting.
/// </para>
/// <para><b>For Beginners:</b> Think of augmentation like taking photos of the same object from different angles.
///
/// If you're teaching a model to recognize cats:
/// - Original image: Cat facing forward
/// - Flipped image: Cat facing right (still a cat!)
/// - Rotated image: Cat at 15 degree angle (still a cat!)
/// - Brighter image: Cat in sunlight (still a cat!)
///
/// By training on all these variations, the model learns that a cat is still a cat
/// regardless of orientation, lighting, or position.
///
/// This is especially useful when you have limited training data.
/// Instead of 100 cat photos, you can create 1000+ variations.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type for pixel values.</typeparam>
public class ImageAugmenter<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new image augmenter.
    /// </summary>
    /// <param name="seed">Random seed for reproducibility (optional).</param>
    public ImageAugmenter(int? seed = null)
    {
        _numOps = NumericOperations<T>.Instance;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    #region Geometric Transformations

    /// <summary>
    /// Flips an image horizontally and/or vertically.
    /// </summary>
    /// <param name="image">Input image [height, width, channels].</param>
    /// <param name="horizontal">Flip horizontally (left-right).</param>
    /// <param name="vertical">Flip vertically (top-bottom).</param>
    /// <returns>Flipped image.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Flipping creates a mirror image.
    ///
    /// Horizontal flip: Mirrors left-right (like a mirror on a wall)
    /// Vertical flip: Mirrors top-bottom (like a reflection in water)
    ///
    /// Use horizontal flips for most objects (cars, animals, people).
    /// Use vertical flips carefully - some objects look unnatural upside down.
    /// </para>
    /// </remarks>
    public Tensor<T> Flip(Tensor<T> image, bool horizontal = true, bool vertical = false)
    {
        if (image.Shape.Length != 3)
            throw new ArgumentException("Image must be 3D tensor [height, width, channels]");

        int height = image.Shape[0];
        int width = image.Shape[1];
        int channels = image.Shape[2];

        var result = new Tensor<T>(image.Shape);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int srcH = vertical ? (height - 1 - h) : h;
                int srcW = horizontal ? (width - 1 - w) : w;

                for (int c = 0; c < channels; c++)
                {
                    result[h, w, c] = image[srcH, srcW, c];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Randomly flips image with given probability.
    /// </summary>
    public Tensor<T> RandomFlip(Tensor<T> image, double probability = 0.5, bool horizontal = true, bool vertical = false)
    {
        if (_random.NextDouble() < probability)
        {
            return Flip(image, horizontal, vertical);
        }
        return image;
    }

    /// <summary>
    /// Rotates image by 90, 180, or 270 degrees.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="times">Number of 90-degree rotations (1=90°, 2=180°, 3=270°).</param>
    /// <returns>Rotated image.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rotation turns the image around its center.
    ///
    /// times=1: Rotate 90 degrees clockwise
    /// times=2: Rotate 180 degrees (upside down)
    /// times=3: Rotate 270 degrees (same as 90 counter-clockwise)
    ///
    /// Only supports 90-degree increments for efficiency.
    /// For arbitrary angles, use more complex interpolation.
    /// </para>
    /// </remarks>
    public Tensor<T> Rotate90(Tensor<T> image, int times = 1)
    {
        if (image.Shape.Length != 3)
            throw new ArgumentException("Image must be 3D tensor");

        times = times % 4; // Normalize to 0-3
        if (times < 0) times += 4;

        if (times == 0)
            return image;

        var result = image;
        for (int i = 0; i < times; i++)
        {
            result = Rotate90Once(result);
        }

        return result;
    }

    private Tensor<T> Rotate90Once(Tensor<T> image)
    {
        int height = image.Shape[0];
        int width = image.Shape[1];
        int channels = image.Shape[2];

        // After 90° rotation: new_height = old_width, new_width = old_height
        var result = new Tensor<T>(new[] { width, height, channels });

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                // 90° clockwise: (h, w) -> (w, height-1-h)
                for (int c = 0; c < channels; c++)
                {
                    result[w, height - 1 - h, c] = image[h, w, c];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Randomly rotates image by 0, 90, 180, or 270 degrees.
    /// </summary>
    public Tensor<T> RandomRotate90(Tensor<T> image)
    {
        int times = _random.Next(4); // 0, 1, 2, or 3
        return Rotate90(image, times);
    }

    /// <summary>
    /// Extracts a random crop from the image.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="cropHeight">Height of crop.</param>
    /// <param name="cropWidth">Width of crop.</param>
    /// <returns>Cropped image.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cropping cuts out a rectangular region.
    ///
    /// Like using the crop tool in a photo editor, this extracts a portion of the image.
    /// Random cropping helps the model learn to recognize objects even when they're:
    /// - Partially visible (cut off at edges)
    /// - At different positions in the frame
    /// - At different scales (if you vary crop size)
    ///
    /// Useful for: Object detection, classification with varying object positions
    /// </para>
    /// </remarks>
    public Tensor<T> RandomCrop(Tensor<T> image, int cropHeight, int cropWidth)
    {
        int height = image.Shape[0];
        int width = image.Shape[1];
        int channels = image.Shape[2];

        if (cropHeight > height || cropWidth > width)
            throw new ArgumentException("Crop size cannot be larger than image size");

        int maxTop = height - cropHeight;
        int maxLeft = width - cropWidth;

        int top = _random.Next(maxTop + 1);
        int left = _random.Next(maxLeft + 1);

        var result = new Tensor<T>(new[] { cropHeight, cropWidth, channels });

        for (int h = 0; h < cropHeight; h++)
        {
            for (int w = 0; w < cropWidth; w++)
            {
                for (int c = 0; c < channels; c++)
                {
                    result[h, w, c] = image[top + h, left + w, c];
                }
            }
        }

        return result;
    }

    #endregion

    #region Color/Intensity Transformations

    /// <summary>
    /// Adjusts image brightness by adding a constant value.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="delta">Brightness adjustment (-1.0 to 1.0).</param>
    /// <returns>Brightness-adjusted image.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Brightness makes the image lighter or darker.
    ///
    /// Positive delta: Increases brightness (adds light)
    /// Negative delta: Decreases brightness (makes darker)
    ///
    /// This helps models work in different lighting conditions:
    /// - Indoor vs outdoor
    /// - Day vs night
    /// - Shadows vs direct light
    ///
    /// Values are clipped to valid range (typically 0-1 or 0-255).
    /// </para>
    /// </remarks>
    public Tensor<T> AdjustBrightness(Tensor<T> image, double delta)
    {
        T deltaT = _numOps.FromDouble(delta);
        return image.Transform(pixel => _numOps.Add(pixel, deltaT));
    }

    /// <summary>
    /// Randomly adjusts brightness.
    /// </summary>
    public Tensor<T> RandomBrightness(Tensor<T> image, double maxDelta = 0.2)
    {
        double delta = (_random.NextDouble() * 2 - 1) * maxDelta; // Range: [-maxDelta, maxDelta]
        return AdjustBrightness(image, delta);
    }

    /// <summary>
    /// Adjusts image contrast by scaling around the mean.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="factor">Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast).</param>
    /// <returns>Contrast-adjusted image.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Contrast is the difference between light and dark areas.
    ///
    /// High contrast: Very distinct light and dark (like a black and white photo)
    /// Low contrast: Washed out, gray, similar tones throughout
    ///
    /// Formula: pixel = mean + factor * (pixel - mean)
    /// - factor > 1: Increases contrast (lights lighter, darks darker)
    /// - factor < 1: Decreases contrast (everything closer to middle gray)
    /// - factor = 1: No change
    ///
    /// Helps models handle:
    /// - Foggy or hazy conditions (low contrast)
    /// - High-contrast scenes (bright sunlight)
    /// </para>
    /// </remarks>
    public Tensor<T> AdjustContrast(Tensor<T> image, double factor)
    {
        // Calculate mean pixel value
        var flattened = image.ToVector();
        T mean = _numOps.Zero;
        for (int i = 0; i < flattened.Length; i++)
        {
            mean = _numOps.Add(mean, flattened[i]);
        }
        mean = _numOps.Divide(mean, _numOps.FromDouble(flattened.Length));

        T factorT = _numOps.FromDouble(factor);

        // Apply contrast adjustment
        return image.Transform(pixel =>
        {
            T diff = _numOps.Subtract(pixel, mean);
            T adjusted = _numOps.Multiply(diff, factorT);
            return _numOps.Add(mean, adjusted);
        });
    }

    /// <summary>
    /// Randomly adjusts contrast.
    /// </summary>
    public Tensor<T> RandomContrast(Tensor<T> image, double minFactor = 0.8, double maxFactor = 1.2)
    {
        double factor = minFactor + _random.NextDouble() * (maxFactor - minFactor);
        return AdjustContrast(image, factor);
    }

    #endregion

    #region Noise and Blur

    /// <summary>
    /// Adds Gaussian noise to the image.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="mean">Mean of Gaussian distribution (default: 0).</param>
    /// <param name="stddev">Standard deviation (default: 0.1).</param>
    /// <returns>Image with added noise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Noise adds random variations to pixel values.
    ///
    /// Think of it like TV static or graininess in photos.
    /// Adding noise during training makes models robust to:
    /// - Low-quality images
    /// - Sensor noise in cameras
    /// - Compression artifacts
    ///
    /// Gaussian noise: Most values near mean, fewer extreme values (bell curve)
    /// Common in real cameras and sensors
    ///
    /// Use sparingly - too much noise makes images unrecognizable!
    /// </para>
    /// </remarks>
    public Tensor<T> AddGaussianNoise(Tensor<T> image, double mean = 0.0, double stddev = 0.1)
    {
        return image.Transform(pixel =>
        {
            // Box-Muller transform for Gaussian random numbers
            double u1 = _random.NextDouble();
            double u2 = _random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            double noise = mean + stddev * randStdNormal;

            return _numOps.Add(pixel, _numOps.FromDouble(noise));
        });
    }

    /// <summary>
    /// Adds salt-and-pepper noise (random black and white pixels).
    /// </summary>
    public Tensor<T> AddSaltPepperNoise(Tensor<T> image, double probability = 0.01)
    {
        return image.Transform(pixel =>
        {
            double rand = _random.NextDouble();
            if (rand < probability / 2)
                return _numOps.Zero; // Pepper (black)
            else if (rand < probability)
                return _numOps.One;  // Salt (white)
            else
                return pixel;
        });
    }

    #endregion

    #region Composition and Batch Operations

    /// <summary>
    /// Applies multiple augmentations sequentially.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="augmentations">List of augmentation functions.</param>
    /// <returns>Augmented image.</returns>
    public Tensor<T> Compose(Tensor<T> image, params Func<Tensor<T>, Tensor<T>>[] augmentations)
    {
        var result = image;
        foreach (var aug in augmentations)
        {
            result = aug(result);
        }
        return result;
    }

    /// <summary>
    /// Applies random augmentations to create multiple versions of an image.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="count">Number of augmented versions to generate.</param>
    /// <returns>List of augmented images.</returns>
    public List<Tensor<T>> GenerateAugmentedBatch(Tensor<T> image, int count = 5)
    {
        var batch = new List<Tensor<T>>();

        for (int i = 0; i < count; i++)
        {
            var augmented = image;

            // Apply random augmentations
            if (_random.NextDouble() < 0.5)
                augmented = RandomFlip(augmented);

            if (_random.NextDouble() < 0.5)
                augmented = RandomRotate90(augmented);

            if (_random.NextDouble() < 0.5)
                augmented = RandomBrightness(augmented);

            if (_random.NextDouble() < 0.5)
                augmented = RandomContrast(augmented);

            if (_random.NextDouble() < 0.2)
                augmented = AddGaussianNoise(augmented, stddev: 0.05);

            batch.Add(augmented);
        }

        return batch;
    }

    #endregion
}
```

---

## Phase 2: Text Augmentation

### AC 2.1: TextAugmenter Class

**File**: `src/Text/TextAugmenter.cs`

```csharp
namespace AiDotNet.Text;

/// <summary>
/// Provides text augmentation techniques for NLP tasks.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Text augmentation creates variations of text while preserving meaning.
///
/// Unlike images where you can flip/rotate, text needs semantic-preserving changes:
/// - Synonym replacement: "happy" → "joyful"
/// - Word insertion: Add filler words
/// - Word deletion: Remove less important words
/// - Word swapping: Change word order
///
/// Benefits:
/// - Increases training data for small datasets
/// - Improves model robustness to paraphrasing
/// - Helps with rare words and variations
///
/// Example:
/// Original: "The cat sat on the mat"
/// Synonym: "The feline sat on the mat"
/// Insertion: "The small cat sat on the mat"
/// Deletion: "Cat sat on mat"
/// Swap: "The cat on the mat sat"
/// </para>
/// </remarks>
public class TextAugmenter
{
    private readonly Random _random;

    // Simple synonym dictionary (in production, use WordNet or similar)
    private readonly Dictionary<string, List<string>> _synonyms;

    public TextAugmenter(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _synonyms = InitializeSynonymDictionary();
    }

    private Dictionary<string, List<string>> InitializeSynonymDictionary()
    {
        // Basic synonym dictionary (expand with proper thesaurus)
        return new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
        {
            { "good", new List<string> { "great", "excellent", "fine", "nice" } },
            { "bad", new List<string> { "poor", "terrible", "awful", "horrible" } },
            { "big", new List<string> { "large", "huge", "enormous", "massive" } },
            { "small", new List<string> { "tiny", "little", "mini", "compact" } },
            { "happy", new List<string> { "joyful", "pleased", "glad", "cheerful" } },
            { "sad", new List<string> { "unhappy", "sorrowful", "depressed", "gloomy" } },
            // Add more synonyms as needed
        };
    }

    /// <summary>
    /// Replaces words with their synonyms randomly.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <param name="probability">Probability of replacing each word (0.0-1.0).</param>
    /// <returns>Text with synonym replacements.</returns>
    public string SynonymReplacement(string text, double probability = 0.3)
    {
        var words = text.Split(' ');
        var result = new List<string>();

        foreach (var word in words)
        {
            if (_random.NextDouble() < probability && _synonyms.ContainsKey(word))
            {
                var synonymList = _synonyms[word];
                var synonym = synonymList[_random.Next(synonymList.Count)];
                result.Add(synonym);
            }
            else
            {
                result.Add(word);
            }
        }

        return string.Join(" ", result);
    }

    /// <summary>
    /// Randomly inserts words from the text at random positions.
    /// </summary>
    public string RandomInsertion(string text, int count = 1)
    {
        var words = text.Split(' ').ToList();

        for (int i = 0; i < count; i++)
        {
            if (words.Count == 0) break;

            // Pick a random word from the text
            var randomWord = words[_random.Next(words.Count)];

            // Insert at random position
            int insertPos = _random.Next(words.Count + 1);
            words.Insert(insertPos, randomWord);
        }

        return string.Join(" ", words);
    }

    /// <summary>
    /// Randomly deletes words from the text.
    /// </summary>
    public string RandomDeletion(string text, double probability = 0.1)
    {
        var words = text.Split(' ');
        if (words.Length == 1) return text; // Don't delete if only one word

        var result = words.Where(w => _random.NextDouble() > probability).ToList();

        if (result.Count == 0) // Don't delete everything
            return words[_random.Next(words.Length)];

        return string.Join(" ", result);
    }

    /// <summary>
    /// Randomly swaps adjacent words.
    /// </summary>
    public string RandomSwap(string text, int swaps = 1)
    {
        var words = text.Split(' ').ToList();
        if (words.Count < 2) return text;

        for (int i = 0; i < swaps; i++)
        {
            int idx1 = _random.Next(words.Count);
            int idx2 = _random.Next(words.Count);

            // Swap
            var temp = words[idx1];
            words[idx1] = words[idx2];
            words[idx2] = temp;
        }

        return string.Join(" ", words);
    }

    /// <summary>
    /// Generates multiple augmented versions of a text.
    /// </summary>
    public List<string> GenerateAugmentedBatch(string text, int count = 5)
    {
        var batch = new List<string>();

        for (int i = 0; i < count; i++)
        {
            var augmented = text;

            // Randomly apply augmentations
            int choice = _random.Next(4);
            switch (choice)
            {
                case 0:
                    augmented = SynonymReplacement(augmented);
                    break;
                case 1:
                    augmented = RandomInsertion(augmented);
                    break;
                case 2:
                    augmented = RandomDeletion(augmented);
                    break;
                case 3:
                    augmented = RandomSwap(augmented);
                    break;
            }

            batch.Add(augmented);
        }

        return batch;
    }
}
```

---

## Phase 3: Tabular Data Augmentation

### AC 3.1: TabularAugmenter Class

**File**: `src/Data/TabularAugmenter.cs`

```csharp
namespace AiDotNet.Data;

/// <summary>
/// Provides augmentation techniques for tabular (structured) data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Tabular augmentation for spreadsheet-like data (rows and columns).
///
/// Unlike images/text, we can't flip or rotate tabular data.
/// Instead we use:
/// - Gaussian noise: Add small random variations to numbers
/// - Mixup: Blend two examples together
/// - Feature permutation: Shuffle values within a column
///
/// Use cases:
/// - Medical records (limited patient data)
/// - Financial data (rare fraud cases)
/// - Sensor data (create variations for robustness)
///
/// Be careful: Adding too much noise or mixing incompatible samples
/// can create unrealistic data that hurts model performance.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
public class TabularAugmenter<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    public TabularAugmenter(int? seed = null)
    {
        _numOps = NumericOperations<T>.Instance;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Adds Gaussian noise to numeric features.
    /// </summary>
    /// <param name="data">Input data matrix [samples, features].</param>
    /// <param name="noiseLevel">Standard deviation of noise relative to feature std (default: 0.05).</param>
    /// <param name="featuresToAugment">Indices of features to augment (null = all).</param>
    /// <returns>Data with added noise.</returns>
    public Matrix<T> AddGaussianNoise(
        Matrix<T> data,
        double noiseLevel = 0.05,
        int[]? featuresToAugment = null)
    {
        var result = new Matrix<T>(data.Rows, data.Columns);

        // Calculate std for each feature
        var featureStdDevs = new T[data.Columns];
        for (int col = 0; col < data.Columns; col++)
        {
            var column = data.GetColumn(col);
            var mean = StatisticsHelper<T>.CalculateMean(column);
            var variance = StatisticsHelper<T>.CalculateVariance(column, mean);
            featureStdDevs[col] = _numOps.Sqrt(variance);
        }

        // Apply noise
        for (int row = 0; row < data.Rows; row++)
        {
            for (int col = 0; col < data.Columns; col++)
            {
                bool shouldAugment = featuresToAugment == null ||
                                      featuresToAugment.Contains(col);

                if (shouldAugment)
                {
                    // Generate Gaussian noise
                    double u1 = _random.NextDouble();
                    double u2 = _random.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                           Math.Sin(2.0 * Math.PI * u2);

                    double stdDev = Convert.ToDouble(featureStdDevs[col]);
                    double noise = noiseLevel * stdDev * randStdNormal;

                    result[row, col] = _numOps.Add(
                        data[row, col],
                        _numOps.FromDouble(noise)
                    );
                }
                else
                {
                    result[row, col] = data[row, col];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Creates synthetic samples using Mixup.
    /// Mixup: x_new = lambda * x1 + (1-lambda) * x2
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <param name="sampleCount">Number of synthetic samples to generate.</param>
    /// <param name="alpha">Beta distribution parameter (default: 0.2).</param>
    /// <returns>Matrix of synthetic samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Mixup blends two examples together.
    ///
    /// Imagine mixing two smoothies:
    /// - Smoothie A: 100% strawberry
    /// - Smoothie B: 100% banana
    /// - Mixed: 70% strawberry + 30% banana = new flavor!
    ///
    /// In ML:
    /// - Sample A: Patient with condition X
    /// - Sample B: Patient with condition Y
    /// - Mixed: Blend of both conditions (useful for boundary cases)
    ///
    /// lambda controls the blend ratio (0.5 = equal parts, 0.8 = mostly first sample)
    /// </para>
    /// </remarks>
    public Matrix<T> Mixup(Matrix<T> data, int sampleCount, double alpha = 0.2)
    {
        var result = new Matrix<T>(sampleCount, data.Columns);

        for (int i = 0; i < sampleCount; i++)
        {
            // Randomly select two samples
            int idx1 = _random.Next(data.Rows);
            int idx2 = _random.Next(data.Rows);

            // Generate mixing coefficient using Beta distribution
            // For simplicity, use uniform random for now
            double lambda = _random.NextDouble();

            // Mix the two samples
            for (int col = 0; col < data.Columns; col++)
            {
                T val1 = data[idx1, col];
                T val2 = data[idx2, col];

                T mixed = _numOps.Add(
                    _numOps.Multiply(val1, _numOps.FromDouble(lambda)),
                    _numOps.Multiply(val2, _numOps.FromDouble(1.0 - lambda))
                );

                result[i, col] = mixed;
            }
        }

        return result;
    }

    /// <summary>
    /// Creates augmented dataset by duplicating and adding noise.
    /// </summary>
    public Matrix<T> AugmentDataset(Matrix<T> data, int augmentationsPerSample = 2)
    {
        var augmented = new List<Matrix<T>> { data };

        for (int i = 0; i < augmentationsPerSample; i++)
        {
            var noisy = AddGaussianNoise(data, noiseLevel: 0.05 * (i + 1));
            augmented.Add(noisy);
        }

        // Concatenate all augmented versions
        return ConcatenateMatrices(augmented);
    }

    private Matrix<T> ConcatenateMatrices(List<Matrix<T>> matrices)
    {
        if (matrices.Count == 0)
            throw new ArgumentException("No matrices to concatenate");

        int totalRows = matrices.Sum(m => m.Rows);
        int cols = matrices[0].Columns;

        var result = new Matrix<T>(totalRows, cols);
        int currentRow = 0;

        foreach (var matrix in matrices)
        {
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int col = 0; col < matrix.Columns; col++)
                {
                    result[currentRow + row, col] = matrix[row, col];
                }
            }
            currentRow += matrix.Rows;
        }

        return result;
    }
}
```

---

## Common Pitfalls to Avoid

1. **Over-Augmentation**: Too much distortion makes data unrealistic
2. **Label-Breaking Transformations**: Don't flip digits "6" and "9"
3. **Test Data Augmentation**: NEVER augment test data (only training)
4. **Inconsistent Augmentation**: Same image should get different augmentations each epoch
5. **Forgetting Random Seeds**: Use seeds for reproducible experiments

---

## Testing Strategy

```csharp
[Fact]
public void ImageAugmenter_Flip_ReversesPixelOrder()
{
    // Create simple test image
    var image = new Tensor<double>(new[] { 2, 2, 1 });
    image[0, 0, 0] = 1.0; image[0, 1, 0] = 2.0;
    image[1, 0, 0] = 3.0; image[1, 1, 0] = 4.0;

    var augmenter = new ImageAugmenter<double>();
    var flipped = augmenter.Flip(image, horizontal: true);

    // Assert pixels are reversed horizontally
    Assert.Equal(2.0, flipped[0, 0, 0]);
    Assert.Equal(1.0, flipped[0, 1, 0]);
}

[Fact]
public void TextAugmenter_SynonymReplacement_PreservesLength()
{
    var augmenter = new TextAugmenter();
    string original = "good bad big small";
    string augmented = augmenter.SynonymReplacement(original, probability: 1.0);

    // Should have same number of words
    Assert.Equal(original.Split(' ').Length, augmented.Split(' ').Length);
}

[Fact]
public void TabularAugmenter_Mixup_BlendsSamples()
{
    var data = new Matrix<double>(new[,] {
        { 0.0, 0.0 },
        { 1.0, 1.0 }
    });
    var augmenter = new TabularAugmenter<double>(seed: 42);

    var mixed = augmenter.Mixup(data, sampleCount: 1);

    // Mixed sample should be between 0 and 1
    Assert.True(mixed[0, 0] >= 0.0 && mixed[0, 0] <= 1.0);
}
```

---

## Next Steps

1. Implement ImageAugmenter with geometric and color transformations
2. Implement TextAugmenter with synonym and word operations
3. Implement TabularAugmenter with noise and mixup
4. Create comprehensive unit tests for each augmentation
5. Add integration tests with training pipelines
6. Create augmentation policy classes (auto-augment)

**Estimated Effort**: 5-6 days for a junior developer

**Files to Create**:
- `src/Images/ImageAugmenter.cs`
- `src/Text/TextAugmenter.cs`
- `src/Data/TabularAugmenter.cs`
- `tests/UnitTests/Images/ImageAugmenterTests.cs`
- `tests/UnitTests/Text/TextAugmenterTests.cs`
- `tests/UnitTests/Data/TabularAugmenterTests.cs`
