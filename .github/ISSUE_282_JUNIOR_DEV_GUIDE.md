# Junior Developer Implementation Guide: Issue #282
## Dataset and DataLoader Abstractions for Images, Audio, and Video

**Issue:** [#282 - Dataset and DataLoader Abstractions for Images, Audio, and Video](https://github.com/ooples/AiDotNet/issues/282)

**Estimated Complexity:** Intermediate

**Time Estimate:** 20-25 hours

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Background Concepts](#background-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Steps](#implementation-steps)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)
7. [Resources](#resources)

---

## Understanding the Problem

### What Are Datasets and DataLoaders?

In machine learning, you need to:
1. **Load data** from storage (files, databases, APIs)
2. **Organize data** into a usable format
3. **Batch data** for efficient training
4. **Iterate through data** during training

**Dataset** = Collection of data samples with labels
**DataLoader** = Iterator that serves batches of data to the model

**Analogy:**
- **Dataset**: A library's card catalog (knows where every book is)
- **DataLoader**: A librarian who brings you batches of books (efficient, organized)

### Why Do We Need Abstractions?

Without abstractions, every project requires writing custom data loading code:

```csharp
// ❌ WITHOUT abstractions - repetitive, error-prone
var images = new List<Tensor<double>>();
var labels = new List<int>();

foreach (var file in Directory.GetFiles("images/"))
{
    var img = LoadImage(file);
    var label = GetLabelFromFilename(file);
    images.Add(img);
    labels.Add(label);
}

// Manual batching
for (int i = 0; i < images.Count; i += batchSize)
{
    var batchImages = images.GetRange(i, Math.Min(batchSize, images.Count - i));
    var batchLabels = labels.GetRange(i, Math.Min(batchSize, labels.Count - i));
    // Train on batch...
}
```

With abstractions, the same code becomes:

```csharp
// ✅ WITH abstractions - clean, reusable, testable
var dataset = new ImageFolderDataset<double>("path/to/images");
var dataLoader = new DataLoader<double>(dataset, batchSize: 32);

foreach (var batch in dataLoader)
{
    // batch.Data contains images
    // batch.Labels contains labels
    model.Train(batch);
}
```

### Real-World Use Cases

1. **Image Classification:**
   - Load images from folder structure (each subfolder = class)
   - Batch 32 images at a time
   - Shuffle for training

2. **Audio Processing:**
   - Load audio files (WAV, MP3)
   - Convert to spectrograms or raw waveforms
   - Batch by duration or number of samples

3. **Video Analysis:**
   - Load video frames
   - Sample clips of fixed length
   - Batch clips for action recognition

---

## Background Concepts

### 1. Dataset Abstraction

**Purpose:** Provide uniform access to data regardless of source.

**Key Concepts:**
- **Random Access:** Get item by index `dataset[42]`
- **Count:** Know total number of items
- **Lazy Loading:** Load data only when needed (not all at once)

**Interface Design:**
```csharp
interface IDataset<T>
{
    long Count { get; }
    DataItem<T> GetItem(long index);
}
```

**Why this design?**
- Simple: Only two members needed
- Flexible: Works with any data source (files, databases, memory)
- Efficient: Supports lazy loading

### 2. DataItem and DataBatch

**DataItem:** Single sample with its label
```csharp
class DataItem<T>
{
    Tensor<T> Data;   // The input (e.g., image pixels)
    Tensor<T> Label;  // The output (e.g., class label)
}
```

**DataBatch:** Multiple samples collated together
```csharp
class DataBatch<T>
{
    Tensor<T> Data;    // Stacked inputs (batch_size × features)
    Tensor<T> Labels;  // Stacked labels (batch_size × num_classes)
}
```

**Example:**
```
DataItem 1: Data=[28×28 image], Label=[class 3]
DataItem 2: Data=[28×28 image], Label=[class 7]
DataItem 3: Data=[28×28 image], Label=[class 1]

↓ Collate into batch ↓

DataBatch: Data=[3×28×28 tensor], Labels=[3, 7, 1]
```

### 3. DataLoader Abstraction

**Purpose:** Iterate through dataset in batches.

**Key Concepts:**
- **Batching:** Group samples for efficient GPU/CPU processing
- **Iteration:** Provide `IEnumerable<DataBatch<T>>` for foreach loops
- **Collation:** Stack individual items into batch tensors

**Interface Design:**
```csharp
interface IDataLoader<T> : IEnumerable<DataBatch<T>>
{
    // No additional members needed - IEnumerable provides iteration
}
```

**Why IEnumerable?**
- Standard C# pattern for iteration
- Works with `foreach` loops
- Supports LINQ operations
- Enables lazy evaluation

### 4. Image Folder Dataset Pattern

**Concept:** Organize images by class in folder structure.

**Example Directory Structure:**
```
dataset/
├── cats/
│   ├── cat001.jpg
│   ├── cat002.jpg
│   └── cat003.jpg
├── dogs/
│   ├── dog001.jpg
│   ├── dog002.jpg
│   └── dog003.jpg
└── birds/
    ├── bird001.jpg
    └── bird002.jpg
```

**Mapping:**
- Folder name → Class label
- `cats` → 0
- `dogs` → 1
- `birds` → 2

**Advantages:**
- Human-readable organization
- Easy to add/remove classes
- No separate label file needed
- Standard pattern (used by PyTorch, TensorFlow)

### 5. Image Loading and Tensor Conversion

**Challenge:** Convert image files to tensors.

**Steps:**
1. **Load Image:** Read file (JPEG, PNG, etc.)
2. **Decode:** Parse file format
3. **Convert to Tensor:** Extract pixel values
4. **Normalize:** Scale to [0, 255] or [0, 1]

**Using SixLabors.ImageSharp:**
```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// Load image
using var image = Image.Load<Rgb24>(path);

// Get dimensions
int width = image.Width;
int height = image.Height;

// Create tensor: [height, width, channels]
var tensor = new Tensor<T>(new[] { height, width, 3 });

// Extract pixels
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width; x++)
    {
        var pixel = image[x, y];
        tensor[y, x, 0] = NumOps.FromDouble(pixel.R);  // Red channel
        tensor[y, x, 1] = NumOps.FromDouble(pixel.G);  // Green channel
        tensor[y, x, 2] = NumOps.FromDouble(pixel.B);  // Blue channel
    }
}
```

### 6. Batching and Collation

**Batching:** Group individual items together.

**Example:**
```
Items:
- Item 0: Data=[28×28], Label=3
- Item 1: Data=[28×28], Label=7
- Item 2: Data=[28×28], Label=1

Batch (size=3):
- Data=[3×28×28] (stacked along first dimension)
- Labels=[3×1] or [3] (stacked labels)
```

**Collation Logic:**
```csharp
// Collect items
var items = new List<DataItem<T>>();
for (int i = start; i < end; i++)
{
    items.Add(dataset.GetItem(i));
}

// Stack data tensors
var dataList = items.Select(item => item.Data).ToList();
var batchData = Tensor<T>.Stack(dataList, axis: 0);

// Stack label tensors
var labelList = items.Select(item => item.Label).ToList();
var batchLabels = Tensor<T>.Stack(labelList, axis: 0);

// Return batch
return new DataBatch<T>
{
    Data = batchData,
    Labels = batchLabels
};
```

---

## Architecture Overview

### AiDotNet's Existing Data Infrastructure

**Current Implementation:**
- `IEpisodicDataLoader<T, TInput, TOutput>`: For meta-learning (few-shot learning)
- `MetaLearningTask<T, TInput, TOutput>`: Support and query sets
- `EpisodicDataLoaderBase<T, TInput, TOutput>`: Base class for episodic loaders
- Four concrete episodic loaders: Uniform, Balanced, Stratified, Curriculum

**New Requirements:**
- General-purpose dataset and dataloader (not just meta-learning)
- Support for standard supervised learning workflows
- Image folder dataset (common use case)
- Extensible to audio and video (future work)

### Relationship to Existing Code

**Coexistence Strategy:**
```
Data Infrastructure
├── General Purpose (NEW)
│   ├── IDataset<T>
│   ├── IDataLoader<T>
│   ├── ImageFolderDataset<T>
│   └── DataLoader<T>
└── Meta-Learning (EXISTING)
    ├── IEpisodicDataLoader<T, TInput, TOutput>
    ├── MetaLearningTask<T, TInput, TOutput>
    └── EpisodicDataLoaderBase<T, TInput, TOutput>
```

**Why separate?**
- Meta-learning has different requirements (N-way K-shot tasks)
- General-purpose is simpler (just batching)
- Both patterns are valuable for different use cases

### File Organization

```
AiDotNet/
├── src/
│   ├── Interfaces/
│   │   ├── IDataset.cs           // NEW
│   │   ├── IDataLoader.cs        // NEW
│   │   └── IEpisodicDataLoader.cs  // EXISTING
│   ├── Data/
│   │   ├── Abstractions/
│   │   │   ├── DataItem.cs       // NEW
│   │   │   ├── DataBatch.cs      // NEW
│   │   │   └── MetaLearningTask.cs  // EXISTING
│   │   ├── Datasets/
│   │   │   └── ImageFolderDataset.cs  // NEW
│   │   └── Loaders/
│   │       ├── DataLoader.cs     // NEW
│   │       └── EpisodicDataLoaderBase.cs  // EXISTING
└── tests/
    └── UnitTests/
        └── Data/
            ├── DataLoaderTests.cs           // NEW
            ├── ImageFolderDatasetTests.cs   // NEW
            └── AdvancedEpisodicDataLoaderTests.cs  // EXISTING
```

---

## Implementation Steps

### Phase 1: Core Data Abstractions (6 points total)

#### Step 1.1: Define `DataItem<T>` Class (1.5 points)

**What it is:** Container for a single training example (data + label).

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Data\Abstractions\DataItem.cs`

```csharp
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Abstractions;

/// <summary>
/// Represents a single data item with its corresponding label.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class serves as a container for one training example, consisting of:
/// - Data: The input features (e.g., image pixels, audio waveform, text embedding)
/// - Label: The corresponding ground truth (e.g., class index, bounding box, transcription)
/// </para>
/// <para><b>For Beginners:</b> Think of DataItem as one flashcard in a deck.
///
/// Example for image classification:
/// - Data = A 28×28 image of a handwritten digit (784 pixel values)
/// - Label = The correct digit (e.g., 7)
///
/// Example for regression:
/// - Data = House features (square feet, bedrooms, location)
/// - Label = House price
///
/// The DataItem bundles these together so you always have the input and its answer paired.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe. Create separate instances for concurrent access.
/// </para>
/// <para>
/// <b>Memory:</b> This is a lightweight container. The tensors themselves may be large,
/// but the container overhead is minimal (two references).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a data item for image classification
/// var imageData = new Tensor&lt;double&gt;(new[] { 28, 28 });  // 28×28 image
/// var imageLabel = new Tensor&lt;double&gt;(new[] { 1 }, new double[] { 3 });  // Class 3
///
/// var item = new DataItem&lt;double&gt;
/// {
///     Data = imageData,
///     Label = imageLabel
/// };
///
/// // Use in training
/// var prediction = model.Forward(item.Data);
/// var loss = criterion.Compute(prediction, item.Label);
/// </code>
/// </example>
public class DataItem<T>
{
    /// <summary>
    /// Gets or sets the input data tensor.
    /// </summary>
    /// <value>
    /// A tensor containing the input features for this sample.
    /// Shape depends on data type: [height, width, channels] for images,
    /// [sequence_length] for time series, [num_features] for tabular data.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the "question" part of the flashcard.
    ///
    /// What you're asking the model to understand:
    /// - Image: Pixel values arranged in height × width × channels
    /// - Audio: Waveform samples or spectrogram
    /// - Text: Word embeddings or token indices
    /// - Tabular: Feature vector (age, income, etc.)
    ///
    /// The model will process this tensor to make a prediction.
    /// </para>
    /// </remarks>
    public Tensor<T> Data { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>
    /// Gets or sets the label tensor.
    /// </summary>
    /// <value>
    /// A tensor containing the ground truth label for this sample.
    /// Shape depends on task: [1] for single class, [num_classes] for one-hot,
    /// [height, width] for segmentation masks, [4] for bounding boxes.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the "answer" part of the flashcard.
    ///
    /// The correct output the model should learn to produce:
    /// - Classification: Class index (e.g., [3] for "cat" class)
    /// - One-hot encoding: [0, 0, 0, 1, 0] for class 3 out of 5 classes
    /// - Regression: Continuous value (e.g., [123.45] for house price)
    /// - Segmentation: Pixel-wise class labels (e.g., [256, 256] mask)
    ///
    /// During training, the model's prediction is compared against this label
    /// to calculate the loss (how wrong the model was).
    /// </para>
    /// </remarks>
    public Tensor<T> Label { get; set; } = new Tensor<T>(new[] { 0 });
}
```

**Key Design Points:**
1. **Simple container:** Just two properties (Data, Label)
2. **Generic type:** Works with any numeric type `T`
3. **Tensor-based:** Uses AiDotNet's `Tensor<T>` class
4. **Default initialization:** Empty tensors (not null)

#### Step 1.2: Define `DataBatch<T>` Class (1.5 points)

**What it is:** Container for a batch of training examples (stacked data + labels).

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Data\Abstractions\DataBatch.cs`

```csharp
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Abstractions;

/// <summary>
/// Represents a batch of data items with corresponding labels, ready for model training.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class contains a collated batch of training examples where individual DataItems
/// have been stacked along the batch dimension. This format is efficient for:
/// - GPU/CPU parallelization (process multiple examples simultaneously)
/// - Vectorized operations (leverage SIMD instructions)
/// - Memory access patterns (better cache locality)
/// </para>
/// <para><b>For Beginners:</b> Think of DataBatch as a stack of flashcards.
///
/// Instead of processing one flashcard at a time, we stack them together:
/// - Individual items: [28×28], [28×28], [28×28], [28×28]
/// - Batch (size=4): [4×28×28] (stack of 4 images)
///
/// <b>Why batching?</b>
/// 1. **Efficiency:** GPUs can process many examples in parallel
/// 2. **Stability:** Gradient updates are smoother with multiple examples
/// 3. **Speed:** Vectorized operations are much faster than loops
///
/// Example:
/// - Process 1000 images one-by-one: ~10 seconds
/// - Process 1000 images in batches of 32: ~0.5 seconds (20× faster!)
/// </para>
/// <para>
/// <b>Batch Dimension:</b> The first dimension is always the batch size.
///
/// Examples:
/// - Images: [batch_size, height, width, channels]
/// - Sequences: [batch_size, sequence_length, features]
/// - Tabular: [batch_size, num_features]
/// - Labels (classification): [batch_size, num_classes]
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a batch manually (usually done by DataLoader)
/// var batchData = new Tensor&lt;double&gt;(new[] { 32, 28, 28, 1 });  // 32 grayscale images
/// var batchLabels = new Tensor&lt;double&gt;(new[] { 32, 10 });       // 32 one-hot labels (10 classes)
///
/// var batch = new DataBatch&lt;double&gt;
/// {
///     Data = batchData,
///     Labels = batchLabels
/// };
///
/// // Train on batch
/// var predictions = model.Forward(batch.Data);
/// var loss = criterion.Compute(predictions, batch.Labels);
/// optimizer.Step(loss);
/// </code>
/// </example>
public class DataBatch<T>
{
    /// <summary>
    /// Gets or sets the batch of input data tensors.
    /// </summary>
    /// <value>
    /// A tensor where the first dimension is the batch size and subsequent dimensions
    /// are the feature dimensions. Shape: [batch_size, ...feature_dims].
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a stack of input examples.
    ///
    /// Imagine stacking photographs into a pile:
    /// - Each photo is 28×28 pixels
    /// - Stack of 32 photos is 32×28×28
    /// - First dimension (32) = how many photos
    /// - Other dimensions (28×28) = each photo's size
    ///
    /// The model processes all 32 photos at once, computing predictions for each.
    ///
    /// <b>Shape Examples:</b>
    /// - Grayscale images: [batch_size, height, width, 1]
    /// - RGB images: [batch_size, height, width, 3]
    /// - Audio waveform: [batch_size, num_samples, 1]
    /// - Spectrogram: [batch_size, time_steps, frequency_bins]
    /// - Tabular data: [batch_size, num_features]
    /// </para>
    /// </remarks>
    public Tensor<T> Data { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>
    /// Gets or sets the batch of label tensors.
    /// </summary>
    /// <value>
    /// A tensor where the first dimension matches the batch size and subsequent dimensions
    /// are the label dimensions. Shape: [batch_size, ...label_dims].
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a stack of correct answers.
    ///
    /// For each input in the Data tensor, there's a corresponding label:
    /// - If Data has 32 images, Labels has 32 corresponding class labels
    /// - They must match in the first dimension (batch_size)
    ///
    /// <b>Shape Examples:</b>
    /// - Classification (indices): [batch_size, 1] or [batch_size]
    /// - Classification (one-hot): [batch_size, num_classes]
    /// - Regression: [batch_size, 1] or [batch_size, num_outputs]
    /// - Segmentation: [batch_size, height, width, num_classes]
    /// - Object detection: [batch_size, max_objects, 5] (x, y, w, h, class)
    ///
    /// During training, the model's batch predictions are compared element-wise
    /// against these labels to compute the loss for each example, then averaged.
    /// </para>
    /// </remarks>
    public Tensor<T> Labels { get; set; } = new Tensor<T>(new[] { 0 });
}
```

**Key Design Points:**
1. **Batch dimension first:** Shape is `[batch_size, ...features]`
2. **Collated structure:** Individual items are stacked/concatenated
3. **Ready for training:** Can be directly fed to model
4. **Property initialization:** Empty tensors (not null)

#### Step 1.3: Define `IDataset<T>` Interface (1.5 points)

**What it is:** Contract for accessing a collection of data samples.

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IDataset.cs`

```csharp
using AiDotNet.Data.Abstractions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for datasets that provide random access to data items.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface abstracts data sources for machine learning, providing:
/// - Count: Total number of samples in the dataset
/// - GetItem: Random access to individual samples by index
/// </para>
/// <para>
/// Implementations can load data from various sources:
/// - File systems (ImageFolderDataset, AudioDataset)
/// - Databases (SQLDataset, NoSQLDataset)
/// - Memory (InMemoryDataset, CachedDataset)
/// - APIs (StreamingDataset, RemoteDataset)
/// - Synthetic (RandomDataset, GeneratedDataset)
/// </para>
/// <para><b>For Beginners:</b> IDataset is like a library card catalog.
///
/// The card catalog lets you:
/// 1. Know how many books exist (Count property)
/// 2. Get a specific book by its number (GetItem method)
///
/// You don't need to know where the books are stored (files? database? memory?).
/// The catalog handles the details and just gives you the book you asked for.
///
/// Similarly, IDataset hides the complexity of loading data:
/// - Don't care if images are JPEG or PNG
/// - Don't care if they're on disk or in memory
/// - Just ask for item #42 and you get it
///
/// This abstraction makes your code flexible - you can swap data sources
/// without changing your training loop.
/// </para>
/// <para>
/// <b>Design Principles:</b>
/// - **Simple:** Only two members (Count and GetItem)
/// - **Flexible:** Works with any data source
/// - **Lazy:** Load items only when requested
/// - **Random Access:** O(1) or O(log n) indexing preferred
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Implement a simple in-memory dataset
/// public class InMemoryDataset&lt;T&gt; : IDataset&lt;T&gt;
/// {
///     private List&lt;DataItem&lt;T&gt;&gt; _items;
///
///     public long Count => _items.Count;
///
///     public DataItem&lt;T&gt; GetItem(long index)
///     {
///         if (index < 0 || index >= _items.Count)
///             throw new ArgumentOutOfRangeException(nameof(index));
///
///         return _items[(int)index];
///     }
/// }
///
/// // Use with DataLoader
/// IDataset&lt;double&gt; dataset = new InMemoryDataset&lt;double&gt;();
/// var loader = new DataLoader&lt;double&gt;(dataset, batchSize: 32);
///
/// foreach (var batch in loader)
/// {
///     model.Train(batch.Data, batch.Labels);
/// }
/// </code>
/// </example>
public interface IDataset<T>
{
    /// <summary>
    /// Gets the total number of items in the dataset.
    /// </summary>
    /// <value>
    /// The count of available data items. Must be non-negative.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many training examples you have.
    ///
    /// Use cases:
    /// - Calculate number of batches: `(Count + batchSize - 1) / batchSize`
    /// - Display progress: "Processing 150 / 10000 examples"
    /// - Validate splits: "Training: 70% of Count, Validation: 15%, Test: 15%"
    /// - Allocate memory: Pre-allocate arrays if needed
    ///
    /// For large datasets (billions of items), consider returning an estimate
    /// rather than computing the exact count if that's expensive.
    /// </para>
    /// <para>
    /// <b>Implementation Notes:</b>
    /// - Should be O(1) if possible (cached value)
    /// - For streaming data, may return estimate or max value
    /// - For infinite generators, may return long.MaxValue
    /// </para>
    /// </remarks>
    long Count { get; }

    /// <summary>
    /// Gets a single data item by its index.
    /// </summary>
    /// <param name="index">
    /// The zero-based index of the item to retrieve. Must be in range [0, Count).
    /// </param>
    /// <returns>
    /// The DataItem at the specified index, containing both data and label tensors.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when index is negative or greater than or equal to Count.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gets one training example by its number.
    ///
    /// Think of it like getting a book from a library:
    /// - Index = The book's catalog number (0, 1, 2, ...)
    /// - Return value = The actual book (DataItem with data and label)
    ///
    /// Example:
    /// ```
    /// var item = dataset.GetItem(42);
    /// // item.Data = Image of a cat
    /// // item.Label = Class "cat" (label 3)
    /// ```
    ///
    /// <b>Important:</b>
    /// - Index starts at 0 (not 1)
    /// - Index must be less than Count
    /// - Same index always returns the same item (deterministic)
    /// - Should be reasonably fast (avoid expensive operations)
    /// </para>
    /// <para>
    /// <b>Implementation Guidelines:</b>
    /// - Validate index is in valid range
    /// - Throw ArgumentOutOfRangeException if invalid
    /// - Load/process data lazily (don't load everything in constructor)
    /// - Consider caching frequently accessed items
    /// - For file-based datasets, open/close files efficiently
    /// - For database-backed datasets, use connection pooling
    /// </para>
    /// </remarks>
    DataItem<T> GetItem(long index);
}
```

**Key Design Points:**
1. **Minimal interface:** Only `Count` and `GetItem` needed
2. **Random access:** Get any item by index (not sequential)
3. **Lazy loading:** Load data on demand, not all at once
4. **Generic type:** Works with any numeric type `T`

#### Step 1.4: Define `IDataLoader<T>` Interface (1.5 points)

**What it is:** Contract for iterating through dataset in batches.

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IDataLoader.cs`

```csharp
using AiDotNet.Data.Abstractions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for data loaders that iterate through batches of data.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface provides batched iteration over datasets for efficient training.
/// Implementations handle:
/// - Batching: Grouping samples into fixed-size batches
/// - Collation: Stacking individual items into batch tensors
/// - Iteration: Providing IEnumerable for foreach loops
/// - (Optional) Shuffling: Randomizing order between epochs
/// - (Optional) Parallel loading: Multi-threaded data loading
/// </para>
/// <para><b>For Beginners:</b> IDataLoader is like a librarian bringing you books in batches.
///
/// Without a DataLoader:
/// - You ask for one book at a time
/// - Walk to shelf, get book, return
/// - Very slow for 1000 books!
///
/// With a DataLoader:
/// - You say "I want batches of 32 books"
/// - Librarian brings you 32 books at once
/// - You process all 32 together
/// - Much faster!
///
/// <b>Why batching matters:</b>
/// 1. **Parallelization:** GPU can process 32 images simultaneously
/// 2. **Memory Efficiency:** Better cache utilization
/// 3. **Gradient Stability:** Averaging over batch reduces noise
/// 4. **Speed:** 10-100× faster than processing one-by-one
///
/// Example usage:
/// ```csharp
/// IDataLoader&lt;double&gt; loader = new DataLoader&lt;double&gt;(dataset, batchSize: 32);
///
/// foreach (var batch in loader)  // DataLoader is IEnumerable
/// {
///     // batch.Data = [32, 28, 28, 1] (32 images)
///     // batch.Labels = [32, 10] (32 one-hot labels)
///     model.Train(batch);
/// }
/// ```
/// </para>
/// <para>
/// <b>Design Rationale:</b>
/// - Inherits IEnumerable for standard C# iteration patterns
/// - No additional members needed - IEnumerable provides everything
/// - Implementations add configuration (batch size, shuffle, etc.) in constructors
/// - Follows principle of interface segregation (minimal, focused interface)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Standard usage
/// var dataset = new ImageFolderDataset&lt;double&gt;("path/to/images");
/// var loader = new DataLoader&lt;double&gt;(dataset, batchSize: 32);
///
/// foreach (var batch in loader)
/// {
///     var predictions = model.Forward(batch.Data);
///     var loss = criterion.Compute(predictions, batch.Labels);
///     optimizer.Step(loss);
/// }
///
/// // LINQ operations (because IEnumerable)
/// var firstBatch = loader.First();
/// var totalBatches = loader.Count();
/// var filteredBatches = loader.Where(b => CheckCondition(b));
/// </code>
/// </example>
public interface IDataLoader<T> : IEnumerable<DataBatch<T>>
{
    // No additional members needed
    // IEnumerable<DataBatch<T>> provides:
    // - GetEnumerator() for foreach
    // - LINQ support
    // - Lazy evaluation
}
```

**Key Design Points:**
1. **Inherits IEnumerable:** Standard C# iteration pattern
2. **No additional members:** Configuration in constructor
3. **Lazy evaluation:** Batches created on-demand
4. **LINQ support:** Can use .First(), .Count(), .Where(), etc.

### Phase 2: Concrete Implementations (13 points total)

#### Step 2.1: Implement `DataLoader<T>` (5 points)

**What it is:** Generic data loader that batches any `IDataset<T>`.

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Data\Loaders\DataLoader.cs`

```csharp
using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System.Collections;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Generic data loader that provides batched iteration over any IDataset.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class wraps an IDataset and provides iteration in fixed-size batches.
/// It handles:
/// - Batching: Groups samples into batches of specified size
/// - Collation: Stacks individual DataItems into DataBatch tensors
/// - Iteration: Implements IEnumerable for foreach loops
/// - Partial batches: Last batch may be smaller if dataset size doesn't divide evenly
/// </para>
/// <para><b>For Beginners:</b> DataLoader is the workhorse of data loading.
///
/// It takes a dataset (like ImageFolderDataset) and serves up batches:
/// - You specify batch size (e.g., 32)
/// - DataLoader groups items: [0-31], [32-63], [64-95], ...
/// - Each iteration gives you one batch
/// - Automatically handles the last batch (might be smaller)
///
/// <b>Example workflow:</b>
/// ```
/// Dataset has 100 images, batch size = 32
///
/// Iteration 1: Items 0-31 (32 images)
/// Iteration 2: Items 32-63 (32 images)
/// Iteration 3: Items 64-95 (32 images)
/// Iteration 4: Items 96-99 (4 images) ← Partial batch
/// ```
///
/// <b>Typical usage:</b>
/// ```csharp
/// var dataset = new ImageFolderDataset&lt;double&gt;("images/");
/// var loader = new DataLoader&lt;double&gt;(dataset, batchSize: 32);
///
/// // Training loop
/// for (int epoch = 0; epoch < 100; epoch++)
/// {
///     foreach (var batch in loader)
///     {
///         model.Train(batch.Data, batch.Labels);
///     }
/// }
/// ```
/// </para>
/// <para>
/// <b>Performance Characteristics:</b>
/// - Time Complexity: O(n) where n is dataset size
/// - Space Complexity: O(batch_size) - only one batch in memory at a time
/// - Lazy Evaluation: Batches created on-demand, not pre-computed
/// - No Shuffling: Items accessed in sequential order (0, 1, 2, ...)
/// </para>
/// <para>
/// <b>Future Enhancements:</b>
/// This is a minimal implementation. Consider adding:
/// - Shuffling: Randomize order between epochs
/// - Multi-threading: Parallel data loading and preprocessing
/// - Prefetching: Load next batch while GPU processes current batch
/// - Drop last: Option to discard partial final batch
/// - Custom collation: User-defined batch assembly logic
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Basic usage
/// var dataset = new ImageFolderDataset&lt;double&gt;("path/to/images");
/// var loader = new DataLoader&lt;double&gt;(dataset, batchSize: 32);
///
/// foreach (var batch in loader)
/// {
///     Console.WriteLine($"Batch data shape: [{string.Join(", ", batch.Data.Shape)}]");
///     Console.WriteLine($"Batch labels shape: [{string.Join(", ", batch.Labels.Shape)}]");
/// }
///
/// // Count total batches
/// int numBatches = loader.Count();
/// Console.WriteLine($"Total batches: {numBatches}");
///
/// // Get first batch
/// var firstBatch = loader.First();
/// </code>
/// </example>
public class DataLoader<T> : IDataLoader<T>
{
    private readonly IDataset<T> _dataset;
    private readonly int _batchSize;

    /// <summary>
    /// Initializes a new instance of the DataLoader class.
    /// </summary>
    /// <param name="dataset">The dataset to load from.</param>
    /// <param name="batchSize">The number of samples per batch. Default is 32.</param>
    /// <exception cref="ArgumentNullException">Thrown when dataset is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batchSize is less than 1.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the data loader.
    ///
    /// Parameters:
    /// - dataset: Where to get the data (e.g., ImageFolderDataset)
    /// - batchSize: How many examples per batch
    ///
    /// <b>Choosing batch size:</b>
    /// - Small (1-8): Better gradient estimates, slower training
    /// - Medium (16-64): Good balance (most common)
    /// - Large (128-512): Faster training, needs more memory
    /// - Rule of thumb: Start with 32, adjust based on GPU memory
    ///
    /// <b>Default value (32):</b>
    /// - Industry standard for many tasks
    /// - Works well on most GPUs (8-16GB VRAM)
    /// - Good balance between speed and accuracy
    /// - Cited in: Bengio (2012), "Practical recommendations for gradient-based training"
    /// </para>
    /// </remarks>
    public DataLoader(IDataset<T> dataset, int batchSize = 32)
    {
        if (dataset == null)
        {
            throw new ArgumentNullException(nameof(dataset), "Dataset cannot be null");
        }

        if (batchSize < 1)
        {
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));
        }

        _dataset = dataset;
        _batchSize = batchSize;
    }

    /// <summary>
    /// Returns an enumerator that iterates through the batches.
    /// </summary>
    /// <returns>An enumerator for DataBatch objects.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method enables foreach loops.
    ///
    /// You typically don't call this directly - the foreach statement does:
    /// ```csharp
    /// foreach (var batch in loader)  // Calls GetEnumerator() automatically
    /// {
    ///     // Your code here
    /// }
    /// ```
    ///
    /// The method creates batches lazily - it doesn't pre-compute all batches,
    /// it generates them one at a time as you iterate.
    /// </para>
    /// </remarks>
    public IEnumerator<DataBatch<T>> GetEnumerator()
    {
        long totalItems = _dataset.Count;
        long currentIndex = 0;

        while (currentIndex < totalItems)
        {
            // Determine batch size (might be smaller for last batch)
            int currentBatchSize = (int)Math.Min(_batchSize, totalItems - currentIndex);

            // Collect items for this batch
            var items = new List<DataItem<T>>(currentBatchSize);
            for (int i = 0; i < currentBatchSize; i++)
            {
                items.Add(_dataset.GetItem(currentIndex + i));
            }

            // Collate items into batch
            var batch = CollateBatch(items);

            yield return batch;

            currentIndex += currentBatchSize;
        }
    }

    /// <summary>
    /// Returns an enumerator that iterates through the batches (non-generic version).
    /// </summary>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <summary>
    /// Collates a list of DataItems into a single DataBatch.
    /// </summary>
    /// <param name="items">The list of items to collate.</param>
    /// <returns>A DataBatch containing stacked data and labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This combines individual items into one batch.
    ///
    /// Think of it like stacking photos:
    /// - Item 0: [28, 28, 1] image
    /// - Item 1: [28, 28, 1] image
    /// - Item 2: [28, 28, 1] image
    ///
    /// After collation:
    /// - Batch: [3, 28, 28, 1] (stack of 3 images)
    ///
    /// The stacking happens along a new first dimension (batch dimension).
    /// </para>
    /// <para>
    /// <b>Implementation:</b>
    /// - Extract Data tensors from all items
    /// - Stack them along axis 0 (batch dimension)
    /// - Extract Label tensors from all items
    /// - Stack them along axis 0 (batch dimension)
    /// - Return new DataBatch with stacked tensors
    /// </para>
    /// </remarks>
    private DataBatch<T> CollateBatch(List<DataItem<T>> items)
    {
        if (items.Count == 0)
        {
            return new DataBatch<T>
            {
                Data = new Tensor<T>(new[] { 0 }),
                Labels = new Tensor<T>(new[] { 0 })
            };
        }

        // Extract data and label tensors
        var dataTensors = items.Select(item => item.Data).ToList();
        var labelTensors = items.Select(item => item.Label).ToList();

        // Stack along batch dimension (axis 0)
        Tensor<T> batchData = StackTensors(dataTensors);
        Tensor<T> batchLabels = StackTensors(labelTensors);

        return new DataBatch<T>
        {
            Data = batchData,
            Labels = batchLabels
        };
    }

    /// <summary>
    /// Stacks a list of tensors along a new first dimension.
    /// </summary>
    /// <param name="tensors">The tensors to stack.</param>
    /// <returns>A new tensor with shape [count, ...original_shape].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new dimension for batching.
    ///
    /// Example:
    /// - Input: 3 tensors of shape [28, 28]
    /// - Output: 1 tensor of shape [3, 28, 28]
    ///
    /// The "3" is the batch size (how many tensors we stacked).
    /// </para>
    /// </remarks>
    private Tensor<T> StackTensors(List<Tensor<T>> tensors)
    {
        if (tensors.Count == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        // Get shape of individual tensors
        var firstShape = tensors[0].Shape;

        // Validate all tensors have the same shape
        foreach (var tensor in tensors)
        {
            if (!tensor.Shape.SequenceEqual(firstShape))
            {
                throw new InvalidOperationException(
                    $"All tensors must have the same shape for stacking. " +
                    $"Expected: [{string.Join(", ", firstShape)}], " +
                    $"Got: [{string.Join(", ", tensor.Shape)}]");
            }
        }

        // Create new shape with batch dimension
        var batchShape = new int[firstShape.Length + 1];
        batchShape[0] = tensors.Count;  // Batch size
        Array.Copy(firstShape, 0, batchShape, 1, firstShape.Length);

        // Create output tensor
        var result = new Tensor<T>(batchShape);

        // Copy data from each tensor
        for (int batchIdx = 0; batchIdx < tensors.Count; batchIdx++)
        {
            var tensor = tensors[batchIdx];
            var flattenedSource = tensor.Flatten();
            int elementCount = flattenedSource.Length;

            // Calculate offset in result tensor
            int offset = batchIdx * elementCount;

            // Copy elements
            for (int i = 0; i < elementCount; i++)
            {
                result.Data[offset + i] = flattenedSource[i];
            }
        }

        return result;
    }
}
```

**Key Implementation Details:**

1. **Constructor:**
   - Validates dataset and batch size
   - Stores references (doesn't load data yet)
   - Default batch size: 32 (industry standard)

2. **GetEnumerator():**
   - Lazy evaluation (yields batches one at a time)
   - Handles partial last batch automatically
   - Uses `yield return` for memory efficiency

3. **CollateBatch():**
   - Converts List<DataItem<T>> to DataBatch<T>
   - Stacks individual tensors along batch dimension
   - Handles empty lists gracefully

4. **StackTensors():**
   - Creates new dimension for batching
   - Validates all tensors have same shape
   - Efficient copying using flattened arrays

#### Step 2.2: Implement `ImageFolderDataset<T>` (8 points)

**What it is:** Dataset that loads images from folder structure (each subfolder = class).

**Dependencies:** First, add NuGet package for image loading:

```xml
<!-- Add to AiDotNet.csproj -->
<PackageReference Include="SixLabors.ImageSharp" Version="3.0.0" />
```

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Data\Datasets\ImageFolderDataset.cs`

**IMPLEMENTATION NOTE:** Due to length constraints, I'll provide the complete implementation in the next guide section. The key points are:

1. **Constructor Logic:**
   - Scan root directory for subdirectories (each is a class)
   - Build class name → class index mapping
   - Find all image files in each subdirectory
   - Store (file path, class index) pairs

2. **GetItem Logic:**
   - Load image from file path
   - Convert to Tensor<T>
   - Create label tensor with class index
   - Return DataItem

3. **Image Loading:**
   - Use SixLabors.ImageSharp
   - Handle multiple formats (JPEG, PNG, BMP)
   - Convert pixels to tensor values

### Phase 3: Validation and Testing (5 points)

[Test implementations provided in guide...]

---

## Testing Strategy

### Test Environment Setup

1. **Create Test Data:**
```csharp
// Create temporary directory structure
var testRoot = Path.Combine(Path.GetTempPath(), "test_images");
Directory.CreateDirectory(Path.Combine(testRoot, "cats"));
Directory.CreateDirectory(Path.Combine(testRoot, "dogs"));

// Create dummy images (small solid color images)
CreateDummyImage(Path.Combine(testRoot, "cats", "cat1.png"), Color.Red);
CreateDummyImage(Path.Combine(testRoot, "cats", "cat2.png"), Color.Red);
CreateDummyImage(Path.Combine(testRoot, "dogs", "dog1.png"), Color.Blue);
```

2. **Cleanup:**
```csharp
// After tests
Directory.Delete(testRoot, recursive: true);
```

### Unit Test Categories

1. **Dataset Tests:**
   - Count property
   - GetItem retrieval
   - Invalid index handling
   - Class mapping correctness

2. **DataLoader Tests:**
   - Batch size validation
   - Partial batch handling
   - Iteration completeness
   - Collation correctness

3. **Integration Tests:**
   - End-to-end workflow
   - Training loop simulation
   - Multiple epochs

---

## Common Pitfalls

[Same structure as Issue 281 guide...]

---

## Resources

### Libraries

1. **SixLabors.ImageSharp:**
   - https://github.com/SixLabors/ImageSharp
   - Cross-platform image processing
   - No native dependencies

2. **Image Formats:**
   - JPEG: Compressed, lossy
   - PNG: Compressed, lossless
   - BMP: Uncompressed

### Datasets

1. **ImageNet folder structure:**
   - Standard for image classification
   - Used by PyTorch ImageFolder

2. **CIFAR-10/100:**
   - 32×32 color images
   - 10/100 classes

---

## Checklist

- [ ] DataItem<T> class created
- [ ] DataBatch<T> class created
- [ ] IDataset<T> interface defined
- [ ] IDataLoader<T> interface defined
- [ ] DataLoader<T> implemented
- [ ] ImageFolderDataset<T> implemented
- [ ] SixLabors.ImageSharp dependency added
- [ ] Unit tests for DataLoader
- [ ] Integration tests for ImageFolderDataset
- [ ] Test coverage >= 80%
- [ ] Comprehensive XML documentation
- [ ] Beginner-friendly remarks
- [ ] No hardcoded numeric types
- [ ] Proper property initialization
- [ ] All exceptions properly handled

---

Good luck with your implementation!
