using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific;

/// <summary>
/// Image patch splitter for image segmentation and patch-based learning tasks.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In image analysis, we often work with patches (small regions)
/// extracted from larger images. Adjacent patches can be highly correlated,
/// so we need to ensure patches from the same region don't appear in both train and test.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Assume each sample is a patch with (imageId, x, y) metadata
/// 2. Group patches by source image
/// 3. Either split entire images, or split with spatial buffer
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Medical image segmentation
/// - Satellite/aerial image analysis
/// - Microscopy image analysis
/// - Any patch-based computer vision task
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ImagePatchSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly int _patchSize;
    private readonly int _overlap;
    private readonly int _imageIdColumn;

    /// <summary>
    /// Creates a new image patch splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="patchSize">Size of each patch. Default is 64.</param>
    /// <param name="overlap">Overlap between adjacent patches. Default is 0.</param>
    /// <param name="imageIdColumn">Column index containing image IDs. Default is 0.</param>
    /// <param name="shuffle">Whether to shuffle images before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ImagePatchSplitter(
        double testSize = 0.2,
        int patchSize = 64,
        int overlap = 0,
        int imageIdColumn = 0,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (patchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be at least 1.");
        }

        if (overlap < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap cannot be negative.");
        }

        _testSize = testSize;
        _patchSize = patchSize;
        _overlap = overlap;
        _imageIdColumn = imageIdColumn;
    }

    /// <inheritdoc/>
    public override string Description => $"Image Patch split (patch={_patchSize}, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        if (_imageIdColumn >= nFeatures)
        {
            throw new ArgumentException(
                $"Image ID column ({_imageIdColumn}) exceeds feature count ({nFeatures}).");
        }

        // Group patches by image ID
        var imagePatches = new Dictionary<int, List<int>>();
        for (int i = 0; i < nSamples; i++)
        {
            int imageId = (int)Convert.ToDouble(X[i, _imageIdColumn]);
            if (!imagePatches.TryGetValue(imageId, out var list))
            {
                list = new List<int>();
                imagePatches[imageId] = list;
            }
            list.Add(i);
        }

        var uniqueImages = imagePatches.Keys.ToArray();
        int nImages = uniqueImages.Length;

        if (nImages < 2)
        {
            throw new ArgumentException("Need at least 2 images for splitting.");
        }

        if (_shuffle)
        {
            ShuffleIndices(uniqueImages);
        }

        int targetTestImages = Math.Max(1, (int)(nImages * _testSize));

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        for (int i = 0; i < nImages; i++)
        {
            int imageId = uniqueImages[i];
            if (i < targetTestImages)
            {
                testIndices.AddRange(imagePatches[imageId]);
            }
            else
            {
                trainIndices.AddRange(imagePatches[imageId]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
