using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific;

/// <summary>
/// Spatial splitter for geographic or coordinate-based data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When working with geographic data (like weather stations,
/// property prices by location, or ecological surveys), spatial autocorrelation
/// means nearby points are often similar. Random splitting would cause data leakage.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Divide the spatial domain into grid blocks
/// 2. Randomly assign entire blocks to train or test
/// 3. All samples within a block stay together
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Remote sensing and satellite imagery
/// - Environmental and ecological modeling
/// - Real estate price prediction
/// - Any geospatial machine learning task
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SpatialSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly int _nBlocksX;
    private readonly int _nBlocksY;
    private readonly int _xColumn;
    private readonly int _yColumn;

    /// <summary>
    /// Creates a new spatial splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="nBlocksX">Number of blocks in X direction. Default is 5.</param>
    /// <param name="nBlocksY">Number of blocks in Y direction. Default is 5.</param>
    /// <param name="xColumn">Column index for X coordinate. Default is 0.</param>
    /// <param name="yColumn">Column index for Y coordinate. Default is 1.</param>
    /// <param name="shuffle">Whether to shuffle blocks before assigning. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SpatialSplitter(
        double testSize = 0.2,
        int nBlocksX = 5,
        int nBlocksY = 5,
        int xColumn = 0,
        int yColumn = 1,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (nBlocksX < 1 || nBlocksY < 1)
        {
            throw new ArgumentOutOfRangeException("Number of blocks must be at least 1 in each direction.");
        }

        _testSize = testSize;
        _nBlocksX = nBlocksX;
        _nBlocksY = nBlocksY;
        _xColumn = xColumn;
        _yColumn = yColumn;
    }

    /// <inheritdoc/>
    public override string Description => $"Spatial split ({_nBlocksX}x{_nBlocksY} blocks, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        if (_xColumn >= nFeatures || _yColumn >= nFeatures)
        {
            throw new ArgumentException(
                $"Coordinate columns ({_xColumn}, {_yColumn}) must be within feature count ({nFeatures}).");
        }

        // Find spatial bounds
        double minX = double.MaxValue, maxX = double.MinValue;
        double minY = double.MaxValue, maxY = double.MinValue;

        for (int i = 0; i < nSamples; i++)
        {
            double x = Convert.ToDouble(X[i, _xColumn]);
            double yr = Convert.ToDouble(X[i, _yColumn]);
            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, yr);
            maxY = Math.Max(maxY, yr);
        }

        double blockWidth = (maxX - minX) / _nBlocksX;
        double blockHeight = (maxY - minY) / _nBlocksY;

        // Avoid zero-size blocks
        if (blockWidth <= 0) blockWidth = 1;
        if (blockHeight <= 0) blockHeight = 1;

        // Assign samples to blocks
        var blockIndices = new Dictionary<(int, int), List<int>>();

        for (int i = 0; i < nSamples; i++)
        {
            double x = Convert.ToDouble(X[i, _xColumn]);
            double yr = Convert.ToDouble(X[i, _yColumn]);

            int blockX = Math.Min((int)((x - minX) / blockWidth), _nBlocksX - 1);
            int blockY = Math.Min((int)((yr - minY) / blockHeight), _nBlocksY - 1);

            var blockKey = (blockX, blockY);
            if (!blockIndices.TryGetValue(blockKey, out var list))
            {
                list = new List<int>();
                blockIndices[blockKey] = list;
            }
            list.Add(i);
        }

        // Get list of non-empty blocks
        var blocks = blockIndices.Keys.ToArray();
        int nBlocks = blocks.Length;

        if (nBlocks < 2)
        {
            throw new ArgumentException(
                "Need at least 2 non-empty spatial blocks. Try reducing nBlocksX or nBlocksY.");
        }

        if (_shuffle)
        {
            // Shuffle blocks
            for (int i = nBlocks - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (blocks[i], blocks[j]) = (blocks[j], blocks[i]);
            }
        }

        int targetTestBlocks = Math.Max(1, (int)(nBlocks * _testSize));

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        for (int i = 0; i < nBlocks; i++)
        {
            var blockKey = blocks[i];
            if (i < targetTestBlocks)
            {
                testIndices.AddRange(blockIndices[blockKey]);
            }
            else
            {
                trainIndices.AddRange(blockIndices[blockKey]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
