using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Provides scanning pattern functions for Vision SSM architectures that process 2D spatial data.
/// </summary>
/// <remarks>
/// <para>
/// Vision Mamba models need to convert 2D patch grids into 1D sequences for SSM processing.
/// Different scanning patterns capture different spatial relationships:
/// </para>
/// <list type="bullet">
/// <item><description><b>Bidirectional:</b> Forward + reverse scans, used in Vision Mamba (Vim).</description></item>
/// <item><description><b>Cross-scan:</b> Four directional scans (L→R, R→L, T→B, B→T), used in VMamba.</description></item>
/// <item><description><b>Continuous:</b> Serpentine/zigzag scan preserving spatial locality, used in PlainMamba.</description></item>
/// <item><description><b>Spatio-temporal:</b> Spatial + temporal scanning for video, used in VideoMamba.</description></item>
/// </list>
/// <para><b>For Beginners:</b> When using Mamba for images instead of text, we need to turn a 2D grid of
/// image patches into a 1D sequence. The order we read the patches matters a lot!
///
/// Think of reading a page:
/// - Left-to-right, top-to-bottom (normal reading) = basic raster scan
/// - Reading both forward and backward = bidirectional scan (catches more patterns)
/// - Reading in all four directions = cross-scan (captures all spatial relationships)
/// - Reading in a zigzag pattern = continuous scan (keeps nearby patches close in sequence)
///
/// Each pattern captures different spatial relationships and works better for different tasks.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public static class ScanPatterns<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;
    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a bidirectional scan by concatenating forward and reversed sequences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used by Vision Mamba (Vim). The input patch sequence is scanned in both directions
    /// and the results are concatenated along the feature dimension. This allows the SSM to
    /// capture both left-to-right and right-to-left dependencies in the patch sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This reads the sequence of patches both forward and backward,
    /// then combines the results. It's like reading a sentence from left to right AND right to left
    /// to get a better understanding of the context from both directions.</para>
    /// </remarks>
    /// <param name="patches">Input tensor [batch, numPatches, dim].</param>
    /// <returns>Bidirectional tensor [batch, numPatches, dim * 2] with forward and reverse concatenated.</returns>
    public static Tensor<T> BidirectionalScan(Tensor<T> patches)
    {
        if (patches.Shape.Length != 3)
            throw new ArgumentException("Input must be 3D [batch, numPatches, dim].", nameof(patches));

        int batchSize = patches.Shape[0];
        int numPatches = patches.Shape[1];
        int dim = patches.Shape[2];

        // Reverse the sequence along the patch dimension
        var reversed = new Tensor<T>(new[] { batchSize, numPatches, dim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                int revP = numPatches - 1 - p;
                for (int d = 0; d < dim; d++)
                {
                    reversed[new[] { b, p, d }] = patches[new[] { b, revP, d }];
                }
            }
        }

        // Concatenate forward and reverse along feature dimension
        var result = new Tensor<T>(new[] { batchSize, numPatches, dim * 2 });
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                for (int d = 0; d < dim; d++)
                {
                    result[new[] { b, p, d }] = patches[new[] { b, p, d }];
                    result[new[] { b, p, d + dim }] = reversed[new[] { b, p, d }];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Creates four directional scans from a 2D patch grid for cross-scanning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used by VMamba. The patch grid is scanned in four directions:
    /// 1. Left-to-right, top-to-bottom (normal raster)
    /// 2. Right-to-left, bottom-to-top (reverse raster)
    /// 3. Top-to-bottom, left-to-right (column-major)
    /// 4. Bottom-to-top, right-to-left (reverse column-major)
    /// </para>
    /// <para>
    /// Each direction captures different spatial dependencies, and the four scans are processed
    /// independently by SSM blocks then merged back together.
    /// </para>
    /// <para><b>For Beginners:</b> Imagine you have a grid of image patches. This reads the grid
    /// in four different directions (like reading a crossword puzzle across and down, and their
    /// reverses). Each direction captures different spatial relationships in the image.</para>
    /// </remarks>
    /// <param name="patches">Input tensor [batch, numPatches, dim] where numPatches = height * width.</param>
    /// <param name="height">Number of patch rows in the 2D grid.</param>
    /// <param name="width">Number of patch columns in the 2D grid.</param>
    /// <returns>List of 4 tensors, each [batch, numPatches, dim], representing the four scan directions.</returns>
    /// <exception cref="ArgumentException">When height * width does not match numPatches.</exception>
    public static List<Tensor<T>> CrossScan(Tensor<T> patches, int height, int width)
    {
        if (patches.Shape.Length != 3)
            throw new ArgumentException("Input must be 3D [batch, numPatches, dim].", nameof(patches));

        int batchSize = patches.Shape[0];
        int numPatches = patches.Shape[1];
        int dim = patches.Shape[2];

        if (height * width != numPatches)
            throw new ArgumentException(
                $"height ({height}) * width ({width}) = {height * width} does not match numPatches ({numPatches}).");

        var results = new List<Tensor<T>>(4);

        // Direction 1: Left-to-right, top-to-bottom (identity - already in raster order)
        var dir1 = new Tensor<T>(new[] { batchSize, numPatches, dim });
        CopyTensor(patches, dir1);
        results.Add(dir1);

        // Direction 2: Right-to-left, bottom-to-top (reverse raster)
        var dir2 = new Tensor<T>(new[] { batchSize, numPatches, dim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                int revP = numPatches - 1 - p;
                for (int d = 0; d < dim; d++)
                {
                    dir2[new[] { b, p, d }] = patches[new[] { b, revP, d }];
                }
            }
        }
        results.Add(dir2);

        // Direction 3: Top-to-bottom, left-to-right (column-major)
        var dir3 = new Tensor<T>(new[] { batchSize, numPatches, dim });
        for (int b = 0; b < batchSize; b++)
        {
            int idx = 0;
            for (int col = 0; col < width; col++)
            {
                for (int row = 0; row < height; row++)
                {
                    int srcIdx = row * width + col;
                    for (int d = 0; d < dim; d++)
                    {
                        dir3[new[] { b, idx, d }] = patches[new[] { b, srcIdx, d }];
                    }
                    idx++;
                }
            }
        }
        results.Add(dir3);

        // Direction 4: Bottom-to-top, right-to-left (reverse column-major)
        var dir4 = new Tensor<T>(new[] { batchSize, numPatches, dim });
        for (int b = 0; b < batchSize; b++)
        {
            int idx = 0;
            for (int col = width - 1; col >= 0; col--)
            {
                for (int row = height - 1; row >= 0; row--)
                {
                    int srcIdx = row * width + col;
                    for (int d = 0; d < dim; d++)
                    {
                        dir4[new[] { b, idx, d }] = patches[new[] { b, srcIdx, d }];
                    }
                    idx++;
                }
            }
        }
        results.Add(dir4);

        return results;
    }

    /// <summary>
    /// Creates a continuous (serpentine/zigzag) scan that preserves spatial locality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used by PlainMamba. Even rows are scanned left-to-right; odd rows are scanned right-to-left.
    /// This creates a continuous path through the 2D grid that keeps spatially adjacent patches
    /// close together in the sequence, which helps the SSM maintain local spatial context.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of jumping from the end of one row to the start of the next
    /// (which is a big jump), the zigzag pattern reads each row in alternating directions. This keeps
    /// nearby patches together in the sequence, which helps the model understand local spatial patterns.</para>
    /// </remarks>
    /// <param name="patches">Input tensor [batch, numPatches, dim] where numPatches = height * width.</param>
    /// <param name="height">Number of patch rows in the 2D grid.</param>
    /// <param name="width">Number of patch columns in the 2D grid.</param>
    /// <returns>Reordered tensor [batch, numPatches, dim] with zigzag scan order.</returns>
    public static Tensor<T> ContinuousScan(Tensor<T> patches, int height, int width)
    {
        if (patches.Shape.Length != 3)
            throw new ArgumentException("Input must be 3D [batch, numPatches, dim].", nameof(patches));

        int batchSize = patches.Shape[0];
        int numPatches = patches.Shape[1];
        int dim = patches.Shape[2];

        if (height * width != numPatches)
            throw new ArgumentException(
                $"height ({height}) * width ({width}) = {height * width} does not match numPatches ({numPatches}).");

        var result = new Tensor<T>(new[] { batchSize, numPatches, dim });

        for (int b = 0; b < batchSize; b++)
        {
            int outIdx = 0;
            for (int row = 0; row < height; row++)
            {
                if (row % 2 == 0)
                {
                    // Even row: left to right
                    for (int col = 0; col < width; col++)
                    {
                        int srcIdx = row * width + col;
                        for (int d = 0; d < dim; d++)
                        {
                            result[new[] { b, outIdx, d }] = patches[new[] { b, srcIdx, d }];
                        }
                        outIdx++;
                    }
                }
                else
                {
                    // Odd row: right to left
                    for (int col = width - 1; col >= 0; col--)
                    {
                        int srcIdx = row * width + col;
                        for (int d = 0; d < dim; d++)
                        {
                            result[new[] { b, outIdx, d }] = patches[new[] { b, srcIdx, d }];
                        }
                        outIdx++;
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Creates spatio-temporal scans for video data, scanning spatially within each frame
    /// and temporally across frames.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used by VideoMamba. For video data, patches have both spatial and temporal dimensions.
    /// This method produces two scan sequences:
    /// 1. Spatial scan: patches within each frame are scanned in raster order, frames concatenated
    /// 2. Temporal scan: for each spatial position, patches across all frames are scanned
    /// </para>
    /// <para><b>For Beginners:</b> For video, we have both space (within each frame) and time (across frames).
    /// This creates two ways to scan:
    /// - Spatial: scan each frame's patches normally, then move to the next frame
    /// - Temporal: for each patch position, scan across all frames over time
    /// This captures both spatial patterns (what's in each frame) and temporal patterns (how things move).</para>
    /// </remarks>
    /// <param name="frames">Input tensor [batch, numFrames * patchesPerFrame, dim].</param>
    /// <param name="height">Number of patch rows per frame.</param>
    /// <param name="width">Number of patch columns per frame.</param>
    /// <param name="numFrames">Number of video frames.</param>
    /// <returns>List of 2 tensors: [spatial scan, temporal scan], each [batch, totalPatches, dim].</returns>
    public static List<Tensor<T>> SpatioTemporalScan(Tensor<T> frames, int height, int width, int numFrames)
    {
        if (frames.Shape.Length != 3)
            throw new ArgumentException("Input must be 3D [batch, totalPatches, dim].", nameof(frames));

        int batchSize = frames.Shape[0];
        int totalPatches = frames.Shape[1];
        int dim = frames.Shape[2];
        int patchesPerFrame = height * width;

        if (patchesPerFrame * numFrames != totalPatches)
            throw new ArgumentException(
                $"height ({height}) * width ({width}) * numFrames ({numFrames}) = {patchesPerFrame * numFrames} does not match totalPatches ({totalPatches}).");

        var results = new List<Tensor<T>>(2);

        // Spatial scan: patches within each frame in raster order, frames concatenated (identity)
        var spatialScan = new Tensor<T>(new[] { batchSize, totalPatches, dim });
        CopyTensor(frames, spatialScan);
        results.Add(spatialScan);

        // Temporal scan: for each spatial position, scan across all frames
        var temporalScan = new Tensor<T>(new[] { batchSize, totalPatches, dim });
        for (int b = 0; b < batchSize; b++)
        {
            int outIdx = 0;
            for (int sp = 0; sp < patchesPerFrame; sp++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    int srcIdx = f * patchesPerFrame + sp;
                    for (int d = 0; d < dim; d++)
                    {
                        temporalScan[new[] { b, outIdx, d }] = frames[new[] { b, srcIdx, d }];
                    }
                    outIdx++;
                }
            }
        }
        results.Add(temporalScan);

        return results;
    }

    /// <summary>
    /// Merges multiple scan outputs by averaging them element-wise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After processing the same input through multiple scan directions (e.g., cross-scan's 4 directions),
    /// this method combines the results by averaging. All scan outputs must have the same shape.
    /// </para>
    /// <para><b>For Beginners:</b> After reading the image patches in multiple directions and processing
    /// each through Mamba, we need to combine the results. Averaging is a simple but effective way to
    /// merge the information captured from all directions.</para>
    /// </remarks>
    /// <param name="scannedOutputs">List of tensors to merge, all with the same shape [batch, numPatches, dim].</param>
    /// <returns>Averaged tensor [batch, numPatches, dim].</returns>
    /// <exception cref="ArgumentException">When the list is empty or tensors have mismatched shapes.</exception>
    public static Tensor<T> MergeScanOutputs(List<Tensor<T>> scannedOutputs)
    {
        if (scannedOutputs == null || scannedOutputs.Count == 0)
            throw new ArgumentException("Must provide at least one scan output.", nameof(scannedOutputs));

        var referenceShape = scannedOutputs[0].Shape;
        for (int i = 1; i < scannedOutputs.Count; i++)
        {
            if (!ShapesEqual(scannedOutputs[i].Shape, referenceShape))
                throw new ArgumentException(
                    $"All scan outputs must have the same shape. Output {i} has shape [{string.Join(",", scannedOutputs[i].Shape)}] but expected [{string.Join(",", referenceShape)}].");
        }

        // Sum all outputs
        var result = scannedOutputs[0];
        for (int i = 1; i < scannedOutputs.Count; i++)
        {
            result = Engine.TensorAdd(result, scannedOutputs[i]);
        }

        // Divide by count to average
        T divisor = NumOps.FromDouble(scannedOutputs.Count);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = NumOps.Divide(result[i], divisor);
        }

        return result;
    }

    private static void CopyTensor(Tensor<T> src, Tensor<T> dst)
    {
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = src[i];
        }
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
}
