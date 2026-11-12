using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// PuzzleMix (Kim et al., 2020) - optimal mixing using saliency and local statistics.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PuzzleMix<T> : ImageMixingAugmenterBase<T>
{
    public int GridSize { get; }

    public PuzzleMix(double alpha = 1.0, int gridSize = 4,
        double probability = 0.5) : base(probability, alpha)
    {
        if (gridSize < 1) throw new ArgumentOutOfRangeException(nameof(gridSize), "GridSize must be at least 1.");
        GridSize = gridSize;
    }

    /// <summary>
    /// Mixes two images using puzzle-style optimal transport.
    /// </summary>
    public ImageTensor<T> ApplyPuzzleMix(ImageTensor<T> image1, ImageTensor<T> image2,
        Vector<T>? labels1, Vector<T>? labels2, AugmentationContext<T> context)
    {
        if (image1.Height != image2.Height || image1.Width != image2.Width)
            image2 = new Resize<T>(image1.Height, image1.Width).Apply(image2, context);

        var result = image1.Clone();
        double lambda = SampleLambda(context);
        LastMixingLambda = NumOps.FromDouble(lambda);

        int cellH = Math.Max(1, image1.Height / GridSize);
        int cellW = Math.Max(1, image1.Width / GridSize);

        // Compute saliency per cell for both images
        var sal1 = ComputeGridSaliency(image1, cellH, cellW);
        var sal2 = ComputeGridSaliency(image2, cellH, cellW);

        // For each cell, choose the image with higher saliency,
        // respecting the overall lambda mixing ratio
        int totalCells = GridSize * GridSize;
        int cellsFromImage2 = (int)(totalCells * (1 - lambda));

        // Rank cells by relative saliency (image2 - image1), highest first; the top cells get image2
        var engine = Engine;
        var scores = engine.Reshape(engine.TensorSubtract(sal2, sal1), [totalCells]);
        var ranked = AiDotNet.Augmentation.Tabular.TabularEngineOps<T>.StableOrder(scores, descending: true);
        int chosen = Math.Max(0, Math.Min(cellsFromImage2, totalCells));
        var useImage2 = new Tensor<T>([totalCells]);
        if (chosen > 0)
        {
            var picked = new Tensor<int>(ranked.Take(chosen).ToArray(), [chosen]);
            useImage2 = engine.TensorScatter(useImage2, picked, ImageTensor<T>.Constant(new Tensor<T>([chosen]), 1.0), 0);
        }

        // Pixel mask: cells' full extents [cell * size, min(cell * size + size, extent)) along each axis
        var rowCells = CellExtent(engine, GridSize, cellH, image1.Height);
        var colCells = CellExtent(engine, GridSize, cellW, image1.Width);
        var pixelMask = engine.TensorMatMul(
            engine.TensorMatMul(engine.TensorTranspose(rowCells), engine.Reshape(useImage2, [GridSize, GridSize])), colCells);

        int channels = Math.Min(image1.Channels, image2.Channels);
        var planar1 = result.ToPlanarTensor();
        var first = ImageTensor<T>.LeadingChannels(planar1, channels);
        var second = ImageTensor<T>.LeadingChannels(image2.ToPlanarTensor(), channels);
        var mask = engine.TensorBroadcastTo(engine.Reshape(pixelMask, [1, image1.Height, image1.Width]), first._shape);
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingBlock(planar1, engine.TensorWhere(mask, second, first)));

        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1, labels2, LastMixingLambda,
                context.SampleIndex, -1, MixingStrategy.Custom);
            RaiseLabelMixing(args);
        }

        return result;
    }

    private Tensor<T> ComputeGridSaliency(ImageTensor<T> image, int cellH, int cellW)
    {
        // Sum the gradient magnitude over the interior of every cell: rows startY + 1 .. endY - 2 and
        // columns startX + 1 .. endX - 2 (those are always interior image pixels), as R * G * C^T
        var engine = Engine;
        var gradient = ImageTensor<T>.GradientMagnitude(image.ToPlanarTensor());
        var rows = CellMembership(engine, GridSize, cellH, image.Height);
        var cols = CellMembership(engine, GridSize, cellW, image.Width);
        return engine.TensorMatMul(engine.TensorMatMul(rows, gradient), engine.TensorTranspose(cols));
    }

    /// <summary>
    /// [cells, size] matrix with 1 where position p lies in [cell * cellSize, min(cell * cellSize + cellSize, size)).
    /// </summary>
    private static Tensor<T> CellExtent(IEngine engine, int cells, int cellSize, int size)
    {
        int[] shape = [cells, size];
        var cellStart = engine.Reshape(engine.TensorMultiplyScalar(engine.TensorArange<int>(0, cells, 1), cellSize), [cells, 1]);
        var start = engine.TensorBroadcastTo(cellStart, shape);
        var end = engine.TensorBroadcastTo(engine.TensorAddScalar(cellStart, cellSize), shape);
        var positions = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<int>(0, size, 1), [1, size]), shape);
        var member = engine.TensorMultiply(
            engine.ScalarMinusTensor(1, engine.TensorLessThan(positions, start)),
            engine.TensorLessThan(positions, end));
        return member.Cast<T>();
    }

    /// <summary>
    /// [cells, size] matrix with 1 where position p lies in [cell * cellSize + 1, min(cell * cellSize + cellSize, size) - 1).
    /// </summary>
    private static Tensor<T> CellMembership(IEngine engine, int cells, int cellSize, int size)
    {
        int[] shape = [cells, size];
        var cellStart = engine.Reshape(engine.TensorMultiplyScalar(engine.TensorArange<int>(0, cells, 1), cellSize), [cells, 1]);
        var first = engine.TensorBroadcastTo(engine.TensorAddScalar(cellStart, 1), shape);
        var end = engine.TensorBroadcastTo(
            engine.TensorSubtractScalar(engine.TensorClampMax(engine.TensorAddScalar(cellStart, cellSize), size), 1), shape);
        var positions = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<int>(0, size, 1), [1, size]), shape);
        var member = engine.TensorMultiply(
            engine.ScalarMinusTensor(1, engine.TensorLessThan(positions, first)),
            engine.TensorLessThan(positions, end));
        return member.Cast<T>();
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["grid_size"] = GridSize;
        return p;
    }
}
