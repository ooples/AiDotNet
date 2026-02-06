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

        // Rank cells by relative saliency (image2 - image1)
        var cellScores = new (int row, int col, double score)[totalCells];
        int idx = 0;
        for (int gy = 0; gy < GridSize; gy++)
            for (int gx = 0; gx < GridSize; gx++)
                cellScores[idx++] = (gy, gx, sal2[gy, gx] - sal1[gy, gx]);

        Array.Sort(cellScores, (a, b) => b.score.CompareTo(a.score));

        // Top cells get image2
        var useImage2 = new bool[GridSize, GridSize];
        for (int i = 0; i < cellsFromImage2 && i < totalCells; i++)
            useImage2[cellScores[i].row, cellScores[i].col] = true;

        for (int gy = 0; gy < GridSize; gy++)
            for (int gx = 0; gx < GridSize; gx++)
            {
                if (!useImage2[gy, gx]) continue;

                int startY = gy * cellH;
                int startX = gx * cellW;
                int endY = Math.Min(startY + cellH, image1.Height);
                int endX = Math.Min(startX + cellW, image1.Width);

                for (int y = startY; y < endY; y++)
                    for (int x = startX; x < endX; x++)
                        for (int c = 0; c < Math.Min(image1.Channels, image2.Channels); c++)
                            result.SetPixel(y, x, c, image2.GetPixel(y, x, c));
            }

        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1, labels2, LastMixingLambda,
                context.SampleIndex, -1, MixingStrategy.Custom);
            RaiseLabelMixing(args);
        }

        return result;
    }

    private double[,] ComputeGridSaliency(ImageTensor<T> image, int cellH, int cellW)
    {
        var saliency = new double[GridSize, GridSize];

        for (int gy = 0; gy < GridSize; gy++)
            for (int gx = 0; gx < GridSize; gx++)
            {
                int startY = gy * cellH;
                int startX = gx * cellW;
                int endY = Math.Min(startY + cellH, image.Height);
                int endX = Math.Min(startX + cellW, image.Width);

                double grad = 0;
                for (int y = startY + 1; y < endY - 1; y++)
                    for (int x = startX + 1; x < endX - 1; x++)
                        for (int c = 0; c < image.Channels; c++)
                        {
                            double dx = NumOps.ToDouble(image.GetPixel(y, Math.Min(x + 1, image.Width - 1), c)) -
                                        NumOps.ToDouble(image.GetPixel(y, Math.Max(x - 1, 0), c));
                            double dy = NumOps.ToDouble(image.GetPixel(Math.Min(y + 1, image.Height - 1), x, c)) -
                                        NumOps.ToDouble(image.GetPixel(Math.Max(y - 1, 0), x, c));
                            grad += Math.Sqrt(dx * dx + dy * dy);
                        }

                saliency[gy, gx] = grad;
            }

        return saliency;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["grid_size"] = GridSize;
        return p;
    }
}
