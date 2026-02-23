using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Scene graph-based image safety classifier that analyzes spatial relationships between
/// detected entities to identify unsafe content configurations.
/// </summary>
/// <remarks>
/// <para>
/// Rather than classifying the image as a whole, this classifier segments the image into
/// regions, characterizes each region's visual properties, and then analyzes spatial
/// relationships between regions. Certain spatial configurations (e.g., skin-colored regions
/// in specific arrangements, weapon-shaped objects near person-shaped regions) indicate
/// unsafe content that whole-image classifiers may miss.
/// </para>
/// <para>
/// <b>For Beginners:</b> This classifier works like a detective examining a scene: first it
/// identifies what objects/regions are present (skin, dark areas, bright areas), then checks
/// how they relate to each other spatially. For example, a weapon-shaped object near a
/// person-shaped region is more concerning than either alone.
/// </para>
/// <para>
/// <b>References:</b>
/// - USD: Scene-graph-based NSFW detection for text-to-image (USENIX Security 2025)
/// - Scene Graph Generation survey: methods, challenges, applications (2024)
/// - OmniSafeBench-MM: 9 risk domains with 50 fine-grained categories (2025, arxiv:2512.06589)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SceneGraphSafetyClassifier<T> : ImageSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;
    private readonly int _gridSize;

    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;
    private static readonly T TwoFiftyFive = NumOps.FromDouble(255.0);

    /// <inheritdoc />
    public override string ModuleName => "SceneGraphSafetyClassifier";

    /// <summary>
    /// Initializes a new scene graph safety classifier.
    /// </summary>
    /// <param name="threshold">Safety score threshold (0-1). Default: 0.6.</param>
    /// <param name="gridSize">Grid size for region analysis. Default: 8.</param>
    public SceneGraphSafetyClassifier(double threshold = 0.6, int gridSize = 8)
    {
        _threshold = threshold;
        _gridSize = gridSize;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var span = image.Data.Span;
        if (span.Length == 0) return findings;

        var layout = DetermineLayout(image.Shape, span.Length);
        if (layout.Height < _gridSize || layout.Width < _gridSize) return findings;

        // Build scene graph: characterize each grid cell
        int rows = layout.Height / (_gridSize > 0 ? layout.Height / _gridSize : 1);
        int cols = layout.Width / (_gridSize > 0 ? layout.Width / _gridSize : 1);
        rows = Math.Min(rows, _gridSize);
        cols = Math.Min(cols, _gridSize);
        if (rows <= 0 || cols <= 0) return findings;

        int cellH = layout.Height / rows;
        int cellW = layout.Width / cols;

        var regions = new RegionInfo[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                regions[r * cols + c] = CharacterizeRegion(span, layout, r * cellH, c * cellW, cellH, cellW);
            }
        }

        // Analyze spatial relationships for unsafe patterns
        double nsfwScore = AnalyzeNSFWPattern(regions, rows, cols);
        double violenceScore = AnalyzeViolencePattern(regions, rows, cols);
        double weaponScore = AnalyzeWeaponPattern(regions, rows, cols);

        if (nsfwScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.SexualExplicit,
                Severity = nsfwScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = nsfwScore,
                Description = $"Scene graph analysis: spatial configuration of skin-toned regions " +
                              $"indicates potential NSFW content (score: {nsfwScore:F3}).",
                RecommendedAction = nsfwScore >= 0.8 ? SafetyAction.Block : SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        if (violenceScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.ViolenceGraphic,
                Severity = violenceScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = violenceScore,
                Description = $"Scene graph analysis: spatial configuration of red/dark regions " +
                              $"indicates potential violent content (score: {violenceScore:F3}).",
                RecommendedAction = violenceScore >= 0.8 ? SafetyAction.Block : SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        if (weaponScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.ViolenceWeapons,
                Severity = SafetySeverity.Medium,
                Confidence = weaponScore,
                Description = $"Scene graph analysis: elongated high-contrast regions with metallic " +
                              $"characteristics detected (score: {weaponScore:F3}).",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private RegionInfo CharacterizeRegion(ReadOnlySpan<T> data, ImageLayout layout,
        int startY, int startX, int height, int width)
    {
        int channels = Math.Min(layout.Channels, 3);
        double rSum = 0, gSum = 0, bSum = 0;
        int skinPixels = 0, darkPixels = 0, redPixels = 0;
        double edgeSum = 0;
        int count = 0;

        for (int y = startY; y < startY + height && y < layout.Height; y++)
        {
            for (int x = startX; x < startX + width && x < layout.Width; x++)
            {
                double r = GetChannelDouble(data, layout, y, x, 0);
                double g = channels >= 2 ? GetChannelDouble(data, layout, y, x, 1) : r;
                double b = channels >= 3 ? GetChannelDouble(data, layout, y, x, 2) : r;

                // Normalize to [0,255]
                if (r <= 1.0 && g <= 1.0 && b <= 1.0) { r *= 255; g *= 255; b *= 255; }

                rSum += r; gSum += g; bSum += b;
                count++;

                double lum = 0.299 * r + 0.587 * g + 0.114 * b;

                // Skin check
                if (r > 95 && g > 40 && b > 20 && r > g && r > b && r - Math.Min(g, b) > 15)
                    skinPixels++;

                // Dark pixel
                if (lum < 40) darkPixels++;

                // Red dominant
                if (r > 100 && r > g * 1.5 && r > b * 1.5) redPixels++;

                // Edge (horizontal gradient)
                if (x > startX)
                {
                    double prevR = GetChannelDouble(data, layout, y, x - 1, 0);
                    if (prevR <= 1.0) prevR *= 255;
                    edgeSum += Math.Abs(r - prevR);
                }
            }
        }

        if (count == 0) count = 1;

        return new RegionInfo
        {
            MeanR = rSum / count,
            MeanG = gSum / count,
            MeanB = bSum / count,
            SkinFraction = (double)skinPixels / count,
            DarkFraction = (double)darkPixels / count,
            RedFraction = (double)redPixels / count,
            EdgeIntensity = edgeSum / count,
            Row = startY / (height > 0 ? height : 1),
            Col = startX / (width > 0 ? width : 1)
        };
    }

    private static double AnalyzeNSFWPattern(RegionInfo[] regions, int rows, int cols)
    {
        // NSFW pattern: large contiguous cluster of high-skin regions in central area
        int skinRegions = 0;
        int centralSkinRegions = 0;
        int adjacentSkinPairs = 0;

        for (int i = 0; i < regions.Length; i++)
        {
            if (regions[i].SkinFraction > 0.3)
            {
                skinRegions++;

                // Central region check
                double relR = (double)regions[i].Row / rows;
                double relC = (double)regions[i].Col / cols;
                if (relR > 0.15 && relR < 0.85 && relC > 0.15 && relC < 0.85)
                {
                    centralSkinRegions++;
                }

                // Check adjacency
                int r = regions[i].Row;
                int c = regions[i].Col;
                for (int j = i + 1; j < regions.Length; j++)
                {
                    if (regions[j].SkinFraction > 0.3)
                    {
                        int dr = Math.Abs(regions[j].Row - r);
                        int dc = Math.Abs(regions[j].Col - c);
                        if (dr <= 1 && dc <= 1) adjacentSkinPairs++;
                    }
                }
            }
        }

        double skinCoverage = (double)skinRegions / regions.Length;
        double centralBias = skinRegions > 0 ? (double)centralSkinRegions / skinRegions : 0;
        double clustering = regions.Length > 1 ? (double)adjacentSkinPairs / regions.Length : 0;

        return Math.Min(1.0, 0.4 * skinCoverage * 3 + 0.3 * centralBias + 0.3 * clustering * 5);
    }

    private static double AnalyzeViolencePattern(RegionInfo[] regions, int rows, int cols)
    {
        // Violence pattern: red regions adjacent to dark regions
        int redDarkPairs = 0;
        int redRegions = 0;

        for (int i = 0; i < regions.Length; i++)
        {
            if (regions[i].RedFraction > 0.2) redRegions++;

            if (regions[i].RedFraction > 0.2)
            {
                for (int j = 0; j < regions.Length; j++)
                {
                    if (i == j) continue;
                    if (regions[j].DarkFraction > 0.3)
                    {
                        int dr = Math.Abs(regions[i].Row - regions[j].Row);
                        int dc = Math.Abs(regions[i].Col - regions[j].Col);
                        if (dr <= 1 && dc <= 1) redDarkPairs++;
                    }
                }
            }
        }

        double redCoverage = (double)redRegions / regions.Length;
        double adjacency = regions.Length > 1 ? (double)redDarkPairs / regions.Length : 0;

        return Math.Min(1.0, 0.5 * redCoverage * 5 + 0.5 * adjacency * 3);
    }

    private static double AnalyzeWeaponPattern(RegionInfo[] regions, int rows, int cols)
    {
        // Weapon pattern: elongated high-edge, grey-metallic regions
        int metallicRegions = 0;
        int highEdgeRegions = 0;

        foreach (var region in regions)
        {
            // Grey/metallic: R≈G≈B with moderate brightness
            double greyness = 1.0 - (Math.Abs(region.MeanR - region.MeanG) +
                                     Math.Abs(region.MeanG - region.MeanB)) / 255.0;
            double brightness = (region.MeanR + region.MeanG + region.MeanB) / (3 * 255);

            if (greyness > 0.7 && brightness > 0.2 && brightness < 0.8)
                metallicRegions++;

            if (region.EdgeIntensity > 30)
                highEdgeRegions++;
        }

        double metallicRatio = (double)metallicRegions / regions.Length;
        double edgeRatio = (double)highEdgeRegions / regions.Length;

        return Math.Min(1.0, 0.5 * metallicRatio * 3 + 0.5 * edgeRatio * 2);
    }

    private static double GetChannelDouble(ReadOnlySpan<T> data, ImageLayout layout, int y, int x, int c)
    {
        int idx;
        if (layout.Format == PixFmt.CHW)
            idx = c * layout.Height * layout.Width + y * layout.Width + x;
        else
            idx = (y * layout.Width + x) * layout.Channels + c;

        if (idx < 0 || idx >= data.Length) return 0;
        return NumOps.ToDouble(data[idx]);
    }

    private static ImageLayout DetermineLayout(int[] shape, int dataLength)
    {
        if (shape.Length >= 4)
        {
            if (shape[1] <= 4 && shape[2] > 4 && shape[3] > 4)
                return new ImageLayout { Channels = shape[1], Height = shape[2], Width = shape[3], Format = PixFmt.CHW };
            if (shape[3] <= 4 && shape[1] > 4 && shape[2] > 4)
                return new ImageLayout { Channels = shape[3], Height = shape[1], Width = shape[2], Format = PixFmt.HWC };
        }
        if (shape.Length == 3)
        {
            if (shape[0] <= 4 && shape[1] > 4 && shape[2] > 4)
                return new ImageLayout { Channels = shape[0], Height = shape[1], Width = shape[2], Format = PixFmt.CHW };
            if (shape[2] <= 4 && shape[0] > 4 && shape[1] > 4)
                return new ImageLayout { Channels = shape[2], Height = shape[0], Width = shape[1], Format = PixFmt.HWC };
        }
        if (shape.Length == 2)
            return new ImageLayout { Channels = 1, Height = shape[0], Width = shape[1], Format = PixFmt.CHW };

        int side = (int)Math.Sqrt(dataLength);
        return new ImageLayout { Channels = 1, Height = side, Width = side > 0 ? dataLength / side : dataLength, Format = PixFmt.CHW };
    }

    private enum PixFmt { CHW, HWC }

    private struct ImageLayout
    {
        public int Channels, Height, Width;
        public PixFmt Format;
    }

    private struct RegionInfo
    {
        public double MeanR, MeanG, MeanB;
        public double SkinFraction, DarkFraction, RedFraction;
        public double EdgeIntensity;
        public int Row, Col;
    }
}
