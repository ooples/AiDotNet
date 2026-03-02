namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Visualization settings and utilities for segmentation outputs.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> After segmenting an image, you usually want to visualize the result.
/// This class provides configuration for creating color-coded overlay images where each class
/// or instance gets a distinct color. The overlay can be blended with the original image
/// to see both the segmentation and the original content.
/// </para>
/// </remarks>
public class SegmentationVisualizationConfig
{
    /// <summary>
    /// Alpha blending factor for overlay (0 = fully transparent, 1 = fully opaque).
    /// Default: 0.5 (50% blend between original image and segmentation colors).
    /// </summary>
    public double Alpha { get; set; } = 0.5;

    /// <summary>
    /// Whether to draw contours/boundaries around segmented regions.
    /// </summary>
    public bool DrawContours { get; set; } = true;

    /// <summary>
    /// Contour thickness in pixels.
    /// </summary>
    public int ContourThickness { get; set; } = 2;

    /// <summary>
    /// Whether to display class labels on the visualization.
    /// </summary>
    public bool ShowLabels { get; set; } = true;

    /// <summary>
    /// Whether to display confidence scores alongside labels.
    /// </summary>
    public bool ShowScores { get; set; }

    /// <summary>
    /// Whether to display instance bounding boxes.
    /// </summary>
    public bool ShowBoundingBoxes { get; set; }

    /// <summary>
    /// Whether to use a fixed color palette or generate random colors.
    /// </summary>
    public bool UseFixedPalette { get; set; } = true;

    /// <summary>
    /// Custom color palette as RGB triplets [numColors, 3] with values in [0, 255].
    /// If null, a default palette is used.
    /// </summary>
    public byte[,]? ColorPalette { get; set; }

    /// <summary>
    /// Minimum confidence threshold for displaying instances.
    /// </summary>
    public double MinDisplayConfidence { get; set; } = 0.3;

    /// <summary>
    /// Background color for areas with no segmentation (R, G, B).
    /// </summary>
    public (byte R, byte G, byte B) BackgroundColor { get; set; } = (0, 0, 0);

    /// <summary>
    /// Gets the default ADE20K color palette (150 classes).
    /// </summary>
    public static byte[,] GetADE20KPalette()
    {
        // Standard ADE20K color palette (first 20 shown, rest generated systematically)
        var palette = new byte[150, 3];
        byte[][] colors =
        [
            [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3]
        ];

        for (int i = 0; i < Math.Min(colors.Length, 150); i++)
        {
            palette[i, 0] = colors[i][0];
            palette[i, 1] = colors[i][1];
            palette[i, 2] = colors[i][2];
        }

        // Generate remaining colors deterministically
        for (int i = colors.Length; i < 150; i++)
        {
            palette[i, 0] = (byte)((i * 37 + 120) % 256);
            palette[i, 1] = (byte)((i * 73 + 60) % 256);
            palette[i, 2] = (byte)((i * 113 + 180) % 256);
        }

        return palette;
    }

    /// <summary>
    /// Gets the default COCO panoptic color palette (133 classes).
    /// </summary>
    public static byte[,] GetCOCOPalette()
    {
        var palette = new byte[133, 3];
        for (int i = 0; i < 133; i++)
        {
            int id = i + 1;
            byte r = 0, g = 0, b = 0;
            for (int j = 0; j < 8; j++)
            {
                r |= (byte)(((id >> (0 + j * 3)) & 1) << (7 - j));
                g |= (byte)(((id >> (1 + j * 3)) & 1) << (7 - j));
                b |= (byte)(((id >> (2 + j * 3)) & 1) << (7 - j));
            }

            palette[i, 0] = r;
            palette[i, 1] = g;
            palette[i, 2] = b;
        }

        return palette;
    }

    /// <summary>
    /// Gets a Cityscapes-style color palette (19 classes).
    /// </summary>
    public static byte[,] GetCityscapesPalette()
    {
        byte[,] palette =
        {
            { 128, 64, 128 },   // road
            { 244, 35, 232 },   // sidewalk
            { 70, 70, 70 },     // building
            { 102, 102, 156 },  // wall
            { 190, 153, 153 },  // fence
            { 153, 153, 153 },  // pole
            { 250, 170, 30 },   // traffic light
            { 220, 220, 0 },    // traffic sign
            { 107, 142, 35 },   // vegetation
            { 152, 251, 152 },  // terrain
            { 70, 130, 180 },   // sky
            { 220, 20, 60 },    // person
            { 255, 0, 0 },      // rider
            { 0, 0, 142 },      // car
            { 0, 0, 70 },       // truck
            { 0, 60, 100 },     // bus
            { 0, 80, 100 },     // train
            { 0, 0, 230 },      // motorcycle
            { 119, 11, 32 }     // bicycle
        };
        return palette;
    }
}
