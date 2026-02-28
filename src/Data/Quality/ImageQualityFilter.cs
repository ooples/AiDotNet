namespace AiDotNet.Data.Quality;

/// <summary>
/// Filters images based on resolution, aspect ratio, and pixel statistics.
/// </summary>
/// <remarks>
/// <para>
/// Checks image metadata and pixel-level statistics to detect low-quality images:
/// blank/solid-color images, extreme aspect ratios, tiny resolutions, and corrupt files.
/// Works on raw pixel data represented as flattened arrays.
/// </para>
/// </remarks>
public class ImageQualityFilter
{
    private readonly ImageQualityFilterOptions _options;

    public ImageQualityFilter(ImageQualityFilterOptions? options = null)
    {
        _options = options ?? new ImageQualityFilterOptions();
    }

    /// <summary>
    /// Checks whether an image passes quality filters based on dimensions.
    /// </summary>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <returns>True if the image passes dimension checks.</returns>
    public bool PassesDimensionCheck(int width, int height)
    {
        if (width < _options.MinWidth || height < _options.MinHeight)
            return false;

        double aspectRatio = width >= height
            ? (double)width / height
            : (double)height / width;

        if (aspectRatio > _options.MaxAspectRatio)
            return false;

        return true;
    }

    /// <summary>
    /// Checks whether pixel values indicate a quality image (not blank, not solid).
    /// </summary>
    /// <param name="pixels">Flattened pixel values (any channel layout).</param>
    /// <returns>True if the image passes pixel statistics checks.</returns>
    public bool PassesPixelCheck(double[] pixels)
    {
        if (pixels.Length == 0)
            return false;

        // Compute mean
        double sum = 0;
        foreach (double p in pixels)
            sum += p;
        double mean = sum / pixels.Length;

        // Compute standard deviation
        double sumSqDiff = 0;
        foreach (double p in pixels)
        {
            double diff = p - mean;
            sumSqDiff += diff * diff;
        }
        double stdDev = Math.Sqrt(sumSqDiff / pixels.Length);

        if (stdDev < _options.MinPixelStdDev)
            return false;

        // Check dominant color ratio
        var valueCounts = new Dictionary<int, int>();
        foreach (double p in pixels)
        {
            int quantized = (int)Math.Round(p);
            valueCounts[quantized] = valueCounts.GetValueOrDefault(quantized, 0) + 1;
        }

        int maxCount = 0;
        foreach (int count in valueCounts.Values)
        {
            if (count > maxCount) maxCount = count;
        }

        if ((double)maxCount / pixels.Length > _options.MaxDominantColorRatio)
            return false;

        if (valueCounts.Count < _options.MinUniqueColors)
            return false;

        return true;
    }

    /// <summary>
    /// Checks whether a file size meets minimum requirements.
    /// </summary>
    /// <param name="fileSize">File size in bytes.</param>
    /// <returns>True if the file size is acceptable.</returns>
    public bool PassesFileSizeCheck(long fileSize)
    {
        return fileSize >= _options.MinFileSize;
    }

    /// <summary>
    /// Filters images by dimension, returning indices of images that should be removed.
    /// </summary>
    /// <param name="widths">Width of each image.</param>
    /// <param name="heights">Height of each image.</param>
    /// <returns>Set of indices that fail quality checks (should be removed).</returns>
    public HashSet<int> FilterByDimensions(IReadOnlyList<int> widths, IReadOnlyList<int> heights)
    {
        var filtered = new HashSet<int>();
        int count = Math.Min(widths.Count, heights.Count);

        for (int i = 0; i < count; i++)
        {
            if (!PassesDimensionCheck(widths[i], heights[i]))
                filtered.Add(i);
        }

        return filtered;
    }
}
