namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>One annotated text region: a polygon in source-image pixels and, optionally, its transcription.</summary>
/// <remarks>
/// Word-level polygons are the standard annotation (ICDAR, Total-Text). CRAFT's weak supervision uses the
/// transcription's length to split a word into character boxes (Baek et al. 2019, Section 3.2). Without a
/// transcription it falls back to the word's aspect ratio.
/// </remarks>
public sealed class TextPolygonTarget
{
    /// <summary>Creates a target from at least three polygon vertices, in order around the region.</summary>
    public TextPolygonTarget(IEnumerable<(double X, double Y)> points, string? transcription = null)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        var owned = points.ToArray();
        if (owned.Length < 3) throw new ArgumentException("A text polygon needs at least three vertices.", nameof(points));
        if (owned.Any(p => double.IsNaN(p.X) || double.IsNaN(p.Y) || double.IsInfinity(p.X) || double.IsInfinity(p.Y)))
            throw new ArgumentException("Polygon vertices must be finite.", nameof(points));
        Points = Array.AsReadOnly(owned);
        Transcription = transcription;
    }

    /// <summary>The vertices, in source-image pixels.</summary>
    public IReadOnlyList<(double X, double Y)> Points { get; }

    /// <summary>The region's text, when known.</summary>
    public string? Transcription { get; }

    /// <summary>An axis-aligned box <c>(x0, y0)</c>..<c>(x1, y1)</c> as a four-vertex polygon.</summary>
    public static TextPolygonTarget FromBox(double x0, double y0, double x1, double y1, string? transcription = null)
        => new(new[] { (x0, y0), (x1, y0), (x1, y1), (x0, y1) }, transcription);
}

/// <summary>The text polygons of every image in a training batch.</summary>
public sealed class TextDetectionTrainingBatch
{
    private readonly IReadOnlyList<TextPolygonTarget>[] _images;

    /// <summary>Creates a batch from one target list per image. An image with no text has an empty list.</summary>
    public TextDetectionTrainingBatch(IEnumerable<IEnumerable<TextPolygonTarget>> images)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        var owned = new List<IReadOnlyList<TextPolygonTarget>>();
        foreach (var image in images)
        {
            if (image is null) throw new ArgumentException("Each image must have a target list, even when empty.", nameof(images));
            var targets = image.ToArray();
            if (targets.Any(target => target is null))
                throw new ArgumentException("Target lists cannot contain null entries.", nameof(images));
            owned.Add(Array.AsReadOnly(targets));
        }
        if (owned.Count == 0) throw new ArgumentException("A training batch must contain at least one image.", nameof(images));
        _images = owned.ToArray();
    }

    /// <summary>Number of images.</summary>
    public int ImageCount => _images.Length;

    /// <summary>The polygons of image <paramref name="imageIndex"/>.</summary>
    public IReadOnlyList<TextPolygonTarget> this[int imageIndex] => _images[imageIndex];
}
