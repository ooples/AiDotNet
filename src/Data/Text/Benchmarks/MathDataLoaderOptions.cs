using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the Hendrycks MATH benchmark loader.
/// </summary>
/// <remarks>
/// <para>
/// MATH (Hendrycks et al. 2021) — 12,500 competition math problems
/// (7,500 train / 5,000 test) drawn from AMC/AIME/HMMT competitions
/// across 7 subject areas (algebra, counting/probability, geometry,
/// intermediate algebra, number theory, prealgebra, precalculus).
/// Each problem has a difficulty level (1..5) and a step-by-step
/// LaTeX-formatted reference solution. Standard advanced math
/// reasoning benchmark.
/// </para>
/// </remarks>
public sealed class MathDataLoaderOptions
{
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Gets or sets whether the dataset is downloaded into the data directory when it is not there.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional subject filter (case-insensitive substring of subject directory name). Null = all 7 subjects.</summary>
    public string? SubjectFilter { get; set; }
    /// <summary>Difficulty level filter (1..5). Null = all levels.</summary>
    public int? LevelFilter { get; set; }
    /// <summary>Gets or sets the token length each problem statement is padded or truncated to.</summary>
    public int MaxProblemLength { get; set; } = 256;
    /// <summary>Gets or sets the token length each solution is padded or truncated to.</summary>
    public int MaxSolutionLength { get; set; } = 512;
    /// <summary>Gets or sets the most distinct tokens in the vocabulary built from the loaded text.</summary>
    public int VocabularySize { get; set; } = 16000;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MaxProblemLength <= 0) throw new ArgumentOutOfRangeException(nameof(MaxProblemLength), "MaxProblemLength must be positive.");
        if (MaxSolutionLength <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSolutionLength), "MaxSolutionLength must be positive.");
        if (VocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(VocabularySize), "VocabularySize must be positive.");
        if (LevelFilter is < 1 or > 5) throw new ArgumentOutOfRangeException(nameof(LevelFilter), "LevelFilter must be in [1, 5] when specified.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
