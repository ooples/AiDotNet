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
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional subject filter (case-insensitive substring of subject directory name). Null = all 7 subjects.</summary>
    public string? SubjectFilter { get; set; }
    /// <summary>Difficulty level filter (1..5). Null = all levels.</summary>
    public int? LevelFilter { get; set; }
    public int MaxProblemLength { get; set; } = 256;
    public int MaxSolutionLength { get; set; } = 512;
    public int VocabularySize { get; set; } = 16000;
    public int? MaxSamples { get; set; }
}
