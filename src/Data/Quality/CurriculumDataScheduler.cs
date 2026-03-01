using AiDotNet.Helpers;

namespace AiDotNet.Data.Quality;

/// <summary>
/// Schedules training data presentation order and pacing for curriculum learning.
/// </summary>
/// <remarks>
/// <para>
/// Curriculum learning presents training samples in a meaningful order (easy to hard).
/// This scheduler determines which samples are available at each epoch based on
/// difficulty scores and a pacing function that controls data pool growth.
/// </para>
/// </remarks>
public class CurriculumDataScheduler
{
    private readonly CurriculumDataSchedulerOptions _options;
    private readonly Random _random;

    public CurriculumDataScheduler(CurriculumDataSchedulerOptions? options = null)
    {
        _options = options ?? new CurriculumDataSchedulerOptions();
        _options.Validate();
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Gets the indices available for training at a given epoch.
    /// </summary>
    /// <param name="difficultyScores">Per-sample difficulty scores (higher = harder). Shape: [numSamples].</param>
    /// <param name="epoch">Current training epoch (0-based).</param>
    /// <returns>Sorted indices of samples available at this epoch.</returns>
    public List<int> GetAvailableIndices(double[] difficultyScores, int epoch)
    {
        if (difficultyScores == null) throw new ArgumentNullException(nameof(difficultyScores));
        if (epoch < 0) throw new ArgumentOutOfRangeException(nameof(epoch), "Epoch must be non-negative.");
        int n = difficultyScores.Length;

        // Sort indices by difficulty
        var sortedIndices = Enumerable.Range(0, n)
            .OrderBy(i => difficultyScores[i])
            .ToArray();

        if (_options.Order == CurriculumOrder.HardToEasy)
            Array.Reverse(sortedIndices);

        // Determine fraction available at this epoch
        double fraction = ComputeFraction(epoch);
        int numAvailable = Math.Max(1, (int)(n * fraction));

        if (_options.Order == CurriculumOrder.Random)
        {
            // Random still respects the curriculum fraction: take the easiest N,
            // then shuffle them so order within the available pool is random.
            var available = sortedIndices.Take(numAvailable).ToList();
            for (int i = available.Count - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (available[i], available[j]) = (available[j], available[i]);
            }
            return available;
        }

        return sortedIndices.Take(numAvailable).ToList();
    }

    /// <summary>
    /// Computes the fraction of data available at a given epoch based on the pacing function.
    /// </summary>
    /// <param name="epoch">Current epoch (0-based).</param>
    /// <returns>Fraction in [InitialFraction, 1.0].</returns>
    public double ComputeFraction(int epoch)
    {
        if (epoch < 0) throw new ArgumentOutOfRangeException(nameof(epoch), "Epoch must be non-negative.");
        if (epoch >= _options.FullDataEpoch)
            return 1.0;

        double progress = (double)epoch / _options.FullDataEpoch;

        double fraction = _options.Pacing switch
        {
            CurriculumPacing.Linear =>
                _options.InitialFraction + (1.0 - _options.InitialFraction) * progress,

            CurriculumPacing.Exponential =>
                _options.InitialFraction * Math.Pow(1.0 / _options.InitialFraction, progress),

            CurriculumPacing.Step =>
                _options.InitialFraction + (1.0 - _options.InitialFraction) * Math.Floor(progress * 5) / 5.0,

            _ => _options.InitialFraction + (1.0 - _options.InitialFraction) * progress
        };

        return Math.Min(1.0, Math.Max(_options.InitialFraction, fraction));
    }
}
