using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Derives a whole archive grid from what a seed population actually measured.</summary>
/// <remarks>
/// <para>
/// A MAP-Elites archive needs a grid, and a grid needs bounds. Writing them by hand is the one piece of setup that
/// cannot be inferred from the code, because it depends on what the descriptors mean and what values the problem
/// produces. Guess too wide and every candidate lands in one cell, so the search quietly degenerates into a plain
/// optimizer; guess too narrow and the grid is mostly empty, so nothing competes. Both failures look like a search
/// that simply is not working, and neither reports an error.
/// </para>
/// <para>
/// This measures the seeds instead. <see cref="EvolutionDescriptorCalibrator"/> already derives one axis from
/// observed values; this derives every axis at once, discovers the axis names when the caller does not name them,
/// and keeps the result deterministic: the seeds are a fixed, ordered set the caller already supplies, so the same
/// seeds give the same grid on any machine and in any culture. That grid enters the archive's definition hash
/// exactly as a hand-written one would, so a checkpoint still refuses to restore into an incompatible archive.
/// Nothing here is population-relative at run time — the grid is fixed before the first proposal, and afterwards
/// only ever widens, in whole bins, under the configured policy.
/// </para>
/// <para>
/// <see cref="FromObservations"/> is the primitive and evaluates nothing: hand it descriptor values you already
/// have. <see cref="CalibrateAsync{TGenome}"/> is the convenience that measures the seeds first.
/// </para>
/// <para><b>For Beginners:</b> Rather than inventing numbers for how "short" or "fast" a candidate can be, hand the
/// seeds you were going to search from anyway to <see cref="CalibrateAsync{TGenome}"/> and pass the definitions it
/// returns to your archive. Anything the search later finds outside that range makes the grid grow rather than
/// being thrown away.</para>
/// </remarks>
public static class EvolutionDescriptorCalibration
{
    /// <summary>Derives descriptor definitions from descriptor values that have already been measured.</summary>
    /// <param name="observations">One dictionary of descriptor values per measured candidate.</param>
    /// <param name="names">
    /// The descriptors to calibrate, in archive axis order, or <see langword="null"/> to calibrate every name any
    /// observation reported, ordered ordinally.
    /// </param>
    /// <param name="options">Calibration settings, or <see langword="null"/> for the defaults.</param>
    /// <returns>One definition per requested descriptor, in the order the axes were requested.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="observations"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// An observation is <see langword="null"/>, <paramref name="names"/> holds a blank or repeated entry, or a
    /// requested descriptor was never reported with a finite value.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">A calibration setting is outside its permitted range.</exception>
    /// <remarks>
    /// A descriptor no observation reported is an error rather than a skipped axis: the caller asked for a grid
    /// dimension, and quietly building one without it would produce a different archive than the one requested.
    /// Values that are not finite are ignored, so one failed measurement does not decide an axis.
    /// </remarks>
    public static IReadOnlyList<EvolutionDescriptorDefinition> FromObservations(
        IReadOnlyList<IReadOnlyDictionary<string, double>> observations,
        IReadOnlyList<string>? names = null,
        EvolutionDescriptorCalibrationOptions? options = null)
    {
        Guard.NotNull(observations);
        EvolutionDescriptorCalibrationOptions settings = (options ?? new EvolutionDescriptorCalibrationOptions()).Clone();
        settings.Validate();

        for (int index = 0; index < observations.Count; index++)
        {
            if (observations[index] is null)
                throw new ArgumentException("An observation cannot be null.", nameof(observations));
        }

        IReadOnlyList<string> axes = names is null ? DiscoverNames(observations) : ValidateNames(names);
        var definitions = new List<EvolutionDescriptorDefinition>(axes.Count);
        foreach (string axis in axes) definitions.Add(Calibrate(axis, observations, settings));
        return definitions;
    }

    /// <summary>Evaluates each seed once and derives descriptor definitions from what they measured.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="task">The task whose evaluations report the descriptors.</param>
    /// <param name="seeds">The seed population, in the order it will be given to the engine.</param>
    /// <param name="names">
    /// The descriptors to calibrate, in archive axis order, or <see langword="null"/> to calibrate every name the
    /// seeds reported, ordered ordinally.
    /// </param>
    /// <param name="options">Calibration settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="seed">The run's root seed, so the calibration pass is reproducible alongside the run.</param>
    /// <param name="cancellationToken">A token that cancels the calibration.</param>
    /// <returns>One definition per requested descriptor, in the order the axes were requested.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="task"/> or <paramref name="seeds"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="seeds"/> is empty or holds a <see langword="null"/> genome, or no seed completed with a finite
    /// value for a requested descriptor.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Seeds are evaluated one at a time, in the given order, so the pass costs the same as the engine's own seeding
    /// round and produces the same answer every time. A seed whose evaluation does not complete contributes nothing
    /// and is not an error: a seed population where some members fail is ordinary, and the survivors still describe
    /// the space.
    /// </para>
    /// <para>
    /// These evaluations happen before the engine exists, so they are separate from the run's own budget. With
    /// result caching configured the engine's seeding round will hit the cache; without it the seeds are measured
    /// twice. For an expensive evaluator, collect descriptors during your own seeding pass and call
    /// <see cref="FromObservations"/> directly instead. For program evolution nothing extra is evaluated at all,
    /// because a program descriptor reads only the program text.
    /// </para>
    /// </remarks>
    public static async Task<IReadOnlyList<EvolutionDescriptorDefinition>> CalibrateAsync<TGenome>(
        IEvolutionTask<TGenome> task,
        IReadOnlyList<TGenome> seeds,
        IReadOnlyList<string>? names = null,
        EvolutionDescriptorCalibrationOptions? options = null,
        ulong seed = 0,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(task);
        Guard.NotNull(seeds);
        if (seeds.Count == 0) throw new ArgumentException("At least one seed is required.", nameof(seeds));

        var observations = new List<IReadOnlyDictionary<string, double>>(seeds.Count);
        for (int index = 0; index < seeds.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (seeds[index] is not { } genome) throw new ArgumentException("A seed cannot be null.", nameof(seeds));

            EvolutionCanonicalGenome<TGenome> canonical = await task
                .CanonicalizeAsync(genome, cancellationToken).ConfigureAwait(false);
            if (canonical is null) throw new InvalidOperationException("The task canonicalized a seed to null.");

            var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
            var candidate = new EvolutionCandidate<TGenome>(index, canonical, lineage);

            // Mirrors the engine's own seeding stream so a calibration pass and the run it precedes draw from the
            // same place, rather than from a stream only this helper knows how to reproduce.
            var context = new EvolutionEvaluationContext(index, seed, unchecked((ulong)index * 8UL + 2UL), 1);

            EvolutionTaskResult result = await task
                .EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
            if (result is null || result.Status != EvolutionEvaluationStatus.Completed) continue;
            observations.Add(result.Descriptors);
        }

        return FromObservations(observations, names, options);
    }

    /// <summary>Collects every descriptor name any observation reported, ordered for a stable axis order.</summary>
    private static IReadOnlyList<string> DiscoverNames(
        IReadOnlyList<IReadOnlyDictionary<string, double>> observations)
    {
        var names = new SortedSet<string>(StringComparer.Ordinal);
        foreach (IReadOnlyDictionary<string, double> observation in observations)
        {
            foreach (KeyValuePair<string, double> pair in observation)
            {
                if (!string.IsNullOrWhiteSpace(pair.Key) && EvolutionDescriptorDefinition.IsFinite(pair.Value))
                    names.Add(pair.Key);
            }
        }

        if (names.Count == 0)
        {
            throw new ArgumentException(
                "No observation reported a finite descriptor, so there is nothing to calibrate. Report at least " +
                "one descriptor from the task's evaluation, or define the archive axes by hand.",
                nameof(observations));
        }

        return new List<string>(names);
    }

    /// <summary>Checks a caller-supplied axis list for blanks and repeats before anything is measured.</summary>
    private static IReadOnlyList<string> ValidateNames(IReadOnlyList<string> names)
    {
        if (names.Count == 0) throw new ArgumentException("At least one descriptor name is required.", nameof(names));

        var seen = new HashSet<string>(StringComparer.Ordinal);
        var axes = new List<string>(names.Count);
        foreach (string name in names)
        {
            if (name is not { } text || text.Trim().Length == 0)
                throw new ArgumentException("A descriptor name cannot be empty or white space.", nameof(names));
            string trimmed = text.Trim();
            if (!seen.Add(trimmed)) throw new ArgumentException("Descriptor names must be distinct.", nameof(names));
            axes.Add(trimmed);
        }

        return axes;
    }

    /// <summary>Derives one axis from the finite values the observations reported for it.</summary>
    private static EvolutionDescriptorDefinition Calibrate(
        string name,
        IReadOnlyList<IReadOnlyDictionary<string, double>> observations,
        EvolutionDescriptorCalibrationOptions options)
    {
        double minimum = 0;
        double maximum = 0;
        bool observed = false;

        foreach (IReadOnlyDictionary<string, double> observation in observations)
        {
            if (!observation.TryGetValue(name, out double value)) continue;
            if (!EvolutionDescriptorDefinition.IsFinite(value)) continue;
            if (!observed)
            {
                minimum = value;
                maximum = value;
                observed = true;
                continue;
            }

            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }

        if (!observed)
        {
            throw new ArgumentException(
                "No observation reported a finite value for descriptor '" + name + "', so its range cannot be " +
                "derived. Report it from the task's evaluation, or define this axis by hand.",
                nameof(observations));
        }

        if (minimum == maximum) return Degenerate(name, minimum, options);

        // Only the extremes are fed in, because the per-axis calibrator keeps nothing else: two observations freeze
        // to the same definition as all of them, and reusing it here means the padding and the finiteness checks
        // cannot drift from the ones a caller gets when calibrating an axis directly.
        var axis = new EvolutionDescriptorCalibrator(name, options.BinCount, options.OutOfRangePolicy);
        axis.Observe(minimum);
        axis.Observe(maximum);
        try
        {
            return axis.Freeze(options.Padding);
        }
        catch (InvalidOperationException exception)
        {
            throw new ArgumentException(
                "The values reported for descriptor '" + name + "' do not produce a finite range. Define this axis " +
                "by hand, reduce the padding, or report values on a smaller scale.",
                nameof(observations), exception);
        }
    }

    /// <summary>Builds an axis for a descriptor every observation agreed on.</summary>
    /// <remarks>
    /// This is an ordinary case, not an error: a descriptor that reads zero for every seed and only moves once the
    /// search finds something. It gets a window of <see cref="EvolutionDescriptorCalibrationOptions.DegenerateSpan"/>
    /// centred on the shared value, which is deliberately not what freezing a single-valued axis does on its own —
    /// that nudges the bounds apart by one representable double, and a bin that narrow would need astronomically
    /// many growth steps to reach the first value that differs.
    /// </remarks>
    private static EvolutionDescriptorDefinition Degenerate(
        string name, double value, EvolutionDescriptorCalibrationOptions options)
    {
        double half = options.DegenerateSpan / 2;
        double minimum = value - half;
        double maximum = value + half;
        if (!EvolutionDescriptorDefinition.IsFinite(minimum) || !EvolutionDescriptorDefinition.IsFinite(maximum) ||
            !EvolutionDescriptorDefinition.IsFinite(maximum - minimum) || maximum <= minimum)
        {
            throw new ArgumentException(
                "Descriptor '" + name + "' reported one value that no span can be centred on. Define this axis by " +
                "hand, or report values on a smaller scale.",
                nameof(value));
        }

        return new EvolutionDescriptorDefinition(name, minimum, maximum, options.BinCount, options.OutOfRangePolicy);
    }
}
