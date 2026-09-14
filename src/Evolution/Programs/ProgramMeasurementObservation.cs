namespace AiDotNet.Evolution.Programs;

/// <summary>One original scalar observation, identified independently of later cache lookups.</summary>
public sealed class ProgramMeasurementObservation
{
    /// <summary>Creates a finite observation; the producer owns truthful acquisition and globally stable identity.</summary>
    public ProgramMeasurementObservation(string sampleId, double value)
    {
        VersionPinnedProgramFitnessEvaluator.ValidateIdentity(sampleId, nameof(sampleId));
        if (double.IsNaN(value) || double.IsInfinity(value)) throw new ArgumentOutOfRangeException(nameof(value));
        SampleId = sampleId; Value = value;
    }

    /// <summary>Gets the original sample identity, not a reuse operation identity.</summary>
    public string SampleId { get; }
    /// <summary>Gets the observed scalar fitness in the declared measurement units.</summary>
    public double Value { get; }
}

/// <summary>Bounded scalar mean and sample-standard-error calculation shared by acquisition and verification.</summary>
public static class ProgramMeasurementStatistics
{
    /// <summary>Fingerprint of Welford arithmetic mean and sqrt(sample variance / n); no confidence interval is inferred.</summary>
    public const string Version = "program-scalar-mean-standard-error-v1";

    /// <summary>Requires 2–256 distinct finite observations; rejects unrepresentable intermediate arithmetic.</summary>
    /// <remarks>The standard error assumes independent samples; this calculation cannot establish independence or stationarity.</remarks>
    public static (double Mean, double StandardError) Calculate(IEnumerable<ProgramMeasurementObservation> observations)
    {
        if (observations is null) throw new ArgumentNullException(nameof(observations));
        var samples = observations.Take(EvolutionMeasurementOrigin.MaximumSampleIds + 1).ToArray();
        if (samples.Length < 2 || samples.Length > EvolutionMeasurementOrigin.MaximumSampleIds ||
            samples.Any(sample => sample is null) || samples.Select(sample => sample.SampleId).Distinct(StringComparer.Ordinal).Count() != samples.Length)
            throw new ArgumentException("Require 2–256 distinct original observations.", nameof(observations));
        double mean = 0, squaredDeviations = 0;
        for (int i = 0; i < samples.Length; i++)
        {
            double delta = samples[i].Value - mean;
            mean += delta / (i + 1);
            squaredDeviations += delta * (samples[i].Value - mean);
        }
        double standardError = Math.Sqrt(squaredDeviations / (samples.Length * (samples.Length - 1)));
        if (double.IsNaN(mean) || double.IsInfinity(mean) || double.IsNaN(standardError) || double.IsInfinity(standardError))
            throw new ArgumentException("Observation statistics exceed finite arithmetic.", nameof(observations));
        return (mean, standardError);
    }
}
