namespace AiDotNet.Evolution.Deployment;

internal sealed class DeploymentValidationEvidence
{
    public DeploymentValidationEvidence() { }
    public int SchemaVersion { get; set; } = 1;
    public string Kind { get; set; } = "paired-deployment-validation";
    public string CandidateId { get; set; } = string.Empty;
    public string IncumbentId { get; set; } = string.Empty;
    public string EnvelopeKey { get; set; } = string.Empty;
    public EvolutionOptimizationDirection Direction { get; set; }
    public double MinimumMeanGain { get; set; }
    public double MaximumPValue { get; set; }
    public double MaximumP95LatencyRatio { get; set; }
    public DeploymentMeasurementPair[] Pairs { get; set; } = Array.Empty<DeploymentMeasurementPair>();

    internal void Validate()
    {
        if (SchemaVersion != 1 || Kind != "paired-deployment-validation" || Pairs is null || Pairs.Length is < 5 or > 64 ||
            !Enum.IsDefined(typeof(EvolutionOptimizationDirection), Direction))
            throw new InvalidDataException("Invalid deployment validation protocol.");
        if (!EvolutionDeploymentMeasurement.Finite(MinimumMeanGain) || MinimumMeanGain < 0 ||
            !EvolutionDeploymentMeasurement.Finite(MaximumPValue) || MaximumPValue <= 0 || MaximumPValue >= 0.5 ||
            !EvolutionDeploymentMeasurement.Finite(MaximumP95LatencyRatio) || MaximumP95LatencyRatio <= 0)
            throw new InvalidDataException("Invalid deployment validation thresholds.");
        try
        {
            DeploymentEncoding.RequireHash(CandidateId); DeploymentEncoding.RequireHash(IncumbentId); DeploymentEncoding.RequireHash(EnvelopeKey);
            foreach (var pair in Pairs)
            {
                if (pair?.Candidate is null || pair.Incumbent is null) throw new InvalidDataException("Missing paired deployment measurement.");
                _ = pair.Candidate.ToMeasurement(Direction); _ = pair.Incumbent.ToMeasurement(Direction);
            }
        }
        catch (ArgumentException error) { throw new InvalidDataException("Invalid stored deployment measurement.", error); }
    }
    internal bool AllPassed => Pairs.All(pair => pair.Candidate.CorrectnessPassed && pair.Candidate.IsFresh && pair.Candidate.HasQuality &&
        pair.Incumbent.CorrectnessPassed && pair.Incumbent.IsFresh && pair.Incumbent.HasQuality);
    internal double Mean(bool candidate) => Mean(Pairs.Select(pair => candidate ? pair.Candidate.Quality : pair.Incumbent.Quality).ToArray());
    internal long P95(bool candidate) => P95(Pairs.Select(pair => candidate ? pair.Candidate.ElapsedTicks : pair.Incumbent.ElapsedTicks).ToArray());
    internal bool Qualifies(EvolutionDeploymentPolicy policy)
    {
        if (!AllPassed) return false;
        double[] gains = Pairs.Select(pair => policy.Direction == EvolutionOptimizationDirection.Maximize
            ? pair.Candidate.Quality - pair.Incumbent.Quality : pair.Incumbent.Quality - pair.Candidate.Quality).ToArray();
        if (gains.Any(value => !EvolutionDeploymentMeasurement.Finite(value))) return false;
        int wins = gains.Count(value => value > 0);
        int nonTies = gains.Count(value => value != 0);
        // Exact Binomial(n, .5) upper tail; ties carry no directional information.
        double probability = Math.Pow(0.5, nonTies), tail = 0;
        for (int k = 0; k <= nonTies; k++)
        {
            if (k >= wins) tail += probability;
            if (k < nonTies) probability *= (nonTies - k) / (k + 1d);
        }
        double gain = Mean(gains);
        return gain > 0 && gain >= policy.MinimumMeanGain && tail <= policy.MaximumPValue &&
            P95(true) / (double)P95(false) <= policy.MaximumP95LatencyRatio;
    }
    internal static double Mean(double[] values)
    {
        double mean = values.Sum(value => value / values.Length);
        if (!EvolutionDeploymentMeasurement.Finite(mean)) throw new InvalidDataException("Unrepresentable deployment mean.");
        return mean;
    }
    internal static long P95(long[] values)
    {
        Array.Sort(values);
        return values[(int)Math.Ceiling(values.Length * 0.95) - 1];
    }
}

internal sealed class DeploymentMeasurementPair
{
    public DeploymentMeasurementPair() { }
    public DeploymentRawMeasurement Candidate { get; set; } = new();
    public DeploymentRawMeasurement Incumbent { get; set; } = new();
}

internal sealed class DeploymentRawMeasurement
{
    public DeploymentRawMeasurement() { }
    public bool CorrectnessPassed { get; set; }
    public bool IsFresh { get; set; }
    public bool HasQuality { get; set; }
    public double Quality { get; set; }
    public long ElapsedTicks { get; set; }
    public double CostUnits { get; set; }
    internal static DeploymentRawMeasurement From(EvolutionDeploymentMeasurement measurement) => new()
    { CorrectnessPassed = measurement.CorrectnessPassed, IsFresh = measurement.IsFresh, HasQuality = measurement.HasQuality, Quality = measurement.Quality,
        ElapsedTicks = measurement.Elapsed.Ticks, CostUnits = measurement.CostUnits };
    internal EvolutionDeploymentMeasurement ToMeasurement(EvolutionOptimizationDirection direction) =>
        new(CorrectnessPassed, Quality, direction, TimeSpan.FromTicks(ElapsedTicks), CostUnits, IsFresh, HasQuality);
}
