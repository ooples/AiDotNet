using System;

namespace AiDotNet.Diffusion.Schedulers
{
    public enum BetaSchedule
    {
        Linear,
        // Cosine and others can be added later
    }

    public enum PredictionType
    {
        Epsilon,
        // Sample or VPrediction can be added later
    }

    public sealed class SchedulerConfig
    {
        public int TrainTimesteps { get; }
        public double BetaStart { get; }
        public double BetaEnd { get; }
        public BetaSchedule BetaSchedule { get; }
        public bool ClipSample { get; }
        public PredictionType PredictionType { get; }

        public SchedulerConfig(
            int trainTimesteps = 1000,
            double betaStart = 0.0001,
            double betaEnd = 0.02,
            BetaSchedule betaSchedule = BetaSchedule.Linear,
            bool clipSample = false,
            PredictionType predictionType = PredictionType.Epsilon)
        {
            if (trainTimesteps <= 1) throw new ArgumentOutOfRangeException("trainTimesteps");
            if (betaStart <= 0 || betaEnd <= 0 || betaEnd <= betaStart) throw new ArgumentOutOfRangeException("beta range invalid");

            TrainTimesteps = trainTimesteps;
            BetaStart = betaStart;
            BetaEnd = betaEnd;
            BetaSchedule = betaSchedule;
            ClipSample = clipSample;
            PredictionType = predictionType;
        }
    }
}

