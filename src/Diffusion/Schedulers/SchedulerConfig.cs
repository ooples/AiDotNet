using System;

namespace AiDotNet.Diffusion.Schedulers
{
    public sealed class SchedulerConfig<T>
    {
        public int TrainTimesteps { get; }
        public T BetaStart { get; }
        public T BetaEnd { get; }
        public BetaSchedule BetaSchedule { get; }
        public bool ClipSample { get; }
        public PredictionType PredictionType { get; }

        public SchedulerConfig(
            int trainTimesteps,
            T betaStart,
            T betaEnd,
            BetaSchedule betaSchedule = BetaSchedule.Linear,
            bool clipSample = false,
            PredictionType predictionType = PredictionType.Epsilon)
        {
            if (trainTimesteps <= 1) throw new ArgumentOutOfRangeException("trainTimesteps");

            TrainTimesteps = trainTimesteps;
            BetaStart = betaStart;
            BetaEnd = betaEnd;
            BetaSchedule = betaSchedule;
            ClipSample = clipSample;
            PredictionType = predictionType;
        }
    }
}
