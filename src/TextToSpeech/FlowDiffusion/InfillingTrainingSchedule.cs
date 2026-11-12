using AiDotNet.LearningRateSchedulers;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The released single-worker LinearLR/SequentialLR recipe in SWivid/F5-TTS model/trainer.py.</summary>
internal static class InfillingTrainingSchedule
{
    internal static LambdaLRScheduler Create(double learningRate, int warmup, int total)
    {
        if (warmup < 0 || total <= warmup)
            throw new ArgumentOutOfRangeException(nameof(total), "TotalSteps must exceed the nonnegative warmup.");
        // PyTorch constructs LinearLR at step zero; the first optimizer update uses the initial factor.
        const double floor = 1e-8;
        var scheduler = new LambdaLRScheduler(learningRate, step => step < warmup
            ? floor + (1 - floor) * step / warmup
            : Math.Max(floor, 1 - (1 - floor) * (step - warmup) / (total - warmup)));
        var state = scheduler.GetState();
        state["current_lr"] = scheduler.GetLearningRateAtStep(0);
        scheduler.LoadState(state);
        return scheduler;
    }
}
