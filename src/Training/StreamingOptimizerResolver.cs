using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

internal static class StreamingOptimizerResolver<T>
{
    /// <summary>
    /// #1662 lever #1: whether a BIT-IDENTICAL full-precision streaming variant exists for this
    /// configured optimizer. The common-case fused-in-backward default (§5a) engages only when
    /// this is true, so a model that fits in memory never silently switches to a non-bit-identical
    /// streaming update. Currently Adam-family (plain Adam + the streaming default); other families
    /// keep the classic path until their full-precision streaming variant lands.
    /// </summary>
    public static bool SupportsFullPrecision(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        bool useStreamingDefaults)
    {
        if (useStreamingDefaults) return true; // streaming default is Adam
        return Resolve(optimizer, 0.0).Kind == StreamingKind.Adam;
    }

    public static string BuildKey(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        bool useStreamingDefaults,
        double streamingWeightDecay,
        bool fullPrecision = false)
    {
        string prec = fullPrecision ? "FP|" : "Q8|";
        if (useStreamingDefaults)
            return prec + "DefaultAdam|" + streamingWeightDecay;

        StreamingConfig c = Resolve(optimizer, streamingWeightDecay);
        return prec + string.Join("|",
            optimizer.GetType().FullName ?? optimizer.GetType().Name,
            c.Kind,
            c.Beta1, c.Beta2, c.Epsilon, c.Decay, c.Rho, c.Momentum, c.WeightDecay,
            c.TrustCoefficient, c.FtrlAlpha, c.FtrlBeta, c.Lambda1, c.Lambda2, c.MaxTrustRatio,
            c.UseAMSGrad, c.UseNesterov, c.ClipTrustRatio, c.UseBiasCorrection, c.MemorySize);
    }

    public static IStreamingOptimizer<T> Create(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        bool useStreamingDefaults,
        double fallbackLearningRate,
        double fallbackWeightDecay,
        bool fullPrecision = false)
    {
        double lr = useStreamingDefaults ? fallbackLearningRate : ResolveLearningRate(optimizer, fallbackLearningRate);

        if (useStreamingDefaults)
        {
            return fullPrecision
                ? new FullPrecisionStreamingAdam<T>(lr, weightDecay: fallbackWeightDecay)
                : new StreamingAdam8Bit<T>(lr, weightDecay: fallbackWeightDecay);
        }

        StreamingConfig c = Resolve(optimizer, fallbackWeightDecay);

        // #1662 lever #1: bit-identical full-precision variants for the common-case fused default.
        // Only reached when SupportsFullPrecision gated engagement (Adam-family today).
        if (fullPrecision && c.Kind == StreamingKind.Adam)
            return new FullPrecisionStreamingAdam<T>(lr, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay);

        switch (c.Kind)
        {
            case StreamingKind.AdamW:
                return new StreamingAdamW8Bit<T>(lr, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay);
            case StreamingKind.AmsGrad:
                return new StreamingAMSGrad8Bit<T>(lr, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay);
            case StreamingKind.Nadam:
                return new StreamingNadam8Bit<T>(lr, c.Beta1, c.Beta2, c.Epsilon);
            case StreamingKind.AdaMax:
                return new StreamingAdaMax8Bit<T>(lr, c.Beta1, c.Beta2, c.Epsilon);
            case StreamingKind.Lion:
                return new StreamingLion8Bit<T>(lr, c.Beta1, c.Beta2, c.WeightDecay);
            case StreamingKind.Lamb:
                return new StreamingLamb8Bit<T>(lr, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay, c.ClipTrustRatio, c.MaxTrustRatio, c.UseBiasCorrection);
            case StreamingKind.Lars:
                return new StreamingLars8Bit<T>(lr, c.Momentum, c.WeightDecay, c.TrustCoefficient, c.Epsilon, c.UseNesterov);
            case StreamingKind.Ftrl:
                return new StreamingFtrl8Bit<T>(c.FtrlAlpha, c.FtrlBeta, c.Lambda1, c.Lambda2);
            case StreamingKind.Lbfgs:
                // Second-order: memory-bounded streaming L-BFGS (8-bit-quantized (s,y) history).
                return new StreamingLBFGS<T>(lr, c.MemorySize);
            case StreamingKind.ConjugateGradient:
                // Hessian-free second-order-like: memory-bounded streaming nonlinear CG.
                return new StreamingConjugateGradient<T>(lr);
            case StreamingKind.AdaDelta:
                return new StreamingAdaDelta8Bit<T>(lr, c.Rho, c.Epsilon);
            case StreamingKind.Adagrad:
                return new StreamingAdagrad8Bit<T>(lr, c.Epsilon);
            case StreamingKind.RmsProp:
                return new StreamingRmsProp8Bit<T>(lr, c.Decay, c.Epsilon);
            case StreamingKind.Momentum:
                return new StreamingMomentum8Bit<T>(lr, c.Momentum);
            case StreamingKind.Nesterov:
                return new StreamingNesterov8Bit<T>(lr, c.Momentum);
            case StreamingKind.Sgd:
                return new StreamingSgd8Bit<T>(lr);
            case StreamingKind.Adam:
            default:
                return new StreamingAdam8Bit<T>(lr, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay);
        }
    }

    /// <summary>
    /// The streaming variant family a configured optimizer maps to.
    /// </summary>
    private enum StreamingKind
    {
        Sgd, Momentum, Nesterov, RmsProp, Adagrad, AdaDelta,
        Adam, AdamW, AmsGrad, Nadam, AdaMax, Lion, Lamb, Lars, Ftrl, Lbfgs, ConjugateGradient
    }

    /// <summary>
    /// Strongly-typed snapshot of the hyperparameters needed to build (and cache-key) a streaming
    /// optimizer. Populated once by <see cref="Resolve"/> from each optimizer's typed options, so
    /// the construction path and the cache key never disagree and never depend on reflection.
    /// </summary>
    private struct StreamingConfig
    {
        public StreamingKind Kind;
        public double Beta1;
        public double Beta2;
        public double Epsilon;
        public double Decay;
        public double Rho;
        public double Momentum;
        public double WeightDecay;
        public double TrustCoefficient;
        public double FtrlAlpha;
        public double FtrlBeta;
        public double Lambda1;
        public double Lambda2;
        public double MaxTrustRatio;
        public bool UseAMSGrad;
        public bool UseNesterov;
        public bool ClipTrustRatio;
        public bool UseBiasCorrection;
        public int MemorySize;
    }

    /// <summary>
    /// Maps a configured optimizer + its strongly-typed options to a <see cref="StreamingConfig"/>.
    /// Reads each option via the optimizer's public <c>GetOptions()</c> downcast to its concrete
    /// options type — compile-time-checked property access, no reflection. A renamed option becomes
    /// a build error here rather than a silently-defaulted hyperparameter.
    /// </summary>
    private static StreamingConfig Resolve(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        double defaultWeightDecay)
    {
        switch (optimizer)
        {
            case AdamWOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig
                {
                    Kind = opt.UseAMSGrad ? StreamingKind.AmsGrad : StreamingKind.AdamW,
                    Beta1 = opt.Beta1, Beta2 = opt.Beta2, Epsilon = opt.Epsilon,
                    WeightDecay = opt.WeightDecay, UseAMSGrad = opt.UseAMSGrad,
                };
            }
            case AdamOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig
                {
                    Kind = opt.UseAMSGrad ? StreamingKind.AmsGrad : StreamingKind.Adam,
                    Beta1 = opt.Beta1, Beta2 = opt.Beta2, Epsilon = opt.Epsilon,
                    WeightDecay = defaultWeightDecay, UseAMSGrad = opt.UseAMSGrad,
                };
            }
            case AMSGradOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (AMSGradOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig
                {
                    Kind = StreamingKind.AmsGrad,
                    Beta1 = opt.Beta1, Beta2 = opt.Beta2, Epsilon = opt.Epsilon,
                    WeightDecay = 0.0, UseAMSGrad = true,
                };
            }
            case NadamOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (NadamOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.Nadam, Beta1 = opt.Beta1, Beta2 = opt.Beta2, Epsilon = opt.Epsilon };
            }
            case AdaMaxOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (AdaMaxOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.AdaMax, Beta1 = opt.Beta1, Beta2 = opt.Beta2, Epsilon = opt.Epsilon };
            }
            case LionOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (LionOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.Lion, Beta1 = opt.Beta1, Beta2 = opt.Beta2, WeightDecay = opt.WeightDecay };
            }
            case LAMBOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (LAMBOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig
                {
                    Kind = StreamingKind.Lamb,
                    Beta1 = opt.Beta1, Beta2 = opt.Beta2, Epsilon = opt.Epsilon, WeightDecay = opt.WeightDecay,
                    ClipTrustRatio = opt.ClipTrustRatio, MaxTrustRatio = opt.MaxTrustRatio, UseBiasCorrection = opt.UseBiasCorrection,
                };
            }
            case LARSOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (LARSOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig
                {
                    Kind = StreamingKind.Lars,
                    Momentum = opt.Momentum, WeightDecay = opt.WeightDecay, TrustCoefficient = opt.TrustCoefficient,
                    Epsilon = opt.Epsilon, UseNesterov = opt.UseNesterov,
                };
            }
            case FTRLOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (FTRLOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig
                {
                    Kind = StreamingKind.Ftrl,
                    FtrlAlpha = opt.Alpha, FtrlBeta = opt.Beta, Lambda1 = opt.Lambda1, Lambda2 = opt.Lambda2,
                };
            }
            case LBFGSOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (LBFGSOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.Lbfgs, MemorySize = opt.MemorySize };
            }
            case Adam8BitOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                // Adam8Bit IS an 8-bit Adam — its streaming form is the 8-bit-state streaming Adam.
                var opt = (Adam8BitOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig
                {
                    Kind = StreamingKind.Adam,
                    Beta1 = opt.Beta1, Beta2 = opt.Beta2, Epsilon = opt.Epsilon, WeightDecay = defaultWeightDecay,
                };
            }
            case ConjugateGradientOptimizer<T, Tensor<T>, Tensor<T>>:
                // Hessian-free O(n) method → first-class streaming nonlinear CG.
                return new StreamingConfig { Kind = StreamingKind.ConjugateGradient };
            case BFGSOptimizer<T, Tensor<T>, Tensor<T>>:
            case DFPOptimizer<T, Tensor<T>, Tensor<T>>:
            case NewtonMethodOptimizer<T, Tensor<T>, Tensor<T>>:
            case TrustRegionOptimizer<T, Tensor<T>, Tensor<T>>:
            case LevenbergMarquardtOptimizer<T, Tensor<T>, Tensor<T>>:
                // Dense-Hessian / full second-order methods keep an O(n^2) Hessian (approximation)
                // that cannot fit a memory budget. The limited-memory quasi-Newton (streaming
                // L-BFGS) is their memory-bounded streaming substitute.
                return new StreamingConfig { Kind = StreamingKind.Lbfgs, MemorySize = 10 };
            case ADMMOptimizer<T, Tensor<T>, Tensor<T>>:
            case CoordinateDescentOptimizer<T, Tensor<T>, Tensor<T>>:
                // Proximal / coordinate gradient methods reduce to a memory-bounded gradient step
                // under per-parameter streaming.
                return new StreamingConfig { Kind = StreamingKind.Sgd };
            case AdaDeltaOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (AdaDeltaOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.AdaDelta, Rho = opt.Rho, Epsilon = opt.Epsilon };
            }
            case AdagradOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (AdagradOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.Adagrad, Epsilon = opt.Epsilon };
            }
            case RootMeanSquarePropagationOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (RootMeanSquarePropagationOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.RmsProp, Decay = opt.Decay, Epsilon = opt.Epsilon };
            }
            case MomentumOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (MomentumOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.Momentum, Momentum = opt.InitialMomentum };
            }
            case NesterovAcceleratedGradientOptimizer<T, Tensor<T>, Tensor<T>> o:
            {
                var opt = (NesterovAcceleratedGradientOptimizerOptions<T, Tensor<T>, Tensor<T>>)o.GetOptions();
                return new StreamingConfig { Kind = StreamingKind.Nesterov, Momentum = opt.InitialMomentum };
            }
            case GradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
            case StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
            case MiniBatchGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
            case ProximalGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>:
                return new StreamingConfig { Kind = StreamingKind.Sgd };
            default:
                return new StreamingConfig
                {
                    Kind = StreamingKind.Adam,
                    Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = defaultWeightDecay,
                };
        }
    }

    public static void RefreshLearningRate(
        IStreamingOptimizer<T> streamingOptimizer,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        bool useStreamingDefaults,
        double fallbackLearningRate)
    {
        if (streamingOptimizer is IStreamingOptimizerLearningRate mutable)
        {
            mutable.SetLearningRate(useStreamingDefaults
                ? fallbackLearningRate
                : ResolveLearningRate(optimizer, fallbackLearningRate));
        }
    }

    private static double ResolveLearningRate(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        double fallbackLearningRate)
    {
        if (optimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> gradientBased)
        {
            return gradientBased.GetCurrentLearningRate();
        }

        return fallbackLearningRate;
    }
}
