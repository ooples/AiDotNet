using System;

namespace AiDotNet.Diffusion.Schedulers
{
    // Minimal DDIM scheduler supporting epsilon prediction and linear beta schedule.
    public sealed class DDIMScheduler : IStepScheduler
    {
        private readonly SchedulerConfig _config;

        private double[] _betas = Array.Empty<double>();
        private double[] _alphas = Array.Empty<double>();
        private double[] _alphasCumprod = Array.Empty<double>();

        public int[] Timesteps { get; private set; }

        public DDIMScheduler(SchedulerConfig config)
        {
            _config = config ?? throw new ArgumentNullException("config");
            InitializeTrainSchedule();
            Timesteps = Array.Empty<int>();
        }

        private void InitializeTrainSchedule()
        {
            int T = _config.TrainTimesteps;
            _betas = new double[T];
            if (_config.BetaSchedule == BetaSchedule.Linear)
            {
                double step = (_config.BetaEnd - _config.BetaStart) / (T - 1);
                for (int i = 0; i < T; i++) _betas[i] = _config.BetaStart + step * i;
            }
            else
            {
                throw new NotSupportedException("Beta schedule not supported");
            }

            _alphas = new double[T];
            _alphasCumprod = new double[T];
            double cum = 1.0;
            for (int i = 0; i < T; i++)
            {
                _alphas[i] = 1.0 - _betas[i];
                cum *= _alphas[i];
                _alphasCumprod[i] = cum;
            }
        }

        public void SetTimesteps(int inferenceSteps)
        {
            if (inferenceSteps <= 0 || inferenceSteps > _config.TrainTimesteps)
                throw new ArgumentOutOfRangeException("inferenceSteps");

            int T = _config.TrainTimesteps;
            int stride = T / inferenceSteps;
            if (stride < 1) stride = 1;

            int[] ts = new int[inferenceSteps];
            int idx = 0;
            for (int i = T - 1; i >= 0 && idx < inferenceSteps; i -= stride)
            {
                ts[idx++] = i;
            }
            if (idx < inferenceSteps)
            {
                Array.Resize(ref ts, idx);
            }
            Timesteps = ts;
        }

        public double[] Step(double[] modelOutput, int timestep, double[] sample, double eta = 0.0, double[]? noise = null)
        {
            if (modelOutput == null) throw new ArgumentNullException("modelOutput");
            if (sample == null) throw new ArgumentNullException("sample");
            if (modelOutput.Length != sample.Length) throw new ArgumentException("modelOutput and sample length mismatch");
            if (timestep < 0 || timestep >= _alphasCumprod.Length) throw new ArgumentOutOfRangeException("timestep");

            int t = timestep;
            int prevT = Math.Max(t - 1, 0);

            double ac = _alphasCumprod[t];
            double acPrev = _alphasCumprod[prevT];
            double sqrtAc = Math.Sqrt(ac);
            double sqrtOneMinusAc = Math.Sqrt(1.0 - ac);

            int n = sample.Length;
            var predOriginal = new double[n];
            for (int i = 0; i < n; i++)
            {
                // epsilon prediction
                predOriginal[i] = (sample[i] - sqrtOneMinusAc * modelOutput[i]) / sqrtAc;
            }

            if (_config.ClipSample)
            {
                for (int i = 0; i < n; i++)
                {
                    if (predOriginal[i] > 1) predOriginal[i] = 1;
                    else if (predOriginal[i] < -1) predOriginal[i] = -1;
                }
            }

            // DDIM variance
            double sigma = 0.0;
            if (eta > 0.0)
            {
                double betaT = _betas[t];
                double var = eta * Math.Sqrt((1 - acPrev) / (1 - ac) * (1 - ac / acPrev));
                sigma = var;
            }

            var prevSample = new double[n];
            double coeffOrig = Math.Sqrt(acPrev);
            double coeffEps = Math.Sqrt(1.0 - acPrev - sigma * sigma);

            var noiseVec = noise;
            if (sigma > 0.0 && noiseVec == null)
            {
                // Deterministic fallback if noise not supplied
                noiseVec = new double[n];
            }

            for (int i = 0; i < n; i++)
            {
                double termOrig = coeffOrig * predOriginal[i];
                double termEps = coeffEps * modelOutput[i];
                double termNoise = 0.0;
                if (sigma > 0.0)
                {
                    termNoise = sigma * noiseVec![i];
                }
                prevSample[i] = termOrig + termEps + termNoise;
            }

            return prevSample;
        }
    }
}
