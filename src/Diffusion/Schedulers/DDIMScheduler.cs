using System;

using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.Schedulers
{
    // Minimal DDIM scheduler supporting epsilon prediction and linear beta schedule.
    public sealed class DDIMScheduler<T> : IStepScheduler<T>
    {
        private readonly SchedulerConfig<T> _config;
        private readonly INumericOperations<T> _ops;

        private T[] _betas = Array.Empty<T>();
        private T[] _alphas = Array.Empty<T>();
        private T[] _alphasCumprod = Array.Empty<T>();

        public int[] Timesteps { get; private set; }

        public DDIMScheduler(SchedulerConfig<T> config)
        {
            _config = config ?? throw new ArgumentNullException("config");
            _ops = MathHelper.GetNumericOperations<T>();
            InitializeTrainSchedule();
            Timesteps = Array.Empty<int>();
        }

        private void InitializeTrainSchedule()
        {
            int steps = _config.TrainTimesteps;
            _betas = new T[steps];
            if (_config.BetaSchedule == BetaSchedule.Linear)
            {
                // Linear interpolation between BetaStart and BetaEnd
                for (int i = 0; i < steps; i++)
                {
                    // beta = start + (end-start) * i/(steps-1)
                    var delta = _ops.Subtract(_config.BetaEnd, _config.BetaStart);
                    var ratio = _ops.Divide(_ops.FromDouble(i), _ops.FromDouble(steps - 1));
                    _betas[i] = _ops.Add(_config.BetaStart, _ops.Multiply(delta, ratio));
                }
            }
            else
            {
                throw new NotSupportedException("Beta schedule not supported");
            }

            _alphas = new T[steps];
            _alphasCumprod = new T[steps];
            var cum = _ops.One;
            for (int i = 0; i < steps; i++)
            {
                _alphas[i] = _ops.Subtract(_ops.One, _betas[i]);
                cum = _ops.Multiply(cum, _alphas[i]);
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

        public Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
        {
            if (modelOutput == null) throw new ArgumentNullException("modelOutput");
            if (sample == null) throw new ArgumentNullException("sample");
            if (modelOutput.Length != sample.Length) throw new ArgumentException("modelOutput and sample length mismatch");
            if (timestep < 0 || timestep >= _alphasCumprod.Length) throw new ArgumentOutOfRangeException("timestep");

            int t = timestep;
            int prevT = Math.Max(t - 1, 0);

            T ac = _alphasCumprod[t];
            T acPrev = _alphasCumprod[prevT];
            T sqrtAc = _ops.Sqrt(ac);
            T oneMinusAc = _ops.Subtract(_ops.One, ac);
            T sqrtOneMinusAc = _ops.Sqrt(oneMinusAc);

            int n = sample.Length;
            var predOriginal = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                // epsilon prediction
                var term = _ops.Subtract(sample[i], _ops.Multiply(sqrtOneMinusAc, modelOutput[i]));
                predOriginal[i] = _ops.Divide(term, sqrtAc);
            }

            if (_config.ClipSample)
            {
                for (int i = 0; i < n; i++)
                {
                    predOriginal[i] = MathHelper.Clamp(predOriginal[i], _ops.Negate(_ops.One), _ops.One);
                }
            }

            // DDIM variance
            T sigma = _ops.Zero;
            if (_ops.GreaterThan(eta, _ops.Zero))
            {
                // var = eta * sqrt((1 - acPrev) / (1 - ac) * (1 - ac/acPrev))
                var oneMinusAcPrev = _ops.Subtract(_ops.One, acPrev);
                var ratio = _ops.Divide(oneMinusAcPrev, oneMinusAc);
                var frac = _ops.Subtract(_ops.One, _ops.Divide(ac, acPrev));
                var inside = _ops.Multiply(ratio, frac);
                sigma = _ops.Multiply(eta, _ops.Sqrt(inside));
            }

            var prevSample = new Vector<T>(n);
            T coeffOrig = _ops.Sqrt(acPrev);
            T sigmaSq = _ops.Multiply(sigma, sigma);
            T coeffEps = _ops.Sqrt(_ops.Subtract(_ops.Subtract(_ops.One, acPrev), sigmaSq));

            var noiseVec = noise;
            if (_ops.GreaterThan(sigma, _ops.Zero) && noiseVec == null)
            {
                // Deterministic fallback if noise not supplied
                noiseVec = new Vector<T>(new T[n]);
            }

            for (int i = 0; i < n; i++)
            {
                T termOrig = _ops.Multiply(coeffOrig, predOriginal[i]);
                T termEps = _ops.Multiply(coeffEps, modelOutput[i]);
                T termNoise = _ops.Zero;
                if (_ops.GreaterThan(sigma, _ops.Zero))
                {
                    termNoise = _ops.Multiply(sigma, noiseVec![i]);
                }
                prevSample[i] = _ops.Add(_ops.Add(termOrig, termEps), termNoise);
            }

            return prevSample;
        }
    }
}
