namespace AiDotNet.Tensors.Helpers;

internal sealed class LockedRandom : Random
{
    private readonly object _gate = new object();

    internal LockedRandom(int seed)
        : base(seed)
    {
    }

    public override int Next()
    {
        lock (_gate)
        {
            return base.Next();
        }
    }

    public override int Next(int maxValue)
    {
        lock (_gate)
        {
            return base.Next(maxValue);
        }
    }

    public override int Next(int minValue, int maxValue)
    {
        lock (_gate)
        {
            return base.Next(minValue, maxValue);
        }
    }

    public override double NextDouble()
    {
        lock (_gate)
        {
            return base.NextDouble();
        }
    }

    public override void NextBytes(byte[] buffer)
    {
        lock (_gate)
        {
            base.NextBytes(buffer);
        }
    }

#if NET8_0_OR_GREATER
    public override void NextBytes(Span<byte> buffer)
    {
        lock (_gate)
        {
            base.NextBytes(buffer);
        }
    }

    public override long NextInt64()
    {
        lock (_gate)
        {
            return base.NextInt64();
        }
    }

    public override long NextInt64(long maxValue)
    {
        lock (_gate)
        {
            return base.NextInt64(maxValue);
        }
    }

    public override long NextInt64(long minValue, long maxValue)
    {
        lock (_gate)
        {
            return base.NextInt64(minValue, maxValue);
        }
    }

    public override float NextSingle()
    {
        lock (_gate)
        {
            return base.NextSingle();
        }
    }
#endif

    protected override double Sample()
    {
        lock (_gate)
        {
            return base.Sample();
        }
    }
}
