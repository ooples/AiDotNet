using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Continuum Memory System (CMS) for nested learning.
/// Provides a spectrum of memory modules operating at different frequencies.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class ContinuumMemorySystem<T> : IContinuumMemorySystem<T>
{
    private readonly int _numFrequencyLevels;
    private readonly int _memoryDimension;
    private Vector<T>[] _memoryStates;
    private T[] _decayRates;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public ContinuumMemorySystem(int memoryDimension, int numFrequencyLevels = 3, T[]? decayRates = null)
    {
        _memoryDimension = memoryDimension;
        _numFrequencyLevels = numFrequencyLevels;

        _memoryStates = new Vector<T>[numFrequencyLevels];
        for (int i = 0; i < numFrequencyLevels; i++)
        {
            _memoryStates[i] = new Vector<T>(memoryDimension);
        }

        _decayRates = decayRates ?? CreateDefaultDecayRates();
    }

    private T[] CreateDefaultDecayRates()
    {
        var rates = new T[_numFrequencyLevels];
        for (int i = 0; i < _numFrequencyLevels; i++)
        {
            double rate = 0.9 + (i * 0.05);
            rates[i] = _numOps.FromDouble(Math.Min(rate, 0.99));
        }
        return rates;
    }

    public void Store(Vector<T> representation, int frequencyLevel)
    {
        if (frequencyLevel < 0 || frequencyLevel >= _numFrequencyLevels)
            throw new ArgumentException($"Invalid frequency level: {frequencyLevel}");

        T decay = _decayRates[frequencyLevel];
        T oneMinusDecay = _numOps.Subtract(_numOps.One, decay);

        var currentMemory = _memoryStates[frequencyLevel];
        var updated = new Vector<T>(_memoryDimension);

        for (int i = 0; i < Math.Min(_memoryDimension, representation.Length); i++)
        {
            T decayed = _numOps.Multiply(currentMemory[i], decay);
            T newVal = _numOps.Multiply(representation[i], oneMinusDecay);
            updated[i] = _numOps.Add(decayed, newVal);
        }

        _memoryStates[frequencyLevel] = updated;
    }

    public Vector<T> Retrieve(Vector<T> query, int frequencyLevel)
    {
        if (frequencyLevel < 0 || frequencyLevel >= _numFrequencyLevels)
            throw new ArgumentException($"Invalid frequency level: {frequencyLevel}");

        return _memoryStates[frequencyLevel];
    }

    public void Update(Vector<T> context, bool[] updateMask)
    {
        if (updateMask.Length != _numFrequencyLevels)
            throw new ArgumentException("Update mask length must match number of frequency levels");

        for (int i = 0; i < _numFrequencyLevels; i++)
        {
            if (updateMask[i])
            {
                Store(context, i);
            }
        }
    }

    public void Consolidate()
    {
        for (int i = 0; i < _numFrequencyLevels - 1; i++)
        {
            var fastMemory = _memoryStates[i];
            var slowMemory = _memoryStates[i + 1];

            double transferRateVal = 0.05 / (i + 1);
            T transferRate = _numOps.FromDouble(transferRateVal);
            T oneMinusTransfer = _numOps.Subtract(_numOps.One, transferRate);

            var consolidated = new Vector<T>(_memoryDimension);
            for (int j = 0; j < _memoryDimension; j++)
            {
                T slow = _numOps.Multiply(slowMemory[j], oneMinusTransfer);
                T fast = _numOps.Multiply(fastMemory[j], transferRate);
                consolidated[j] = _numOps.Add(slow, fast);
            }

            _memoryStates[i + 1] = consolidated;
        }
    }

    public int NumberOfFrequencyLevels => _numFrequencyLevels;

    public T[] DecayRates
    {
        get => _decayRates;
        set
        {
            if (value.Length != _numFrequencyLevels)
                throw new ArgumentException("Decay rates length must match number of frequency levels");
            _decayRates = value;
        }
    }

    public Vector<T>[] MemoryStates => _memoryStates;

    public void Reset()
    {
        for (int i = 0; i < _numFrequencyLevels; i++)
        {
            _memoryStates[i] = new Vector<T>(_memoryDimension);
        }
    }
}
