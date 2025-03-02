namespace AiDotNet.NeuralNetworks;

public class Experience<T>
{
    public Vector<T> State { get; }
    public int Action { get; }
    public T Reward { get; }
    public Vector<T> NextState { get; }
    public bool Done { get; }

    public Experience(Vector<T> state, int action, T reward, Vector<T> nextState, bool done)
    {
        State = state;
        Action = action;
        Reward = reward;
        NextState = nextState;
        Done = done;
    }
}