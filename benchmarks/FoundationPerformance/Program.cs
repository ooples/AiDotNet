using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;

bool gpu = args.Contains("gpu");
using var backend = gpu ? new OpenClBackend() : null;
if (gpu && !backend!.IsAvailable) throw new Exception("OpenCL required");
using var direct = gpu ? (DirectGpuEngine)Activator.CreateInstance(typeof(DirectGpuEngine), BindingFlags.Instance | BindingFlags.NonPublic, null, new object[]{backend!}, null)! : null;
using var device = gpu ? new DirectGpuTensorEngine(direct!) : null;
IEngine engine = device is null ? new CpuEngine() : device;
AiDotNetEngine.Current = engine;
const int warmup = 5, repeats = 21;
var results = new List<object>();
void Measure(string name, Action step, Func<double> checksum, object shape, Action synchronize)
{
    for (int i=0;i<warmup;i++) { step(); synchronize(); }
    var times = new double[repeats]; var bytes = new long[repeats];
    for (int i=0;i<repeats;i++)
    {
        backend?.Synchronize(); long startBytes=GC.GetTotalAllocatedBytes(true);
        var timer=Stopwatch.StartNew(); step(); synchronize(); timer.Stop();
        times[i]=timer.Elapsed.TotalMilliseconds; bytes[i]=GC.GetTotalAllocatedBytes(true)-startBytes;
    }
    long downloads, uploads;
    using(var scope=GpuResidencyScope.Begin()) { step(); backend?.Synchronize(); downloads=scope.Downloads; uploads=scope.Uploads; }
    Array.Sort(times); Array.Sort(bytes);
    results.Add(new { name, shape, median_ms=times[repeats/2], min_ms=times[0], max_ms=times[^1], allocated_bytes=bytes[repeats/2], downloads, uploads, checksum=checksum() });
}
foreach (var shape in new[] { new[]{4,8,16,32}, new[]{16,32,32,64} })
{
    int b=shape[0], t=shape[1], inputs=shape[2], h=shape[3];
    var x=new Tensor<float>(new[]{b,t,inputs});
    for(int i=0;i<x.Length;i++) x[i]=(float)Math.Sin(i*.01)*.1f;
    var layer=new LSTMLayer<float>(h); layer.SetTrainingMode(true); layer.Forward(x);
    var weights=layer.GetTrainableParameters();
    foreach(var w in weights) for(int i=0;i<w.Length;i++) w[i]=.01f;
    // Reset the layer's packed caches through its supported parameter mutation contract.
    layer.SetParameters(layer.GetParameters());
    Tensor<float>? output=null; Dictionary<Tensor<float>,Tensor<float>>? gradients=null;
    var sources=new[]{x}.Concat(weights).ToArray();
    void Forward() { layer.ResetState(); output=layer.Forward(x); }
    void Train() { using var tape=new GradientTape<float>(); Forward(); gradients=tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(output!,output!),null),sources); }
    Measure("LSTMLayer.forward",Forward,()=>output!.ToArray().Sum(v=>(double)v),shape,()=>{if(gpu) output!.ToArray();});
    Measure("LSTMLayer.forward_backward",Train,()=>gradients![x].ToArray().Sum(v=>(double)v),shape,()=>{if(gpu) gradients![x].ToArray();});
}
if (!gpu)
{
    int b=16,t=32,inputs=32,h=64;
    var x=new Tensor<double>(new[]{b,t,inputs});
    for(int i=0;i<x.Length;i++) x[i]=Math.Sin(i*.01)*.1;
    var layer=new LSTMLayer<double>(h); layer.SetTrainingMode(true); layer.Forward(x);
    var weights=layer.GetTrainableParameters(); foreach(var w in weights) for(int i=0;i<w.Length;i++) w[i]=.01;
    layer.SetParameters(layer.GetParameters());
    Tensor<double>? output=null; Dictionary<Tensor<double>,Tensor<double>>? gradients=null;
    var sources=new[]{x}.Concat(weights).ToArray();
    void Forward(){layer.ResetState();output=layer.Forward(x);}
    void Train(){using var tape=new GradientTape<double>();Forward();gradients=tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(output!,output!),null),sources);}
    Measure("LSTMLayer.float64.forward",Forward,()=>output!.ToArray().Sum(),new[]{b,t,inputs,h},()=>{});
    Measure("LSTMLayer.float64.forward_backward",Train,()=>gradients![x].ToArray().Sum(),new[]{b,t,inputs,h},()=>{});
}
foreach(var shape in new[]{new[]{8,64},new[]{2,16,16,32}})
{
    int hidden=shape[^1]; var gateShape=(int[])shape.Clone(); gateShape[^1]*=4;
    var gates=new Tensor<float>(gateShape);var old=new Tensor<float>(shape);
    for(int i=0;i<gates.Length;i++) gates[i]=(float)Math.Sin(i*.01)*.1f;
    for(int i=0;i<old.Length;i++) old[i]=.1f;
    var method=typeof(IEngine).GetMethods().SingleOrDefault(m=>m.Name=="LstmCellFromGates")?.MakeGenericMethod(typeof(float));
    Tensor<float>? output=null; Dictionary<Tensor<float>,Tensor<float>>? gradients=null;
    void Forward()
    {
        Tensor<float> h,c;
        if(method is not null){var a=new object?[]{gates,old,null};h=(Tensor<float>)method.Invoke(engine,a)!;c=(Tensor<float>)a[2]!;}
        else {Tensor<float> G(int k)=>engine.TensorNarrow(gates,shape.Length-1,k*hidden,hidden);
            c=engine.TensorAdd(engine.TensorMultiply(engine.Sigmoid(G(1)),old),engine.TensorMultiply(engine.Sigmoid(G(0)),engine.Tanh(G(2))));
            h=engine.TensorMultiply(engine.Sigmoid(G(3)),engine.Tanh(c));}
        output=engine.TensorAdd(h,c);
    }
    void Train(){using var tape=new GradientTape<float>();Forward();gradients=tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(output!,output!),null),new[]{gates,old});}
    Measure("projected_gates.forward_backward",Train,()=>gradients![gates].ToArray().Sum(v=>(double)v),shape,()=>{if(gpu) gradients![gates].ToArray();});
}
var live=Enumerable.Range(0,8).Select(_=>new Tensor<float>(new[]{256,256})).ToArray();
foreach(var p in live) engine.TensorCopy(engine.TensorAddScalar(p,.1f),p);
var snapshotType=typeof(LSTMLayer<float>).Assembly.GetType("AiDotNet.Training.TensorCheckpoint`1")?.MakeGenericType(typeof(float));
Tensor<float>[]? saved=null;
var capture=snapshotType?.GetMethod("Capture"); var restore=snapshotType?.GetMethod("Restore");
Action checkpoint;
if(capture is not null)
    checkpoint=()=>{saved=(Tensor<float>[])capture.Invoke(null,new object?[]{live,saved})!;restore!.Invoke(null,new object[]{live,saved});};
else
    checkpoint=()=>{var packed=live.SelectMany(p=>p.ToArray()).ToArray();int offset=0;foreach(var p in live){var values=new float[p.Length];Array.Copy(packed,offset,values,0,p.Length);engine.TensorCopy(new Tensor<float>(values,p.Shape.ToArray()),p);offset+=p.Length;}};
Measure("checkpoint.capture_restore",checkpoint,()=>live[0].ToArray().Sum(v=>(double)v),new[]{8,256,256},()=>{if(gpu) live[0].ToArray();});
Console.WriteLine(JsonSerializer.Serialize(new{backend=gpu?"OpenCL":"CPU",warmup,repeats,results},new JsonSerializerOptions{WriteIndented=true}));
