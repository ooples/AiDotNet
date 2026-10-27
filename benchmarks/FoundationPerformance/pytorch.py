import argparse, json, math, statistics, time, warnings
import torch
p=argparse.ArgumentParser(); p.add_argument('--gpu',action='store_true'); args=p.parse_args()
torch.set_num_threads(1); torch.set_num_interop_threads(1)
warnings.filterwarnings('error')  # A DirectML CPU fallback invalidates the measurement.
if args.gpu:
    import torch_directml
    device=torch_directml.device(); backend='DirectML explicit recurrence'
else:
    device=torch.device('cpu'); backend='CPU native nn.LSTM'
results=[]
def measure(name, step, shape):
    for _ in range(5): out=step(); out.cpu() if args.gpu else None
    times=[]
    for _ in range(21):
        start=time.perf_counter(); out=step(); out.cpu() if args.gpu else None
        times.append((time.perf_counter()-start)*1000)
    results.append(dict(name=name,shape=shape,median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),checksum=out.detach().cpu().double().sum().item()))
for b,t,i,h in [(4,8,16,32),(16,32,32,64)]:
    shape=[b,t,i,h]
    x=torch.tensor([math.sin(j*.01)*.1 for j in range(b*t*i)],dtype=torch.float32).reshape(b,t,i).to(device).requires_grad_()
    model=torch.nn.LSTM(i,h,batch_first=True).to(device)
    with torch.no_grad():
        for param in model.parameters(): param.fill_(.01)
        model.bias_hh_l0.zero_()
    model.bias_hh_l0.requires_grad_(False)
    def forward():
        if not args.gpu: return model(x)[0]
        hidden=torch.zeros(b,h,device=device); cell=torch.zeros_like(hidden); ys=[]
        for j in range(t):
            ig,fg,cg,og=(x[:,j]@model.weight_ih_l0.T+hidden@model.weight_hh_l0.T+model.bias_ih_l0).chunk(4,dim=-1)
            cell=fg.sigmoid()*cell+ig.sigmoid()*cg.tanh(); hidden=og.sigmoid()*cell.tanh(); ys.append(hidden)
        return torch.stack(ys,dim=1)
    def inference():
        with torch.no_grad(): return forward()
    def train():
        model.zero_grad(set_to_none=True); x.grad=None
        forward().square().sum().backward(); return x.grad
    measure('LSTMLayer.forward',inference,shape)
    measure('LSTMLayer.forward_backward',train,shape)
if not args.gpu:
    b,t,i,h=16,32,32,64
    x=torch.tensor([math.sin(j*.01)*.1 for j in range(b*t*i)],dtype=torch.float64).reshape(b,t,i).requires_grad_()
    model=torch.nn.LSTM(i,h,batch_first=True).double()
    with torch.no_grad():
        for param in model.parameters(): param.fill_(.01)
        model.bias_hh_l0.zero_()
    model.bias_hh_l0.requires_grad_(False)
    measure('LSTMLayer.float64.forward',inference,[b,t,i,h])
    measure('LSTMLayer.float64.forward_backward',train,[b,t,i,h])
for shape in [(8,64),(2,16,16,32)]:
    n=math.prod(shape)*4
    gates=torch.tensor([math.sin(j*.01)*.1 for j in range(n)],dtype=torch.float32).reshape(*shape[:-1],4*shape[-1]).to(device).requires_grad_()
    old=torch.full(shape,.1,device=device,dtype=torch.float32,requires_grad=True)
    def gate_train():
        gates.grad=None;old.grad=None
        ig,fg,cg,og=gates.chunk(4,dim=-1)
        c=fg.sigmoid()*old+ig.sigmoid()*cg.tanh();h=og.sigmoid()*c.tanh()
        (h+c).square().sum().backward();return gates.grad
    measure('projected_gates.forward_backward',gate_train,shape)
live=[torch.full((256,256),.1,device=device,dtype=torch.float32) for _ in range(8)]; saved=[torch.empty_like(x) for x in live]
def checkpoint():
    with torch.no_grad():
        for src,dst in zip(live,saved): dst.copy_(src)
        for src,dst in zip(saved,live): dst.copy_(src)
    return live[0]
measure('checkpoint.capture_restore',checkpoint,[8,256,256])
print(json.dumps(dict(backend=backend,torch=torch.__version__,warmup=5,repeats=21,results=results),indent=2))
