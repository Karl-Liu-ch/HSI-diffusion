import torch
from torch.autograd import Variable
from models.transformer.DT_attn import DTN

device = torch.device('cuda')
model = DTN(3, 31, [128, 128], 8, [2,4], 4, 1).to(device)

def cal_batch_size(model, input_shape):
    freememmory = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    print(freememmory, 'GB')
    batch_size = 1
    input_data = Variable(torch.rand([batch_size, input_shape[0], input_shape[1], input_shape[2]]).to(device))
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
    model.train()
    output = model(input_data)
    output.mean().backward()
    memory_usage = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"] - initial_memory
    print("GPU memory usage for batch size {} is {} GB".format(batch_size, memory_usage / 1024 ** 3))
    print(initial_memory / 1024 ** 3, 'GB')
    max_batch_size = (freememmory - initial_memory / 1024 ** 3) // (memory_usage / 1024 ** 3)
    print(int(max_batch_size))
    return int(max_batch_size)

cal_batch_size(model, [3, 128, 128])