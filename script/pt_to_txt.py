import sys
import torch

full = torch.load(sys.argv[1]) # full_*.pt file
iteration = sys.argv[1].split('.')[0].split('_')[-1]

def save_graph_to_txt():
    model = full
    f=open('MPC_actor'+'_'+iteration+".txt",'w')

    content = ''

    for w in model.items():
        w = w[1] #weight tensor
        if w.ndim == 2:
            w = w.transpose(0,1).flatten()
        for i in w:
            content += str(i.item())+", " # tensor to float

    f.write(content[:-2])

def save_graph_to_txt_full():
    model = full['actor_architecture_state_dict']
    f=open('MPC_actor'+'_'+iteration+".txt",'w')

    content = ''

    for w in model.items():
        w = w[1] #weight tensor
        if w.ndim == 2:
            w = w.transpose(0,1).flatten()
        for i in w:
            content += str(i.item())+", " # tensor to float

    f.write(content[:-2])

save_graph_to_txt_full()
