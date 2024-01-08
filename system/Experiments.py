import subprocess
models = ["TextCNN","fastText"]
localsteps = [2,16]
algorithms = ["FedProx"]
batchsize = [32]
for i in models:
    for j in localsteps:
        for k in algorithms:
            for l in batchsize:
                subprocess.run([
                    "python3", "main.py", f"--model={i}", f"--local_steps={j}", f"--algorithm={k}", f"--batch_size={l}" 
                ])
