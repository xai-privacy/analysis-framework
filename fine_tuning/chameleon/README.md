# Chameleon KVM instance

These scripts use the `chi` Python client with the `KVM@TACC` site. We alreay have Security Groups comfigured in KVM to allow SSH connections. We also have the SSH key pair generated and OAuth credentials shared privately. The lease reservation is only needed for Bare Metal servers.

## Launch an instance

For KVM, launch the instance directly, install python-chi first; **you do not need to reserve a lease
first**. From this directory, run:

```bash
source <app-cred-file-path>
python3 launch_server.py
```

The script authenticates, creates a server, waits for it to become active, and associates a
floating IP. Its `key_name` (`my-chi-key`) must match the name of your
Chameleon SSH key pair::

```bash
ssh -i <private key path> cc@<floating-ip>
```

Keep the private key secure and do not commit or share it.

## Optional lease example

[`chameleon_reserve_lease.py`](chameleon_reserve_lease.py) is an example of
creating a lease; it is not a prerequisite for launching a KVM instance. It
shows a reservation for a `g1.h100.vgpu.1g.12gb` flavor, which is separate from
the direct KVM launch configured in [`launch_server.py`](launch_server.py).

The OAuth app-credentials file is shared privately.
It can be used when configuring credentials to run the lease example.

## A Fine-Tuning Example

A LEET-Arg fine-tuning example is below. It uses the prepared 58/15
train/validation split and the CUDA QLoRA scripts in `fine_tuning/trinav/`.

If the transfer has not completed, run these commands from the local
`fine_tuning` directory. Keep your private key local; do not upload it.

```bash
ssh -o StrictHostKeyChecking=no -i my-chi-key.pem \
	cc@<public IP> 'mkdir -p ~/xai'

scp -o StrictHostKeyChecking=no -i my-chi-key.pem -r trinav \
	cc@<public IP>:~/xai/
```

Connect to the VM and verify that the folder arrived:

```bash
ssh -o StrictHostKeyChecking=no -i my-chi-key.pem cc@<public IP>
ls ~/xai/trinav
cd ~/xai/trinav
```

Prepare the Python environment. The system-site-packages option keeps the
CUDA-enabled PyTorch supplied by the Chameleon image visible inside the venv.
The GPU check must pass before continuing.

```bash
nvidia-smi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch/GPU is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install --upgrade pip
python -m pip install -r requirements-chameleon.txt
hf auth login
```

Accept the `meta-llama/Llama-3.2-3B-Instruct` license on Hugging Face before
authenticating. Then fine-tune for 180 steps, evaluating and saving every 20:

```bash
python train_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--train-data data/train.jsonl \
	--valid-data data/valid.jsonl \
	--split data/split.json \
	--output-dir adapters_chameleon \
	--max-steps 180 \
	--eval-steps 20 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

Plot training loss, validation loss, learning rate, and epoch after training:

```bash
python plot_training_metrics.py \
	--trainer-state adapters_chameleon/trainer_state.json
```

Optionally evaluate the base model and the trained adapter on validation data:

```bash
python evaluate_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--format-hint \
	--out results/base_hint.json

python evaluate_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--adapter-path adapters_chameleon \
	--out results/finetuned.json
```
