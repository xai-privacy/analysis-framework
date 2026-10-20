# Chameleon KVM instance

These scripts use the `chi` Python client with the `KVM@TACC` site. We already have Security Groups configured in KVM to allow SSH connections. We also have the SSH key pair generated and OAuth credentials shared privately.

**KVM@TACC requires a lease.** GPU flavors (such as `g1.h100.vgpu.1g.12gb`) cannot be booted on demand. You first reserve a lease for the flavor, wait for it to become `ACTIVE`, and then launch the instance against that lease. The workflow is:

1. Reserve a lease with [`chameleon_reserve_lease.py`](chameleon_reserve_lease.py)
2. Wait until the lease status is `ACTIVE`
3. Launch the instance with [`launch_server.py`](launch_server.py), passing the lease name

## Setup

Install `python-chi` and load the OAuth app-credentials file (shared privately) in every new shell:

```bash
pip install python-chi
source <app-cred-file-path>
```

## Step 1: Reserve a lease

From this directory, run:

```bash
python3 chameleon_reserve_lease.py
```

The script reserves one `g1.h100.vgpu.1g.12gb` flavor (`flavor:instance` resource type) for **6 hours**, starting now, under the lease name `trinav-xai-lease`. It prints the lease ID on success. To change the lease name or duration, edit the `create_lease(...)` call and the `lease_duration(hours=...)` argument in the script. Change the amount if you want to launch multiple instances with this lease.

If lease creation fails, the error is printed by `python-chi` and the script exits with `Lease creation failed`. Do not re-run the script after a successful creation, or you will create a duplicate lease.

## Step 2: Wait for the lease to become ACTIVE

A new lease starts in `PENDING` and usually becomes `ACTIVE` within a minute or two. Check the status in the Chameleon Horizon dashboard for `KVM@TACC` (Reservations > Leases). Do not continue until the status is `ACTIVE`; `launch_server.py` exits if the lease is in any other state.

## Step 3: Launch an instance

Pass the lease name to the launch script:

```bash
python3 launch_server.py trinav-xai-lease
```

The script looks up the lease by name, checks that it is `ACTIVE`, picks its active reservation (for flavor reservations, the reservation ID is the flavor ID), creates a server with that flavor, waits for it to become active, and associates a floating IP. Its `key_name` (`my-chi-key`) must match the name of your Chameleon SSH key pair. The GPU-based instances take a few seconds to start up and be ready to connect.

Once it prints the floating IP, connect with:

```bash
ssh -i <private key path> cc@<floating-ip>
```

Keep the private key secure and do not commit or share it. If the host key verification fails, use ssh with StrictHostKeyChecking=no or delete the offending key in the ssh config file and reconnect. Use option vv with SSH for more debugging information.

## Lease lifetime and cleanup

- The lease is time-limited (6 hours by default). When it ends, the instance is terminated and anything stored on it is lost. Copy adapters, logs, and results off the VM before the lease expires.
- Launch the instance soon after the lease becomes active so you use as much of the window as possible.
- If you finish early, delete the server and the lease in Horizon so the reserved capacity and SUs are released.
- To use the GPU again after a lease ends, repeat from Step 1.

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
python -m pip install --upgrade pip
python -m pip install -r requirements-chameleon.txt
hf auth login
sudo reboot
```

Accept the `meta-llama/Llama-3.2-3B-Instruct` license on Hugging Face before
authenticating. Log back into the same directory again. Then fine-tune for 180 steps, evaluating and saving every 20:

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