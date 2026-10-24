# Chameleon KVM instance

We perform fine-tuning using [Chameleon](https://www.chameleoncloud.org/) cloud instances. If you do not have a Chameleon account, contact @SebastianZimmeck.

The scripts in this folder use the `chi` Python client with the `KVM@TACC` site. We already have Security Groups configured in KVM to allow SSH connections. We also have the SSH key pair generated and OAuth credentials shared privately.

**KVM@TACC requires a lease.** GPU flavors (such as `g1.h100.vgpu.1g.12gb`) cannot be booted on demand. You first reserve a lease for the flavor, wait for it to become `ACTIVE`, and then launch the instance against that lease. The workflow is:

1. Preliminary setup
2. Reserve a lease with [`chameleon_reserve_lease.py`](chameleon_reserve_lease.py)
3. Wait until the lease status is `ACTIVE`
4. Launch the instance with [`launch_server.py`](launch_server.py), passing the lease name
5. Run fine-tuning

Instructions are tested for macOS.

## 1. Preliminary setup

We recommend to first create a Python virtual environment:

```bash
python3 -m venv ft_venv
source ft_venv/bin/activate
```

For the Fish shell use `source ft_venv/bin/activate.fish`. (To stop the virtual environment run `deactivate`.)

Install `python-chi` and `chameleon-blazarclient` and place the OAuth app-credentials file in the same directory:

```bash
pip install python-chi
pip install chameleon-blazarclient
source app-cred-python-chi-kvm-openrc.sh
```

The SSH private key, `my-chi-key.pem`, is needed to log in after launching an instance. Place the file in the same directory as the OAuth app-credentials, `app-cred-python-chi-kvm-openrc.sh`, file. The OAuth app-credentials file is needed for OAuth login into Chameleon and using a bare metal server. If you do not have these files, contact @SebastianZimmeck.

## 2. Reserve a lease

From this directory, run:

```bash
python3 chameleon_reserve_lease.py
```

The script reserves one `g1.h100.vgpu.1g.12gb` flavor (`flavor:instance` resource type) for **6 hours**, starting now, under the lease name `trinav-xai-lease`. It prints the lease ID on success. To change the lease name or duration, edit the `create_lease(...)` call and the `lease_duration(hours=...)` argument in the script. Change the amount if you want to launch multiple instances with this lease.

If lease creation fails, the error is printed by `python-chi` and the script exits with `Lease creation failed`. Do not re-run the script after a successful creation, or you will create a duplicate lease.

## 3. Wait for the lease to become ACTIVE

A new lease starts in `PENDING` and usually becomes `ACTIVE` within a minute or two. Check the status in the Chameleon Horizon dashboard for `KVM@TACC` at <https://kvm.tacc.chameleoncloud.org> under Reservations > Leases. Do not continue until the status is `ACTIVE`; `launch_server.py` exits if the lease is in any other state.

## 4. Launch an instance

Pass the lease name to the launch script:

```bash
python3 launch_server.py trinav-xai-lease
```

The script looks up the lease by name, checks that it is `ACTIVE`, picks its active reservation (for flavor reservations, the reservation ID is the flavor ID), creates a server with that flavor, waits for it to become active, and associates a floating IP. Its `key_name` (`my-chi-key`) must match the name of your Chameleon SSH key pair. The GPU-based instances take a few seconds to start up and be ready to connect.

You can find the floating IP at <https://kvm.tacc.chameleoncloud.org> under Network > Floating IPs. Connect with:

```bash
ssh -i my-chi-key.pem cc@<floating-ip>
```

If asked whether you are sure you want to continue connecting, select yes.

If the permissions of your key file are too open, change them with:

```bash
chmod 600 my-chi-key.pem
```

Keep the private key secure and do not commit or share it.

If the host key verification fails, use ssh with StrictHostKeyChecking=no or delete the offending key in the ssh config file and reconnect. Use option `vv` with SSH for more debugging information.

## 5. Run fine-tuning

### 5.1 Lease lifetime and cleanup

- The lease is time-limited (6 hours by default). When it ends, the instance is terminated and anything stored on it is lost. Copy adapters, logs, and results off the VM before the lease expires.
- Launch the instance soon after the lease becomes active so you use as much of the window as possible.
- If you finish early, delete the instance and the lease in Horizon so the reserved capacity and SUs are released.
  You can do so at <https://kvm.tacc.chameleoncloud.org> under Compute > Instances > Delete Instances.
- To use the GPU again after a lease ends, repeat from step 2 above.

### 5.2 A fine-tuning example

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
authenticating. Log back into the same directory again. Then fine-tune for 60 steps, evaluating and saving every 10:

```bash
python train_chameleon.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --train-data data/train.jsonl \
  --valid-data data/valid.jsonl \
  --split data/split.json \
  --output-dir adapters_chameleon_60step \
  --max-steps 60 \
  --eval-steps 10 \
  --learning-rate 1e-4 \
  --max-seq-length 4096
```

Plot training loss, validation loss, learning rate, and epoch after training:

```bash
python plot_training_metrics.py \
  --trainer-state adapters_chameleon_60step/trainer_state.json
```

Optionally evaluate the base model and the trained adapter on validation data:

```bash
python evaluate_chameleon.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --format-hint \
  --out results/base_hint.json

python evaluate_chameleon.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --adapter-path adapters_chameleon_60step \
  --format-hint \
  --out results/finetuned.json
```

Please note that the output directory name will change according to the number of steps.
