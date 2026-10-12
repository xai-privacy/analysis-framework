# Chameleon KVM instance

These scripts use the `chi` Python client with the `KVM@TACC` site. We alreay have Security Groups comfigured in KVM to allow SSH connections. We also have the SSH key pair generated and OAuth credentials if somebody wants to create a lease (not needed though). The lease rservation is only needed for Bare Metal servers.

## Launch an instance

For KVM, launch the instance directly, install python-chi first; **you do not need to reserve a lease
first**. From this directory, run:

```bash
python3 launch_server.py
```

The script creates a server, waits for it to become active, and associates a
floating IP. Its `key_name` (`my-chi-key`) must match the name of your
Chameleon SSH key pair. The corresponding private SSH key is
[`../trinav/my-chi-key.pem`](../trinav/my-chi-key.pem); use it to connect to the
printed IP:

```bash
ssh -i ../trinav/my-chi-key.pem cc@<floating-ip>
```

Keep the private key secure and do not commit or share it.

## Optional lease example

[`chameleon_reserve_lease.py`](chameleon_reserve_lease.py) is an example of
creating a lease; it is not a prerequisite for launching a KVM instance. It
shows a reservation for a `g1.h100.vgpu.1g.12gb` flavor, which is separate from
the direct KVM launch configured in [`launch_server.py`](launch_server.py).

The optional OAuth app-credentials file is shared privately.
It can be used when configuring credentials to run the lease example. OAuth
credentials and lease reservation are not needed for the direct KVM launch.
