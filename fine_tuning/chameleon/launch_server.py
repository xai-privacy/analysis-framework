## USE AS python3 launch_server.py <lease_name>

import chi
import chi.server
import chi.lease
import sys

# 1. Configure Site
chi.use_site("KVM@TACC")

key_name = "my-chi-key"  # Make sure this matches your KVM@TACC key pair name

print("Booting VM using the provided lease...")

if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <lease_name>")

lease_name = sys.argv[1]
print(f"Booting VM using lease '{lease_name}'...")

leases = [l for l in chi.blazar().lease.list() if l["name"] == lease_name]
if not leases:
    sys.exit(f"No lease named '{lease_name}' at this site/project.")
lease = leases[0]

if lease["status"] != "ACTIVE":
    sys.exit(f"Lease '{lease_name}' is {lease['status']}, not ACTIVE.")

reservations = [r for r in lease["reservations"] if r.get("status") == "active"]
if not reservations:
    sys.exit(f"No active reservations found in lease '{lease_name}'.")

flavor_id = reservations[0]["id"]  # for flavor reservations, reservation ID == flavor ID
print(f"Using flavor ID: {flavor_id}")

# 2. Launch the server using the custom reservation flavor
server = chi.server.create_server(
    server_name="xai-gpu-vm",
    image_name="CC-Ubuntu24.04-CUDA-VGPU",
    flavor_id=flavor_id,
    key_name=key_name,
    network_name="sharednet1"
)

# 3. Wait for boot and assign IP
chi.server.wait_for_active(server.id)
ip_address = chi.server.associate_floating_ip(server.id)

print(f"VM is active!")
print(f"Connect using: ssh cc@{ip_address} -i /path/to/your/downloaded/key.pem")