import chi
import chi.server

# 1. Configure Site
chi.use_site("KVM@TACC")

key_name = "my-chi-key"  # Make sure this matches your KVM@TACC key pair name

print("Booting H100 VM directly (no lease needed!)...")

server = chi.server.create_server(
    server_name="test-cpu-vm",
    image_name="CC-Ubuntu22.04",
    flavor_name="m1.medium",
    key_name=key_name,
    network_name="sharednet1"
)

# 3. Wait for boot and assign IP
chi.server.wait_for_active(server.id)
ip_address = chi.server.associate_floating_ip(server.id)

print(f"VM is active!")
print(f"Connect using: ssh cc@{ip_address} -i /path/to/your/downloaded/key.pem")