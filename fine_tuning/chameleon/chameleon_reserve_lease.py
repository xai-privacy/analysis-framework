import chi
import chi.lease
import chi.server

# 1. Configure Site
chi.use_site("KVM@TACC")

# 2. Build the reservation list
# We manually construct the dictionary that Blazar expects for a VM flavor
reservations = [{
    "resource_type": "virtual:instance",
    "flavor_id": chi.server.get_flavor_id("g1.h100.vgpu.1g.12gb"),
    "amount": 1,
    "affinity": False
}]

# 3. Calculate times dynamically (Start: Now, End: +6 hours)
start_date, end_date = chi.lease.lease_duration(hours=6)
lease_name = "trinav-xai-lease"

# 4. Submit the lease
print("Submitting lease...")
my_lease = chi.lease.create_lease(
    lease_name,
    reservations,
    start_date=start_date,
    end_date=end_date
)

print(f"VM Lease created successfully! ID: {my_lease['id']}")