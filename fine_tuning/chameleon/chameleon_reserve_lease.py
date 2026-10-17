import chi
import chi.lease
import chi.server

chi.use_site("KVM@TACC")

reservations = [{
    "resource_type": "flavor:instance",   # was virtual:instance
    "flavor_id": chi.server.get_flavor_id("g1.h100.vgpu.1g.12gb"),
    "amount": 1,
    "affinity": None,                     # matches the "-" in your earlier leases
}]

start_date, end_date = chi.lease.lease_duration(hours=6)

my_lease = chi.lease.create_lease(
    "trinav-xai-lease",
    reservations,
    start_date=start_date,
    end_date=end_date,
)
if my_lease is None:
    raise RuntimeError("Lease creation failed, see the error printed above")

print("Lease ID:", my_lease["id"])