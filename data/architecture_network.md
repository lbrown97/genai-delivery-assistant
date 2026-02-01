# Architecture - Network Segmentation
- Datastores run in private subnets.
- No direct public access to Qdrant or databases.
- API and UI are exposed via internal load balancer in production.
