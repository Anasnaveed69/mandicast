# Deploying MandiCast to AWS EC2

The deployment is one EC2 instance running the `docker-compose.yml` stack. GitHub Actions builds the images, pushes them to GHCR, and rolls them out over SSH.

## 1. Launch the instance

- Ubuntu 24.04, **t3.medium** (4 GB RAM) recommended. t3.small works because the setup script adds 4 GB of swap, but training will be slow.
- 20 GB gp3 disk.
- Security group: inbound 22 from your IP, 80 from anywhere.
- Optional: attach an IAM role with read/write access to the S3 bucket used by DVC. With a role you don't need AWS keys in `.env`.
- Allocate an Elastic IP so the public address doesn't change between restarts.

## 2. Prepare the server (once)

```bash
ssh -i key.pem ubuntu@<ip> 'bash -s' < deploy/ec2-setup.sh
```

This installs Docker and the compose plugin, adds swap, clones the repo to `~/mandicast`, and creates `.env` with a random `ADMIN_TOKEN` and Postgres password. Log out and back in afterwards so the docker group applies.

## 3. GitHub secrets

Repo → Settings → Secrets and variables → Actions:

| Secret | Value |
|---|---|
| `EC2_HOST` | public IP or DNS of the instance |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | contents of the private key (`.pem`) |

Also create an environment named `production` (Settings → Environments). You can add required reviewers there if you want a manual approval before each deploy.

If `EC2_HOST` isn't set, `cd` still publishes the images but skips the deploy step.

After the first `cd` run, open the four `mandicast-*` packages on your GitHub profile and set them to **public**. The server can then pull them without logging in, for example when you run `deploy.sh` by hand.

## 4. Data

Choose one option:

- **DVC remote on S3.** Set `DVC_REMOTE_URL=s3://<bucket>/mandicast` (and keys, if there's no IAM role) in `~/mandicast/.env`. The pipeline job runs `dvc pull` before training and `dvc push` after it.
- **Copy the CSV.** `scp data/raw/amis_lahore.csv ubuntu@<ip>:~/mandicast/data/raw/`

## 5. First deploy and training

```bash
git push origin main                 # ci -> cd -> deployed (or on the server: bash deploy/deploy.sh)
ssh ubuntu@<ip>
cd ~/mandicast
docker compose --profile train run --rm pipeline   # dvc repro -> MLflow registry -> API reload
curl localhost/api/ready
```

On a fresh server the MLflow registry is empty. When that happens, the pipeline job forces the `register` stage, even if `dvc repro` finds nothing else to run. From then on, `retrain.yml` repeats this every Sunday.

## Operating it

| Task | Command |
|---|---|
| status | `docker compose ps` |
| logs | `docker compose logs -f api` (or `scheduler`, `mlflow`, ...) |
| MLflow UI | `ssh -L 5000:localhost:5000 ubuntu@<ip>`, then open http://localhost:5000 |
| refresh data now | `docker compose exec scheduler python -m mandicast.jobs refresh` |
| reload models | `curl -X POST -H "X-Admin-Token: $ADMIN_TOKEN" localhost/api/admin/reload` |
| roll back | `IMAGE_TAG=<older sha> docker compose up -d` (every image is tagged with its commit) |

Public endpoints: `http://<ip>/` (UI), `http://<ip>/api/docs` (API docs), `http://<ip>/api/v1/forecast`.
