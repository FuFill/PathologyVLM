from __future__ import annotations

import io
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError


_ENDPOINT = os.environ.get(
    "MINIO_ENDPOINT", "https://api.blackhole2.ai.innopolis.university:443"
)
_BUCKET = os.environ.get("MINIO_BUCKET", "pershin-medailab")
_PREFIX = os.environ.get("MINIO_PREFIX", "Pathomorphology/CAMELYON")


def get_s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=_ENDPOINT,
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


def s3_key(path: str) -> str:
    return f"{_PREFIX}/{path}" if not path.startswith(_PREFIX) else path


def read_csv_from_s3(s3_path: str) -> pd.DataFrame:
    client = get_s3_client()
    key = s3_key(s3_path)
    obj = client.get_object(Bucket=_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def upload_to_s3(local_path: str, s3_path: str) -> str:
    client = get_s3_client()
    key = s3_key(s3_path)
    client.upload_file(local_path, _BUCKET, key)
    return f"s3://{_BUCKET}/{key}"


def download_tar_and_extract(
    tar_path: str,
    internal_path: str,
    dest_dir: str | Path,
) -> Optional[Path]:
    client = get_s3_client()
    key = s3_key(tar_path)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    try:
        client.download_file(_BUCKET, key, tmp.name)
        with tarfile.open(tmp.name, "r:gz") as tf:
            member = tf.getmember(internal_path.lstrip("/"))
            tf.extract(member, path=str(dest))
            return dest / member.path
    except (ClientError, KeyError, tarfile.TarError) as exc:
        print(f"[s3_utils] WARNING: could not extract {internal_path} from {tar_path}: {exc}")
        return None
    finally:
        os.unlink(tmp.name)


def get_minio_path_components(minio_path: str) -> tuple[str, str]:
    """Parse minio_path like 's3://bucket/key.tar.gz!/internal/path.png'
    Returns (tar_key, internal_path)
    """
    if "!" in minio_path:
        tar_part, internal = minio_path.split("!", 1)
        tar_part = tar_part.replace("s3://", "").split("/", 1)[1] if "s3://" in tar_part else tar_part
        return tar_part.lstrip("/"), internal.lstrip("/")
    return minio_path, ""
