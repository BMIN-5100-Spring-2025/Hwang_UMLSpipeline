import logging
from pathlib import Path
import boto3

def download_from_s3(bucket, prefix, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    local_dir = Path(local_dir)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel_path = Path(key).relative_to(prefix) if key.startswith(prefix) else Path(key)
            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Downloading s3://{bucket}/{key} to {local_path}")
            s3.download_file(bucket, key, str(local_path))

def upload_to_s3(bucket, prefix, local_dir):
    s3 = boto3.client('s3')
    local_dir = Path(local_dir)
    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_dir)
            s3_key = str(Path(prefix) / rel_path)
            logging.info(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
            s3.upload_file(str(file_path), bucket, s3_key) 