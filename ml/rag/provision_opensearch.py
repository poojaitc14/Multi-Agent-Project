"""Provision the OpenSearch Serverless collection for refund-policy RAG
(project-plan.md Q54), local dev, dev-prefixed (Q51's convention).

Creates (idempotently -- safe to re-run): an encryption security policy, a
network security policy (public access, simplest for a solo dev project),
the VECTORSEARCH collection itself, and a data access policy granting the
current caller full access. Waits for the collection to become ACTIVE and
prints its endpoint.
"""

import os
import time

import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
COLLECTION_NAME = os.environ.get("OPENSEARCH_COLLECTION_NAME", "refund-policy-dev")

client = boto3.client("opensearchserverless", region_name=AWS_REGION)
sts = boto3.client("sts", region_name=AWS_REGION)


def caller_principal() -> str:
    """Resolves the current caller to a principal ARN usable in a data
    access policy. For an SSO-assumed role, that's the underlying IAM role
    ARN (data access policies match on the role, not the session)."""
    identity = sts.get_caller_identity()
    arn = identity["Arn"]
    if ":assumed-role/" in arn:
        # arn:aws:sts::<acct>:assumed-role/<role-name>/<session> -> the role's IAM ARN
        account = identity["Account"]
        role_name = arn.split(":assumed-role/")[1].split("/")[0]
        iam = boto3.client("iam", region_name=AWS_REGION)
        try:
            role = iam.get_role(RoleName=role_name)
            return role["Role"]["Arn"]
        except Exception as e:
            print(f"Could not resolve exact role ARN via IAM ({e!r}), falling back to a guess")
            return f"arn:aws:iam::{account}:role/{role_name}"
    return arn


def ensure_encryption_policy():
    try:
        client.create_security_policy(
            name=COLLECTION_NAME,
            type="encryption",
            policy=f'{{"Rules":[{{"ResourceType":"collection","Resource":["collection/{COLLECTION_NAME}"]}}],"AWSOwnedKey":true}}',
        )
        print(f"Created encryption policy '{COLLECTION_NAME}'")
    except client.exceptions.ConflictException:
        print(f"Encryption policy '{COLLECTION_NAME}' already exists")


def ensure_network_policy():
    try:
        client.create_security_policy(
            name=COLLECTION_NAME,
            type="network",
            policy=f'[{{"Rules":[{{"ResourceType":"collection","Resource":["collection/{COLLECTION_NAME}"]}},{{"ResourceType":"dashboard","Resource":["collection/{COLLECTION_NAME}"]}}],"AllowFromPublic":true}}]',
        )
        print(f"Created network policy '{COLLECTION_NAME}'")
    except client.exceptions.ConflictException:
        print(f"Network policy '{COLLECTION_NAME}' already exists")


def ensure_collection():
    try:
        client.create_collection(name=COLLECTION_NAME, type="VECTORSEARCH")
        print(f"Requested collection '{COLLECTION_NAME}' (type=VECTORSEARCH)")
    except client.exceptions.ConflictException:
        print(f"Collection '{COLLECTION_NAME}' already exists")

    for _ in range(30):
        details = client.batch_get_collection(names=[COLLECTION_NAME])["collectionDetails"]
        if details and details[0]["status"] == "ACTIVE":
            return details[0]
        time.sleep(10)
    raise TimeoutError("Collection did not become ACTIVE in time")


def ensure_data_access_policy(principal: str):
    policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{COLLECTION_NAME}"],
                    "Permission": ["aoss:*"],
                },
                {
                    "ResourceType": "index",
                    "Resource": [f"index/{COLLECTION_NAME}/*"],
                    "Permission": ["aoss:*"],
                },
            ],
            "Principal": [principal],
        }
    ]
    import json

    try:
        client.create_access_policy(name=COLLECTION_NAME, type="data", policy=json.dumps(policy))
        print(f"Created data access policy '{COLLECTION_NAME}' for {principal}")
    except client.exceptions.ConflictException:
        client.update_access_policy(
            name=COLLECTION_NAME,
            type="data",
            policy=json.dumps(policy),
            policyVersion=client.get_access_policy(name=COLLECTION_NAME, type="data")["accessPolicyDetail"][
                "policyVersion"
            ],
        )
        print(f"Updated existing data access policy '{COLLECTION_NAME}' for {principal}")


def main():
    principal = caller_principal()
    print(f"Principal for data access policy: {principal}")
    ensure_encryption_policy()
    ensure_network_policy()
    ensure_data_access_policy(principal)
    collection = ensure_collection()
    print(f"\nCollection ACTIVE: {collection['name']}")
    print(f"Endpoint: {collection['collectionEndpoint']}")
    return collection


if __name__ == "__main__":
    main()
