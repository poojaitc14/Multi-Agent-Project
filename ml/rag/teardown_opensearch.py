"""Tears down everything provision_opensearch.py created: the collection
and its 3 policies. Real, billable resources -- run this to stop the
OCU-based cost floor between work sessions."""

from provision_opensearch import COLLECTION_NAME, client


def main():
    try:
        details = client.batch_get_collection(names=[COLLECTION_NAME])["collectionDetails"]
        if details:
            client.delete_collection(id=details[0]["id"])
            print(f"Deleted collection '{COLLECTION_NAME}' (id={details[0]['id']})")
        else:
            print(f"Collection '{COLLECTION_NAME}' already gone")
    except client.exceptions.ResourceNotFoundException:
        print(f"Collection '{COLLECTION_NAME}' already gone")

    for policy_type in ("data", ):
        try:
            client.delete_access_policy(name=COLLECTION_NAME, type=policy_type)
            print(f"Deleted {policy_type} access policy '{COLLECTION_NAME}'")
        except client.exceptions.ResourceNotFoundException:
            print(f"{policy_type} access policy '{COLLECTION_NAME}' already gone")

    for policy_type in ("encryption", "network"):
        try:
            client.delete_security_policy(name=COLLECTION_NAME, type=policy_type)
            print(f"Deleted {policy_type} security policy '{COLLECTION_NAME}'")
        except client.exceptions.ResourceNotFoundException:
            print(f"{policy_type} security policy '{COLLECTION_NAME}' already gone")


if __name__ == "__main__":
    main()
