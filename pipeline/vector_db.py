import milvus_lite
from pymilvus import MilvusClient


def create_milvus_client(uri: str, db_name: str) -> MilvusClient:
    client = MilvusClient(uri=uri, db_name=db_name)
    return client

def create_local_milvus_client(db_name: str) -> MilvusClient:
    if not db_name.startswith(("./", "/")):
        db_name = f"./{db_name}"
    client = MilvusClient(uri=db_name)
    return client

def create_milvus_collection(client: MilvusClient, collection_name: str, dimension: int):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )
    print(f"Collection '{collection_name}' created with dimension {dimension}.")