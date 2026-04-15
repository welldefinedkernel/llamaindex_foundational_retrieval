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