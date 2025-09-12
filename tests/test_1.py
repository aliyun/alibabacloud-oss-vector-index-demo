import json
import logging
import os
import random
import time
from collections import defaultdict
from http import HTTPStatus
from typing import List

import alibabacloud_oss_v2.vectors as oss_vectors
import dashscope
from PIL import Image
from dashscope import MultiModalEmbeddingItemImage, MultiModalEmbeddingItemText

from gradio_app import Util as util

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def multi_modal_embedding(image_url: str = None, text: str = None) -> List:
    if image_url is None and text is None:
        raise Exception("image_url and text is None")
    if image_url is not None and text is not None:
        raise Exception("image_url and text is not None")
    if image_url is not None:
        dash_input = [MultiModalEmbeddingItemImage(image=image_url, factor=1.0)]
    else:
        dash_input = [MultiModalEmbeddingItemText(text=text, factor=1.0)]
    resp = dashscope.MultiModalEmbedding.call(model="multimodal-embedding-v1", input=dash_input)
    if resp.status_code == HTTPStatus.OK:
        return resp.output["embeddings"][0]["embedding"]
    else:
        raise Exception(f"{resp.status_code}: {resp.message}")


def test_put_bucket():
    util.client.put_vector_bucket(
        oss_vectors.models.PutVectorBucketRequest(
            bucket=util.vector_bucket_name,
        )
    )


def test_list_bucket():
    buckets = util.client.list_vector_buckets(oss_vectors.models.ListVectorBucketsRequest())
    print([x.name for x in buckets.buckets])


def test_delete_bucket():
    result = util.client.delete_vector_bucket(
        oss_vectors.models.DeleteVectorBucketRequest(
            bucket=util.vector_bucket_name,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")


def test_get_bucket():
    result = util.client.get_vector_bucket(
        oss_vectors.models.GetVectorBucketRequest(
            bucket=util.vector_bucket_name,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")


def test_put_vector_index():
    result = util.client.put_vector_index(
        oss_vectors.models.PutVectorIndexRequest(
            bucket=util.vector_bucket_name,
            index_name=util.vector_index_name,
            dimension=util.dimension,
            data_type="float32",
            distance_metric="cosine",
            metadata={"nonFilterableMetadataKeys": ["key1", "key2"]},
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")


def test_list_vector_index():
    result = util.client.list_vector_indexes(
        oss_vectors.models.ListVectorIndexesRequest(
            bucket=util.vector_bucket_name,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")
    print(result.indexes)
    print([x["indexName"] for x in result.indexes])


def test_delete_vector_index():
    result = util.client.delete_vector_index(
        oss_vectors.models.DeleteVectorIndexRequest(
            bucket=util.vector_bucket_name,
            index_name="t1",
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")


def test_get_vector_index():
    result = util.client.get_vector_index(
        oss_vectors.models.GetVectorIndexRequest(
            bucket=util.vector_bucket_name,
            index_name=util.vector_index_name,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")
    print(result.index)


def test_put_vector():
    vectors = []
    for idx in range(10):
        vectors.append({"key": str(idx), "data": {"float32": [i for i in range(util.dimension)]}, "metadata": {"text": str(idx % 2), "color": "2"}})
    result = util.client.put_vectors(
        oss_vectors.models.PutVectorsRequest(
            bucket=util.vector_bucket_name,
            index_name=util.vector_index_name,
            vectors=vectors,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")


def test_get_vector():
    result = util.client.get_vectors(
        oss_vectors.models.GetVectorsRequest(
            bucket=util.vector_bucket_name,
            index_name=util.vector_index_name,
            keys=["0", "1", "2"],
            return_data=True,
            return_metadata=True,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")
    print(result.vectors)


def test_list_vector():
    result = util.client.list_vectors(
        oss_vectors.models.ListVectorsRequest(
            bucket=util.vector_bucket_name,
            index_name=util.vector_index_name,
            max_results=1000,
            return_data=False,
            return_metadata=True,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")


def test_delete_vector():
    result = util.client.delete_vectors(
        oss_vectors.models.DeleteVectorsRequest(
            bucket=util.vector_bucket_name,
            index_name=util.vector_index_name,
            keys=[str(idx) for idx in range(100)],
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")


def test_query_vectors():
    result = util.client.query_vectors(
        oss_vectors.models.QueryVectorsRequest(
            bucket=util.vector_bucket_name,
            index_name=util.vector_index_name,
            query_vector={"float32": [i * 2222 for i in range(util.dimension)]},
            # filter={
            #     "text": "0"
            # },
            top_k=30,
            return_distance=True,
            return_metadata=False,
        )
    )
    print(f"status code: {result.status_code}, request id: {result.request_id}")
    print(result.vectors)


def test_generate_images():
    prefix = "http://oss-vector-resources.oss-cn-hangzhou.aliyuncs.com/photograph/"
    files = os.listdir("data/photograph")
    info_json = {}
    for file in files[:]:
        time.sleep(0.5)
        logger.info(f"processing {file}")
        img = Image.open("data/photograph/" + file)
        width, height = img.size
        info_json[file] = {
            "image_url": prefix + file,
            "embedding": multi_modal_embedding(image_url=prefix + file),
            "width": str(width),
            "height": str(height),
            "city": random.choice(["hangzhou", "shanghai", "beijing", "shenzhen", "guangzhou"]),
        }
    json.dump(info_json, open("data/info.json", "w"), indent=4, ensure_ascii=False)


def test_images_stats():
    files = os.listdir("data/photograph")
    width_dict = defaultdict(int)
    height_dict = defaultdict(int)
    for file in files[:]:
        img = Image.open("data/photograph/" + file)
        width, height = img.size
        width_dict[width] += 1
        height_dict[height] += 1

    sorted_items = sorted(width_dict.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_items:
        print(f"width_dict {word}: {count}")
    sorted_items = sorted(height_dict.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_items:
        print(f"height_dict {word}: {count}")


def test_generate_v2():
    with open("data/info.json", "r") as info_f, open("data/data.json", "w") as info_w:
        info_json = json.load(info_f)
        data_json = []
        for key in info_json:
            info = info_json[key]
            data_json.append(
                {
                    "key": key,
                    "data": {"float32": info["embedding"]},
                    "metadata": {
                        "image_url": info["image_url"],
                        "width": info["width"],
                        "height": info["height"],
                        "city": info["city"],
                    },
                }
            )
        print(data_json)
        info_w.write(json.dumps(data_json, indent=0, ensure_ascii=False))


def test_search():
    util.query_text(text="狗狗", top_k=5)
