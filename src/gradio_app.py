# -*- coding: utf-8 -*-

import json
import logging
import os

import alibabacloud_oss_v2 as oss
import alibabacloud_oss_v2.vectors as oss_vectors
import dashscope
import gradio as gr
from PIL import Image
from dashscope import MultiModalEmbeddingItemText

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Util:
    access_key_id = os.environ.get('oss_test_access_key_id')
    access_key_secret = os.environ.get('oss_test_access_key_secret')
    region = os.environ.get('oss_test_region')
    account_id = os.environ.get('oss_test_account_id')

    cfg = oss.config.load_default()
    cfg.credentials_provider = oss.credentials.StaticCredentialsProvider(access_key_id, access_key_secret)
    cfg.region = region
    cfg.account_id = account_id
    client = oss_vectors.Client(cfg)

    vector_bucket_name = "my-test-2"
    vector_index_name = "test1"
    dimension = 1024

    @staticmethod
    def embedding(text) -> list[float]:
        return dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=[MultiModalEmbeddingItemText(text=text, factor=1.0)]
        ).output["embeddings"][0]["embedding"]

    @staticmethod
    def query_text(text: str, top_k: int = 5, city: list[str] = None, height: list[str] = None, return_meta: bool = True, return_distance: bool = True) -> list[tuple[Image.Image, str]]:
        logger.info(f"search text:{text}, top_k:{top_k}, city:{city}, height:{height}")

        sub_filter = []
        if city is not None and len(city) > 0:
            sub_filter.append({"city": {"$in": city}})
        if height is not None and len(height) > 0:
            sub_filter.append({"height": {"$in": height}})
        if len(sub_filter) > 0:
            filter_body = {"$and": sub_filter}
        else:
            filter_body = None

        result = Util.client.query_vectors(oss_vectors.models.QueryVectorsRequest(
            bucket=Util.vector_bucket_name,
            index_name=Util.vector_index_name,
            query_vector={
                "float32": Util.embedding(text)
            },
            filter=filter_body,
            top_k=top_k,
            return_distance=return_distance,
            return_metadata=return_meta,
        ))

        gallery_data = []
        current_dir = os.path.dirname(os.path.abspath(__file__))

        for vector in result.vectors:
            file_path = os.path.join(current_dir, "../data/photograph/", vector["key"])
            img = Image.open(file_path)
            gallery_data.append((img, json.dumps(vector)))
        ret = gallery_data
        logger.info(f"search text:{text}, top_k:{top_k}, request_id:{result.request_id}, ret:{ret}")
        return ret

    @staticmethod
    def on_gallery_box_select(evt: gr.SelectData):
        result = ""
        img_data = evt.value["caption"]
        img_data = json.loads(img_data)
        for key in img_data:
            img_data_item = img_data[key]
            if type(img_data_item) is str:
                img_data_item = img_data_item.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
            if type(img_data_item) is dict:
                for sub_key in img_data_item:
                    img_data_item[sub_key] = img_data_item[sub_key].replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
                    result += f' - **{sub_key}**: &nbsp; {img_data_item[sub_key]}\r\n'
                continue
            result += f' - **{key}**: &nbsp; {img_data_item}\r\n'
        return result


with gr.Blocks(title="OSS Demo") as demo:
    with gr.Tab("OSS QueryVector 图片示例") as search_tab:
        with gr.Row():
            query_text_box = gr.Textbox(label='query_text', interactive=True, value="狗狗")
            top_k_box = gr.Slider(minimum=1, maximum=30, value=10, step=1, label='top_k', interactive=True)
            with gr.Column():
                return_meta_box = gr.Checkbox(label='return_meta', interactive=True, value=True)
                return_distance_box = gr.Checkbox(label='return_distance', interactive=True, value=True)
        with gr.Row():
            city_box = gr.Dropdown(label='city', multiselect=True, choices=["hangzhou", "shanghai", "beijing", "shenzhen", "guangzhou"])
            height_box = gr.Dropdown(label='height', multiselect=True, choices=["1024", "683", "768", "576"])
        with gr.Row():
            query_button = gr.Button(value="query", variant='primary')
        with gr.Row():
            with gr.Column(scale=8):
                gallery_box = gr.Gallery(columns=5, show_label=False, preview=False, allow_preview=False, visible=True, show_download_button=False)
            with gr.Column(scale=2):
                with gr.Row(variant="panel"):
                    md_box = gr.Markdown(visible=True, elem_classes="image_detail")
            gallery_box.select(Util.on_gallery_box_select, [], [md_box])
        query_button.click(
            Util.query_text,
            inputs=[
                query_text_box,
                top_k_box,
                city_box,
                height_box,
                return_meta_box,
                return_distance_box
            ],
            outputs=[
                gallery_box,
            ],
            concurrency_limit=1,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
