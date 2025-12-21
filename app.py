import gradio as gr
from webui import create_TI2I
from webui import Runner
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

def main():
    runner = Runner()

    with gr.Blocks(analytics_enabled=False,
                   title='WGFF',
                   ) as demo:

        md_txt = "# WGFF" \
                 "\nOfficial demo of the paper []()"
        gr.Markdown(md_txt)

        with gr.Tabs(selected='WGFF'):
            with gr.TabItem("WGFF", id='WGFF'):
                create_TI2I(runner=runner)

        demo.launch(share=False, debug=False)

if __name__ == '__main__':
    main()

