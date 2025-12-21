import os
from PIL import Image
import gradio as gr


def create_TI2I(runner):
    with gr.Blocks():
        with gr.Row():
            gr.Markdown('1. Upload the content image and set target prompt.\n'
                        '2. Choose the generative model.\n'
                        '3. Cilck `Run` to start generation.')

        with gr.Row():
            with gr.Column():
                gr.Markdown('#### Input Image:\n')
                # input image
                content_image = gr.Image(label='Input Content Image', type='pil')

                # input prompt
                prompt = gr.Textbox(label='Target Prompt', value='')
                

                run_button = gr.Button(value='Run')

            with gr.Column():
                with gr.Accordion('Options', open=True):
                    tau_f = gr.Slider(label='End Steps', minimum=0., maximum=1., value=0.5, step=0.01)
                    unconditional_guidancd_scale = gr.Slider(label='Guidancd Scale', minimum=5, maximum=10, value=7.5, step=0.1)
                    seed = gr.Number(label='Seed', value=2025, precision=0, minimum=0, maximum=2 ** 31)
                    ll_weight = gr.Slider(label='LL weight', minimum=0, maximum=1, value=0.9, step=0.01)
                    lh_weight = gr.Slider(label='LH weight', minimum=0, maximum=1, value=0.1, step=0.01)
                    hl_weight = gr.Slider(label='HL weight', minimum=0, maximum=1, value=0.1, step=0.01)
                    hh_weight = gr.Slider(label='HH weight', minimum=0, maximum=1, value=0.1, step=0.01)

            with gr.Column():
                gr.Markdown('#### Output Image:\n')
                result_gallery = gr.Gallery(label='Output', elem_id='gallery', columns=2, height='auto', preview=True)
                gr.Examples(
                    [
                        [Image.open('assets/1.png').convert('RGB'), 'Eagle'],
                        [Image.open('assets/2.png').convert('RGB'), 'Young lady'],
                        [Image.open('assets/3.png').convert('RGB'), 'Blue Hair'],
                        [Image.open('assets/4.png').convert('RGB'), 'Lion']
                    ],
                    [content_image, prompt]
                )

        input_parameters = [
            content_image, prompt, ll_weight, lh_weight, hl_weight, hh_weight, tau_f, unconditional_guidancd_scale, seed
        ]

        run_button.click(fn=runner.run_WGFF, inputs=input_parameters, outputs=result_gallery)
