import gradio as gr
import numpy as np
from itertools import combinations


# Gradio components for the banning tab
def banning_interface(row_ban, col_ban):
    """Handles the banning tab."""
    return 0
    result = ban_deck(row_ban, col_ban)
    return str(result)


# Gradio components for the picking tab
def picking_interface(pick_size):
    """Handles the picking tab."""
    return 0
    result = compute_best_picking_policy(pick_size)
    return result


# Gradio app with two tabs
with gr.Blocks() as demo:
    gr.Markdown("## Deck Management WebUI")

    with gr.Tab("Banning"):
        gr.Markdown("### Ban Decks (Remove Rows and Columns)")
        row_input = gr.Number(label="Row to Ban (Index)", value=0, precision=0)
        col_input = gr.Number(label="Column to Ban (Index)", value=0, precision=0)
        ban_button = gr.Button("Ban")
        ban_output = gr.Textbox(label="Modified Win Rate Matrix")
        ban_button.click(banning_interface, inputs=[row_input, col_input], outputs=ban_output)

    with gr.Tab("Picking"):
        gr.Markdown("### Compute Best Picking Policy")
        pick_size_input = gr.Number(label="Number of Decks to Pick", value=3, precision=0)
        pick_button = gr.Button("Compute Best Policy")
        pick_output = gr.Textbox(label="Best Picking Result")
        pick_button.click(picking_interface, inputs=[pick_size_input], outputs=pick_output)


if __name__ == "__main__":
    demo.launch(share=True)
