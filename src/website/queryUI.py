import os 
from pathlib import Path
import gradio as gr

from ..main import tandem_dimple
from .SAV_handler import _read_text_file_safely, process_sav_txt, handle_sav_input
from .STR_handler import process_structure_txt

MAX_BYTES = 100000

def upload_file(filepath, _type='SAV'):
    """
    Input: Gradio UploadedFile or a path str (depends on version).
    Output: (UploadButton update, DownloadButton update, message string)
    """
    
    save_dir = Path("uploads")
    save_dir.mkdir(exist_ok=True)
    dest = save_dir / Path(filepath).name
    if not os.path.exists(filepath):
        msg = f"❌ No file received."
        state = False

    # Size guard
    try:
        size = os.path.getsize(filepath)
        if size > MAX_BYTES:
            msg = f"❌ File too large ({size} bytes). Max allowed is {MAX_BYTES} bytes."
            state = False
    except OSError:
        pass
    
    if _type == 'SAV':
        # Read & parse
        txt = _read_text_file_safely(filepath)
        msg, state = process_sav_txt(txt)
    elif _type == 'STR': # structure file
        msg, state = f"✅ Received 1 file.", True
        pass
    
    return [
        gr.update(visible=True, label="Re-upload file"),
        f"✅ **{dest.name}** uploaded!\n\n{msg}",
        state
    ]

def UI_SAVinput():
    gr.Markdown("UniProt ID with Single Amino Acid Variant (SAV)")
    with gr.Row(elem_classes="sav-query"):
        
        sav_txt_state = gr.State(False)
        sav_btn_state = gr.State(False)
        
        with gr.Column(scale=5, min_width=320):
            sav_txt = gr.Textbox(
                show_label=False,
                placeholder="O14508 52 S N\nP29033 217 Y D\n...",
                max_lines=5,
                lines=4,
                elem_id="sav-txt",
            )
            sav_txt_msg = gr.Markdown()
            sav_txt.change(process_sav_txt, sav_txt, [gr.State(False), sav_txt_msg, sav_txt_state])

        with gr.Column(scale=3, min_width=200):
            sav_btn = gr.UploadButton(
                label="Upload file",
                file_count="single",
                elem_id="sav-btn",
                file_types=[".txt"],
                
            )
            sav_btn_msg = gr.Markdown("Upload a text file (≤150KB)", elem_id="btn-msg")
            sav_btn.upload(upload_file, [sav_btn, gr.State('SAV')], [sav_btn, sav_btn_msg, sav_btn_state])
    
    return sav_txt, sav_txt_state, sav_btn, sav_btn_state

def UI_STRinput():
    
    def _toggle(checked: bool):
        return [
            gr.update(visible=checked),
            gr.update(interactive=True),
            gr.update(interactive=True),
        ]
        
    checkbox = gr.Checkbox(
        label="Use customized structure",
        value=False, # unchecked by default
    )
    with gr.Row(visible=True, elem_classes="custom-str") as custom_str:

        str_txt_state = gr.State(False)
        str_btn_state = gr.State(False)
        with gr.Column(scale=5, min_width=320):
            str_txt = gr.Textbox(
                show_label=False,
                placeholder="PDB ID (e.g., 1G0D) or AF2 ID (e.g., O14508)",
                lines=1,
                elem_id="custom-str-txt",
                interactive=False,
            )
            str_txt_msg = gr.Markdown()
            str_txt.change(process_structure_txt, str_txt, [str_txt_msg, str_txt_state])
            
        with gr.Column(scale=3, min_width=200):
            str_btn = gr.UploadButton(
                label="Upload file",
                file_count="single",
                file_types=[".cif", ".pdb"],
                elem_id="button",
                interactive=False,
            )
            str_btn_msg = gr.Markdown("Upload a customized structure file (.cif/.pdb, ≤150KB)", elem_id="btn-msg")
            str_btn.upload(upload_file, [str_btn, gr.State('STR')], [str_btn, str_btn_msg, str_btn_state])
    
    checkbox.change(_toggle, checkbox, [custom_str, str_txt, str_btn])
    return str_txt, str_txt_state, str_btn, str_btn_state

# def submit_job(
#         session_id,
#         sav_txt, sav_txt_state, sav_btn, sav_btn_state,
#         str_txt, str_txt_state, str_btn, str_btn_state
#     ):
    
#     if sav_btn_state:
#         SAV_input = sav_btn
#     elif sav_txt_state:
#         SAV_input = sav_txt
#     else:
#         SAV_input = None
    
#     SAV_input = handle_sav_input(SAV_input)
    
#     if str_btn_state:
#         STR_input = str_btn
#     elif str_txt_state:
#         STR_input = str_txt
#     else:
#         STR_input = None
    
#     td = tandem_dimple(
#         query=SAV_input,
#         job_name=str(session_id),
#         custom_PDB=STR_input,
#         refresh=False,
#     )  