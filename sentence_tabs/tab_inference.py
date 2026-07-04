"""
Inference tab page.

Three reference audio modes:
  1. Select from recorded sentences
  2. Upload custom reference audio
  3. Live recording as reference

Calls the upstream inference API (api_v2.py) via HTTP.
"""

import os
import time
import gradio as gr
import requests

from sentence_recorder.model_utils import (start_inference_api, stop_inference_api,
                                          is_api_running, set_inference_weights,
                                          set_refer_audio, scan_gpt_weights,
                                          scan_sovits_weights, get_upstream_path)
from sentence_recorder.mapping import MappingManager
from sentence_recorder.recorder import RecordingManager
from sentence_recorder.project_manager import ProjectManager
from sentence_recorder.state import AppState


API_PORT = 9880
API_BASE = f"http://127.0.0.1:{API_PORT}"


def create_inference_tab():
    """Create the inference tab UI and return components."""

    _recorder = RecordingManager()

    # ==================== Helper Functions ====================

    def get_recorded_sentences() -> list:
        """Get list of recorded sentences for dropdown."""
        project = AppState.get_current_project()
        mapping_path = ProjectManager.get_mapping_path(project)
        if not os.path.exists(mapping_path):
            return []
        mm = MappingManager()
        mm.load(mapping_path)
        entries = mm.get_confirmed()
        choices = []
        for e in entries:
            display = f"#{e.idx}: {e.text[:50]}..."
            choices.append((display, str(e.idx)))
        return choices

    def get_model_choices() -> tuple:
        """Get GPT and SoVITS model choices."""
        gpt_weights = scan_gpt_weights()
        sovits_weights = scan_sovits_weights()
        gpt_choices = [(w, w) for w in gpt_weights]
        sovits_choices = [(w, w) for w in sovits_weights]
        return gpt_choices, sovits_choices

    # ==================== Build UI ====================

    with gr.Column() as infer_tab:
        # ─── Reference Audio Section ───
        with gr.Group():
            gr.Markdown("### Reference Audio Settings")

            ref_source = gr.Radio(
                choices=["Select from recordings", "Upload custom audio", "Live record reference"],
                value="Select from recordings",
                label="Reference audio source"
            )

            # Mode 1: From recordings
            with gr.Column(visible=True) as ref_from_recordings:
                ref_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select a recorded sentence",
                    info="Recorded sentences from your current project"
                )
                ref_text_from_dropdown = gr.Textbox(
                    value="",
                    label="Reference text (auto-filled)",
                    interactive=False,
                    lines=2
                )

            # Mode 2: Upload custom
            with gr.Column(visible=False) as ref_upload:
                ref_audio_upload = gr.Audio(
                    type="filepath",
                    label="Upload reference audio (WAV)"
                )
                ref_text_input = gr.Textbox(
                    value="",
                    label="Reference text (leave empty for ASR recognition)",
                    placeholder="Type the text in the reference audio, or leave empty for ASR",
                    lines=2
                )
                btn_asr = gr.Button("🎤 ASR Recognize", variant="secondary")

            # Mode 3: Live record
            with gr.Column(visible=False) as ref_live:
                btn_live_record = gr.Button("🔴 Click to Record (3s auto-stop)", variant="secondary")
                btn_live_play = gr.Button("▶ Play recorded reference", interactive=False)
                btn_live_redo = gr.Button("🔄 Re-record", interactive=False)
                ref_live_text = gr.Textbox(
                    value="",
                    label="Reference text for live recording",
                    placeholder="Type the content of what you just recorded",
                    lines=2
                )

        # ─── Model Selection ───
        with gr.Group():
            gr.Markdown("### Model Selection")
            with gr.Row():
                gpt_model = gr.Dropdown(
                    choices=[], label="GPT Model",
                    info="Select GPT weights"
                )
                sovits_model = gr.Dropdown(
                    choices=[], label="SoVITS Model",
                    info="Select SoVITS weights"
                )
            btn_refresh_models = gr.Button("🔄 Refresh model list")

        # ─── Inference Parameters ───
        with gr.Group():
            gr.Markdown("### TTS Parameters")

            tts_text = gr.Textbox(
                value="",
                label="Text to synthesize",
                placeholder="Enter the text you want to convert to speech...",
                lines=3
            )

            with gr.Row():
                text_lang = gr.Dropdown(
                    choices=["zh", "en", "jp", "kr", "all_zh", "auto"],
                    value="zh", label="Text language"
                )
                prompt_lang = gr.Dropdown(
                    choices=["zh", "en", "jp", "kr", "all_zh", "auto"],
                    value="zh", label="Reference language"
                )

            with gr.Row():
                top_k = gr.Slider(minimum=1, maximum=100, value=15, step=1, label="Top-K")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Top-P")
                temperature = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.05, label="Temperature")
                speed = gr.Slider(minimum=0.6, maximum=1.65, value=1.0, step=0.05, label="Speed")

            with gr.Row():
                cut_method = gr.Dropdown(
                    choices=["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"],
                    value="cut5", label="Text split method"
                )
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

            with gr.Row():
                streaming = gr.Checkbox(value=False, label="Streaming output")
                super_sampling = gr.Checkbox(value=False, label="Super sampling")
                parallel_infer = gr.Checkbox(value=True, label="Parallel inference")

        # ─── Synthesis Control ───
        with gr.Group():
            btn_synthesize = gr.Button("🎧 Synthesize", variant="primary")
            audio_output = gr.Audio(label="Synthesized audio", type="filepath")
            btn_save = gr.Button("💾 Save audio", variant="secondary")

        # Status
        api_status = gr.Markdown("**API Status:** Unknown")

    # ==================== State for live recording ====================
    _live_recorded_path = ""

    # ==================== Event Handlers ====================

    def toggle_ref_source(source: str):
        """Show/hide reference source UI panels."""
        vis_from = gr.update(visible=(source == "Select from recordings"))
        vis_upload = gr.update(visible=(source == "Upload custom audio"))
        vis_live = gr.update(visible=(source == "Live record reference"))
        return vis_from, vis_upload, vis_live

    def refresh_dropdowns():
        """Refresh recorded sentences and model dropdowns."""
        # Recorded sentences
        choices = get_recorded_sentences()
        if not choices:
            choices = [("(no recordings)", "0")]

        # Models
        gpt_choices, sovits_choices = get_model_choices()
        if not gpt_choices:
            gpt_choices = [("(no models found)", "")]
        if not sovits_choices:
            sovits_choices = [("(no models found)", "")]

        # API status
        running = is_api_running(API_PORT)
        status = "**API Status:** 🟢 Running" if running else "**API Status:** 🔴 Stopped"

        return {
            ref_dropdown: gr.update(choices=choices),
            gpt_model: gr.update(choices=gpt_choices),
            sovits_model: gr.update(choices=sovits_choices),
            api_status: status,
        }

    def on_select_recording(idx_str: str):
        """When user selects a recorded sentence, auto-fill text."""
        if not idx_str or idx_str == "0":
            return ""
        idx = int(idx_str)
        project = AppState.get_current_project()
        mapping_path = ProjectManager.get_mapping_path(project)
        mm = MappingManager()
        mm.load(mapping_path)
        entry = mm.get(idx)
        if entry:
            return entry.text
        return ""

    def on_synthesize(
        ref_source_val, selected_idx_str, upload_audio, upload_text,
        live_text, gpt, sovits, text, text_lang_val, prompt_lang_val,
        top_k_val, top_p_val, temp_val, speed_val, cut, seed_val,
        streaming_val, super_sampling_val, parallel
    ):
        """Synthesize speech using the inference API."""
        # Ensure API is running
        if not is_api_running(API_PORT):
            status = start_inference_api(API_PORT)
            if not status:
                return [None, "**API Status:** 🔴 Failed to start API"]
            time.sleep(3)  # Wait for model loading

        # Get reference audio path
        ref_path = ""
        ref_text = ""
        if ref_source_val == "Select from recordings" and selected_idx_str and selected_idx_str != "0":
            idx = int(selected_idx_str)
            project = AppState.get_current_project()
            mapping_path = ProjectManager.get_mapping_path(project)
            mm = MappingManager()
            mm.load(mapping_path)
            entry = mm.get(idx)
            if entry:
                ref_path = os.path.join(ProjectManager.get_project_dir(project), entry.wav_path)
                ref_text = entry.text

        elif ref_source_val == "Upload custom audio":
            ref_path = upload_audio if upload_audio else ""
            ref_text = upload_text if upload_text else ""

        elif ref_source_val == "Live record reference":
            global _live_recorded_path
            ref_path = _live_recorded_path
            ref_text = live_text if live_text else ""

        if not ref_path or not os.path.exists(ref_path):
            return [None, "**Status:** Reference audio not found or not set"]

        if not text.strip():
            return [None, "**Status:** Please enter text to synthesize"]

        # Set weights
        if gpt:
            try:
                requests.get(f"{API_BASE}/set_gpt_weights", params={
                    "weights_path": get_upstream_path(gpt)
                }, timeout=10)
            except Exception as e:
                print(f"[WARN] Failed to set GPT weights: {e}")

        if sovits:
            try:
                requests.get(f"{API_BASE}/set_sovits_weights", params={
                    "weights_path": get_upstream_path(sovits)
                }, timeout=10)
            except Exception as e:
                print(f"[WARN] Failed to set SoVITS weights: {e}")

        # Set refer audio
        try:
            requests.get(f"{API_BASE}/set_refer_audio", params={
                "ref_audio_path": ref_path,
                "prompt_text": ref_text,
                "prompt_lang": prompt_lang_val,
            }, timeout=10)
        except Exception as e:
            print(f"[WARN] Failed to set reference audio: {e}")

        # Call TTS
        try:
            response = requests.post(f"{API_BASE}/tts", json={
                "text": text,
                "text_lang": text_lang_val,
                "ref_audio_path": ref_path,
                "prompt_text": ref_text,
                "prompt_lang": prompt_lang_val,
                "top_k": top_k_val,
                "top_p": top_p_val,
                "temperature": temp_val,
                "speed_factor": speed_val,
                "text_split_method": cut,
                "batch_size": 1,
                "streaming_mode": streaming_val,
                "media_type": "wav",
                "seed": int(seed_val),
                "parallel_infer": parallel,
                "super_sampling": super_sampling_val,
            }, timeout=120)

            if response.status_code == 200:
                # Save to temp file
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "projects", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                output_path = os.path.join(temp_dir, f"tts_output_{int(time.time())}.wav")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return [output_path, "**API Status:** 🟢 Running - Synthesis complete"]
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                return [None, f"**API Status:** 🔴 TTS failed: {error_text}"]

        except Exception as e:
            return [None, f"**API Status:** 🔴 API error: {str(e)}"]

    def on_live_record():
        """Start/cancel live recording for reference."""
        global _live_recorded_path
        if _recorder.is_recording:
            result = _recorder.stop_recording()
            if result and result["path"]:
                _live_recorded_path = result["path"]
                return [gr.update(value="🔴 Click to Record (3s auto-stop)", variant="secondary"),
                        gr.update(interactive=True), gr.update(interactive=True)]
            return [gr.update(value="🔴 Click to Record (3s auto-stop)", variant="secondary"),
                    gr.update(interactive=False), gr.update(interactive=False)]
        else:
            # Start recording
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "projects", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            save_path = os.path.join(temp_dir, f"live_ref_{int(time.time())}.wav")
            success = _recorder.start_recording(save_path)
            if success:
                return [gr.update(value="⏹ Stop Recording", variant="stop"),
                        gr.update(interactive=False), gr.update(interactive=False)]
            return [gr.update(value="🔴 Click to Record (3s auto-stop)", variant="secondary"),
                    gr.update(interactive=False), gr.update(interactive=False)]

    def on_asr(audio_path: str):
        """ASR recognition for uploaded audio."""
        if not audio_path:
            return "No audio file provided"
        # Placeholder - real ASR would call upstream FasterWhisper
        # For now, just notify
        return "ASR: Audio received. Please enter text manually, or ASR will be implemented in a future update."

    def save_audio(audio_path: str):
        """Save synthesized audio to project directory."""
        if not audio_path or not os.path.exists(audio_path):
            return "No audio to save"
        project = AppState.get_current_project()
        save_dir = ProjectManager.get_recorded_dir(project)
        save_name = f"synthesized_{int(time.time())}.wav"
        save_path = os.path.join(save_dir, save_name)
        import shutil
        shutil.copy2(audio_path, save_path)
        return f"Saved to {save_path}"

    # ==================== Wire Events ====================

    ref_source.change(
        fn=toggle_ref_source,
        inputs=[ref_source],
        outputs=[ref_from_recordings, ref_upload, ref_live]
    )

    ref_dropdown.change(
        fn=on_select_recording,
        inputs=[ref_dropdown],
        outputs=[ref_text_from_dropdown]
    )

    btn_refresh_models.click(
        fn=refresh_dropdowns,
        inputs=[],
        outputs=[ref_dropdown, gpt_model, sovits_model, api_status]
    )

    btn_synthesize.click(
        fn=on_synthesize,
        inputs=[
            ref_source, ref_dropdown, ref_audio_upload, ref_text_input,
            ref_live_text, gpt_model, sovits_model, tts_text,
            text_lang, prompt_lang, top_k, top_p, temperature, speed,
            cut_method, seed, streaming, super_sampling, parallel_infer
        ],
        outputs=[audio_output, api_status]
    )

    btn_live_record.click(
        fn=on_live_record,
        inputs=[],
        outputs=[btn_live_record, btn_live_play, btn_live_redo]
    )

    btn_asr.click(
        fn=on_asr,
        inputs=[ref_audio_upload],
        outputs=[ref_text_input]
    )

    btn_save.click(
        fn=save_audio,
        inputs=[audio_output],
        outputs=[]
    )

    # Return components
    return {
        "infer_tab": infer_tab,
        "ref_source": ref_source,
        "ref_dropdown": ref_dropdown,
        "gpt_model": gpt_model,
        "sovits_model": sovits_model,
        "btn_refresh_models": btn_refresh_models,
        "api_status": api_status,
        "audio_output": audio_output,
    }
