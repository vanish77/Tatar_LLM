"""
Gradio ���-��������� ��� ��������� LLM
���������: python app.py
��� ���������� �� Hugging Face Spaces
"""

import torch
import gradio as gr
from utils import load_model, load_tokenizer, generate_text


# ���������� ���������� ��� ������
model = None
tokenizer = None
tok_config = None
device = None


def initialize_model():
    """�������������� ������ ��� �������"""
    global model, tokenizer, tok_config, device
    
    print("�������� ������...")
    
    # ���������� ����������
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    # ��������� ������ � �����������
    model, _ = load_model("models/best_model.pt", device)
    tokenizer, tok_config = load_tokenizer("tokenizer")
    
    print(f"������ ��������� �� ����������: {device}")
    return True


def generate(
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    num_samples: int = 1
):
    """
    ���������� ����� �� ������ �������
    """
    if not prompt:
        return "����������, ������� ������ �� ��������� �����."
    
    try:
        results = []
        for _ in range(num_samples):
            text = generate_text(
                model, tokenizer, tok_config, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            results.append(text)
        
        # ���������� ����������
        if num_samples == 1:
            return results[0]
        else:
            return "\n\n---\n\n".join(f"**������� {i+1}:**\n{text}" 
                                      for i, text in enumerate(results))
    
    except Exception as e:
        return f"������ ���������: {str(e)}"


# ������� ��������
examples = [
    ["����� �???��", 40, 0.8, 40, 1],
    ["���������", 50, 0.7, 40, 1],
    ["����� ����", 40, 0.8, 40, 1],
    ["���", 30, 0.9, 40, 2],
]


# ������ Gradio ���������
def create_interface():
    """������ Gradio ���������"""
    
    with gr.Blocks(title="��������� LLM", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # ???? ��������� LLM
        
        ��������� �������� ������, ��������� � ���� �� ��������� �����.
        
        **�����������**: GPT (decoder-only, 6 ����, 512 dim, ~23M ����������)  
        **�����������**: Rotary Embeddings, Mixed Precision, MPS support
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="������ �� ��������� �����",
                    placeholder="����� �???��...",
                    lines=3
                )
                
                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="����. ���������� �������"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="�����������"
                    )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=40,
                        step=1,
                        label="Top-k"
                    )
                    
                    num_samples_slider = gr.Slider(
                        minimum=1,
                        maximum=3,
                        value=1,
                        step=1,
                        label="���������� ���������"
                    )
                
                generate_btn = gr.Button("?? ������������", variant="primary")
            
            with gr.Column(scale=2):
                output_text = gr.Markdown(
                    label="��������������� �����"
                )
        
        # �������
        gr.Examples(
            examples=examples,
            inputs=[
                prompt_input,
                max_tokens_slider,
                temperature_slider,
                top_k_slider,
                num_samples_slider
            ],
            label="������� ��������"
        )
        
        # ����������
        gr.Markdown("""
        ---
        ### ?? ������ �� �������������:
        
        - **����������� 0.3-0.5**: ����� ������������� ���������
        - **����������� 0.7-0.9**: ���������������� ��������� (�������������)
        - **����������� 1.0-1.5**: ����� ���������� � ������������� ���������
        
        - **Top-k 10-20**: ����� ��������������� ���������
        - **Top-k 40-50**: ���������������� ��������� (�������������)
        - **Top-k 100+**: ����������� ������������� ���������
        
        ### ?? � �������:
        
        ��� ������ ������� � ���� �� ��������� ������� �� Leipzig Wortschatz.
        ����������� ����������� �����������: Rotary Position Embeddings, 
        Mixed Precision Training, ��������� Apple Silicon (MPS).
        
        **GitHub**: [������ �� ��� �����������]
        """)
        
        # ���������� �������
        generate_btn.click(
            fn=generate,
            inputs=[
                prompt_input,
                max_tokens_slider,
                temperature_slider,
                top_k_slider,
                num_samples_slider
            ],
            outputs=output_text
        )
    
    return demo


if __name__ == "__main__":
    # �������������� ������
    initialize_model()
    
    # ������ � ��������� ���������
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # ���������� True ��� ��������� ������
    )


