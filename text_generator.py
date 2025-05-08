import customtkinter as ctk
from transformers import GPT2LMHeadModel, GPT2Tokenizer




model_name = "sberbank-ai/rugpt3small_based_on_gpt2"  # Русский язык
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


def generate_text():
    input_text = entry.get().strip()
    if not input_text:
        output_textbox.delete("1.0", "end")
        output_textbox.insert("1.0", "Введите промпт для генерации!")
        return

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    output_textbox.delete("1.0", "end")
    output_textbox.insert("1.0", generated_text)


# GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Генератор текста на GPT-2")
app.geometry("800x600")

# Основной фрейм
frame = ctk.CTkFrame(app)
frame.pack(pady=20, padx=20, fill="both", expand=True)

label = ctk.CTkLabel(frame, text="Введите промпт:", font=("Arial", 14))
label.pack(pady=10)

entry = ctk.CTkEntry(frame, width=600, height=40)
entry.pack()


button = ctk.CTkButton(
    frame,
    text="Сгенерировать текст",
    command=generate_text,
    corner_radius=10,
    font=("Arial", 14)
)
button.pack(pady=20)

output_textbox = ctk.CTkTextbox(
    frame,
    width=600,
    height=300,
    wrap="word",
    font=("Arial", 12)
)
output_textbox.pack()

app.mainloop()


