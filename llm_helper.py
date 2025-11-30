import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def ask_groq_commentary(context_text, style="commentator"):
    """
    呼叫 Groq 生成文字
    Call Groq to generate text
    style: 'commentator' or （主播）或 'strategist' （策略師）
    """
    if style == "commentator":
        system_prompt = (
            "You are a energetic F1 commentator like David Croft. "
            "Speak fast, be dramatic, and use F1 terminology."
        )
    else:
        system_prompt = (
            "You are a calm F1 Strategy Engineer. "
            "Analyze the data logically and explain the tyre strategies."
        )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": context_text,
                },
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=1000,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"AI連線錯誤 Error calling Groq API: {str(e)}")
        return None
