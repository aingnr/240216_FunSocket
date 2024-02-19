from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import pandas as pd
import tempfile
import os
import json

# 환경 변수 로드
load_dotenv()

# 환경 변수에서 OpenAI API 키 가져오기
openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = "gpt-3.5-turbo"

app = FastAPI()

# 정적 파일 설정
app.mount("/static", StaticFiles(directory="C:\\gitcode\\240216_FunSocket\\StaticFiles"), name="static")

@app.get("/")
async def get_root():
    # 루트 엔드포인트에 대한 HTML 반환
    return HTMLResponse("""
    <html>
        <head>
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <div id="chat-container">
                <div id="chat-box"></div>
                <textarea id="user-input" placeholder="Enter your question..."></textarea>
                <button id="send-btn">Send</button>
            </div>
            <script src="/static/script.js"></script>
        </body>
    </html>
    """)

class ChatApplication:
    def __init__(self, openai_api_key, model_name):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.chatgpt = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=model_name,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    async def process_data(self, data_dict):
        question = ", ".join(data_dict.values())
        vectordb = Chroma(persist_directory="chroma\\0129_malcang1\\0129_mal", embedding_function=self.embeddings)

        try:
            docs = vectordb.similarity_search(question)
            docs_str = docs[0].page_content if docs else "There are no related documents."
        except Exception:
            docs_str = "An error occurred while searching."

        file_path = '0129_dog.csv'
        df = pd.read_csv(file_path, encoding='utf-8' if os.path.exists(file_path) else 'cp949')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)

        try:
            agent = create_csv_agent(self.chatgpt, temp_file.name, agent_type=AgentType.OPENAI_FUNCTIONS)
            response = agent.invoke(question + " 내용에 적합한 '제품 이름' 2개를 '제품 특성' 기준으로 추천해줘.")
        except Exception:
            response = "An error occurred while processing your question."

    # 최종 답변 생성
    template = f"""
            유저의 질문: {question}
            관련 문서 내용: {docs_str}
            CSV 데이터를 바탕으로 생성된 질문: {response}
            
            위 내용을 바탕으로 아래 마크다운 형식으로 구성된 답변을 제시해 주어야 해.
        
            [Healthy Recipe]
            1. Veterinarian’s diagnosis
            2. Care program that can be done at home
            3. Recommended nutritional ingredients
            4. Recommended feed products
            5. Conclusion
        
            For '4. Recommended feed products' above, 'Product Name' and 'Product Characteristics' in the {response} content must be applied.
            At this time, please express the contents of the two recommended products in the format 1) and 2).
            """
    try:
        final_answer = chatgpt.invoke(template)
        response_content = final_answer.response if hasattr(final_answer, 'response') else "응답을 처리할 수 없습니다."
    except Exception as e:
        final_answer = "답변 생성 중 오류가 발생했습니다."

    response_text = json.dumps({"message": response_content}, ensure_ascii=False)
    return response_text

    # 임시 파일 삭제
    temp_file.close()
    os.unlink(temp_file.name)

    # return final_answer

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1/", port=8000, reload=True)
