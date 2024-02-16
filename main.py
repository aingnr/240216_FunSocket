#1st!zzz
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

# OpenAI API 키 환경 변수에서 검색
openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = "gpt-3.5-turbo"

app = FastAPI()

# 정적 파일을 위한 설정
app.mount("/static", StaticFiles(directory=r"C:\\wiz\\240216_TEST\\StaticFiles"), name="static")

@app.get("/")
async def get_root():
    # HTML 반환
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            data_dict = json.loads(data)  # 받은 데이터를 JSON 객체로 변환
            response_text = await process_data(data_dict)  # 사용자 입력 처리
            await websocket.send_text(response_text)  # 처리 결과를 클라이언트에 전송
        except WebSocketDisconnect:
            print("WebSocket connection closed")
            break

async def process_data(data):
    # ChatGPT 인스턴스 생성
    chatgpt = ChatOpenAI(
            openai_api_key=openai_api_key, 
            model=model_name,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0
        )

    # 임베딩 및 벡터 데이터베이스 설정
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma(persist_directory="chroma\\0129_malcang1\\0129_mal", embedding_function=embeddings)

    question = ", ".join(data.values())

    # 문서 검색 및 처리
    try:
        docs = vectordb.similarity_search(question)
        docs_str = docs[0].page_content if docs else "관련 문서가 없습니다."
    except Exception as e:
        docs_str = "검색 중 오류가 발생했습니다."

    # CSV 파일 처리
    file_path = '0129_dog.csv'

    df = pd.read_csv(file_path, encoding='utf-8' if os.path.exists(file_path) else 'cp949')

    # 임시 파일 생성
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)

    # 에이전트 생성 및 질문 처리
    try:
        agent = create_csv_agent(chatgpt, temp_file.name, agent_type=AgentType.OPENAI_FUNCTIONS)
        response = agent.invoke(question + "내용에 적합한 '제품 이름' 2개를 '제품 특성' 기준으로 추천해줘")
    except Exception as e:
        response = "질문 처리 중 오류가 발생했습니다."

    # 최종 답변 생성
    template = f"""
            유저의 질문: {question}
            관련 문서 내용: {docs_str}
            CSV 데이터를 바탕으로 생성된 질문: {response}
            
            위 내용을 바탕으로 아래 마크다운 형식으로 구성된 답변을 제시해 주어야 해.
        
            [건강 레시피]
            1.수의사의 진단
            2.집에서 할 수 있는 케어 프로그램
            3.추천 영양성분
            4.추천 사료 제품
            5.맺음말
        
            위에서 '4.추천 사료 제품'은 {response} 내용 중 '제품 이름'과 '제품 특성'을 반드시 적용해 주어야 해.
            이때, 추천된 2개 제품은 1), 2) 형식으로 내용을 표현해줘.
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
