from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import os
from dotenv import load_dotenv
import pandas as pd
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_community.vectorstores import Chroma

class ChatApplication:
    def __init__(self, openai_api_key, model_name):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.chatgpt = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model=self.model_name,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

    async def process_data(self, data_dict):
        question = ", ".join(data_dict.values())
        vectordb = Chroma(persist_directory="chroma\\0129_malcang1\\0129_mal", embedding_function=self.embeddings)
        response_text = await self.generate_response(question, vectordb)
        return response_text

    async def generate_response(self, question, vectordb):
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
            response = agent.invoke(question + "내용에 적합한 '제품 이름' 2개를 '제품 특성' 기준으로 추천해줘")
        except Exception:
            response = "An error occurred while processing your question."

        template = self.construct_template(question, docs_str, response)

        try:
            final_answer = self.chatgpt.invoke(template)
            response_content = final_answer.response if hasattr(final_answer, 'response') else "The response could not be processed."
        except Exception:
            final_answer = "An error occurred while generating the answer."

        temp_file.close()
        os.unlink(temp_file.name)
        
        return json.dumps({"message": response_content}, ensure_ascii=False)

    def construct_template(self, question, docs_str, response):
        return f"""
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

# 애플리케이션 초기화 및 실행
if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    model_name = "gpt-3.5-turbo"
    chat_app = ChatApplication(openai_api_key, model_name)
    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            try:
                data = await websocket.receive_text()
                data_dict = json.loads(data)
                response_text = await chat_app.process_data(data_dict)
                await websocket.send_text(response_text)
            except WebSocketDisconnect:
                print("WebSocket connection closed")
                break
