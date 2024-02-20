const websocket = new WebSocket('ws://localhost:8000/ws');
const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

function renderAnswer(answerText) {
  const formattedText = answerText.replace(/\n/g, '<br>'); // 줄바꿈을 <br> 태그로 변환
  document.querySelector('.answer-paragraph').innerHTML = formattedText;
}

websocket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', data.sender);

    // 메시지가 'bot'으로부터 온 것이면, renderAnswer 함수를 사용하여 줄바꿈 처리
    if(data.sender === 'bot') {
      renderAnswer(data.message);
    } else {
      messageElement.textContent = data.message;
    }

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // 채팅 박스를 가장 아래로 스크롤
};

sendButton.onclick = function() {
    const message = userInput.value;
    if(message) {
        websocket.send(message);
        userInput.value = ''; // 입력 필드 초기화
    }
};

userInput.addEventListener("keyup", function(event) {
    if (event.keyCode === 13) { // "Enter" 키의 코드는 13입니다.
        event.preventDefault();
        sendButton.click(); // 엔터를 누르면 '보내기' 버튼을 클릭하는 것과 같은 효과
    }
});