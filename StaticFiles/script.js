const ws = new WebSocket('ws://127.0.0.1:8000/ws');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);  // 서버로부터 받은 JSON 문자열을 객체로 변환
    displayMessage(data.message);  // 메시지 표시 함수 호출
};

sendBtn.addEventListener('click', function() {
    const message = userInput.value.trim();
    if (message) {
        console.log(`Sending message to server: ${message}`);  // 서버로 전송하는 메시지 로깅
        ws.send(JSON.stringify({ message: message }));
        userInput.value = '';
    }
});

function displayMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}
