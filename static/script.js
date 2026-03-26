const sessionId = "session_" + Math.random().toString(36).substr(2, 9);
const messagesArea = document.getElementById("messages-area");
const inputField = document.getElementById("chat-input");
const typingIndicator = document.createElement("div");

// Setup typing indicator
typingIndicator.className = "typing-indicator message";
typingIndicator.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';

inputField.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

function sendQuickMessage(text) {
    inputField.value = text;
    sendMessage();
}

async function sendMessage() {
    const text = inputField.value.trim();
    if (!text) return;

    // Add user message to UI
    appendMessage(text, 'user-message');
    inputField.value = "";
    
    // Show typing...
    messagesArea.appendChild(typingIndicator);
    typingIndicator.style.display = "block";
    scrollToBottom();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                query: text
            })
        });

        const data = await response.json();
        
        // Hide typing...
        typingIndicator.style.display = "none";
        
        if (response.ok) {
            // Format citations if it's the raw RAG context (by splitting on 'Source:' or formatting nicely)
            let formattedText = data.response;
            if (formattedText.includes("Source: ")) {
                formattedText = formattedText.replace(/Source: ([^\n|]+)/g, '<br><span class="citation">Source: $1</span>');
            }
            appendHTMLMessage(formattedText, 'bot-message');
        } else {
            appendMessage("Sorry, I encountered an error connecting to the backend API.", 'bot-message');
        }
    } catch (error) {
        typingIndicator.style.display = "none";
        appendMessage("Network error. Make sure the backend API is running.", 'bot-message');
    }
}

function appendMessage(text, className) {
    const div = document.createElement("div");
    div.className = "message " + className;
    div.textContent = text;
    messagesArea.appendChild(div);
    scrollToBottom();
}

function appendHTMLMessage(htmlContent, className) {
    const div = document.createElement("div");
    div.className = "message " + className;
    // Replace newlines with <br> for HTML rendering, but avoid double breaking existing HTML tags
    div.innerHTML = htmlContent.replace(/\n\n/g, '<br><br>').replace(/\n(?![<])/g, '<br>');
    messagesArea.appendChild(div);
    scrollToBottom();
}

function scrollToBottom() {
    messagesArea.scrollTop = messagesArea.scrollHeight;
}
