const API_URL = 'http://localhost:8000';
let sessionId = null;

const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const darkModeToggle = document.getElementById('darkModeToggle');

document.addEventListener('DOMContentLoaded', () => {
    messageInput.addEventListener('input', autoResize);
    messageInput.addEventListener('keydown', handleKeyPress);
    darkModeToggle.addEventListener('click', toggleDarkMode);
    
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
        darkModeToggle.querySelector('.icon').textContent = '☀️';
    }
    
    messageInput.focus();
});

function autoResize() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
}

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    darkModeToggle.querySelector('.icon').textContent = isDark ? '☀️' : '🌙';
    localStorage.setItem('darkMode', isDark ? 'enabled' : 'disabled');
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    messageInput.disabled = true;
    sendButton.disabled = true;
    
    addMessage('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    const typing = addTypingIndicator();
    
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId })
        });
        
        if (!response.ok)
            throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        sessionId = data.session_id;
        
        typing.remove();
        addMessage('assistant', data.reply, data.sources);
        
    } catch (error) {
        typing.remove();
        addMessage('assistant', 'Σφάλμα σύνδεσης με τον server.');
        console.error(error);
    }
    
    messageInput.disabled = false;
    sendButton.disabled = false;
    messageInput.focus();
}

function addMessage(role, content, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message';
    
    const rowDiv = document.createElement('div');
    rowDiv.className = `message-row ${role}-row`;
    
    if (role === 'assistant') {
        const avatar = document.createElement('div');
        avatar.className = 'avatar bot-avatar';
        avatar.textContent = '🤖';
        rowDiv.appendChild(avatar);
    }
    
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${role}-bubble`;
    bubble.innerHTML = formatMessage(content);
    
    rowDiv.appendChild(bubble);
    messageDiv.appendChild(rowDiv);
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.innerHTML = `
            <div class="sources-title">📚 Πηγές:</div>
            ${sources.map(s => `<div class="source-item">• ${escapeHtml(s)}</div>`).join('')}
        `;
        messageDiv.appendChild(sourcesDiv);
    }
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message';
    
    const rowDiv = document.createElement('div');
    rowDiv.className = 'message-row';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar bot-avatar';
    avatar.textContent = '🤖';
    
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator active';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    
    rowDiv.appendChild(avatar);
    rowDiv.appendChild(indicator);
    messageDiv.appendChild(rowDiv);
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

function formatMessage(text) {

    text = escapeHtml(text);
    
    text = text.replace(/```([\s\S]*?)```/g, (match, code) => {
        const id = 'code-' + Math.random().toString(36).substr(2, 9);
        return `
            <div class="code-block-wrapper">
                <button class="copy-code-button" onclick="copyCode('${id}')">Copy</button>
                <pre><code id="${id}">${code.trim()}</code></pre>
            </div>
        `;
    });
    
    text = text.replace(/`([^`]+)`/g, '<span class="inline-code">$1</span>');

    text = text.replace(/\n/g, '<br>');
    
    return text;
}

function copyCode(id) {
    const code = document.getElementById(id).textContent;
    navigator.clipboard.writeText(code).then(() => {
        const btn = document.getElementById(id).parentElement.querySelector('.copy-code-button');
        const orig = btn.textContent;
        btn.textContent = 'Copied';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = orig;
            btn.classList.remove('copied');
        }, 2000);
    });
}

async function resetChat() {
    if (!confirm('Νέα συνομιλία;')) return;
    
    if (sessionId) {
        try {
            await fetch(`${API_URL}/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
        } catch (e) {}
    }
    
    chatContainer.innerHTML = `
        <div class="welcome-message">
            <div class="avatar bot-avatar">🤖</div>
            <div class="message-bubble assistant-bubble">
                Γεια σου! Είμαι ο AI Tutor σου για Java OOP. Ρώτησέ με οτιδήποτε!
            </div>
        </div>
    `;
    
    sessionId = null;
    messageInput.focus();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}