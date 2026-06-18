// Ίδιο origin με το frontend -> χρησιμοποιούμε σχετικά paths.
// Αν ποτέ χρειαστεί άλλος server, άλλαξε μόνο αυτό (π.χ. 'http://localhost:8000').
const API_BASE = '';

let sessionId = null;
let initialChatHTML = '';

const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const resetButton = document.getElementById('resetButton');
const darkModeToggle = document.getElementById('darkModeToggle');

document.addEventListener('DOMContentLoaded', () => {
    // Κρατάμε το αρχικό περιεχόμενο (welcome) για να το επαναφέρουμε στο reset
    // -> μία πηγή αλήθειας, χωρίς διπλό κείμενο.
    initialChatHTML = chatContainer.innerHTML;

    messageInput.addEventListener('input', autoResize);
    messageInput.addEventListener('keydown', handleKeyPress);
    sendButton.addEventListener('click', sendMessage);
    resetButton.addEventListener('click', resetChat);
    darkModeToggle.addEventListener('click', toggleDarkMode);

    // Copy buttons μέσα στα code blocks (event delegation, χωρίς inline onclick)
    chatContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('.copy-code-button');
        if (btn) copyCode(btn);
    });

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
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        sessionId = data.session_id;

        typing.remove();
        addMessage('assistant', data.reply, data.sources);
    } catch (error) {
        typing.remove();
        addMessage('assistant', 'Σφάλμα σύνδεσης με τον server.');
        console.error(error);
    } finally {
        // Πάντα ξανα-ενεργοποιούμε το input, ακόμα κι αν κάτι σκάσει παραπάνω.
        messageInput.disabled = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

function addMessage(role, content, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message';

    const rowDiv = document.createElement('div');
    rowDiv.className = `message-row ${role}-row`;

    if (role === 'assistant') {
        const avatar = document.createElement('div');
        avatar.className = 'avatar bot-avatar';
        avatar.setAttribute('aria-hidden', 'true');
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
    avatar.setAttribute('aria-hidden', 'true');
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
    // Πρώτα κάνουμε escape ΟΛΟ το κείμενο -> XSS-safe.
    text = escapeHtml(text);

    // 1) Βγάζουμε ΠΡΩΤΑ τα code blocks σε placeholder, ώστε να μην τα
    //    πειράξει το \n -> <br> πιο κάτω. Το (?:...\n) "τρώει" προαιρετικά
    //    το όνομα της γλώσσας (π.χ. ```java) ΜΟΝΟ όταν ακολουθείται από newline,
    //    οπότε δεν χαλάει μονόγραμμα code blocks.
    const codeBlocks = [];
    text = text.replace(/```(?:[a-zA-Z0-9+#.-]*[ \t]*\n)?([\s\S]*?)```/g, (match, code) => {
        codeBlocks.push(
            '<div class="code-block-wrapper">' +
            '<button class="copy-code-button" type="button">Copy</button>' +
            `<pre><code>${code.replace(/\n+$/, '')}</code></pre>` +
            '</div>'
        );
        return `\u0000CODE${codeBlocks.length - 1}\u0000`;
    });

    // 2) Inline code
    text = text.replace(/`([^`]+)`/g, '<span class="inline-code">$1</span>');

    // 3) Σβήνουμε τις κενές γραμμές γύρω από τα code blocks
    //    (το .code-block-wrapper έχει ήδη δικό του margin για το spacing).
    text = text.replace(/\n*\u0000CODE(\d+)\u0000\n*/g, '\u0000CODE$1\u0000');

    // 4) Newlines -> <br> ΜΟΝΟ στο κανονικό κείμενο
    text = text.replace(/\n/g, '<br>');

    // 5) Επαναφέρουμε τα code blocks
    text = text.replace(/\u0000CODE(\d+)\u0000/g, (m, i) => codeBlocks[i]);

    return text;
}

function copyCode(btn) {
    const code = btn.parentElement.querySelector('code');
    if (!code) return;

    navigator.clipboard.writeText(code.textContent).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Copied';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = orig;
            btn.classList.remove('copied');
        }, 2000);
    }).catch((err) => {
        console.error('Copy failed:', err);
        btn.textContent = 'Σφάλμα';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    });
}

async function resetChat() {
    if (!confirm('Νέα συνομιλία;')) return;

    if (sessionId) {
        try {
            await fetch(`${API_BASE}/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
        } catch (e) {
            console.error(e);
        }
    }

    chatContainer.innerHTML = initialChatHTML;
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