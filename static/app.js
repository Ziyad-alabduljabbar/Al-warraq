document.addEventListener('DOMContentLoaded', () => {

    const body = document.body;
    if (body) {
        const currentUserFlag = body.dataset.currentUser === 'true';
        setupRecommendationModeControl(currentUserFlag);
        setupInterestForm();
        setupPasswordValidation();
    }

    const alerts = document.querySelectorAll('.alert'); 
    if (alerts.length > 0) {
        setTimeout(() => {
            alerts.forEach(alert => {
                alert.style.transition = 'opacity 0.5s ease';
                alert.style.opacity = '0'; 
                setTimeout(() => alert.remove(), 500);
            });
        }, 3000); 
    }

    const loginForm = document.querySelector('form[action="/login"]') || document.querySelector('.auth-form');
    if (loginForm && window.location.pathname === '/login') {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault(); 
            const formData = new FormData(loginForm);
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    body: formData,
                    headers: { 'X-Requested-With': 'XMLHttpRequest' }
                });

                if (response.status === 429) {
                    const data = await response.json();
                    showNotification(data.error, 'error'); 
                } else if (response.redirected) {
                    window.location.href = response.url; 
                } else if (response.ok) {
                    const htmlText = await response.text();
                    if (htmlText.includes('Invalid email or password')) {
                        showNotification('Invalid email or password', 'error');
                    } else {
                        document.body.innerHTML = htmlText;
                    }
                }
            } catch (error) {
                console.error('Login Error:', error);
            }
        });
    }

    
    const chatInput = document.getElementById('chatInput');
if (chatInput) {
    chatInput.addEventListener('input', function() {
        
        this.style.height = 'auto'; 
        
        
        let scrollH = this.scrollHeight;
        this.style.height = (scrollH > 40 ? scrollH : 40) + 'px';

        
        if (scrollH >= 95) {
            this.style.overflowY = 'auto';
        } else {
            this.style.overflowY = 'hidden';
        }
    });

    chatInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            const sendBtn = this.nextElementSibling;
            if (sendBtn) sendBtn.click();
            
            
            this.value = '';
            this.style.height = '40px';
            this.style.overflowY = 'hidden';
        }
    });
}
});


function setupRecommendationModeControl(isLoggedIn) {
  const modeSelect = document.querySelector('[data-recommendation-select]');
  if (!modeSelect) return;

  if (!isLoggedIn) {
    if (modeSelect.value !== 'tfidf') {
      modeSelect.value = 'tfidf';
    }
    modeSelect.setAttribute('disabled', 'disabled');
    modeSelect.classList.add('disabled-select');
    appendLoginNotice(modeSelect);
    return;
  }

  modeSelect.removeAttribute('disabled');
  modeSelect.classList.remove('disabled-select');
  modeSelect.addEventListener('change', () => {
    if (!modeSelect.value) {
      modeSelect.value = 'tfidf';
    }
  });
}

function appendLoginNotice(targetSelect) {
  const info = document.createElement('p');
  info.className = 'login-notice';
  info.textContent = 'Log in or create an account to unlock personalized recommendation modes.';
  const parent = targetSelect.closest('.search-panel') || targetSelect.parentElement;
  if (!parent) return;

  if (!parent.querySelector('.login-notice')) {
    parent.appendChild(info);
  }
}

function setupInterestForm() {
  const form = document.querySelector('[data-interest-form]');
  if (!form) return;

  const checkboxes = Array.from(form.querySelectorAll('input[type="checkbox"][name="interests"]'));
  const summary = form.querySelector('[data-interest-summary]');
  const emptyState = form.querySelector('[data-interest-empty]');

  form.addEventListener('submit', () => {
    const existingInputs = form.querySelectorAll('input[type="hidden"][name="interests"]');
    existingInputs.forEach((hidden) => hidden.remove());

    checkboxes
      .filter((box) => box.checked)
      .forEach((box) => {
        const hidden = document.createElement('input');
        hidden.type = 'hidden';
        hidden.name = 'interests';
        hidden.value = box.value;
        form.appendChild(hidden);
      });
  });

  const updateSummary = () => {
    if (!summary || !emptyState) return;

    summary.innerHTML = '';
    const selected = checkboxes.filter((box) => box.checked);

    if (!selected.length) {
      summary.classList.add('is-hidden');
      emptyState.classList.remove('is-hidden');
      return;
    }

    selected.forEach((box) => {
      const chip = document.createElement('span');
      chip.className = 'badge badge--compact';
      chip.textContent = box.value;
      summary.appendChild(chip);
    });

    summary.classList.remove('is-hidden');
    emptyState.classList.add('is-hidden');
  };

  checkboxes.forEach((box) => {
    box.addEventListener('change', updateSummary);
  });

  updateSummary();
}

// --- AI Chatbot Logic ---

// Function to format text (Bold, Bullet points, and New lines)
function formatMessage(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/^(\*|\-)\s+(.*)/gm, '• $2')             // Bullet points
        .replace(/\n/g, '<br>');                         // New lines
}

// Function to append messages to the chat history
function appendMessage(sender, text, id = '') {
    const chatHistory = document.getElementById('chatHistory');
    if (!chatHistory) return;

    const wrapper = document.createElement('div');
    wrapper.className = `msg-wrapper ${sender}`;
    if (id) wrapper.id = id;
    
    const bubble = document.createElement('div');
    bubble.className = `msg-bubble ${sender}`;
    bubble.setAttribute('dir', 'auto');
    bubble.innerHTML = formatMessage(text); 
    
    wrapper.appendChild(bubble);
    chatHistory.appendChild(wrapper);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Function to handle sending messages
async function sendChatMessage(bookData, similarBooks) {
    const inputField = document.getElementById('chatInput');
    const message = inputField.value.trim();
    if (!message) return;

    appendMessage('user', message);
    inputField.value = '';
    const loadingId = 'loading-' + Date.now();
    appendMessage('bot', 'Thinking...', loadingId);

    const payload = {
        message: message,
        context: {
            title: bookData.title,
            author: bookData.author,
            category: bookData.category,
            tags: bookData.tags,
            full_desc: bookData.full_desc,
            similar_books: similarBooks
        }
    };

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();
        appendMessage('bot', data.response);
    } catch (e) {
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();
        appendMessage('bot', 'Connection error.');
    }
}

// --- Password Validation & Toggle Logic ---
function setupPasswordValidation() {
    const form = document.getElementById('changePasswordForm');
    if (!form) return; 

    
    const toggleBtn = document.getElementById('togglePasswordBtn');
    const formContainer = document.getElementById('passwordFormContainer');

    if (toggleBtn && formContainer) {
        toggleBtn.addEventListener('click', () => {
            if (formContainer.style.display === 'none' || formContainer.style.display === '') {
                formContainer.style.display = 'block';
                toggleBtn.innerHTML = '❌ Cancel'; 
            } else {
                formContainer.style.display = 'none';
                toggleBtn.innerHTML = '🔒 Change Password'; 
            }
        });
    }

    
    form.addEventListener('submit', function(event) {
        const newPassword = document.getElementById('new_password').value;
        const confirmPassword = document.getElementById('confirm_password').value;
        const errorText = document.getElementById('password-error');

        if (newPassword !== confirmPassword) {
            event.preventDefault(); 
            errorText.style.display = 'block'; 
        } else {
            errorText.style.display = 'none';
        }
    });
}

function showNotification(message, type) {
    const flashContainer = document.querySelector('.flash-messages') || document.querySelector('.auth-form');
    
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(a => a.remove());

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.style.marginBottom = '1rem';
    alertDiv.textContent = message;
    
    if (flashContainer) {
        flashContainer.prepend(alertDiv);
    } else {
        document.body.prepend(alertDiv);
    }

    setTimeout(() => {
        alertDiv.style.transition = 'opacity 0.5s ease';
        alertDiv.style.opacity = '0';
        setTimeout(() => alertDiv.remove(), 500);
    }, 4000);
}
