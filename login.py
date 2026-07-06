
import streamlit as st
import json
import os
import hashlib
import secrets


DB_FILE = "users.json"

def get_users_db():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_users_db(users):
    with open(DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password, salt=None):
    """PBKDF2-SHA256 with a per-user random salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()
    return salt, digest

def register_new_user(email, name, password):
    users = get_users_db()
    salt, digest = hash_password(password)
    users[email.strip().lower()] = {"name": name.strip(), "salt": salt, "password": digest}
    save_users_db(users)

def verify_credentials(email, password):
    users = get_users_db()
    email = email.strip().lower()
    user = users.get(email)
    if not user:
        return None
    if "salt" in user:
        _, digest = hash_password(password, user["salt"])
        if secrets.compare_digest(digest, user["password"]):
            return user["name"]
    else:
        # Legacy unsalted SHA-256 account — verify once, then upgrade to salted PBKDF2
        legacy = hashlib.sha256(password.encode()).hexdigest()
        if secrets.compare_digest(legacy, user["password"]):
            user["salt"], user["password"] = hash_password(password)
            users[email] = user
            save_users_db(users)
            return user["name"]
    return None

def show_login_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg: #070B12;
        --panel: #0D131C;
        --panel-2: #0A0F17;
        --line: rgba(57,255,136,0.16);
        --line-dim: rgba(57,255,136,0.07);
        --phosphor: #39FF88;
        --phosphor-dim: #1F9A5C;
        --amber: #FFB020;
        --amber-dim: #8A6420;
        --text: #E7F0F7;
        --text-dim: #5A7288;
        --text-faint: #2E4257;
        --danger: #FF5D5D;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section[data-testid="stMain"] > div {
        background: var(--bg) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stAppViewContainer"] {
        background-image:
            radial-gradient(ellipse 90% 45% at 50% -8%, rgba(57,255,136,0.10) 0%, transparent 60%),
            radial-gradient(ellipse 60% 30% at 85% 100%, rgba(255,176,32,0.06) 0%, transparent 55%) !important;
        background-color: var(--bg) !important;
    }

    [data-testid="stHeader"], header[data-testid="stHeader"] { display: none !important; }
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

    .block-container {
        max-width: 460px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin: 0 auto !important;
    }

    /* ══════════ MONITOR BEZEL (signature element) ══════════ */
    .monitor {
        position: relative;
        border: 1px solid var(--line);
        border-radius: 4px;
        background:
            repeating-linear-gradient(0deg, var(--line-dim) 0 1px, transparent 1px 18px),
            repeating-linear-gradient(90deg, var(--line-dim) 0 1px, transparent 1px 18px),
            var(--panel-2);
        padding: 1.1rem 1.2rem 0.9rem;
        margin-bottom: 1.4rem;
        overflow: hidden;
    }
    .monitor::before, .monitor::after,
    .monitor .br-tl, .monitor .br-tr, .monitor .br-bl, .monitor .br-br {
        content: '';
        position: absolute;
        width: 14px; height: 14px;
        border: 2px solid var(--phosphor);
        opacity: 0.65;
    }
    .monitor .br-tl { top: -1px; left: -1px; border-right: none; border-bottom: none; }
    .monitor .br-tr { top: -1px; right: -1px; border-left: none; border-bottom: none; }
    .monitor .br-bl { bottom: -1px; left: -1px; border-right: none; border-top: none; }
    .monitor .br-br { bottom: -1px; right: -1px; border-left: none; border-top: none; }

    .monitor-topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.62rem;
        color: var(--text-faint);
        letter-spacing: 0.14em;
        margin-bottom: 0.5rem;
    }
    .live-dot {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--phosphor);
        box-shadow: 0 0 6px var(--phosphor);
        margin-right: 5px;
        animation: blink 1.6s ease-in-out infinite;
    }
    @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

    .vitals-row {
        display: flex;
        gap: 0.55rem;
        margin-bottom: 0.65rem;
        flex-wrap: wrap;
    }
    .vital-chip {
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
        border-radius: 3px;
        padding: 0.28rem 0.55rem;
        font-size: 0.66rem;
        color: var(--text-dim);
        letter-spacing: 0.05em;
        line-height: 1.3;
    }
    .vital-chip b {
        display: block;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0;
    }
    .vital-chip.hr b { color: var(--phosphor); }
    .vital-chip.spo2 b { color: var(--amber); }
    .vital-chip.bp b { color: var(--text); }

    .trace-wrap { width: 100%; }
    .trace-wrap svg { display: block; width: 100%; height: 54px; }
    .trace-ecg {
        fill: none; stroke: var(--phosphor); stroke-width: 1.8;
        stroke-linecap: round; stroke-linejoin: round;
        stroke-dasharray: 1300; stroke-dashoffset: 1300;
        filter: drop-shadow(0 0 3px rgba(57,255,136,0.6));
        animation: draw 2.1s cubic-bezier(0.4,0,0.2,1) 0.1s forwards,
                   pulse-g 2.6s 2.4s ease-in-out infinite;
    }
    .trace-pleth {
        fill: none; stroke: var(--amber); stroke-width: 1.4;
        stroke-linecap: round; stroke-linejoin: round;
        stroke-dasharray: 900; stroke-dashoffset: 900;
        opacity: 0.75;
        animation: draw2 2.4s cubic-bezier(0.4,0,0.2,1) 0.35s forwards;
    }
    @keyframes draw { to { stroke-dashoffset: 0; } }
    @keyframes draw2 { to { stroke-dashoffset: 0; } }
    @keyframes pulse-g {
        0%,100% { filter: drop-shadow(0 0 3px rgba(57,255,136,0.5)); }
        50% { filter: drop-shadow(0 0 8px rgba(57,255,136,0.95)); }
    }

    /* ══════════ BRAND ══════════ */
    .brand { text-align: center; margin-bottom: 1.3rem; }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.35rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -1.2px;
        line-height: 1;
        margin: 0 0 0.4rem;
    }
    .brand-title .accent {
        color: var(--phosphor);
        text-shadow: 0 0 18px rgba(57,255,136,0.45);
    }
    .brand-sub {
        font-size: 0.66rem;
        color: var(--text-faint);
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-weight: 500;
    }

    /* ══════════ TABS ══════════ */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(57,255,136,0.14) !important;
        border-radius: 0 !important;
        padding: 0 !important;
        gap: 1.6rem !important;
        margin-bottom: 1.3rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0 !important;
        color: var(--text-faint) !important;
        font-weight: 600 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.12em !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.1rem 0.05rem 0.6rem !important;
        transition: all 0.15s ease !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--phosphor) !important;
        border-bottom: 2px solid var(--phosphor) !important;
        text-shadow: 0 0 10px rgba(57,255,136,0.35);
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ══════════ LABELS ══════════ */
    .stTextInput label p, .stTextInput label {
        font-size: 0.66rem !important;
        font-weight: 600 !important;
        color: var(--text-faint) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        margin-bottom: 0.3rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ══════════ TEXT INPUTS — terminal line style ══════════ */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.015) !important;
        border: none !important;
        border-bottom: 1px solid rgba(90,114,136,0.35) !important;
        border-radius: 2px !important;
        color: var(--text) !important;
        font-size: 0.88rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        caret-color: var(--phosphor) !important;
        padding: 0.55rem 0.15rem !important;
        transition: border-color 0.15s, background 0.15s !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input:focus {
        border-bottom: 1px solid var(--phosphor) !important;
        background: rgba(57,255,136,0.035) !important;
        box-shadow: 0 1px 0 0 rgba(57,255,136,0.4) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--text-faint) !important; }
    .stTextInput > div { border: none !important; box-shadow: none !important; }

    /* ══════════ SUBMIT BUTTON ══════════ */
    [data-testid="stFormSubmitButton"] > button,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #1F9A5C 0%, #0E5C38 100%) !important;
        color: #EAFFF4 !important;
        border: 1px solid rgba(57,255,136,0.4) !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        font-family: 'JetBrains Mono', monospace !important;
        box-shadow: 0 0 22px rgba(57,255,136,0.18) !important;
        height: 2.5rem !important;
        width: 100% !important;
        transition: box-shadow 0.18s ease, transform 0.18s ease !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 0 32px rgba(57,255,136,0.32) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 0 14px rgba(57,255,136,0.22) !important;
    }

    /* ══════════ GUEST OVERRIDE BUTTON ══════════ */
    .stButton > button {
        background: transparent !important;
        color: var(--amber) !important;
        border: 1px dashed rgba(255,176,32,0.4) !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        font-family: 'JetBrains Mono', monospace !important;
        height: 2.4rem !important;
        transition: background 0.18s, border-color 0.18s, box-shadow 0.18s !important;
    }
    .stButton > button:hover {
        background: rgba(255,176,32,0.06) !important;
        border-color: rgba(255,176,32,0.7) !important;
        box-shadow: 0 0 16px rgba(255,176,32,0.14) !important;
    }

    /* ══════════ ALERTS ══════════ */
    [data-testid="stAlertContainer"], .stAlert {
        border-radius: 3px !important;
        font-size: 0.78rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-left: 3px solid !important;
    }

    /* ══════════ SUBHEADER ══════════ */
    h3 {
        color: var(--text-dim) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        margin-bottom: 1.1rem !important;
        margin-top: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    hr { border-color: rgba(57,255,136,0.06) !important; margin: 0.5rem 0 !important; }

    /* ══════════ FOOTER ══════════ */
    .login-foot {
        text-align: center;
        color: var(--text-faint);
        font-size: 0.64rem;
        letter-spacing: 0.06em;
        margin-top: 1.6rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
    }
    .login-foot .tag {
        display: inline-block;
        border: 1px solid var(--danger);
        color: var(--danger);
        border-radius: 3px;
        padding: 0.05rem 0.35rem;
        margin-right: 0.4rem;
        font-size: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── MONITOR BEZEL: vitals + dual trace hero ──
    st.markdown("""
    <div class="monitor">
      <span class="br-tl"></span><span class="br-tr"></span><span class="br-bl"></span><span class="br-br"></span>
      <div class="monitor-topline">
        <span><span class="live-dot"></span>PATIENT MONITOR — LIVE</span>
        <span>LEAD II · SPO2</span>
      </div>
      <div class="vitals-row">
        <div class="vital-chip hr">HR (bpm)<b>72</b></div>
        <div class="vital-chip spo2">SpO2 (%)<b>98</b></div>
        <div class="vital-chip bp">BP (mmHg)<b>118/76</b></div>
      </div>
      <div class="trace-wrap">
        <svg viewBox="0 0 500 54" xmlns="http://www.w3.org/2000/svg">
          <path class="trace-pleth" d="
            M0,34 C15,26 25,26 40,34 C55,42 65,42 80,34
            C95,26 105,26 120,34 C135,42 145,42 160,34
            C175,26 185,26 200,34 C215,42 225,42 240,34
            C255,26 265,26 280,34 C295,42 305,42 320,34
            C335,26 345,26 360,34 C375,42 385,42 400,34
            C415,26 425,26 440,34 C455,42 465,42 480,34
            C490,30 495,30 500,34
          "/>
          <path class="trace-ecg" d="
            M0,32 L44,32
            Q52,20 60,32
            L74,32 L78,38 L83,2 L88,46 L93,32
            Q103,18 114,32
            L162,32
            Q170,20 178,32
            L192,32 L196,38 L201,2 L206,46 L211,32
            Q221,18 232,32
            L280,32
            Q288,20 296,32
            L310,32 L314,38 L319,2 L324,46 L329,32
            Q339,18 350,32
            L398,32
            Q406,20 414,32
            L428,32 L432,38 L437,2 L442,46 L447,32
            Q457,18 468,32
            L500,32
          "/>
        </svg>
      </div>
    </div>
    <div class="brand">
      <div class="brand-title">Doctor <span class="accent">AI</span></div>
      <div class="brand-sub">Secure Medical Intelligence Portal</div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["▸ LOG IN", "▸ SIGN UP"])

    with tab_login:
        st.subheader("// AUTHENTICATE SESSION")
        with st.form("login_form"):
            email_input = st.text_input("Email Address", placeholder="you@example.com")
            pass_input = st.text_input("Password", type="password", placeholder="••••••••")
            submit_login = st.form_submit_button("Log In →", use_container_width=True)

            if submit_login:
                if not email_input or not pass_input:
                    st.warning("⚠️ Please enter both email and password.")
                else:
                    user_name = verify_credentials(email_input, pass_input)
                    if user_name:
                        st.session_state['logged_in'] = True
                        st.session_state['current_user_name'] = user_name
                        st.rerun()
                    else:
                        st.error("❌ Invalid Email or Password!")

        st.write("")
        if st.button("⚡ Guest Override — Hack Club Judges", use_container_width=True):
            st.session_state['logged_in'] = True
            st.session_state['current_user_name'] = "Hack Club Judge"
            st.rerun()

    with tab_signup:
        st.subheader("// REGISTER NEW PATIENT RECORD")
        with st.form("signup_form"):
            new_name = st.text_input("Full Name", placeholder="Jane Smith")
            new_email = st.text_input("Email Address", placeholder="you@example.com")
            new_pass = st.text_input("Create Password", type="password", placeholder="At least 6 characters")
            confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")
            submit_signup = st.form_submit_button("Create Account →", use_container_width=True)

            if submit_signup:
                db = get_users_db()
                if not new_name or not new_email or not new_pass:
                    st.warning("⚠️ Please fill in all required fields.")
                elif new_pass != confirm_pass:
                    st.error("❌ Passwords do not match!")
                elif len(new_pass) < 6:
                    st.warning("⚠️ Password must be at least 6 characters long.")
                elif new_email.strip().lower() in db:
                    st.error("❌ This email is already registered! Please log in.")
                else:
                    register_new_user(new_email, new_name, new_pass)
                    st.success("✅ Account created! Please switch to the 'Log In' tab.")

    st.markdown("""
    <div class="login-foot"><span class="tag">!</span>Not a substitute for professional medical advice</div>
    """, unsafe_allow_html=True)
    
