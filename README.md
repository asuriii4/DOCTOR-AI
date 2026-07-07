# DOCTOR-AI
Doctor AI is a production-grade medical chatbot powered by Groq LLM and RAG technology. It combines local medical knowledge bases with live web search to provide evidence-based health information, complete with safety disclaimers and professional guidance.


<img width="1815" height="795" alt="image" src="https://github.com/user-attachments/assets/a626f332-7c15-474e-9471-a736902f3b57" />


# Why I Built This

I started learning cybersecurity and web security through CTF competitions and 
phishing detection research. But I realized: **security is only meaningful if 
it protects something people actually care about.**

Medical AI caught my attention because:

1. **Safety > Intelligence** — Most medical chatbots just output text. I wanted 
   to build one that *refuses* to be confident when it shouldn't be. Every 
   response includes mandatory disclaimers and links to real professionals.

2. **RAG as a Control Mechanism** — Instead of just trusting an LLM, I learned 
   that retrieval-augmented generation (RAG) lets you ground responses in 
   verified medical literature. This was a natural extension of my security 
   mindset: verify first, trust the model second.

3. **Production-Grade Code** — I refactored the initial prototype into modular, 
   tested architecture because I wanted to prove I could build things that 
   scale, not just demos. This mirrors my journey from CTF problem-solving 
   (one-off exploits) to real engineering (maintainable systems).


The project taught me:
- How LLMs can be controlled and constrained (prompt engineering at scale)
- Vector databases and semantic search (RAG fundamentals)
- Proper error handling and logging for production systems
- User authentication and session management

**This is the kind of project I want to build**: systems where the security and 
safety architecture is as important as the feature set.
