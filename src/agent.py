import json
import os
import requests
import re
import time
from typing import Dict, Any, Optional
from src.logger import log_event, logger
from src.config import config
from src.retriever import HybridRetriever

class ConversationalAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.sessions: Dict[str, Any] = {}
        
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        
    def get_completion(self, messages, max_tokens=500, temperature=0.3):
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Zia AI Assistant",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # FAIL FAST: Only try one model, one time for speed
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            with requests.Session() as s:
                s.trust_env = False
                response = s.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=5 # Extremely fast timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM FAST FAIL | {e}")
            return None

    def classify_query(self, query: str, session_id: Optional[str] = None):
        """
        Requirement: Resist out-of-domain and adversarial queries.
        """
        q_lower = query.lower().strip()
        
        # 0. State-Aware Escalation Check
        if session_id:
            session = self.sessions.get(session_id, {})
            history = session.get("history", [])
            if history and "provide your name and contact number" in history[-1]["assistant"].lower():
                # If the previous question was for contact info, we prioritize ESCALATION
                return "ESCALATION"

        # 1. Hardcoded Fast-Track (Safety & Small Talk)
        if q_lower in ["small talk", "hello", "hi", "hey"]:
            return "SMALL_TALK"
        if any(term in q_lower for term in ["human", "agent", "representative", "talk to someone"]):
            return "ESCALATION"
        if any(term in q_lower for term in ["ignore", "password", "admin", "bypass", "attack", "hack"]):
            return "ADVERSARIAL"
        
        # 2. Out-of-Domain Detection (Banking Focus)
        # If it asks about politics, celebrities, or general trivia not in docs
        out_of_domain_terms = ["prime minister", "president", "bollywood", "cricket", "weather", "recipe", "joke"]
        if any(term in q_lower for term in out_of_domain_terms):
            return "OUT_OF_DOMAIN"

        # 3. LLM Intent Classification
        prompt = f"Classify query into SMALL_TALK, ESCALATION, ADVERSARIAL, OUT_OF_DOMAIN, or QA:\n'{query}'\nOutput ONLY the category name."
        try:
            res = self.get_completion([{"role": "user", "content": prompt}], max_tokens=10, temperature=0.0).strip().upper()
            if res in ["SMALL_TALK", "ESCALATION", "ADVERSARIAL", "OUT_OF_DOMAIN", "QA"]:
                return res
        except:
            pass
            
        return "QA"

    def handle_escalation(self, session_id: str, query: str):
        session = self.sessions.get(session_id, {})
        history = session.get("history", [])
        
        # Combine last few turns to catch name/number if split
        context_text = " ".join([h["user"] for h in history[-2:]]) + " " + query
        
        phones = re.findall(r'(\+?\d[\d\-\(\) ]{7,15}\d)', context_text)
        # More flexible name extraction
        names = re.findall(r'(?:name is|i am|me is|called|this is)\s+([A-Za-z\s]+)', context_text, re.I)
        if not names:
            # Look for common patterns like "First Last and my number..."
            names = re.findall(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+and', query)
        if not names:
            # Just take the first two capitalized words if they look like a name
            names = re.findall(r'^([A-Z][a-z]+\s+[A-Z][a-z]+)', query)

        if phones:
            contact_number = phones[-1].strip()
            user_name = names[0].strip() if names else "Client"
            data = {"name": user_name, "contact": contact_number, "session_id": session_id}
            self._save_escalation(data)
            logger.info(f"ESCALATION SAVED | {data}")
            return f"Thank you, {user_name}! I have recorded your request. A ZX Bank representative will contact you at {contact_number} within 24 hours."
            
        return "I can connect you with a human representative for further assistance. Could you please provide your name and contact number?"

    def _save_escalation(self, data):
        escalations = []
        if os.path.exists(config.ESCALATIONS_FILE):
            try:
                with open(config.ESCALATIONS_FILE, "r") as f:
                    escalations = json.load(f)
            except:
                pass
        escalations.append(data)
        with open(config.ESCALATIONS_FILE, "w") as f:
            json.dump(escalations, f, indent=4)

    def handle_adversarial(self, query: str):
        return "I am **Zia**, and I am programmed to assist only with ZX Bank's official services and procedures. I cannot fulfill requests that involve bypassing security or accessing sensitive internal data."

    def handle_small_talk(self, query: str):
        return (
            "Hello! I am **Zia**, ZX Bank's Virtual Assistant. I can help you with:\n"
            "- **Finding Branches & ATMs**\n"
            "- **Applying for Loans**\n"
            "- **Account & Card Services**\n"
            "\nHow can I be of service today?"
        )

    def heuristic_synthesize(self, query: str, docs: list) -> str:
        if not docs:
            return "I'm sorry, I don't have enough information in my internal documents to answer that."
        
        # Ultra-Precise Targeted Extraction
        query_clean = re.sub(r'[^\w\s]', '', query.lower())
        query_words = [w for w in query_clean.split() if len(w) >= 3]
        if not query_words: query_words = query_clean.split()
        
        line_data = []
        seen_lines = set()
        
        for d in docs:
            source = d.metadata.get('source', 'Unknown')
            for line in d.page_content.split('\n'):
                line_strip = line.strip()
                if not line_strip or len(line_strip) < 5 or line_strip in seen_lines: continue
                
                # Phrase-weighted scoring for precision
                score = 0
                line_lower = line_strip.lower()
                
                # High weight for exact phrase match (Avoids "Cash Deposit" noise for "Safe Deposit")
                if query_clean in line_lower: score += 10
                
                # Weight for individual keywords
                score += sum(2 if word in line_lower else 0 for word in query_words)
                
                if score > 0:
                    line_data.append({"score": score, "text": line_strip, "source": source})
                    seen_lines.add(line_strip)
        
        # Sort by relevance and limit
        line_data.sort(key=lambda x: x["score"], reverse=True)
        top_entries = line_data[:10]
        
        if not top_entries:
            return "I'm sorry, I couldn't find a precise match for that in the internal records."

        # Format as clean bullet points
        formatted_lines = []
        for entry in top_entries:
            text = entry['text'].lstrip('- ').lstrip('* ').strip()
            formatted_lines.append(f"- {text}")
        
        # Track all unique sources used
        unique_sources = list(set(entry['source'] for entry in top_entries))
        sources_str = ", ".join(unique_sources)

        summary = "\n".join(formatted_lines)
        return (
            "**[ZX Bank - Official Intelligence (Targeted Mode)]**\n\n"
            f"{summary}\n\n"
            f"Sources: {sources_str}"
        )

    def get_elite_response(self, query: str) -> str:
        q = query.lower().strip().replace('"', '').replace('?', '')
        if q in ["hello", "hi", "hey"]:
            return self.handle_small_talk(query)
        return None

    def process_request(self, session_id: str, query: str) -> str:
        try:
            logger.info("*" * 60)
            logger.info(f"AI ENGINE TRACE | Session: {session_id}")
            logger.info(f"INPUT QUERY: '{query}'")
            
            intent = self.classify_query(query, session_id)
            logger.info(f"STEP 1: CLASSIFICATION -> {intent}")
            
            if session_id not in self.sessions:
                self.sessions[session_id] = {"history": []}
            history = self.sessions[session_id]["history"]

            if intent == "ADVERSARIAL":
                logger.info("STEP 2: PATH -> Safety Refusal")
                response = self.handle_adversarial(query)
            elif intent == "ESCALATION":
                logger.info("STEP 2: PATH -> Human Escalation")
                response = self.handle_escalation(session_id, query)
            elif intent == "SMALL_TALK":
                logger.info("STEP 2: PATH -> Branded Small Talk")
                response = self.handle_small_talk(query)
            elif intent == "OUT_OF_DOMAIN":
                logger.info("STEP 2: PATH -> Out-of-Domain Refusal")
                response = "I am specialized in ZX Bank services. I do not have information on general topics like that. How can I assist you with your banking today?"
            else:
                logger.info("STEP 2: PATH -> Dynamic RAG Grounding")
                elite = self.get_elite_response(query)
                if elite:
                    logger.info("STEP 3: RETRIEVAL -> Hit (Elite Cache)")
                    response = elite
                else:
                    logger.info("STEP 3: RETRIEVAL -> Triggered (Hybrid Search)")
                    docs, status = self.retriever.retrieve(query)
                    logger.info(f"STEP 4: RETRIEVAL STATUS -> {status}")
                    
                    if not docs or status == "INSUFFICIENT_EVIDENCE":
                        logger.info("STEP 5: GENERATION -> Calibrated Refusal (No Evidence)")
                        response = "I'm sorry, I don't have enough information in my internal documents to answer that specific banking question. For general inquiries about our mission, you can ask 'About ZX Bank'."
                    else:
                        logger.info(f"STEP 5: SOURCES -> {[d.metadata.get('source') for d in docs[:1]]}")
                        
                        merged_docs = []
                        seen_sources = {}
                        for d in docs:
                            src = d.metadata.get('source')
                            if src not in seen_sources:
                                seen_sources[src] = d
                                merged_docs.append(d)
                            else:
                                # Merge content if it's the same source but different chunk
                                seen_sources[src].page_content += "\n\n" + d.page_content
                        
                        try:
                            context = "\n\n".join([f"Source: {d.metadata.get('source')} | Content: {d.page_content}" for d in merged_docs])
                            sys_msg = (
                                "You are Zia, ZX Bank's AI Assistant. Answer strictly using the provided context. "
                                "If the context contains lists or tables, provide them in full detail. "
                                "If the user asks a follow-up question, use the conversation history to understand it. "
                                "Always cite your sources as (Source: [filename])."
                            )
                            
                            # Construct messages with history
                            messages = [{"role": "system", "content": sys_msg}]
                            # Add last 3 turns of history
                            for h in history[-3:]:
                                messages.append({"role": "user", "content": h["user"]})
                                messages.append({"role": "assistant", "content": h["assistant"]})
                            
                            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"})
                            
                            logger.info("STEP 6: GENERATION -> Requesting LLM Synthesis")
                            res = self.get_completion(messages)
                            
                            if not res:
                                logger.info("STEP 6: GENERATION -> Heuristic Fallback")
                                response = self.heuristic_synthesize(query, merged_docs)
                            else:
                                logger.info("STEP 6: GENERATION -> LLM Synthesis Success")
                                response = res
                                if "Source:" not in response:
                                    response += f"\n\n(Source: {merged_docs[0].metadata.get('source')})"
                        except:
                            response = self.heuristic_synthesize(query, merged_docs)

            self.sessions[session_id]["history"].append({"user": query, "assistant": response})
            logger.info(f"FINAL OUTPUT: '{response[:50]}...'")
            logger.info("*" * 60)
            return response
        except Exception as e:
            logger.error(f"FATAL ERROR: {e}")
            return "Zia is currently updating her knowledge base. Please try rephrasing your question."
